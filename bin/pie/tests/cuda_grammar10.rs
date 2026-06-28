//! #10-verify: uniqueness-aware cross-request accumulation — GPU end-to-end
//! (alpha policy ∥ bravo attach-hash ∥ delta fire). Proves the #10 accumulation
//! policy's **distinct-program accounting** flows correctly attach → `on_arrival`
//! → `distinct_program_count()` → the fire trace, on the REAL driver — the thing
//! no unit test can settle (real `program_identity_hash`es threaded from attach
//! through the scheduler into a genuine merged fire).
//!
//! ## The witness line (load-bearing, asserted — not inferred)
//! `boot_4090` runs the engine IN-PROCESS, so the scheduler's `eprintln!` lands on
//! THIS process's fd 2; each phase `dup2`s fd 2 to a file (the `cuda_grammar_r2`
//! capture pattern) and asserts on:
//!   `[pie-sched-trace] driver=… fire requests=N … distinct_programs=K`
//! where `N` = `batch.len()` (co-batched requests) and `K` =
//! `policy.distinct_program_count()` (the unioned `program_identity_hash` set size
//! for the firing batch — identical programs self-dedup to one entry).
//!
//! ## The three gates (each a distinct, deterministic claim)
//!   - **G1 DEDUP** — `N` IDENTICAL programs (same `grammar-late`) co-batch to ONE
//!     fire with `distinct_programs = 1`. The dedup key collapses identical
//!     bytecode+manifest to a single compile/finalize wall, so `K=1` though `N≥2`.
//!   - **G2 DISTINCT** — `N` programs with DISTINCT bytecode/manifest co-batch to a
//!     fire with `distinct_programs ≥ 2` (`> G1`'s 1). This is the load-bearing
//!     contrast: the SAME co-batch size yields `K=1` (all identical) vs `K≥2`
//!     (distinct) ⇒ the policy is uniqueness-AWARE, not request-counting.
//!   - **G3 NO-REGRESSION** — a SOLE request fires un-coalesced (`requests=1`): a
//!     sparse arrival is never inflated into a phantom co-batch (the
//!     low-concurrency common path the #10 window must never penalise).
//!
//! Co-batching uses alpha's deterministic test hold (`PIE_SCHED_ACCUM_HOLD_US`,
//! the same lever `cuda_grammar_r2` proved coalesces ≥2 prefills) so the merge is
//! reproducible; `on_arrival` unions each request's identities regardless of the
//! adaptive window, so `distinct_programs` is exact on the held batch.
//!
//! Run (GPU; needs the #10 base = alpha policy + bravo attach-hash):
//!   cargo test -p pie-bin --features driver-cuda --test cuda_grammar10 -- --ignored --nocapture

mod common;

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Deterministic co-batch hold (µs). 120ms comfortably coalesces ≥2 cross-session
/// prefills into one drain window (`cuda_grammar_r2` proved 50ms suffices for 2;
/// the extra margin absorbs the 3-distinct G2 launch). CLI-overridable.
const ACCUM_HOLD_US: &str = "120000";

/// Small token budget — the co-batch witness is on the (prefill) fire; keep the
/// decode tail short so the phases stay quick.
const MAX_TOKENS: u32 = 2;

/// A disjoint alphabet for the masking inferlets (kept in-range / small so the
/// natural argmax is forced out; identical across the G1 dedup procs so they hash
/// to ONE identity).
const ALPHABET: [u32; 4] = [10, 11, 12, 13];

/// RAII capture of fd 2 to a temp file (the in-process engine writes its scheduler
/// trace there). Restores fd 2 on `finish()`/`Drop`. Mirrors `cuda_grammar_r2`.
struct StderrCapture {
    saved: Option<i32>,
    path: PathBuf,
    tag: String,
}

impl StderrCapture {
    fn start(tag: &str) -> Result<Self> {
        use std::io::Write;
        let path = std::env::temp_dir()
            .join(format!("grammar10_{}_{}.log", tag, std::process::id()));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .context("open capture temp file")?;
        let _ = std::io::stderr().flush();
        // SAFETY: dup/dup2 on the process's own stderr; `file` stays open until the
        // dup2 completes (its fd is consumed by the kernel, not the File drop).
        let saved = unsafe { libc::dup(2) };
        anyhow::ensure!(saved >= 0, "dup(stderr) failed");
        let rc = unsafe { libc::dup2(file.as_raw_fd(), 2) };
        anyhow::ensure!(rc >= 0, "dup2(file, stderr) failed");
        drop(file);
        Ok(Self {
            saved: Some(saved),
            path,
            tag: tag.to_string(),
        })
    }

    fn restore(&mut self) {
        use std::io::Write;
        let _ = std::io::stderr().flush();
        if let Some(saved) = self.saved.take() {
            // SAFETY: restore the saved real stderr onto fd 2, then close the dup.
            unsafe {
                libc::dup2(saved, 2);
                libc::close(saved);
            }
        }
    }

    /// Restore fd 2, read + echo the captured text (so it lands in the test log),
    /// and return it for assertions.
    fn finish(mut self) -> String {
        self.restore();
        let content = std::fs::read_to_string(&self.path).unwrap_or_default();
        eprintln!(
            "---- captured engine stderr ({}) ----\n{content}\n---- end captured ({}) ----",
            self.tag, self.tag
        );
        let _ = std::fs::remove_file(&self.path);
        content
    }
}

impl Drop for StderrCapture {
    fn drop(&mut self) {
        self.restore();
        let _ = std::fs::remove_file(&self.path);
    }
}

/// One observed `fire` trace line's `(requests, distinct_programs)`.
#[derive(Clone, Copy, Debug)]
struct Fire {
    requests: usize,
    distinct: usize,
}

/// Parse every `[pie-sched-trace] … fire requests=N … distinct_programs=K` line.
fn parse_fires(trace: &str) -> Vec<Fire> {
    trace
        .lines()
        .filter(|l| l.contains("fire requests="))
        .filter_map(|l| {
            let requests = field(l, "fire requests=")?;
            let distinct = field(l, "distinct_programs=")?;
            Some(Fire { requests, distinct })
        })
        .collect()
}

/// Extract the `usize` immediately following `key` on a trace line.
fn field(line: &str, key: &str) -> Option<usize> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse::<usize>().ok()
}

/// Launch each `(program_name, input)` on its OWN client session (separate
/// sessions give genuine concurrency; the hold then coalesces their forwards),
/// obtaining ALL handles before awaiting any so their first forwards land in the
/// same drain window. Returns the result JSONs. Clients are kept alive until all
/// returns are collected.
async fn launch_cobatch(
    listen_addr: &std::net::SocketAddr,
    reqs: &[(String, String)],
) -> Result<Vec<String>> {
    let mut clients = Vec::with_capacity(reqs.len());
    let mut procs = Vec::with_capacity(reqs.len());
    for (program, input) in reqs {
        let c = Client::connect_with_identity(&format!("ws://{listen_addr}/v1/ws"), "test-user")
            .await
            .context("connect proc session")?;
        c.authenticate("test-user", &None)
            .await
            .context("auth proc session")?;
        let p = c
            .launch_process(program.clone(), input.clone(), true)
            .await
            .with_context(|| format!("launch {program}"))?;
        procs.push(p);
        clients.push(c);
    }
    let mut results = Vec::with_capacity(reqs.len());
    for (i, mut proc) in procs.into_iter().enumerate() {
        let json = proc
            .wait_for_return()
            .await
            .with_context(|| format!("wait_for_return proc {i}"))?;
        results.push(json);
    }
    drop(clients);
    Ok(results)
}

fn build_wasm(ws: &Path, packages: &[&str]) -> Result<()> {
    let mut args = vec!["build", "--target", "wasm32-wasip2"];
    for p in packages {
        args.push("-p");
        args.push(p);
    }
    let ok = Command::new("cargo")
        .args(&args)
        .current_dir(ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for {packages:?}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#10-verify: needs the 4090 + cuda + qwen-3-0.6b + the #10 base (alpha policy + bravo attach-hash)"]
async fn grammar10_distinct_program_accounting_on_real_driver() -> Result<()> {
    // Scheduler levers MUST be set pre-boot (OnceLock / run-loop reads).
    // SAFETY: set before any engine threads spawn.
    unsafe {
        if std::env::var_os("PIE_SCHED_ACCUM_HOLD_US").is_none() {
            std::env::set_var("PIE_SCHED_ACCUM_HOLD_US", ACCUM_HOLD_US);
        }
        std::env::set_var("PIE_SCHED_TRACE", "1");
    }
    let hold_us =
        std::env::var("PIE_SCHED_ACCUM_HOLD_US").unwrap_or_else(|_| ACCUM_HOLD_US.to_string());
    common::init_trace();

    // Build all inferlets used across the gates (one cargo invocation) BEFORE the
    // captures so the verbose build logs stay out of the trace files.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    build_wasm(&ws, &["grammar-late", "grammar", "mirostat"])?;

    let mask_input = format!("{{\"alphabet\":{ALPHABET:?},\"max_tokens\":{MAX_TOKENS}}}");
    let plain_input = format!("{{\"max_tokens\":{MAX_TOKENS}}}");

    let pie = common::boot_4090().await?;
    eprintln!("[grammar10] booted, listen_addr={}", pie.listen_addr);

    // Install all programs once (one setup session).
    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup.authenticate("test-user", &None).await.context("auth setup")?;
    for (pkg, manifest) in [
        ("grammar_late", "grammar-late"),
        ("grammar", "grammar"),
        ("mirostat", "mirostat"),
    ] {
        let wasm = ws.join(format!("target/wasm32-wasip2/debug/{pkg}.wasm"));
        let man = ws.join(format!("{manifest}/Pie.toml"));
        setup
            .add_program(&wasm, &man, true)
            .await
            .with_context(|| format!("add_program {pkg}"))?;
    }
    eprintln!("[grammar10] programs installed; PIE_SCHED_ACCUM_HOLD_US={hold_us}");

    // ── G1 DEDUP: 2 IDENTICAL grammar-late → distinct_programs = 1 ────────────
    let g1_cap = StderrCapture::start("G1-dedup")?;
    let g1 = launch_cobatch(
        &pie.listen_addr,
        &[
            ("grammar-late@0.1.0".into(), mask_input.clone()),
            ("grammar-late@0.1.0".into(), mask_input.clone()),
        ],
    )
    .await;
    let g1_trace = g1_cap.finish();
    let g1 = g1?;
    let g1_fires = parse_fires(&g1_trace);
    eprintln!("[grammar10] G1 results={g1:?}");
    eprintln!("[grammar10] G1 fires={g1_fires:?}");
    // The dedup witness: a co-batched fire (requests≥2) whose distinct count is 1.
    let g1_dedup = g1_fires
        .iter()
        .find(|f| f.requests >= 2 && f.distinct == 1)
        .copied();
    anyhow::ensure!(
        g1_dedup.is_some(),
        "G1 DEDUP ABSENT — no `fire requests>=2 … distinct_programs=1` (2 identical \
         grammar-late should co-batch AND dedup to one distinct program). Fires: \
         {g1_fires:?}. If only `requests=1` lines: the hold didn't coalesce — bump \
         PIE_SCHED_ACCUM_HOLD_US (current {hold_us}µs)."
    );
    let g1_distinct = g1_dedup.unwrap().distinct;

    // ── G2 DISTINCT: 3 DISTINCT programs → distinct_programs ≥ 2 (> G1) ───────
    let g2_cap = StderrCapture::start("G2-distinct")?;
    let g2 = launch_cobatch(
        &pie.listen_addr,
        &[
            ("grammar-late@0.1.0".into(), mask_input.clone()),
            ("grammar@0.1.0".into(), plain_input.clone()),
            ("mirostat@0.1.0".into(), plain_input.clone()),
        ],
    )
    .await;
    let g2_trace = g2_cap.finish();
    let g2 = g2?;
    let g2_fires = parse_fires(&g2_trace);
    eprintln!("[grammar10] G2 results={g2:?}");
    eprintln!("[grammar10] G2 fires={g2_fires:?}");
    let g2_max_distinct = g2_fires.iter().map(|f| f.distinct).max().unwrap_or(0);
    eprintln!(
        "[grammar10] G2 max distinct_programs={g2_max_distinct} (vs G1 distinct={g1_distinct})"
    );
    // Distinct-counting witness: a co-batch of distinct programs yields a distinct
    // count > the dedup baseline (≥2). This is the load-bearing contrast — the
    // SAME co-batch mechanism gives K=1 for identical (G1) vs K≥2 for distinct.
    anyhow::ensure!(
        g2_max_distinct >= 2,
        "G2 DISTINCT FAILED — max distinct_programs={g2_max_distinct} (<2). Distinct \
         programs did not co-batch into a fire with ≥2 distinct identities. Fires: \
         {g2_fires:?}. (If all `distinct_programs=1`: either no co-batch — bump the \
         hold — or the programs hashed identically, which would be a real identity \
         bug.)"
    );
    anyhow::ensure!(
        g2_max_distinct > g1_distinct,
        "G2 vs G1 CONTRAST FAILED — distinct programs ({g2_max_distinct}) did not \
         exceed the dedup baseline ({g1_distinct}); the policy is not uniqueness-aware."
    );

    // ── G3 NO-REGRESSION: a SOLE request fires un-coalesced (requests=1) ──────
    let g3_cap = StderrCapture::start("G3-noregress")?;
    let g3 = launch_cobatch(
        &pie.listen_addr,
        &[("grammar-late@0.1.0".into(), mask_input.clone())],
    )
    .await;
    let g3_trace = g3_cap.finish();
    let g3 = g3?;
    let g3_fires = parse_fires(&g3_trace);
    eprintln!("[grammar10] G3 results={g3:?}");
    eprintln!("[grammar10] G3 fires={g3_fires:?}");
    anyhow::ensure!(
        g3_fires.iter().any(|f| f.requests == 1),
        "G3 NO-REGRESSION FAILED — no `fire requests=1` for a sole request (a sparse \
         arrival must not be inflated into a phantom co-batch). Fires: {g3_fires:?}."
    );

    pie.shutdown().await;

    eprintln!(
        "[grammar10] #10-verify GREEN: G1 dedup (requests≥2, distinct_programs={g1_distinct}) ∧ \
         G2 distinct (max distinct_programs={g2_max_distinct} > {g1_distinct}) ∧ \
         G3 no-regression (sole request fires requests=1)."
    );
    Ok(())
}
