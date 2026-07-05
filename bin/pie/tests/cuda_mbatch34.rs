//! #34 M-batch `num_rows=N` occupancy verify — GPU (delta).
//!
//! Drives N co-batched, identical-bytecode, DISJOINT-alphabet `grammarmb`
//! requests (single-`[Token]` `grammar(vocab)`, `Readiness::Late` mask → M-batch
//! ELIGIBLE) through the real driver with echo's env-gated M-batch dispatch
//! (`PIE_SAMPLING_IR_MBATCH`). The N identical sampling programs (same bytecode,
//! per-row Late masks) co-batch in the decode phase → the M-batch groups them into
//! ONE `num_rows=N` launch (fire-axis occupancy, the bench's 10-122x) instead of N
//! `num_rows=1` `fire_one`s.
//!
//! The honest non-degenerate bar (both required):
//!   1. **GROUPING WITNESS** — env-ON emits `[ir-trace] mbatch group n=N` (echo's
//!      permanent witness at `mbatched=true`). Asserts the M-batch ACTUALLY grouped
//!      (≥1 group of n≥2), ruling out a silent fallback-to-`fire_one` false-green
//!      (a dense/gather miss produces correct tokens but no occupancy). env-OFF
//!      MUST emit NO such line (the env gate works).
//!   2. **PER-REQUEST CONFORM** — every request's tokens under env-ON == its
//!      env-OFF (`num_rows=1` sequential) control, AND each token ∈ that request's
//!      DISJOINT alphabet (the inferlet's self-check). A wrong-grouping / mask
//!      scatter (request i gets request j's gathered mask) ⇒ token ∈ alphabet_j ∉
//!      alphabet_i ⇒ caught by both the alphabet check AND the ON≠OFF compare.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_mbatch34 -- --ignored --nocapture

mod common;

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Deterministic co-batch hold (µs) so the N requests coalesce into one drain
/// window (their decodes then co-batch into the M-batch-able fire).
const ACCUM_HOLD_US: &str = "120000";
const MAX_TOKENS: u32 = 6;

/// N requests with DISJOINT alphabets — same `grammar` bytecode (dedup → one
/// distinct program, contiguous), per-row Late masks, distinct constrained tokens.
const ALPHABETS: [[u32; 4]; 4] = [
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
    [40, 41, 42, 43],
];

/// RAII capture of fd 2 → a temp file (the in-process engine writes `[ir-trace]`
/// there). Restores fd 2 on `finish()`/`Drop`. Mirrors `cuda_grammar10`.
struct StderrCapture {
    saved: Option<i32>,
    path: PathBuf,
    tag: String,
}

impl StderrCapture {
    fn start(tag: &str) -> Result<Self> {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("mbatch34_{}_{}.log", tag, std::process::id()));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .context("open capture temp file")?;
        let _ = std::io::stderr().flush();
        // SAFETY: dup/dup2 on the process's own stderr.
        let saved = unsafe { libc::dup(2) };
        anyhow::ensure!(saved >= 0, "dup(stderr) failed");
        let rc = unsafe { libc::dup2(file.as_raw_fd(), 2) };
        anyhow::ensure!(rc >= 0, "dup2(file, stderr) failed");
        drop(file);
        Ok(Self { saved: Some(saved), path, tag: tag.to_string() })
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

/// Launch each request on its OWN client session (genuine concurrency; the hold
/// coalesces their forwards), obtaining ALL handles before awaiting any so their
/// decodes land in the same drain window. Returns the result JSONs in order.
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
        c.authenticate("test-user", &None).await.context("auth proc session")?;
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

/// The max `n` over all `[ir-trace] mbatch group n=N …` lines (0 ⇒ none present).
fn max_mbatch_group_n(trace: &str) -> u32 {
    trace
        .lines()
        .filter(|l| l.contains("mbatch group n="))
        .filter_map(|l| {
            let s = l.find("mbatch group n=")? + "mbatch group n=".len();
            let rest = &l[s..];
            let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
            rest[..end].parse::<u32>().ok()
        })
        .max()
        .unwrap_or(0)
}

/// Parse `tokens=[a, b, c]` from a `grammarmb` result JSON.
fn parse_tokens(json: &str) -> Option<Vec<u32>> {
    let s = json.find("tokens=[")? + "tokens=[".len();
    let rest = &json[s..];
    let end = rest.find(']')?;
    Some(
        rest[..end]
            .split(',')
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect(),
    )
}

fn build_wasm(ws: &Path, pkg: &str) -> Result<()> {
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", pkg])
        .current_dir(ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for {pkg}");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#34 M-batch occupancy verify: needs the 4090 + cuda + qwen-3-0.6b + echo's env-gated M-batch"]
async fn mbatch34_num_rows_n_occupancy_on_real_driver() -> Result<()> {
    // Scheduler co-batch hold + ir-trace MUST be set pre-boot (OnceLock / run-loop).
    // SAFETY: set before any engine threads spawn.
    unsafe {
        if std::env::var_os("PIE_SCHED_ACCUM_HOLD_US").is_none() {
            std::env::set_var("PIE_SCHED_ACCUM_HOLD_US", ACCUM_HOLD_US);
        }
        std::env::set_var("PIE_SAMPLING_IR_TRACE", "1");
        // Start with M-batch OFF (the control); toggled per phase below.
        std::env::remove_var("PIE_SAMPLING_IR_MBATCH");
    }
    common::init_trace();

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    build_wasm(&ws, "grammarmb")?;

    let pie = common::boot_4090().await?;
    eprintln!("[mbatch34] booted, listen_addr={}", pie.listen_addr);

    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup.authenticate("test-user", &None).await.context("auth setup")?;
    let wasm = ws.join("target/wasm32-wasip2/debug/grammarmb.wasm");
    let man = ws.join("grammarmb/Pie.toml");
    setup.add_program(&wasm, &man, true).await.context("add_program grammarmb")?;
    eprintln!("[mbatch34] grammarmb installed");

    let reqs: Vec<(String, String)> = ALPHABETS
        .iter()
        .map(|a| {
            (
                "grammarmb@0.1.0".to_string(),
                format!("{{\"alphabet\":{a:?},\"max_tokens\":{MAX_TOKENS}}}"),
            )
        })
        .collect();
    let n = reqs.len() as u32;

    // ── Phase OFF (control): M-batch unset → individual num_rows=1 fires. ──
    let off_cap = StderrCapture::start("OFF")?;
    let off = launch_cobatch(&pie.listen_addr, &reqs).await;
    let off_trace = off_cap.finish();
    let off = off?;
    eprintln!("[mbatch34] OFF results={off:?}");
    let off_group_n = max_mbatch_group_n(&off_trace);
    let off_tokens: Vec<Option<Vec<u32>>> = off.iter().map(|j| parse_tokens(j)).collect();
    let off_all_ok = off.iter().all(|j| j.contains("GRAMMARMB_OK=true"));

    // ── Phase ON (test): M-batch on → group N into one num_rows=N fire. ──
    // SAFETY: getenv is read per-dispatch (not cached), so toggling here re-arms it.
    unsafe { std::env::set_var("PIE_SAMPLING_IR_MBATCH", "1") };
    let on_cap = StderrCapture::start("ON")?;
    let on = launch_cobatch(&pie.listen_addr, &reqs).await;
    let on_trace = on_cap.finish();
    let on = on?;
    eprintln!("[mbatch34] ON results={on:?}");
    let on_group_n = max_mbatch_group_n(&on_trace);
    let on_tokens: Vec<Option<Vec<u32>>> = on.iter().map(|j| parse_tokens(j)).collect();
    let on_all_ok = on.iter().all(|j| j.contains("GRAMMARMB_OK=true"));

    pie.shutdown().await;

    eprintln!(
        "[mbatch34] off_group_n={off_group_n} on_group_n={on_group_n} \
         off_all_ok={off_all_ok} on_all_ok={on_all_ok}\n  off_tokens={off_tokens:?}\n  on_tokens={on_tokens:?}"
    );

    // ── (0) baseline: the OFF control must itself be correct + grammar-conformant. ──
    anyhow::ensure!(
        off_all_ok,
        "OFF control degenerate — a grammarmb request was not GRAMMARMB_OK \
         (constrained token left its alphabet) BEFORE M-batch: {off:?}"
    );
    // ── (1a) the env gate: OFF must NOT group (no witness when unset). ──
    anyhow::ensure!(
        off_group_n == 0,
        "env gate BROKEN — `mbatch group` witnessed with PIE_SAMPLING_IR_MBATCH \
         unset (max n={off_group_n}); the default path must be byte-identical: {off_trace:?}"
    );
    // ── (1b) GROUPING WITNESS: ON must group ≥2 rows into one num_rows=N fire. ──
    anyhow::ensure!(
        on_group_n >= 2,
        "M-batch did NOT group — no `[ir-trace] mbatch group n>=2` with \
         PIE_SAMPLING_IR_MBATCH=1 on {n} identical contiguous grammars. A silent \
         fallback to per-row fire_one (dense/gather miss) produces correct tokens \
         but NO occupancy = false-green; this gate catches it. ON trace: {on_trace:?}"
    );
    // ── (2a) per-request grammar conform under M-batch (each token in its alphabet). ──
    anyhow::ensure!(
        on_all_ok,
        "M-batch CONFORM failed — a request's constrained token left its DISJOINT \
         alphabet under num_rows=N grouping (a mask scatter: request got another \
         row's gathered mask): {on:?}"
    );
    // ── (2b) per-request token-exactness: ON == the OFF (num_rows=1) control. ──
    anyhow::ensure!(
        on_tokens == off_tokens && on_tokens.iter().all(|t| t.is_some()),
        "M-batch token MISMATCH — a request's grouped (num_rows=N) tokens differ \
         from its sequential (num_rows=1) control ⇒ wrong-grouping/scatter:\n  \
         off={off_tokens:?}\n  on={on_tokens:?}"
    );

    eprintln!(
        "[mbatch34] PASS — M-batch grouped n={on_group_n} (num_rows=N occupancy) \
         with per-request conform (tokens == sequential control, each in its disjoint \
         alphabet); env-off control byte-identical (no grouping)."
    );
    Ok(())
}
