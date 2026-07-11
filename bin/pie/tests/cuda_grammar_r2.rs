//! Gate-1: de-hardwired masking **merge-survival** verify (`forward_R=2`) — GPU
//! (bravo barrier ∥ alpha hold ∥ echo scatter ∥ delta fire). This is the LAST
//! masking seal: it proves the per-request grammar masks survive a genuine
//! BATCH-MERGED forward (`forward_R=2`, the cut-#1 supply-drift class), distinct
//! from item-1's concurrency claim (concurrent procs do NOT co-batch — run-ahead
//! serializes fires per-driver → R=1 each; `cuda_grammar_late` proves that).
//!
//! ## Why item-1's concurrency verify is NOT enough (the false-green trap)
//! On TWO independent `R=1` fires (`2×R=1`), each proc masks correctly on its own
//! fire — so `conform` AND `tok_A != tok_B` BOTH pass without any merge ever
//! happening. That re-proves concurrent masking (item-1, done), NOT merge-survival
//! (gate-1). The output asserts alone cannot tell `R=2` from `2×R=1`. So the gate
//! MUST hard-assert the merge actually happened — the load-bearing witness.
//!
//! ## The stack (delta folds + fires)
//!   - **alpha's hold** (`PIE_SCHED_ACCUM_HOLD_US`): the scheduler waits up to the
//!     hold for more requests before firing → the two procs' forwards land in ONE
//!     drain window → deterministic `forward_R=2` (without it, the A-wake→B-arrival
//!     race splits to `2×R=1`).
//!   - **echo's scatter**: the merged fire runs the per-program `per_req[r]` scatter
//!     (gated `num_programs>1 && num_programs==R`), marshaling EACH request's own
//!     grammar mask + token (no `per_req[0]→both` aliasing).
//!   - **bravo's barrier (this harness)**: two DISTINCT-alphabet grammar procs
//!     (A allows `[10..13]`, B allows `[20..23]`) on two sessions, so the
//!     constrained tokens are distinguishable BY CONSTRUCTION (`tok_A ∈ {10..13}` ≠
//!     `tok_B ∈ {20..23}`) — if `per_req[0]` aliased A's mask into B, B's token
//!     would land in `[10..13]` ⇒ B's grammar-violation check fails loud AND
//!     `tok_A == tok_B`.
//!
//! ## The locked gate (ALL must hold; any one missing ⇒ RED)
//!   1. **MERGE WITNESS** (load-bearing): `[pie-sched-trace] … fire requests=2`
//!      present (the hold coalesced 2→1 fire; `requests`=`batch.len()`=`forward_R`).
//!      A `2×R=1` signature (only `fire requests=1`) ⇒ the hold didn't coalesce ⇒
//!      RED, bump `PIE_SCHED_ACCUM_HOLD_US` (do NOT pass). `boot_4090` runs the
//!      engine IN-PROCESS, so this `eprintln!` lands on THIS process's fd 2 — the
//!      harness `dup2`'s fd 2 to a file to capture + assert it.
//!   2. **OWN-MASK CONFORM**: both procs `LATE_MASK_OK=true` (device token ==
//!      byte-identical CPU `apply_mask_argmax` ∧ natural argmax forced out ∧ token
//!      in ITS OWN alphabet).
//!   3. **DISTINGUISHABLE**: `tok_A[0] != tok_B[0]` (rules out `per_req[0]→both`).
//!
//! delta confirms the redundant driver-side witnesses on the same fire under
//! `PIE_SAMPLING_IR_TRACE=1` (`[ir-trace] forward_R=2` ∧ echo's `de-hardwire merged
//! multi-program HANDLED programs=2 R=2`). This harness sets that env too so they
//! are captured for the record.
//!
//! Run (GPU; needs alpha's hold + echo's scatter folded in):
//!   cargo test -p pie-bin --features driver-cuda --test cuda_grammar_r2 -- --ignored --nocapture

mod common;

use std::os::fd::AsRawFd;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// The two DISJOINT grammar alphabets. Disjoint ⇒ the constrained tokens are
/// distinguishable by construction, so a `per_req[0]→both` mask aliasing on the
/// merged fire is caught loud (B's token would land in A's alphabet).
const ALPHABET_A: [u32; 4] = [10, 11, 12, 13];
const ALPHABET_B: [u32; 4] = [20, 21, 22, 23];

/// Accumulation hold (µs) — default 50ms. The two procs' FIRST forwards are
/// prefills whose cross-proc startup gap exceeds 5ms (delta's gate-1 fire: 5ms ⇒
/// `2×R=1` RED, 50ms ⇒ genuine `forward_R=2` GREEN), so 50ms comfortably coalesces
/// them into one merged fire out-of-the-box. CLI-overridable via the env (below).
const ACCUM_HOLD_US: &str = "50000";

/// RAII capture of the process's stderr (fd 2) to a temp file. `boot_4090` runs
/// the engine IN-PROCESS, so the scheduler/driver trace witnesses are written to
/// THIS process's fd 2; capture lets the harness assert on them. Restores fd 2 on
/// `finish()` or `Drop` (so output is visible again even on early-return/panic).
struct StderrCapture {
    saved: Option<i32>,
    path: std::path::PathBuf,
}

impl StderrCapture {
    fn start() -> Result<Self> {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("gate1_r2_stderr_{}.log", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .context("open capture temp file")?;
        let _ = std::io::stderr().flush();
        // SAFETY: dup/dup2 on the process's own stderr; `file` stays open until the
        // dup2 completes (its fd is consumed by the kernel, not by the File drop).
        let saved = unsafe { libc::dup(2) };
        anyhow::ensure!(saved >= 0, "dup(stderr) failed");
        let rc = unsafe { libc::dup2(file.as_raw_fd(), 2) };
        anyhow::ensure!(rc >= 0, "dup2(file, stderr) failed");
        // `file` (the original fd) can drop now; fd 2 holds an independent dup.
        drop(file);
        Ok(Self {
            saved: Some(saved),
            path,
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

    /// Restore fd 2, read the captured text, echo it to the (restored) stderr so
    /// the trace is in the test log, and return it for assertions.
    fn finish(mut self) -> String {
        self.restore();
        let content = std::fs::read_to_string(&self.path).unwrap_or_default();
        eprintln!(
            "---- captured engine stderr (gate-1 R=2) ----\n{content}\n---- end captured stderr ----"
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

/// Parse the first generated token from an inferlet result line
/// (`… tokens=[N, M, …]`). The grammar-late inferlet reports `tokens={vec:?}`.
fn first_token(result: &str) -> Option<u32> {
    let start = result.find("tokens=[")? + "tokens=[".len();
    let rest = &result[start..];
    let end = rest.find(']')?;
    rest[..end]
        .split(',')
        .next()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse::<u32>().ok())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "gate-1 R=2 merge-survival: needs the 4090 + cuda + qwen-3-0.6b + alpha's PIE_SCHED_ACCUM_HOLD_US hold + echo's per-program scatter + the 0x65 mask kernel"]
async fn grammar_r2_merge_survival_on_real_driver() -> Result<()> {
    // Scheduler levers MUST be set before boot: `PIE_SCHED_TRACE` is cached on
    // first read (OnceLock) and `PIE_SCHED_ACCUM_HOLD_US` is read in the run loop.
    // The hold coalesces the two procs' forwards into ONE drain window ⇒ genuine
    // `forward_R=2`. `PIE_SAMPLING_IR_TRACE` surfaces delta's redundant driver
    // witnesses (`forward_R=2`, `programs=2 R=2`) for the record.
    //
    // SAFETY: set before any threads that read these envs are spawned (pre-boot).
    // Respect an existing PIE_SCHED_ACCUM_HOLD_US so the hold can be bumped beyond
    // the 50ms default from the CLI (e.g. `PIE_SCHED_ACCUM_HOLD_US=100000 cargo test
    // …`) without editing the harness, should the inter-proc gap ever exceed 50ms.
    unsafe {
        if std::env::var_os("PIE_SCHED_ACCUM_HOLD_US").is_none() {
            std::env::set_var("PIE_SCHED_ACCUM_HOLD_US", ACCUM_HOLD_US);
        }
        std::env::set_var("PIE_SCHED_TRACE", "1");
        std::env::set_var("PIE_SAMPLING_IR_TRACE", "1");
    }
    let hold_us =
        std::env::var("PIE_SCHED_ACCUM_HOLD_US").unwrap_or_else(|_| ACCUM_HOLD_US.to_string());
    common::init_trace();

    // Build the (alphabet-parameterized) grammar-late inferlet to wasm BEFORE the
    // capture, so the verbose cargo build logs stay live + out of the trace file.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "grammar-late"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "grammar-late wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/grammar_late.wasm");
    let manifest = ws.join("grammar-late/Pie.toml");

    let input_a = format!("{{\"alphabet\":{:?},\"max_tokens\":4}}", ALPHABET_A);
    let input_b = format!("{{\"alphabet\":{:?},\"max_tokens\":4}}", ALPHABET_B);

    // Everything that drives the engine runs INSIDE the capture so the in-process
    // scheduler/driver trace witnesses are recorded. We always `finish()` the
    // capture (restore fd 2 + dump) before propagating any error or asserting.
    let cap = StderrCapture::start()?;
    let run: Result<Vec<String>> = async {
        let pie = common::boot_4090().await?;
        eprintln!("[grammar-r2] booted, listen_addr={}", pie.listen_addr);

        // Install the program once (one setup session).
        let setup =
            Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
                .await
                .context("connect setup")?;
        setup
            .authenticate("test-user", &None)
            .await
            .context("auth setup")?;
        setup
            .add_program(&wasm, &manifest, true)
            .await
            .context("add_program")?;
        eprintln!(
            "[grammar-r2] program installed; launching 2 DISTINCT-alphabet decodes \
             (A={ALPHABET_A:?} / B={ALPHABET_B:?}) on SEPARATE sessions with \
             PIE_SCHED_ACCUM_HOLD_US={hold_us} → expect one merged forward_R=2 fire"
        );

        // Each proc on its OWN client session (item-1's lesson: gateway
        // session_driver serializes turns per session — separate sessions give
        // genuine concurrency; the hold then coalesces their forwards). Launch BOTH
        // back-to-back (both handles obtained before awaiting either) so their first
        // forwards arrive within the hold window. Keep clients alive until return.
        let inputs = [input_a.clone(), input_b.clone()];
        let mut clients = Vec::with_capacity(2);
        let mut procs = Vec::with_capacity(2);
        for input in inputs.iter() {
            let c = Client::connect_with_identity(
                &format!("ws://{}/v1/ws", pie.listen_addr),
                "test-user",
            )
            .await
            .context("connect proc session")?;
            c.authenticate("test-user", &None)
                .await
                .context("auth proc session")?;
            let p = c
                .launch_process("grammar-late@0.1.0".to_string(), input.clone(), true)
                .await
                .context("launch")?;
            procs.push(p);
            clients.push(c);
        }

        let mut results = Vec::with_capacity(2);
        for (i, mut proc) in procs.into_iter().enumerate() {
            let json = proc
                .wait_for_return()
                .await
                .with_context(|| format!("wait_for_return proc {i}"))?;
            eprintln!("[grammar-r2] proc {i} returned: {json}");
            results.push(json);
        }
        drop(clients);
        pie.shutdown().await;
        Ok(results)
    }
    .await;

    let captured = cap.finish();
    let results = run?;
    anyhow::ensure!(
        results.len() == 2,
        "expected 2 proc results, got {}",
        results.len()
    );

    // ── Witness 1 (load-bearing): the MERGE actually happened. ────────────────
    // `[pie-sched-trace] … fire requests=2` ⇒ the hold coalesced the two procs'
    // forwards into ONE fire (genuine forward_R=2). Only `fire requests=1` lines ⇒
    // 2×R=1 (hold didn't coalesce) ⇒ RED (bump the hold), NOT a pass.
    let n_merge = captured.matches("fire requests=2").count();
    let n_single = captured.matches("fire requests=1").count();
    // echo's redundant driver-side scatter witness (only emitted on the genuine
    // merged multi-program path; reported for cross-confirmation with delta).
    let driver_scatter_witness = captured.contains("programs=2 R=2");
    let driver_shape_witness = captured.contains("forward_R=2");
    eprintln!(
        "[grammar-r2] merge witnesses: fire_requests=2 ×{n_merge} (fire_requests=1 ×{n_single}); \
         driver scatter(programs=2 R=2)={driver_scatter_witness} shape(forward_R=2)={driver_shape_witness}"
    );
    anyhow::ensure!(
        n_merge >= 1,
        "MERGE WITNESS ABSENT — no `fire requests=2` in the scheduler trace (saw \
         {n_single}× `fire requests=1`). The hold did NOT coalesce the two forwards \
         into one fire ⇒ this is 2×R=1 (concurrent masking, already proven in \
         item-1), NOT merge-survival. Bump PIE_SCHED_ACCUM_HOLD_US (current {hold_us}µs); \
         do NOT pass."
    );

    // ── Witness 2: each proc conforms to its OWN mask. ────────────────────────
    for (i, json) in results.iter().enumerate() {
        anyhow::ensure!(
            json.contains("LATE_MASK_OK=true"),
            "OWN-MASK CONFORM FAILED (proc {i}) — the merged-fire per-request mask \
             diverged from the host CPU reference, or the disallowed natural argmax \
             was not forced out (a `per_req[0]→both` alias / dropped carrier on the \
             merge ⇒ wrong mask / SkippedLateBindMiss): {json}"
        );
    }

    // ── Witness 3: the two outputs are distinguishable (no per_req[0]→both alias). ─
    let tok_a = first_token(&results[0])
        .with_context(|| format!("parse tok_A from proc 0 result: {}", results[0]))?;
    let tok_b = first_token(&results[1])
        .with_context(|| format!("parse tok_B from proc 1 result: {}", results[1]))?;
    eprintln!(
        "[grammar-r2] tok_A[0]={tok_a} (∈{ALPHABET_A:?}?) tok_B[0]={tok_b} (∈{ALPHABET_B:?}?)"
    );
    anyhow::ensure!(
        ALPHABET_A.contains(&tok_a),
        "tok_A[0]={tok_a} not in alphabet A {ALPHABET_A:?} — proc A masked wrong"
    );
    anyhow::ensure!(
        ALPHABET_B.contains(&tok_b),
        "tok_B[0]={tok_b} not in alphabet B {ALPHABET_B:?} — proc B masked wrong \
         (if it's in A's alphabet, that's the per_req[0]→both aliasing bug)"
    );
    anyhow::ensure!(
        tok_a != tok_b,
        "DISTINGUISHABLE FAILED — tok_A[0]={tok_a} == tok_B[0]={tok_b}. Disjoint \
         alphabets should make these differ; equality ⇒ per_req[0]→both aliasing \
         (both requests got the same mask on the merge)."
    );

    eprintln!(
        "[grammar-r2] GATE-1 GREEN: merge-survival proven — forward_R=2 (fire requests=2 ×{n_merge}) \
         ∧ both LATE_MASK_OK=true (own-mask conform) ∧ tok_A[0]={tok_a} != tok_B[0]={tok_b}."
    );
    Ok(())
}
