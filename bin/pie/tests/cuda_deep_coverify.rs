//! **Piece-2 co-verify — carrier deep pre-submission × wait-for-all (capstone
//! precondition C3, the carrier 8× lever).** Boots the 4090 + real CUDA driver +
//! Qwen3-0.6B, selects the wait-for-all fire rule (`PIE_SCHED_POLICY=waitall`)
//! at depth `PIE_SCHED_MAX_IN_FLIGHT=k`, launches an 8-pipeline fleet of bravo's
//! device-resident deep-pre-submission decode (`decode_pipelined_deep(k)`), and
//! asserts BOTH:
//!
//!   1. **DEEP-k byte-identity** — every pipeline's k-deep carrier chain decodes
//!      byte-identical to the synchronous greedy reference (`MATCH=true` /
//!      `DEEP*_MATCH=true`, + `ANCHOR_OK` non-degeneracy when the inferlet emits
//!      it). Greedy ⇒ any divergence is a real chain-ordering / carrier / WAR bug.
//!   2. **Co-batch density** (delta's M-AB gauges on the wait-for-all path):
//!      `avg_active_pipelines_at_fire ≈ FLEET` (the wait-set is PERSISTENT, not
//!      transient), `avg_missing_at_fire ≈ 0` (waves fire all-ready, not on the
//!      deadline), and `mean_batch ≈ FLEET` (the 8× lift the carrier unlocks by
//!      cutting the round-trip so pipelines cycle fast). Printed always;
//!      hard-gated opt-in via `PIE_DEEP_HARD=1`.
//!
//! This is the turnkey acceptance for the delta N+k chain-stash
//! (`stash_chain_continuation`) × bravo `decode_pipelined_deep` × cap=k firing.
//! The chain-stash keeps each pipeline's N+1…N+k links in FIFO order (never
//! co-batched with their own head) while co-batching ACROSS pipelines into dense
//! waves — so byte-identity (1) and density (2) hold together.
//!
//! `#[ignore]`, driver-cuda + profile-fire. Run:
//!   PIE_COMPILER_LAUNCHER=env RUSTC_WRAPPER=sccache CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda,profile-fire \
//!     --test cuda_deep_coverify -- --ignored --nocapture
//!
//! Config knobs (finalized to bravo's `lowlevel-chat` pilot on
//! `bravo-carrier-prod` — env-overridable):
//!   PIE_DEEP_INFERLET = the deep-pre-submission inferlet pkg (default `lowlevel-chat`)
//!   PIE_DEEP_DEPTH    = the chain depth k = the in-flight cap (default 4)
//!   PIE_DEEP_INPUT    = the launch input the pilot parses (default `"32 depth=<k>"`:
//!                       `<max_tokens>` + `depth=<k>` pre-submission window)
//!   PIE_DEEP_HARD=1   = hard-gate the density thresholds (byte-identity is always
//!                       hard-gated)

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Fleet width — the 8-pipeline homogeneous decode fleet.
const FLEET: usize = 8;

/// Chain depth k = the in-flight cap. bravo's `decode_pipelined_deep(k)` submits
/// a k-deep carrier chain up front; the wave fires k deep at cap=k. Overridable
/// via `PIE_DEEP_DEPTH` (kept == `PIE_SCHED_MAX_IN_FLIGHT`).
fn depth() -> usize {
    std::env::var("PIE_DEEP_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&k| k >= 1)
        .unwrap_or(4)
}

/// The deep-pre-submission inferlet package name — bravo's `lowlevel-chat`
/// pilot (`decode_pipelined_deep_eos`, on `bravo-carrier-prod`). Override via
/// `PIE_DEEP_INFERLET`.
fn deep_inferlet() -> String {
    std::env::var("PIE_DEEP_INFERLET").unwrap_or_else(|_| "lowlevel-chat".to_string())
}

/// The launch input the pilot parses: `"<max_tokens> depth=<k>"` — bravo's
/// `lowlevel-chat` reads the first whitespace token as `max_tokens` and
/// `depth=<k>` as the deep carrier's pre-submission window (aligned to
/// `PIE_SCHED_MAX_IN_FLIGHT=k`). Override via `PIE_DEEP_INPUT`.
fn deep_input() -> String {
    std::env::var("PIE_DEEP_INPUT").unwrap_or_else(|_| format!("32 depth={}", depth()))
}

/// A per-pipeline return is DEEP-k byte-identity-correct iff it reports
/// `DEEP_MATCH=true` (the depth-k carrier == synchronous greedy stream — the
/// primary gate). A `DEEP_MATCH=false` (or a degenerate `ANCHOR_OK=false`) is a
/// HARD fail even if the depth-1 `MATCH` passed. Falls back to a generic
/// `MATCH=true` for deep inferlets that don't emit `DEEP_MATCH`.
fn pipeline_match_ok(json: &str) -> bool {
    let deep_true = json.contains("DEEP_MATCH=true")
        || json.contains("DEEP4_MATCH=true")
        || json.contains("DEEPK_MATCH=true");
    let generic_true = json.contains("MATCH=true");
    let failed = json.contains("DEEP_MATCH=false") || json.contains("ANCHOR_OK=false");
    (deep_true || generic_true) && !failed
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "piece-2 co-verify: needs the 4090 + cuda + qwen-3-0.6b + decode_pipelined_deep"]
async fn deep_presubmit_coverify_on_real_driver() -> Result<()> {
    let k = depth();
    // Scheduler policy + cap MUST be selected pre-boot (OnceLock read at loop
    // init). Wait-for-all + cap=k = the carrier's back-to-back firing depth.
    // SAFETY: set before any engine threads spawn.
    unsafe {
        std::env::set_var("PIE_SCHED_POLICY", "waitall");
        std::env::set_var("PIE_SCHED_MAX_IN_FLIGHT", k.to_string());
        // Expose the depth to the inferlet build/launch side too (belt + braces
        // with the launch input, in case bravo's loop reads an env for depth).
        std::env::set_var("PIE_DEEP_DEPTH", k.to_string());
    }
    common::init_trace();

    let pkg = deep_inferlet();
    let prog = format!("{pkg}@0.1.0");
    let input = deep_input();

    // Build the deep-pre-submission inferlet (wasm32-wasip2, raw-WIT carrier).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", &pkg])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "{pkg} wasm build failed");
    let wasm = ws.join(format!("target/wasm32-wasip2/debug/{pkg}.wasm"));
    let manifest = ws.join(format!("{pkg}/Pie.toml"));

    let pie = common::boot_4090().await?;
    eprintln!(
        "[deep-coverify] booted, listen_addr={}  policy=waitall  cap={k}  inferlet={pkg}  input={input}",
        pie.listen_addr
    );

    // Install the program once.
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
    eprintln!("[deep-coverify] program installed; launching {FLEET}-pipeline deep(k={k}) fleet");

    // Launch the whole fleet BEFORE awaiting any — all deep chains are in flight
    // at once, so the wait-for-all scheduler forms dense cross-pipeline waves.
    let mut clients = Vec::with_capacity(FLEET);
    for _ in 0..FLEET {
        let c =
            Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
                .await
                .context("connect fleet")?;
        c.authenticate("test-user", &None)
            .await
            .context("auth fleet")?;
        let proc = c
            .launch_process(prog.clone(), input.clone(), true)
            .await
            .context("launch fleet")?;
        clients.push((c, proc));
    }

    // Drain every pipeline's return + tally byte-identity correctness.
    let mut n_match = 0usize;
    let mut first_bad: Option<String> = None;
    for (i, (_c, mut proc)) in clients.into_iter().enumerate() {
        match proc.wait_for_return().await {
            Ok(json) => {
                if pipeline_match_ok(&json) {
                    n_match += 1;
                } else {
                    eprintln!("[deep-coverify] pipeline {i} NOT byte-identical: {json}");
                    first_bad.get_or_insert(json);
                }
            }
            Err(e) => {
                eprintln!("[deep-coverify] pipeline {i} errored: {e:#}");
                first_bad.get_or_insert(format!("errored: {e:#}"));
            }
        }
    }
    eprintln!("[deep-coverify] fleet done: {n_match}/{FLEET} pipelines DEEP-k byte-identical");

    // Read the wait-for-all wave gauges in-process (the engine ran here).
    let stats = pie_engine::scheduler::get_stats().await;
    pie.shutdown().await;

    let total_batches = stats.total_batches;
    let total_requests = stats.total_requests_processed;
    let mean_batch = if total_batches > 0 {
        total_requests as f64 / total_batches as f64
    } else {
        0.0
    };
    let q = &stats.fire.quorum;
    eprintln!(
        "═══════════ piece-2 co-verify — deep pre-submission × wait-for-all (cap={k}) ═══════════"
    );
    eprintln!(
        "  fleet={FLEET}  depth(k)={k}  total_batches={total_batches}  total_requests={total_requests}  mean_batch={mean_batch:.2}"
    );
    eprintln!(
        "  batch_size_hist (1,2-3,4-7,8-15,16-31,32-63,64-127,128+) = {:?}",
        stats.batch_size_hist
    );
    eprintln!("  ── wait-for-all wave (M-AB) ──");
    eprintln!("  wave_fires                     = {}", q.wave_fires);
    eprintln!(
        "  avg_active_pipelines_at_fire   = {}   (≈{FLEET} ⇒ persistent wait-set; ≈1 ⇒ transient)",
        q.avg_active_pipelines_at_fire
    );
    eprintln!(
        "  avg_missing_at_fire            = {}   (≈0 ⇒ dense all-ready fire; >0 ⇒ deadline hold+partial)",
        q.avg_missing_at_fire
    );
    eprintln!("  escape_fires                   = {}", q.escape_fires);
    eprintln!("  cold_hold_fires                = {}", q.cold_hold_fires);
    eprintln!("  ── carrier reduce-R (context) ──");
    eprintln!(
        "  post_dispatch_to_fire_us (avg) = {}   (host round-trip — the carrier's target)",
        stats.fire.avg_post_dispatch_to_fire_us
    );
    eprintln!(
        "═══════════════════════════════════════════════════════════════════════════════════════"
    );

    // ── Gate 1: DEEP-k byte-identity (ALWAYS hard) ──────────────────────────
    // The scheduler chain-stash + the carrier must produce the exact greedy
    // stream; any miss is a real ordering/carrier/WAR corruption.
    anyhow::ensure!(
        total_batches > 0,
        "no batches fired — scheduler/probe plumbing broken (nothing to verify)"
    );
    anyhow::ensure!(
        n_match == FLEET,
        "DEEP-{k} byte-identity FAILED: only {n_match}/{FLEET} pipelines matched the synchronous \
         greedy reference — chain-ordering / carrier / WAR corruption. First bad: {}",
        first_bad.as_deref().unwrap_or("<none captured>")
    );
    eprintln!("[deep-coverify] ✅ DEEP-{k} byte-identity: {FLEET}/{FLEET} pipelines exact");

    // ── Gate 2: co-batch density (opt-in hard via PIE_DEEP_HARD=1) ──────────
    // The carrier cuts the round-trip → pipelines cycle fast → the wait-for-all
    // wave co-batches all FLEET pipelines' same-depth links → mean_batch ≈ FLEET
    // with a persistent wait-set (avg_active ≈ FLEET) and no deadline holds
    // (avg_missing ≈ 0). Printed always; enforced only when explicitly gating,
    // since density depends on the carrier landing + the reduce-R spacing.
    if std::env::var("PIE_DEEP_HARD").as_deref() == Ok("1") {
        // 0.75·FLEET floor: allows warm-up/tail singleton fires while proving the
        // steady state is dense (a transient/singleton bug reads mean_batch≈1).
        let dense_floor = (FLEET as f64) * 0.75;
        anyhow::ensure!(
            mean_batch >= dense_floor,
            "DENSITY FAILED: mean_batch {mean_batch:.2} < {dense_floor:.2} (0.75·FLEET) — the \
             wave is NOT co-batching the fleet (transient wait-set or the carrier not cutting R)"
        );
        anyhow::ensure!(
            q.wave_fires > 0 && q.avg_active_pipelines_at_fire as usize >= FLEET - 1,
            "DENSITY FAILED: avg_active_pipelines_at_fire {} < {} (FLEET-1) — the wait-set is \
             TRANSIENT, not persistent (membership not accumulating across waves)",
            q.avg_active_pipelines_at_fire,
            FLEET - 1
        );
        anyhow::ensure!(
            q.avg_missing_at_fire <= 1,
            "DENSITY FAILED: avg_missing_at_fire {} > 1 — waves are HOLDING to the deadline and \
             firing partial (stragglers absent within the deadline; reduce-R spacing too wide)",
            q.avg_missing_at_fire
        );
        eprintln!(
            "[deep-coverify] ✅ DENSITY: mean_batch={mean_batch:.2}  avg_active={}  avg_missing={}",
            q.avg_active_pipelines_at_fire, q.avg_missing_at_fire
        );
    }

    Ok(())
}
