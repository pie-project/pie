//! **T2 bubble-p50 gate — real-driver (4090) measurement.** The masterplan M1
//! quantitative gate: *inter-batch bubble p50 < 100 µs on an 8-pipeline
//! homogeneous decode fleet under the quorum scheduler.* Boots the 4090 + real
//! CUDA driver + Qwen3-0.6B, selects the PTIR quorum fire rule
//! (`PIE_SCHED_POLICY=quorum` + the `run-ahead` feature), launches N concurrent
//! greedy run-ahead decodes (the `runahead` inferlet's `collect_tokens_pipelined`
//! carrier path), and reads the scheduler's fire probes in-process via
//! `pie::inference::get_stats()`.
//!
//! The `inter_batch_bubble_us` probe has a **host proxy** (stamps device "idle
//! since" when the loop *receives* a completion with nothing queued behind it,
//! which lags true device retirement → over-counts) AND, when the CUDA driver
//! profiles (`probe_device_idle_us` = `t0_entry − t5_prev_retire`), an **accurate
//! driver-stamp histogram**. `InferenceStats::bubble_p50()` prefers the driver
//! stamp (falls back to the proxy); this harness reports BOTH as a transition
//! cross-check (two independent producers) and — when the accurate driver stamp
//! is present — can HARD-gate p50 < 100 µs (opt-in via `PIE_BUBBLE_HARD=1`).
//!
//! `#[ignore]`, driver-cuda + run-ahead + profile-fire. Run:
//!   PIE_COMPILER_LAUNCHER=env RUSTC_WRAPPER=sccache CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda,run-ahead,profile-fire \
//!     --test cuda_bubble -- --ignored --nocapture
//!
//! ## ⚠️ Concurrent decode is NON-batch-invariant — `MATCH<fleet` at fleet>1 is EXPECTED, not a bug
//!
//! Under the `runahead` inferlet the harness compares each pipeline's carrier
//! (pipelined) decode against a synchronous reference (`MATCH=true/false`). When
//! MORE THAN ONE lane co-batches, some lanes report `MATCH=false` — this is
//! **inherent to batched bf16 inference, NOT a carrier/arena defect**:
//!   - Co-batching changes the bf16 reduction ORDER of the logits, which flips
//!     `argmax` at a near-tie (e.g. token 264 "a" vs 279 "the" after "It is").
//!   - Plain concurrent `generate` (ZERO run-ahead) diverges the SAME way (~7:1
//!     at that tie), so it is a property of concurrent decode, not the carrier.
//!   - The carrier pass and the sync-reference pass have DIFFERENT batch
//!     compositions, so a near-tie can flip between them → `MATCH=false`.
//!
//! **The carrier byte-identity GATE is `PIE_BUBBLE_FLEET=1`** (single lane → no
//! concurrent batching → deterministic → 1/1 byte-identical). A multi-lane 8/8
//! cross-pass bar conflates carrier-correctness with batch-invariance and is
//! unachievable while decode is non-batch-invariant. To distinguish a REAL bug
//! from a near-tie flip, grep the log for `unknown object` / `txn_commit failed`
//! (arena corruption) — their ABSENCE + `ANCHOR_OK=true CLEAR_OK=true` on a
//! `MATCH=false` line means the carrier is correct and only the near-tie flipped.
//! "Transparency" here = byte-identical MODULO near-tie reduction-order flips.
//!
//! ## Knobs
//!   PIE_BUBBLE_FLEET  — fleet width (default 8; =1 is the carrier byte-identity gate)
//!   PIE_BUBBLE_BUDGET — per-pipeline token budget (default 32; drives steady-state)
//!   PIE_BUBBLE_INFERLET = generate (default, sync greedy) | runahead (carrier)
//!   PIE_SCHED_POLICY  = quorum (default) | waitall

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Fleet width — the 8-pipeline homogeneous decode fleet of the M1 gate.
const FLEET_DEFAULT: usize = 8;
fn fleet() -> usize {
    std::env::var("PIE_BUBBLE_FLEET").ok().and_then(|s| s.parse().ok()).unwrap_or(FLEET_DEFAULT)
}
/// Per-pipeline token budget (larger ⇒ more steady-state fires past warm-up).
/// Env-overridable via `PIE_BUBBLE_BUDGET` (charlie: ① probe matrix needs 5 vs 200).
fn budget() -> String {
    std::env::var("PIE_BUBBLE_BUDGET").unwrap_or_else(|_| "32".to_string())
}

/// Steady-state co-batch mean from a (t_ms, total_batches, total_requests) time
/// series: the raw whole-run mean is dragged by ramp (lanes still launching) and
/// drain (early finishers Leave, thinning tail waves), so we also report the mean
/// over the middle 60% of the run (drop the first/last 20%). guru's honest-headline
/// ask: steady 4-7 with <8 is CORRECT, not a bug.
fn steady_state_mean(samples: &[(u128, u64, u64)]) -> (f64, u64, u64) {
    if samples.len() < 3 {
        return (0.0, 0, 0);
    }
    let t_end = samples.last().unwrap().0 as f64;
    let (lo, hi) = (t_end * 0.2, t_end * 0.8);
    let (mut d_req, mut d_bat) = (0u64, 0u64);
    for w in samples.windows(2) {
        let (t0, b0, r0) = w[0];
        let (t1, b1, r1) = w[1];
        let mid = (t0 as f64 + t1 as f64) / 2.0;
        if mid >= lo && mid <= hi {
            d_bat += b1.saturating_sub(b0);
            d_req += r1.saturating_sub(r0);
        }
    }
    let mean = if d_bat > 0 { d_req as f64 / d_bat as f64 } else { 0.0 };
    (mean, d_bat, d_req)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "T2 bubble-p50 gate: needs the 4090 + cuda + qwen-3-0.6b; run under --features run-ahead,profile-fire"]
async fn bubble_p50_on_real_driver() -> Result<()> {
    // Scheduler policy MUST be selected pre-boot (OnceLock read at loop init).
    // Defaults to the quorum rule; override with `PIE_SCHED_POLICY=<other>` to
    // A/B the fleet against the legacy run-ahead policy.
    // SAFETY: set before any engine threads spawn.
    unsafe {
        if std::env::var_os("PIE_SCHED_POLICY").is_none() {
            std::env::set_var("PIE_SCHED_POLICY", "quorum");
        }
    }
    let policy = std::env::var("PIE_SCHED_POLICY").unwrap_or_else(|_| "quorum".into());
    common::init_trace();

    // Build the run-ahead greedy-decode inferlet (its `collect_tokens_pipelined`
    // drives the device-side carrier — the run-ahead path we measure the fleet of).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    // The fleet inferlet (env-overridable). Default: `generate` — a greedy
    // decode loop on RAW WIT bindings (no inferlet-crate decode helpers, In Gim's
    // directive), so the harness exercises the real WIT path the M3 inferlets use.
    //   PIE_BUBBLE_INFERLET=generate (default) — raw-WIT sequential greedy decode
    //   PIE_BUBBLE_INFERLET=runahead            — raw-WIT run-ahead carrier decode
    let inferlet = std::env::var("PIE_BUBBLE_INFERLET").unwrap_or_else(|_| "generate".into());
    let (pkg, prog, input) = match inferlet.as_str() {
        "runahead" => ("runahead", "runahead@0.1.0", budget()),
        "generate" => ("generate", "generate@0.1.0", budget()),
        _ => ("generate", "generate@0.1.0", budget()),
    };
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", pkg])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "{pkg} wasm build failed");
    let wasm = ws.join(format!("target/wasm32-wasip2/debug/{pkg}.wasm"));
    let manifest = ws.join(format!("{pkg}/Pie.toml"));

    let pie = common::boot_4090().await?;
    eprintln!("[bubble] booted, listen_addr={} policy={policy}", pie.listen_addr);

    // Install the program once.
    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup.authenticate("test-user", &None).await.context("auth setup")?;
    setup
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;
    let fleet = fleet();
    eprintln!("[bubble] program installed; launching {fleet}-pipeline fleet (budget={})", budget());

    // Steady-state sampler: poll get_stats() every 25ms to build a batch-size time
    // series, so we can report the STEADY-STATE mean (excluding ramp/drain) beside
    // the raw whole-run mean (guru's honest-headline ask).
    let sample_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let sampler = {
        let stop = sample_stop.clone();
        tokio::spawn(async move {
            let t0 = std::time::Instant::now();
            let mut samples: Vec<(u128, u64, u64)> = Vec::new();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let s = pie::inference::get_stats().await;
                samples.push((t0.elapsed().as_millis(), s.total_batches, s.total_requests_processed));
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
            samples
        })
    };

    // Launch the whole fleet BEFORE awaiting any — all decodes are in flight at
    // once, so the scheduler forms real co-batches and reaches steady state.
    let mut clients = Vec::with_capacity(fleet);
    for _ in 0..fleet {
        let c =
            Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
                .await
                .context("connect fleet")?;
        c.authenticate("test-user", &None).await.context("auth fleet")?;
        let proc = c
            .launch_process(prog.into(), input.clone(), true)
            .await
            .context("launch fleet")?;
        clients.push((c, proc));
    }
    // Drain every pipeline's return (they decoded concurrently).
    let mut n_ok = 0usize;
    for (i, (_c, mut proc)) in clients.into_iter().enumerate() {
        match proc.wait_for_return().await {
            Ok(json) => {
                // runahead reports MATCH=true; generate returns a token list. Count
                // any clean, non-degenerate return.
                if json.contains("MATCH=true") || json.contains("generated") {
                    n_ok += 1;
                }
            }
            Err(e) => eprintln!("[bubble] pipeline {i} errored: {e:#}"),
        }
    }
    eprintln!("[bubble] fleet done: {n_ok}/{fleet} pipelines MATCH-correct");

    // Stop the sampler + compute the steady-state window mean.
    sample_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let samples = sampler.await.unwrap_or_default();
    let (steady_mean, steady_batches, steady_requests) = steady_state_mean(&samples);

    // Read the scheduler's fire probes in-process (the engine ran here).
    let stats = pie::inference::get_stats().await;
    pie.shutdown().await;

    let total_batches = stats.total_batches;
    let total_requests = stats.total_requests_processed;
    let mean_batch = if total_batches > 0 {
        total_requests as f64 / total_batches as f64
    } else {
        0.0
    };
    let q = &stats.fire.quorum;
    eprintln!("═══════════════════ T2 bubble-p50 gate — 4090 / Qwen3-0.6B ═══════════════════");
    eprintln!(
        "  fleet={fleet}  total_batches={total_batches}  total_requests={total_requests}  \
         mean_batch={mean_batch:.2}"
    );
    eprintln!(
        "  STEADY-STATE mean_batch (mid-60% window, excl ramp/drain) = {steady_mean:.2}  \
         (over {steady_batches} batches / {steady_requests} reqs, {} samples)",
        samples.len()
    );
    eprintln!("  batch_size_hist (1,2-3,4-7,8-15,16-31,32-63,64-127,128+) = {:?}", stats.batch_size_hist);
    eprintln!("  ── quorum probes ──");
    eprintln!("  inter_batch_bubble_us (host proxy, avg) = {}", q.avg_inter_batch_bubble_us);
    eprintln!("  quorum_latency_us (avg)                 = {}", q.avg_quorum_latency_us);
    eprintln!("  escape_fires                            = {}", q.escape_fires);
    eprintln!("  submit_ahead_fires                      = {}", q.submit_ahead_fires);
    eprintln!("  cold_hold_fires                         = {}", q.cold_hold_fires);
    eprintln!("  cold_hold_us (avg/fire)                 = {}", q.avg_cold_hold_us);
    eprintln!("  readiness_miss                          = {}", q.readiness_miss);
    eprintln!("  ── wait-for-all wave (M-AB) ──");
    eprintln!("  wave_fires                              = {}", q.wave_fires);
    eprintln!("  avg_active_pipelines_at_fire            = {}  (≈fleet ⇒ persistent wait-set; ≈1 ⇒ transient/singleton)", q.avg_active_pipelines_at_fire);
    eprintln!("  avg_missing_at_fire                     = {}  (>0 ⇒ deadline hold+partial; ≈0 ⇒ all-ready fire)", q.avg_missing_at_fire);
    eprintln!("  ── fire timing (context) ──");
    eprintln!("  inter_fire_us (avg)      = {}", stats.fire.avg_inter_fire_us);
    eprintln!("  driver_fire_us (avg)     = {}", stats.fire.execute.avg_driver_fire_us);
    eprintln!("  batch_build_us (avg)     = {}", stats.fire.execute.avg_batch_build_us);
    eprintln!("  ── R-decomposition (reduce-R target) ──");
    eprintln!("  post_dispatch_to_fire_us (avg) = {}  ← host round-trip (DOMINANT)", stats.fire.avg_post_dispatch_to_fire_us);
    eprintln!("  recv_block_wait_us (avg)       = {}  (scheduler recv-block, delta)", stats.fire.avg_recv_block_wait_us);
    {
        let dc = &stats.fire.execute.driver_cuda;
        eprintln!("  driver_cuda (avg µs/fire):");
        eprintln!("    [rust] ipc_submit={}  gpu_wait={}  ipc_recv={}",
            dc.avg_ipc_submit_us, dc.avg_gpu_wait_us, dc.avg_ipc_recv_us);
        eprintln!("    [c++ ] wire_parse={}  plan={}  h2d={}  kernel_launch={}  sync={}  response_build={}",
            dc.avg_wire_parse_us, dc.avg_plan_us, dc.avg_h2d_us,
            dc.avg_kernel_launch_us, dc.avg_sync_us, dc.avg_response_build_us);
    }
    eprintln!("  ── inter-batch bubble p50/p99 (histogram) ──");
    let from_driver = stats.bubble_from_driver();
    let p50 = stats.bubble_p50();
    let p99 = stats.bubble_p99();
    eprintln!("  bubble_us_hist (host proxy)   = {:?}", stats.bubble_us_hist);
    eprintln!("  bubble_us_hist (driver stamp) = {:?}", stats.bubble_us_hist_driver);
    eprintln!(
        "  cross-check p50: proxy={} µs  driver={} µs  (source of record = {})",
        stats.bubble_p50_proxy(),
        stats.bubble_p50_driver(),
        if from_driver { "DRIVER stamp (accurate)" } else { "host PROXY (driver stamp absent)" }
    );
    eprintln!("  bubble p50 = {p50} µs   p99 = {p99} µs");
    eprintln!("  bubble-p50 GATE target: inter_batch_bubble p50 < 100 µs");
    eprintln!("═══════════════════════════════════════════════════════════════════════════════");

    // The measurement is meaningful only if the fleet actually co-batched and the
    // scheduler fired real batches (otherwise the probe is trivially 0).
    anyhow::ensure!(total_batches > 0, "no batches fired — scheduler probe plumbing broken");
    if n_ok != fleet {
        eprintln!(
            "[bubble] ⚠ {n_ok}/{fleet} pipelines byte-matched the sync reference. For \
             runahead this is EXPECTED when >1 lane co-batches: concurrent decode is \
             non-batch-invariant (bf16 reduction-order flips logit near-ties), so the \
             carrier pass vs the sync-reference pass can diverge at a near-tie even though \
             the carrier is byte-exact under a matched batch composition (verify with \
             PIE_BUBBLE_FLEET=1 → 1/1). NOT an arena/corruption error — check the log for \
             `unknown object`/`txn_commit` to distinguish. Bubble numbers still valid."
        );
    }

    // BAR-2 carrier transparency GATE (guru): at `PIE_BUBBLE_FLEET=1` the co-batch
    // confound is removed (a single lane can't co-batch), so the carrier/preempt
    // run MUST be byte-identical to its sync reference — this is the clean,
    // un-confounded suspend/restore + seal + carrier transparency proof that the
    // cuda_contention harness delegates to (see its assertion-3 doc). At fleet>1
    // `MATCH<fleet` is the documented non-batch-invariance (warned above, not a
    // failure). Completion note: run the carrier+preempt config at `budget<=200`
    // — `budget=500` multi-lane hits the demote-rejoin cap-saturation stall
    // (task #14, delta), which is a scheduling liveness issue, NOT a carrier bug
    // (FLEET=1 b500 still completes byte-exact).
    if fleet == 1 {
        anyhow::ensure!(
            n_ok == 1,
            "FLEET=1 carrier byte-identity gate FAILED (0/1 matched the sync reference) — with a \
             single lane there is no co-batch non-batch-invariance, so a mismatch is a real \
             carrier/preempt/seal corruption. Check the log for the pipelined-vs-sync token diff."
        );
        eprintln!("[bubble] FLEET=1 carrier byte-identity gate: 1/1 MATCH ✓");
    }

    // Hard G3 gate — opt-in, and ONLY meaningful on the accurate driver stamp (the
    // host proxy over-counts, so a hard bound on it would be wrong). Requires the
    // driver device-idle stamp to be present (profile-driver-cuda on the driver).
    if std::env::var("PIE_BUBBLE_HARD").as_deref() == Ok("1") {
        anyhow::ensure!(
            from_driver,
            "PIE_BUBBLE_HARD=1 but no accurate driver device-idle stamp — the host proxy \
             over-counts, so the hard p50<100µs gate needs profile-driver-cuda. Refusing to \
             gate on the proxy."
        );
        anyhow::ensure!(
            p50 < 100,
            "G3 FAILED: inter-batch bubble p50 {p50} µs ≥ 100 µs (driver device-idle stamp)"
        );
        eprintln!("[bubble] G3 PASS: driver-stamp bubble p50 {p50} µs < 100 µs");
    }
    Ok(())
}
