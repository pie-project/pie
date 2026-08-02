//! **Task-B KV preempt/restore over-capacity e2e** (alpha — the M-AB validation
//! gate for the contention rework).
//!
//! The rework's headline claim (In Gim): KV contention is **preempt/restore, not
//! admission**. When a fleet's combined KV demand EXCEEDS the physical pool, an
//! alloc-fail must NOT surface a `WorkingSetError` to the inferlet — instead it
//! routes to the contention orchestrator (always on):
//!   OOM in prep → `acquire()` (FCFS wait / preempt a younger process) → a
//!   completing/terminating process frees its pages (the WS-drop drain hook) →
//!   `on_blocks_freed` wakes the FIFO waiter → the prep retries → succeeds.
//!
//! This test proves it end-to-end + that preemption is TRANSPARENT. Transparency
//! is established by a COMPOSITE proof, because the naive "concurrent tokens ==
//! solo tokens" check is structurally confounded — even ENGAGED: engagement
//! REQUIRES co-batching (a single lane never contends), and co-batched decode is
//! non-batch-invariant at logit near-ties (batch composition → bf16 reduction
//! order → a flipped argmax), so a lane can diverge from its batch-of-1 solo
//! reference with NO KV corruption (verified: tp=16 engaged, suspends=101, still
//! ~9 near-tie flips). The composite (charlie's Phase-2 finding):
//!   (a) the surviving `cuda_concurrent` single-lane reference — SEAL + carrier transparency with the
//!       co-batch confound REMOVED (a single long-decode lane forced to preempt
//!       via `[batching].total_pages`, byte-identical to its sync reference);
//!   (b) engaged multi-lane `restore_attributable == 0` HERE — a divergence first
//!       appearing at KV position >= page_size (a SEALED + restored page) is real
//!       suspend/restore corruption, hard-gated (assertion 3);
//!   (c) position-0 batch-invariance HERE — the first token off the identical
//!       prefill is deterministic, so a position-0 divergence is real prefill/
//!       page-0 corruption, hard-gated (assertion 3);
//!   (d) engagement counters (assertions 2+4) — preempt/restore provably ran.
//!   Divergences in the [1, page_size) band are the documented non-batch-
//!   invariance confound (logged, non-gating) — (a) covers transparency there.
//!
//! Assertions (engagement checked FIRST — a dormant config fails actionably on ②
//! rather than as a non-determinism trap on ③):
//!   1. **Zero WorkingSetError** — every pipeline in an over-capacity fleet
//!      completes with a token stream (none fails with "out of blocks"). This is
//!      the preempt/restore correctness: waiters block + wake, never error.
//!   2. **Engagement** — the orchestrator's contention counters prove preempt/
//!      restore actually happened: `waiters_parked > 0` + `waiters_woken > 0`
//!      (a lane's demand outran the pool, parked in the queue, and was woken
//!      when freed pages covered it). So a fleet that merely FIT the pool (no
//!      contention) cannot pass trivially. This is what makes ② decisive.
//!   3. **Transparency (classified)** — no position-0 divergence (prefill/page-0
//!      corruption) AND no `restore_attributable` divergence (first divergence at
//!      KV position >= page_size = a sealed/restored page). The [1, page_size)
//!      band is logged non-gating (non-batch-invariance; see the composite above).
//!   4. **Active self-suspend** — `suspends > 0 && restores > 0`: victims
//!      SAVE their own KV state (D2H offload) and RESTORE it (H2D). Preempt
//!      is always on; this fires on every contended profile run.
//!
//! Construction-based contention proof: the fleet is sized so its combined KV
//! demand provably exceeds the (small) pool — so if all complete with zero
//! WorkingSetError, the orchestrator MUST have engaged (an unmanaged pool would
//! have errored the over-capacity lanes).
//!
//! `#[ignore]` (needs the 4090 + cuda + qwen3-0.6b). Run:
//!   PIE_COMPILER_LAUNCHER=env \
//!     cargo test -p pie-gpu-tests --features driver-cuda \
//!     --test cuda_contention -- --ignored --nocapture
//!
//! Quantitative A/B (each command is a separate process because boot is
//! process-global; the optional record file prints the ratio after both
//! runs). The pre-rewrite/legacy side of an A/B is measured by checking out
//! the baseline git ref — the runtime no longer has an error-path mode:
//!   PIE_CONTENTION_PROFILE_MODE=baseline PIE_CONTENTION_PROFILE_RECORD=/tmp/pie-contention.json \
//!     cargo test -p pie-gpu-tests --features driver-cuda --test cuda_contention \
//!     over_capacity_fleet_preempts_and_restores_transparently -- --ignored --exact --nocapture
//!   PIE_CONTENTION_PROFILE_MODE=contended PIE_CONTENTION_TOTAL_PAGES=8 \
//!   PIE_CONTENTION_PROFILE_RECORD=/tmp/pie-contention.json \
//!     cargo test -p pie-gpu-tests --features driver-cuda --test cuda_contention \
//!     over_capacity_fleet_preempts_and_restores_transparently -- --ignored --exact --nocapture
//!
//! COORDINATION (charlie owns the harness + boot config + GPU): the small-pool
//! boot knob below (`SMALL_POOL_GPU_MEM_UTIL`) is a first-cut forcing mechanism —
//! a low `gpu_mem_utilization` shrinks the KV pool so a modest fleet over-fills
//! it. charlie: tune the util (or add an explicit `total_pages`/`max_num_kv_pages`
//! cap to the driver options — the dummy driver currently floors at 256 pages, so
//! a host-runnable variant needs that cap) so the fleet reliably contends without
//! OOM-killing the box. alpha owns the FLEET shape + the two assertions.

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_bin::sweep::fleet::{self, FleetRun};

/// The inferlet every lane runs. Named here rather than inside the fleet
/// helper because the sweep and this test drive different programs through
/// the same load generator.
const GENERATE: &str = "generate@0.1.0";
use pie_client::client::Client;

/// A fleet large enough that its combined KV demand exceeds the shrunk pool
/// (see `SMALL_POOL_GPU_MEM_UTIL`) — forcing the preempt/restore path.
///
/// **Live-tunable via `PIE_CONTENTION_FLEET`**. Contention is forced by the
/// explicit KV-page cap (`PIE_CONTENTION_TOTAL_PAGES` → `[batching].total_pages`),
/// so the fleet just needs `fleet × pages_per_lane > total_pages` (each lane
/// holds ~a handful of pages). The solo-reference guard confirms a single lane
/// still fits.
fn fleet_size() -> usize {
    std::env::var("PIE_CONTENTION_FLEET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24)
}

/// Distinct prompts long enough that each lane holds several KV pages, so the
/// fleet's aggregate demand exceeds the small pool. Greedy decode ⇒ deterministic
/// per prompt, so the solo reference is exact.
fn prompt(k: usize) -> String {
    // `PIE_CONTENTION_BUDGET=N`: emit a bare-int budget so the generate
    // inferlet decodes N tokens — LONGER-lived lanes sustain KV pool
    // pressure long enough for the suspend/restore cycle to reliably
    // engage (5-token lanes finish before contention forms → flaky parked=0).
    // Unset → the original JSON prompt (default 5-token lanes preserved).
    if let Ok(b) = std::env::var("PIE_CONTENTION_BUDGET") {
        if b.trim().parse::<usize>().is_ok() {
            return b.trim().to_string();
        }
    }
    let _ = k;
    format!(
        "{{\"prompt\": \"Lane {k}: the quick brown fox jumps over the lazy dog, \
         and then the clever cat considers the consequences carefully before it\"}}"
    )
}

const BUBBLE_HIST_UPPER_US: [u64; 16] = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    100,
    150,
    250,
    500,
    1_000,
    2_000,
    8_000,
    32_000,
    u64::MAX,
];

fn bubble_percentile(histogram: &[u64], percentile: u64) -> u64 {
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return 0;
    }
    let target = (total * percentile).div_ceil(100);
    let mut cumulative = 0;
    for (index, count) in histogram.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return BUBBLE_HIST_UPPER_US[index];
        }
    }
    *BUBBLE_HIST_UPPER_US.last().unwrap()
}

fn write_profile_record(
    mode: &str,
    fleet: usize,
    token_budget: usize,
    total_pages: u32,
    run: &FleetRun,
    batch_avg: f64,
    bubble_p50_us: u64,
    bubble_p99_us: u64,
) -> Result<()> {
    let Ok(path) = std::env::var("PIE_CONTENTION_PROFILE_RECORD") else {
        return Ok(());
    };
    let total_tokens: usize = run
        .outputs
        .iter()
        .filter_map(Option::as_ref)
        .map(Vec::len)
        .sum();
    let mut document: serde_json::Value = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    document[mode] = serde_json::json!({
        "fleet": fleet,
        "token_budget": token_budget,
        "total_pages": total_pages,
        "elapsed_us": run.elapsed.as_micros(),
        "total_tokens": total_tokens,
        "throughput_tok_s": total_tokens as f64 / run.elapsed.as_secs_f64().max(1e-9),
        "lane_p50_us": run.lane_percentile_us(50),
        "lane_p95_us": run.lane_percentile_us(95),
        "lane_p99_us": run.lane_percentile_us(99),
        "batch_avg": batch_avg,
        "bubble_p50_us": bubble_p50_us,
        "bubble_p99_us": bubble_p99_us,
    });
    if let Some(planner) = pie_engine::planner::planner() {
        let diagnostics = planner.diagnostics();
        document[mode]["contention"] = serde_json::json!({
            "parked": diagnostics.parks_total,
            "woken": diagnostics.serves_total,
            "suspends": diagnostics.evictions_total,
            "restores": diagnostics.restores_total,
            "d2h_pages": diagnostics.d2h_pages_total,
            "h2d_pages": diagnostics.h2d_pages_total,
            "d2h_copy_us": diagnostics.d2h_copy_us_total,
            "h2d_copy_us": diagnostics.h2d_copy_us_total,
            "eviction_rollbacks": diagnostics.eviction_rollbacks_total,
            "restore_failures": diagnostics.restore_failures_total,
            "starvations": diagnostics.starvations_total,
            "host_swap_exhaustions": diagnostics.host_swap_exhaustions_total,
        });
    }
    std::fs::write(&path, serde_json::to_vec_pretty(&document)?)?;

    if let (Some(baseline), Some(contended)) = (document.get("baseline"), document.get("contended"))
        && baseline.get("fleet") == contended.get("fleet")
        && baseline.get("token_budget") == contended.get("token_budget")
    {
        let baseline_throughput = baseline["throughput_tok_s"].as_f64().unwrap_or_default();
        let contended_throughput = contended["throughput_tok_s"].as_f64().unwrap_or_default();
        let baseline_p95 = baseline["lane_p95_us"].as_u64().unwrap_or_default() as f64;
        let contended_p95 = contended["lane_p95_us"].as_u64().unwrap_or_default() as f64;
        eprintln!(
            "[contention-profile] A/B throughput={:.3}x p95-latency={:.3}x record={path}",
            contended_throughput / baseline_throughput.max(f64::EPSILON),
            contended_p95 / baseline_p95.max(1.0),
        );
    }
    if let (Some(legacy), Some(baseline)) = (document.get("legacy"), document.get("baseline"))
        && legacy.get("fleet") == baseline.get("fleet")
        && legacy.get("token_budget") == baseline.get("token_budget")
    {
        let legacy_throughput = legacy["throughput_tok_s"].as_f64().unwrap_or_default();
        let baseline_throughput = baseline["throughput_tok_s"].as_f64().unwrap_or_default();
        let legacy_p95 = legacy["lane_p95_us"].as_u64().unwrap_or_default() as f64;
        let baseline_p95 = baseline["lane_p95_us"].as_u64().unwrap_or_default() as f64;
        eprintln!(
            "[contention-profile] hot-path throughput={:.3}x p95-latency={:.3}x record={path}",
            baseline_throughput / legacy_throughput.max(f64::EPSILON),
            baseline_p95 / legacy_p95.max(1.0),
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "contention over-capacity e2e: needs the 4090 + cuda + qwen3-0.6b"]
async fn over_capacity_fleet_preempts_and_restores_transparently() -> Result<()> {
    common::init_trace();

    let profile_mode =
        std::env::var("PIE_CONTENTION_PROFILE_MODE").unwrap_or_else(|_| "contended".to_string());
    anyhow::ensure!(
        matches!(profile_mode.as_str(), "baseline" | "contended" | "ceiling"),
        "PIE_CONTENTION_PROFILE_MODE must be baseline, contended, or ceiling \
         (the legacy error-path profile is gone with the mode knob; measure \
         the pre-rewrite ref via git for A/B). `ceiling` is the capped-but-\
         uncontended fitting-width run that anchors gap-to-ceiling (§6): same \
         page cap, a fleet that fits, engagement not required."
    );
    let total_pages = if profile_mode == "baseline" {
        0
    } else {
        std::env::var("PIE_CONTENTION_TOTAL_PAGES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(8)
    };

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    anyhow::ensure!(
        Command::new("cargo")
            .args(["build", "--target", "wasm32-wasip2", "-p", "generate"])
            .current_dir(&ws)
            .status()?
            .success(),
        "generate wasm build failed"
    );
    let wasm = ws.join("target/wasm32-wasip2/debug/generate.wasm");
    let manifest = ws.join("generate/Pie.toml");

    // Boot with a SMALL KV pool so a modest fleet over-fills it (charlie tunes the
    // exact knob — see the module COORDINATION note).
    let pie = common::boot_4090_kv_cap(total_pages).await?;
    let addr = pie.listen_addr.to_string();
    let fleet = fleet_size();
    eprintln!("[contention] booted small-pool, addr={addr}, fleet={fleet}");

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    setup.authenticate("test-user", &None).await?;
    setup
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;

    let inputs: Vec<String> = (0..fleet).map(prompt).collect();

    // Reference: each lane ALONE (fits the small pool solo, so no contention) —
    // the deterministic greedy ground truth for that prompt.
    let mut reference = Vec::new();
    for (k, input) in inputs.iter().enumerate() {
        let r = fleet::run(&addr, GENERATE, std::slice::from_ref(input))
            .await
            .outputs
            .pop()
            .flatten();
        anyhow::ensure!(
            r.is_some(),
            "solo lane {k} produced no tokens — the pool is too small even for ONE lane; \
             raise the small-pool util so a single lane fits (contention must come from the FLEET, not a solo lane)"
        );
        reference.push(r);
    }
    let scheduler_before = pie_engine::scheduler::get_stats().await;

    // Over-capacity: launch the whole fleet at once. Combined demand > pool ⇒ the
    // losers OOM in prep → acquire() → wait → a finisher frees → drain → retry.
    // PIE_CONTENTION_TRACE=1: dump ContentionStats every ~1s during the concurrent
    // run so a HANG (never returns) still captures the stuck state, and a PASS shows
    // the counters climb-then-quiesce-by-completion (Phase-2 re-validation).
    let ctrace_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ctrace_task = if std::env::var("PIE_CONTENTION_TRACE").is_ok() {
        let stop = ctrace_stop.clone();
        Some(tokio::spawn(async move {
            let t0 = std::time::Instant::now();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(planner) = pie_engine::planner::planner() {
                    let d = planner.diagnostics();
                    eprintln!(
                        "[contention-trace] t={}ms parked={} woken={} suspends={} restores={} \
                         restore_failures={} h2d_pages={} h2d_us={} \
                         free={}/{} host_free={}/{} accum={} queue={} states={:?}",
                        t0.elapsed().as_millis(),
                        d.parks_total,
                        d.serves_total,
                        d.evictions_total,
                        d.restores_total,
                        d.restore_failures_total,
                        d.h2d_pages_total,
                        d.h2d_copy_us_total,
                        d.device_pages_free,
                        d.device_pages_total,
                        d.host_slots_free,
                        d.host_slots_total,
                        d.accumulation,
                        d.queue.len(),
                        d.proc_states,
                    );
                }
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
            }
        }))
    } else {
        None
    };
    let fleet_run = fleet::run(&addr, GENERATE, &inputs).await;
    let concurrent = &fleet_run.outputs;
    ctrace_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(t) = ctrace_task {
        t.abort();
    }
    let total_output_tokens: usize = concurrent
        .iter()
        .filter_map(Option::as_ref)
        .map(Vec::len)
        .sum();
    let scheduler_after = pie_engine::scheduler::get_stats().await;
    let batches = scheduler_after
        .total_batches
        .saturating_sub(scheduler_before.total_batches);
    let requests = scheduler_after
        .total_requests_processed
        .saturating_sub(scheduler_before.total_requests_processed);
    let batch_avg = requests as f64 / batches.max(1) as f64;
    let bubble_histogram: Vec<_> = scheduler_after
        .bubble_us_hist
        .iter()
        .zip(scheduler_before.bubble_us_hist)
        .map(|(after, before)| after.saturating_sub(before))
        .collect();
    let bubble_p50_us = bubble_percentile(&bubble_histogram, 50);
    let bubble_p99_us = bubble_percentile(&bubble_histogram, 99);
    eprintln!(
        "[contention-profile] mode={profile_mode} pages={total_pages} fleet={fleet} \
         elapsed={:.3}s output_tokens={} throughput={:.1}tok/s \
         lane_p50={:.3}ms lane_p95={:.3}ms lane_p99={:.3}ms \
         batch_avg={:.1} bubble_p50={}us bubble_p99={}us",
        fleet_run.elapsed.as_secs_f64(),
        total_output_tokens,
        total_output_tokens as f64 / fleet_run.elapsed.as_secs_f64().max(1e-9),
        fleet_run.lane_percentile_us(50) as f64 / 1_000.0,
        fleet_run.lane_percentile_us(95) as f64 / 1_000.0,
        fleet_run.lane_percentile_us(99) as f64 / 1_000.0,
        batch_avg,
        bubble_p50_us,
        bubble_p99_us,
    );
    if let Some(planner) = pie_engine::planner::planner() {
        let diagnostics = planner.diagnostics();
        eprintln!(
            "[contention-profile] parked={} woken={} suspends={} restores={} \
             d2h_pages={} h2d_pages={} d2h={:.3}ms h2d={:.3}ms",
            diagnostics.parks_total,
            diagnostics.serves_total,
            diagnostics.evictions_total,
            diagnostics.restores_total,
            diagnostics.d2h_pages_total,
            diagnostics.h2d_pages_total,
            diagnostics.d2h_copy_us_total as f64 / 1_000.0,
            diagnostics.h2d_copy_us_total as f64 / 1_000.0,
        );
    }
    write_profile_record(
        &profile_mode,
        fleet,
        std::env::var("PIE_CONTENTION_BUDGET")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(5),
        total_pages,
        &fleet_run,
        batch_avg,
        bubble_p50_us,
        bubble_p99_us,
    )?;

    // Early engagement snapshot: print the final ContentionStats BEFORE any
    // assertion so a transparency RED still surfaces whether preempt/restore
    // fired (a degenerate-replay mismatch with suspends==0 falsifies the
    // "restore corruption" framing → the bug is not on the restore path).
    if let Some(planner) = pie_engine::planner::planner() {
        let d = planner.diagnostics();
        eprintln!(
            "[contention] final-engagement: parked={} woken={} suspends={} restores={}",
            d.parks_total, d.serves_total, d.evictions_total, d.restores_total,
        );
    }

    // Assertion 1: ZERO WorkingSetError — every lane completed with tokens.
    let errored: Vec<usize> = (0..fleet).filter(|&k| concurrent[k].is_none()).collect();
    assert!(
        errored.is_empty(),
        "preempt/restore FAILED: lanes {errored:?} returned no tokens (a WorkingSetError \
         surfaced instead of the orchestrator absorbing the over-capacity). Zero errors expected."
    );

    // Assertion 2: ENGAGEMENT — preempt/restore provably HAPPENED, checked FIRST.
    // Ordering is load-bearing (charlie's Phase-2 finding): under a DORMANT
    // config (suspends=0, parked=0 — e.g. a too-loose total_pages cap) the
    // multi-lane token comparison measures concurrent-decode NON-BATCH-
    // INVARIANCE, not preempt correctness — the sync single-lane reference
    // itself near-tie loops. Failing on engagement FIRST turns a dormant
    // config into the actionable "fix your config" error instead of a
    // non-determinism trap. Without this a
    // fleet that merely FIT the pool (no contention) passes assertions 1+2
    // trivially. Engagement = allocation waiters that PARKED (demand outran
    // the pool → blocked in acquire's queue) and were later WOKEN (freed
    // pages covered them → the drain released them → they succeeded).
    let (parked, woken, suspends, restores) = pie_engine::planner::planner()
        .map(|planner| {
            let d = planner.diagnostics();
            (
                d.parks_total,
                d.serves_total,
                d.evictions_total,
                d.restores_total,
            )
        })
        .unwrap_or((0, 0, 0, 0));
    eprintln!(
        "[contention] engagement: parked={parked} woken={woken} suspends={suspends} restores={restores}"
    );
    if profile_mode == "contended" {
        assert!(
            parked > 0 && woken > 0,
            "preempt/restore did NOT ENGAGE (waiters_parked={parked}, waiters_woken={woken}) — the \
             fleet never over-filled the pool, so assertions 1+2 proved nothing. Shrink the pool \
             (PIE_CONTENTION_UTIL / a total_pages cap) or grow PIE_CONTENTION_FLEET until it contends."
        );
    } else if profile_mode == "baseline" {
        assert_eq!(
            (parked, suspends, restores),
            (0, 0, 0),
            "roomy baseline unexpectedly contended"
        );
    }
    // `ceiling` asserts neither way: the fitting-width run may brush the cap
    // transiently; only its physics (throughput at width) matters.

    // Assertion 3: TRANSPARENCY (classified) — preempt+restore preserved KV
    // content (W1). The naive `concurrent[k] == solo_reference[k]` is confounded
    // even on an ENGAGED run (charlie's Phase-2 completion of the engaged case):
    // engagement REQUIRES co-batching (a single lane never contends), and
    // co-batched decode is non-batch-invariant at logit near-ties (batch
    // composition → bf16 reduction order → a flipped argmax), so a lane can
    // diverge from its batch-of-1 solo reference with NO KV corruption — even at
    // parked/suspends>0 (verified: tp=16 engaged, suspends=101, still ~9 lanes
    // near-tie flip). The exact-MATCH thus cannot be a hard gate here; the clean,
    // un-confounded suspend/restore transparency proof is the single-lane
    // `cuda_concurrent` reference
    // (a single long-decode lane forced to preempt via `[batching].total_pages`,
    // byte-identical to its sync reference). What this harness CAN hard-gate is
    // genuine KV corruption, by the FIRST-DIVERGENCE POSITION of each mismatch
    // (guru's `restore_attributable == 0` refinement — unconfounded even where
    // token identity is not):
    //   • position 0 — the first token off the identical `"hello world"` prefill
    //     is batch-invariant (deterministic greedy from identical rows), so ANY
    //     position-0 divergence is real prefill/page-0 restore corruption. FAIL.
    //   • position >= `page_size` (KV page = 32 tokens, driver `kv_page_size`,
    //     config.hpp:53) — provably inside a SEALED/RESTORED page (a generated
    //     token at index `i` sits at KV position `prompt_len + i >= i`, so
    //     `i >= page_size` ⇒ KV position >= page_size regardless of prompt length
    //     ⇒ a page has sealed and, under preemption, been offloaded + restored).
    //     A divergence first appearing there is restore/seal corruption, not a
    //     near-tie flip. FAIL (`restore_attributable`). Vacuous when the token
    //     budget < page_size (no page seals); active for longer-budget runs.
    //   • position in [1, page_size) — the confounded band (non-batch-invariance
    //     ≡ partial-page-0 restore, inseparable here). LOGGED, non-gating; clean
    //     transparency in this band is the FLEET=1 bubble proof above.
    const PAGE_SIZE: usize = 32; // driver kv_page_size (config.hpp:53)
    let first_divergence = |a: &Option<Vec<i64>>, b: &Option<Vec<i64>>| -> Option<usize> {
        match (a, b) {
            (Some(x), Some(y)) => (0..x.len().min(y.len()))
                .find(|&i| x[i] != y[i])
                .or_else(|| (x.len() != y.len()).then_some(x.len().min(y.len()))),
            _ => None,
        }
    };
    let mut nbi_lanes = Vec::new(); // [1, page_size): non-batch-invariance (tolerated)
    let mut corrupt_lanes = Vec::new(); // position 0: prefill/page-0 corruption
    let mut restore_attributable = Vec::new(); // >= page_size: sealed/restored-page corruption
    for k in 0..fleet {
        if concurrent[k] != reference[k] {
            let pos = first_divergence(&concurrent[k], &reference[k]);
            eprintln!(
                "[contention] lane {k} diverged@{pos:?} conc={:?} ref={:?}",
                concurrent[k], reference[k]
            );
            match pos {
                Some(0) => corrupt_lanes.push(k),
                Some(p) if p >= PAGE_SIZE => restore_attributable.push(k),
                _ => nbi_lanes.push(k),
            }
        }
    }
    if !nbi_lanes.is_empty() {
        eprintln!(
            "[contention] {} lane(s) {nbi_lanes:?} diverged in [1, {PAGE_SIZE}) — concurrent-decode \
             non-batch-invariance (co-batch bf16 near-tie flip), NOT KV corruption. Suspend/restore \
             transparency in this band is gated by the single-lane cuda_concurrent reference. Non-gating.",
            nbi_lanes.len()
        );
    }
    assert!(
        corrupt_lanes.is_empty(),
        "KV/prefix corruption: lanes {corrupt_lanes:?} diverged at position 0 — the first token off \
         the identical `hello world` prefill is batch-invariant, so a position-0 divergence is a real \
         restore/prefill KV corruption (not a near-tie flip)."
    );
    assert!(
        restore_attributable.is_empty(),
        "restore-attributable KV corruption: lanes {restore_attributable:?} first diverged at position \
         >= {PAGE_SIZE} (a SEALED + preempt-restored page), so the divergence is real suspend/restore \
         corruption of a sealed page, not a pre-seal near-tie flip."
    );

    // Assertion 4 (active self-suspend): a victim doesn't just park on the
    // pool — at its next safe point (`yield_point`) it SAVES its KV state
    // (D2H offload) and RESTORES it (H2D) on release, so `suspends`/
    // `restores` MUST fire.
    if profile_mode == "contended" {
        assert!(
            suspends > 0 && restores > 0,
            "active self-suspend did NOT ENGAGE (suspends={suspends}, restores={restores}) — \
             victims must actively save (suspend) and restore their KV state. \
             Zero means the active path never ran."
        );
        eprintln!("[contention] self-suspend engaged: suspends={suspends} restores={restores} ✓");
    }

    eprintln!(
        "[contention] mode={profile_mode} {fleet}/{fleet} lanes: zero WorkingSetError + \
         classified transparency (parked={parked}, woken={woken})"
    );
    Ok(())
}
