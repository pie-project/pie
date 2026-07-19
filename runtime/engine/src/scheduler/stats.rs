//! `SchedulerStats` — the per-driver, lock-free telemetry snapshot. Updated
//! atomically after each fire by the single scheduler thread; read (Relaxed)
//! by [`aggregate`]. Self-contained: no dependency on the
//! scheduler batch/request types — pure counters + histograms + the fire/driver
//! probe sub-structs (`crate::scheduler::probe`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

// =============================================================================
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

/// Inter-batch bubble histogram: exclusive upper bounds (µs) per bucket. A
/// boundary is pinned at 100 (the masterplan p50 gate) so "p50 < 100 µs" reads as
/// "the p50 bucket's upper bound ≤ 100". `u64::MAX` is the overflow catch-all.
pub const BUBBLE_HIST_UPPER_US: [u64; 16] = [
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

/// Cumulative stats exposed for monitoring. Updated atomically after each batch.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    // ── Always-on counters (no Instant::now needed). ────────────────────────
    pub total_batches: AtomicU64,
    pub total_tokens_processed: AtomicU64,
    /// Total request count across all batches (sum of batch sizes).
    /// Divide by `total_batches` for mean batch size in requests.
    pub total_requests_processed: AtomicU64,
    /// Largest forward request count ever fired by this scheduler.
    pub max_forward_requests_observed: AtomicU64,
    /// Coarse histogram of batch sizes. Buckets:
    /// [0]=1, [1]=2-3, [2]=4-7, [3]=8-15, [4]=16-31,
    /// [5]=32-63, [6]=64-127, [7]=128+.
    pub batch_size_hist: [AtomicU64; 8],
    pub last_batch_latency_us: AtomicU64,
    pub cumulative_latency_us: AtomicU64,
    /// Inter-batch bubble histogram (always-on, HOST PROXY) — one count per fire,
    /// bucketed by [`BUBBLE_HIST_UPPER_US`]. Yields a true p50/p99 (the masterplan
    /// gate) across ALL fires (0 when enqueue-ahead covered the gap), unlike the
    /// probe-gated `fire.quorum.inter_batch_bubble_us` accumulator (average,
    /// non-zero only). The host stamp includes scheduler wake/submit overhead, so
    /// it is a conservative upper-bound measurement.
    pub bubble_us_hist: [AtomicU64; BUBBLE_HIST_UPPER_US.len()],
    // ── Fire-domain probes (gated behind `profile-fire` feature). ───────────
    //
    // Hierarchy + invariants documented in `crate::scheduler::probe`. Writers
    // use the `probe_fire!` macro from that module so the fetch_add
    // disappears when the feature is off. The struct itself is always
    // defined so callers and readers compile uniformly.
    pub fire: crate::scheduler::probe::FireProbes,
}

impl SchedulerStats {
    /// Record one fire's inter-batch bubble (µs) into the HOST-PROXY histogram.
    /// Called only from the single per-driver scheduler thread (race-free plain
    /// fetch_add).
    pub fn record_bubble_us(&self, us: u64) {
        self.bubble_us_hist[Self::bubble_bucket(us)].fetch_add(1, Relaxed);
    }

    #[inline]
    fn bubble_bucket(us: u64) -> usize {
        BUBBLE_HIST_UPPER_US
            .iter()
            .position(|&upper| us < upper)
            .unwrap_or(BUBBLE_HIST_UPPER_US.len() - 1)
    }
}

/// Fold a completed batch's always-on counters into the shared stats.
/// `latency` is the off-thread forward (GPU) wait —
/// the dominant component of the batch's wall time under the overlapped
/// fire (the host build/enqueue overlaps the prior in-flight batch).
pub(crate) fn record_fire_stats(
    stats: &SchedulerStats,
    latency: Duration,
    batch_size: u64,
    total_tokens: usize,
) {
    crate::probe_fire!(stats.fire.post_dispatch.stats_update_us, {
        stats.total_batches.fetch_add(1, Relaxed);
        stats
            .total_tokens_processed
            .fetch_add(total_tokens as u64, Relaxed);
        stats
            .total_requests_processed
            .fetch_add(batch_size, Relaxed);
        stats
            .max_forward_requests_observed
            .fetch_max(batch_size, Relaxed);
        let bucket = match batch_size {
            0 | 1 => 0,
            2..=3 => 1,
            4..=7 => 2,
            8..=15 => 3,
            16..=31 => 4,
            32..=63 => 5,
            64..=127 => 6,
            _ => 7,
        };
        stats.batch_size_hist[bucket].fetch_add(1, Relaxed);
        stats
            .last_batch_latency_us
            .store(latency.as_micros() as u64, Relaxed);
        stats
            .cumulative_latency_us
            .fetch_add(latency.as_micros() as u64, Relaxed);
    });
}

// =============================================================================
// AggregateStats (cross-driver stats, aggregated over per-worker stats)
// =============================================================================

use std::sync::Arc;

/// Aggregated scheduler stats across every driver (was `InferenceStats`).
///
/// Always-on counters live at the top; per-domain probe averages
/// (currently just `fire`) are nested so the shape mirrors the probe
/// hierarchy in `crate::scheduler::probe`.
#[derive(Debug, Default, serde::Serialize)]
pub struct AggregateStats {
    pub total_batches: u64,
    pub total_tokens_processed: u64,
    pub total_requests_processed: u64,
    pub max_forward_requests_observed: u64,
    /// Histogram buckets (1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+).
    pub batch_size_hist: [u64; 8],
    pub last_batch_latency_us: u64,
    pub cumulative_batch_latency_us: u64,
    pub avg_batch_latency_us: u64,

    /// Fire-domain probe averages. Values are 0 when built without
    /// `profile-fire`. Mirrors `crate::scheduler::probe::FireProbes`.
    pub fire: FireStats,

    /// Inter-batch bubble histogram (one count per fire), bucketed by
    /// [`BUBBLE_HIST_UPPER_US`] — for the p50/p99 bubble gate (masterplan M1).
    /// HOST PROXY (device-idle stamped at the Rust enqueue point → over-counts by
    /// the host submit/handshake delay). See [`Self::bubble_p50`].
    pub bubble_us_hist: [u64; BUBBLE_HIST_UPPER_US.len()],
}

impl AggregateStats {
    /// Inter-batch bubble p50 (µs) from the scheduler's host-side histogram.
    pub fn bubble_p50(&self) -> u64 {
        self.bubble_percentile(0.50)
    }

    /// Inter-batch bubble p99 (µs).
    pub fn bubble_p99(&self) -> u64 {
        self.bubble_percentile(0.99)
    }

    /// Alias retained for callers that explicitly request the host proxy.
    pub fn bubble_p50_proxy(&self) -> u64 {
        Self::hist_percentile(&self.bubble_us_hist, 0.50)
    }

    fn bubble_percentile(&self, q: f64) -> u64 {
        Self::hist_percentile(&self.bubble_us_hist, q)
    }

    /// The upper bound of the bucket containing the `q`-quantile of `hist`
    /// (0 if the histogram is empty).
    fn hist_percentile(hist: &[u64; BUBBLE_HIST_UPPER_US.len()], q: f64) -> u64 {
        let total: u64 = hist.iter().sum();
        if total == 0 {
            return 0;
        }
        let target = (total as f64 * q).ceil() as u64;
        let mut cum = 0u64;
        for (i, &count) in hist.iter().enumerate() {
            cum += count;
            if cum >= target {
                return BUBBLE_HIST_UPPER_US[i];
            }
        }
        *BUBBLE_HIST_UPPER_US.last().unwrap()
    }
}

#[derive(Debug, Default, serde::Serialize)]
pub struct FireStats {
    pub avg_inter_fire_us: u64,
    pub avg_post_dispatch_to_fire_us: u64,
    pub avg_recv_block_wait_us: u64,
    pub inter_fire_us_sum: u64,
    pub post_dispatch_to_fire_us_sum: u64,
    pub recv_block_wait_us_sum: u64,
    pub accumulate: AccumulateStats,
    pub pre_dispatch: PreDispatchStats,
    pub execute: ExecuteStats,
    pub post_dispatch: PostDispatchStats,
    pub quorum: QuorumStats,
}

/// Quorum-rule probe averages/counters (overview §7.2; thrust-2 §3 F1–F6).
/// The wave counters (`wave_*`, `cold_hold_fires`) populate in every build.
/// Legacy straggler counters remain zero under strict wait-all.
/// The latency probes (bubble/quorum-latency sums, escape/submit-ahead)
/// still require `profile-fire`. Populated by the quorum core (thrust-2
/// phase S5). Mirrors `crate::scheduler::probe::QuorumProbes`.
#[derive(Debug, Default, serde::Serialize)]
pub struct QuorumStats {
    /// Mean device idle between a batch retiring and the next launching (F1).
    pub avg_inter_batch_bubble_us: u64,
    pub inter_batch_bubble_us_sum: u64,
    /// Mean last-ready → enqueue latency (F1 quorum completion).
    pub avg_quorum_latency_us: u64,
    pub quorum_latency_us_sum: u64,
    /// Idle-escape (F2) fire count; divide by `total_batches` for escape rate.
    pub escape_fires: u64,
    /// Depth-2 submit-ahead (G3 bubble) fire count; divide by `total_batches`
    /// for the submit-ahead rate (steady-state decode-fleet bubble-filler).
    pub submit_ahead_fires: u64,
    /// Mean cold-hold window per cold-hold fire (F3 occupancy).
    pub avg_cold_hold_us: u64,
    pub cold_hold_us_sum: u64,
    /// Count of fires through the cold-hold path (F3).
    pub cold_hold_fires: u64,
    /// Legacy field: strict wait-all never fires narrow.
    pub straggler_fires: u64,
    /// Legacy field: strict wait-all never demotes pipelines.
    pub straggler_demotions: u64,
    /// Dummy-run / readiness-miss count (M3 gate: rate < 1%).
    pub readiness_miss: u64,
    /// Wait-for-all wave (M-AB): mean active_pipelines (wait-set size) sampled
    /// at each WaitAll fire. ≈ fleet width ⇒ persistent wait-set (waves should
    /// be dense); ≈1 ⇒ transient/singleton. 0 if no WaitAll fire.
    pub avg_active_pipelines_at_fire: u64,
    /// Mean absentees per wave fire; strict wait-all keeps this at zero.
    pub avg_missing_at_fire: u64,
    /// Cumulative numerator for `avg_active_pipelines_at_fire`.
    pub wave_active_sum: u64,
    /// Cumulative numerator for `avg_missing_at_fire`.
    pub wave_missing_sum: u64,
    /// Count of WaitAll wave fires (denominator for the two averages above).
    pub wave_fires: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct AccumulateStats {
    pub avg_accum_loop_us: u64,
    pub accum_loop_us_sum: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct PreDispatchStats {
    pub avg_fire_prepare_us: u64,
    pub fire_prepare_us_sum: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct ExecuteStats {
    pub avg_total_us: u64,
    pub avg_batch_build_us: u64,
    pub avg_driver_fire_us: u64,
    pub total_us_sum: u64,
    pub batch_build_us_sum: u64,
    pub driver_fire_us_sum: u64,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct PostDispatchStats {
    pub avg_context_tick_us: u64,
    pub avg_stats_update_us: u64,
    pub context_tick_us_sum: u64,
    pub stats_update_us_sum: u64,
}

/// Aggregate stats across every per-driver `SchedulerStats` (was
/// `InferenceService::aggregate_stats`; now a plain function over the
/// registry's `Vec<Arc<SchedulerStats>>` — the atomics are lock-free, so no
/// actor round-trip is needed).
pub(crate) fn aggregate(scheduler_stats: &[Arc<SchedulerStats>]) -> AggregateStats {
    let mut total_batches = 0u64;
    let mut total_tokens = 0u64;
    let mut total_requests = 0u64;
    let mut max_forward_requests = 0u64;
    let mut hist = [0u64; 8];
    let mut bubble_hist = [0u64; BUBBLE_HIST_UPPER_US.len()];
    let mut last_latency = 0u64;
    let mut cumulative_latency = 0u64;
    // Per-driver sums of probe atomics. Walked in the same shape
    // as AggregateStats.fire / FireProbes so the relationship is
    // self-evident.
    let mut fire_inter = 0u64;
    let mut fire_post_dispatch_to_fire = 0u64;
    let mut fire_recv_block_wait = 0u64;
    let mut fire_accumulate_accum_loop = 0u64;
    let mut fire_pre_dispatch_fire_prepare = 0u64;
    let mut fire_execute_total = 0u64;
    let mut fire_execute_batch_build = 0u64;
    let mut fire_execute_driver_fire = 0u64;
    let mut fire_post_dispatch_context_tick = 0u64;
    let mut fire_post_dispatch_stats_update = 0u64;
    let mut q_inter_batch_bubble = 0u64;
    let mut q_quorum_latency = 0u64;
    let mut q_escape_fires = 0u64;
    let mut q_submit_ahead_fires = 0u64;
    let mut q_cold_hold = 0u64;
    let mut q_cold_hold_fires = 0u64;
    let mut q_straggler_fires = 0u64;
    let mut q_straggler_demotions = 0u64;
    let mut q_readiness_miss = 0u64;
    let mut q_wave_active_sum = 0u64;
    let mut q_wave_missing_sum = 0u64;
    let mut q_wave_fires = 0u64;
    for s in scheduler_stats {
        total_batches += s.total_batches.load(Relaxed);
        total_tokens += s.total_tokens_processed.load(Relaxed);
        total_requests += s.total_requests_processed.load(Relaxed);
        max_forward_requests =
            max_forward_requests.max(s.max_forward_requests_observed.load(Relaxed));
        for (dst, src) in hist.iter_mut().zip(s.batch_size_hist.iter()) {
            *dst += src.load(Relaxed);
        }
        last_latency = last_latency.max(s.last_batch_latency_us.load(Relaxed));
        cumulative_latency += s.cumulative_latency_us.load(Relaxed);
        let f = &s.fire;
        fire_inter += f.inter_fire_us.load(Relaxed);
        fire_post_dispatch_to_fire += f.post_dispatch_to_fire_us.load(Relaxed);
        fire_recv_block_wait += f.recv_block_wait_us.load(Relaxed);
        fire_accumulate_accum_loop += f.accumulate.accum_loop_us.load(Relaxed);
        fire_pre_dispatch_fire_prepare += f.pre_dispatch.fire_prepare_us.load(Relaxed);
        fire_execute_total += f.execute.total_us.load(Relaxed);
        fire_execute_batch_build += f.execute.batch_build_us.load(Relaxed);
        fire_execute_driver_fire += f.execute.driver_fire_us.load(Relaxed);
        fire_post_dispatch_context_tick += f.post_dispatch.context_tick_us.load(Relaxed);
        fire_post_dispatch_stats_update += f.post_dispatch.stats_update_us.load(Relaxed);
        q_inter_batch_bubble += f.quorum.inter_batch_bubble_us.load(Relaxed);
        q_quorum_latency += f.quorum.quorum_latency_us.load(Relaxed);
        q_escape_fires += f.quorum.escape_fires.load(Relaxed);
        q_submit_ahead_fires += f.quorum.submit_ahead_fires.load(Relaxed);
        q_cold_hold += f.quorum.cold_hold_us.load(Relaxed);
        q_cold_hold_fires += f.quorum.cold_hold_fires.load(Relaxed);
        q_straggler_fires += f.quorum.straggler_fires.load(Relaxed);
        q_straggler_demotions += f.quorum.straggler_demotions.load(Relaxed);
        q_readiness_miss += f.quorum.readiness_miss.load(Relaxed);
        q_wave_active_sum += f.quorum.wave_active_sum.load(Relaxed);
        q_wave_missing_sum += f.quorum.wave_missing_sum.load(Relaxed);
        q_wave_fires += f.quorum.wave_fires.load(Relaxed);
        for (dst, src) in bubble_hist.iter_mut().zip(s.bubble_us_hist.iter()) {
            *dst += src.load(Relaxed);
        }
    }

    let avg = |value: u64| {
        if total_batches > 0 {
            value / total_batches
        } else {
            0
        }
    };
    // Inter-fire is sampled starting at the 2nd batch (first one has
    // no prior to diff against), so divide by max(total_batches-1, 1)
    // to get a stable mean.
    let avg_pair = |value: u64| {
        if total_batches > 1 {
            value / (total_batches - 1)
        } else {
            0
        }
    };
    AggregateStats {
        total_batches,
        total_tokens_processed: total_tokens,
        total_requests_processed: total_requests,
        max_forward_requests_observed: max_forward_requests,
        batch_size_hist: hist,
        last_batch_latency_us: last_latency,
        cumulative_batch_latency_us: cumulative_latency,
        avg_batch_latency_us: avg(cumulative_latency),
        fire: FireStats {
            avg_inter_fire_us: avg_pair(fire_inter),
            avg_post_dispatch_to_fire_us: avg_pair(fire_post_dispatch_to_fire),
            avg_recv_block_wait_us: avg_pair(fire_recv_block_wait),
            inter_fire_us_sum: fire_inter,
            post_dispatch_to_fire_us_sum: fire_post_dispatch_to_fire,
            recv_block_wait_us_sum: fire_recv_block_wait,
            accumulate: AccumulateStats {
                avg_accum_loop_us: avg(fire_accumulate_accum_loop),
                accum_loop_us_sum: fire_accumulate_accum_loop,
            },
            pre_dispatch: PreDispatchStats {
                avg_fire_prepare_us: avg(fire_pre_dispatch_fire_prepare),
                fire_prepare_us_sum: fire_pre_dispatch_fire_prepare,
            },
            execute: ExecuteStats {
                avg_total_us: avg(fire_execute_total),
                avg_batch_build_us: avg(fire_execute_batch_build),
                avg_driver_fire_us: avg(fire_execute_driver_fire),
                total_us_sum: fire_execute_total,
                batch_build_us_sum: fire_execute_batch_build,
                driver_fire_us_sum: fire_execute_driver_fire,
            },
            post_dispatch: PostDispatchStats {
                avg_context_tick_us: avg(fire_post_dispatch_context_tick),
                avg_stats_update_us: avg(fire_post_dispatch_stats_update),
                context_tick_us_sum: fire_post_dispatch_context_tick,
                stats_update_us_sum: fire_post_dispatch_stats_update,
            },
            quorum: QuorumStats {
                avg_inter_batch_bubble_us: avg(q_inter_batch_bubble),
                inter_batch_bubble_us_sum: q_inter_batch_bubble,
                avg_quorum_latency_us: avg(q_quorum_latency),
                quorum_latency_us_sum: q_quorum_latency,
                escape_fires: q_escape_fires,
                submit_ahead_fires: q_submit_ahead_fires,
                avg_cold_hold_us: if q_cold_hold_fires > 0 {
                    q_cold_hold / q_cold_hold_fires
                } else {
                    0
                },
                cold_hold_us_sum: q_cold_hold,
                cold_hold_fires: q_cold_hold_fires,
                straggler_fires: q_straggler_fires,
                straggler_demotions: q_straggler_demotions,
                readiness_miss: q_readiness_miss,
                avg_active_pipelines_at_fire: if q_wave_fires > 0 {
                    q_wave_active_sum / q_wave_fires
                } else {
                    0
                },
                avg_missing_at_fire: if q_wave_fires > 0 {
                    q_wave_missing_sum / q_wave_fires
                } else {
                    0
                },
                wave_active_sum: q_wave_active_sum,
                wave_missing_sum: q_wave_missing_sum,
                wave_fires: q_wave_fires,
            },
        },
        bubble_us_hist: bubble_hist,
    }
}
