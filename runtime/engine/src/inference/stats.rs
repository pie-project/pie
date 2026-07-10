//! `SchedulerStats` — the per-driver, lock-free telemetry snapshot. Updated
//! atomically after each fire by the single scheduler thread; read (Relaxed)
//! by `InferenceService::aggregate_stats`. Self-contained: no dependency on the
//! scheduler batch/request types — pure counters + histograms + the fire/driver
//! probe sub-structs (`crate::probe::*`).

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

// =============================================================================
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

/// Bounded per-identity (C3) co-batch table width. Fleets carry ~10 distinct
/// program identities; a first-seen table of this many slots covers them with
/// headroom. A fleet exceeding it reads `identities_dropped > 0` (fail-loud
/// "saturated" — never silently missing an identity).
pub const PER_IDENTITY_BATCH_CAP: usize = 32;

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
    // ── Per-identity (C3) co-batch density (pentathlon per-identity-batch-density
    //    probe = rows / fires per program identity). First-seen bounded table keyed
    //    on `program_identity_hash` (bytecode ⊕ manifest — the SAME key the quorum
    //    scheduler batches / dedups / compiles on, so the telemetry can't disagree
    //    with the batching identity it measures). Written by the single per-driver
    //    scheduler thread; read (Relaxed) by `aggregate_stats`.
    /// Slot key = `program_identity_hash` (0 = empty slot).
    pub per_identity_hash: [AtomicU64; PER_IDENTITY_BATCH_CAP],
    /// Fires that included ≥1 request of this identity.
    pub per_identity_fires: [AtomicU64; PER_IDENTITY_BATCH_CAP],
    /// Total rows (co-batched requests) fired for this identity.
    pub per_identity_rows: [AtomicU64; PER_IDENTITY_BATCH_CAP],
    /// Fire-records dropped because the table was saturated (drop-new). > 0 ⇒ the
    /// fleet has more than `PER_IDENTITY_BATCH_CAP` distinct identities; the
    /// per-identity table is a bounded sample, NOT silently missing.
    pub identities_dropped: AtomicU64,

    /// Inter-batch bubble histogram (always-on, HOST PROXY) — one count per fire,
    /// bucketed by [`BUBBLE_HIST_UPPER_US`]. Yields a true p50/p99 (the masterplan
    /// gate) across ALL fires (0 when enqueue-ahead covered the gap), unlike the
    /// probe-gated `fire.quorum.inter_batch_bubble_us` accumulator (average,
    /// non-zero only). The host stamp includes scheduler wake/submit overhead, so
    /// it is a conservative upper-bound measurement.
    pub bubble_us_hist: [AtomicU64; BUBBLE_HIST_UPPER_US.len()],
    // ── Fire-domain probes (gated behind `profile-fire` feature). ───────────
    //
    // Hierarchy + invariants documented in `crate::probe::fire`. Writers
    // use the `probe_fire!` macro from that module so the fetch_add
    // disappears when the feature is off. The struct itself is always
    // defined so callers and readers compile uniformly.
    pub fire: crate::probe::fire::FireProbes,
}

impl SchedulerStats {
    /// Record one fire's contribution for `hash`: +1 fire, +`rows` co-batched
    /// requests. First-seen linear insertion into the bounded per-identity table;
    /// drop-new + bump `identities_dropped` on saturation (fail-loud). Called only
    /// from the single per-driver scheduler thread, so plain load/store is race-free.
    pub fn record_identity_fire(&self, hash: u64, rows: u64) {
        if hash == 0 {
            return; // 0 is the empty-slot sentinel; a hashless plain forward isn't C3-keyed.
        }
        for slot in 0..PER_IDENTITY_BATCH_CAP {
            let cur = self.per_identity_hash[slot].load(Relaxed);
            if cur == hash {
                self.per_identity_fires[slot].fetch_add(1, Relaxed);
                self.per_identity_rows[slot].fetch_add(rows, Relaxed);
                return;
            }
            if cur == 0 {
                self.per_identity_hash[slot].store(hash, Relaxed);
                self.per_identity_fires[slot].fetch_add(1, Relaxed);
                self.per_identity_rows[slot].fetch_add(rows, Relaxed);
                return;
            }
        }
        // Table saturated → drop this record, count it (fail-loud).
        self.identities_dropped.fetch_add(1, Relaxed);
    }

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
