//! Batch scheduling policies.
//!
//! Three policies live here, in order of increasing patience:
//!
//!   - `GreedyPolicy` — fire immediately, zero waiting.
//!   - `EagerPolicy` — wait for the *peer cohort* to gather, then fire
//!     even if the GPU is still finishing the previous batch (the
//!     scheduler's semaphore handles GPU contention).
//!   - `AdaptivePolicy` (default) — wait for the peer cohort *and* the
//!     GPU before firing. Kept in `Wait` while the GPU is busy so
//!     incoming arrivals during decode keep growing the next batch.
//!
//! Selection via the per-model `[model.scheduler].policy` field in
//! the TOML config (default = `adaptive`). The other two are retained
//! as references: greedy for sanity-checking the scheduler itself,
//! eager as a slightly simpler alternative that's marginally faster
//! in pure steady-state and marginally slower under bursty / churning
//! workloads.
//!
//! ## Design history (why these are what they are)
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a batch and the policy decides when to fire.
//!
//! An earlier version of this file used `AdaptiveThroughputPolicy`,
//! which fired when `idle > ref_l/B` (where `ref_l` was a leaky-max
//! of observed batch latencies, decayed by `0.9999` per batch). The
//! intuition was: bigger batches need shorter waits because they're
//! already "full enough". That policy had three failure modes that
//! showed up clearly under benchmark:
//!
//!   1. **Low-RPS over-wait.** With a single in-flight context, B=1
//!      and `ref_l/B = ref_l`. The 50 ms `max_wait_time` cap fires
//!      the batch every iteration, paying ~50 ms of pure waiting per
//!      token. Per-request latency at RPS=1 was ~3× a no-wait
//!      baseline.
//!
//!   2. **Idle-phase pathology under churn.** `ref_l` is a max-ratchet,
//!      so once a high-RPS phase elevates it, decay is glacial
//!      (half-life ~7000 batches). A subsequent idle phase still has
//!      `ref_l` at the old peak, so every B=1 batch waits its full
//!      50 ms cap. Idle-phase p50 of 995 ms vs 541 ms for the new
//!      policy under a 0.5↔64 RPS churn pattern.
//!
//!   3. **Premature mid-burst fire.** As B ramps up during peer
//!      accumulation, `ref_l/B` shrinks aggressively. A 0.5 ms gap
//!      between two peer arrivals at B=4 trips `idle > ref_l/B` and
//!      the batch fires fragmented (`B=4` fired with `pinned=13` in
//!      instrumented runs). Stragglers go in the next batch and pay
//!      an extra full iteration of latency.
//!
//! Several intermediate fixes were tried — adding a "cohort cap"
//! (`B >= pinned_count → fire`), faster `DECAY`, idle-based grace
//! periods, λ-tracking — and the path through them led to the
//! present design, which is **simpler than any intermediate**:
//!
//!   - The cohort cap alone fixes (1) and (2).
//!   - Removing the `idle > ref_l/B` rule entirely fixes (3).
//!   - That removal also drops the need for `DECAY` smoothing,
//!     since the latency value is no longer used as a
//!     sensitivity-amplified budget.
//!
//! That cohort-cap-plus-bound design is `EagerPolicy` below. Adding
//! a GPU-busy gate on top — keeping the policy in `Wait` while the
//! driver is occupied — yields `AdaptivePolicy`. The gate has a small
//! consistent benefit under churn (~20 ms p50 improvement on average,
//! up to ~50 ms on saturated bursts) at the cost of a similarly small
//! steady-state regression. Promoted to default because the workloads
//! Pie targets are bursty.

use std::time::{Duration, Instant};

use super::scheduler::{Decision, SchedulingPolicy};

const COLD_START_COHORT_WAIT: Duration = Duration::from_millis(10);

// =============================================================================
// AdaptivePolicy — the default. Cohort cap + GPU-busy gate + safety bound.
// =============================================================================
//
// **Firing rule** (in evaluation order):
//
//   1. `B >= max_forward_requests`              — structural driver limit.
//   2. `in_flight`                        — GPU is occupied by the
//                                           previous batch. Wait.
//                                           Arrivals during the wait
//                                           grow this batch.
//   3. `last_latency == 0`                — cold start, no measurement
//                                           yet; briefly wait for the
//                                           initial serving cohort, then
//                                           fire to make progress.
//   4. `B >= pinned_count(driver_idx)`    — every active peer is in
//                                           the batch (cohort cap).
//                                           Fire.
//   5. `start.elapsed() >= last_latency`  — safety bound. Waiting
//                                           longer than one batch's
//                                           compute time costs more
//                                           in queue delay than any
//                                           plausible saving from a
//                                           larger batch.
//   otherwise: `Wait(last_latency - start.elapsed())`
//
// Conditions (3)–(5) all act as wait gates: we fire only when the
// GPU is free *and* either the cohort is full or the safety bound
// has elapsed.
//
// **Why the cohort cap is load-bearing.** In iteration-level
// batching, `pinned_count(driver_idx)` is the count of currently-active
// contexts on that driver. After a batch fires, every one of those
// contexts will eventually submit its next forward-pass request.
// Once we have all of them in the accumulator (`B == pinned_count`),
// waiting longer can't produce a bigger batch from existing peers —
// it can only catch brand-new contexts that pin during the wait. At
// realistic Poisson arrival rates that's rarely worth one extra
// batch's worth of delay for the existing peers. So: fire.
//
// `pinned_count` over-counts if a peer's inferlet is mid-CPU-work
// (pinned but not currently submitting). The safety bound is the
// fallback for that case.
//
// **Why the GPU-busy gate (vs `EagerPolicy`).** Without the gate,
// the cohort-full condition can fire while the previous batch is
// still on the driver. The scheduler then enters its `Fire` branch
// and blocks on `acquire_owned().await` for the in-flight permit.
// During that block the scheduler stops draining the request
// channel, so any arrivals queued during that window fall into the
// *next* batch. With the gate, the policy stays in `Wait` until the
// GPU is free; the scheduler's `tokio::select` keeps absorbing
// arrivals into the *current* batch right up to the moment of fire.
//
// **Why `last_latency` (not `ref_l` with smoothing).** The original
// policy used a leaky-max of recent batch latencies. The smoothing
// served two purposes: (a) damping spikes, (b) preventing a death
// spiral where a single small batch shrinks the budget so far that
// all subsequent batches are also small. With the cohort cap as the
// primary fire mechanism, neither problem materializes:
//
//   - The bound only fires when `B < pinned_count` — i.e., when
//     peers are stuck. In that regime, firing fast IS correct;
//     there's nothing to wait for.
//   - When the bound *does* fire, the resulting batch's actual
//     decode time becomes the next `last_latency`. The value
//     re-calibrates within 1–2 batches.
//
// Empirically validated under: steady RPS sweeps {1,4,16,64},
// symmetric churn (32↔1), severe idle↔burst churn (0.5↔64), and
// rapid 2-second bursts. No spiral or runaway observed.

pub(super) struct AdaptivePolicy {
    max_forward_requests: usize,
    /// Index of the driver this policy is scheduling for. Used to
    /// read `pinned_count` (lock-free atomic, updated on context
    /// pin/unpin).
    driver_idx: usize,
    /// `Some(t)` while a batch is accumulating; `None` after fire
    /// and before the next first arrival.
    batch_start_time: Option<Instant>,
    /// Most recent batch's compute time, in seconds. Updated on
    /// `on_complete`. Used as the absolute upper bound on how long
    /// to wait when `B < pinned_count`. Zero until the second batch
    /// completes (the first is skipped due to CUDA/warmup overhead).
    last_latency: f64,
    /// Total batches the policy has seen complete. Used to skip the
    /// first batch's `on_complete` (warmup latency is
    /// unrepresentative).
    batches_completed: usize,
    /// Set in `on_fired` and cleared in `on_complete` — true while
    /// the driver is executing a batch.
    in_flight: bool,
    /// Maximum `pinned_count` observed since the last fire. The cohort
    /// cap uses this instead of the instantaneous `pinned_count` so a
    /// transient drop — every inferlet calls `unpin()` immediately
    /// after its forward returns and `pin()` again before the next
    /// step, so `pinned_count` briefly goes through every value in
    /// `[0, R]` between fires — doesn't cause us to fire with a
    /// partial cohort. Reset on `on_fired`. Effective only while
    /// `in_flight == false` (when we're accumulating the next batch).
    cohort_high_water: usize,
    /// Highest batch size that has fired since the policy was created.
    /// Used as the cohort target so the scheduler waits for the
    /// expected steady-state cohort to assemble even when the first
    /// inferlet to arrive happens to be the one that just re-pinned
    /// (a race vs the other 7+ inferlets still going through their
    /// post-fire CPU step). Without this, the cohort cap fires at
    /// B=1 because `pinned_count == 1` at that exact moment. Persists
    /// across fires; only ratchets upward.
    fired_high_water: usize,
}

impl AdaptivePolicy {
    pub fn new(max_forward_requests: usize, driver_idx: usize) -> Self {
        Self {
            max_forward_requests,
            driver_idx,
            batch_start_time: None,
            last_latency: 0.0,
            batches_completed: 0,
            in_flight: false,
            cohort_high_water: 0,
            fired_high_water: 0,
        }
    }
}

impl SchedulingPolicy for AdaptivePolicy {
    fn on_arrival(&mut self) {
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        self.batches_completed += 1;
        // Skip the first batch: CUDA graph capture, kernel-cache
        // warmup, and one-shot allocations make its latency
        // unrepresentative of steady-state. After that, just take
        // the latest value — no smoothing, no decay.
        if self.batches_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
        self.in_flight = false;
    }

    fn on_fired(&mut self, fired_size: usize) {
        self.batch_start_time = None;
        self.in_flight = true;
        // Reset per-batch high-water; next batch starts fresh.
        self.cohort_high_water = 0;
        // Ratchet the persistent fired-cohort estimate to the actual
        // observed cohort size. Do not round near-full cohorts up to
        // `max_forward_requests`: admission can legitimately settle at
        // R=max-1, and rounding would force every later batch to wait for
        // a request that cannot arrive.
        if fired_size > self.fired_high_water {
            self.fired_high_water = fired_size;
        }
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        // (1) Structural cap. Always fires, even when in_flight,
        //     because the batch can't grow further.
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        // (2) GPU-busy gate. While the GPU is occupied, never fire
        //     — let the scheduler stay in its Wait branch and keep
        //     growing this batch with incoming requests.
        if self.in_flight {
            return Decision::Wait(Duration::from_secs(60));
        }
        // (3) Cold start. Once the GPU is free and there is no latency
        //     measurement yet, give the first serving cohort a small window
        //     to assemble. This avoids permanently paying for an initial
        //     singleton batch when many inferlets are launched concurrently
        //     but arrive a few milliseconds apart.
        if self.last_latency == 0.0 {
            let Some(start) = self.batch_start_time else {
                // Defensive: if decide is called before on_arrival, there is
                // no accumulation window to measure, so fire.
                return Decision::Fire;
            };
            let elapsed = start.elapsed();
            if elapsed < COLD_START_COHORT_WAIT {
                return Decision::Wait(COLD_START_COHORT_WAIT - elapsed);
            }
            return Decision::Fire;
        }
        // GPU is free below this point.
        // (4) Cohort cap. The target is `max(this-batch high-water,
        //     last fired R)`:
        //   - The per-batch high-water absorbs the case where the
        //     pinned_count climbs through intermediate values as
        //     inferlets re-pin between fires.
        //   - `fired_high_water` is the running max across all prior
        //     fires; it stops us from firing a partial batch when the
        //     FIRST inferlet to re-pin wins the race against the
        //     other 7+ inferlets still finishing their post-fire CPU
        //     step. Without it, `B = pinned_count = 1` at that moment
        //     trips the cap and we fire at B=1.
        let pinned = crate::context::pinned_count(self.driver_idx);
        let (resident_active, resident_pinned) = crate::context::resident_count(self.driver_idx);
        let resident = resident_active.saturating_add(resident_pinned);
        let mut active = pinned;
        // If a dense serving cohort is already mostly assembled, use the
        // resident count to wait for stragglers. This prevents a first
        // 440/512 fire from becoming the permanent cohort size, while
        // sparse agent workloads still fire from the pinned count below.
        if resident > active
            && current_forward_requests.saturating_mul(4) >= resident.saturating_mul(3)
        {
            active = resident.min(self.max_forward_requests);
        }
        if active > self.cohort_high_water {
            self.cohort_high_water = active;
        }
        let target = self.cohort_high_water.max(self.fired_high_water);
        if target > 0 && current_forward_requests >= target {
            return Decision::Fire;
        }
        // (5) Safety bound — never wait longer than one previous
        //     batch's compute time.
        let Some(start) = self.batch_start_time else {
            // Defensive: batch_start_time should always be Some when
            // decide is called (set on first on_arrival).
            return Decision::Fire;
        };
        let dense_resident_cohort =
            target >= self.max_forward_requests && resident >= target && target > 0;
        let wait_bound = if dense_resident_cohort {
            self.last_latency * 2.0
        } else {
            self.last_latency
        };
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= wait_bound {
            return Decision::Fire;
        }
        Decision::Wait(Duration::from_secs_f64(wait_bound - elapsed))
    }
}

// =============================================================================
// EagerPolicy — cohort cap + safety bound, no GPU-busy gate.
// =============================================================================
//
// Same firing logic as `AdaptivePolicy` minus the GPU-busy gate:
// fires as soon as the cohort is ready, even if the previous batch
// is still on the driver. The scheduler's semaphore handles GPU
// contention by blocking in the `Fire` branch.
//
// Slightly faster than `AdaptivePolicy` in pure steady-state (no
// GPU-busy waiting overhead, ~10 ms p50 improvement at
// RPS={1,4,16,64}), slightly slower under churn (~20 ms p50
// degradation on average, up to ~50 ms on saturated bursts —
// because arrivals during the semaphore-acquire fall into the next
// batch instead of the current one). Use this if your workload
// skews steady-state rather than bursty.

pub(super) struct EagerPolicy {
    max_forward_requests: usize,
    driver_idx: usize,
    batch_start_time: Option<Instant>,
    last_latency: f64,
    batches_completed: usize,
    cohort_high_water: usize,
}

impl EagerPolicy {
    pub fn new(max_forward_requests: usize, driver_idx: usize) -> Self {
        Self {
            max_forward_requests,
            driver_idx,
            batch_start_time: None,
            last_latency: 0.0,
            batches_completed: 0,
            cohort_high_water: 0,
        }
    }
}

impl SchedulingPolicy for EagerPolicy {
    fn on_arrival(&mut self) {
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
    }

    fn on_complete(&mut self, latency: Duration) {
        self.batches_completed += 1;
        if self.batches_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
    }

    fn on_fired(&mut self, _fired_size: usize) {
        self.batch_start_time = None;
        self.cohort_high_water = 0;
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        let active = crate::context::pinned_count(self.driver_idx);
        if active > self.cohort_high_water {
            self.cohort_high_water = active;
        }
        let target = self.cohort_high_water;
        if target > 0 && current_forward_requests >= target {
            return Decision::Fire;
        }
        if self.last_latency == 0.0 {
            return Decision::Fire;
        }
        let Some(start) = self.batch_start_time else {
            return Decision::Fire;
        };
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= self.last_latency {
            return Decision::Fire;
        }
        Decision::Wait(Duration::from_secs_f64(self.last_latency - elapsed))
    }
}

// =============================================================================
// GreedyPolicy — fire immediately. Zero state.
// =============================================================================
//
// Retained as a reference baseline. Useful when:
//   - Debugging the scheduler itself: any non-trivial policy bug
//     looks like a scheduler bug. Swap in greedy and see if the
//     symptom persists; if not, the policy is the cause.
//   - Workloads where you genuinely want zero coalescing (RPS=1,
//     latency above all else, or a single interactive session).
//
// At any RPS > 1, greedy produces small B=1–2 batches and pays
// per-batch fixed overhead on every token — measured p50 stays at
// ~1100–1250 ms regardless of load (vs 350–800 ms for adaptive
// across the same range). Throughput at saturation suffers
// correspondingly (~6% below adaptive at RPS=64 in our matrix).

pub(super) struct GreedyPolicy;

impl GreedyPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl SchedulingPolicy for GreedyPolicy {
    fn on_arrival(&mut self) {}
    fn on_complete(&mut self, _latency: Duration) {}
    fn on_fired(&mut self, _fired_size: usize) {}

    fn decide(&mut self, _current_forward_requests: usize) -> Decision {
        // The scheduler only calls `decide` when the batch is
        // non-empty, and the BatchAccumulator already enforces
        // `max_forward_requests` upstream of the policy. So: just fire,
        // every time.
        Decision::Fire
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AdaptivePolicy -----------------------------------------------------

    #[test]
    fn adaptive_cold_start_fires() {
        let mut policy = AdaptivePolicy::new(512, 0);
        // Defensive case: no batches completed and no accumulation window was
        // started, so fire to make progress.
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn adaptive_cold_start_waits_for_initial_cohort() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_arrival();
        assert!(matches!(policy.decide(1), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_fires_at_max_forward_requests_even_when_in_flight() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(0);
        // The structural cap fires unconditionally.
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn adaptive_waits_while_gpu_busy() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(0);
        // Even at large B (cohort cap would normally fire), we wait
        // because the GPU is occupied.
        assert!(matches!(policy.decide(100), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_waits_while_gpu_busy_even_before_latency_calibration() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_fired(1);
        assert!(matches!(policy.decide(1), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_resumes_after_complete() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(0);
        assert!(matches!(policy.decide(1), Decision::Wait(_)));
        policy.on_complete(Duration::from_millis(20));
        // GPU free, no batch_start_time → Fire (defensive).
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn adaptive_skips_first_batch_in_latency_update() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        assert_eq!(
            policy.last_latency, 0.0,
            "first batch is skipped (CUDA/warmup overhead inflates it)"
        );
    }

    #[test]
    fn adaptive_last_latency_tracks_most_recent_completed_batch() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(30));
        assert!((policy.last_latency - 0.030).abs() < 1e-6);
        policy.on_complete(Duration::from_millis(5));
        assert!((policy.last_latency - 0.005).abs() < 1e-6);
    }

    // ---- EagerPolicy --------------------------------------------------------

    #[test]
    fn eager_cold_start_fires() {
        let mut policy = EagerPolicy::new(512, 0);
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn eager_fires_at_max_forward_requests() {
        let mut policy = EagerPolicy::new(512, 0);
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn eager_skips_first_batch_in_latency_update() {
        let mut policy = EagerPolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        assert_eq!(policy.last_latency, 0.0);
    }

    // ---- GreedyPolicy -------------------------------------------------------

    #[test]
    fn greedy_always_fires() {
        let mut policy = GreedyPolicy::new();
        assert!(matches!(policy.decide(1), Decision::Fire));
        assert!(matches!(policy.decide(100), Decision::Fire));
        assert!(matches!(policy.decide(512), Decision::Fire));
    }
}
