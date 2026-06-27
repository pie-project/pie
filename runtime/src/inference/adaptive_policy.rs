//! Batch scheduling policies.
//!
//! Three policies live here:
//!
//!   - `GreedyPolicy` — fire immediately, zero waiting.
//!   - `EagerPolicy` — wait for the peer cohort to gather (using a
//!     `pinned_count`-driven cap and a `last_latency` safety bound).
//!   - `AdaptivePolicy` (default) — fire on a structural cap or on
//!     `fired_high_water`, with `last_latency` as a deadlock
//!     watchdog. One heuristic (`fired_high_water`), explicitly
//!     scoped.
//!
//! Selection via the per-model `[model.scheduler].policy` field in
//! the TOML config (default = `adaptive`).
//!
//! ## Pie batching model
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a batch and the policy decides when to fire.

use std::time::{Duration, Instant};

use super::scheduler::{Decision, SchedulingPolicy};

// =============================================================================
// AdaptivePolicy — fire on structural cap or historical cohort size.
// =============================================================================
//
// Firing rule (in evaluation order):
//
//   1. `B >= max_forward_requests`  — structural driver limit.
//   2. `fired_high_water == 0`      — no historical fire yet; fire
//                                     whatever we have (cold start).
//   3. `B >= fired_high_water`      — we've matched the largest cohort
//                                     ever fired; firing more won't
//                                     produce a bigger batch from
//                                     existing peers.
//   4. `elapsed >= last_latency`    — deadlock watchdog: the cohort
//                                     never grew back to
//                                     `fired_high_water` (active
//                                     concurrency dropped — e.g.,
//                                     inferlets finished). Fire what
//                                     we have so the system makes
//                                     progress. `last_latency` is a
//                                     natural, self-recalibrating
//                                     bound (don't wait longer than
//                                     one fire takes); not a tuning
//                                     parameter.
//   otherwise: `Wait(last_latency - elapsed)`.
//
// `fired_high_water` is the only firing-rule concept; `last_latency`
// is a watchdog to prevent indefinite parking, not a steady-state
// trigger.

pub(super) struct AdaptivePolicy {
    max_forward_requests: usize,
    /// Largest batch size that has fired. Monotonic; only ratchets
    /// upward. The single firing-rule concept.
    fired_high_water: usize,
    /// Previous fire's compute time, in seconds. Used solely as the
    /// deadlock watchdog (rule 4). Zero until the second fire
    /// completes (first fire's latency is unrepresentative).
    last_latency: f64,
    /// Total fires the policy has observed completing. Used to
    /// suppress the first `on_complete` from setting `last_latency`.
    fires_completed: usize,
    /// When the current batch started accumulating. Reset on fire.
    batch_start_time: Option<Instant>,
}

impl AdaptivePolicy {
    pub fn new(max_forward_requests: usize, _driver_idx: usize) -> Self {
        Self {
            max_forward_requests,
            fired_high_water: 0,
            last_latency: 0.0,
            fires_completed: 0,
            batch_start_time: None,
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
        self.fires_completed += 1;
        if self.fires_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
    }

    fn on_fired(&mut self, fired_size: usize) {
        self.batch_start_time = None;
        if fired_size > self.fired_high_water {
            self.fired_high_water = fired_size;
        }
    }

    fn decide(&mut self, current_forward_requests: usize) -> Decision {
        if current_forward_requests >= self.max_forward_requests {
            return Decision::Fire;
        }
        if self.fired_high_water == 0 || current_forward_requests >= self.fired_high_water {
            return Decision::Fire;
        }
        let Some(start) = self.batch_start_time else {
            // Defensive: `batch_start_time` should be `Some` whenever
            // `decide` runs (set on first `on_arrival`).
            return Decision::Fire;
        };
        // Cold start: no fire has completed yet, no bound to apply.
        if self.last_latency == 0.0 {
            return Decision::Fire;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= self.last_latency {
            return Decision::Fire;
        }
        Decision::Wait(Duration::from_secs_f64(self.last_latency - elapsed))
    }
}

// =============================================================================
// EagerPolicy — pinned_count cohort cap + last_latency safety bound.
// =============================================================================
//
// Fires when the live `pinned_count(driver_idx)` cohort has fully
// assembled in the batch, or when one `last_latency` has elapsed.
// Tracks `cohort_high_water` per-batch (monotonic over the
// accumulation window) so transient `pinned_count` dips between
// pin/unpin don't fire a partial batch.

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
        // Per-driver pinned-context count is gone with the context actor
        // (Phase 5); the cohort-high-water path degrades to the latency-based
        // decision below.
        let active = 0;
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
        // No fire has happened yet (fired_high_water == 0); fire to
        // make progress.
        let mut policy = AdaptivePolicy::new(512, 0);
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn adaptive_fires_at_structural_cap() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        // Structural cap fires unconditionally.
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn adaptive_fires_at_fired_high_water() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        policy.on_arrival();
        // Matched the historical cohort size; fire.
        assert!(matches!(policy.decide(64), Decision::Fire));
    }

    #[test]
    fn adaptive_waits_below_fired_high_water() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        policy.on_arrival();
        // Below the historical cohort — wait for stragglers.
        assert!(matches!(policy.decide(32), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_watchdog_fires_after_last_latency() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_micros(1)); // tiny → watchdog elapses fast
        policy.on_fired(64);
        policy.on_arrival();
        std::thread::sleep(Duration::from_millis(2));
        // last_latency was 1 µs; 2 ms has elapsed → watchdog fires
        // even though batch is below fired_high_water.
        assert!(matches!(policy.decide(32), Decision::Fire));
    }

    #[test]
    fn adaptive_cold_start_fires_even_without_batch_start_time() {
        // No arrival recorded yet (batch_start_time is None). The
        // defensive branch fires immediately.
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(64);
        assert!(matches!(policy.decide(32), Decision::Fire));
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

    #[test]
    fn adaptive_fired_high_water_ratchets_only_upward() {
        let mut policy = AdaptivePolicy::new(512, 0);
        policy.on_fired(40);
        policy.on_fired(80);
        policy.on_fired(60); // smaller — must not pull the watermark down
        assert_eq!(policy.fired_high_water, 80);
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
