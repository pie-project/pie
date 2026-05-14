//! Batch scheduling policies.
//!
//! Three policies live here, in order of increasing patience:
//!
//!   - `GreedyPolicy`   — fire immediately, zero state.
//!   - `EagerPolicy`    — cohort cap (`pinned_count`) + `last_latency`
//!                        safety bound, no GPU-busy gate.
//!   - `AdaptivePolicy` — hot-cohort fast path on top of the eager
//!                        firing rule, plus a GPU-busy gate. Default.
//!
//! Selection via the per-model `[model.scheduler].batch_policy` field
//! in TOML (default = `adaptive`). The validator in `server/config.rs`
//! restricts the value to one of `adaptive | eager | greedy`, so the
//! selector below panics on anything else (defense-in-depth — the
//! string never reaches it unless config validation is bypassed).
//!
//! ## Why `AdaptivePolicy` is a hybrid
//!
//! It combines two signals:
//!
//!   - **Hot drain** (the just-fired cohort has re-queued) — fast
//!     path when most pinned ctxs ARE the previously-fired cohort.
//!     A clean drain says "everyone we expect to be here, is here".
//!   - **Pinned-count cap** (`B >= pinned_count(device)`) — the
//!     fallback when the hot cohort is a small subset of pinned
//!     ctxs (e.g., a few fast ctxs cycling while many slow ones are
//!     mid-WASM). Without this gate, the hot cohort would re-cycle
//!     a tiny batch indefinitely while the cold tail keeps missing
//!     the train.
//!
//! When the hot cohort drains but `B < pinned_count`, we wait up to
//! `last_latency` for the rest — the same budget Eager uses. When
//! the hot cohort can't drain at all (some ctxs got terminated or
//! stuck mid-CPU), the `hot_window` (default 5 ms, env-tunable via
//! `PIE_HOT_WINDOW_MS`) safety bound fires.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use super::scheduler::{Decision, SchedulingPolicy};

// =============================================================================
// AdaptivePolicy — default. Hot drain + pinned_count cap + GPU-busy gate.
// =============================================================================
//
// Firing rule (in evaluation order):
//
//   1. `B >= max_batch_size`                       — structural device limit.
//   2. `last_latency == 0`                         — cold start; fire to make
//                                                    progress.
//   3. `in_flight`                                 — GPU occupied; wait. The
//                                                    scheduler's tokio::select
//                                                    wakes us on completion.
//   4. `pending_hot.is_empty()` AND
//      (`pinned_count == 0` OR `B >= pinned_count`)
//                                                  — hot cohort drained AND
//                                                    every pinned ctx is in
//                                                    the batch; fire.
//   5. `pending_hot.is_empty()` AND
//      `elapsed >= last_latency`                   — adaptive safety bound;
//                                                    cold tail isn't coming.
//   6. `!pending_hot.is_empty()` AND
//      `elapsed >= hot_window`                     — hot safety bound; some hot
//                                                    ctx is stuck/terminated.
//   otherwise: `Wait(remaining budget)`.
//
// The `pinned_count == 0` clause in step 4 is a defensive "no info,
// fire" path: covers the brief window where ctxs unpin between
// batches, and lets the unit tests (which don't go through the real
// pin() path) exercise the hot-drain logic.

pub(super) struct AdaptivePolicy {
    max_batch_size: usize,
    device_idx: usize,
    /// Contexts in the previously-fired batch that haven't re-queued
    /// yet. Drained by `on_arrival`.
    pending_hot: HashSet<u64>,
    /// Safety bound for waiting on the hot drain (default 5 ms).
    hot_window: Duration,
    /// `Some(t)` while a batch is accumulating; `None` after fire.
    batch_start_time: Option<Instant>,
    /// Most recent batch's compute time, in seconds. Updated on
    /// `on_complete`. Zero until the second batch completes (the
    /// first is skipped due to CUDA/warmup overhead).
    last_latency: f64,
    batches_completed: usize,
    /// Set in `on_fired`, cleared in `on_complete`.
    in_flight: bool,
}

impl AdaptivePolicy {
    pub fn new(max_batch_size: usize, device_idx: usize, hot_window: Duration) -> Self {
        Self {
            max_batch_size,
            device_idx,
            pending_hot: HashSet::new(),
            hot_window,
            batch_start_time: None,
            last_latency: 0.0,
            batches_completed: 0,
            in_flight: false,
        }
    }
}

impl SchedulingPolicy for AdaptivePolicy {
    fn on_arrival(&mut self, ctx_id: u64) {
        if self.batch_start_time.is_none() {
            self.batch_start_time = Some(Instant::now());
        }
        self.pending_hot.remove(&ctx_id);
    }

    fn on_complete(&mut self, latency: Duration) {
        self.batches_completed += 1;
        if self.batches_completed > 1 {
            self.last_latency = latency.as_secs_f64();
        }
        self.in_flight = false;
    }

    fn on_fired(&mut self, fired_ctx_ids: &[u64]) {
        self.batch_start_time = None;
        self.in_flight = true;
        self.pending_hot.clear();
        self.pending_hot.extend(fired_ctx_ids.iter().copied());
    }

    fn decide(&self, current_batch_size: usize) -> Decision {
        // (1) Structural cap.
        if current_batch_size >= self.max_batch_size {
            return Decision::Fire;
        }
        // (2) Cold start.
        if self.last_latency == 0.0 {
            return Decision::Fire;
        }
        // (3) GPU-busy gate. Park until on_complete wakes us via
        //     the scheduler's tokio::select.
        if self.in_flight {
            return Decision::Wait(Duration::MAX);
        }

        let elapsed = self
            .batch_start_time
            .map(|t| t.elapsed())
            .unwrap_or_default();
        let last = Duration::from_secs_f64(self.last_latency);

        if self.pending_hot.is_empty() {
            // Hot cohort drained (or never existed). Fall back to
            // pinned-count cohort cap + last_latency safety bound.
            let active = crate::context::pinned_count(self.device_idx);
            if active == 0 || current_batch_size >= active {
                return Decision::Fire;
            }
            if elapsed >= last {
                return Decision::Fire;
            }
            return Decision::Wait(last - elapsed);
        }

        // Hot cohort still draining. Bounded by hot_window.
        if elapsed >= self.hot_window {
            return Decision::Fire;
        }
        Decision::Wait(self.hot_window - elapsed)
    }
}

// =============================================================================
// EagerPolicy — cohort cap (`pinned_count`) + safety bound, no GPU-busy gate.
// =============================================================================
//
// Same firing logic as `AdaptivePolicy` minus the hot-cohort drain
// signal AND the GPU-busy gate: fires as soon as the cohort is full
// or the latency budget elapses, even mid-flight. The scheduler's
// semaphore handles GPU contention by blocking in the `Fire` branch.
//
// Slightly faster in pure steady-state (no GPU-busy waiting overhead,
// no hot-cohort hysteresis), slightly slower under churn — because
// arrivals during the semaphore-acquire fall into the next batch
// rather than the current one. Use when your workload skews
// steady-state rather than bursty.

pub(super) struct EagerPolicy {
    max_batch_size: usize,
    device_idx: usize,
    batch_start_time: Option<Instant>,
    last_latency: f64,
    batches_completed: usize,
}

impl EagerPolicy {
    pub fn new(max_batch_size: usize, device_idx: usize) -> Self {
        Self {
            max_batch_size,
            device_idx,
            batch_start_time: None,
            last_latency: 0.0,
            batches_completed: 0,
        }
    }
}

impl SchedulingPolicy for EagerPolicy {
    fn on_arrival(&mut self, _ctx_id: u64) {
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

    fn on_fired(&mut self, _fired_ctx_ids: &[u64]) {
        self.batch_start_time = None;
    }

    fn decide(&self, current_batch_size: usize) -> Decision {
        if current_batch_size >= self.max_batch_size {
            return Decision::Fire;
        }
        let active = crate::context::pinned_count(self.device_idx);
        if active > 0 && current_batch_size >= active {
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
// Retained as a reference baseline. Useful for:
//   - Debugging the scheduler itself: swap in greedy, see whether
//     the symptom persists. If it doesn't, the policy is the cause.
//   - Workloads where you genuinely want zero coalescing (RPS=1,
//     latency above all else, single interactive session).

pub(super) struct GreedyPolicy;

impl GreedyPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl SchedulingPolicy for GreedyPolicy {
    fn on_arrival(&mut self, _ctx_id: u64) {}
    fn on_complete(&mut self, _latency: Duration) {}
    fn on_fired(&mut self, _fired_ctx_ids: &[u64]) {}
    fn decide(&self, _current_batch_size: usize) -> Decision {
        Decision::Fire
    }
}

// =============================================================================
// Selector
// =============================================================================

/// Build the scheduling policy named in `[model.scheduler].batch_policy`.
///
/// Panics on unknown names. The validator in `server/config.rs`
/// restricts the value at config-parse time, so this is reachable
/// only if validation is bypassed.
pub(super) fn make_policy(
    name: &str,
    max_batch_size: usize,
    device_idx: usize,
    hot_window: Duration,
) -> Box<dyn SchedulingPolicy> {
    match name {
        "adaptive" => Box::new(AdaptivePolicy::new(max_batch_size, device_idx, hot_window)),
        "eager" => Box::new(EagerPolicy::new(max_batch_size, device_idx)),
        "greedy" => Box::new(GreedyPolicy::new()),
        other => panic!(
            "unknown batch_policy {:?} (expected one of 'adaptive' | 'eager' | 'greedy')",
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AdaptivePolicy ------------------------------------------------------

    #[test]
    fn adaptive_cold_start_fires() {
        let policy = AdaptivePolicy::new(512, 0, Duration::from_millis(5));
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn adaptive_fires_at_max_batch_size_even_when_in_flight() {
        let mut policy = AdaptivePolicy::new(512, 0, Duration::from_millis(5));
        policy.on_complete(Duration::from_millis(500)); // skipped
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(&[1, 2, 3]);
        // Structural cap fires unconditionally.
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn adaptive_waits_while_gpu_busy() {
        let mut policy = AdaptivePolicy::new(512, 0, Duration::from_millis(5));
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(&[1, 2, 3]);
        assert!(matches!(policy.decide(100), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_fires_when_cohort_reformed() {
        // No pinned ctxs in tests → step 4's `active == 0` clause
        // fires as soon as pending_hot drains.
        let mut policy = AdaptivePolicy::new(512, 0, Duration::from_millis(5));
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(&[1, 2, 3]);
        policy.on_complete(Duration::from_millis(20)); // GPU free
        policy.on_arrival(1);
        assert!(matches!(policy.decide(1), Decision::Wait(_)));
        policy.on_arrival(2);
        assert!(matches!(policy.decide(2), Decision::Wait(_)));
        policy.on_arrival(3);
        assert!(matches!(policy.decide(3), Decision::Fire));
    }

    #[test]
    fn adaptive_new_arrival_is_ignored_for_cohort_gate() {
        let mut policy = AdaptivePolicy::new(512, 0, Duration::from_millis(5));
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(&[1, 2, 3]);
        policy.on_complete(Duration::from_millis(20));
        // A brand-new context arrives — it isn't in pending_hot, so
        // pending_hot is still {1,2,3} and the policy keeps waiting.
        policy.on_arrival(99);
        assert!(matches!(policy.decide(1), Decision::Wait(_)));
    }

    #[test]
    fn adaptive_safety_bound_fires_after_window() {
        let mut policy = AdaptivePolicy::new(512, 0, Duration::from_millis(5));
        policy.on_complete(Duration::from_millis(500));
        policy.on_complete(Duration::from_millis(20));
        policy.on_fired(&[1, 2, 3]);
        policy.on_complete(Duration::from_millis(20));
        // Only one hot ctx returns; the window elapses while we wait.
        policy.on_arrival(1);
        std::thread::sleep(Duration::from_millis(8));
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    // ---- EagerPolicy ---------------------------------------------------------

    #[test]
    fn eager_cold_start_fires() {
        let policy = EagerPolicy::new(512, 0);
        assert!(matches!(policy.decide(1), Decision::Fire));
    }

    #[test]
    fn eager_fires_at_max_batch_size() {
        let policy = EagerPolicy::new(512, 0);
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    #[test]
    fn eager_skips_first_batch_in_latency_update() {
        let mut policy = EagerPolicy::new(512, 0);
        policy.on_complete(Duration::from_millis(500));
        assert_eq!(policy.last_latency, 0.0);
    }

    // ---- GreedyPolicy --------------------------------------------------------

    #[test]
    fn greedy_always_fires() {
        let policy = GreedyPolicy::new();
        assert!(matches!(policy.decide(1), Decision::Fire));
        assert!(matches!(policy.decide(100), Decision::Fire));
        assert!(matches!(policy.decide(512), Decision::Fire));
    }

    // ---- Selector ------------------------------------------------------------

    #[test]
    #[should_panic(expected = "unknown batch_policy")]
    fn selector_panics_on_unknown() {
        let _ = make_policy("nonsense", 512, 0, Duration::from_millis(5));
    }
}
