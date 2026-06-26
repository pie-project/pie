//! Batch scheduling policy.
//!
//! Under FCFS the engine fires greedily:
//!
//!   - `GreedyPolicy` — fire immediately, zero waiting.
//!
//! ## Pie batching model
//!
//! Pie performs **iteration-level batching**: each in-flight context
//! re-submits a forward-pass request after every token. The scheduler
//! accumulates these into a batch and the policy decides when to fire.

use std::time::Duration;

use super::scheduler::{Decision, SchedulingPolicy};

// =============================================================================
// GreedyPolicy — fire immediately. Zero state.
// =============================================================================
//
// Under FCFS this is the only policy: every non-empty batch fires immediately,
// with no coalescing knob. Iteration-level batching still arises naturally —
// each in-flight context re-submits after every token and the accumulator
// drains all immediately-available requests before firing.

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

    // ---- GreedyPolicy -------------------------------------------------------

    #[test]
    fn greedy_always_fires() {
        let mut policy = GreedyPolicy::new();
        assert!(matches!(policy.decide(1), Decision::Fire));
        assert!(matches!(policy.decide(100), Decision::Fire));
        assert!(matches!(policy.decide(512), Decision::Fire));
    }
}
