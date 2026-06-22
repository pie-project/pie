//! Health-check axis (cross-cutting).
//!
//! Tracks liveness by heartbeat. Each node periodically heartbeats; the checker
//! grades a node [`HealthStatus::Healthy`] / [`Degraded`] / [`Unreachable`]
//! based on how long ago its last heartbeat arrived.
//!
//! Time is modelled as a monotonic logical clock the caller advances with
//! [`HealthChecker::tick`], rather than reading the wall clock directly. That
//! keeps coordination logic deterministic and unit-testable now; the
//! process-mode serve loop will drive `tick` from a real timer once networking
//! is wired in a later step.
//!
//! [`Degraded`]: HealthStatus::Degraded
//! [`Unreachable`]: HealthStatus::Unreachable

use std::collections::HashMap;

use pie_schema::{HealthStatus, WorkerId};

/// Heartbeat-based liveness tracker.
#[derive(Debug)]
pub struct HealthChecker {
    /// Logical tick at which each node last heartbeated.
    last_seen: HashMap<WorkerId, u64>,
    /// Current logical time.
    now: u64,
    /// Ticks of silence after which a node is graded `Degraded`.
    degrade_after: u64,
    /// Ticks of silence after which a node is graded `Unreachable`.
    unreachable_after: u64,
}

impl HealthChecker {
    /// New checker. A node is `Degraded` after `degrade_after` silent ticks and
    /// `Unreachable` after `unreachable_after` (which must be `>= degrade_after`).
    pub fn new(degrade_after: u64, unreachable_after: u64) -> Self {
        debug_assert!(unreachable_after >= degrade_after);
        Self {
            last_seen: HashMap::new(),
            now: 0,
            degrade_after,
            unreachable_after,
        }
    }

    /// Start tracking `node`, treating registration as its first heartbeat.
    pub fn track(&mut self, node: WorkerId) {
        self.last_seen.insert(node, self.now);
    }

    /// Stop tracking `node` (it left the cluster).
    pub fn forget(&mut self, node: WorkerId) {
        self.last_seen.remove(&node);
    }

    /// Record a heartbeat from `node` at the current tick.
    pub fn heartbeat(&mut self, node: WorkerId) {
        self.last_seen.insert(node, self.now);
    }

    /// Advance the logical clock by one tick.
    pub fn tick(&mut self) {
        self.now += 1;
    }

    /// Current liveness verdict for `node`, or `None` if it is not tracked.
    pub fn status(&self, node: WorkerId) -> Option<HealthStatus> {
        self.last_seen.get(&node).map(|&seen| {
            let silent = self.now.saturating_sub(seen);
            if silent >= self.unreachable_after {
                HealthStatus::Unreachable
            } else if silent >= self.degrade_after {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            }
        })
    }

    /// Snapshot of every tracked node's status.
    pub fn report(&self) -> Vec<(WorkerId, HealthStatus)> {
        self.last_seen
            .keys()
            .map(|&node| (node, self.status(node).expect("tracked node has a status")))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grades_by_silence() {
        let mut h = HealthChecker::new(2, 4);
        h.track(WorkerId(1));
        assert_eq!(h.status(WorkerId(1)), Some(HealthStatus::Healthy));

        h.tick(); // silent 1
        assert_eq!(h.status(WorkerId(1)), Some(HealthStatus::Healthy));
        h.tick(); // silent 2 -> degraded
        assert_eq!(h.status(WorkerId(1)), Some(HealthStatus::Degraded));
        h.tick();
        h.tick(); // silent 4 -> unreachable
        assert_eq!(h.status(WorkerId(1)), Some(HealthStatus::Unreachable));
    }

    #[test]
    fn heartbeat_resets() {
        let mut h = HealthChecker::new(2, 4);
        h.track(WorkerId(1));
        h.tick();
        h.tick(); // degraded
        h.heartbeat(WorkerId(1));
        assert_eq!(h.status(WorkerId(1)), Some(HealthStatus::Healthy));
    }

    #[test]
    fn untracked_is_none() {
        let h = HealthChecker::new(2, 4);
        assert_eq!(h.status(WorkerId(7)), None);
    }

    #[test]
    fn report_lists_tracked() {
        let mut h = HealthChecker::new(2, 4);
        h.track(WorkerId(1));
        h.track(WorkerId(2));
        assert_eq!(h.report().len(), 2);
        h.forget(WorkerId(1));
        assert_eq!(h.report().len(), 1);
    }
}
