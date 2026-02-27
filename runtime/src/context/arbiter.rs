//! # Arbiter Module
//!
//! Invested-importance scheduling policy for KV cache pages.
//!
//! The arbiter decides which process node should yield pages when GPU
//! memory is under pressure. It uses an invested importance model:
//!
//!   π_d = w_i · p_d
//!
//! where `w_i` is the SRPT-derived process weight (`total_steps / remaining_steps`)
//! and `p_d` is the page count for that process on device `d`.
//! Higher invested importance = harder to evict.
//!
//! ## Per-Device Decomposition
//!
//! Each device is an independent subproblem: a process's priority on
//! device `d` depends only on its pages on `d`, not on other devices.
//!
//! ## Eviction Condition
//!
//! Evict victim V for requester R iff:
//!   π_V < π_R(p_R + n) = w_R · (p_R + n)
//!
//! This is the post-allocation floor: only evict if the requester's
//! invested importance AFTER allocation exceeds the victim's current.

use std::collections::HashMap;
use std::time::Instant;

use crate::device::DeviceId;
use crate::process::ProcessId;

// =============================================================================
// DevicePages (per-device accounting)
// =============================================================================

/// Per-device page accounting for a process node.
#[derive(Debug, Clone)]
struct DevicePages {
    committed: usize,
    working: usize,
}

impl DevicePages {
    fn new() -> Self {
        DevicePages { committed: 0, working: 0 }
    }

    fn total(&self) -> usize {
        self.committed + self.working
    }
}

// =============================================================================
// Node (internal)
// =============================================================================

/// Per-process state. Page accounting is per-device.
#[derive(Debug, Clone)]
struct Node {
    /// Number of active (non-suspended) contexts for this process (global).
    active_contexts: usize,
    /// SRPT weight: `total_steps / remaining_steps` for the owning workflow.
    /// Higher = workflow closer to completion = harder to evict. Default 1.0.
    weight: f64,
    /// Per-device page accounting.
    devices: HashMap<DeviceId, DevicePages>,
    /// Last access time.
    last_access: Instant,
}

impl Node {
    fn new() -> Self {
        Node {
            active_contexts: 0,
            weight: 1.0,
            devices: HashMap::new(),
            last_access: Instant::now(),
        }
    }

    /// Get or create per-device accounting.
    fn device_mut(&mut self, dev: DeviceId) -> &mut DevicePages {
        self.devices.entry(dev).or_insert_with(DevicePages::new)
    }

    /// Pages on a specific device.
    fn pages_on(&self, dev: DeviceId) -> usize {
        self.devices.get(&dev).map(|d| d.total()).unwrap_or(0)
    }

    /// Invested importance on a specific device.
    ///
    /// π_d = w_i · p_d
    fn priority_on(&self, dev: DeviceId) -> f64 {
        self.priority_on_at(dev, self.pages_on(dev))
    }

    /// Invested importance at an arbitrary page count on a device.
    fn priority_on_at(&self, _dev: DeviceId, p: usize) -> f64 {
        self.weight * p as f64
    }

    /// Total pages across all devices.
    fn total_pages(&self) -> usize {
        self.devices.values().map(|d| d.total()).sum()
    }
}

// =============================================================================
// Arbiter
// =============================================================================

/// The arbiter tracks per-process, per-device page budgets and answers
/// policy queries.
#[derive(Debug)]
pub struct Arbiter {
    nodes: HashMap<ProcessId, Node>,
}

impl Arbiter {
    pub fn new() -> Self {
        Arbiter {
            nodes: HashMap::new(),
        }
    }

    // ==================== Node Lifecycle ====================

    pub fn remove_node(&mut self, pid: &ProcessId) {
        self.nodes.remove(pid);
    }

    pub fn node_is_empty(&self, pid: &ProcessId) -> bool {
        self.nodes.get(pid).map_or(true, |n| n.total_pages() == 0)
    }

    // ==================== DAG Weights ====================

    pub fn set_node_weight(&mut self, pid: ProcessId, weight: f64) {
        let n = self.nodes.entry(pid).or_insert_with(Node::new);
        n.weight = weight;
    }

    // ==================== Accounting ====================

    pub fn activate(&mut self, pid: ProcessId) {
        let n = self.node_mut(pid);
        n.active_contexts += 1;
        n.last_access = Instant::now();
    }

    pub fn deactivate(&mut self, pid: ProcessId) {
        let n = self.node_mut(pid);
        n.active_contexts = n.active_contexts.saturating_sub(1);
    }

    pub fn add_working(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let n = self.node_mut(pid);
        let d = n.device_mut(device);
        d.working += pages;
        n.last_access = Instant::now();
    }

    pub fn remove_working(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let n = self.node_mut(pid);
        let d = n.device_mut(device);
        d.working = d.working.saturating_sub(pages);
    }

    pub fn commit(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let n = self.node_mut(pid);
        let d = n.device_mut(device);
        d.committed += pages;
        d.working = d.working.saturating_sub(pages);
    }

    pub fn uncommit(&mut self, pid: ProcessId, device: DeviceId, pages: usize) {
        let n = self.node_mut(pid);
        let d = n.device_mut(device);
        d.committed = d.committed.saturating_sub(pages);
    }

    pub fn suspend(&mut self, pid: ProcessId, device: DeviceId, committed: usize, working: usize) {
        let n = self.node_mut(pid);
        n.active_contexts = n.active_contexts.saturating_sub(1);
        let d = n.device_mut(device);
        d.committed = d.committed.saturating_sub(committed);
        d.working = d.working.saturating_sub(working);
    }

    pub fn restore(&mut self, pid: ProcessId, device: DeviceId, committed: usize, working: usize) {
        let n = self.node_mut(pid);
        n.active_contexts += 1;
        let d = n.device_mut(device);
        d.committed += committed;
        d.working += working;
    }

    pub fn touch(&mut self, pid: ProcessId) {
        if let Some(n) = self.nodes.get_mut(&pid) {
            n.last_access = Instant::now();
        }
    }

    // ==================== Policy Queries ====================

    /// Invested importance of a process on a specific device: w_i · p_d.
    pub fn priority(&self, pid: &ProcessId, device: DeviceId) -> f64 {
        self.nodes.get(pid).map(|n| n.priority_on(device)).unwrap_or(0.0)
    }

    /// Pages held by a process on a specific device.
    pub fn node_pages(&self, pid: &ProcessId, device: DeviceId) -> usize {
        self.nodes.get(pid).map(|n| n.pages_on(device)).unwrap_or(0)
    }

    /// Invested importance at an arbitrary page count on a device.
    pub fn priority_at(&self, pid: &ProcessId, device: DeviceId, pages: usize) -> f64 {
        self.nodes.get(pid).map(|n| n.priority_on_at(device, pages)).unwrap_or(0.0)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ==================== Internal Helpers ====================

    fn node_mut(&mut self, pid: ProcessId) -> &mut Node {
        self.nodes.entry(pid).or_insert_with(Node::new)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn pid(n: u64) -> ProcessId { Uuid::from_u128(n as u128) }
    const DEV0: DeviceId = 0;
    const DEV1: DeviceId = 1;

    #[test]
    fn node_created_on_activate() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        assert_eq!(arb.node_count(), 1);
    }

    #[test]
    fn per_device_isolation() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.add_working(pid(1), DEV0, 50);
        arb.add_working(pid(1), DEV1, 10);

        // Priority is device-specific
        let p0 = arb.priority(&pid(1), DEV0);
        let p1 = arb.priority(&pid(1), DEV1);
        assert!(p0 > p1, "device 0 (50p) should have higher priority than device 1 (10p): {p0} > {p1}");

        // Pages are device-specific
        assert_eq!(arb.node_pages(&pid(1), DEV0), 50);
        assert_eq!(arb.node_pages(&pid(1), DEV1), 10);
    }

    #[test]
    fn priority_increases_with_more_pages() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));

        arb.add_working(pid(1), DEV0, 5);
        let p5 = arb.priority(&pid(1), DEV0);

        arb.add_working(pid(1), DEV0, 15);
        let p20 = arb.priority(&pid(1), DEV0);

        assert!(p20 > p5, "more pages = higher invested importance: p20={p20} > p5={p5}");
    }

    #[test]
    fn higher_weight_means_higher_priority() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.activate(pid(2));
        arb.add_working(pid(1), DEV0, 10);
        arb.add_working(pid(2), DEV0, 10);

        arb.set_node_weight(pid(1), 5.0);
        arb.set_node_weight(pid(2), 1.0);

        let p1 = arb.priority(&pid(1), DEV0);
        let p2 = arb.priority(&pid(2), DEV0);
        assert!(p1 > p2, "higher weight should mean higher priority: {p1} > {p2}");
    }

    #[test]
    fn commit_moves_working_to_committed() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.add_working(pid(1), DEV0, 10);

        assert_eq!(arb.node_pages(&pid(1), DEV0), 10);

        arb.commit(pid(1), DEV0, 5);
        assert_eq!(arb.node_pages(&pid(1), DEV0), 10); // total unchanged
    }

    #[test]
    fn suspend_removes_pages_on_device() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.add_working(pid(1), DEV0, 5);
        arb.commit(pid(1), DEV0, 5);
        arb.add_working(pid(1), DEV0, 3);
        arb.add_working(pid(1), DEV1, 7); // pages on another device

        assert_eq!(arb.node_pages(&pid(1), DEV0), 8);
        assert_eq!(arb.node_pages(&pid(1), DEV1), 7);

        arb.suspend(pid(1), DEV0, 5, 3);
        assert_eq!(arb.node_pages(&pid(1), DEV0), 0);
        assert_eq!(arb.node_pages(&pid(1), DEV1), 7); // unaffected
    }

    #[test]
    fn bigger_node_has_higher_priority() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.add_working(pid(1), DEV0, 100);

        arb.activate(pid(2));
        arb.add_working(pid(2), DEV0, 10);

        let p1 = arb.priority(&pid(1), DEV0);
        let p2 = arb.priority(&pid(2), DEV0);
        assert!(p1 > p2, "bigger node should have higher priority: {p1} > {p2}");
    }

    #[test]
    fn cross_device_pages_dont_affect_priority() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.activate(pid(2));

        // Node 1: 30 pages on DEV0, 40 on DEV1
        arb.add_working(pid(1), DEV0, 30);
        arb.add_working(pid(1), DEV1, 40);

        // Node 2: 30 pages on DEV0 only
        arb.add_working(pid(2), DEV0, 30);

        // On DEV0, both have 30 pages → same priority
        let p1 = arb.priority(&pid(1), DEV0);
        let p2 = arb.priority(&pid(2), DEV0);
        assert!((p1 - p2).abs() < 1e-10,
            "same device pages should yield same priority: {p1} vs {p2}");

        // Node 1's DEV1 pages don't inflate its DEV0 priority
        assert_eq!(arb.node_pages(&pid(1), DEV0), 30);
        assert_eq!(arb.node_pages(&pid(2), DEV0), 30);
    }

    /// Proves the anti-thrashing property of invested importance.
    ///
    /// If A (0 pages) evicts B (30 pages) by requesting 30 pages,
    /// then B (now 0 pages) requesting 30 pages CANNOT evict A back,
    /// because B's post-allocation floor = w_B·30 = A's current priority.
    #[test]
    fn post_allocation_floor_prevents_thrashing() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1)); // Node A
        arb.activate(pid(2)); // Node B

        // State after A evicted B: A has 30 pages, B has 0
        arb.add_working(pid(1), DEV0, 30);

        // B wants to restore 30 pages. Its post-allocation floor:
        let b_current = arb.node_pages(&pid(2), DEV0); // = 0
        let b_floor = arb.priority_at(&pid(2), DEV0, b_current + 30);
        // b_floor = w·30 = 30

        // A's current priority on DEV0:
        let a_priority = arb.priority(&pid(1), DEV0);
        // a_priority = w·30 = 30

        // Key: A's priority >= B's floor → A is NOT evictable by B
        assert!(a_priority >= b_floor,
            "A ({a_priority}) should NOT be evictable by B (floor={b_floor})");

        // They're equal: symmetric nodes at same target = no eviction
        assert!((a_priority - b_floor).abs() < 1e-10);
    }

    #[test]
    fn suspend_restore_round_trip() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.add_working(pid(1), DEV0, 5);
        arb.commit(pid(1), DEV0, 5);
        arb.add_working(pid(1), DEV0, 3);

        let pre_pages = arb.node_pages(&pid(1), DEV0);
        assert_eq!(pre_pages, 8); // 5 committed + 3 working

        // suspend zeros everything
        arb.suspend(pid(1), DEV0, 5, 3);
        assert_eq!(arb.node_pages(&pid(1), DEV0), 0);

        // restore should bring it back exactly
        arb.restore(pid(1), DEV0, 5, 3);
        assert_eq!(arb.node_pages(&pid(1), DEV0), pre_pages,
            "round-trip should preserve page count");
    }

    /// Simulates the replay restore scenario:
    /// suspend all → restore only prefix-matched committed → add_working → commit replayed.
    /// Verifies no double-counting of committed pages.
    #[test]
    fn suspend_partial_restore_then_commit() {
        let mut arb = Arbiter::new();
        arb.activate(pid(1));
        arb.add_working(pid(1), DEV0, 10);
        arb.commit(pid(1), DEV0, 10);
        arb.add_working(pid(1), DEV0, 3);

        let pre_pages = arb.node_pages(&pid(1), DEV0);
        assert_eq!(pre_pages, 13); // 10 committed + 3 working

        // Suspend: zeros committed and working
        arb.suspend(pid(1), DEV0, 10, 3);
        assert_eq!(arb.node_pages(&pid(1), DEV0), 0);

        // Restore working pages (swap-in)
        arb.add_working(pid(1), DEV0, 3);

        // Restore only prefix-matched committed (6 of 10 matched)
        let prefix_matched = 6;
        arb.restore(pid(1), DEV0, prefix_matched, 0);

        // Replay remaining 4 pages: add as working, then commit
        arb.add_working(pid(1), DEV0, 4);
        arb.commit(pid(1), DEV0, 4); // working → committed

        // Total should equal pre-suspend: 10 committed + 3 working = 13
        assert_eq!(arb.node_pages(&pid(1), DEV0), pre_pages,
            "partial restore + replay commit must not double-count");
    }
}
