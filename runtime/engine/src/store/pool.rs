//! Typed physical-id free list over a driver-preallocated static pool.
//!
//! One pool per resource kind (`KvBackingPool`, `StateBackingPool`, ...).
//! A pool only reserves and releases stable ids over static device memory; it
//! owns no CoW logic, hash maintenance, mapping, residency, or refcounts
//! (kv_refact.md, `store/pool.rs`). Freed ids are recycled only after the
//! completion epoch of their last in-flight user retires.
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

/// A typed physical id backed by a pool slot. Implemented by
/// `PhysicalKvPageId` and RS-specific ids.
pub trait PoolId: Copy {
    fn from_index(index: u32) -> Self;
    fn index(self) -> u32;
}

/// Free list with completion-epoch-delayed recycling.
pub struct Pool<I> {
    free: Vec<I>,
    /// Ids waiting for their epoch to retire before becoming allocatable.
    pending: Vec<(u64, Vec<I>)>,
    base: u32,
    capacity: u32,
}

impl<I: PoolId> Pool<I> {
    pub fn new(capacity: u32) -> Self {
        Self::new_range(0, capacity)
    }

    pub fn new_range(base: u32, capacity: u32) -> Self {
        let end = base
            .checked_add(capacity)
            .expect("pool id range overflows u32");
        Self {
            // Pop order: ascending ids first (cosmetic, deterministic tests).
            free: (base..end).rev().map(I::from_index).collect(),
            pending: Vec::new(),
            base,
            capacity,
        }
    }

    /// Allocate one id, or `None` on exhaustion. Exhaustion propagates up to
    /// the scheduler's contention ladder; the pool itself never blocks.
    pub fn try_alloc(&mut self) -> Option<I> {
        self.free.pop()
    }

    /// Allocate `n` ids all-or-nothing.
    pub fn try_alloc_n(&mut self, n: usize) -> Option<Vec<I>> {
        if self.free.len() < n {
            return None;
        }
        let at = self.free.len() - n;
        Some(self.free.split_off(at))
    }

    /// Queue ids for recycling once `epoch` retires.
    pub fn recycle_after_epoch(&mut self, ids: Vec<I>, epoch: u64) {
        if !ids.is_empty() {
            self.pending.push((epoch, ids));
        }
    }

    /// Return ids that were reserved but never published or submitted to a
    /// driver operation. No completion epoch is required because no device
    /// user could have observed them.
    pub fn release_reserved(&mut self, ids: Vec<I>) {
        debug_assert!(ids.iter().all(|id| {
            id.index() >= self.base && id.index() < self.base.saturating_add(self.capacity)
        }));
        debug_assert!(
            ids.iter()
                .all(|id| !self.free.iter().any(|free| free.index() == id.index()))
        );
        self.free.extend(ids);
    }

    /// Retire all epochs `<= epoch`, returning their ids to the free list.
    pub fn retire_through(&mut self, epoch: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= epoch {
                let (_, ids) = self.pending.swap_remove(i);
                self.free.extend(ids);
            } else {
                i += 1;
            }
        }
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn pending_recycle(&self) -> usize {
        self.pending.iter().map(|(_, ids)| ids.len()).sum()
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}
