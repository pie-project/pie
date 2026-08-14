//! Typed physical-id free list over a driver-preallocated static pool.
//!
//! One pool per resource kind (`KvBackingPool`, `StateBackingPool`, ...).
//! A pool only reserves and releases stable ids over static device memory; it
//! owns no CoW logic, hash maintenance, mapping, residency, or refcounts
//! (kv_refact.md, `store/pool.rs`). Freed ids are recycled only after the
//! completion epoch of their last in-flight user retires.
//!
//! Allocation order is a CAPACITY CONTRACT, not a cosmetic choice: the CUDA
//! driver's elastic arena commits a physical PREFIX of the id space up to the
//! highest id any frame references (`required_kv_pages` is a high-water over
//! translated page ids), so every allocation takes the LOWEST free id. That
//! keeps the committed prefix equal to actual residency. The previous
//! LIFO free list let alloc/free churn ratchet the high-water far above the
//! live page count, and frames whose real footprint fit the budget were
//! rejected with "frame commit impossible" (tts-bench emergency #2017:
//! kv_required 11.3k-15.4k pages against a budget sized for the live set).
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::marker::PhantomData;

/// A typed physical id backed by a pool slot. Implemented by
/// `PhysicalKvPageId` and RS-specific ids.
pub trait PoolId: Copy {
    fn from_index(index: u32) -> Self;
    fn index(self) -> u32;
}

/// Lowest-id-first free set with completion-epoch-delayed recycling.
pub struct Pool<I> {
    /// Free ids, ordered so allocation always takes the smallest (see the
    /// module docs for why this is load-bearing).
    free: BTreeSet<u32>,
    /// Ids waiting for their epoch to retire before becoming allocatable.
    pending: Vec<(u64, Vec<I>)>,
    base: u32,
    capacity: u32,
    _marker: PhantomData<I>,
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
            free: (base..end).collect(),
            pending: Vec::new(),
            base,
            capacity,
            _marker: PhantomData,
        }
    }

    /// Allocate the lowest free id, or `None` on exhaustion. Exhaustion
    /// propagates up to the scheduler's contention ladder; the pool itself
    /// never blocks.
    pub fn try_alloc(&mut self) -> Option<I> {
        self.free.pop_first().map(I::from_index)
    }

    /// Allocate the `n` lowest free ids all-or-nothing.
    pub fn try_alloc_n(&mut self, n: usize) -> Option<Vec<I>> {
        if self.free.len() < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let index = self.free.pop_first().expect("length checked above");
            out.push(I::from_index(index));
        }
        Some(out)
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
        debug_assert!(ids.iter().all(|id| !self.free.contains(&id.index())));
        self.free.extend(ids.iter().map(|id| id.index()));
    }

    /// Retire all epochs `<= epoch`, returning their ids to the free set.
    pub fn retire_through(&mut self, epoch: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= epoch {
                let (_, ids) = self.pending.swap_remove(i);
                self.free.extend(ids.iter().map(|id| id.index()));
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

    pub fn highest_in_use_exclusive(&self) -> u32 {
        let end = self.base.saturating_add(self.capacity);
        (self.base..end)
            .rev()
            .find(|index| !self.free.contains(index))
            .map_or(0, |index| index - self.base + 1)
    }
}

#[cfg(test)]
mod low_water_tests {
    use super::{Pool, PoolId};

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Id(u32);
    impl PoolId for Id {
        fn from_index(index: u32) -> Self {
            Self(index)
        }
        fn index(self) -> u32 {
            self.0
        }
    }

    /// Allocation is lowest-id-first from a fresh pool.
    #[test]
    fn allocation_starts_at_the_lowest_id() {
        let mut pool: Pool<Id> = Pool::new(8);
        let ids = pool.try_alloc_n(3).unwrap();
        assert_eq!(ids.iter().map(|id| id.0).collect::<Vec<_>>(), [0, 1, 2]);
    }

    /// The prefix-commit contract: after churn frees low ids, the next
    /// allocation reuses THEM, not fresh high ids — the id high-water
    /// tracks the live count, so the driver's prefix commit prices actual
    /// residency instead of allocation history.
    #[test]
    fn freed_low_ids_are_reused_before_fresh_high_ids() {
        let mut pool: Pool<Id> = Pool::new(64);
        let first = pool.try_alloc_n(8).unwrap();
        // Free the low half out of order (churn), keep 4..8 live.
        pool.release_reserved(first[..4].iter().rev().copied().collect());
        let reused = pool.try_alloc_n(4).unwrap();
        assert_eq!(reused.iter().map(|id| id.0).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!(pool.highest_in_use_exclusive(), 8);
    }

    /// Epoch-recycled ids come back into the ordered set, not to the tail.
    #[test]
    fn recycled_ids_rejoin_the_ordered_set() {
        let mut pool: Pool<Id> = Pool::new(16);
        let ids = pool.try_alloc_n(4).unwrap();
        pool.recycle_after_epoch(ids[..2].to_vec(), 7);
        assert_eq!(pool.try_alloc().unwrap().0, 4);
        pool.retire_through(7);
        assert_eq!(pool.try_alloc().unwrap().0, 0);
    }

    /// A ranged pool allocates from its own base upward.
    #[test]
    fn ranged_pool_allocates_from_its_base() {
        let mut pool: Pool<Id> = Pool::new_range(100, 4);
        assert_eq!(pool.try_alloc().unwrap().0, 100);
        assert_eq!(pool.highest_in_use_exclusive(), 1);
    }
}
