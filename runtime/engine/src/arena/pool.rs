//! `BlockPool` — a single physical id-space free list.
//!
//! Ported from `context::pagestore::PagePool` and `context::rs_cache::RsStore`.
//! Each pool owns one contiguous id-space `0..capacity` of fixed-size blocks
//! (the arena's KV-page-sized accounting unit). The arena holds several pools —
//! one per `(kind, tier)` region — because the driver consumes distinct id
//! namespaces (`kv_page_indices` vs `rs_slot_ids`), so block ids are only
//! meaningful relative to their pool.
//!
//! Allocation is bulk (`split_off`-style, mirroring `PagePool::alloc_n`). A
//! `free_set` guards against double-free, the single most common page-pool bug
//! the old code logged loudly.

use rustc_hash::FxHashSet;

use super::BlockId;

/// A free list over a single physical id-space of fixed-size blocks.
#[derive(Debug)]
pub(crate) struct BlockPool {
    capacity: u32,
    free: Vec<BlockId>,
    free_set: FxHashSet<BlockId>,
}

impl BlockPool {
    /// Create a pool owning ids `0..capacity`, all initially free.
    pub(crate) fn new(capacity: u32) -> Self {
        let free: Vec<BlockId> = (0..capacity).collect();
        let free_set: FxHashSet<BlockId> = (0..capacity).collect();
        BlockPool {
            capacity,
            free,
            free_set,
        }
    }

    /// Total block capacity of this pool.
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Blocks currently free (allocatable).
    pub(crate) fn available(&self) -> u32 {
        self.free.len() as u32
    }

    /// Blocks currently handed out.
    pub(crate) fn used(&self) -> u32 {
        self.capacity - self.free.len() as u32
    }

    /// Allocate `n` blocks, or `None` if fewer than `n` are free. Bulk alloc via
    /// `split_off` — the returned ids are not guaranteed contiguous.
    pub(crate) fn alloc(&mut self, n: u32) -> Option<Vec<BlockId>> {
        let n = n as usize;
        if self.free.len() < n {
            return None;
        }
        let start = self.free.len() - n;
        let blocks = self.free.split_off(start);
        for b in &blocks {
            self.free_set.remove(b);
        }
        Some(blocks)
    }

    /// Return blocks to the free list. Double-frees are dropped with a logged
    /// error rather than corrupting the pool.
    pub(crate) fn free(&mut self, ids: &[BlockId]) {
        for &id in ids {
            if id >= self.capacity {
                tracing::error!(
                    block = id,
                    capacity = self.capacity,
                    "arena: free of out-of-range block id"
                );
                continue;
            }
            if !self.free_set.insert(id) {
                tracing::error!(block = id, "arena: double free of block id");
                continue;
            }
            self.free.push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free_roundtrip() {
        let mut pool = BlockPool::new(8);
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.available(), 8);

        let a = pool.alloc(3).expect("alloc 3");
        assert_eq!(a.len(), 3);
        assert_eq!(pool.used(), 3);
        assert_eq!(pool.available(), 5);

        pool.free(&a);
        assert_eq!(pool.available(), 8);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn alloc_fails_when_exhausted() {
        let mut pool = BlockPool::new(2);
        let _ = pool.alloc(2).expect("alloc 2");
        assert!(pool.alloc(1).is_none());
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn double_free_is_ignored() {
        let mut pool = BlockPool::new(4);
        let a = pool.alloc(1).expect("alloc 1");
        pool.free(&a);
        // Second free of the same id must not inflate the pool past capacity.
        pool.free(&a);
        assert_eq!(pool.available(), 4);
    }
}
