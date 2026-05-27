//! Runtime-owned recurrent-state cache slot pool.
//!
//! Linear-attention models keep a compact recurrent state beside the
//! paged KV cache. Unlike KV, this state is not content-addressed or
//! paged: each resident context owns one physical slot. The context
//! actor owns allocation so drivers consume stable slot ids instead of
//! maintaining a hidden context_id -> slot LRU.

use super::pagestore::PhysicalPageId;

pub type RsSlotId = PhysicalPageId;

pub const RS_FLAG_RESET: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RsState {
    /// This driver/model has no recurrent-state cache.
    Unsupported,
    /// The model uses rs_cache, but this context has no accumulated state yet.
    Empty,
    /// The context owns a live physical rs_cache slot on its driver.
    Resident(RsSlotId),
    /// The context has token history, but its recurrent state was evicted.
    /// Restore must replay lineage into a fresh slot before user forwards.
    Missing,
}

impl RsState {
    pub(crate) fn resident_slot(self) -> Option<RsSlotId> {
        match self {
            RsState::Resident(slot) => Some(slot),
            _ => None,
        }
    }

    pub(crate) fn is_missing(self) -> bool {
        matches!(self, RsState::Missing)
    }
}

#[derive(Debug)]
pub(crate) struct RsStore {
    free: Vec<RsSlotId>,
    total: usize,
}

impl RsStore {
    pub(crate) fn new(num_slots: usize) -> Self {
        let mut free: Vec<RsSlotId> = (0..num_slots as RsSlotId).collect();
        // Pop from the end; reverse keeps slot assignment ascending.
        free.reverse();
        Self {
            free,
            total: num_slots,
        }
    }

    pub(crate) fn total_slots(&self) -> usize {
        self.total
    }

    pub(crate) fn available(&self) -> usize {
        self.free.len()
    }

    pub(crate) fn alloc(&mut self) -> Option<RsSlotId> {
        self.free.pop()
    }

    pub(crate) fn free(&mut self, slot: RsSlotId) {
        if (slot as usize) < self.total && !self.free.contains(&slot) {
            // Reuse older free slots first. This avoids immediately handing a
            // just-released recurrent-state slab to another context while
            // process cleanup messages are still being drained.
            self.free.insert(0, slot);
        }
    }
}
