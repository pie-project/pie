//! The one queue (D1). Allocation waiters and restore entries order by one
//! spawn clock; the head — the oldest unmet entry, of either kind — has
//! first claim on every freed page and accumulates until its demand fits.
//! Younger entries are served only from surplus beyond the head's
//! accumulation, in the same spawn order. Drainage is provable because the
//! head's demand is finite and freed pages flow to it monotonically;
//! victim-restore thrash is impossible because a victim is by construction
//! younger than the waiter it yielded to, so its restore entry queues
//! behind. There is no separate restore phase, no utilization pause, and no
//! aging override — head-first-claim subsumes all three.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Notify;

use super::ProcessId;
use super::grant::{Demand, DevicePageReservation};

/// What kind of service an entry waits for.
pub(super) enum EntryKind {
    /// A parked allocation; its `acquire` future waits on `notify`.
    Allocation {
        quorum_id: ProcessId,
        notify: Arc<Notify>,
        /// Duplicate callers from the same pipeline scope aggregated here.
        waiters: usize,
        /// Whether the A×B pipeline-leave was already reported for this entry.
        parked_reported: bool,
        /// The issued grant, parked until its `acquire` collects it.
        grant: Option<super::grant::AllocationGrant>,
    },
    /// A suspended process awaiting its restore grant.
    Restore {
        /// When the process finished suspending — restore-wait diagnostics.
        suspended_at: Instant,
    },
}

/// One queue entry. `seq` is the owning process's spawn clock; entries with
/// the same seq (duplicate pipelines of one process) tie-break by insertion.
pub(super) struct Entry {
    pub pid: ProcessId,
    pub demand: Demand,
    pub kind: EntryKind,
    /// Head-first-claim accumulation: concrete pages reserved toward this
    /// entry. Only the head accumulates partially; younger entries fill in
    /// one piece from surplus. Bounded by one process's demand — the price
    /// of the drainage guarantee.
    pub kv_accum: DevicePageReservation,
}

impl Entry {
    /// Device pages still missing before this entry's KV side is covered.
    pub fn kv_missing(&self) -> u32 {
        self.demand
            .kv_pages
            .saturating_sub(self.kv_accum.len() as u32)
    }

    /// Whether this entry still waits for service (an issued, uncollected
    /// grant no longer competes for pages).
    pub fn is_unmet(&self) -> bool {
        match &self.kind {
            EntryKind::Allocation { grant, .. } => grant.is_none(),
            EntryKind::Restore { .. } => true,
        }
    }
}

/// Key: spawn clock first, insertion order second.
pub(super) type EntryKey = (u64, u64);

/// The single service queue, ordered by `(spawn seq, insertion id)`.
#[derive(Default)]
pub(super) struct Queue {
    entries: BTreeMap<EntryKey, Entry>,
}

impl Queue {
    pub fn insert(&mut self, key: EntryKey, entry: Entry) {
        let displaced = self.entries.insert(key, entry);
        debug_assert!(displaced.is_none(), "queue keys are unique");
    }

    pub fn get(&self, key: &EntryKey) -> Option<&Entry> {
        self.entries.get(key)
    }

    pub fn get_mut(&mut self, key: &EntryKey) -> Option<&mut Entry> {
        self.entries.get_mut(key)
    }

    pub fn remove(&mut self, key: &EntryKey) -> Option<Entry> {
        self.entries.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&EntryKey, &Entry)> {
        self.entries.iter()
    }

    /// The head: the oldest unmet entry, of either kind.
    pub fn head(&self) -> Option<(&EntryKey, &Entry)> {
        self.entries.iter().find(|(_, entry)| entry.is_unmet())
    }

    /// Remove every entry satisfying `predicate`, returning them (the caller
    /// disposes reservations/grants OUTSIDE the orchestrator lock).
    pub fn extract(&mut self, mut predicate: impl FnMut(&Entry) -> bool) -> Vec<Entry> {
        let keys: Vec<EntryKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| predicate(entry))
            .map(|(key, _)| *key)
            .collect();
        keys.iter()
            .filter_map(|key| self.entries.remove(key))
            .collect()
    }

    /// Find the allocation entry aggregating `(pid, quorum_id)`.
    pub fn find_allocation(
        &mut self,
        pid: ProcessId,
        quorum_id: ProcessId,
    ) -> Option<(&EntryKey, &mut Entry)> {
        self.entries.iter_mut().find_map(|(key, entry)| {
            (entry.pid == pid
                && matches!(&entry.kind, EntryKind::Allocation { quorum_id: q, .. } if *q == quorum_id))
            .then_some((key, entry))
        })
    }

    /// Move up to `count` accumulated pages from `from` to `to` — the head
    /// exercising its first claim over a younger entry's stranded partial
    /// accumulation. Pure bookkeeping under the caller's lock; returns the
    /// pages moved.
    pub fn transfer_accum(&mut self, from: &EntryKey, to: &EntryKey, count: usize) -> usize {
        let Some(donor) = self.entries.get_mut(from) else {
            return 0;
        };
        let donation = donor.kv_accum.donate(count);
        let moved = donation.len();
        if moved > 0 {
            let recipient = self
                .entries
                .get_mut(to)
                .expect("transfer target exists under the same lock");
            recipient.kv_accum.absorb(donation);
        }
        moved
    }

    /// The restore entry for `pid`, if queued.
    pub fn find_restore(&self, pid: ProcessId) -> Option<(&EntryKey, &Entry)> {
        self.entries
            .iter()
            .find(|(_, entry)| entry.pid == pid && matches!(entry.kind, EntryKind::Restore { .. }))
    }

    /// Counts for the packed summary word: (allocation entries, restore
    /// entries) — computed, never mirrored.
    pub fn counts(&self) -> (usize, usize) {
        let mut allocations = 0;
        let mut restores = 0;
        for entry in self.entries.values() {
            match entry.kind {
                EntryKind::Allocation { .. } => allocations += 1,
                EntryKind::Restore { .. } => restores += 1,
            }
        }
        (allocations, restores)
    }
}

/// Wake the acquire futures of displaced allocation entries so they observe
/// the removal (the caller then drops the entries OUTSIDE the lock —
/// forgetting the wake wedges a parked acquire).
pub(super) fn wake_displaced(entries: &[Entry]) {
    for entry in entries {
        if let EntryKind::Allocation { notify, .. } = &entry.kind {
            notify.notify_waiters();
        }
    }
}
