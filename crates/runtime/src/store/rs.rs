#![allow(dead_code)]

pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::store::genmap::{GenKey, GenMap};
use crate::store::pool::{Pool, PoolId};
use write::{
    RsBufferIntent, RsBufferTarget, RsPendingFold, RsPendingFolds, RsPreparedWrite, RsPublished,
    RsStateTarget,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsWsMarker {}
pub type RsWorkingSetId = GenKey<RsWsMarker>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RsSlotId(pub u32);

impl PoolId for RsSlotId {
    fn from_index(index: u32) -> Self {
        Self(index)
    }
    fn index(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsGeometry {
    pub state_size: u64,
    pub buffer_page_tokens: u32,
    pub fold_granularity: u32,
}

impl RsGeometry {
    fn normalized_granularity(&self) -> u32 {
        self.fold_granularity.max(1)
    }
}

/// The bounded half of a model whose state is a window: `tokens` is the
/// window its rows are read through, `page_size` the tokens a page holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingGeometry {
    pub tokens: u32,
    pub page_size: u32,
}

/// A window's state. Its `buffer` holds the windowed page of each kv page
/// index it was written for; `committed` is the position the window is
/// anchored at and `head` the position written up to, in tokens; `retired`
/// counts the leading pages, in sequence order, already given back;
/// `stated` is the size the guest's buffer calls declared, so they answer
/// as on a slot store.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RingState {
    retired: u32,
    committed: u32,
    head: u32,
    stated: u32,
}

/// What a fire moves a ring by once it is published: `fold` of the tokens
/// up to `commit_end` are committed, and the head lands at `head`. With a
/// device-decided length (no `fold`) the host commits nothing, but every
/// token before `settled`, the first the fire writes, is part of the
/// sequence it continues, so the ring catches up to it. `positioned` says
/// the host knows the fire's positions: only then do pages retire, through
/// the sequence order of its kv page indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RingAdvance {
    pub(crate) positioned: bool,
    pub(crate) settled: u32,
    pub(crate) fold: Option<u32>,
    pub(crate) commit_end: u32,
    pub(crate) head: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageRange {
    pub start: u32,
    pub len: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RsError {
    #[error("unknown rs working set")]
    UnknownWorkingSet,
    #[error("fold: tokens must be > 0")]
    FoldZero,
    #[error("fold: {tokens} tokens exceed buffered capacity {capacity}")]
    FoldExceedsBuffer { tokens: u32, capacity: u32 },
    #[error("fold: {tokens} tokens is not a positive multiple of fold granularity {granularity}")]
    FoldGranularity { tokens: u32, granularity: u32 },
    #[error("discard: {count} tokens exceed the {buffered} buffered")]
    DiscardExceedsBuffer { count: u32, buffered: u32 },
    #[error("rs working set: index {index} out of range (size {size})")]
    IndexOutOfRange { index: u32, size: u32 },
    #[error("rs working set: duplicate index {index}")]
    DuplicateIndex { index: u32 },
    #[error("rs batch contains the same working set more than once")]
    DuplicateWorkingSet,
    #[error(
        "the folded boundary is device-resident: at most {bound} buffered token(s) remain, but \
         the exact count is not host-known. Free the buffer to settle it before a fire that \
         must replay it"
    )]
    BufferOccupancyIndeterminate { bound: u32 },
    #[error("rs working set: permutation is not a bijection over 0..{size}")]
    BadPermutation { size: u32 },
    #[error(
        "rs working set: buffer token range [{start}, {start}+{len}) exceeds capacity {capacity}"
    )]
    BufferRangeOutOfRange { start: u32, len: u32, capacity: u32 },
    #[error("rs working set: buffered slot {index} read before it was written")]
    UnmaterializedRead { index: u32 },
    #[error("rs pool exhausted: requested {requested}, available {available}")]
    OutOfSlots { requested: usize, available: usize },
    #[error("rs slot grant mismatch: required {required}, granted {granted}")]
    GrantMismatch { required: usize, granted: usize },
    #[error("rs index key must be 1..={max} bytes", max = crate::store::kv::MAX_INDEX_KEY_BYTES)]
    BadIndexKey,
    #[error("rs working set is suspended to host")]
    Suspended,
}

pub const RS_TRANSLATION_UNMAPPED: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Occupancy {
    Exact(u32),
    AtMost(u32),
}

impl Occupancy {
    const EMPTY: Self = Occupancy::Exact(0);

    fn at_most(n: u32) -> Self {
        if n == 0 {
            Occupancy::Exact(0)
        } else {
            Occupancy::AtMost(n)
        }
    }

    fn exact(self) -> Option<u32> {
        match self {
            Occupancy::Exact(n) => Some(n),
            Occupancy::AtMost(_) => None,
        }
    }

    fn bound(self) -> u32 {
        match self {
            Occupancy::Exact(n) | Occupancy::AtMost(n) => n,
        }
    }

    fn map(self, f: impl FnOnce(u32) -> u32) -> Self {
        match self {
            Occupancy::Exact(n) => Occupancy::Exact(f(n)),
            Occupancy::AtMost(n) => Occupancy::at_most(f(n)),
        }
    }

    fn into_bound(self) -> Self {
        Occupancy::at_most(self.bound())
    }
}

struct RsEntry {
    geom: RsGeometry,
    folded: Option<RsSlotId>,
    buffer: Vec<Option<RsSlotId>>,
    occupancy: Occupancy,
    buffer_head: u32,
    window_phase: bool,
    ring: Option<RingState>,
}

/// A suspend or restore: `(working set, from, to)` slot moves, host slots
/// being ids past the device range.
#[derive(Debug)]
pub struct RsResidencyTxn {
    scope: Vec<RsWorkingSetId>,
    moves: Vec<(RsSlotId, RsSlotId)>,
    host_base: u32,
    ring: bool,
}

impl RsResidencyTxn {
    /// The engine's slot ids: each device slot and the host row it pairs with.
    pub fn copy_plan(&self) -> (Vec<u32>, Vec<u32>) {
        self.moves
            .iter()
            .map(|&(from, to)| {
                let (device, host) = if from.0 >= self.host_base {
                    (to, from)
                } else {
                    (from, to)
                };
                (device.0, host.0 - self.host_base)
            })
            .unzip()
    }

    pub fn slot_count(&self) -> usize {
        self.moves.len()
    }

    /// The moves are windowed kv pages, copied through the kv planes.
    pub fn is_ring(&self) -> bool {
        self.ring
    }
}

#[derive(Debug, Default)]
struct Touch {
    writes: Vec<RsSlotId>,
    reads: Vec<RsSlotId>,
}

pub struct RsStore {
    pool: Pool<RsSlotId>,
    host: Pool<RsSlotId>,
    host_base: u32,
    /// The window, when the model's state is one: the slots are then its
    /// pages, id 0 the engine's null page.
    ring: Option<RingGeometry>,
    refs: HashMap<RsSlotId, u32>,
    working_sets: GenMap<RsWsMarker, RsEntry>,
    seq: u64,
    /// Unsettled fires by seq, with the slots each writes and reads.
    inflight: BTreeMap<u64, Touch>,
    /// Index snapshots: hidden forks no process holds.
    indexes: HashMap<Vec<u8>, RsWorkingSetId>,
}

impl RsStore {
    pub fn new(capacity: u32) -> Self {
        Self::new_with_host(capacity, 0)
    }

    pub fn new_with_host(capacity: u32, host_capacity: u32) -> Self {
        Self::build(
            Pool::new(capacity),
            Pool::new_range(capacity, host_capacity),
            capacity,
            None,
        )
    }

    /// A store for a model whose state is a window of `geom`: `pages`
    /// ring ids (0 the null page every page behind a window reads) and
    /// `host_pages` host ones a suspended ring parks on.
    pub fn new_ring(geom: RingGeometry, pages: u32, host_pages: u32) -> Self {
        let host_base = pages.max(1);
        Self::build(
            Pool::new_range(1, pages.saturating_sub(1)),
            Pool::new_range(host_base, host_pages),
            host_base,
            Some(geom),
        )
    }

    fn build(
        pool: Pool<RsSlotId>,
        host: Pool<RsSlotId>,
        host_base: u32,
        ring: Option<RingGeometry>,
    ) -> Self {
        Self {
            pool,
            host,
            host_base,
            ring,
            refs: HashMap::new(),
            working_sets: GenMap::new(),
            seq: 0,
            inflight: BTreeMap::new(),
            indexes: HashMap::new(),
        }
    }

    pub fn is_ring(&self) -> bool {
        self.ring.is_some()
    }

    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    pub fn retire_idle(&mut self) {
        let completed = match self.inflight.keys().next() {
            Some(&oldest) => oldest.saturating_sub(1),
            None => u64::MAX,
        };
        self.pool.retire_through(completed);
    }

    /// Records slots the fire published under `seq` reads beyond those it
    /// writes or copies, so none is recycled while it still reads them.
    pub fn note_reads(&mut self, seq: u64, ids: impl IntoIterator<Item = RsSlotId>) {
        if let Some(touch) = self.inflight.get_mut(&seq) {
            touch.reads.extend(ids);
        }
    }

    fn busy_slots(&self) -> HashSet<RsSlotId> {
        self.inflight
            .values()
            .flat_map(|touch| touch.writes.iter().copied())
            .collect()
    }

    fn live_ring(&self, entry: &RsEntry) -> Option<RingState> {
        entry.ring.filter(|_| self.pool.capacity() > 0)
    }

    pub fn create_working_set(&mut self, geom: RsGeometry) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: None,
            buffer: Vec::new(),
            occupancy: Occupancy::EMPTY,
            buffer_head: 0,
            window_phase: false,
            ring: self.ring.map(|_| RingState::default()),
        })
    }

    pub fn fork(&mut self, ws: RsWorkingSetId) -> Result<RsWorkingSetId, RsError> {
        let (geom, folded, buffer, occupancy, buffer_head, ring) = {
            let entry = self.entry(ws)?;
            (
                entry.geom,
                entry.folded,
                entry.buffer.clone(),
                entry.occupancy,
                entry.buffer_head,
                entry.ring,
            )
        };
        if let Some(id) = folded {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        for id in buffer.iter().flatten() {
            *self.refs.entry(*id).or_insert(1) += 1;
        }
        Ok(self.working_sets.insert(RsEntry {
            geom,
            folded,
            buffer,
            occupancy,
            buffer_head,
            window_phase: false,
            ring,
        }))
    }

    /// Releases `ws`, returning how many slots it was the last holder of.
    pub fn release_working_set(&mut self, ws: RsWorkingSetId) -> usize {
        let Some(entry) = self.working_sets.remove(ws) else {
            return 0;
        };
        entry
            .folded
            .into_iter()
            .chain(entry.buffer.into_iter().flatten())
            .filter(|&id| self.decref(id))
            .count()
    }

    fn slots(entry: &RsEntry) -> impl Iterator<Item = RsSlotId> + '_ {
        entry
            .folded
            .into_iter()
            .chain(entry.buffer.iter().flatten().copied())
    }

    fn validate_index_key(key: &[u8]) -> Result<(), RsError> {
        if key.is_empty() || key.len() > crate::store::kv::MAX_INDEX_KEY_BYTES {
            return Err(RsError::BadIndexKey);
        }
        Ok(())
    }

    /// Indexes a snapshot of `ws` under `key`, returning the slots a replaced
    /// snapshot freed.
    pub fn update_index(&mut self, key: Vec<u8>, ws: RsWorkingSetId) -> Result<usize, RsError> {
        Self::validate_index_key(&key)?;
        if Self::slots(self.entry(ws)?).any(|id| self.is_host(id)) {
            return Err(RsError::Suspended);
        }
        let snapshot = self.fork(ws)?;
        Ok(match self.indexes.insert(key, snapshot) {
            Some(old) => self.release_working_set(old),
            None => 0,
        })
    }

    #[allow(
        clippy::wrong_self_convention,
        reason = "`from-index` is the WIT name, as on the kv store"
    )]
    pub fn from_index(&mut self, key: &[u8]) -> Result<Option<RsWorkingSetId>, RsError> {
        Self::validate_index_key(key)?;
        // A snapshot a fire still writes is not readable yet: it misses until
        // that fire settles, so publishing never waits on one.
        let settled = |snapshot: &RsWorkingSetId| {
            self.working_sets.get(*snapshot).is_some_and(|entry| {
                let busy = self.busy_slots();
                !Self::slots(entry).any(|id| busy.contains(&id))
            })
        };
        self.indexes
            .get(key)
            .copied()
            .filter(settled)
            .map(|snapshot| self.fork(snapshot))
            .transpose()
    }

    pub fn remove_index(&mut self, key: &[u8]) -> Result<(bool, usize), RsError> {
        Self::validate_index_key(key)?;
        Ok(match self.indexes.remove(key) {
            Some(snapshot) => (true, self.release_working_set(snapshot)),
            None => (false, 0),
        })
    }

    /// Drops the index snapshots `drop` picks, returning how many went and
    /// the slots they were the last holders of.
    fn drop_indexes(&mut self, drop: impl Fn(&Self, RsWorkingSetId) -> bool) -> (usize, usize) {
        let keys: Vec<Vec<u8>> = self
            .indexes
            .iter()
            .filter(|&(_, &snapshot)| drop(self, snapshot))
            .map(|(key, _)| key.clone())
            .collect();
        let mut freed = 0;
        for key in &keys {
            if let Some(snapshot) = self.indexes.remove(key) {
                freed += self.release_working_set(snapshot);
            }
        }
        (keys.len(), freed)
    }

    /// Drops the snapshots sharing a slot with `working_sets`, so their next
    /// write lands in place instead of copying to a slot the pool lacks: a
    /// cache entry never makes live work wait. Returns how many were dropped.
    pub fn yield_indexes(&mut self, working_sets: &[RsWorkingSetId]) -> usize {
        let sharing = self.sharing_indexes(&working_sets.iter().copied().collect());
        self.drop_indexes(|_, snapshot| sharing.contains(&snapshot))
            .0
    }

    /// The rs half of the planner's idle reclaim: drops every snapshot that
    /// holds a slot no live working set does, returning the slots freed.
    pub fn drop_unused_indexes(&mut self) -> usize {
        if self.indexes.is_empty() {
            return 0;
        }
        let snapshots: HashSet<RsWorkingSetId> = self.indexes.values().copied().collect();
        let live: HashSet<RsSlotId> = self
            .working_sets
            .iter()
            .filter(|(ws, _)| !snapshots.contains(ws))
            .flat_map(|(_, entry)| Self::slots(entry))
            .collect();
        let (_, freed) = self.drop_indexes(|store, snapshot| {
            store
                .working_sets
                .get(snapshot)
                .is_some_and(|entry| Self::slots(entry).any(|id| !live.contains(&id)))
        });
        self.retire_idle();
        freed
    }

    pub fn alloc_buffer(&mut self, ws: RsWorkingSetId, n: u32) -> Result<PageRange, RsError> {
        let entry = self.entry_mut(ws)?;
        // A ring's pages are taken by the fires that write them; the guest's
        // buffer is what it may write ahead of the commit, kept as a count.
        if let Some(ring) = entry.ring.as_mut() {
            let start = ring.stated;
            ring.stated = ring.stated.saturating_add(n);
            return Ok(PageRange { start, len: n });
        }
        let start = entry.buffer.len() as u32;
        entry.buffer.resize(entry.buffer.len() + n as usize, None);
        Ok(PageRange { start, len: n })
    }

    pub fn discard_buffered(&mut self, ws: RsWorkingSetId, count: u32) -> Result<(), RsError> {
        let entry = self.entry_mut(ws)?;
        if let Some(ring) = entry.ring.as_mut() {
            let buffered = ring.head.saturating_sub(ring.committed);
            if count > buffered {
                return Err(RsError::DiscardExceedsBuffer { count, buffered });
            }
            ring.head -= count;
            return Ok(());
        }
        if count > entry.occupancy.bound() {
            return Err(RsError::DiscardExceedsBuffer {
                count,
                buffered: entry.occupancy.bound(),
            });
        }
        entry.occupancy = entry.occupancy.map(|n| n - count);
        if entry.occupancy.bound() == 0 {
            entry.buffer_head = 0;
        }
        Ok(())
    }

    pub fn free_buffer(&mut self, ws: RsWorkingSetId, indices: &[u32]) -> Result<(), RsError> {
        let entry = self.entry(ws)?;
        let size = entry
            .ring
            .map_or(entry.buffer.len() as u32, |ring| ring.stated);
        let mut remove = vec![false; size as usize];
        for &index in indices {
            if index >= size {
                return Err(RsError::IndexOutOfRange { index, size });
            }
            if remove[index as usize] {
                return Err(RsError::DuplicateIndex { index });
            }
            remove[index as usize] = true;
        }
        if let Some(ring) = self.entry_mut(ws)?.ring.as_mut() {
            ring.stated = size - indices.len() as u32;
            return Ok(());
        }
        let old = std::mem::take(&mut self.entry_mut(ws)?.buffer);
        let mut kept = Vec::with_capacity(old.len() - indices.len());
        let mut dropped = Vec::new();
        for (index, slot) in old.into_iter().enumerate() {
            if remove[index] {
                if let Some(id) = slot {
                    dropped.push(id);
                }
            } else {
                kept.push(slot);
            }
        }
        self.entry_mut(ws)?.buffer = kept;
        {
            let entry = self.entry_mut(ws)?;
            if remove[0] || entry.buffer.is_empty() {
                entry.buffer_head = 0;
            }
            let capacity = (entry.buffer.len() as u32)
                .saturating_mul(entry.geom.buffer_page_tokens.max(1))
                .saturating_sub(entry.buffer_head);
            entry.occupancy = entry.occupancy.map(|n| n.min(capacity));
        }
        for id in dropped {
            self.decref(id);
        }
        Ok(())
    }

    pub fn reorder_buffer(&mut self, ws: RsWorkingSetId, perm: &[u32]) -> Result<(), RsError> {
        let entry = self.entry_mut(ws)?;
        let size = entry
            .ring
            .map_or(entry.buffer.len(), |ring| ring.stated as usize);
        if perm.len() != size {
            return Err(RsError::BadPermutation { size: size as u32 });
        }
        let mut seen = vec![false; size];
        for &p in perm {
            if (p as usize) >= size || seen[p as usize] {
                return Err(RsError::BadPermutation { size: size as u32 });
            }
            seen[p as usize] = true;
        }
        if entry.ring.is_some() {
            return Ok(());
        }
        let old = entry.buffer.clone();
        for (i, &p) in perm.iter().enumerate() {
            entry.buffer[i] = old[p as usize];
        }
        Ok(())
    }

    /// The ring indexes a fire writing kv page indexes `pages` of `entry`
    /// writes, with the page each holds; none where the engine keeps no
    /// windowed pool.
    fn ring_targets(
        &self,
        entry: &RsEntry,
        pages: &[u32],
    ) -> Result<Vec<(u32, Option<RsSlotId>)>, RsError> {
        if self.live_ring(entry).is_none() {
            return Ok(Vec::new());
        }
        let mut pages = pages.to_vec();
        pages.sort_unstable();
        pages.dedup();
        Ok(pages
            .into_iter()
            .map(|page| (page, entry.buffer.get(page as usize).copied().flatten()))
            .collect())
    }

    /// The slots writing `targets` takes: a fresh one for each the set holds
    /// none for, or shares.
    fn targets_demand(&self, targets: &[(u32, Option<RsSlotId>)]) -> usize {
        targets
            .iter()
            .filter(|(_, slot)| slot.is_none_or(|id| self.ref_count(id) > 1))
            .count()
    }

    /// The pages a fire writing kv page indexes `pages` of ring `ws` takes.
    pub fn ring_demand(&self, ws: RsWorkingSetId, pages: &[u32]) -> Result<usize, RsError> {
        Ok(self.targets_demand(&self.ring_targets(self.entry(ws)?, pages)?))
    }

    /// Prepares a fire that writes `pages` of ring `ws`; `ring_advance`
    /// moves the ring once the fire's translation has been read.
    pub fn prepare_ring(
        &mut self,
        ws: RsWorkingSetId,
        pages: &[u32],
        granted: Option<&mut Vec<RsSlotId>>,
    ) -> Result<RsPreparedWrite, RsError> {
        let targets = self.ring_targets(self.entry(ws)?, pages)?;
        self.prepare_targets(ws, false, None, targets, None, granted)
    }

    /// The ring page of each kv page index in `indexes`, 0 behind the ring
    /// or where none is written yet.
    pub fn ring_translation(
        &self,
        ws: RsWorkingSetId,
        indexes: impl IntoIterator<Item = u32>,
    ) -> Result<Vec<u32>, RsError> {
        let entry = self.entry(ws)?;
        if self.live_ring(entry).is_none() {
            return Ok(Vec::new());
        }
        Ok(indexes
            .into_iter()
            .map(|page| {
                entry
                    .buffer
                    .get(page as usize)
                    .copied()
                    .flatten()
                    .map_or(0, |id| id.0)
            })
            .collect())
    }

    /// A ring's `(committed, head)` positions.
    #[cfg(test)]
    pub fn ring_positions(&self, ws: RsWorkingSetId) -> Result<Option<(u32, u32)>, RsError> {
        Ok(self.entry(ws)?.ring.map(|ring| (ring.committed, ring.head)))
    }

    /// Moves ring `ws` by what its published fire commits: the pages of
    /// `order`, the kv page indexes in sequence order, wholly behind the
    /// committed position's window retire once no fire still reads them.
    pub fn ring_advance(&mut self, ws: RsWorkingSetId, advance: RingAdvance, order: &[u32]) {
        let Some(geom) = self.ring else {
            return;
        };
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let Some(ring) = entry.ring.as_mut() else {
            return;
        };
        if advance.fold.is_none() {
            ring.committed = ring.committed.max(advance.settled.min(ring.head));
        }
        if let Some(head) = advance.head {
            ring.head = head;
        }
        if let Some(fold) = advance.fold {
            let tail = advance.commit_end.saturating_sub(ring.committed);
            ring.committed = ring.committed.saturating_add(fold.min(tail));
        }
        let first = ring.committed.saturating_sub(geom.tokens) / geom.page_size.max(1);
        if !advance.positioned || first <= ring.retired {
            return;
        }
        let behind =
            &order[(ring.retired as usize).min(order.len())..(first as usize).min(order.len())];
        ring.retired = first;
        let dropped: Vec<RsSlotId> = behind
            .iter()
            .filter_map(|&page| entry.buffer.get_mut(page as usize).and_then(Option::take))
            .collect();
        for id in dropped {
            self.decref(id);
        }
    }

    pub fn resolve_buffer(
        &self,
        ws: RsWorkingSetId,
        start_token: u32,
        len_tokens: u32,
    ) -> Result<Vec<RsSlotId>, RsError> {
        if len_tokens == 0 {
            return Ok(Vec::new());
        }
        let entry = self.entry(ws)?;
        let (first, last) = page_span(entry, start_token, len_tokens)?;
        let mut ids = Vec::with_capacity(last - first + 1);
        for index in first..=last {
            match entry.buffer[index] {
                Some(id) => ids.push(id),
                None => {
                    return Err(RsError::UnmaterializedRead {
                        index: index as u32,
                    });
                }
            }
        }
        Ok(ids)
    }

    pub fn validate_fold(
        &self,
        ws: RsWorkingSetId,
        tokens: u32,
        buffer_tokens: Option<(u32, u32)>,
        intent: RsBufferIntent,
    ) -> Result<(), RsError> {
        let entry = self.entry(ws)?;
        if tokens == 0 {
            return Err(RsError::FoldZero);
        }
        let granularity = entry.geom.normalized_granularity();
        if granularity > 1 && !tokens.is_multiple_of(granularity) {
            return Err(RsError::FoldGranularity {
                tokens,
                granularity,
            });
        }
        let capacity = (entry.buffer.len() as u32)
            .saturating_mul(entry.geom.buffer_page_tokens)
            .saturating_sub(entry.buffer_head);
        if tokens > capacity {
            return Err(RsError::FoldExceedsBuffer { tokens, capacity });
        }
        let live = match (intent, buffer_tokens) {
            (RsBufferIntent::Write, Some((start, len))) => {
                entry.occupancy.bound().max(start.saturating_add(len))
            }
            _ => entry.occupancy.bound(),
        };
        if tokens > live {
            return Err(RsError::FoldExceedsBuffer {
                tokens,
                capacity: live,
            });
        }
        Ok(())
    }

    pub fn prepare_write(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        buffer_tokens: Option<(u32, u32)>,
    ) -> Result<RsPreparedWrite, RsError> {
        self.prepare(
            ws,
            write_state,
            None,
            buffer_tokens,
            RsBufferIntent::Write,
            None,
        )
    }

    pub fn prepare_write_reserved(
        &mut self,
        ws: RsWorkingSetId,
        granted: &mut Vec<RsSlotId>,
    ) -> Result<RsPreparedWrite, RsError> {
        self.prepare(ws, true, None, None, RsBufferIntent::Write, Some(granted))
    }

    pub fn prepare_general(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fold_tokens: Option<u32>,
        buffer_tokens: Option<(u32, u32)>,
        buffer_intent: RsBufferIntent,
    ) -> Result<RsPreparedWrite, RsError> {
        if let Some(tokens) = fold_tokens {
            self.validate_fold(ws, tokens, buffer_tokens, buffer_intent)?;
        }
        self.prepare(
            ws,
            write_state,
            fold_tokens,
            buffer_tokens,
            buffer_intent,
            None,
        )
    }

    pub fn prepare_reserved(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fold_tokens: Option<u32>,
        buffer_tokens: Option<(u32, u32)>,
        buffer_intent: RsBufferIntent,
        granted: &mut Vec<RsSlotId>,
    ) -> Result<RsPreparedWrite, RsError> {
        if let Some(tokens) = fold_tokens {
            self.validate_fold(ws, tokens, buffer_tokens, buffer_intent)?;
        }
        self.prepare(
            ws,
            write_state,
            fold_tokens,
            buffer_tokens,
            buffer_intent,
            Some(granted),
        )
    }

    pub fn write_state_demand(&self, ws: RsWorkingSetId) -> Result<usize, RsError> {
        Ok(match self.entry(ws)?.folded {
            None => 1,
            Some(id) if self.ref_count(id) > 1 => 1,
            Some(_) => 0,
        })
    }

    pub fn write_demand(
        &self,
        ws: RsWorkingSetId,
        write_state: bool,
        buffer_tokens: Option<(u32, u32)>,
    ) -> Result<usize, RsError> {
        let state = if write_state {
            self.write_state_demand(ws)?
        } else {
            0
        };
        let Some((start, len)) = buffer_tokens.filter(|(_, len)| *len > 0) else {
            return Ok(state);
        };
        let entry = self.entry(ws)?;
        let (first, last) = page_span(entry, start, len)?;
        let targets: Vec<_> = (first..=last)
            .map(|index| (index as u32, entry.buffer[index]))
            .collect();
        Ok(state + self.targets_demand(&targets))
    }

    pub fn prepare_fold(
        &mut self,
        ws: RsWorkingSetId,
        tokens: u32,
    ) -> Result<RsPreparedWrite, RsError> {
        self.validate_fold(ws, tokens, None, RsBufferIntent::Replay)?;
        self.prepare(ws, true, Some(tokens), None, RsBufferIntent::Replay, None)
    }

    fn prepare(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fold_tokens: Option<u32>,
        buffer_tokens: Option<(u32, u32)>,
        buffer_intent: RsBufferIntent,
        reserved: Option<&mut Vec<RsSlotId>>,
    ) -> Result<RsPreparedWrite, RsError> {
        let targets = match buffer_tokens {
            Some((start, len)) if len > 0 => {
                let entry = self.entry(ws)?;
                let (first, last) = page_span(entry, start, len)?;
                (first..=last)
                    .map(|index| (index as u32, entry.buffer[index]))
                    .collect()
            }
            _ => Vec::new(),
        };
        let span = buffer_tokens
            .filter(|(_, len)| *len > 0)
            .map(|(start, len)| (start, len, buffer_intent));
        self.prepare_targets(ws, write_state, fold_tokens, targets, span, reserved)
    }

    /// Prepares a write of the state (when `write_state`) and of the buffer
    /// `targets`, taking a fresh slot for each the set holds none for or
    /// shares, from `reserved` or the pool.
    fn prepare_targets(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fold_tokens: Option<u32>,
        buffer_targets_src: Vec<(u32, Option<RsSlotId>)>,
        buffer_span: Option<(u32, u32, RsBufferIntent)>,
        reserved: Option<&mut Vec<RsSlotId>>,
    ) -> Result<RsPreparedWrite, RsError> {
        let folded = {
            let entry = self.entry(ws)?;
            // Fires lease the kv fence the suspend raised and quiesced, so none reaches here.
            debug_assert!(
                !self.holds_host(entry),
                "an rs write on a suspended working set"
            );
            entry.folded
        };
        let state_needs_alloc = write_state && folded.is_none_or(|id| self.ref_count(id) > 1);
        let need = usize::from(state_needs_alloc) + self.targets_demand(&buffer_targets_src);
        let allocated = match reserved {
            Some(granted) => {
                if granted.len() < need {
                    return Err(RsError::GrantMismatch {
                        required: need,
                        granted: granted.len(),
                    });
                }
                granted.drain(..need).collect()
            }
            None => self.pool.try_alloc_n(need).ok_or(RsError::OutOfSlots {
                requested: need,
                available: self.pool.available(),
            })?,
        };
        let mut fresh_ids = allocated.iter().copied();

        let state = if write_state {
            Some(match folded {
                None => RsStateTarget {
                    slot: fresh_ids.next().expect("allocated for fresh state"),
                    reset: true,
                    copy_from: None,
                    fold_tokens,
                },
                Some(old) if self.ref_count(old) > 1 => RsStateTarget {
                    slot: fresh_ids.next().expect("allocated for cow state"),
                    reset: false,
                    copy_from: Some(old),
                    fold_tokens,
                },
                Some(old) => RsStateTarget {
                    slot: old,
                    reset: false,
                    copy_from: None,
                    fold_tokens,
                },
            })
        } else {
            None
        };

        let buffers: Vec<RsBufferTarget> = buffer_targets_src
            .into_iter()
            .map(|(index, slot)| match slot {
                None => RsBufferTarget::Fresh {
                    index,
                    dst: fresh_ids.next().expect("allocated covers materialize"),
                },
                Some(src) if self.ref_count(src) > 1 => RsBufferTarget::Cow {
                    index,
                    src,
                    dst: fresh_ids.next().expect("allocated covers cow"),
                },
                Some(src) => RsBufferTarget::InPlace { index, dst: src },
            })
            .collect();

        self.seq += 1;
        let touch = Touch {
            writes: state
                .iter()
                .map(|state| state.slot)
                .chain(buffers.iter().map(RsBufferTarget::dst))
                .collect(),
            reads: state
                .iter()
                .filter_map(|state| state.copy_from)
                .chain(buffers.iter().filter_map(RsBufferTarget::src))
                .collect(),
        };
        self.inflight.insert(self.seq, touch);
        Ok(RsPreparedWrite {
            fold_len_is_bound: false,
            ws,
            state,
            buffers,
            allocated,
            buffer_span,
            seq: self.seq,
        })
    }

    pub fn publish_prepared(&mut self, prepared: RsPreparedWrite) -> Result<RsPublished, RsError> {
        let (published, folds) = self.publish_batch(vec![prepared])?;
        self.commit_folds(folds);
        Ok(published)
    }

    pub fn publish_batch(
        &mut self,
        prepared: Vec<RsPreparedWrite>,
    ) -> Result<(RsPublished, RsPendingFolds), RsError> {
        let validation = (|| {
            let mut seen = Vec::with_capacity(prepared.len());
            for write in &prepared {
                self.entry(write.ws)?;
                if seen.contains(&write.ws) {
                    return Err(RsError::DuplicateWorkingSet);
                }
                seen.push(write.ws);
            }
            Ok(())
        })();
        if let Err(error) = validation {
            self.cancel_batch(prepared);
            return Err(error);
        }
        let seqs = prepared
            .iter()
            .map(RsPreparedWrite::seq)
            .collect::<Vec<_>>();
        let mut folds = RsPendingFolds::default();
        for write in prepared {
            self.publish_prevalidated(write, &mut folds);
        }
        Ok((RsPublished::new(seqs), folds))
    }

    pub fn commit_folds(&mut self, folds: RsPendingFolds) {
        for RsPendingFold {
            ws,
            tokens,
            len_is_bound: is_bound,
        } in folds.0
        {
            if is_bound {
                if let Ok(entry) = self.entry_mut(ws) {
                    entry.occupancy = entry.occupancy.into_bound();
                }
            } else {
                self.advance_fold(ws, tokens);
            }
        }
    }

    fn publish_prevalidated(&mut self, prepared: RsPreparedWrite, folds: &mut RsPendingFolds) {
        let ws = prepared.ws;
        if let Some(state) = &prepared.state {
            let old = self.entry(ws).expect("batch prevalidated").folded;
            if old != Some(state.slot) {
                self.refs.insert(state.slot, 1);
                self.entry_mut(ws).expect("batch prevalidated").folded = Some(state.slot);
                if let Some(old) = old {
                    self.decref(old);
                }
            }
        }

        for target in &prepared.buffers {
            match *target {
                RsBufferTarget::Fresh { index, dst } => {
                    self.refs.insert(dst, 1);
                    self.set_buffer_page(ws, index, dst);
                }
                RsBufferTarget::Cow { index, src, dst } => {
                    self.refs.insert(dst, 1);
                    self.set_buffer_page(ws, index, dst);
                    self.decref(src);
                }
                RsBufferTarget::InPlace { .. } => {}
            }
        }

        if let Some((start, len, RsBufferIntent::Write)) = prepared.buffer_span {
            let entry = self.entry_mut(ws).expect("batch prevalidated");
            entry.occupancy = entry.occupancy.map(|n| n.max(start.saturating_add(len)));
        }

        if let Some(tokens) = prepared.state.as_ref().and_then(|state| state.fold_tokens) {
            folds.0.push(RsPendingFold {
                ws,
                tokens,
                len_is_bound: prepared.fold_len_is_bound,
            });
        }
    }

    /// A ring grows as its sequence does; a slot buffer was sized by the guest.
    fn set_buffer_page(&mut self, ws: RsWorkingSetId, index: u32, dst: RsSlotId) {
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let index = index as usize;
        if entry.buffer.len() <= index {
            entry.buffer.resize(index + 1, None);
        }
        entry.buffer[index] = Some(dst);
    }

    pub fn settle(&mut self, published: RsPublished) {
        for seq in published.seqs() {
            self.inflight.remove(seq);
        }
        self.retire_idle();
    }

    pub fn cancel_prepared(&mut self, prepared: RsPreparedWrite) {
        self.pool
            .recycle_after_epoch(prepared.allocated, prepared.seq);
        self.inflight.remove(&prepared.seq);
        self.retire_idle();
    }

    pub fn cancel_batch(&mut self, prepared: Vec<RsPreparedWrite>) {
        for write in prepared {
            self.cancel_prepared(write);
        }
    }

    pub fn retire_through(&mut self, _epoch: u64) {
        self.retire_idle();
    }

    fn advance_fold(&mut self, ws: RsWorkingSetId, tokens: u32) {
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let page = entry.geom.buffer_page_tokens.max(1);
        let head = entry.buffer_head.saturating_add(tokens);
        let drop = ((head / page) as usize).min(entry.buffer.len());
        entry.buffer_head = head - (drop as u32) * page;
        entry.occupancy = entry.occupancy.map(|n| n.saturating_sub(tokens));
        if entry.occupancy.bound() == 0 {
            entry.buffer_head = 0;
        }
        let dropped: Vec<RsSlotId> = entry.buffer.drain(..drop).flatten().collect();
        let capacity = (entry.buffer.len() as u32)
            .saturating_mul(page)
            .saturating_sub(entry.buffer_head);
        entry.occupancy = entry.occupancy.map(|n| n.min(capacity));
        for id in dropped {
            self.decref(id);
        }
    }

    pub fn geometry(&self, ws: RsWorkingSetId) -> Result<RsGeometry, RsError> {
        Ok(self.entry(ws)?.geom)
    }

    pub fn buffer_size(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        let entry = self.entry(ws)?;
        Ok(entry
            .ring
            .map_or(entry.buffer.len() as u32, |ring| ring.stated))
    }

    pub fn buffer_tokens(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        let entry = self.entry(ws)?;
        if let Some(ring) = entry.ring {
            return Ok(ring.head.saturating_sub(ring.committed));
        }
        entry
            .occupancy
            .exact()
            .ok_or(RsError::BufferOccupancyIndeterminate {
                bound: entry.occupancy.bound(),
            })
    }

    pub fn buffer_tokens_bound(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        let entry = self.entry(ws)?;
        Ok(match entry.ring {
            Some(ring) => ring.head.saturating_sub(ring.committed),
            None => entry.occupancy.bound(),
        })
    }

    pub fn buffer_tokens_exact(&self, ws: RsWorkingSetId) -> bool {
        self.entry(ws)
            .map(|e| e.occupancy.exact().is_some())
            .unwrap_or(true)
    }

    pub fn buffer_head(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer_head)
    }

    pub fn buffer_translation(&self, ws: RsWorkingSetId) -> Result<Vec<u32>, RsError> {
        Ok(self
            .entry(ws)?
            .buffer
            .iter()
            .map(|slot| slot.map_or(RS_TRANSLATION_UNMAPPED, |id| id.0))
            .collect())
    }

    pub fn window_phase(&self, ws: RsWorkingSetId) -> Result<bool, RsError> {
        Ok(self.entry(ws)?.window_phase)
    }

    pub fn toggle_window_phase(&mut self, ws: RsWorkingSetId) {
        if let Ok(entry) = self.entry_mut(ws) {
            entry.window_phase = !entry.window_phase;
        }
    }

    pub fn folded_slot(&self, ws: RsWorkingSetId) -> Result<Option<RsSlotId>, RsError> {
        Ok(self.entry(ws)?.folded)
    }

    pub fn available_slots(&self) -> usize {
        self.pool.available()
    }

    /// The distinct slots these working sets hold, a slot shared by forks counted once.
    pub fn held_slots(&self, working_sets: impl IntoIterator<Item = RsWorkingSetId>) -> usize {
        let mut held = BTreeSet::new();
        for ws in working_sets {
            if let Some(entry) = self.working_sets.get(ws) {
                held.extend(Self::slots(entry).filter(|&id| !self.is_host(id)));
            }
        }
        held.len()
    }

    fn is_host(&self, id: RsSlotId) -> bool {
        id.0 >= self.host_base
    }

    fn holds_host(&self, entry: &RsEntry) -> bool {
        Self::slots(entry).any(|id| self.is_host(id))
    }

    /// Swaps `from` for `to` in every working set of `scope`, returning
    /// how many held it.
    fn swap_within(&mut self, scope: &[RsWorkingSetId], from: RsSlotId, to: RsSlotId) -> u32 {
        let mut held = 0;
        for &ws in scope {
            if let Some(entry) = self.working_sets.get_mut(ws) {
                for id in entry
                    .folded
                    .iter_mut()
                    .chain(entry.buffer.iter_mut().flatten())
                {
                    if *id == from {
                        *id = to;
                        held += 1;
                    }
                }
            }
        }
        held
    }

    fn held_within(&self, scope: &HashSet<RsWorkingSetId>) -> BTreeMap<RsSlotId, u32> {
        let mut held = BTreeMap::new();
        for entry in scope.iter().filter_map(|&ws| self.working_sets.get(ws)) {
            for id in Self::slots(entry) {
                *held.entry(id).or_insert(0) += 1;
            }
        }
        held
    }

    /// The index snapshots sharing a slot with `working_sets`.
    fn sharing_indexes(&self, working_sets: &HashSet<RsWorkingSetId>) -> HashSet<RsWorkingSetId> {
        let held = self.held_within(working_sets);
        self.indexes
            .values()
            .copied()
            .filter(|&snapshot| {
                self.working_sets
                    .get(snapshot)
                    .is_some_and(|entry| Self::slots(entry).any(|id| held.contains_key(&id)))
            })
            .collect()
    }

    /// The device slots of `working_sets` held by no working set outside
    /// them but index snapshots, which a suspend drops: an index is cache,
    /// and left holding a parked set's slot it would pin it on the device.
    fn private(&self, working_sets: &HashSet<RsWorkingSetId>) -> Vec<RsSlotId> {
        let mut scope = self.sharing_indexes(working_sets);
        scope.extend(working_sets.iter().copied());
        let shared = self.held_within(&scope);
        self.held_within(working_sets)
            .into_keys()
            .filter(|id| !self.is_host(*id) && self.ref_count(*id) == shared[id])
            .collect()
    }

    /// The slots a suspend frees once the evict drains the set's settled fires.
    pub fn suspendable_slots(&self, working_sets: &HashSet<RsWorkingSetId>) -> usize {
        self.private(working_sets).len()
    }

    fn swapped(&self, working_sets: &HashSet<RsWorkingSetId>) -> Vec<RsSlotId> {
        self.held_within(working_sets)
            .into_keys()
            .filter(|&id| self.is_host(id))
            .collect()
    }

    pub fn swapped_slots(&self, working_sets: &HashSet<RsWorkingSetId>) -> usize {
        self.swapped(working_sets).len()
    }

    pub fn host_available(&self) -> usize {
        self.host.available()
    }

    pub fn host_capacity(&self) -> u32 {
        self.host.capacity()
    }

    /// Parks the sets' private, settled slots in host rows, when rows for
    /// all of them are free; otherwise they stay on the device, and `None`
    /// says so.
    pub fn prepare_suspend(
        &mut self,
        working_sets: &HashSet<RsWorkingSetId>,
    ) -> Option<RsResidencyTxn> {
        let busy = self.busy_slots();
        let slots: Vec<RsSlotId> = self
            .private(working_sets)
            .into_iter()
            .filter(|id| !busy.contains(id))
            .collect();
        if slots.is_empty() {
            return None;
        }
        let host = self.host.try_alloc_n(slots.len())?;
        let moving: HashSet<RsSlotId> = slots.iter().copied().collect();
        self.drop_indexes(|store, snapshot| {
            store
                .working_sets
                .get(snapshot)
                .is_some_and(|entry| Self::slots(entry).any(|id| moving.contains(&id)))
        });
        Some(self.txn(working_sets, slots.into_iter().zip(host).collect()))
    }

    fn txn(
        &self,
        working_sets: &HashSet<RsWorkingSetId>,
        moves: Vec<(RsSlotId, RsSlotId)>,
    ) -> RsResidencyTxn {
        RsResidencyTxn {
            scope: working_sets.iter().copied().collect(),
            moves,
            host_base: self.host_base,
            ring: self.ring.is_some(),
        }
    }

    /// Swaps each copied slot for its host copy, returning the device slots freed.
    /// A slot shared outside the set or dropped while the copy ran stays as it is.
    pub fn commit_suspend(&mut self, txn: RsResidencyTxn) -> usize {
        let epoch = self.seq;
        let busy = self.busy_slots();
        let scope: HashSet<RsWorkingSetId> = txn.scope.iter().copied().collect();
        let held = self.held_within(&scope);
        let mut freed = 0;
        for (device, host) in txn.moves {
            // A set released while the copy ran holds nothing; what it still
            // holds is as private and settled as the prepare found it, since
            // the process is fenced.
            let within = held.get(&device).copied().unwrap_or(0);
            if within == 0 {
                self.host.release_reserved(vec![host]);
                continue;
            }
            debug_assert!(
                self.ref_count(device) == within && !busy.contains(&device),
                "an rs slot went shared or busy under its suspend"
            );
            self.swap_within(&txn.scope, device, host);
            if let Some(count) = self.refs.remove(&device) {
                self.refs.insert(host, count);
            }
            self.pool.recycle_after_epoch(vec![device], epoch);
            freed += 1;
        }
        self.retire_idle();
        freed
    }

    pub fn abort_suspend(&mut self, txn: RsResidencyTxn) {
        self.host
            .release_reserved(txn.moves.into_iter().map(|(_, host)| host).collect());
    }

    pub fn prepare_restore(
        &mut self,
        working_sets: &HashSet<RsWorkingSetId>,
        granted: &mut Vec<RsSlotId>,
    ) -> Result<RsResidencyTxn, RsError> {
        let swapped = self.swapped(working_sets);
        if granted.len() < swapped.len() {
            return Err(RsError::GrantMismatch {
                required: swapped.len(),
                granted: granted.len(),
            });
        }
        let devices = granted.drain(..swapped.len());
        Ok(self.txn(working_sets, swapped.into_iter().zip(devices).collect()))
    }

    /// Swaps each host slot back for its device copy, returning the slots restored.
    pub fn commit_restore(&mut self, txn: RsResidencyTxn) -> usize {
        let mut restored = 0;
        for (host, device) in txn.moves {
            if self.swap_within(&txn.scope, host, device) == 0 {
                self.pool.release_reserved(vec![device]);
                continue;
            }
            if let Some(count) = self.refs.remove(&host) {
                self.refs.insert(device, count);
            }
            self.host.release_reserved(vec![host]);
            restored += 1;
        }
        restored
    }

    pub fn abort_restore(&mut self, txn: RsResidencyTxn) {
        self.pool
            .release_reserved(txn.moves.into_iter().map(|(_, device)| device).collect());
    }

    pub fn capacity_slots(&self) -> u32 {
        self.pool.capacity()
    }

    pub fn reserve_slots(&mut self, count: usize) -> Option<Vec<RsSlotId>> {
        self.pool.try_alloc_n(count)
    }

    pub fn release_slot_reservation(&mut self, slots: Vec<RsSlotId>) {
        self.pool.release_reserved(slots);
    }

    fn entry(&self, ws: RsWorkingSetId) -> Result<&RsEntry, RsError> {
        self.working_sets.get(ws).ok_or(RsError::UnknownWorkingSet)
    }

    fn entry_mut(&mut self, ws: RsWorkingSetId) -> Result<&mut RsEntry, RsError> {
        self.working_sets
            .get_mut(ws)
            .ok_or(RsError::UnknownWorkingSet)
    }

    fn ref_count(&self, id: RsSlotId) -> u32 {
        self.refs.get(&id).copied().unwrap_or(1)
    }

    /// Drops a reference to `id`, recycling it once none remains, behind
    /// the newest unsettled fire that writes or reads it: no fire still
    /// reading a slot sees it reused, and none that never did holds it.
    fn decref(&mut self, id: RsSlotId) -> bool {
        let count = self.refs.entry(id).or_insert(1);
        *count -= 1;
        if *count != 0 {
            return false;
        }
        self.refs.remove(&id);
        if self.is_host(id) {
            self.host.release_reserved(vec![id]);
            return true;
        }
        let epoch = self
            .inflight
            .iter()
            .rev()
            .find(|(_, touch)| touch.writes.contains(&id) || touch.reads.contains(&id))
            .map_or_else(
                || {
                    self.inflight
                        .keys()
                        .next()
                        .map_or(self.seq, |&oldest| oldest - 1)
                },
                |(&seq, _)| seq,
            );
        self.pool.recycle_after_epoch(vec![id], epoch);
        true
    }
}

fn page_span(
    entry: &RsEntry,
    start_token: u32,
    len_tokens: u32,
) -> Result<(usize, usize), RsError> {
    let page = entry.geom.buffer_page_tokens.max(1);
    let capacity = (entry.buffer.len() as u32).saturating_mul(page);
    let start = entry.buffer_head.saturating_add(start_token);
    let end = start
        .checked_add(len_tokens)
        .filter(|&e| e <= capacity)
        .ok_or(RsError::BufferRangeOutOfRange {
            start: start_token,
            len: len_tokens,
            capacity,
        })?;
    debug_assert!(len_tokens > 0);
    let first = (start / page) as usize;
    let last = ((end - 1) / page) as usize;
    Ok((first, last))
}
