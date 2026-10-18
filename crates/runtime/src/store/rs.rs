pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::inferlet::process::ProcessId;
use crate::rt::Instant;

use crate::store::genmap::{GenKey, GenMap};
use crate::store::kv::hash::Hash256;
use crate::store::pool::{Pool, PoolId};
use write::{
    RsBufferTarget, RsPendingFold, RsPendingFolds, RsPreparedWrite, RsPublished, RsSnapshot,
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
/// counts the leading pages, in sequence order, already given back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RingState {
    retired: u32,
    committed: u32,
    head: u32,
}

/// What a fire moves a ring by once it is published: `fold` of the tokens
/// up to `commit_end` are committed, and the head lands at `head`. With a
/// device-decided length (no `fold`) the host commits nothing, but every
/// token before `settled`, the first the fire writes, is part of the
/// sequence it continues, so the ring catches up to it. `positioned` says
/// the host knows the fire's positions: the pages behind the committed
/// position's window retire, through the sequence order of its kv page
/// indexes. A device-positioned fire retires behind `reach`, the first token
/// it may write, as no row's window reaches back past it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RingAdvance {
    pub(crate) positioned: bool,
    pub(crate) settled: u32,
    pub(crate) reach: u32,
    pub(crate) fold: Option<u32>,
    pub(crate) commit_end: u32,
    pub(crate) head: Option<u32>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RsError {
    #[error("unknown rs working set")]
    UnknownWorkingSet,
    #[error("a fire writes slot {slot}, which its state already holds (it holds {frozen})")]
    BelowFrozen { slot: u32, frozen: u32 },
    #[error("rs batch contains the same working set more than once")]
    DuplicateWorkingSet,
    #[error("rs pools exhausted: requested {requested}, available {available}")]
    OutOfSlots {
        requested: RsUnits,
        available: RsUnits,
    },
    #[error("rs grant mismatch: required {required}, granted {granted}")]
    GrantMismatch { required: RsUnits, granted: RsUnits },
}

pub const RS_TRANSLATION_UNMAPPED: u32 = u32::MAX;

/// A count in each recurrent pool: state slots and buffer pages.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RsUnits {
    pub slots: u32,
    pub pages: u32,
}

impl std::fmt::Display for RsUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} state slot(s) and {} buffer page(s)",
            self.slots, self.pages
        )
    }
}

impl RsUnits {
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.slots == 0 && self.pages == 0
    }

    #[must_use]
    pub fn covers(self, need: RsUnits) -> bool {
        self.slots >= need.slots && self.pages >= need.pages
    }

    #[must_use]
    pub fn add(self, other: RsUnits) -> RsUnits {
        RsUnits {
            slots: self.slots.saturating_add(other.slots),
            pages: self.pages.saturating_add(other.pages),
        }
    }

    #[must_use]
    pub fn sub(self, other: RsUnits) -> RsUnits {
        RsUnits {
            slots: self.slots.saturating_sub(other.slots),
            pages: self.pages.saturating_sub(other.pages),
        }
    }
}

#[derive(Debug, Default)]
pub struct RsGrant {
    pub slots: Vec<RsSlotId>,
    pub pages: Vec<RsSlotId>,
}

impl RsGrant {
    #[must_use]
    pub fn units(&self) -> RsUnits {
        RsUnits {
            slots: self.slots.len() as u32,
            pages: self.pages.len() as u32,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty() && self.pages.is_empty()
    }
}

/// In the engine's numbering: each kind from 0.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RsCopyPlan {
    pub slots: Vec<(u32, u32)>,
    pub pages: Vec<(u32, u32)>,
}

/// Token `t` of page `p` holds slot `base + p * page + t`.
#[derive(Clone)]
struct RsEntry {
    geom: RsGeometry,
    folded: Option<RsSlotId>,
    buffer: Vec<Option<RsSlotId>>,
    base: u32,
    frozen: u32,
    ring: Option<RingState>,
}

#[derive(Debug)]
pub struct RsResidencyTxn {
    scope: Vec<RsWorkingSetId>,
    moves: Vec<(RsSlotId, RsSlotId)>,
    plan: RsCopyPlan,
    ring: bool,
}

impl RsResidencyTxn {
    pub fn copy_plan(&self) -> &RsCopyPlan {
        &self.plan
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

/// One numbering covers four pools: device state slots from 0, then device
/// buffer pages, then host slots and pages. An engine numbers each kind from 0
/// ([`RsStore::engine_id`]).
pub struct RsStore {
    slots: Pool<RsSlotId>,
    pages: Pool<RsSlotId>,
    host_slots: Pool<RsSlotId>,
    host_pages: Pool<RsSlotId>,
    pages_at: u32,
    /// Page 0 is the engine's null page, and there are no slots.
    ring: Option<RingGeometry>,
    refs: HashMap<RsSlotId, u32>,
    working_sets: GenMap<RsWsMarker, RsEntry>,
    seq: u64,
    /// Unsettled fires by seq, with the slots each writes and reads.
    inflight: BTreeMap<u64, Touch>,
    /// Checkpoints: hidden forks no process holds.
    checkpoints: HashMap<Hash256, RsCheckpoint>,
}

#[derive(Debug, Clone, Copy)]
struct RsCheckpoint {
    snapshot: RsWorkingSetId,
    owner: Option<ProcessId>,
    last_use: Instant,
}

impl RsStore {
    pub fn new(slots: u32, pages: u32, host_slots: u32, host_pages: u32) -> Self {
        let host = slots + pages;
        Self::build(
            Pool::new(slots),
            Pool::new_range(slots, pages),
            Pool::new_range(host, host_slots),
            Pool::new_range(host + host_slots, host_pages),
            slots,
            None,
        )
    }

    /// A store for a model whose state is a window of `geom`: `pages`
    /// ring pages (0 the null page every page behind a window reads) and
    /// `host_pages` host ones a suspended ring parks on.
    pub fn new_ring(geom: RingGeometry, pages: u32, host_pages: u32) -> Self {
        let host = pages.max(1);
        Self::build(
            Pool::new(0),
            Pool::new_range(1, pages.saturating_sub(1)),
            Pool::new_range(host, 0),
            Pool::new_range(host, host_pages),
            0,
            Some(geom),
        )
    }

    fn build(
        slots: Pool<RsSlotId>,
        pages: Pool<RsSlotId>,
        host_slots: Pool<RsSlotId>,
        host_pages: Pool<RsSlotId>,
        pages_at: u32,
        ring: Option<RingGeometry>,
    ) -> Self {
        Self {
            slots,
            pages,
            host_slots,
            host_pages,
            pages_at,
            ring,
            refs: HashMap::new(),
            working_sets: GenMap::new(),
            seq: 0,
            inflight: BTreeMap::new(),
            checkpoints: HashMap::new(),
        }
    }

    /// The state is a window ring of kv pages, drawn from the buffer pages.
    pub fn is_ring(&self) -> bool {
        self.ring.is_some()
    }

    pub fn retire_idle(&mut self) {
        let completed = match self.inflight.keys().next() {
            Some(&oldest) => oldest.saturating_sub(1),
            None => u64::MAX,
        };
        self.slots.retire_through(completed);
        self.pages.retire_through(completed);
    }

    #[must_use]
    pub fn engine_id(&self, id: RsSlotId) -> u32 {
        if self.host_pages.contains(id) {
            id.0 - self.host_pages.base()
        } else if self.host_slots.contains(id) {
            id.0 - self.host_slots.base()
        } else if self.is_page(id) {
            id.0 - self.pages_at
        } else {
            id.0
        }
    }

    #[must_use]
    pub fn page_id(&self, page: u32) -> RsSlotId {
        RsSlotId(page + self.pages_at)
    }

    fn is_page(&self, id: RsSlotId) -> bool {
        (id.0 >= self.pages_at && id.0 < self.host_slots.base()) || self.host_pages.contains(id)
    }

    fn units(&self, ids: impl IntoIterator<Item = RsSlotId>) -> RsUnits {
        let mut units = RsUnits::default();
        for id in ids {
            if self.is_page(id) {
                units.pages += 1;
            } else {
                units.slots += 1;
            }
        }
        units
    }

    fn device_pool(&mut self, id: RsSlotId) -> &mut Pool<RsSlotId> {
        if self.is_page(id) {
            &mut self.pages
        } else {
            &mut self.slots
        }
    }

    fn host_pool(&mut self, id: RsSlotId) -> &mut Pool<RsSlotId> {
        if self.is_page(id) {
            &mut self.host_pages
        } else {
            &mut self.host_slots
        }
    }

    fn recycle(&mut self, ids: Vec<RsSlotId>, epoch: u64) {
        for id in ids {
            self.device_pool(id).recycle_after_epoch(vec![id], epoch);
        }
    }

    fn release_device(&mut self, ids: impl IntoIterator<Item = RsSlotId>) {
        for id in ids {
            self.device_pool(id).release_reserved(vec![id]);
        }
    }

    fn release_host(&mut self, ids: impl IntoIterator<Item = RsSlotId>) {
        for id in ids {
            self.host_pool(id).release_reserved(vec![id]);
        }
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
        entry.ring.filter(|_| self.pages.capacity() > 0)
    }

    pub fn create_working_set(&mut self, geom: RsGeometry) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: None,
            buffer: Vec::new(),
            base: 0,
            frozen: 0,
            ring: self.ring.map(|_| RingState::default()),
        })
    }

    pub fn fork(&mut self, ws: RsWorkingSetId) -> Result<RsWorkingSetId, RsError> {
        let entry = self.entry(ws)?.clone();
        for id in Self::slots(&entry) {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        Ok(self.working_sets.insert(entry))
    }

    /// Releases `ws`, returning how many slots it was the last holder of.
    pub fn release_working_set(&mut self, ws: RsWorkingSetId) -> usize {
        let Some(entry) = self.working_sets.remove(ws) else {
            return 0;
        };
        Self::slots(&entry).filter(|&id| self.decref(id)).count()
    }

    fn slots(entry: &RsEntry) -> impl Iterator<Item = RsSlotId> + '_ {
        entry
            .folded
            .into_iter()
            .chain(entry.buffer.iter().flatten().copied())
    }

    fn drop_checkpoints(&mut self, drop: impl Fn(&Self, RsWorkingSetId) -> bool) -> usize {
        let keys: Vec<Hash256> = self
            .checkpoints
            .iter()
            .filter(|&(_, entry)| drop(self, entry.snapshot))
            .map(|(&key, _)| key)
            .collect();
        self.drop_keys(&keys);
        keys.len()
    }

    fn drop_keys(&mut self, keys: &[Hash256]) -> usize {
        let mut freed = 0;
        for key in keys {
            if let Some(entry) = self.checkpoints.remove(key) {
                freed += self.release_working_set(entry.snapshot);
            }
        }
        freed
    }

    /// Drops the snapshots sharing a slot with `working_sets`, so their next
    /// write lands in place instead of copying to a slot the pool lacks: a
    /// cache entry never makes live work wait. Returns how many were dropped.
    pub fn yield_checkpoints(&mut self, working_sets: &[RsWorkingSetId]) -> usize {
        let sharing = self.sharing_checkpoints(&working_sets.iter().copied().collect());
        self.drop_checkpoints(|_, snapshot| sharing.contains(&snapshot))
    }

    /// The rs half of idle reclaim, in the kv half's order.
    pub fn reclaim_cache(&mut self, keys: impl crate::policy::retention::Keys) -> usize {
        if self.checkpoints.is_empty() {
            return 0;
        }
        let snapshots: HashSet<RsWorkingSetId> = self
            .checkpoints
            .values()
            .map(|entry| entry.snapshot)
            .collect();
        let live: HashSet<RsSlotId> = self
            .working_sets
            .iter()
            .filter(|(ws, _)| !snapshots.contains(ws))
            .flat_map(|(_, entry)| Self::slots(entry))
            .collect();
        let mut groups: HashMap<(Option<ProcessId>, i64), (Vec<Hash256>, Instant)> = HashMap::new();
        for (&key, entry) in &self.checkpoints {
            let Some(ws) = self.working_sets.get(entry.snapshot) else {
                continue;
            };
            if Self::slots(ws).any(|id| !live.contains(&id)) {
                let at = u64::from(ws.frozen).saturating_sub(1);
                let group = groups
                    .entry((entry.owner, keys.key_at(entry.owner, at)))
                    .or_insert_with(|| (Vec::new(), entry.last_use));
                group.0.push(key);
                group.1 = group.1.max(entry.last_use);
            }
        }
        let mut order: Vec<_> = groups
            .into_iter()
            .map(|((_, key), (keys, last_use))| ((key, last_use), keys))
            .collect();
        order.sort_unstable_by_key(|(order, _)| *order);
        let mut freed = 0;
        for (_, keys) in order {
            freed += self.drop_keys(&keys);
            if freed > 0 {
                break;
            }
        }
        self.retire_idle();
        freed
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn discard_cache(&mut self, owner: ProcessId) -> usize {
        let keys: Vec<Hash256> = self
            .checkpoints
            .iter()
            .filter(|(_, entry)| entry.owner == Some(owner))
            .map(|(&key, _)| key)
            .collect();
        let freed = self.drop_keys(&keys);
        self.retire_idle();
        freed
    }

    fn unsettled(&self, snapshot: RsWorkingSetId) -> bool {
        let busy = self.busy_slots();
        self.working_sets
            .get(snapshot)
            .is_none_or(|entry| Self::slots(entry).any(|id| busy.contains(&id)))
    }

    /// Not when a fire on its way still writes it, or a suspend took it to the
    /// host.
    pub fn checkpoint(
        &mut self,
        ws: RsWorkingSetId,
        key: Hash256,
        owner: Option<ProcessId>,
    ) -> Result<bool, RsError> {
        let entry = self.entry(ws)?;
        if entry.ring.is_some()
            || entry.folded.is_none()
            || self.holds_host(entry)
            || self.unsettled(ws)
        {
            return Ok(false);
        }
        let snapshot = self.fork(ws)?;
        self.keep(snapshot, key, owner);
        Ok(true)
    }

    /// Keeps `snapshot`, a hidden working set no process holds, as `key`.
    pub fn keep(&mut self, snapshot: RsWorkingSetId, key: Hash256, owner: Option<ProcessId>) {
        let kept = RsCheckpoint {
            snapshot,
            owner,
            last_use: Instant::now(),
        };
        if let Some(old) = self.checkpoints.insert(key, kept) {
            self.release_working_set(old.snapshot);
        }
    }

    pub fn has_checkpoint(&self, key: &Hash256) -> bool {
        self.checkpoints
            .get(key)
            .is_some_and(|entry| !self.unsettled(entry.snapshot))
    }

    /// Sharing its slots until a write copies them.
    pub fn adopt_checkpoint(&mut self, ws: RsWorkingSetId, key: &Hash256) -> Result<bool, RsError> {
        let fresh = {
            let entry = self.entry(ws)?;
            entry.folded.is_none() && entry.buffer.iter().all(Option::is_none)
        };
        let Some(snapshot) = self
            .checkpoints
            .get(key)
            .map(|entry| entry.snapshot)
            .filter(|&snapshot| fresh && !self.unsettled(snapshot))
        else {
            return Ok(false);
        };
        let from = self.entry(snapshot)?.clone();
        for id in Self::slots(&from) {
            *self.refs.entry(id).or_insert(1) += 1;
        }
        let entry = self.entry_mut(ws)?;
        entry.folded = from.folded;
        entry.buffer = from.buffer;
        entry.base = from.base;
        entry.frozen = from.frozen;
        if let Some(entry) = self.checkpoints.get_mut(key) {
            entry.last_use = Instant::now();
        }
        Ok(true)
    }

    pub fn forget_state(&mut self, ws: RsWorkingSetId) -> Result<(), RsError> {
        let (folded, buffer) = {
            let entry = self.entry_mut(ws)?;
            entry.base = 0;
            entry.frozen = 0;
            (entry.folded.take(), std::mem::take(&mut entry.buffer))
        };
        for id in folded.into_iter().chain(buffer.into_iter().flatten()) {
            self.decref(id);
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

    /// The pages writing `targets` takes: a fresh one for each the set holds
    /// none for, or shares.
    fn targets_demand(&self, targets: &[(u32, Option<RsSlotId>)]) -> u32 {
        targets
            .iter()
            .filter(|(_, slot)| slot.is_none_or(|id| self.ref_count(id) > 1))
            .count() as u32
    }

    /// The pages a fire writing kv page indexes `pages` of ring `ws` takes.
    pub fn ring_demand(&self, ws: RsWorkingSetId, pages: &[u32]) -> Result<RsUnits, RsError> {
        Ok(RsUnits {
            slots: 0,
            pages: self.targets_demand(&self.ring_targets(self.entry(ws)?, pages)?),
        })
    }

    /// Prepares a fire that writes `pages` of ring `ws`; `ring_advance`
    /// moves the ring once the fire's translation has been read.
    pub fn prepare_ring(
        &mut self,
        ws: RsWorkingSetId,
        pages: &[u32],
        granted: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        let targets = self.ring_targets(self.entry(ws)?, pages)?;
        self.prepare_targets(ws, false, false, None, targets, granted)
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
                    .map_or(0, |id| self.engine_id(id))
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
        let reach = if advance.positioned {
            ring.committed
        } else {
            advance.reach
        };
        let first = reach.saturating_sub(geom.tokens) / geom.page_size.max(1);
        if first <= ring.retired {
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

    /// To a fresh slot when `fresh`: other lanes of the fire read the one it
    /// holds.
    pub fn group_demand(
        &self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<RsUnits, RsError> {
        let entry = self.entry(ws)?;
        let state = write_state && (fresh || entry.folded.is_none_or(|id| self.ref_count(id) > 1));
        let targets = self.buffer_targets(entry, writes)?;
        Ok(RsUnits {
            slots: u32::from(state),
            pages: self.targets_demand(&targets),
        })
    }

    fn buffer_targets(
        &self,
        entry: &RsEntry,
        writes: Option<std::ops::Range<u32>>,
    ) -> Result<Vec<(u32, Option<RsSlotId>)>, RsError> {
        let Some(writes) = writes.filter(|writes| !writes.is_empty()) else {
            return Ok(Vec::new());
        };
        if writes.start < entry.frozen {
            return Err(RsError::BelowFrozen {
                slot: writes.start,
                frozen: entry.frozen,
            });
        }
        let page = entry.geom.buffer_page_tokens.max(1);
        let base = if entry.buffer.is_empty() {
            entry.frozen
        } else {
            entry.base
        };
        let first = (writes.start - base) / page;
        let last = (writes.end - 1 - base) / page;
        Ok((first..=last)
            .map(|index| (index, entry.buffer.get(index as usize).copied().flatten()))
            .collect())
    }

    pub fn prepare_group(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        fold: Option<u32>,
        writes: Option<std::ops::Range<u32>>,
        reserved: Option<&mut RsGrant>,
    ) -> Result<RsPreparedWrite, RsError> {
        if let Some(writes) = writes.as_ref().filter(|writes| !writes.is_empty()) {
            let entry = self.entry_mut(ws)?;
            if entry.buffer.is_empty() {
                entry.base = entry.frozen;
            }
            let page = entry.geom.buffer_page_tokens.max(1);
            let pages = writes.end.saturating_sub(entry.base).div_ceil(page) as usize;
            if entry.buffer.len() < pages {
                entry.buffer.resize(pages, None);
            }
        }
        let targets = self.buffer_targets(self.entry(ws)?, writes)?;
        self.prepare_targets(ws, write_state, fresh, fold, targets, reserved)
    }

    /// Prepares a write of the state (when `write_state`) and of the buffer
    /// `targets`, taking a fresh slot or page for each the set holds none
    /// for or shares, from `reserved` or the pools.
    fn prepare_targets(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fresh: bool,
        fold_tokens: Option<u32>,
        buffer_targets_src: Vec<(u32, Option<RsSlotId>)>,
        reserved: Option<&mut RsGrant>,
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
        let need = RsUnits {
            slots: u32::from(
                write_state && (fresh || folded.is_none_or(|id| self.ref_count(id) > 1)),
            ),
            pages: self.targets_demand(&buffer_targets_src),
        };
        let (slots, pages) = match reserved {
            Some(granted) => {
                if !granted.units().covers(need) {
                    return Err(RsError::GrantMismatch {
                        required: need,
                        granted: granted.units(),
                    });
                }
                (
                    granted.slots.drain(..need.slots as usize).collect(),
                    granted.pages.drain(..need.pages as usize).collect(),
                )
            }
            None => self.alloc(need).ok_or(RsError::OutOfSlots {
                requested: need,
                available: self.available(),
            })?,
        };
        let mut fresh_slots = slots.iter().copied();
        let mut fresh_ids = pages.iter().copied();
        let allocated: Vec<RsSlotId> = slots.iter().chain(&pages).copied().collect();

        let state = if write_state {
            Some(match folded {
                None => RsStateTarget {
                    slot: fresh_slots.next().expect("allocated for fresh state"),
                    reset: true,
                    copy_from: None,
                    fold_tokens,
                },
                Some(old) if fresh || self.ref_count(old) > 1 => RsStateTarget {
                    slot: fresh_slots.next().expect("allocated for cow state"),
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
            ws,
            state,
            buffers,
            allocated,
            seq: self.seq,
            snapshots: Vec::new(),
        })
    }

    /// A fresh slot for each of `rows`, where the fire `prepared` lands its
    /// working set's state as of that row; each becomes a hidden working set
    /// on publish ([`RsPublished::snapshots`]).
    pub fn prepare_snapshots(
        &mut self,
        prepared: &mut RsPreparedWrite,
        rows: &[u32],
        reserved: Option<&mut RsGrant>,
    ) -> Result<(), RsError> {
        if rows.is_empty() {
            return Ok(());
        }
        let need = RsUnits {
            slots: rows.len() as u32,
            pages: 0,
        };
        let slots: Vec<RsSlotId> = match reserved {
            Some(granted) if granted.slots.len() >= rows.len() => {
                granted.slots.drain(..rows.len()).collect()
            }
            Some(granted) => {
                return Err(RsError::GrantMismatch {
                    required: need,
                    granted: granted.units(),
                });
            }
            None => {
                self.alloc(need)
                    .ok_or(RsError::OutOfSlots {
                        requested: need,
                        available: self.available(),
                    })?
                    .0
            }
        };
        if let Some(touch) = self.inflight.get_mut(&prepared.seq) {
            touch.writes.extend(&slots);
        }
        prepared.allocated.extend(&slots);
        prepared.snapshots = rows.iter().copied().zip(slots).collect();
        Ok(())
    }

    #[cfg(test)]
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
        let mut snapshots = Vec::new();
        for write in prepared {
            self.publish_prevalidated(write, &mut folds, &mut snapshots);
        }
        Ok((RsPublished::new(seqs, snapshots), folds))
    }

    pub fn commit_folds(&mut self, folds: RsPendingFolds) {
        for RsPendingFold { ws, tokens } in folds.0 {
            self.fold(ws, tokens);
        }
    }

    fn publish_prevalidated(
        &mut self,
        prepared: RsPreparedWrite,
        folds: &mut RsPendingFolds,
        snapshots: &mut Vec<RsSnapshot>,
    ) {
        let ws = prepared.ws;
        let (geom, frozen) = {
            let entry = self.entry(ws).expect("batch prevalidated");
            (entry.geom, entry.frozen)
        };
        for &(row, slot) in &prepared.snapshots {
            self.refs.insert(slot, 1);
            let snapshot = self.working_sets.insert(RsEntry {
                geom,
                folded: Some(slot),
                buffer: Vec::new(),
                base: frozen + row,
                frozen: frozen + row,
                ring: None,
            });
            snapshots.push(RsSnapshot {
                ws,
                row,
                slot,
                snapshot,
            });
        }
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

        if let Some(tokens) = prepared.state.as_ref().and_then(|state| state.fold_tokens) {
            folds.0.push(RsPendingFold { ws, tokens });
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
        self.recycle(prepared.allocated, prepared.seq);
        self.inflight.remove(&prepared.seq);
        self.retire_idle();
    }

    pub fn cancel_batch(&mut self, prepared: Vec<RsPreparedWrite>) {
        for write in prepared {
            self.cancel_prepared(write);
        }
    }

    /// The buffer pages wholly below them go.
    pub fn fold(&mut self, ws: RsWorkingSetId, slots: u32) {
        let Ok(entry) = self.entry_mut(ws) else {
            return;
        };
        entry.frozen = entry.frozen.saturating_add(slots);
        let page = entry.geom.buffer_page_tokens.max(1);
        let below = (entry.frozen.saturating_sub(entry.base) / page) as usize;
        let drop = below.min(entry.buffer.len());
        entry.base += drop as u32 * page;
        let dropped: Vec<RsSlotId> = entry.buffer.drain(..drop).flatten().collect();
        if entry.buffer.is_empty() {
            entry.base = entry.frozen;
        }
        for id in dropped {
            self.decref(id);
        }
    }

    pub fn frozen(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.frozen)
    }

    pub fn base(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        let entry = self.entry(ws)?;
        Ok(if entry.buffer.is_empty() {
            entry.frozen
        } else {
            entry.base
        })
    }

    #[cfg(test)]
    pub fn buffer_size(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer.len() as u32)
    }

    pub fn buffer_translation(&self, ws: RsWorkingSetId) -> Result<Vec<u32>, RsError> {
        Ok(self
            .entry(ws)?
            .buffer
            .iter()
            .map(|slot| slot.map_or(RS_TRANSLATION_UNMAPPED, |id| self.engine_id(id)))
            .collect())
    }

    pub fn folded_slot(&self, ws: RsWorkingSetId) -> Result<Option<RsSlotId>, RsError> {
        Ok(self.entry(ws)?.folded)
    }

    pub fn available(&self) -> RsUnits {
        RsUnits {
            slots: self.slots.available() as u32,
            pages: self.pages.available() as u32,
        }
    }

    pub fn capacity(&self) -> RsUnits {
        RsUnits {
            slots: self.slots.capacity(),
            pages: self.pages.capacity(),
        }
    }

    /// One shared by forks counted once.
    pub fn held(&self, working_sets: impl IntoIterator<Item = RsWorkingSetId>) -> RsUnits {
        let mut held = BTreeSet::new();
        for ws in working_sets {
            if let Some(entry) = self.working_sets.get(ws) {
                held.extend(Self::slots(entry).filter(|&id| !self.is_host(id)));
            }
        }
        self.units(held)
    }

    fn is_host(&self, id: RsSlotId) -> bool {
        id.0 >= self.host_slots.base()
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

    fn sharing_checkpoints(
        &self,
        working_sets: &HashSet<RsWorkingSetId>,
    ) -> HashSet<RsWorkingSetId> {
        let held = self.held_within(working_sets);
        self.checkpoints
            .values()
            .map(|entry| entry.snapshot)
            .filter(|&snapshot| {
                self.working_sets
                    .get(snapshot)
                    .is_some_and(|entry| Self::slots(entry).any(|id| held.contains_key(&id)))
            })
            .collect()
    }

    /// The device slots of `working_sets` held by no working set outside
    /// them but checkpoints, which a suspend drops: a checkpoint is cache,
    /// and left holding a parked set's slot it would pin it on the device.
    fn private(&self, working_sets: &HashSet<RsWorkingSetId>) -> Vec<RsSlotId> {
        let mut scope = self.sharing_checkpoints(working_sets);
        scope.extend(working_sets.iter().copied());
        let shared = self.held_within(&scope);
        self.held_within(working_sets)
            .into_keys()
            .filter(|id| !self.is_host(*id) && self.ref_count(*id) == shared[id])
            .collect()
    }

    pub fn suspendable(&self, working_sets: &HashSet<RsWorkingSetId>) -> RsUnits {
        self.units(self.private(working_sets))
    }

    fn swapped(&self, working_sets: &HashSet<RsWorkingSetId>) -> Vec<RsSlotId> {
        self.held_within(working_sets)
            .into_keys()
            .filter(|&id| self.is_host(id))
            .collect()
    }

    pub fn swapped_units(&self, working_sets: &HashSet<RsWorkingSetId>) -> RsUnits {
        self.units(self.swapped(working_sets))
    }

    pub fn host_available(&self) -> RsUnits {
        RsUnits {
            slots: self.host_slots.available() as u32,
            pages: self.host_pages.available() as u32,
        }
    }

    pub fn host_capacity(&self) -> RsUnits {
        RsUnits {
            slots: self.host_slots.capacity(),
            pages: self.host_pages.capacity(),
        }
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
        let (pages, slots): (Vec<RsSlotId>, Vec<RsSlotId>) =
            slots.into_iter().partition(|&id| self.is_page(id));
        let host_slots = self.host_slots.try_alloc_n(slots.len())?;
        let Some(host_pages) = self.host_pages.try_alloc_n(pages.len()) else {
            self.host_slots.release_reserved(host_slots);
            return None;
        };
        let slots: Vec<RsSlotId> = slots.into_iter().chain(pages).collect();
        let host: Vec<RsSlotId> = host_slots.into_iter().chain(host_pages).collect();
        let moving: HashSet<RsSlotId> = slots.iter().copied().collect();
        self.drop_checkpoints(|store, snapshot| {
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
        let mut plan = RsCopyPlan::default();
        for &(from, to) in &moves {
            let (device, host) = if self.is_host(from) {
                (to, from)
            } else {
                (from, to)
            };
            let pair = (self.engine_id(device), self.engine_id(host));
            if self.is_page(device) {
                plan.pages.push(pair);
            } else {
                plan.slots.push(pair);
            }
        }
        RsResidencyTxn {
            scope: working_sets.iter().copied().collect(),
            moves,
            plan,
            ring: self.ring.is_some(),
        }
    }

    /// One shared outside the set or dropped while the copy ran stays as it is.
    pub fn commit_suspend(&mut self, txn: RsResidencyTxn) -> RsUnits {
        let epoch = self.seq;
        let busy = self.busy_slots();
        let scope: HashSet<RsWorkingSetId> = txn.scope.iter().copied().collect();
        let held = self.held_within(&scope);
        let mut freed = Vec::new();
        for (device, host) in txn.moves {
            // A set released while the copy ran holds nothing; what it still
            // holds is as private and settled as the prepare found it, since
            // the process is fenced.
            let within = held.get(&device).copied().unwrap_or(0);
            if within == 0 {
                self.release_host([host]);
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
            self.recycle(vec![device], epoch);
            freed.push(device);
        }
        self.retire_idle();
        self.units(freed)
    }

    pub fn abort_suspend(&mut self, txn: RsResidencyTxn) {
        self.release_host(txn.moves.into_iter().map(|(_, host)| host));
    }

    pub fn prepare_restore(
        &mut self,
        working_sets: &HashSet<RsWorkingSetId>,
        granted: &mut RsGrant,
    ) -> Result<RsResidencyTxn, RsError> {
        let swapped = self.swapped(working_sets);
        let need = self.units(swapped.iter().copied());
        if !granted.units().covers(need) {
            return Err(RsError::GrantMismatch {
                required: need,
                granted: granted.units(),
            });
        }
        let mut slots = granted.slots.drain(..need.slots as usize);
        let mut pages = granted.pages.drain(..need.pages as usize);
        let moves = swapped
            .into_iter()
            .map(|host| {
                let device = if self.is_page(host) {
                    pages.next()
                } else {
                    slots.next()
                };
                (host, device.expect("the grant covers what was swapped"))
            })
            .collect();
        drop((slots, pages));
        Ok(self.txn(working_sets, moves))
    }

    /// Swaps each host slot back for its device copy, returning the slots restored.
    pub fn commit_restore(&mut self, txn: RsResidencyTxn) -> usize {
        let mut restored = 0;
        for (host, device) in txn.moves {
            if self.swap_within(&txn.scope, host, device) == 0 {
                self.release_device([device]);
                continue;
            }
            if let Some(count) = self.refs.remove(&host) {
                self.refs.insert(device, count);
            }
            self.release_host([host]);
            restored += 1;
        }
        restored
    }

    pub fn abort_restore(&mut self, txn: RsResidencyTxn) {
        self.release_device(txn.moves.into_iter().map(|(_, device)| device));
    }

    fn alloc(&mut self, need: RsUnits) -> Option<(Vec<RsSlotId>, Vec<RsSlotId>)> {
        let slots = self.slots.try_alloc_n(need.slots as usize)?;
        let Some(pages) = self.pages.try_alloc_n(need.pages as usize) else {
            self.slots.release_reserved(slots);
            return None;
        };
        Some((slots, pages))
    }

    pub fn reserve(&mut self, need: RsUnits) -> Option<RsGrant> {
        self.alloc(need)
            .map(|(slots, pages)| RsGrant { slots, pages })
    }

    pub fn release_reservation(&mut self, grant: RsGrant) {
        self.slots.release_reserved(grant.slots);
        self.pages.release_reserved(grant.pages);
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
            self.release_host([id]);
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
        self.recycle(vec![id], epoch);
        true
    }
}
