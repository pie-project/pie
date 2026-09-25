#![allow(dead_code)]

pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::{BTreeSet, HashMap};

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
}

pub struct RsStore {
    pool: Pool<RsSlotId>,
    refs: HashMap<RsSlotId, u32>,
    working_sets: GenMap<RsWsMarker, RsEntry>,
    seq: u64,
    outstanding: BTreeSet<u64>,
}

impl RsStore {
    pub fn new(capacity: u32) -> Self {
        Self {
            pool: Pool::new(capacity),
            refs: HashMap::new(),
            working_sets: GenMap::new(),
            seq: 0,
            outstanding: BTreeSet::new(),
        }
    }

    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    pub fn retire_idle(&mut self) {
        let completed = match self.outstanding.iter().next() {
            Some(&oldest) => oldest.saturating_sub(1),
            None => u64::MAX,
        };
        self.pool.retire_through(completed);
    }

    pub fn create_working_set(&mut self, geom: RsGeometry) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: None,
            buffer: Vec::new(),
            occupancy: Occupancy::EMPTY,
            buffer_head: 0,
            window_phase: false,
        })
    }

    pub fn fork(&mut self, ws: RsWorkingSetId) -> Result<RsWorkingSetId, RsError> {
        let (geom, folded, buffer, occupancy, buffer_head) = {
            let entry = self.entry(ws)?;
            (
                entry.geom,
                entry.folded,
                entry.buffer.clone(),
                entry.occupancy,
                entry.buffer_head,
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
        }))
    }

    pub fn release_working_set(&mut self, ws: RsWorkingSetId, epoch: u64) {
        let Some(entry) = self.working_sets.remove(ws) else {
            return;
        };
        if let Some(id) = entry.folded {
            self.decref(id, epoch);
        }
        for id in entry.buffer.into_iter().flatten() {
            self.decref(id, epoch);
        }
    }

    pub fn alloc_buffer(&mut self, ws: RsWorkingSetId, n: u32) -> Result<PageRange, RsError> {
        let entry = self.entry_mut(ws)?;
        let start = entry.buffer.len() as u32;
        entry.buffer.resize(entry.buffer.len() + n as usize, None);
        Ok(PageRange { start, len: n })
    }

    pub fn discard_buffered(&mut self, ws: RsWorkingSetId, count: u32) -> Result<(), RsError> {
        let entry = self.entry_mut(ws)?;
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

    pub fn free_buffer(
        &mut self,
        ws: RsWorkingSetId,
        indices: &[u32],
        epoch: u64,
    ) -> Result<(), RsError> {
        let entry = self.entry(ws)?;
        let size = entry.buffer.len() as u32;
        let mut remove = vec![false; entry.buffer.len()];
        for &index in indices {
            if index >= size {
                return Err(RsError::IndexOutOfRange { index, size });
            }
            if remove[index as usize] {
                return Err(RsError::DuplicateIndex { index });
            }
            remove[index as usize] = true;
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
            self.decref(id, epoch);
        }
        Ok(())
    }

    pub fn reorder_buffer(&mut self, ws: RsWorkingSetId, perm: &[u32]) -> Result<(), RsError> {
        let entry = self.entry_mut(ws)?;
        let size = entry.buffer.len();
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
        let old = entry.buffer.clone();
        for (i, &p) in perm.iter().enumerate() {
            entry.buffer[i] = old[p as usize];
        }
        Ok(())
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
        let buffers = (first..=last)
            .filter(|&index| match entry.buffer[index] {
                None => true,
                Some(id) => self.ref_count(id) > 1,
            })
            .count();
        Ok(state + buffers)
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
        let (folded, buffer_targets_src) = {
            let entry = self.entry(ws)?;
            let src: Vec<(u32, Option<RsSlotId>)> = match buffer_tokens {
                Some((start, len)) if len > 0 => {
                    let (first, last) = page_span(entry, start, len)?;
                    (first..=last)
                        .map(|index| (index as u32, entry.buffer[index]))
                        .collect()
                }
                _ => Vec::new(),
            };
            (entry.folded, src)
        };

        let state_needs_alloc = write_state
            && match folded {
                None => true,
                Some(id) => self.ref_count(id) > 1,
            };
        let buffer_needs_alloc = buffer_targets_src
            .iter()
            .filter(|(_, slot)| match slot {
                None => true,
                Some(id) => self.ref_count(*id) > 1,
            })
            .count();

        let need = usize::from(state_needs_alloc) + buffer_needs_alloc;
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

        let buffers = buffer_targets_src
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
        self.outstanding.insert(self.seq);
        Ok(RsPreparedWrite {
            fold_len_is_bound: false,
            ws,
            state,
            buffers,
            allocated,
            buffer_span: buffer_tokens
                .filter(|(_, len)| *len > 0)
                .map(|(start, len)| (start, len, buffer_intent)),
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
        let epoch = self.seq;
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
                self.advance_fold(ws, tokens, epoch);
            }
        }
    }

    fn publish_prevalidated(&mut self, prepared: RsPreparedWrite, folds: &mut RsPendingFolds) {
        let ws = prepared.ws;
        let epoch = self.seq;
        if let Some(state) = &prepared.state {
            let old = self.entry(ws).expect("batch prevalidated").folded;
            if old != Some(state.slot) {
                self.refs.insert(state.slot, 1);
                self.entry_mut(ws).expect("batch prevalidated").folded = Some(state.slot);
                if let Some(old) = old {
                    self.decref(old, epoch);
                }
            }
        }

        for target in &prepared.buffers {
            match *target {
                RsBufferTarget::Fresh { index, dst } => {
                    self.refs.insert(dst, 1);
                    self.entry_mut(ws).expect("batch prevalidated").buffer[index as usize] =
                        Some(dst);
                }
                RsBufferTarget::Cow { index, src, dst } => {
                    self.refs.insert(dst, 1);
                    self.entry_mut(ws).expect("batch prevalidated").buffer[index as usize] =
                        Some(dst);
                    self.decref(src, epoch);
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

    pub fn settle(&mut self, published: RsPublished) {
        for seq in published.seqs() {
            self.outstanding.remove(seq);
        }
        self.retire_idle();
    }

    pub fn cancel_prepared(&mut self, prepared: RsPreparedWrite) {
        self.pool
            .recycle_after_epoch(prepared.allocated, prepared.seq);
        self.outstanding.remove(&prepared.seq);
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

    fn advance_fold(&mut self, ws: RsWorkingSetId, tokens: u32, epoch: u64) {
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
            self.decref(id, epoch);
        }
    }

    pub fn geometry(&self, ws: RsWorkingSetId) -> Result<RsGeometry, RsError> {
        Ok(self.entry(ws)?.geom)
    }

    pub fn buffer_size(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer.len() as u32)
    }

    pub fn buffer_tokens(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        let entry = self.entry(ws)?;
        entry
            .occupancy
            .exact()
            .ok_or(RsError::BufferOccupancyIndeterminate {
                bound: entry.occupancy.bound(),
            })
    }

    pub fn buffer_tokens_bound(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.occupancy.bound())
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
                held.extend(entry.folded);
                held.extend(entry.buffer.iter().flatten());
            }
        }
        held.len()
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

    fn decref(&mut self, id: RsSlotId, epoch: u64) {
        let count = self.refs.entry(id).or_insert(1);
        *count -= 1;
        if *count == 0 {
            self.refs.remove(&id);
            let epoch = epoch.max(self.seq);
            self.pool.recycle_after_epoch(vec![id], epoch);
        }
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
