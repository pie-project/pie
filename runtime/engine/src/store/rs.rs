//! RS (recurrent-state) store (kv_refact.md, `store/rs/`).
//!
//! Separately owns RS WorkingSets, folded state, buffered pages, the typed
//! static pool, CoW/reset/fold/promotion, and its own prepare/commit/abort
//! protocol. Deliberately does NOT reuse `KvPageTable`, trie structure, or KV
//! hash semantics: RS state is a model-defined composite slot plus a dense
//! ordered array of buffered page slots, shared across forks by slot
//! reference counts internal to this store (the no-refcount rule of
//! kv_refact.md invariant 8 is specific to the KV page metadata model).
//!
//! One `RsSlotId` space backs both folded states and buffered pages: the
//! driver's RS pool is a single id space (`rs_slot_ids` address it directly),
//! so splitting would collide when lowered to launch descriptors.
//!
//! Semantics carried over from the retired `working_set/rs.rs`:
//! - The folded slot is lazily allocated with `reset` on the first
//!   write/fold, copied on write while shared after a fork, and written in
//!   place when uniquely owned.
//! - Buffered slots are reserved logically (`alloc_buffer`) and materialized
//!   on first write; shared materialized slots copy-on-write.
//! - `fold(n)` is validated against the model fold granularity before any
//!   driver dispatch; a committed fold advances the folded boundary and drops
//!   the fully covered head buffer pages. No rollback across a committed
//!   fold: the pre-fold state survives only through a fork taken before it.
//!
//! Like `KvStore`, prepare computes targets without mutating the mapping and
//! commit applies them; at most one prepared write may be in flight per
//! WorkingSet (the sequencer's same-WorkingSet batching rule).
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

pub mod working_set;
pub mod write;

#[cfg(test)]
mod tests;

use std::collections::HashMap;

use crate::store::genmap::{GenKey, GenMap};
use crate::store::pool::{Pool, PoolId};
use write::{RsBufferTarget, RsPreparedWrite, RsStateTarget};

/// Marker for RS WorkingSet ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsWsMarker {}
pub type RsWorkingSetId = GenKey<RsWsMarker>;

/// One slot in the RS backing pool: a model-defined composite folded state or
/// one buffered RS page. Stable while live; the driver addresses its RS pool
/// by this id.
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

/// Per-model RS geometry (from driver capabilities / `model.wit` caps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsGeometry {
    /// Bytes of one folded recurrent-state object.
    pub state_size: u64,
    /// Tokens per buffered RS page.
    pub buffer_page_tokens: u32,
    /// Fold granularity in tokens (0 is normalized to 1).
    pub fold_granularity: u32,
}

impl RsGeometry {
    fn normalized_granularity(&self) -> u32 {
        self.fold_granularity.max(1)
    }
}

/// A contiguous half-open span of buffered page slots.
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
    #[error("rs working set: index {index} out of range (size {size})")]
    IndexOutOfRange { index: u32, size: u32 },
    #[error("rs working set: duplicate index {index}")]
    DuplicateIndex { index: u32 },
    #[error("rs batch contains the same working set more than once")]
    DuplicateWorkingSet,
    #[error("rs working set: permutation is not a bijection over 0..{size}")]
    BadPermutation { size: u32 },
    #[error(
        "rs working set: buffer token range [{start}, {start}+{len}) exceeds capacity {capacity}"
    )]
    BufferRangeOutOfRange { start: u32, len: u32, capacity: u32 },
    #[error("rs working set: buffered slot {index} read before it was written")]
    UnmaterializedRead { index: u32 },
    /// Pool exhaustion; the scheduler routes this through the contention
    /// ladder, like `KvStoreError::OutOfPages`.
    #[error("rs pool exhausted: requested {requested}, available {available}")]
    OutOfSlots { requested: usize, available: usize },
}

struct RsEntry {
    geom: RsGeometry,
    /// Folded composite state; `None` until the first write/fold commits.
    folded: Option<RsSlotId>,
    /// Dense ordered buffered page slots. `None` = reserved, unmaterialized.
    buffer: Vec<Option<RsSlotId>>,
}

/// The RS store: WorkingSets + the typed backing pool.
pub struct RsStore {
    pool: Pool<RsSlotId>,
    refs: HashMap<RsSlotId, u32>,
    working_sets: GenMap<RsWsMarker, RsEntry>,
    /// See `KvStore::seq`: submission sequence for epoch retirement.
    seq: u64,
    in_flight: u64,
}

impl RsStore {
    pub fn new(capacity: u32) -> Self {
        Self {
            pool: Pool::new(capacity),
            refs: HashMap::new(),
            working_sets: GenMap::new(),
            seq: 0,
            in_flight: 0,
        }
    }

    /// The epoch to tag frees with right now.
    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    /// Retire everything immediately when no prepared write is in flight.
    pub fn retire_idle(&mut self) {
        if self.in_flight == 0 {
            self.pool.retire_through(self.seq);
        }
    }

    // ------------------------------------------------------------------
    // WorkingSet lifecycle
    // ------------------------------------------------------------------

    pub fn create_working_set(&mut self, geom: RsGeometry) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: None,
            buffer: Vec::new(),
        })
    }

    /// Fork: shares the folded slot and every materialized buffered slot by
    /// reference; the first write on a shared slot copies it.
    pub fn fork(&mut self, ws: RsWorkingSetId) -> Result<RsWorkingSetId, RsError> {
        let (geom, folded, buffer) = {
            let entry = self.entry(ws)?;
            (entry.geom, entry.folded, entry.buffer.clone())
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

    // ------------------------------------------------------------------
    // Buffer structure (dense ordered array)
    // ------------------------------------------------------------------

    /// Append `n` reserved (unmaterialized) buffered page slots.
    pub fn alloc_buffer(&mut self, ws: RsWorkingSetId, n: u32) -> Result<PageRange, RsError> {
        let entry = self.entry_mut(ws)?;
        let start = entry.buffer.len() as u32;
        entry.buffer.resize(entry.buffer.len() + n as usize, None);
        Ok(PageRange { start, len: n })
    }

    /// Remove the buffered slots at `indices` and densely compact the array.
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
        for id in dropped {
            self.decref(id, epoch);
        }
        Ok(())
    }

    /// Reorder buffered slots by the full bijection `perm`: new slot `i`
    /// takes old slot `perm[i]`.
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

    /// Materialized buffered ids covering the token range, for an RS read.
    /// Reading a reserved (never-written) slot is an error.
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

    // ------------------------------------------------------------------
    // Fold validation
    // ------------------------------------------------------------------

    pub fn validate_fold(&self, ws: RsWorkingSetId, tokens: u32) -> Result<(), RsError> {
        let entry = self.entry(ws)?;
        if tokens == 0 {
            return Err(RsError::FoldZero);
        }
        let granularity = entry.geom.normalized_granularity();
        if granularity > 1 && tokens % granularity != 0 {
            return Err(RsError::FoldGranularity {
                tokens,
                granularity,
            });
        }
        let capacity = (entry.buffer.len() as u32).saturating_mul(entry.geom.buffer_page_tokens);
        if tokens > capacity {
            return Err(RsError::FoldExceedsBuffer { tokens, capacity });
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Prepare / commit / abort
    // ------------------------------------------------------------------

    /// Prepare an in-forward folded-state write (GDN / linear-attention
    /// `commit_len` path) and/or a buffered-page token-range write, without
    /// mutating the committed mapping.
    pub fn prepare_write(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        buffer_tokens: Option<(u32, u32)>,
    ) -> Result<RsPreparedWrite, RsError> {
        self.prepare(ws, write_state, None, buffer_tokens)
    }

    /// Prepare an explicit `fold(tokens)`: validated against the fold
    /// granularity before any driver dispatch. A committed fold advances the
    /// folded boundary (dropping fully covered head buffer pages).
    pub fn prepare_fold(
        &mut self,
        ws: RsWorkingSetId,
        tokens: u32,
    ) -> Result<RsPreparedWrite, RsError> {
        self.validate_fold(ws, tokens)?;
        self.prepare(ws, true, Some(tokens), None)
    }

    fn prepare(
        &mut self,
        ws: RsWorkingSetId,
        write_state: bool,
        fold_tokens: Option<u32>,
        buffer_tokens: Option<(u32, u32)>,
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

        // Classify before allocating so failures leak nothing.
        let state_needs_alloc = write_state
            && match folded {
                None => true,
                Some(id) => self.ref_count(id) > 1, // shared -> CoW
            };
        let buffer_needs_alloc = buffer_targets_src
            .iter()
            .filter(|(_, slot)| match slot {
                None => true,                        // materialize
                Some(id) => self.ref_count(*id) > 1, // CoW
            })
            .count();

        let need = usize::from(state_needs_alloc) + buffer_needs_alloc;
        let allocated = self.pool.try_alloc_n(need).ok_or(RsError::OutOfSlots {
            requested: need,
            available: self.pool.available(),
        })?;
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
        self.in_flight += 1;
        Ok(RsPreparedWrite {
            ws,
            state,
            buffers,
            allocated,
            seq: self.seq,
        })
    }

    /// Commit after driver success at `epoch`: adopt the folded slot, apply
    /// buffer repoints, advance the fold boundary, release displaced slots
    /// after the epoch retires.
    pub fn commit(&mut self, prepared: RsPreparedWrite, epoch: u64) -> Result<(), RsError> {
        self.commit_batch(vec![prepared], epoch)
    }

    /// Atomically commit every recurrent-state row of one forward fire.
    ///
    /// All working sets are validated before any mapping is changed. If a
    /// handle was released or the batch aliases one working set twice, every
    /// prepared target is aborted and no row is adopted.
    pub fn commit_batch(
        &mut self,
        prepared: Vec<RsPreparedWrite>,
        epoch: u64,
    ) -> Result<(), RsError> {
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
            self.abort_batch(prepared, epoch);
            return Err(error);
        }
        for write in prepared {
            self.commit_prevalidated(write, epoch);
        }
        Ok(())
    }

    fn commit_prevalidated(&mut self, prepared: RsPreparedWrite, epoch: u64) {
        let ws = prepared.ws;
        if let Some(state) = &prepared.state {
            let old = self.entry(ws).expect("batch prevalidated").folded;
            if old != Some(state.slot) {
                self.refs.insert(state.slot, 1);
                self.entry_mut(ws).expect("batch prevalidated").folded = Some(state.slot);
                if let Some(old) = old {
                    self.decref(old, epoch);
                }
            }
            if let Some(tokens) = state.fold_tokens {
                self.advance_fold(ws, tokens, epoch);
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
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    /// Driver failure/poison/dummy-run: release pending slots; the committed
    /// state stays authoritative.
    pub fn abort(&mut self, prepared: RsPreparedWrite, epoch: u64) {
        self.pool.recycle_after_epoch(prepared.allocated, epoch);
        self.in_flight = self.in_flight.saturating_sub(1);
    }

    pub fn abort_batch(&mut self, prepared: Vec<RsPreparedWrite>, epoch: u64) {
        for write in prepared {
            self.abort(write, epoch);
        }
    }

    pub fn retire_through(&mut self, epoch: u64) {
        self.pool.retire_through(epoch);
    }

    /// Advance the folded boundary after a committed fold: drop the head
    /// buffer pages fully covered by the folded prefix; a partial tail page
    /// stays buffered (the inferlet owns token<->slot bookkeeping).
    fn advance_fold(&mut self, ws: RsWorkingSetId, tokens: u32, epoch: u64) {
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let page = entry.geom.buffer_page_tokens.max(1);
        let drop = ((tokens / page) as usize).min(entry.buffer.len());
        let dropped: Vec<RsSlotId> = entry.buffer.drain(..drop).flatten().collect();
        for id in dropped {
            self.decref(id, epoch);
        }
    }

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    pub fn geometry(&self, ws: RsWorkingSetId) -> Result<RsGeometry, RsError> {
        Ok(self.entry(ws)?.geom)
    }

    pub fn buffer_size(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer.len() as u32)
    }

    pub fn folded_slot(&self, ws: RsWorkingSetId) -> Result<Option<RsSlotId>, RsError> {
        Ok(self.entry(ws)?.folded)
    }

    pub fn available_slots(&self) -> usize {
        self.pool.available()
    }

    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

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
            self.pool.recycle_after_epoch(vec![id], epoch);
        }
    }
}

/// Inclusive page-index span covering the token range, validated against the
/// buffered capacity.
fn page_span(
    entry: &RsEntry,
    start_token: u32,
    len_tokens: u32,
) -> Result<(usize, usize), RsError> {
    let page = entry.geom.buffer_page_tokens.max(1);
    let capacity = (entry.buffer.len() as u32).saturating_mul(page);
    let end = start_token
        .checked_add(len_tokens)
        .filter(|&e| e <= capacity)
        .ok_or(RsError::BufferRangeOutOfRange {
            start: start_token,
            len: len_tokens,
            capacity,
        })?;
    debug_assert!(len_tokens > 0);
    let first = (start_token / page) as usize;
    let last = ((end - 1) / page) as usize;
    Ok((first, last))
}
