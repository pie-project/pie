//! RS (recurrent-state) store: WorkingSets, folded state, buffered pages, a
//! typed static pool, and its own prepare/commit/abort protocol. Separate
//! from `KvPageTable` since RS state is a composite slot plus a dense
//! buffered-page array, refcounted internally rather than under the KV
//! no-refcount rule. Mapping publishes in guest submission order, like
//! `KvStore`: `prepare` classifies and allocates, `publish_batch` commits
//! under the same lock, and only `settle` waits for the device.

// Some methods here are not yet called by the live single-model fire path but
// are exercised by this module's own tests and reserved for upcoming
// increments.
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

/// Marker for RS WorkingSet ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RsWsMarker {}
pub type RsWorkingSetId = GenKey<RsWsMarker>;

/// One slot in the RS backing pool: a model-defined composite folded state or
/// one buffered RS page. Stable while live; the engine addresses its RS pool
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

/// Per-model RS geometry (from engine capabilities / `model.wit` caps).
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
    #[error("discard: {count} tokens exceed the {buffered} buffered")]
    DiscardExceedsBuffer { count: u32, buffered: u32 },
    #[error("rs working set: index {index} out of range (size {size})")]
    IndexOutOfRange { index: u32, size: u32 },
    #[error("rs working set: duplicate index {index}")]
    DuplicateIndex { index: u32 },
    #[error("rs batch contains the same working set more than once")]
    DuplicateWorkingSet,
    /// A fold committed with a device-resident length, so the host holds only
    /// an upper bound on the live buffer. Cleared by `free_buffer`.
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
    /// Pool exhaustion; the scheduler routes this through the contention
    /// ladder, like `KvStoreError::OutOfPages`.
    #[error("rs pool exhausted: requested {requested}, available {available}")]
    OutOfSlots { requested: usize, available: usize },
    /// A reserved-path prepare was handed fewer slots than it needs.
    #[error("rs slot grant mismatch: required {required}, granted {granted}")]
    GrantMismatch { required: usize, granted: usize },
}

/// A buffer page that is reserved but has no physical slot behind it yet.
/// Distinct from every real slot id because the pool is capacity-bounded far
/// below `u32::MAX`.
pub const RS_TRANSLATION_UNMAPPED: u32 = u32::MAX;

/// How many tokens the buffer holds, and whether the host knows that
/// exactly or only bounds it (counted in tokens actually written, not
/// `buffer.len() * page`). The host loses the exact count when a fold
/// commits whose length it never learned (device-computed, unread); `n` is
/// then only an upper bound, since replaying it as exact would double-fold.
/// So reading the count forces the caller to say which guarantee it needs:
/// [`exact`](Self::exact) refuses on a bound, [`bound`](Self::bound) never
/// refuses and is correct only where an over-count is safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Occupancy {
    /// The buffer holds exactly this many tokens.
    Exact(u32),
    /// The buffer holds at most this many; never zero (see
    /// [`Occupancy::at_most`]).
    AtMost(u32),
}

impl Occupancy {
    const EMPTY: Self = Occupancy::Exact(0);

    /// An upper bound of `n`, normalized: a bound of zero pins the true count
    /// at exactly zero, so `AtMost(0)` collapses to `Exact(0)`.
    fn at_most(n: u32) -> Self {
        if n == 0 {
            Occupancy::Exact(0)
        } else {
            Occupancy::AtMost(n)
        }
    }

    /// The count when the host knows it, `None` when it only bounds it.
    fn exact(self) -> Option<u32> {
        match self {
            Occupancy::Exact(n) => Some(n),
            Occupancy::AtMost(_) => None,
        }
    }

    /// The upper bound, always known. Correct only where an over-count is
    /// safe (capacity, allocation); a replay must go through
    /// [`exact`](Self::exact) and take the refusal instead.
    fn bound(self) -> u32 {
        match self {
            Occupancy::Exact(n) | Occupancy::AtMost(n) => n,
        }
    }

    /// Apply `f` to the count, preserving exactness (a result of zero
    /// re-collapses to exact via [`at_most`](Self::at_most)).
    fn map(self, f: impl FnOnce(u32) -> u32) -> Self {
        match self {
            Occupancy::Exact(n) => Occupancy::Exact(f(n)),
            Occupancy::AtMost(n) => Occupancy::at_most(f(n)),
        }
    }

    /// Downgrade to a bound: a fold committed whose length the host never
    /// learned.
    fn into_bound(self) -> Self {
        Occupancy::at_most(self.bound())
    }
}

struct RsEntry {
    geom: RsGeometry,
    /// Folded composite state; `None` until the first write/fold commits.
    folded: Option<RsSlotId>,
    /// Dense ordered buffered page slots. `None` = reserved, unmaterialized.
    buffer: Vec<Option<RsSlotId>>,
    /// Buffered tokens, and whether that count is exact — see [`Occupancy`].
    /// `free_buffer` (the guest's "I am done with this window") is what
    /// restores exactness after a device-resident fold takes it away.
    occupancy: Occupancy,
    /// Where logical buffer token 0 physically sits, in tokens from the
    /// start of page 0. Always `< buffer_page_tokens`. A fold can land
    /// mid-page (only whole covered pages release), so survivors keep
    /// their physical offsets: logical token `k` lives at physical
    /// `buffer_head + k`.
    buffer_head: u32,
}

/// The RS store: WorkingSets + the typed backing pool.
pub struct RsStore {
    pool: Pool<RsSlotId>,
    refs: HashMap<RsSlotId, u32>,
    working_sets: GenMap<RsWsMarker, RsEntry>,
    /// See `KvStore::seq`: submission sequence for epoch retirement.
    seq: u64,
    /// Submission sequences prepared but not yet settled or cancelled.
    /// Ordered: retirement is bounded by the smallest element, since every
    /// sequence below it has completed on the device.
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

    /// The epoch to tag frees with right now.
    pub fn current_epoch(&self) -> u64 {
        self.seq
    }

    /// Retire every free whose epoch is provably complete on the device.
    /// A slot is recycled with the epoch of its free, and `decref` pins
    /// that epoch to the newest sequence ever issued, so retiring through
    /// the newest completed sequence (smallest outstanding minus one)
    /// hands out only slots nothing can still be reading or writing.
    pub fn retire_idle(&mut self) {
        let completed = match self.outstanding.iter().next() {
            Some(&oldest) => oldest.saturating_sub(1),
            None => u64::MAX,
        };
        self.pool.retire_through(completed);
    }

    // ------------------------------------------------------------------
    // WorkingSet lifecycle
    // ------------------------------------------------------------------

    pub fn create_working_set(&mut self, geom: RsGeometry) -> RsWorkingSetId {
        self.working_sets.insert(RsEntry {
            geom,
            folded: None,
            buffer: Vec::new(),
            occupancy: Occupancy::EMPTY,
            buffer_head: 0,
        })
    }

    /// Fork: shares the folded slot and every materialized buffered slot by
    /// reference; the first write on a shared slot copies it.
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

    /// Forget the last `count` buffered tokens: they never happened. The
    /// twin of a fold (which moves the boundary right, irreversibly); this
    /// moves the live end left and costs nothing. Releases tokens, not
    /// pages — `free_buffer` remains the capacity operation.
    pub fn discard_buffered(&mut self, ws: RsWorkingSetId, count: u32) -> Result<(), RsError> {
        let entry = self.entry_mut(ws)?;
        if count > entry.occupancy.bound() {
            return Err(RsError::DiscardExceedsBuffer {
                count,
                buffered: entry.occupancy.bound(),
            });
        }
        // Legal even while the boundary is device-resident: discarding from
        // the tail shifts the bound down by `count` and leaves it a bound
        // (exact only where that hits zero).
        entry.occupancy = entry.occupancy.map(|n| n - count);
        // Nothing survives to hold in place, so the head rebases to 0 for the
        // next append (same reasoning as `advance_fold`).
        if entry.occupancy.bound() == 0 {
            entry.buffer_head = 0;
        }
        Ok(())
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
        // Freeing pages discards the tokens they held; clamp to surviving
        // capacity (still exact when freeing everything, never an
        // under-count otherwise).
        {
            let entry = self.entry_mut(ws)?;
            // Removing page 0, or emptying the buffer, rebases physical
            // storage: the next append starts at physical 0.
            if remove[0] || entry.buffer.is_empty() {
                entry.buffer_head = 0;
            }
            // Freeing everything drives the clamped count to zero, and
            // `Occupancy::at_most` collapses that back to exact — the
            // guest's "I am done with this window" settling the boundary.
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

    /// `buffer_tokens`/`intent` decide how far a fold may reach — page
    /// capacity alone isn't enough, since it would let a fold gather
    /// never-written slab tokens. Live extent is the buffer's occupancy
    /// for a replay, or `start + len` for a write (the fire's own new
    /// tokens extend the space it folds through); capacity is still
    /// checked, since the gather itself is physical.
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
        // Occupancy may be only an upper bound; bounding against it is then
        // permissive but sound — it can only admit what an exact fill would.
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
        self.prepare(
            ws,
            write_state,
            None,
            buffer_tokens,
            RsBufferIntent::Write,
            None,
        )
    }

    /// Prepare a folded-state write from caller-owned reserved slots,
    /// consuming exactly the required prefix of `granted` (lend semantics:
    /// failure consumes nothing, surplus stays caller-owned).
    pub fn prepare_write_reserved(
        &mut self,
        ws: RsWorkingSetId,
        granted: &mut Vec<RsSlotId>,
    ) -> Result<RsPreparedWrite, RsError> {
        self.prepare(ws, true, None, None, RsBufferIntent::Write, Some(granted))
    }

    /// The general prepare: any combination of a folded-state write, an
    /// explicit fold, and a buffered-page write, allocating from the pool.
    /// A fold is validated against the granularity and buffered capacity
    /// before anything is allocated.
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

    /// [`prepare_general`] from caller-owned reserved slots (the acquisition
    /// grant), consuming exactly the required prefix of `granted`.
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

    /// Phase-A demand: slots a folded-state write for `ws` would allocate
    /// (1 for a fresh or CoW folded target, 0 for an in-place write). Pure —
    /// no allocation, transaction, or refcount change.
    pub fn write_state_demand(&self, ws: RsWorkingSetId) -> Result<usize, RsError> {
        Ok(match self.entry(ws)?.folded {
            None => 1,
            Some(id) if self.ref_count(id) > 1 => 1,
            Some(_) => 0,
        })
    }

    /// Phase-A demand for a whole prepared write, folded target plus buffered
    /// pages: exactly what [`RsStore::prepare`] would allocate. A buffered
    /// page costs a slot when it is still reserved (first write materializes
    /// it) or shared after a fork (copy-on-write). Pure.
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

    /// Prepare an explicit `fold(tokens)`: validated against the fold
    /// granularity before any engine dispatch. A committed fold advances the
    /// folded boundary (dropping fully covered head buffer pages).
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

    /// Publish one guest-ordered prepared write into the committed mapping:
    /// adopt the folded slot, apply buffer repoints, advance the fold
    /// boundary, release displaced slots. Physical content arrives later on
    /// the same pipeline stream; only pool retirement waits for the device.
    pub fn publish_prepared(&mut self, prepared: RsPreparedWrite) -> Result<RsPublished, RsError> {
        let (published, folds) = self.publish_batch(vec![prepared])?;
        self.commit_folds(folds);
        Ok(published)
    }

    /// Atomically publish every recurrent-state row of one forward fire.
    /// All working sets are validated before any mapping is changed: if a
    /// handle was released or the batch aliases one working set twice,
    /// every prepared target is cancelled and no row is adopted.
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

    /// Apply the fold advances `publish_batch` deferred. Must stay
    /// deferred: wire arrays describe the buffer as this fire's own rows
    /// were laid out, so advancing the boundary first would report a
    /// post-fold head to a fire whose rows were built before it.
    pub fn commit_folds(&mut self, folds: RsPendingFolds) {
        let epoch = self.seq;
        for RsPendingFold {
            ws,
            tokens,
            len_is_bound: is_bound,
        } in folds.0
        {
            if is_bound {
                // `tokens` is only an upper bound: advancing by it could
                // drop pages still needed, advancing by less could
                // double-fold. Neither is recoverable, so retain every
                // page and downgrade to a bound until `free_buffer`
                // settles it.
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
        // Displaced slots are recycled against the current sequence;
        // `retire_idle` only hands them back once nothing is in flight.
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

        // Write covers [start, start+len), so the buffer now holds at least
        // that; `max` not `+=` so a rewrite doesn't double-count (a `Replay`
        // span is a gather and isn't counted). Runs before the fold, since a
        // fold's arithmetic is over the buffer the write leaves behind.
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

    /// Settle a published write after its fire resolves, successfully or not.
    /// The mapping is already authoritative (fail-stop, as in `KvStore`); all
    /// that remains is releasing the in-flight hold on pool retirement.
    pub fn settle(&mut self, published: RsPublished) {
        for seq in published.seqs() {
            self.outstanding.remove(seq);
        }
        self.retire_idle();
    }

    /// Roll back a prepare that never published — a lowering or submission
    /// failure between `prepare` and `publish_batch`. The committed mapping
    /// was never touched, so only the allocation is returned.
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

    /// Retire completion epochs `<= epoch`, making recycled slots
    /// allocatable. Gated on the global in-flight count rather than the
    /// epoch: no recycled slot is handed out while any prepared write is
    /// still outstanding.
    pub fn retire_through(&mut self, _epoch: u64) {
        self.retire_idle();
    }

    /// Advance the folded boundary after a committed fold: drop the head
    /// buffer pages fully covered by the folded prefix; a partial tail page
    /// stays buffered (the inferlet owns token<->slot bookkeeping).
    fn advance_fold(&mut self, ws: RsWorkingSetId, tokens: u32, epoch: u64) {
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let page = entry.geom.buffer_page_tokens.max(1);
        // Only whole covered pages release; the remainder is recorded in
        // `buffer_head` so survivors keep their physical offsets. Dropping
        // pages rebases those offsets, hence `head - drop * page`.
        let head = entry.buffer_head.saturating_add(tokens);
        let drop = ((head / page) as usize).min(entry.buffer.len());
        entry.buffer_head = head - (drop as u32) * page;
        entry.occupancy = entry.occupancy.map(|n| n.saturating_sub(tokens));
        // Fold absorbed the whole buffer: no survivor to hold in place, so
        // the head rebases and the next append starts at 0.
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

    // ------------------------------------------------------------------
    // Introspection
    // ------------------------------------------------------------------

    pub fn geometry(&self, ws: RsWorkingSetId) -> Result<RsGeometry, RsError> {
        Ok(self.entry(ws)?.geom)
    }

    pub fn buffer_size(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer.len() as u32)
    }

    /// Buffered tokens actually written — the exact `B` a fire must be
    /// classified against. Not `buffer_size() * buffer_page_tokens`: that
    /// page-granular bound would make a freshly reserved page look occupied
    /// before anything was written to it.
    pub fn buffer_tokens(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        let entry = self.entry(ws)?;
        entry
            .occupancy
            .exact()
            .ok_or(RsError::BufferOccupancyIndeterminate {
                bound: entry.occupancy.bound(),
            })
    }

    /// The upper bound on buffered tokens, which is always known. Use this
    /// only where an over-count is safe (capacity and allocation); anything
    /// that REPLAYS the buffer must go through `buffer_tokens` and take the
    /// refusal, because replaying an over-count double-folds.
    pub fn buffer_tokens_bound(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.occupancy.bound())
    }

    /// Whether this working set's folded boundary is known exactly.
    pub fn buffer_tokens_exact(&self, ws: RsWorkingSetId) -> bool {
        self.entry(ws)
            .map(|e| e.occupancy.exact().is_some())
            .unwrap_or(true)
    }

    /// Physical offset of logical buffer token 0 within page 0. The engine
    /// needs it because a fold that lands mid-page leaves the survivors where
    /// they were rather than compacting them down.
    pub fn buffer_head(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer_head)
    }

    /// This working set's buffer-page translation: WorkingSet-relative
    /// buffer page index -> physical slot id, dense, in page order. A
    /// reserved but unmaterialized page lowers as
    /// [`RS_TRANSLATION_UNMAPPED`] — a fire-geometry error rather than
    /// readable garbage.
    pub fn buffer_translation(&self, ws: RsWorkingSetId) -> Result<Vec<u32>, RsError> {
        Ok(self
            .entry(ws)?
            .buffer
            .iter()
            .map(|slot| slot.map_or(RS_TRANSLATION_UNMAPPED, |id| id.0))
            .collect())
    }

    pub fn folded_slot(&self, ws: RsWorkingSetId) -> Result<Option<RsSlotId>, RsError> {
        Ok(self.entry(ws)?.folded)
    }

    pub fn available_slots(&self) -> usize {
        self.pool.available()
    }

    pub fn capacity_slots(&self) -> u32 {
        self.pool.capacity()
    }

    /// Reserve concrete slot ids for one acquisition grant. The caller owns
    /// them until consumed by a reserved-path prepare or released.
    pub fn reserve_slots(&mut self, count: usize) -> Option<Vec<RsSlotId>> {
        self.pool.try_alloc_n(count)
    }

    /// Return unconsumed reserved slot ids to the pool.
    pub fn release_slot_reservation(&mut self, slots: Vec<RsSlotId>) {
        self.pool.release_reserved(slots);
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
            // Pin the tag to the newest sequence ever issued rather than
            // trusting the caller's: sequences are monotonic, so every write
            // that could still reference this slot has a sequence at or below
            // `self.seq`, and the slot stays unallocatable until it retires.
            let epoch = epoch.max(self.seq);
            self.pool.recycle_after_epoch(vec![id], epoch);
        }
    }
}

/// Inclusive page-index span covering the token range, validated against the
/// buffered capacity. `start_token` is logical (an offset from the oldest
/// unfolded token); resolved through `buffer_head` so callers never have to
/// know a fold landed mid-page.
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
