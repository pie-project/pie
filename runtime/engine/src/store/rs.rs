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
//! Like `KvStore`, the mapping is published in guest SUBMISSION order, not
//! completion order: `prepare` classifies and allocates, `publish_batch`
//! folds the result into the committed mapping while still holding the store
//! lock the prepare ran under, and only pool retirement (`settle`) waits for
//! the device. Physical content arrives later on the same pipeline stream.
//!
//! That seam is what makes RS run-ahead safe. A successor fire prepared
//! before its predecessor completes still classifies against a mapping that
//! already contains the predecessor's decision, so it cannot RESET a slot
//! twice or re-CoW an already-privatized one — no host barrier required.
//! Correctness of the *contents* comes from stream order, exactly as it
//! already does for the CoW pre-launch copy.
//!
//! A failed fire is fail-stop, not rolled back (`KvStore` is identical):
//! its published mapping stays authoritative, the pipeline fails, and the
//! pre-failure state survives only through a fork taken before it — the same
//! rule this store already documents for a committed fold.
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

/// How many tokens the buffer holds — and whether the host knows that number
/// or merely bounds it.
///
/// Buffered occupancy is counted in tokens actually WRITTEN, never as
/// `buffer.len() * page`: reserving a page does not buffer a token, and
/// classifying fires against the page-granular bound made every freshly
/// allocated buffer look full and rejected the legal empty-buffer append.
///
/// The host loses the exact count when a fold commits whose LENGTH it never
/// learned — the device computed it and only the driver ever saw the value.
/// The fold absorbed somewhere between 1 and `n` tokens, so the live buffer is
/// somewhere in `0..=n`. Retaining `n` is safe for MEMORY (no live page is
/// dropped) but NOT for the READ PATH: a later fire that replayed the bound
/// would replay tokens the fold already absorbed, which is a double fold and
/// unrecoverable.
///
/// Both variants carry a `u32`, so this type buys no arithmetic. What it buys
/// is that *reading* the number forces the caller to say which guarantee it
/// needs: [`exact`](Self::exact) refuses when the answer is a bound,
/// [`bound`](Self::bound) never refuses and is only correct where an
/// over-count is safe. That distinction used to live in a sibling `bool` that
/// every reader had to remember to consult.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Occupancy {
    /// The buffer holds exactly this many tokens.
    Exact(u32),
    /// The buffer holds AT MOST this many; the true count is `1..=n`.
    ///
    /// Never zero — see [`Occupancy::at_most`].
    AtMost(u32),
}

impl Occupancy {
    const EMPTY: Self = Occupancy::Exact(0);

    /// An upper bound of `n`, normalized.
    ///
    /// A bound of zero pins the true count at exactly zero, so `AtMost(0)` is
    /// not a state the buffer can be in. Answering that here rather than at
    /// each call site is most of the reason this is a type: the old code asked
    /// "did this operation drive the bound to zero, and does that restore
    /// exactness?" separately in `discard_buffered`, `free_buffer` and
    /// `advance_fold`, and had to get it right three times.
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

    /// The upper bound, which is always known.
    ///
    /// Correct only where an over-count is safe — capacity and allocation.
    /// Anything that REPLAYS the buffer must go through [`exact`](Self::exact)
    /// and take the refusal, because replaying an over-count double-folds.
    fn bound(self) -> u32 {
        match self {
            Occupancy::Exact(n) | Occupancy::AtMost(n) => n,
        }
    }

    /// Apply `f` to the count, preserving exactness.
    ///
    /// Arithmetic on a bound yields a bound: subtracting a known quantity from
    /// an unknown one leaves it unknown. The exception is a result of zero,
    /// which [`at_most`](Self::at_most) collapses back to exact.
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
    ///
    /// `free_buffer` — the guest's explicit "I am done with this window" — is
    /// what restores exactness once a device-resident fold has taken it away.
    /// That is not a limitation bolted on: it is the shape `mtp-native-verify`
    /// already has (buffer a window, commit the accepted prefix, free the
    /// buffer, open the next window).
    occupancy: Occupancy,
    /// Where logical buffer token 0 physically sits, in tokens from the start
    /// of page 0. Always `< buffer_page_tokens`.
    ///
    /// A fold absorbs tokens off the FRONT of the buffer, but only WHOLE
    /// covered pages can be released — `fold_granularity` is 1 in production
    /// while a buffer page is the KV page size, so a fold routinely lands
    /// mid-page. The survivors keep their physical offsets, so logical token
    /// `k` lives at physical `buffer_head + k`. Without this, a partial fold
    /// silently re-aims every later read and write one fold earlier: the
    /// replay would re-scan tokens that are ALREADY inside the folded state,
    /// and the append would overwrite live ones.
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
    ///
    /// Ordered because retirement is bounded by the SMALLEST element: every
    /// sequence below it has completed on the device, so anything freed at or
    /// before that epoch can no longer be referenced.
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
    ///
    /// A slot is recycled with the epoch of the free, and `decref` pins that
    /// epoch to `self.seq` — the newest sequence ever issued — so no device
    /// operation referencing the slot can carry a sequence ABOVE its tag.
    /// Retiring through the newest completed sequence therefore hands out only
    /// slots nothing can still be reading or writing.
    ///
    /// "Newest completed" is the smallest outstanding sequence minus one, not
    /// the newest settled one: fires settle out of order, so a young fire
    /// finishing first says nothing about an older one still on the device.
    /// With nothing outstanding every epoch is releasable.
    ///
    /// The two halves of that invariant are what make this sound, and both are
    /// load-bearing, because the obvious half-measures are not. Gating on
    /// idleness alone (`outstanding.is_empty()`) is safe but starves: under
    /// sustained load the store is never idle, so freed slots accumulate
    /// unretired and the pool reads empty while every slot in it is
    /// releasable — on Qwen3.6-27B (24 slots, admission pinned to 16) that
    /// failed 8 of 32 requests with "every RS folded slot is held". Retiring
    /// on the free epoch alone is fast but lies: it treats the epoch as if it
    /// had completed merely because a newer write exists, and hands the slot
    /// out from under the device. Measured, that version won +31% on the
    /// pinned-admission path and then hung the uncapped one (32/32 completed
    /// before, a 300 s stall after), which is what aliasing live state looks
    /// like from outside. Waiting for the epoch to actually COMPLETE is what
    /// buys the throughput without the aliasing.
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

    /// Forget the last `count` buffered tokens: they never happened.
    ///
    /// The twin of a fold, on the other end of the buffer. `fold` moves the
    /// folded boundary F to the RIGHT and is irreversible; this moves the
    /// live end F+B to the LEFT and costs nothing, because the slots it
    /// releases are simply overwritten by the next append. Between them they
    /// are the two ways a buffer shrinks, and until now only one of them
    /// existed.
    ///
    /// That asymmetry is why a speculative decode needs two fires per window.
    /// A verify fire buffers `k+1` tokens, of which a prefix is accepted; the
    /// only way to get rid of the rejected tail was `free_buffer`, which
    /// empties the buffer WHOLESALE and therefore forces the accepted prefix
    /// to be folded away first — in its own fire, since the accepted length
    /// is not known until the verify has run. With the tail discardable, the
    /// next window's fire folds the previous window's prefix while writing
    /// its own tokens (§10.2.9's fold-behind shape) and the commit fire
    /// disappears.
    ///
    /// Capacity is untouched: this releases TOKENS, not pages. `free_buffer`
    /// remains the capacity operation, and the two now divide cleanly —
    /// content versus capacity.
    pub fn discard_buffered(&mut self, ws: RsWorkingSetId, count: u32) -> Result<(), RsError> {
        let entry = self.entry_mut(ws)?;
        if count > entry.occupancy.bound() {
            return Err(RsError::DiscardExceedsBuffer {
                count,
                buffered: entry.occupancy.bound(),
            });
        }
        // Legal even while the boundary is device-resident. The uncertainty
        // there is how many tokens a fold absorbed off the FRONT; discarding
        // from the TAIL shifts the bound down by exactly `count` and leaves
        // it a bound. It does not RESTORE exactness — only the guest saying
        // "I am done with this window" does that — except at zero, which
        // `Occupancy::at_most` collapses because a bound of zero pins the
        // true count.
        entry.occupancy = entry.occupancy.map(|n| n - count);
        // Nothing survives to hold in place, so the head may rebase and the
        // next append starts at physical 0 -- the same reasoning
        // `advance_fold` uses when a fold absorbs the whole buffer. Without
        // it the head creeps forward across windows and the reservation
        // grows to match.
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
        // Freeing pages discards the tokens they held. Which tokens is the
        // guest's business (`free_buffer` takes arbitrary indices), so clamp
        // to the surviving capacity: still exact for the cases that matter
        // (freeing everything empties the buffer) and never an under-count.
        {
            let entry = self.entry_mut(ws)?;
            // Removing page 0 rebases physical storage, so the head no longer
            // names anything. An empty buffer has no survivors to hold in
            // place either. Either way the next append starts at physical 0.
            if remove[0] || entry.buffer.is_empty() {
                entry.buffer_head = 0;
            }
            // Discarding pages is the guest saying which tokens it no longer
            // needs, which is exactly the statement an indeterminate boundary
            // was missing. Freeing everything leaves zero surviving capacity,
            // so the clamp below drives the count to zero and
            // `Occupancy::at_most` makes it exact again — the guest's "I am
            // done with this window" settling the boundary.
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

    /// `buffer_tokens` / `intent` are the prepare's own, and they decide what
    /// the fold is allowed to reach. Page capacity is NOT the answer: a
    /// two-page buffer holding three live tokens would happily accept a fold
    /// of six and gather three slab tokens that were never written, which is
    /// silent corruption of the recurrent state rather than a visible
    /// failure. The live extent is the buffer's occupancy for a REPLAY (a commit
    /// gathers only what is already buffered) and `start + len` for a WRITE
    /// (the fire's own new tokens are part of the extended space it folds
    /// through). Capacity is still checked, because the gather is physical.
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
        if granularity > 1 && tokens % granularity != 0 {
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
        // Occupancy may be only an UPPER BOUND (a device-resident fold length
        // was never read back). Bounding against it is then permissive but
        // still sound -- it can only admit a fold the exact fill would too.
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
    /// granularity before any driver dispatch. A committed fold advances the
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
    ///
    /// All working sets are validated before any mapping is changed. If a
    /// handle was released or the batch aliases one working set twice, every
    /// prepared target is cancelled and no row is adopted.
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

    /// Apply the fold advances a `publish_batch` deferred.
    ///
    /// The fold is deferred, and MUST be, because the wire arrays describe the
    /// buffer as this fire's own rows are laid out: extended row `j` is
    /// physical `buffer_head + j`. Advancing the boundary first would report a
    /// head — and a page list — from AFTER the fold to a fire whose rows were
    /// built before it, which is only invisible when the fold either moves
    /// nothing or empties the buffer (the two cases that existed before the
    /// interior boundary). An interior fold is neither.
    pub fn commit_folds(&mut self, folds: RsPendingFolds) {
        let epoch = self.seq;
        for RsPendingFold {
            ws,
            tokens,
            len_is_bound: is_bound,
        } in folds.0
        {
            if is_bound {
                // `tokens` is only an upper bound here. Advancing by it would
                // drop pages the fold may still need; advancing by less would
                // leave the host thinking tokens are live that the fold has
                // already absorbed, and the next fire would replay them — a
                // double fold. Neither is recoverable, so the boundary simply
                // stops being a host-side number: retain every page, downgrade
                // the occupancy to the bound it now is, and refuse to state an
                // exact count until `free_buffer` settles it.
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
        // Displaced slots are recycled against the current submission
        // sequence; `retire_idle` is what actually hands them back, and it
        // only fires while nothing is in flight.
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

        // Exact occupancy: the write covers tokens [start, start+len), so the
        // buffer now holds at least start+len. `max` (not `+=`) because a
        // rewrite of an already-buffered span must not double-count. A
        // `Replay` span is a gather, not a write, and is not counted at all.
        //
        // This runs BEFORE the fold. A fire may do both — buffer its new
        // tokens and fold a prefix of the resulting buffer in the same pass —
        // and the fold's arithmetic is over the buffer the write leaves
        // behind, not the one it found.
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
    /// allocatable. Retirement is gated on the global in-flight count rather
    /// than the epoch: no recycled slot is ever handed out while any prepared
    /// write is still outstanding. Note this is the coarse rule the KV store
    /// USED to share — `KvStore::settle` now tracks the outstanding sequence
    /// set and retires through the oldest, because waiting for global
    /// quiescence there cost a 4.5 ms per-completion supply drip (analysis.md
    /// 10.16-10.17). RS slots have not shown the same pressure.
    pub fn retire_through(&mut self, _epoch: u64) {
        self.retire_idle();
    }

    /// Advance the folded boundary after a committed fold: drop the head
    /// buffer pages fully covered by the folded prefix; a partial tail page
    /// stays buffered (the inferlet owns token<->slot bookkeeping).
    fn advance_fold(&mut self, ws: RsWorkingSetId, tokens: u32, epoch: u64) {
        let entry = self.entry_mut(ws).expect("batch prevalidated");
        let page = entry.geom.buffer_page_tokens.max(1);
        // The fold absorbed `tokens` buffered tokens into the folded prefix,
        // so they are no longer buffered. Only WHOLE covered pages can be
        // released; whatever the fold consumed of the next page is recorded in
        // `buffer_head` so the survivors keep their physical offsets. Dropping
        // pages rebases those offsets, hence `head - drop * page`.
        let head = entry.buffer_head.saturating_add(tokens);
        let drop = ((head / page) as usize).min(entry.buffer.len());
        entry.buffer_head = head - (drop as u32) * page;
        entry.occupancy = entry.occupancy.map(|n| n.saturating_sub(tokens));
        // A fold that absorbed the whole buffer leaves no survivor to hold in
        // place, so the head can rebase and the next append starts at 0.
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
    /// classified against.
    ///
    /// NOT `buffer_size() * buffer_page_tokens`: that is a page-granular upper
    /// bound, and reserving a page in order to buffer into it made the buffer
    /// look occupied before a single token had been written, so the legal
    /// "append onto an empty buffer" fire was refused as an append onto a full
    /// one.
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

    /// Physical offset of logical buffer token 0 within page 0. The driver
    /// needs it because a fold that lands mid-page leaves the survivors where
    /// they were rather than compacting them down.
    pub fn buffer_head(&self, ws: RsWorkingSetId) -> Result<u32, RsError> {
        Ok(self.entry(ws)?.buffer_head)
    }

    /// This working set's buffer-page translation: WorkingSet-relative buffer
    /// page index -> physical slot id, dense, in page order.
    ///
    /// The RS twin of `KvWorkingSet::translation`, and the reason the guest's
    /// `rs-geometry` channels can name pages at all: a guest never holds a
    /// physical slot id, so channel-resolved buffer geometry is meaningless
    /// until it is mapped through this. A page that is reserved but not yet
    /// materialized has no physical backing and lowers as
    /// [`RS_TRANSLATION_UNMAPPED`] -- naming one is a fire-geometry error,
    /// not readable garbage, because unmaterialized activations would fold
    /// silently into the state.
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

    pub fn committed_high_water_slots(&self) -> u32 {
        self.pool.highest_in_use_exclusive()
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
            // trusting the caller's. Sequences are monotonic, so every write
            // that could still reference this slot has a sequence at or below
            // `self.seq`, and nothing prepared later can reach it: the slot is
            // unreferenced and stays unallocatable until it retires. A caller
            // holding a stale epoch would otherwise tag the free BELOW a write
            // still in flight against it, which is the aliasing `retire_idle`
            // documents.
            let epoch = epoch.max(self.seq);
            self.pool.recycle_after_epoch(vec![id], epoch);
        }
    }
}

/// Inclusive page-index span covering the token range, validated against the
/// buffered capacity.
///
/// `start_token` is LOGICAL — an offset from the oldest unfolded token. The
/// span is resolved against physical page storage through `buffer_head`, so
/// callers never have to know a fold landed mid-page.
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
