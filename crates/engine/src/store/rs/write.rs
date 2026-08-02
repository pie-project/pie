//! `RsPreparedWrite`: the per-fire prepared RS operation (kv_refact.md,
//! `store/rs/write.rs`). Same lifecycle discipline as `KvPreparedWrite`:
//! prepare classifies and allocates, `RsStore::publish_batch` folds the
//! result into the committed mapping in submission order, and
//! [`RsPublished`] is the receipt the fire holds until it settles.
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

use super::{RsSlotId, RsWorkingSetId};

/// The folded-slot write target for one fire (or explicit fold).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsStateTarget {
    /// Slot the driver writes the advanced folded state into.
    pub slot: RsSlotId,
    /// Freshly allocated slab: the driver must zero it first (`RS_FLAG_RESET`).
    pub reset: bool,
    /// Device copy to issue before the write when the folded slab was shared
    /// (first write after a fork copies it).
    pub copy_from: Option<RsSlotId>,
    /// Tokens to fold (`rs_fold_lens`) when this is an explicit fold; a
    /// committed fold advances the folded boundary by this many tokens.
    pub fold_tokens: Option<u32>,
}

/// What a prepared buffer span MEANS. Both intents name the same pages, but
/// they move tokens in opposite directions, so the occupancy each implies is
/// opposite too.
///
/// - `Write` — the fire scatters activations INTO the span, so the buffer now
///   holds at least `start + len` tokens.
/// - `Replay` — the fire gathers the span on its way into the folded state.
///   Those tokens LEAVE the buffer; counting them as written would re-add
///   exactly what `advance_fold` just subtracted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RsBufferIntent {
    Write,
    Replay,
}

/// One buffered-page write target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsBufferTarget {
    /// Reserved slot materialized by its first write: fresh page, no copy.
    Fresh { index: u32, dst: RsSlotId },
    /// Uniquely owned page: write in place.
    InPlace { index: u32, dst: RsSlotId },
    /// Shared page: fresh slot, copy the preserved content first.
    Cow {
        index: u32,
        src: RsSlotId,
        dst: RsSlotId,
    },
}

impl RsBufferTarget {
    pub fn dst(&self) -> RsSlotId {
        match *self {
            RsBufferTarget::Fresh { dst, .. }
            | RsBufferTarget::InPlace { dst, .. }
            | RsBufferTarget::Cow { dst, .. } => dst,
        }
    }
}

/// A prepared, not-yet-published RS write for one fire.
#[derive(Debug)]
pub struct RsPreparedWrite {
    pub(crate) ws: RsWorkingSetId,
    pub(crate) state: Option<RsStateTarget>,
    pub(crate) buffers: Vec<RsBufferTarget>,
    pub(crate) allocated: Vec<RsSlotId>,
    /// The `(start, len)` token span this write buffers, if any. Pages are the
    /// allocation unit but NOT the accounting unit: a reserved page is not a
    /// written token, and conflating the two made a freshly allocated (and
    /// genuinely empty) buffer read as a full page of occupancy. Publishing
    /// this span is what lets `RsStore::buffer_tokens` be exact.
    pub(crate) buffer_span: Option<(u32, u32, RsBufferIntent)>,
    /// Submission sequence stamped at prepare (see `KvPreparedWrite::seq`).
    pub(crate) seq: u64,
    /// The fold this write commits has a length the HOST never learned: the
    /// device computed it and only the driver resolved it. The state target
    /// still carries a `fold_tokens`, but it is an upper bound, so publishing
    /// it must not advance the boundary by that much or retire pages against
    /// it — see `store::rs::Occupancy`.
    pub(crate) fold_len_is_bound: bool,
}

impl RsPreparedWrite {
    pub fn working_set(&self) -> RsWorkingSetId {
        self.ws
    }

    /// Declare that this write's fold length is device-resident. The caller
    /// supplied the host's upper bound so the allocation and the CSR had a
    /// shape; the real length reaches only the driver.
    pub fn mark_fold_len_device(&mut self) {
        self.fold_len_is_bound = true;
    }

    pub fn fold_len_is_bound(&self) -> bool {
        self.fold_len_is_bound
    }

    /// Submission sequence for epoch retirement at finalize.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn state(&self) -> Option<&RsStateTarget> {
        self.state.as_ref()
    }

    pub fn buffer_targets(&self) -> &[RsBufferTarget] {
        &self.buffers
    }

    /// Buffer `(src, dst)` copies the driver must issue before the launch.
    pub fn buffer_copy_plan(&self) -> impl Iterator<Item = (RsSlotId, RsSlotId)> + '_ {
        self.buffers.iter().filter_map(|t| match *t {
            RsBufferTarget::Cow { src, dst, .. } => Some((src, dst)),
            _ => None,
        })
    }
}

/// The fold advances a `publish_batch` deferred, to be applied with
/// `RsStore::commit_folds` once the fire's wire arrays have been built
/// against the pre-fold buffer they describe.
#[derive(Debug, Default)]
#[must_use = "a deferred fold that is never committed leaves the boundary behind"]
pub struct RsPendingFolds(pub(crate) Vec<RsPendingFold>);

/// One deferred fold. `tokens` is exact unless `len_is_bound`, in which case
/// the length lives on the device and was never read back — the store may
/// then only narrow its bound on the live buffer, never state it.
#[derive(Debug, Clone, Copy)]
pub struct RsPendingFold {
    pub(crate) ws: super::RsWorkingSetId,
    pub(crate) tokens: u32,
    pub(crate) len_is_bound: bool,
}

/// The receipt for one fire's published RS rows. The mapping is already
/// authoritative when this exists; it only carries what settlement needs —
/// the submission sequences this fire holds open, so settling can release
/// exactly them from the store's outstanding set.
///
/// The individual sequences are kept rather than just the newest one because
/// retirement is bounded by the OLDEST sequence still outstanding across all
/// fires. Fires settle out of order, so a receipt that only remembered its
/// maximum could not say which sequences its settlement actually released,
/// and the watermark would have to fall back to "nothing is in flight".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsPublished {
    seqs: Vec<u64>,
}

impl RsPublished {
    pub(super) fn new(seqs: Vec<u64>) -> Self {
        Self { seqs }
    }

    /// Submission sequence of the newest row in the batch.
    pub fn seq(&self) -> u64 {
        self.seqs.iter().copied().max().unwrap_or(0)
    }

    /// Every submission sequence this receipt holds open.
    pub fn seqs(&self) -> &[u64] {
        &self.seqs
    }

    /// Prepared rows this receipt covers.
    pub fn rows(&self) -> usize {
        self.seqs.len()
    }
}
