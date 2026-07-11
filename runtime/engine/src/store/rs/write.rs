//! `RsPreparedWrite`: the per-fire prepared RS operation (kv_refact.md,
//! `store/rs/write.rs`). Same lifecycle discipline as `KvPreparedWrite`:
//! prepare classifies and allocates, `RsStore` commits on driver success or
//! aborts on failure.
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

/// A prepared, not-yet-committed RS write for one fire.
#[derive(Debug)]
pub struct RsPreparedWrite {
    pub(crate) ws: RsWorkingSetId,
    pub(crate) state: Option<RsStateTarget>,
    pub(crate) buffers: Vec<RsBufferTarget>,
    pub(crate) allocated: Vec<RsSlotId>,
    /// Submission sequence stamped at prepare (see `KvPreparedWrite::seq`).
    pub(crate) seq: u64,
}

impl RsPreparedWrite {
    pub fn working_set(&self) -> RsWorkingSetId {
        self.ws
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
