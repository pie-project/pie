//! `KvPreparedWrite`: the per-fire prepared KV operation (kv_refact.md,
//! `store/kv/write.rs`).
//!
//! Holds freshly allocated physical ids, the CoW copy plan, and pending
//! mapping/hash deltas, retained until async driver completion confirms the
//! epoch. This is not a transaction manager: `KvStore` commits it on success
//! and aborts/releases it on failure. `PendingFire` owns
//! `Option<KvPreparedWrite>`.
//!
//! Complete typed-store API (kv_refact.md): some methods here are not yet
//! called by the live single-model fire path (only a subset of the typed
//! store surface is currently wired) but are exercised by this module's
//! own unit test suite and reserved for upcoming increments (contention/
//! reclaim expansion, RS buffer-write paths, etc.) — kept rather than
//! deleted, allowed rather than silently masked.
#![allow(dead_code)]

use super::hash::Hash256;
use super::page_table::{NodeId, PhysicalKvPageId, WorkingSetId};

/// One write target, classified by the CoW rules ("Every PTIR KV output is a
/// write intent"):
/// - fresh reserved slot -> fresh backing, no copy;
/// - private, unobserved owned page -> write in place;
/// - shared or retained page -> fresh slot, copy the preserved cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedTarget {
    Fresh {
        index: u64,
        dst: PhysicalKvPageId,
    },
    InPlace {
        index: u64,
        dst: PhysicalKvPageId,
    },
    Cow {
        index: u64,
        src: PhysicalKvPageId,
        dst: PhysicalKvPageId,
    },
}

impl PreparedTarget {
    pub fn index(&self) -> u64 {
        match *self {
            PreparedTarget::Fresh { index, .. }
            | PreparedTarget::InPlace { index, .. }
            | PreparedTarget::Cow { index, .. } => index,
        }
    }

    /// The physical page the driver writes for this target.
    pub fn dst(&self) -> PhysicalKvPageId {
        match *self {
            PreparedTarget::Fresh { dst, .. }
            | PreparedTarget::InPlace { dst, .. }
            | PreparedTarget::Cow { dst, .. } => dst,
        }
    }
}

/// A prepared, not-yet-committed KV write for one fire.
#[derive(Debug)]
pub struct KvPreparedWrite {
    pub(crate) ws: WorkingSetId,
    /// Terminal snapshot pin taken at prepare; released at commit/abort.
    pub(crate) pinned: Option<NodeId>,
    /// Ordered: in-place targets, then the CoW region ascending, then fresh
    /// appends ascending.
    pub(crate) targets: Vec<PreparedTarget>,
    pub(crate) allocated: Vec<PhysicalKvPageId>,
    pub(crate) old_mapped: u64,
    /// Start of the rebased tail region, when any committed page is CoW'd.
    pub(crate) cow_start: Option<u64>,
    /// Submission sequence stamped at prepare; fires complete in FIFO order,
    /// so the finalizer may `retire_through(seq)` after commit/abort.
    pub(crate) seq: u64,
}

impl KvPreparedWrite {
    pub fn working_set(&self) -> WorkingSetId {
        self.ws
    }

    /// Submission sequence for epoch retirement at finalize.
    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn targets(&self) -> &[PreparedTarget] {
        &self.targets
    }

    /// `(src, dst)` pairs the driver must copy (preserved cells) before the
    /// launch writes new cells.
    pub fn copy_plan(&self) -> impl Iterator<Item = (PhysicalKvPageId, PhysicalKvPageId)> + '_ {
        self.targets.iter().filter_map(|t| match *t {
            PreparedTarget::Cow { src, dst, .. } => Some((src, dst)),
            _ => None,
        })
    }
}

/// Committed metadata for one prepared target, in target order: the final
/// token-slot hashes and page hash of the page after the write.
#[derive(Debug, Clone)]
pub struct PageCommit {
    pub token_hashes: Vec<Option<Hash256>>,
    pub page_hash: Option<Hash256>,
}
