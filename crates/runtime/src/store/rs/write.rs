use super::{RsSlotId, RsWorkingSetId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsStateTarget {
    pub slot: RsSlotId,
    pub reset: bool,
    pub copy_from: Option<RsSlotId>,
    pub fold_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RsBufferTarget {
    Fresh {
        index: u32,
        dst: RsSlotId,
    },
    InPlace {
        index: u32,
        dst: RsSlotId,
    },
    Cow {
        index: u32,
        src: RsSlotId,
        dst: RsSlotId,
    },
}

impl RsBufferTarget {
    pub fn src(&self) -> Option<RsSlotId> {
        match *self {
            RsBufferTarget::Cow { src, .. } => Some(src),
            _ => None,
        }
    }

    pub fn dst(&self) -> RsSlotId {
        match *self {
            RsBufferTarget::Fresh { dst, .. }
            | RsBufferTarget::InPlace { dst, .. }
            | RsBufferTarget::Cow { dst, .. } => dst,
        }
    }
}

#[derive(Debug)]
pub struct RsPreparedWrite {
    pub(crate) ws: RsWorkingSetId,
    pub(crate) state: Option<RsStateTarget>,
    pub(crate) buffers: Vec<RsBufferTarget>,
    pub(crate) allocated: Vec<RsSlotId>,
    pub(crate) seq: u64,
    /// Rows the fire lands the state as of, each in a slot of its own.
    pub(crate) snapshots: Vec<(u32, RsSlotId)>,
}

impl RsPreparedWrite {
    pub fn seq(&self) -> u64 {
        self.seq
    }

    pub fn state(&self) -> Option<&RsStateTarget> {
        self.state.as_ref()
    }

    pub fn buffer_targets(&self) -> &[RsBufferTarget] {
        &self.buffers
    }

    pub fn buffer_copy_plan(&self) -> impl Iterator<Item = (RsSlotId, RsSlotId)> + '_ {
        self.buffers.iter().filter_map(|t| match *t {
            RsBufferTarget::Cow { src, dst, .. } => Some((src, dst)),
            _ => None,
        })
    }
}

#[derive(Debug, Default)]
#[must_use = "a deferred fold that is never committed leaves the boundary behind"]
pub struct RsPendingFolds(pub(crate) Vec<RsPendingFold>);

#[derive(Debug, Clone, Copy)]
pub struct RsPendingFold {
    pub(crate) ws: super::RsWorkingSetId,
    pub(crate) tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsPublished {
    seqs: Vec<u64>,
    snapshots: Vec<RsSnapshot>,
}

/// A working set's state as of `row` of the fire, held by `snapshot`, a
/// hidden working set no process holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsSnapshot {
    pub ws: RsWorkingSetId,
    pub row: u32,
    pub slot: RsSlotId,
    pub snapshot: RsWorkingSetId,
}

impl RsPublished {
    pub(super) fn new(seqs: Vec<u64>, snapshots: Vec<RsSnapshot>) -> Self {
        Self { seqs, snapshots }
    }

    pub fn seqs(&self) -> &[u64] {
        &self.seqs
    }

    pub fn snapshots(&self) -> &[RsSnapshot] {
        &self.snapshots
    }
}
