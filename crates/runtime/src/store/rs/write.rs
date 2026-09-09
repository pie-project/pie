#![allow(dead_code)]

use super::{RsSlotId, RsWorkingSetId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsStateTarget {
    pub slot: RsSlotId,
    pub reset: bool,
    pub copy_from: Option<RsSlotId>,
    pub fold_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RsBufferIntent {
    Write,
    Replay,
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
    pub(crate) buffer_span: Option<(u32, u32, RsBufferIntent)>,
    pub(crate) seq: u64,
    pub(crate) fold_len_is_bound: bool,
}

impl RsPreparedWrite {
    pub fn working_set(&self) -> RsWorkingSetId {
        self.ws
    }

    pub fn mark_fold_len_device(&mut self) {
        self.fold_len_is_bound = true;
    }

    pub fn fold_len_is_bound(&self) -> bool {
        self.fold_len_is_bound
    }

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
    pub(crate) len_is_bound: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsPublished {
    seqs: Vec<u64>,
}

impl RsPublished {
    pub(super) fn new(seqs: Vec<u64>) -> Self {
        Self { seqs }
    }

    pub fn seq(&self) -> u64 {
        self.seqs.iter().copied().max().unwrap_or(0)
    }

    pub fn seqs(&self) -> &[u64] {
        &self.seqs
    }

    pub fn rows(&self) -> usize {
        self.seqs.len()
    }
}
