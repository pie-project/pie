use super::hash::Hash256;
use super::page_table::{PhysicalKvPageId, WorkingSetId};

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

    #[cfg(test)]
    pub fn dst(&self) -> PhysicalKvPageId {
        match *self {
            PreparedTarget::Fresh { dst, .. }
            | PreparedTarget::InPlace { dst, .. }
            | PreparedTarget::Cow { dst, .. } => dst,
        }
    }
}

#[derive(Debug)]
pub struct KvPreparedWrite {
    pub(crate) ws: WorkingSetId,
    pub(crate) targets: Vec<PreparedTarget>,
    pub(crate) allocated: Vec<PhysicalKvPageId>,
    pub(crate) cow_start: Option<u64>,
    pub(crate) seq: u64,
}

impl KvPreparedWrite {
    pub fn targets(&self) -> &[PreparedTarget] {
        &self.targets
    }

    pub fn copy_plan(&self) -> impl Iterator<Item = (PhysicalKvPageId, PhysicalKvPageId)> + '_ {
        self.targets.iter().filter_map(|t| match *t {
            PreparedTarget::Cow { src, dst, .. } => Some((src, dst)),
            _ => None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PageCommit {
    pub token_hashes: Vec<Option<Hash256>>,
    pub page_hash: Option<Hash256>,
}
