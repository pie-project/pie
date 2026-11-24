//! A program's state root: every region of PEs that keeps state across launches, its slots and
//! sizes, and the PEs the host addresses it through (`host.json`'s `state`). The engine acts on
//! a region's fields alone, so a new kind of state is a new region, not new engine code.

use serde::{Deserialize, Serialize};

/// A PE the host writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Pe {
    pub x: u32,
    pub y: u32,
}

/// A paged region's pages: `size` positions each, `pages` a column (its last the scratch page:
/// an idle row's), each of its `slots` lane slots holding at most `pages`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pages {
    pub size: u32,
    pub pages: u32,
}

/// One region of state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Region {
    /// What it holds (`kv`: a K / V cache of positions; `gdn`, `kda`, `mamba2`: a recurrence's
    /// state a slot), as the compiler names it.
    pub kind: String,
    /// The layer block and the head it serves.
    pub block: u32,
    pub head: u32,
    /// Its slots (a launch's lane slots: a lane's state, or a lane's page table).
    pub slots: u32,
    /// Its paging, where it pages its positions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paging: Option<Pages>,
    /// Its PEs the host tells a step's first position (`pos`, a word).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pos: Vec<Pe>,
    /// Its PEs the host tells a step's rows' (lane slot, state slot, position) (`rowp`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<Pe>,
    /// Its PEs the host tells each lane slot's pages (`ptab`: `slots` x `pages` words).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<Pe>,
}

/// A program's state: its lane slots a launch (a step's rows) and its regions.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRoot {
    pub lanes: u32,
    pub regions: Vec<Region>,
    /// Every `rowp` PE advances each row's position at a launch's end (`Column::advances`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub advances: bool,
}

impl StateRoot {
    /// The paging its paged regions share (their sizes and pages, with its lane slots), where
    /// any pages.
    #[must_use]
    pub fn paging(&self) -> Option<(Pages, u32)> {
        self.regions
            .iter()
            .find_map(|r| r.paging)
            .map(|g| (g, self.lanes))
    }

    /// Every region's PEs of one kind of address (`f` picks it), sorted.
    #[must_use]
    pub fn pes(&self, f: impl Fn(&Region) -> &[Pe]) -> Vec<Pe> {
        let mut v: Vec<Pe> = self.regions.iter().flat_map(|r| f(r).to_vec()).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}
