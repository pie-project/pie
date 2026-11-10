//! What each PE is: one [`PeRole`] a PE, set by the floor. A role is the kernel slot it fills
//! ([`Kind`]: open, a block declares its own) and its static place in the program ([`Place`]: the
//! wafer program architecture's positions, no model in them); the queues and routes it uses are the
//! fabric's ([`super::fabric::PeIo`]).

use serde::{Serialize, Serializer};

/// The kernel slot a PE fills: the key of the layout's kind map, of the kernel builds and of the
/// rcfg dumps. Open: the architecture's own kinds are the consts here; a block that needs another
/// (gpt-oss's attention / MoE halves, a storage column) declares a const in its module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kind {
    /// The name in the rcfg dumps and the levers (`w.FIFOW:16`).
    pub name: &'static str,
    /// The file stem of its source in the build directory (`build*.py`'s names).
    pub stem: &'static str,
}

impl Kind {
    pub const fn new(name: &'static str, stem: &'static str) -> Kind {
        Kind { name, stem }
    }

    /// Unused PEs: the empty filler kernel.
    pub const IDLE: Kind = Kind::new("filler", "filler");
    /// A unit's weight PEs.
    pub const WEIGHT: Kind = Kind::new("w", "weight");
    /// A unit row's control PE (row reductions' root, x handoff in / out, embedding in).
    pub const CX: Kind = Kind::new("cx", "cx");
    /// A unit column's control PE (column reductions' root, norms, RoPE, attention core, act).
    pub const CY: Kind = Kind::new("cy", "cy");
    /// A head weight PE (vocabulary rows of the lm_head).
    pub const HEAD: Kind = Kind::new("head", "head");
    /// The head control line (row reductions of the head, the token loop's max-reduce).
    pub const HEAD_CTL: Kind = Kind::new("hctl", "hctl");
    /// A gap slot PE: routes only, relays the vertical handoff.
    pub const RELAY: Kind = Kind::new("relay", "relay");
}

impl Serialize for Kind {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name)
    }
}

/// Where a PE sits in the wafer program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Place {
    /// No place (a filler corner, the wafer's spare area, a route-only PE).
    None,
    /// A unit's weight PE at local weight row / column (control lines skipped).
    Weight {
        unit: u32,
        row: u32,
        col: u32,
    },
    /// A unit row's control PE; `first` takes the embedding / the token loop's winner, `last`
    /// sends the final x to the head.
    Cx {
        unit: u32,
        row: u32,
        first: bool,
        last: bool,
    },
    Cy {
        unit: u32,
        col: u32,
    },
    Head {
        row: u32,
        col: u32,
    },
    /// `root`: the token loop's root (ring, force, nforce).
    HeadCtl {
        row: u32,
        root: bool,
    },
    /// A storage column's PE: it holds items `[first, first + count)` (none: a route-only PE of
    /// the column), at `level` from the column's foot, `top` the column's top PE.
    Store {
        first: u32,
        count: u32,
        level: u32,
        top: bool,
    },
}

/// A PE's role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct PeRole {
    pub kind: Kind,
    pub place: Place,
}

impl PeRole {
    pub const IDLE: PeRole = PeRole {
        kind: Kind::IDLE,
        place: Place::None,
    };

    pub const fn new(kind: Kind, place: Place) -> PeRole {
        PeRole { kind, place }
    }

    pub fn is_idle(&self) -> bool {
        self.kind == Kind::IDLE
    }
}
