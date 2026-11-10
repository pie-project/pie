//! What each PE is: one [`PeRole`] a PE, set by the floor. A role says which kernel the PE runs
//! and its static place in the program (unit, local row / column, variant); the queues and
//! routes it uses are the fabric's ([`super::fabric::PeIo`]).

use serde::{Deserialize, Serialize};

/// A PE's role.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeRole {
    /// Unused (a filler corner, the wafer's spare area): an empty kernel.
    Idle,
    Weight(Weight),
    /// A unit row's control PE (row reductions' root, x handoff in / out, embedding in).
    Cx(Cx),
    /// A unit column's control PE (column reductions' root, norms, RoPE, attention core, act).
    Cy(Cy),
    /// A head weight PE (vocabulary rows of the lm_head).
    Head(Head),
    /// The head control line (row reductions of the head, the token loop's max-reduce).
    HeadCtl(HeadCtl),
    /// A gap slot PE: routes only, relays the vertical handoff.
    Relay,
    /// Gemma's PLE storage PE.
    Store(Store),
}

/// The kernel a role runs: the key of the layout's kind map and of the kernel params.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RoleKind {
    Idle,
    Weight,
    /// gpt-oss's attention / MoE weight halves (`wa` / `wm`).
    WeightAttn,
    WeightMoe,
    Cx,
    Cy,
    /// gpt-oss's CY halves (`cya` / `cym`).
    CyAttn,
    CyMoe,
    Head,
    HeadCtl,
    Relay,
    Store,
}

impl PeRole {
    pub fn kind(&self) -> RoleKind {
        match self {
            PeRole::Idle => RoleKind::Idle,
            PeRole::Weight(w) => match w.part {
                WeightPart::Attn => RoleKind::WeightAttn,
                WeightPart::Moe => RoleKind::WeightMoe,
                _ => RoleKind::Weight,
            },
            PeRole::Cx(_) => RoleKind::Cx,
            PeRole::Cy(c) => match c.part {
                WeightPart::Attn => RoleKind::CyAttn,
                WeightPart::Moe => RoleKind::CyMoe,
                _ => RoleKind::Cy,
            },
            PeRole::Head(_) => RoleKind::Head,
            PeRole::HeadCtl(_) => RoleKind::HeadCtl,
            PeRole::Relay => RoleKind::Relay,
            PeRole::Store(_) => RoleKind::Store,
        }
    }
}

impl RoleKind {
    /// The kind's name in the rcfg dumps and the kernel layout (`geo/*.rs::kind`,
    /// `mkfull*.py`'s kind maps): "filler", "w", "wa", "wm", "cx", "cy", "cya", "cym", "head",
    /// "hctl", "relay", "ple".
    pub fn name(self) -> &'static str {
        match self {
            RoleKind::Idle => "filler",
            RoleKind::Weight => "w",
            RoleKind::WeightAttn => "wa",
            RoleKind::WeightMoe => "wm",
            RoleKind::Cx => "cx",
            RoleKind::Cy => "cy",
            RoleKind::CyAttn => "cya",
            RoleKind::CyMoe => "cym",
            RoleKind::Head => "head",
            RoleKind::HeadCtl => "hctl",
            RoleKind::Relay => "relay",
            RoleKind::Store => "ple",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Weight {
    pub unit: u32,
    /// Local weight row / column (control lines skipped).
    pub row: u32,
    pub col: u32,
    /// Which part of the unit row it serves.
    pub part: WeightPart,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightPart {
    /// Every layer's projections (Qwen, Gemma).
    Dense,
    /// gpt-oss's attention columns.
    Attn,
    /// gpt-oss's MoE columns.
    Moe,
    /// Gemma 12B's rows: attention rows (scores segments, KV) or plain rows (projections only).
    AttnRow,
    PlainRow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cx {
    pub unit: u32,
    pub row: u32,
    /// The first unit (takes the embedding / the token loop's winner).
    pub first: bool,
    /// The last unit (sends the final x to the head).
    pub last: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cy {
    pub unit: u32,
    pub col: u32,
    pub part: WeightPart,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Head {
    pub row: u32,
    pub col: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadCtl {
    pub row: u32,
    /// The token loop's root (ring, force, nforce).
    pub root: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Store {
    /// First token, tokens held, its level in the store column, the top PE of the column.
    pub first_token: u32,
    pub tokens: u32,
    pub level: u32,
    pub top: bool,
}
