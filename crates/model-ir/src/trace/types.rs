use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dim {
    Tokens,

    Requests,

    Const(u32),

    MoeAlignedRoutes {
        top_k: u32,
        experts: u32,
        block: u32,
    },
}

impl Dim {
    pub fn moe_aligned_rows(n: u32, top_k: u32, experts: u32, block: u32) -> u32 {
        let routes = n * top_k;
        let padded = routes + experts.min(routes) * block.saturating_sub(1);
        padded.div_ceil(block.max(1)) * block.max(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<Dim>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    BF16,

    F16,
    F32,
    I32,
}

pub type ValueId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynAxis {
    PerToken,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormVariant {
    #[default]
    Plain,
    Gemma,
}

impl NormVariant {
    pub fn is_plain(&self) -> bool {
        *self == NormVariant::Plain
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RopeKind {
    Standard,
    Yarn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FireClass {
    Decode,
    Prefill,
}

impl FireClass {
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardPred {
    HasWriteDesc,

    TokensLE(u32),

    TokensGT(u32),

    TokensMultipleOf(u32),

    WantsAttnScore,

    HasCustomMask,

    HasStageHooks,

    HasLora,

    WindowOne,
}

impl GuardPred {
    pub fn wire(&self) -> (u32, u32) {
        match *self {
            GuardPred::HasWriteDesc => (0, 0),
            GuardPred::TokensLE(k) => (1, k),
            GuardPred::TokensGT(k) => (2, k),
            GuardPred::WantsAttnScore => (3, 0),
            GuardPred::HasCustomMask => (4, 0),
            GuardPred::HasStageHooks => (5, 0),
            GuardPred::HasLora => (6, 0),
            GuardPred::WindowOne => (7, 0),

            GuardPred::TokensMultipleOf(k) => (10, k),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookStage {
    OnAttnProj,
    OnAttn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardArm {
    pub pred: GuardPred,
    pub ops: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeamStatement {
    pub seam: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<u32>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<ValueId>,
}
