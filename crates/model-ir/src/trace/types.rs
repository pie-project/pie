//! Shared trace vocabulary types; wire discriminants are ABI.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dim {
    /// The fire's token rows (`N`; equals `Requests` on a pure-decode fire).
    Tokens,
    /// The fire's request rows (`R`).
    Requests,
    /// A load-time constant: hidden size, head count x head dim, vocab.
    Const(u32),
    /// MoE aligned route count: routes bucketed by expert and block-padded.
    MoeAlignedRoutes {
        top_k: u32,
        experts: u32,
        block: u32,
    },
}

impl Dim {
    /// Aligned route count for `n` token rows.
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
    /// Half; used when MXFP4 routed GEMVs consume cast activations.
    F16,
    F32,
    I32,
}

pub type ValueId = u32;

/// Marks values whose contents choose lowering-relevant structure per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynAxis {
    PerToken,
}

/// RMSNorm weight convention: `Gemma` uses `(1 + w)`, `Plain` uses `w`.
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

/// Fire-shape class a lowered trace is specialized to; semantic traces
/// have no class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FireClass {
    Decode,
    Prefill,
    // Wire numbers for retired classes stay reserved; the ABI is append-only.
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
    /// Explicit KV-write descriptors. Wire kind 0, payload unused.
    HasWriteDesc,
    /// `N <= k`. Wire kind 1, payload `k`.
    TokensLE(u32),
    /// `N > k`. Wire kind 2, payload `k`.
    TokensGT(u32),
    /// `N % k == 0` with `k != 0`. Wire kind 10, payload `k`;
    /// a threshold is not equivalent for fixed-tile kernels.
    TokensMultipleOf(u32),
    /// Attached programs want attention scores. Wire kind 3, payload unused.
    WantsAttnScore,
    /// Custom attention mask present. Wire kind 4, payload unused.
    HasCustomMask,
    /// Retired stage-hook predicate. Wire kind 5 is reserved.
    HasStageHooks,
    /// Usable LoRA lanes; q/v adapter delta must precede fused decode-QKV.
    /// Wire kind 6, payload unused.
    HasLora,
    /// Every row is a one-token query window. Wire kind 7, payload unused.
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
            // 10 and not 8: 8 and 9 are `driver-cuda`'s two Peel slots, placed
            // above the GuardPred range when it ended at 7, so a guard added
            // at 8 would quietly become a Peel in the device predicate word.
            GuardPred::TokensMultipleOf(k) => (10, k),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookStage {
    /// Before attention; a page-mask sink narrows the same stated kernel.
    OnAttnProj,
    OnAttn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardArm {
    pub pred: GuardPred,
    pub ops: u32,
}

/// One seam statement in text order; `op` is the carrying op when one exists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeamStatement {
    pub seam: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<u32>,
    /// Values exposed by the statement, not inferred from neighbours.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<ValueId>,
}
