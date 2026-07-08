//! Stages, descriptor ports, intrinsic scoping, well-known sink names, and the
//! bind-time [`ModelProfile`] — the shared vocabulary between the container,
//! the validator, and every backend. Wire tags here are frozen constants
//! (mirrored into `include/ptir_abi.h`).

use alloc::string::String;
use alloc::vec::Vec;

use super::op::IntrinsicId;
use crate::types::DType;

/// Attachment stage of a traced program (overview §5.3). Wire tags stable.
/// Boundary stages run once per pass; the anatomical taps run once per layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Stage {
    /// Before any KV read — weight swap, pass-wide config sinks.
    Prologue = 0,
    /// Per layer, before attention (query in scope).
    OnAttnProj = 1,
    /// Per layer, after attention.
    OnAttn = 2,
    /// After the forward — sampling programs.
    Epilogue = 3,
}

impl Stage {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Stage::Prologue,
            1 => Stage::OnAttnProj,
            2 => Stage::OnAttn,
            3 => Stage::Epilogue,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Stage::Prologue => "prologue",
            Stage::OnAttnProj => "on_attn_proj",
            Stage::OnAttn => "on_attn",
            Stage::Epilogue => "epilogue",
        }
    }
    /// True for the per-layer anatomical taps.
    pub fn per_layer(self) -> bool {
        matches!(self, Stage::OnAttnProj | Stage::OnAttn)
    }
}

/// Execution order of the pass's channel-touching phases — the global
/// per-channel program order (overview §1) is stage order, then op order
/// within a stage. The descriptor (port peeks/takes) sits between the
/// prologue and the per-layer taps. `0xFF` is the descriptor's tag in the
/// readiness table (it is not a program stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    Prologue,
    Descriptor,
    OnAttnProj,
    OnAttn,
    Epilogue,
}

/// Wire tag for [`Phase`] in the readiness table (`Stage` tags + 0xFF).
pub const PHASE_DESCRIPTOR_TAG: u8 = 0xFF;

impl Phase {
    pub fn tag(self) -> u8 {
        match self {
            Phase::Prologue => Stage::Prologue as u8,
            Phase::Descriptor => PHASE_DESCRIPTOR_TAG,
            Phase::OnAttnProj => Stage::OnAttnProj as u8,
            Phase::OnAttn => Stage::OnAttn as u8,
            Phase::Epilogue => Stage::Epilogue as u8,
        }
    }
    pub fn of_stage(s: Stage) -> Phase {
        match s {
            Stage::Prologue => Phase::Prologue,
            Stage::OnAttnProj => Phase::OnAttnProj,
            Stage::OnAttn => Phase::OnAttn,
            Stage::Epilogue => Phase::Epilogue,
        }
    }
    /// All phases in execution order.
    pub const ORDER: [Phase; 5] =
        [Phase::Prologue, Phase::Descriptor, Phase::OnAttnProj, Phase::OnAttn, Phase::Epilogue];
}

/// Descriptor ports (overview §5.1): the forward's ragged-tensor families.
/// Consumption discipline is fixed per port: the token family **takes**
/// (a token is spent by the pass that embeds it), geometry and masks **read**
/// (state, not a message).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Port {
    EmbedTokens = 0,
    EmbedIndptr = 1,
    Positions = 2,
    Pages = 3,
    PageIndptr = 4,
    KvLen = 5,
    WSlot = 6,
    WOff = 7,
    Readout = 8,
    AttnMask = 9,
}

impl Port {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Port::EmbedTokens,
            1 => Port::EmbedIndptr,
            2 => Port::Positions,
            3 => Port::Pages,
            4 => Port::PageIndptr,
            5 => Port::KvLen,
            6 => Port::WSlot,
            7 => Port::WOff,
            8 => Port::Readout,
            9 => Port::AttnMask,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Port::EmbedTokens => "embed_tokens",
            Port::EmbedIndptr => "embed_indptr",
            Port::Positions => "positions",
            Port::Pages => "pages",
            Port::PageIndptr => "page_indptr",
            Port::KvLen => "kv_len",
            Port::WSlot => "w_slot",
            Port::WOff => "w_off",
            Port::Readout => "readout",
            Port::AttnMask => "attn_mask",
        }
    }
    /// True iff a channel bound to this port is **consumed** (take) by the
    /// pass; false = peeked (read). §5.1: the token-indexed family (embed,
    /// positions, `w_slot`/`w_off`) consumes — a token is spent by the pass
    /// that embeds it; geometry and masks are state, peeked.
    pub fn consumes(self) -> bool {
        matches!(self, Port::EmbedTokens | Port::Positions | Port::WSlot | Port::WOff)
    }
}

/// Where a configuration sink's effect is consumed — drives the T11
/// stage-precedence check: a sink call is legal only at a stage strictly
/// preceding its consumption point (pass-wide ⇒ prologue only;
/// attention-scoped ⇒ prologue (all layers) or `on_attn_proj` (that layer)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkScope {
    /// Consumed by the whole forward (e.g. `lora`, `minference_sparse`).
    PassWide,
    /// Consumed by a layer's attention (e.g. `attn_page_mask`).
    Attention,
}

/// Well-known first-party sink names and their scopes. Second-party sinks are
/// container-named; their scope comes from the [`ModelProfile`] entry.
pub const KNOWN_SINKS: &[(&str, SinkScope)] = &[
    ("attn_page_mask", SinkScope::Attention),
    ("lora", SinkScope::PassWide),
    ("minference_sparse", SinkScope::PassWide),
];

/// Intrinsic value scope: which stages may materialize it (overview §5.3).
pub fn intrinsic_stages(intr: IntrinsicId) -> &'static [Stage] {
    match intr {
        IntrinsicId::Logits | IntrinsicId::MtpLogits | IntrinsicId::Hidden | IntrinsicId::ValueHead => {
            &[Stage::Epilogue]
        }
        IntrinsicId::MtpDrafts => &[Stage::Epilogue],
        IntrinsicId::Query | IntrinsicId::Layer => &[Stage::OnAttnProj, Stage::OnAttn],
    }
}

/// True iff the intrinsic's availability is a **model property** checked at
/// bind (overview §4).
pub fn intrinsic_model_gated(intr: IntrinsicId) -> bool {
    matches!(intr, IntrinsicId::MtpLogits | IntrinsicId::MtpDrafts | IntrinsicId::ValueHead)
}

/// A second-party kernel/sink the backend provides (bind-time availability,
/// overview §4). `replayable = false` violates §1's corollary (a time- or
/// load-varying return is a register read in disguise) and is rejected at
/// bind — the T10 lint.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelInfo {
    pub name: String,
    /// For sinks: where the effect is consumed. `None` = a value-returning
    /// kernel (not a sink).
    pub sink_scope: Option<SinkScope>,
    pub replayable: bool,
}

/// Everything bind needs from the model/backend: the trace-known constants,
/// the model-gated intrinsics, and the second-party registry.
#[derive(Clone, Debug)]
pub struct ModelProfile {
    pub vocab: u32,
    pub page_size: u32,
    pub num_layers: u32,
    /// Concrete dtype `ACT` resolves to (bf16/fp8 quantized types are the
    /// backend's; the *interpreter-visible* materialization is F32).
    pub activation: DType,
    pub has_mtp_logits: bool,
    /// `[k]` I32 draft tokens intrinsic ([`IntrinsicId::MtpDrafts`]) available —
    /// a model with an MTP head serving device-resident spec-decode drafts.
    pub has_mtp_drafts: bool,
    pub has_value_head: bool,
    /// Available second-party kernels + sinks, by name.
    pub kernels: Vec<KernelInfo>,
}

impl ModelProfile {
    pub fn kernel(&self, name: &str) -> Option<&KernelInfo> {
        self.kernels.iter().find(|k| k.name == name)
    }

    /// A small test/dummy-driver profile.
    pub fn dummy() -> Self {
        ModelProfile {
            vocab: 32,
            page_size: 4,
            num_layers: 2,
            activation: DType::F32,
            has_mtp_logits: true,
            has_mtp_drafts: true,
            has_value_head: true,
            kernels: Vec::new(),
        }
    }
}
