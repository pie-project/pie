//! Stages, descriptor ports, intrinsic scoping, well-known sink names, and the
//! bind-time [`ModelProfile`] — the shared vocabulary between the container,
//! the validator, and every backend. Wire tags here are frozen constants
//! (mirrored into `include/ptir_abi.h`).

use alloc::string::String;
use alloc::vec::Vec;

use super::op::IntrinsicId;
use crate::types::DType;

crate::declare_tagged_enum! {
    /// Attachment stage of a traced program. Wire tags stable.
    /// Boundary stages run once per pass; the anatomical taps run once per layer.
    pub enum Stage {
        /// Before any KV read — weight swap, pass-wide config sinks.
        Prologue = 0, "prologue";
        /// Per layer, before attention (query in scope).
        OnAttnProj = 1, "on_attn_proj";
        /// Per layer, after attention.
        OnAttn = 2, "on_attn";
        /// After the forward — sampling programs.
        Epilogue = 3, "epilogue";
    }
}

impl Stage {
    /// True for the per-layer anatomical taps.
    pub fn per_layer(self) -> bool {
        matches!(self, Stage::OnAttnProj | Stage::OnAttn)
    }
}

/// Execution order of the pass's channel-touching phases — the global
/// per-channel program order is stage order, then op order
/// within a stage. The descriptor (port peeks/takes) sits between the
/// prologue and the per-layer taps. `0xFF` is the descriptor's tag in the
/// readiness table (it is not a program stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Phase {
    /// Before the forward pass runs.
    Prologue,
    /// Descriptor-port reads, between the prologue and the layer taps.
    Descriptor,
    /// At each layer's attention projection.
    OnAttnProj,
    /// At each layer's attention.
    OnAttn,
    /// After the forward pass completes.
    Epilogue,
}

/// Wire tag for [`Phase`] in the readiness table (`Stage` tags + 0xFF).
pub const PHASE_DESCRIPTOR_TAG: u8 = 0xFF;

impl Phase {
    /// This phase's tag in the readiness table.
    pub fn tag(self) -> u8 {
        match self {
            Phase::Prologue => Stage::Prologue as u8,
            Phase::Descriptor => PHASE_DESCRIPTOR_TAG,
            Phase::OnAttnProj => Stage::OnAttnProj as u8,
            Phase::OnAttn => Stage::OnAttn as u8,
            Phase::Epilogue => Stage::Epilogue as u8,
        }
    }
    /// The phase a program stage runs in.
    ///
    /// Total, because every stage has one; [`Phase::Descriptor`] is the one
    /// phase with no stage, since port reads are not a program body.
    pub fn of_stage(s: Stage) -> Phase {
        match s {
            Stage::Prologue => Phase::Prologue,
            Stage::OnAttnProj => Phase::OnAttnProj,
            Stage::OnAttn => Phase::OnAttn,
            Stage::Epilogue => Phase::Epilogue,
        }
    }
    /// All phases in execution order.
    pub const ORDER: [Phase; 5] = [
        Phase::Prologue,
        Phase::Descriptor,
        Phase::OnAttnProj,
        Phase::OnAttn,
        Phase::Epilogue,
    ];
}

crate::declare_tagged_enum! {
    /// Descriptor ports: the forward's ragged-tensor families.
    /// Consumption discipline is fixed per port: the token family **takes**
    /// (a token is spent by the pass that embeds it), geometry and masks **read**
    /// (state, not a message).
    pub enum Port {
        /// The token ids to embed, one flat run per request. Taken.
        EmbedTokens = 0, "embed_tokens";
        /// Row offsets splitting `embed_tokens` into per-request runs; one
        /// entry more than there are requests. Read.
        EmbedIndptr = 1, "embed_indptr";
        /// Each token's position in its sequence, driving both RoPE and the
        /// causal masks. Taken.
        Positions = 2, "positions";
        /// The KV pages each request may address. Read.
        Pages = 3, "pages";
        /// Row offsets splitting `pages` per request. Read.
        PageIndptr = 4, "page_indptr";
        /// Per-request readable KV extent after this pass's writes land.
        /// Read.
        KvLen = 5, "kv_len";
        /// Which adapter slot each token routes to. Taken.
        WSlot = 6, "w_slot";
        /// Each token's offset within its adapter slot. Taken.
        WOff = 7, "w_off";
        /// Which token rows the epilogue reads out; absent means the last
        /// row of each request. Read.
        Readout = 8, "readout";
        /// An explicit attention mask replacing the derived causal one.
        /// Read.
        AttnMask = 9, "attn_mask";
        // ── Recurrent-state buffered-slot family. Wire-additive: tags 0-9 are
        // unmoved, so a pure-attention guest's container is byte-identical.
        //
        // NO GUEST BINDS THESE ANY MORE. `rs-geometry` used to carry the
        // buffer's addressing, and the runtime derived the same values from
        // the `RsStore` it is authoritative for and refused any fire whose
        // guest copy disagreed — five channels of page arithmetic with one
        // satisfying assignment. The tags are RESERVED rather than reclaimed:
        // renumbering would silently change the meaning of already-compiled
        // containers, which is a far worse trade than five unused names.
        //
        // `RsBufferLen` is the exception, and its direction is INVERTING. The
        // live buffered token count is exactly the quantity t15 makes
        // device-resident, so it comes back not as something the guest states
        // but as something the device writes and the host reads as an upper
        // bound. `FireGeometry::rs_buffer_lens` is already staged for it.
        /// RESERVED. The buffer's page pool. Derived by the runtime.
        RsBufferPages = 10, "rs_buffer_pages";
        /// RESERVED. Row offsets splitting `rs_buffer_pages` per request.
        /// Derived by the runtime.
        RsBufferIndptr = 11, "rs_buffer_indptr";
        /// RESERVED, and inverting: the live buffered token count, which t15
        /// makes device-resident. Staged as `FireGeometry::rs_buffer_lens`.
        RsBufferLen = 12, "rs_buffer_len";
        /// RESERVED. Each buffered token's slab. Derived by the runtime.
        RsWSlot = 13, "rs_w_slot";
        /// RESERVED. Each buffered token's offset within its slab. Derived by
        /// the runtime.
        RsWOff = 14, "rs_w_off";
        /// How far the folded boundary advances, per request. Unlike 10-14
        /// this is a real guest decision, and the only RS port whose value the
        /// host is allowed not to know: a device-computed accepted count
        /// reaches the recurrence through here instead of round-tripping
        /// through the host. Read.
        RsFoldLen = 15, "rs_fold_len";
    }
}

impl Port {
    /// True iff a channel bound to this port is **consumed** (take) by the
    /// pass; false = peeked (read). The token-indexed family (embed,
    /// positions, `w_slot`/`w_off`) consumes — a token is spent by the pass
    /// that embeds it; geometry and masks are state, peeked.
    pub fn consumes(self) -> bool {
        matches!(
            self,
            Port::EmbedTokens | Port::Positions | Port::WSlot | Port::WOff | Port::RsWSlot
                | Port::RsWOff
        )
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

/// Intrinsic value scope: which stages may materialize it.
pub fn intrinsic_stages(intr: IntrinsicId) -> &'static [Stage] {
    match intr {
        IntrinsicId::Logits
        | IntrinsicId::MtpLogits
        | IntrinsicId::Hidden
        | IntrinsicId::ValueHead => &[Stage::Epilogue],
        IntrinsicId::MtpDrafts => &[Stage::Epilogue],
        // `OnAttn` only, and that is forced rather than chosen: the scores do
        // not exist until this layer's attention has run, and `OnAttnProj`
        // fires before it. That is also why an eviction policy cannot read
        // and act in the same fire — see the T11 note in `validate.rs`.
        IntrinsicId::AttnScore => &[Stage::OnAttn],
        IntrinsicId::Query | IntrinsicId::Layer => &[Stage::OnAttnProj, Stage::OnAttn],
    }
}

/// Whether `profile` provides `intr`.
///
/// Exhaustive on purpose, and deliberately one function.
///
/// The tempting split is two: a list of which intrinsics are model-gated, and
/// a mapping from each gated one to its profile flag with a `_ => true` arm.
/// Under that shape a new gated intrinsic added to the list but missed in the
/// mapping is available on every model — a capability check that passes by
/// being forgotten, which is the worst way for one to fail. One exhaustive
/// match with no catch-all makes the compiler ask the question instead.
pub fn intrinsic_available(intr: IntrinsicId, profile: &ModelProfile) -> bool {
    match intr {
        IntrinsicId::MtpLogits => profile.has_mtp_logits,
        IntrinsicId::MtpDrafts => profile.has_mtp_drafts,
        IntrinsicId::ValueHead => profile.has_value_head,
        IntrinsicId::AttnScore => profile.has_attn_score,
        IntrinsicId::Logits | IntrinsicId::Hidden | IntrinsicId::Query | IntrinsicId::Layer => true,
    }
}

/// A second-party kernel/sink the backend provides, resolved for
/// availability at bind time.
///
/// `replayable = false` is rejected at bind: a time- or load-varying return
/// is a register read in disguise, and a trace containing one cannot be
/// replayed, cached by hash, or batched with an identical trace.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelInfo {
    /// The name a [`KernelCall`](crate::op::Op::KernelCall) or
    /// [`SinkCall`](crate::op::Op::SinkCall) resolves to.
    pub name: String,
    /// For sinks: where the effect is consumed. `None` = a value-returning
    /// kernel (not a sink).
    pub sink_scope: Option<SinkScope>,
    /// Whether the same arguments always give the same result. `false` is
    /// rejected at bind.
    pub replayable: bool,
}

/// Everything bind needs from the model/backend: the trace-known constants,
/// the model-gated intrinsics, and the second-party registry.
#[derive(Clone, Debug)]
pub struct ModelProfile {
    /// Token-vocabulary size; the trailing extent of a logits row.
    pub vocab: u32,
    /// Tokens per KV page.
    pub page_size: u32,
    /// How many transformer layers the per-layer taps fire for.
    pub num_layers: u32,
    /// Concrete dtype `ACT` resolves to (bf16/fp8 quantized types are the
    /// backend's; the *interpreter-visible* materialization is F32).
    pub activation: DType,
    /// `[k, vocab]` F32 draft logits intrinsic available — a model with a
    /// multi-token-prediction head.
    pub has_mtp_logits: bool,
    /// `[k]` I32 draft tokens intrinsic ([`IntrinsicId::MtpDrafts`]) available —
    /// a model with an MTP head serving device-resident spec-decode drafts.
    pub has_mtp_drafts: bool,
    /// A scalar value-head intrinsic is available.
    pub has_value_head: bool,
    /// `[kv_max]` F32 head-folded attention weights
    /// ([`IntrinsicId::AttnScore`]) available. Unlike the MTP flags this is a *backend* property as much as
    /// a model one: it needs a score-observing attention kernel, and it is
    /// refused for soft-capped or sliding-window attention, where the captured
    /// row would not be the softmax the eviction papers define.
    pub has_attn_score: bool,
    /// The backend honours an `attn_page_mask` sink. `attn_page_mask` is a
    /// first-party name, so without this flag a program would validate against
    /// every backend and then fail at its first fire on the ones that cannot
    /// enforce it -- the opposite of the bind-time contract.
    pub has_attn_page_mask: bool,
    /// The backend honours a `lora` sink: it advertises that it can consume
    /// the sink's A/B/SITES configuration and apply the low-rank delta at the
    /// declared projection sites. Same shape of contract as
    /// `has_attn_page_mask` -- `lora` is a first-party name (its
    /// [`KNOWN_SINKS`] entry, reserved until now, is live), so a program
    /// naming it type-checks against every backend, and the backend's ability
    /// to HONOUR it must be a bind-time refusal rather than a silent no-op
    /// adapter.
    pub has_lora: bool,
    /// Available second-party kernels + sinks, by name.
    pub kernels: Vec<KernelInfo>,
}

impl ModelProfile {
    /// The registry entry for `name`, or `None` if this backend does not
    /// offer it.
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
            has_attn_score: true,
            has_attn_page_mask: true,
            has_lora: true,
            kernels: Vec::new(),
        }
    }
}
