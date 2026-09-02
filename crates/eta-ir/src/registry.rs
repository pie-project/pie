//! Stages, descriptor ports, intrinsic scoping, sink names, and the
//! bind-time [`ModelProfile`] — shared vocabulary between container,
//! validator, and backends.

use alloc::string::String;
use alloc::vec::Vec;

use super::op::IntrinsicId;
use crate::types::Dtype;

crate::declare_tagged_enum! {
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    /// Attachment stage of a traced program; wire tags stable. Boundary
    /// stages run once per pass, taps once per layer.
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

/// Execution order of the pass's channel-touching phases: stage order,
/// then op order within a stage. The descriptor sits between the prologue
/// and the per-layer taps; `0xFF` is its readiness-table tag (not a
/// program stage).
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
    /// The phase a program stage runs in. Total: every stage has one;
    /// [`Phase::Descriptor`] is the one phase with no stage.
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
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    /// Descriptor ports: the forward's ragged-tensor families. Consumption
    /// is fixed per port: the token family takes, geometry and masks read.
    pub enum Port {
        /// The token ids to embed, one flat run per request. Taken.
        EmbedTokens = 0, "embed_tokens";
        /// Row offsets splitting `embed_tokens` into per-request runs; one
        /// entry more than there are requests. Read.
        EmbedIndptr = 1, "embed_indptr";
        /// Each token's position in its sequence, driving both RoPE and the
        /// causal masks. Taken.
        Positions = 2, "positions";
        /// The KV pages each request may address, as working-set-relative
        /// indexes (survives the copy-on-write that moves the physical
        /// page). Translated through the working set's flat table before
        /// use. Read.
        Pages = 3, "pages";
        /// Row offsets splitting `pages` per request. Read.
        PageIndptr = 4, "page_indptr";
        /// Per-request readable KV extent after this pass's writes land.
        /// Read.
        KvLen = 5, "kv_len";
        /// The explicit KV write descriptor's page half: which page each
        /// token row is appended into ([`Port::Pages`]'s space). Stated,
        /// not derived — several lanes sharing one pool can't all use
        /// `have + row`. Taken.
        WSlot = 6, "w_slot";
        /// That row's offset inside the page [`Port::WSlot`] names. An offset
        /// is in no page space and is never translated. Taken.
        WOff = 7, "w_off";
        /// Which token rows the epilogue reads out; absent means the last
        /// row of each request. Read.
        Readout = 8, "readout";
        /// An explicit attention mask replacing the derived causal one.
        /// Read.
        AttnMask = 9, "attn_mask";
        // Recurrent-state buffered-slot family. Wire-additive: tags 0-9
        // stay unmoved. Reserved rather than reclaimed — renumbering would
        // change already-compiled containers' meaning.
        /// RESERVED. Page tables for the recurrent-state buffered slots.
        RsBufferPages = 10, "rs_buffer_pages";
        /// RESERVED. Row offsets splitting `rs_buffer_pages` per request.
        /// Derived by the runtime.
        RsBufferIndptr = 11, "rs_buffer_indptr";
        /// RESERVED, and inverting: the live buffered token count,
        /// device-resident. Staged as `FireGeometry::rs_buffer_lens`.
        RsBufferLen = 12, "rs_buffer_len";
        /// RESERVED. Each buffered token's slab. Derived by the runtime.
        RsWSlot = 13, "rs_w_slot";
        /// RESERVED. Each buffered token's offset within its slab. Derived by
        /// the runtime.
        RsWOff = 14, "rs_w_off";
        /// How far the folded boundary advances, per request — unlike
        /// 10-14, a real guest decision; the only RS port the host may not
        /// know the value of. Read.
        RsFoldLen = 15, "rs_fold_len";
    }
}

impl Port {
    /// True iff a channel bound to this port is consumed (take) by the
    /// pass; false = peeked (read). Token-indexed ports (embed, positions,
    /// `w_slot`/`w_off`) consume; geometry and masks are state, peeked.
    ///
    /// `RsFoldLen` also consumes: a count spent by the fire that folds on
    /// it, not state a later fire re-reads.
    pub fn consumes(self) -> bool {
        matches!(
            self,
            Port::EmbedTokens
                | Port::Positions
                | Port::WSlot
                | Port::WOff
                | Port::RsWSlot
                | Port::RsWOff
                | Port::RsFoldLen
        )
    }
}

/// A set of descriptor ports, as one word. Bit `p as u8` is port `p` — one
/// numbering, the container's own, so a port added to [`Port`] gets a bit
/// for free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PortMask(u32);

impl PortMask {
    /// The empty set.
    pub const NONE: PortMask = PortMask(0);

    /// The three ports a decode envelope resolves: token ids, positions,
    /// and each request's readable KV extent — enough to run a decode step
    /// (one row per request, host owns the page table).
    pub const DECODE_ENVELOPE: PortMask = PortMask::of(&[
        Port::EmbedTokens,
        Port::Positions,
        Port::KvLen,
    ]);

    /// The seven ports a device-resolved geometry resolves: decode
    /// envelope plus page table, row split, and adapter routing — the
    /// device derives the whole fire geometry, host stages only the descriptor.
    pub const DEVICE_GEOMETRY: PortMask = PortMask::of(&[
        Port::EmbedTokens,
        Port::Positions,
        Port::KvLen,
        Port::Pages,
        Port::PageIndptr,
        Port::WSlot,
        Port::WOff,
    ]);

    /// The recurrent-state buffered-slot family (tags 10-14) — reserved.
    pub const RS_BUFFER: PortMask = PortMask::of(&[
        Port::RsBufferPages,
        Port::RsBufferIndptr,
        Port::RsBufferLen,
        Port::RsWSlot,
        Port::RsWOff,
    ]);

    /// The set holding exactly the listed ports.
    pub const fn of(ports: &[Port]) -> PortMask {
        let mut bits = 0u32;
        let mut index = 0;
        while index < ports.len() {
            bits |= 1u32 << (ports[index] as u8);
            index += 1;
        }
        PortMask(bits)
    }

    /// The set this raw word denotes. Bit `p as u8` is port `p`.
    #[must_use]
    pub const fn from_bits(bits: u32) -> PortMask {
        PortMask(bits)
    }

    /// This set as a raw word.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Is `port` in the set?
    #[must_use]
    pub const fn contains(self, port: Port) -> bool {
        self.0 & (1u32 << (port as u8)) != 0
    }

    /// Is every port of `other` in this set?
    #[must_use]
    pub const fn covers(self, other: PortMask) -> bool {
        self.0 & other.0 == other.0
    }

    /// The set with `port` added.
    #[must_use]
    pub const fn with(self, port: Port) -> PortMask {
        PortMask(self.0 | (1u32 << (port as u8)))
    }

    /// The union of two sets.
    #[must_use]
    pub const fn union(self, other: PortMask) -> PortMask {
        PortMask(self.0 | other.0)
    }

    /// Is the set empty?
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Every port in the set, in wire-tag order.
    pub fn iter(self) -> impl Iterator<Item = Port> {
        Port::ALL.iter().copied().filter(move |&p| self.contains(p))
    }
}

/// How much of a fire's geometry the device resolves for itself — three
/// points on one axis (how far the descriptor ports reach). The classes
/// are the port sets they name; [`GeometryClass::ports`] is where that's written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GeometryClass {
    /// The host resolves everything and the device binds no descriptor port.
    #[default]
    Host,
    /// The device resolves a decode step's envelope
    /// ([`PortMask::DECODE_ENVELOPE`]); the host still owns the page table.
    DecodeEnvelope,
    /// The device resolves the whole fire geometry
    /// ([`PortMask::DEVICE_GEOMETRY`]).
    DeviceGeometry,
}

impl GeometryClass {
    /// The ports this class requires a device to serve.
    #[must_use]
    pub const fn ports(self) -> PortMask {
        match self {
            GeometryClass::Host => PortMask::NONE,
            GeometryClass::DecodeEnvelope => PortMask::DECODE_ENVELOPE,
            GeometryClass::DeviceGeometry => PortMask::DEVICE_GEOMETRY,
        }
    }

    /// The most demanding class `served` can carry, widest first. Total:
    /// [`GeometryClass::Host`] asks for nothing, so it's always an answer.
    #[must_use]
    pub fn admitted_by(served: PortMask) -> GeometryClass {
        if served.covers(PortMask::DEVICE_GEOMETRY) {
            GeometryClass::DeviceGeometry
        } else if served.covers(PortMask::DECODE_ENVELOPE) {
            GeometryClass::DecodeEnvelope
        } else {
            GeometryClass::Host
        }
    }
}

/// Where a configuration sink's effect is consumed — drives the
/// stage-precedence check: a sink call is legal only strictly before its
/// consumption point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

/// Every [`IntrinsicId::AttnScore`] row is this many F32 slots wide: the
/// live prefix carries the mass, the rest is exactly `0.0`. A constant
/// (not a profile field) because the guest side has no host call for a
/// score ceiling.
///
/// A request whose live KV exceeds it is refused by name rather than
/// truncated — a truncated row would not sum to one.
pub const ATTN_SCORE_KV_MAX: u32 = 2048;

/// Intrinsic value scope: which stages may materialize it.
pub fn intrinsic_stages(intr: IntrinsicId) -> &'static [Stage] {
    match intr {
        IntrinsicId::Logits
        | IntrinsicId::MtpLogits
        | IntrinsicId::Hidden
        | IntrinsicId::ValueHead => &[Stage::Epilogue],
        IntrinsicId::MtpDrafts => &[Stage::Epilogue],
        // Epilogue only: the observability contract. The capture arm
        // accumulates per-key mass as the graph runs; the epilogue reads
        // the whole rectangle at once, so scores are exactly as fresh as
        // the fire that produced them.
        IntrinsicId::AttnScore => &[Stage::Epilogue],
        IntrinsicId::Query | IntrinsicId::Layer => &[Stage::OnAttnProj, Stage::OnAttn],
    }
}

/// Whether `profile` provides `intr`. Deliberately one exhaustive match,
/// not a gated-list + mapping split — that shape lets a newly gated
/// intrinsic default to available if the mapping misses it.
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
/// `replayable = false` is rejected at bind: a time/load-varying return
/// can't be replayed, cached by hash, or batched with an identical trace.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelProfile {
    /// Token-vocabulary size; the trailing extent of a logits row.
    pub vocab: u32,
    /// Tokens per KV page.
    pub page_size: u32,
    /// How many transformer layers the per-layer taps fire for.
    pub num_layers: u32,
    /// Concrete dtype `ACT` resolves to (interpreter-visible materialization is F32).
    pub activation: Dtype,
    /// `[k, vocab]` F32 draft logits intrinsic available (multi-token-prediction head).
    pub has_mtp_logits: bool,
    /// `[k]` I32 draft tokens intrinsic ([`IntrinsicId::MtpDrafts`])
    /// available (device-resident spec-decode drafts).
    pub has_mtp_drafts: bool,
    /// A scalar value-head intrinsic is available.
    pub has_value_head: bool,
    /// `[layers * heads, kv_max]` F32 per-key attention mass
    /// ([`IntrinsicId::AttnScore`]) available at the epilogue. A backend
    /// property too: needs a score-observing kernel, refused for
    /// soft-capped or sliding-window attention.
    pub has_attn_score: bool,
    /// The backend honours an `attn_page_mask` sink. Without this flag a
    /// program would validate against every backend and fail at its first
    /// fire on ones that can't enforce it.
    pub has_attn_page_mask: bool,
    /// The backend honours a `lora` sink (A/B/sites config, low-rank
    /// delta at declared projection sites). Same contract as
    /// `has_attn_page_mask`: naming it type-checks everywhere, honouring
    /// it must be a bind-time refusal, not a silent no-op.
    pub has_lora: bool,
    /// Available second-party kernels + sinks, by name.
    pub kernels: Vec<KernelInfo>,
}

impl ModelProfile {
    /// The lowercase name of [`ModelProfile::activation`], for a digest or
    /// diagnostic. A method because the answer belongs to ETA, not the
    /// field: `worker`'s boot-consistency digest hashes these bytes
    /// without otherwise knowing this crate.
    pub fn activation_name(&self) -> &'static str {
        crate::types::name_or_unknown(self.activation)
    }

    /// The registry entry for `name`, or `None` if this backend does not
    /// offer it.
    pub fn kernel(&self, name: &str) -> Option<&KernelInfo> {
        self.kernels.iter().find(|k| k.name == name)
    }

    /// A small test/dummy-engine profile.
    pub fn dummy() -> Self {
        ModelProfile {
            vocab: 32,
            page_size: 4,
            num_layers: 2,
            activation: Dtype::F32,
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
