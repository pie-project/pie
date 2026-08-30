//! Stages, descriptor ports, intrinsic scoping, well-known sink names, and the
//! bind-time [`ModelProfile`] — the shared vocabulary between the container,
//! the validator, and every backend. Wire tags here are frozen constants, read
//! from this module by every backend rather than mirrored into one.

use alloc::string::String;
use alloc::vec::Vec;

use super::op::IntrinsicId;
use crate::types::Dtype;

crate::declare_tagged_enum! {
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
        /// The KV pages each request may address, as WORKING-SET-RELATIVE
        /// indexes — the space `kv-working-set` hands a guest and the only
        /// space a guest ever holds, because a relative index survives the
        /// copy-on-write that moves the physical page under it. Whoever
        /// resolves this port translates it through the working set's flat
        /// table before it becomes an address: the runtime for a host-folded
        /// value, the engine for one it reads off a channel. Read.
        Pages = 3, "pages";
        /// Row offsets splitting `pages` per request. Read.
        PageIndptr = 4, "page_indptr";
        /// Per-request readable KV extent after this pass's writes land.
        /// Read.
        KvLen = 5, "kv_len";
        /// **The explicit KV write descriptor's page half**: which page each
        /// token row is appended into, in [`Port::Pages`]'s space and
        /// translated with it. It is stated rather than derived because a
        /// derivation cannot spell a write that is not the page run's tail —
        /// several lanes appending into one shared pool each need their own
        /// cell, and `have + row` names one cell for all of them. Taken.
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
        // ── Recurrent-state buffered-slot family. Wire-additive: tags 0-9
        // stay unmoved so a pure-attention guest's container is
        // byte-identical. RESERVED rather than reclaimed: renumbering would
        // silently change the meaning of already-compiled containers.
        // `RsBufferLen` inverts direction — the device writes the live
        // buffered token count and the host reads it as an upper bound
        // (`FireGeometry::rs_buffer_lens`), rather than the guest stating it.
        /// RESERVED. Page tables for the recurrent-state buffered slots.
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
    ///
    /// `RsFoldLen` also consumes: it is a count spent by the fire that folds
    /// on it, not state a later fire re-reads. Peeking it would let a device
    /// handoff (fire A computes the count, fire B folds on it) make fire A
    /// implicitly take its own not-yet-produced output, deadlocking the
    /// launch.
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

/// A SET of descriptor ports, as one word.
///
/// **THE PORTS CAME HOME** (palo design §7, decision 19). This set used to be
/// spelled `PIE_DEVICE_PORT_*` in `engine`: thirteen `u32` bit constants
/// in a second, private numbering that agreed with [`Port`]'s wire tags
/// nowhere — `PIE_DEVICE_PORT_PAGES` was bit 1 while [`Port::Pages`] is tag 3,
/// and `EmbedIndptr` and `Readout` had no bit at all. Two numberings for one
/// registry is a translation table somebody has to keep, and the only way to
/// find out it drifted is a fire that binds the wrong buffer.
///
/// So the set is built from the tags themselves: bit `p as u8` is port `p`.
/// There is one numbering, it is the container's, and a port added to [`Port`]
/// gets a bit for free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PortMask(u32);

impl PortMask {
    /// The empty set.
    pub const NONE: PortMask = PortMask(0);

    /// The three ports a **decode envelope** resolves: the token ids, their
    /// positions, and each request's readable KV extent. The narrow contract —
    /// a device that serves this and no more can run a decode step, because a
    /// decode step's geometry is one row per request and the page table is the
    /// host's.
    pub const DECODE_ENVELOPE: PortMask = PortMask::of(&[
        Port::EmbedTokens,
        Port::Positions,
        Port::KvLen,
    ]);

    /// The seven ports a **device-resolved geometry** resolves: the decode
    /// envelope plus the page table, the row split, and the adapter routing.
    /// A device that serves this set derives the whole fire geometry itself
    /// and the host stages nothing per fire but the descriptor.
    pub const DEVICE_GEOMETRY: PortMask = PortMask::of(&[
        Port::EmbedTokens,
        Port::Positions,
        Port::KvLen,
        Port::Pages,
        Port::PageIndptr,
        Port::WSlot,
        Port::WOff,
    ]);

    /// The recurrent-state buffered-slot family (tags 10-14) — RESERVED, in
    /// the same sense [`Port`]'s own doc comments mean it.
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

/// How much of a fire's geometry the device resolves for itself.
///
/// Three points on one axis — how far the descriptor ports reach — and it came
/// home with them (decision 19). It was `engine`'s `GeometryClass` beside
/// a `PIE_GEOMETRY_CLASS_*` triple of `u32`s that a `const` assertion held in
/// step with it; here the classes ARE the port sets they name, and
/// [`GeometryClass::ports`] is the one place the correspondence is written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GeometryClass {
    /// The host resolves everything and the device binds no descriptor port.
    #[default]
    Host,
    /// The device resolves a decode step's envelope
    /// ([`PortMask::DECODE_ENVELOPE`]) and the host still owns the page table.
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

    /// The most demanding class `served` can carry, widest first.
    ///
    /// Total: [`GeometryClass::Host`] asks for nothing, so it is always an
    /// answer.
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

/// Where a configuration sink's effect is consumed — drives the T11
/// stage-precedence check: a sink call is legal only at a stage strictly
/// preceding its consumption point (pass-wide ⇒ prologue only;
/// attention-scoped ⇒ prologue (all layers) or `on_attn_proj` (that layer)).
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

/// **THE SCORE ROW'S KV CEILING — A STATUTE, NOT A CONSTITUTION**
/// (`.wiki/alto/attn-score.md` §4: "Sampling cadence, window width, layer
/// sets: statutes, not constitution").
///
/// Every [`IntrinsicId::AttnScore`] row is this many F32 slots wide: the
/// live prefix carries the mass and everything past it is exactly `0.0`. It
/// is a CONSTANT rather than a profile field for one reason, and the reason
/// is the guest side. A program's trace-known constants reach it through the
/// SDK's model host calls (`vocab`, `page_size`); there is no host call for
/// a score ceiling and adding one is a WIT change. A number both the author
/// and the backend must agree on, that neither can be told, has exactly one
/// honest home: a published constant in the registry both already read.
///
/// **AND THE NUMBER CARRIES ITS ARGUMENT.** The slab an engine carves for
/// this is `lanes × layers × heads × this × 4 B` — for a six-attention-layer
/// 16-head text at 256 lanes, 201 MiB at 2048 and 805 MiB at 8192. 2048 is
/// the widest ceiling that keeps the observability slab an order of
/// magnitude under a small model's own weights, which is the bar a facility
/// nobody asked for has to clear. A request whose live KV exceeds it is
/// refused by name rather than truncated: a truncated row is a distribution
/// that does not sum to one, and every consumer of this axis divides by that
/// sum.
pub const ATTN_SCORE_KV_MAX: u32 = 2048;

/// Intrinsic value scope: which stages may materialize it.
pub fn intrinsic_stages(intr: IntrinsicId) -> &'static [Stage] {
    match intr {
        IntrinsicId::Logits
        | IntrinsicId::MtpLogits
        | IntrinsicId::Hidden
        | IntrinsicId::ValueHead => &[Stage::Epilogue],
        IntrinsicId::MtpDrafts => &[Stage::Epilogue],
        // **EPILOGUE ONLY, AND THAT IS THE OBSERVABILITY CONTRACT**
        // (`.wiki/alto/attn-score.md` §4). This row used to say `OnAttn`, and
        // that spelling asked the graph to be torn open once per layer so a
        // guest could stand inside it — the third boundary palo §9 abolished.
        // What replaced it is a WRITE the graph already contained: the
        // capture arm accumulates per-key mass into a rectangle as it runs,
        // and the epilogue — a boundary that already exists — reads the whole
        // rectangle at once, every exported layer and every head of it. So
        // the scores are still exactly as fresh as the fire that produced
        // them, and no stage of a guest program runs mid-forward to get them.
        IntrinsicId::AttnScore => &[Stage::Epilogue],
        IntrinsicId::Query | IntrinsicId::Layer => &[Stage::OnAttnProj, Stage::OnAttn],
    }
}

/// Whether `profile` provides `intr`.
///
/// Deliberately one exhaustive match, not a gated-list + mapping split:
/// that shape lets a newly gated intrinsic default to available if the
/// mapping misses it — a capability check that passes by being forgotten.
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
    /// Concrete dtype `ACT` resolves to (bf16/fp8 quantized types are the
    /// backend's; the *interpreter-visible* materialization is F32).
    pub activation: Dtype,
    /// `[k, vocab]` F32 draft logits intrinsic available — a model with a
    /// multi-token-prediction head.
    pub has_mtp_logits: bool,
    /// `[k]` I32 draft tokens intrinsic ([`IntrinsicId::MtpDrafts`]) available —
    /// a model with an MTP head serving device-resident spec-decode drafts.
    pub has_mtp_drafts: bool,
    /// A scalar value-head intrinsic is available.
    pub has_value_head: bool,
    /// `[layers * heads, kv_max]` F32 per-key attention mass
    /// ([`IntrinsicId::AttnScore`]) available at the epilogue. Unlike the MTP
    /// flags this is a *backend* property as much as a model one: it needs a
    /// score-observing attention kernel, and it is refused for soft-capped or
    /// sliding-window attention, where the captured row would not be the
    /// softmax the eviction papers define.
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
    /// The lowercase name of [`ModelProfile::activation`], for a digest or a
    /// diagnostic.
    ///
    /// A method because the answer belongs to ETA and the field does not: the
    /// dtype is `dtype::Dtype`, which the loader and the kernels also name,
    /// and `types::name` is ETA's spelling of the four it computes in. The one
    /// reader is `worker`'s boot-consistency digest, which hashes these bytes
    /// and does not otherwise know this crate. It read `activation.name()`
    /// while the dtype was this crate's own enum.
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
