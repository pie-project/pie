//! Bind-time validation: the single gate every inferlet-supplied trace passes
//! before it reaches a backend. It is the one place the following are
//! decided, and nothing downstream re-decides them:
//!
//! * SPSC channel endpoints — a channel the host writes has no stage putting
//!   to it, and vice versa.
//! * The per-channel **first-op direction table**, which is what the fire-time
//!   readiness predicate is computed from.
//! * Sink stage-precedence: a sink must precede the point its value is
//!   consumed.
//! * Shape closure, so every result shape is known from the trace alone.
//! * Model-gated intrinsic availability, and the replayability lint that
//!   rejects a kernel returning a time- or load-varying value.
//! * The in-place channel classification the lowering tiers share.
//!
//! [`bind`] consumes a decoded [`TraceContainer`] plus a [`ModelProfile`] and
//! returns a [`BoundTrace`] — the validated, typed artifact the reference
//! interpreter and the CUDA tiers execute. A backend that re-derives any of
//! the above is a second validator, and the one that disagrees is the one
//! nobody tested.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::container::{ChannelDecl, ExternDir, HostRole, PortSource, TraceContainer};
use super::infer::{BodyCtx, BodyError, BodyErrorKind, body_types};
use super::op::{ChannelUse, IntrinsicId, Op};
use super::registry::{
    KNOWN_SINKS, ModelProfile, Phase, Port, SinkScope, Stage, intrinsic_available, intrinsic_stages,
};
use crate::types::{DType, Shape, ValueType};

/// Which bit a channel's first in-pass op needs (the fire-time structural
/// predicate): `take`/`read` need **full**; a leading `put` needs **empty**
/// (back-pressure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// The channel must hold a value: the first op consumes one.
    NeedsFull,
    /// The channel must have room: the first op produces one.
    NeedsEmpty,
}

/// One row of the readiness table: channel × the phase owning its first
/// in-pass op × the required bit. Emitted per pass; the wait-word
/// machinery consumes it (contract C2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessEntry {
    /// The channel this row is about, in declaration order.
    pub chan: u32,
    /// The phase owning the channel's first in-pass op.
    pub phase: Phase,
    /// Which bit that op needs before it can run.
    pub dir: Direction,
}

/// Lowering class, computed at registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelClass {
    /// Host-visible (or non-linear device use): full epoch ring.
    FullRing,
    /// Linear `take`→`put` with no fallible stage after it: in-place, no
    /// undo (predicated on the accumulated flag at stage entry).
    InPlace,
    /// Linear `take`→`put` ahead of a fallible stage: in-place with undo at
    /// mutation granularity.
    InPlaceUndo,
}

/// A validated + typed trace, bound to a model profile.
#[derive(Clone, Debug)]
pub struct BoundTrace {
    /// The program this binding validated.
    pub container: TraceContainer,
    /// The model the program was checked against; availability and type
    /// rules are only meaningful relative to it.
    pub profile: ModelProfile,
    /// C3 identity: FNV-1a over the canonical container bytes.
    pub hash: u64,
    /// Program-side channel element types (`ACT` materialized).
    pub channel_types: Vec<ValueType>,
    /// Per stage (container order): the body's SSA type table.
    pub stage_types: Vec<Vec<ValueType>>,
    /// The first-op direction table (one entry per touched channel).
    pub readiness: Vec<ReadinessEntry>,
    /// Lowering class per channel (declaration order).
    pub classes: Vec<ChannelClass>,
}

/// A bind failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ValidateError {
    /// SSA/shape/dtype error inside a stage body.
    Body {
        /// The stage the offending op sits in.
        stage: Stage,
        /// What the body check rejected.
        err: BodyError,
    },
    /// At most one program per stage; stages sorted by tag.
    DuplicateStage(Stage),
    /// Stage programs are not in tag order.
    StagesUnsorted,
    /// The name table is not sorted, or repeats an entry.
    NamesUnsortedOrDuplicate,
    /// Ports sorted by tag, unique.
    DuplicatePort(Port),
    /// Port bindings are not in tag order.
    PortsUnsorted,
    /// A port binds a channel index outside the declaration table.
    PortChannelOutOfRange {
        /// The offending port.
        port: Port,
        /// The offending channel, in declaration order.
        chan: u32,
    },
    /// Const payload length must equal `numel × elem_size`.
    PortConstPayload {
        /// The offending port.
        port: Port,
    },
    /// Every attention trace must define the post-write readable extent.
    EmbedTokensWithoutKvLen,
    /// Every attention trace must bind its complete geometry explicitly.
    EmbedTokensWithoutGeometry {
        /// The offending port.
        port: Port,
    },
    /// Channel capacity must be ≥ 1 (trace-known constructor arg).
    ZeroCapacity {
        /// The offending channel, in declaration order.
        chan: u32,
    },
    /// SPSC: the host writes this channel — no stage may put.
    SecondProducer {
        /// The offending channel, in declaration order.
        chan: u32,
        /// The stage the offending op sits in.
        stage: Stage,
    },
    /// SPSC: the host reads this channel — no stage may take/read, and
    /// no port may bind it.
    SecondConsumer {
        /// The offending channel, in declaration order.
        chan: u32,
        /// The stage the offending op sits in.
        stage: Stage,
    },
    /// A sink at a stage that does not precede its consumption point
    /// (pass-wide ⇒ prologue-only; attention ⇒ prologue or attn-proj).
    SinkMisplaced {
        /// Index of the name in the container's name table.
        name_index: u16,
        /// The name, resolved for the diagnostic.
        name: String,
        /// The stage the offending op sits in.
        stage: Stage,
    },
    /// A `SinkCall` names something the profile knows as a value kernel, or
    /// a `KernelCall` names a sink.
    SinkKernelKindMismatch {
        /// Index of the name in the container's name table.
        name_index: u16,
        /// The name, resolved for the diagnostic.
        name: String,
    },
    /// Bind-time availability: the backend lacks this
    /// second-party name.
    KernelUnavailable {
        /// Index of the name in the container's name table.
        name_index: u16,
        /// The name, resolved for the diagnostic.
        name: String,
    },
    /// The named kernel returns a time-/load-varying value.
    NotReplayable {
        /// Index of the name in the container's name table.
        name_index: u16,
        /// The name, resolved for the diagnostic.
        name: String,
    },
    /// Stage-scoped intrinsic used outside its stages.
    IntrinsicWrongStage {
        /// The offending intrinsic.
        intr: IntrinsicId,
        /// The stage the offending op sits in.
        stage: Stage,
    },
    /// Model-gated intrinsic the profile lacks.
    IntrinsicUnavailable {
        /// The offending intrinsic.
        intr: IntrinsicId,
    },
    /// Declared intrinsic type violates the registry rule (e.g. `logits`
    /// must be `[n_out, vocab]` F32 for the bound model).
    IntrinsicTypeRule {
        /// The offending intrinsic.
        intr: IntrinsicId,
        /// The stage the offending op sits in.
        stage: Stage,
    },
    /// v1.1: extern table not sorted by channel / duplicate channel.
    ExternsUnsortedOrDup,
    /// v1.1: an extern channel must be device-role (`host_role = None`) and
    /// unseeded (the producing instance fills it).
    ExternDeclConflict {
        /// The offending channel, in declaration order.
        chan: u32,
    },
    /// v1.1: extern name index outside the name table.
    ExternNameOutOfRange {
        /// The offending channel, in declaration order.
        chan: u32,
    },
    /// v1.1 SPSC across the pair: a stage op (or port) on the wrong side of
    /// the extern direction (put on an Import; take/read/port on an Export).
    ExternDirViolation {
        /// The offending channel, in declaration order.
        chan: u32,
        /// The stage the offending op sits in.
        stage: Stage,
    },
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ValidateError::*;
        match self {
            Body { stage, err } => write!(f, "stage {}: {err}", stage.name()),
            DuplicateStage(s) => write!(f, "duplicate program for stage {}", s.name()),
            StagesUnsorted => f.write_str("stage programs must be sorted by stage tag"),
            NamesUnsortedOrDuplicate => {
                f.write_str("name table must be strictly sorted and unique")
            }
            DuplicatePort(p) => write!(f, "duplicate binding for port {}", p.name()),
            PortsUnsorted => f.write_str("port bindings must be sorted by port tag"),
            PortChannelOutOfRange { port, chan } => {
                write!(f, "port {}: channel index {chan} out of range", port.name())
            }
            PortConstPayload { port } => {
                write!(f, "port {}: const payload length mismatch", port.name())
            }
            EmbedTokensWithoutKvLen => f.write_str("embed_tokens requires a kv_len port binding"),
            EmbedTokensWithoutGeometry { port } => {
                write!(
                    f,
                    "embed_tokens requires an explicit {} port binding",
                    port.name()
                )
            }
            ZeroCapacity { chan } => write!(f, "channel {chan}: capacity must be >= 1"),
            SecondProducer { chan, stage } => write!(
                f,
                "channel {chan}: SPSC violation — host is the writer but stage {} puts",
                stage.name()
            ),
            SecondConsumer { chan, stage } => write!(
                f,
                "channel {chan}: SPSC violation — host is the reader but stage {} consumes",
                stage.name()
            ),
            SinkMisplaced {
                name_index,
                name,
                stage,
            } => write!(
                f,
                "sink `{name}` (name #{name_index}) at stage {} does not precede \
                 its consumption point",
                stage.name()
            ),
            SinkKernelKindMismatch { name_index, name } => {
                write!(
                    f,
                    "`{name}` (name #{name_index}): sink/kernel kind mismatch with \
                     the profile — the program calls it as one and the backend \
                     declares it as the other"
                )
            }
            KernelUnavailable { name_index, name } => {
                write!(
                    f,
                    "`{name}` (name #{name_index}): the backend's model profile does \
                     not advertise this kernel/sink, so binding it would either \
                     fail mid-fire or — far worse — run as a silent no-op. Either \
                     the driver does not implement it for this model family, or it \
                     is implemented but switched off (on the CUDA driver \
                     `envelope_dot`, `attn_page_mask` and `attn_score` all require \
                     PIE_CUDA_KV_ENVELOPES=1)"
                )
            }
            NotReplayable { name_index, name } => write!(
                f,
                "`{name}` (name #{name_index}): time-/load-varying return — a \
                 register read in disguise"
            ),
            IntrinsicWrongStage { intr, stage } => {
                write!(
                    f,
                    "intrinsic {} not in scope at stage {}",
                    intr.name(),
                    stage.name()
                )
            }
            IntrinsicUnavailable { intr } => {
                write!(
                    f,
                    "model-gated intrinsic {} unavailable on this model",
                    intr.name()
                )
            }
            IntrinsicTypeRule { intr, stage } => write!(
                f,
                "intrinsic {} at stage {}: declared type violates the registry rule",
                intr.name(),
                stage.name()
            ),
            ExternsUnsortedOrDup => f.write_str("extern table must be sorted by channel, unique"),
            ExternDeclConflict { chan } => write!(
                f,
                "extern channel {chan}: must be host_role=none and unseeded (the peer instance fills it)"
            ),
            ExternNameOutOfRange { chan } => {
                write!(f, "extern channel {chan}: name index out of range")
            }
            ExternDirViolation { chan, stage } => write!(
                f,
                "extern channel {chan}: stage {} op violates the extern direction \
                 (import ⇒ consume-only, export ⇒ produce-only — SPSC across the pair)",
                stage.name()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ValidateError {}

/// Program-side view of a channel's element type.
pub(crate) fn channel_value_type(decl: &ChannelDecl) -> ValueType {
    ValueType::new(decl.shape, decl.dtype.program_dtype())
}

/// Validate a container against a profile; returns the typed, bound trace.
pub fn bind(container: TraceContainer, profile: ModelProfile) -> Result<BoundTrace, ValidateError> {
    // One pass per rule group, in the order the rules depend on each other.
    // The order is not cosmetic: `check_spsc_endpoints` reads a channel decl by
    // the index an op carries, and `check_bodies` is what proved that index is
    // in range. Reading a channel decl as a bare
    // `container.channels[chan as usize]` hides that dependency: the index is
    // safe only because a check hundreds of lines away already ran, and
    // nothing in the expression says so. Going through the accessor makes it
    // return the same error `check_bodies` would have, so reordering these
    // calls costs a wrong *message* rather than a panic on decoded bytes.
    check_structure(&container)?;
    check_externs(&container)?;
    let channel_types: Vec<ValueType> = container.channels.iter().map(channel_value_type).collect();
    let stage_types = check_bodies(&container, &channel_types)?;
    check_intrinsics(&container, &profile)?;
    check_second_party_names(&container, &profile)?;
    check_spsc_endpoints(&container)?;

    let readiness = readiness_table(&container);
    let classes = classify_channels(&container, &readiness);

    let hash = container.hash();
    Ok(BoundTrace {
        container,
        profile,
        hash,
        channel_types,
        stage_types,
        readiness,
        classes,
    })
}

/// The channel decl an op names, or the error [`check_bodies`] would have
/// raised for it.
///
/// Every caller runs after [`check_bodies`], so the `None` arm is unreachable.
/// It is kept anyway: "in range because a different function ran first" is not
/// something the reader of a slice index can see, and it stops being true the
/// moment the passes are reordered. Total instead, at the cost of one
/// `match`.
fn channel_decl(
    container: &TraceContainer,
    chan: u32,
    stage: Stage,
    op_index: u32,
) -> Result<&ChannelDecl, ValidateError> {
    container
        .channels
        .get(chan as usize)
        .ok_or(ValidateError::Body {
            stage,
            err: BodyError {
                op_index,
                kind: BodyErrorKind::ChannelOutOfRange(chan),
            },
        })
}

/// Container-level shape: sorted/unique tables, non-zero capacities, port
/// sources in range, and the geometry ports `embed_tokens` implies.
///
/// Establishes that every `PortSource::Channel` index is in range, which
/// [`check_spsc_endpoints`] relies on.
fn check_structure(container: &TraceContainer) -> Result<(), ValidateError> {
    if container.names.windows(2).any(|names| names[0] >= names[1]) {
        return Err(ValidateError::NamesUnsortedOrDuplicate);
    }
    for w in container.stages.windows(2) {
        if w[0].stage == w[1].stage {
            return Err(ValidateError::DuplicateStage(w[0].stage));
        }
        if w[0].stage > w[1].stage {
            return Err(ValidateError::StagesUnsorted);
        }
    }
    for w in container.ports.windows(2) {
        if w[0].port == w[1].port {
            return Err(ValidateError::DuplicatePort(w[0].port));
        }
        if w[0].port > w[1].port {
            return Err(ValidateError::PortsUnsorted);
        }
    }
    for (i, ch) in container.channels.iter().enumerate() {
        if ch.capacity == 0 {
            return Err(ValidateError::ZeroCapacity {
                chan: u32::try_from(i).unwrap_or(u32::MAX),
            });
        }
    }
    for p in &container.ports {
        match &p.source {
            PortSource::Channel(c) => {
                if *c as usize >= container.channels.len() {
                    return Err(ValidateError::PortChannelOutOfRange {
                        port: p.port,
                        chan: *c,
                    });
                }
            }
            PortSource::Const { dtype, shape, data } => {
                // Computed in u64 and compared by widening `data.len()`, not
                // by narrowing the element count: `usize` is 32 bits on the
                // wasm guest, so a shape whose `numel` leaves `u32` would wrap
                // to a small expected size that a short payload then matches.
                let elem_size = super::container::const_elem_size(*dtype) as u64;
                let expect = shape.numel().checked_mul(elem_size);
                if expect != Some(data.len() as u64) {
                    return Err(ValidateError::PortConstPayload { port: p.port });
                }
            }
        }
    }
    let has_embed = container
        .ports
        .iter()
        .any(|binding| binding.port == Port::EmbedTokens);
    if has_embed
        && !container
            .ports
            .iter()
            .any(|binding| binding.port == Port::KvLen)
    {
        return Err(ValidateError::EmbedTokensWithoutKvLen);
    }
    if has_embed {
        for port in [
            Port::Positions,
            Port::Pages,
            Port::PageIndptr,
            Port::WSlot,
            Port::WOff,
        ] {
            if !container.ports.iter().any(|binding| binding.port == port) {
                return Err(ValidateError::EmbedTokensWithoutGeometry { port });
            }
        }
    }
    Ok(())
}

/// v1.1 extern channels: sorted and unique, and each one declared in a way that
/// leaves the endpoint free for its peer.
fn check_externs(container: &TraceContainer) -> Result<(), ValidateError> {
    for w in container.externs.windows(2) {
        if w[0].chan >= w[1].chan {
            return Err(ValidateError::ExternsUnsortedOrDup);
        }
    }
    for e in &container.externs {
        let Some(decl) = container.channels.get(e.chan as usize) else {
            return Err(ValidateError::ExternDeclConflict { chan: e.chan });
        };
        if decl.host_role != HostRole::None || decl.seeded {
            return Err(ValidateError::ExternDeclConflict { chan: e.chan });
        }
        if e.name as usize >= container.names.len() {
            return Err(ValidateError::ExternNameOutOfRange { chan: e.chan });
        }
    }
    Ok(())
}

/// Per-stage bodies: SSA numbering, shapes and dtypes.
///
/// Establishes that every channel and name index an op carries is in range —
/// the precondition the passes below are written against.
fn check_bodies(
    container: &TraceContainer,
    channel_types: &[ValueType],
) -> Result<Vec<Vec<ValueType>>, ValidateError> {
    let ctx = BodyCtx {
        channel_types,
        n_names: u32::try_from(container.names.len()).unwrap_or(u32::MAX),
    };
    let mut stage_types = Vec::with_capacity(container.stages.len());
    for sp in &container.stages {
        let types = body_types(&sp.ops, &ctx).map_err(|err| ValidateError::Body {
            stage: sp.stage,
            err,
        })?;
        stage_types.push(types);
    }
    Ok(stage_types)
}

/// Intrinsics: stage scope, model gating, and the registry's type rules.
fn check_intrinsics(
    container: &TraceContainer,
    profile: &ModelProfile,
) -> Result<(), ValidateError> {
    for sp in &container.stages {
        for op in &sp.ops {
            if let Op::IntrinsicVal { intr, shape, dtype } = *op {
                if !intrinsic_stages(intr).contains(&sp.stage) {
                    return Err(ValidateError::IntrinsicWrongStage {
                        intr,
                        stage: sp.stage,
                    });
                }
                if !intrinsic_available(intr, profile) {
                    return Err(ValidateError::IntrinsicUnavailable { intr });
                }
                if !intrinsic_type_ok(intr, shape, dtype, profile) {
                    return Err(ValidateError::IntrinsicTypeRule {
                        intr,
                        stage: sp.stage,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Second-party names: availability, replayability, and the sink/kernel
/// kind and placement rules.
///
/// Runs after [`check_bodies`], which is what puts every `name` index in range.
fn check_second_party_names(
    container: &TraceContainer,
    profile: &ModelProfile,
) -> Result<(), ValidateError> {
    for sp in &container.stages {
        for op in &sp.ops {
            match op {
                Op::KernelCall { name, .. } => {
                    let n = resolve_name(container, *name);
                    let info =
                        profile
                            .kernel(n)
                            .ok_or_else(|| ValidateError::KernelUnavailable {
                                name_index: *name,
                                name: n.into(),
                            })?;
                    if info.sink_scope.is_some() {
                        return Err(ValidateError::SinkKernelKindMismatch {
                            name_index: *name,
                            name: n.into(),
                        });
                    }
                    if !info.replayable {
                        return Err(ValidateError::NotReplayable {
                            name_index: *name,
                            name: n.into(),
                        });
                    }
                }
                Op::SinkCall { name, .. } => {
                    let n = resolve_name(container, *name);
                    // First-party sinks have spec-owned scopes; second-party
                    // sinks come from the profile.
                    // A first-party sink name always type-checks, so the
                    // backend's ability to HONOUR it has to be checked here or
                    // not at all: a program emitting `attn_page_mask` against a
                    // driver that cannot compact its page table would bind
                    // cleanly and then either fail mid-fire or -- far worse --
                    // run as a silent no-op whose eviction policy never
                    // evicts anything.
                    if n == "attn_page_mask" && !profile.has_attn_page_mask {
                        return Err(ValidateError::KernelUnavailable {
                            name_index: *name,
                            name: n.into(),
                        });
                    }
                    let scope = KNOWN_SINKS
                        .iter()
                        .find(|(k, _)| *k == n)
                        .map(|(_, s)| *s)
                        .or_else(|| profile.kernel(n).and_then(|i| i.sink_scope));
                    let scope = match scope {
                        Some(s) => s,
                        None => {
                            // Unknown name: available? (bind-time rule)
                            let info = profile.kernel(n).ok_or_else(|| {
                                ValidateError::KernelUnavailable {
                                    name_index: *name,
                                    name: n.into(),
                                }
                            })?;
                            match info.sink_scope {
                                Some(s) => s,
                                None => {
                                    return Err(ValidateError::SinkKernelKindMismatch {
                                        name_index: *name,
                                        name: n.into(),
                                    });
                                }
                            }
                        }
                    };
                    // The call must precede the consumption point.
                    let ok = match scope {
                        SinkScope::PassWide => sp.stage == Stage::Prologue,
                        SinkScope::Attention => {
                            matches!(sp.stage, Stage::Prologue | Stage::OnAttnProj)
                        }
                    };
                    if !ok {
                        return Err(ValidateError::SinkMisplaced {
                            name_index: *name,
                            name: n.into(),
                            stage: sp.stage,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// SPSC endpoints, extended across the extern pair: at most one
/// producer and one consumer per channel, counting the host role and the
/// descriptor ports as endpoints.
///
/// Runs after [`check_structure`] and [`check_bodies`]; see [`channel_decl`].
fn check_spsc_endpoints(container: &TraceContainer) -> Result<(), ValidateError> {
    let extern_dir = |chan: u32| -> Option<ExternDir> {
        container
            .externs
            .iter()
            .find(|e| e.chan == chan)
            .map(|e| e.dir)
    };
    for sp in &container.stages {
        for (op_index, op) in sp.ops.iter().enumerate() {
            let Some((use_, chan)) = op.channel_use() else {
                continue;
            };
            // Which endpoint a local op occupies, and therefore which peer it
            // would be a second copy of.
            let (local_role, peer_dir) = match use_ {
                ChannelUse::Put => (HostRole::Writer, ExternDir::Import),
                // A non-consuming read still occupies the consumer endpoint.
                ChannelUse::Take | ChannelUse::Read => (HostRole::Reader, ExternDir::Export),
            };
            let decl = channel_decl(
                container,
                chan,
                sp.stage,
                u32::try_from(op_index).unwrap_or(u32::MAX),
            )?;
            if decl.host_role == local_role {
                return Err(if use_ == ChannelUse::Put {
                    ValidateError::SecondProducer {
                        chan,
                        stage: sp.stage,
                    }
                } else {
                    ValidateError::SecondConsumer {
                        chan,
                        stage: sp.stage,
                    }
                });
            }
            if extern_dir(chan) == Some(peer_dir) {
                return Err(ValidateError::ExternDirViolation {
                    chan,
                    stage: sp.stage,
                });
            }
        }
    }
    // Ports consume too: an Export channel bound to a port would make this
    // pass a second consumer of its own export.
    for p in &container.ports {
        if let PortSource::Channel(c) = p.source
            && extern_dir(c) == Some(ExternDir::Export)
        {
            return Err(ValidateError::ExternDirViolation {
                chan: c,
                stage: Stage::Prologue,
            });
        }
    }
    // Ports are pass-side consumers too.
    for p in &container.ports {
        if let PortSource::Channel(c) = p.source
            && channel_decl(container, c, Stage::Prologue, 0)?.host_role == HostRole::Reader
        {
            // Attribute to the descriptor; report with the epilogue tag
            // absent a stage — use Prologue? Keep a dedicated message via
            // SecondConsumer with the earliest stage marker.
            return Err(ValidateError::SecondConsumer {
                chan: c,
                stage: Stage::Prologue,
            });
        }
    }
    Ok(())
}

fn resolve_name(c: &TraceContainer, idx: u16) -> &str {
    c.names.get(idx as usize).map(|s| s.as_str()).unwrap_or("")
}

/// Registry type rules for the stage-scoped intrinsics, against the bound
/// model. `query` is deliberately loose (backend-shaped): rank ≥ 1, F32.
fn intrinsic_type_ok(
    intr: IntrinsicId,
    shape: Shape,
    dtype: DType,
    profile: &ModelProfile,
) -> bool {
    match intr {
        IntrinsicId::Logits | IntrinsicId::MtpLogits => {
            dtype == DType::F32
                && shape.rank() == 2
                && shape.dims()[1] == profile.vocab
                && shape.dims()[0] >= 1
        }
        IntrinsicId::Hidden => dtype == DType::F32 && shape.rank() == 2 && shape.dims()[0] >= 1,
        IntrinsicId::ValueHead => dtype == DType::F32 && shape.rank() == 1,
        IntrinsicId::Query => dtype == DType::F32 && shape.rank() >= 1,
        IntrinsicId::Layer => dtype == DType::U32 && shape.is_scalar(),
        // `[k]` I32 draft tokens (k = row count, trace-known).
        IntrinsicId::MtpDrafts => dtype == DType::I32 && shape.rank() == 1 && shape.dims()[0] >= 1,
        // `[kv_max]` — the program's own KV ceiling, exactly like
        // `envelope_dot`'s `p_max`. The head axis is folded by the backend
        // (see `registry::intrinsic_stages`), and the live `kv_len` is not
        // trace-known, so the declared extent is a ceiling the backend
        // zero-pads to rather than a shape it must match.
        IntrinsicId::AttnScore => dtype == DType::F32 && shape.rank() == 1 && shape.dims()[0] >= 1,
    }
}

/// Walk the pass's phases in execution order (prologue → descriptor →
/// attn-proj → attn → epilogue) and record, per channel, the FIRST op's
/// required bit. Take/read ⇒ full; put ⇒ empty.
pub(crate) fn readiness_table(c: &TraceContainer) -> Vec<ReadinessEntry> {
    let mut seen: Vec<bool> = alloc::vec![false; c.channels.len()];
    let mut out = Vec::new();
    let mut visit = |chan: u32, phase: Phase, dir: Direction, seen: &mut Vec<bool>| {
        let i = chan as usize;
        if i < seen.len() && !seen[i] {
            seen[i] = true;
            out.push(ReadinessEntry { chan, phase, dir });
        }
    };
    let stage_prog = |s: Stage| c.stages.iter().find(|p| p.stage == s);
    for phase in Phase::ORDER {
        match phase {
            Phase::Descriptor => {
                for p in &c.ports {
                    if let PortSource::Channel(chan) = p.source {
                        visit(chan, phase, Direction::NeedsFull, &mut seen);
                    }
                }
            }
            _ => {
                let stage = match phase {
                    Phase::Prologue => Stage::Prologue,
                    Phase::OnAttnProj => Stage::OnAttnProj,
                    Phase::OnAttn => Stage::OnAttn,
                    Phase::Epilogue => Stage::Epilogue,
                    Phase::Descriptor => unreachable!(),
                };
                if let Some(sp) = stage_prog(stage) {
                    for op in &sp.ops {
                        if let Some((use_, chan)) = op.channel_use() {
                            let direction = match use_ {
                                ChannelUse::Take | ChannelUse::Read => Direction::NeedsFull,
                                ChannelUse::Put => Direction::NeedsEmpty,
                            };
                            visit(chan, phase, direction, &mut seen);
                        }
                    }
                }
            }
        }
    }
    out
}

/// In-place classification, computed at registration:
/// host-visible ⇒ full ring; device-private **linear** `take`→`put` (the
/// taken value flows into exactly one put on the same channel) ⇒ in-place —
/// without undo when no fallible stage follows the mutating stage, with
/// row-granularity undo otherwise; anything else ⇒ full ring (always safe).
///
/// A stage is *fallible* here when it owns the first op of a host-coupled
/// channel (a late host edge — the only readiness fire time cannot settle).
/// The descriptor phase inherits fallibility from host-fed ports.
pub(crate) fn classify_channels(
    c: &TraceContainer,
    readiness: &[ReadinessEntry],
) -> Vec<ChannelClass> {
    // Fallible phases: first-use of a host-coupled OR extern channel (an
    // extern edge crosses pipelines — fire time cannot settle it, like a
    // late host edge).
    let is_extern = |chan: u32| c.externs.iter().any(|e| e.chan == chan);
    let mut fallible = [false; 5];
    for e in readiness {
        if c.channels[e.chan as usize].host_role != HostRole::None || is_extern(e.chan) {
            let pi = Phase::ORDER.iter().position(|p| *p == e.phase).unwrap();
            fallible[pi] = true;
        }
    }

    let mut classes = Vec::with_capacity(c.channels.len());
    'chan: for (ci, decl) in c.channels.iter().enumerate() {
        let ci = u32::try_from(ci).unwrap_or(u32::MAX);
        if decl.host_role != HostRole::None || is_extern(ci) {
            // Host-visible and extern channels always keep the full ring
            // (the other endpoint peeks/commits on its own clock).
            classes.push(ChannelClass::FullRing);
            continue;
        }
        // Gather (stage, op-index) of takes and puts of this channel.
        let mut take: Option<(usize, usize, u32)> = None; // (stage idx, op idx, value id)
        let mut put: Option<(usize, usize, u32)> = None; // (stage idx, op idx, put value)
        let mut extra = false;
        for (si, sp) in c.stages.iter().enumerate() {
            let mut next_id = 0u32;
            for (oi, op) in sp.ops.iter().enumerate() {
                if let Some((use_, ch)) = op.channel_use()
                    && ch == ci
                {
                    match use_ {
                        ChannelUse::Take => {
                            if take.is_some() {
                                extra = true;
                            }
                            take = Some((si, oi, next_id));
                        }
                        // A peek is a second reader, so never linear.
                        ChannelUse::Read => extra = true,
                        ChannelUse::Put => {
                            if put.is_some() {
                                extra = true;
                            }
                            if let Op::ChanPut { value, .. } = *op {
                                put = Some((si, oi, value));
                            }
                        }
                    }
                }
                next_id += op.result_count();
            }
        }
        // Descriptor peeks count as extra consumers (not linear).
        for p in &c.ports {
            if matches!(p.source, PortSource::Channel(ch) if ch == ci) {
                extra = true;
            }
        }
        let (Some((tsi, toi, tid)), Some((psi, poi, pval)), false) = (take, put, extra) else {
            classes.push(ChannelClass::FullRing);
            continue;
        };
        // Same stage, take before put, and the put's value depends on the
        // taken value ("the taken value flows into exactly one put on the
        // same channel"). Other non-put readers are fine: a taken value may
        // also feed, say, a similarity matmul without costing the channel its
        // in-place class.
        if tsi != psi || toi >= poi {
            classes.push(ChannelClass::FullRing);
            continue;
        }
        let ops = &c.stages[tsi].ops;
        // Reachability: does `pval` depend on `tid`?
        let mut reach =
            alloc::vec![false; ops.iter().map(|o| o.result_count()).sum::<u32>() as usize];
        if (tid as usize) < reach.len() {
            reach[tid as usize] = true;
        }
        let mut next_id = 0u32;
        for op in ops.iter() {
            let dep = op
                .operands()
                .iter()
                .any(|&v| reach.get(v as usize).copied().unwrap_or(false));
            for r in 0..op.result_count() {
                let id = (next_id + r) as usize;
                if dep {
                    reach[id] = true;
                }
            }
            next_id += op.result_count();
        }
        if !reach.get(pval as usize).copied().unwrap_or(false) {
            classes.push(ChannelClass::FullRing);
            continue 'chan;
        }
        // Fallible-stage analysis: any fallible phase strictly after the
        // mutating stage's phase?
        let mstage = c.stages[tsi].stage;
        let mpi = Phase::ORDER
            .iter()
            .position(|p| *p == Phase::of_stage(mstage))
            .unwrap();
        let followed_by_fallible = fallible[mpi + 1..].iter().any(|&f| f);
        classes.push(if followed_by_fallible {
            ChannelClass::InPlaceUndo
        } else {
            ChannelClass::InPlace
        });
    }
    classes
}

#[cfg(test)]
mod tests {
    //! The one check that has to see inside.
    //!
    //! Everything reachable through [`bind`] is tested from outside, in
    //! `tests/validate.rs`. This stays here because it calls a pass directly,
    //! which is the whole point of it.

    use super::*;
    use crate::container::{ChanDType, ChannelDecl, PortBinding, StageProgram};
    use alloc::vec;

    /// A `ChanPut` naming a channel that does not exist.
    ///
    /// `check_bodies` rejects it, so `check_spsc_endpoints` never sees one in
    /// production. This asks for the other half: that the pass which *would*
    /// have indexed the table blind also answers, and answers the same thing.
    /// Before the split that was `container.channels[chan as usize]` three
    /// hundred lines below the check that made it safe — a panic waiting on
    /// whoever moved the blocks around.
    #[test]
    fn a_pass_that_runs_second_still_answers_when_run_first() {
        let mut container = TraceContainer {
            names: vec![],
            channels: vec![ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(DType::I32),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            }],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![Op::Iota { len: 1 }, Op::ChanPut { chan: 7, value: 0 }],
            }],
            externs: vec![],
        };

        let direct = check_spsc_endpoints(&container).unwrap_err();
        let whole = bind(container.clone(), ModelProfile::dummy()).unwrap_err();
        for err in [&direct, &whole] {
            assert!(
                matches!(
                    err,
                    ValidateError::Body {
                        err: BodyError {
                            kind: BodyErrorKind::ChannelOutOfRange(7),
                            ..
                        },
                        ..
                    }
                ),
                "out-of-range channel should read the same from either pass, got {err:?}"
            );
        }

        // And the same for a descriptor port, which is the second raw index.
        container.stages[0].ops.pop();
        container.ports = vec![PortBinding {
            port: Port::EmbedTokens,
            source: PortSource::Channel(7),
        }];
        assert!(matches!(
            check_spsc_endpoints(&container).unwrap_err(),
            ValidateError::Body {
                err: BodyError {
                    kind: BodyErrorKind::ChannelOutOfRange(7),
                    ..
                },
                ..
            }
        ));
    }
}
