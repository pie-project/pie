//! Bind-time validation: the single gate every inferlet-supplied trace passes
//! before it reaches a backend, realizing the thrust-3 locked decisions —
//! SPSC endpoints (T2), the per-channel **first-op direction table** (the
//! readiness predicate input, T3/§7.1), sink stage-precedence (T11),
//! shape closure, model-gated intrinsic availability and the T10 lint
//! (overview §4/§1), plus the §7.1 in-place channel classification the
//! lowering tiers share.
//!
//! [`bind`] consumes a decoded [`TraceContainer`] + a [`ModelProfile`] and
//! returns a [`BoundTrace`] — the validated, typed artifact the reference
//! interpreter ([`super::interp`]) and the CUDA tiers execute.

use alloc::vec::Vec;
use core::fmt;

use super::container::{ChannelDecl, ExternDir, HostRole, PortSource, TraceContainer};
use super::infer::{BodyCtx, BodyError, body_types};
use super::op::{IntrinsicId, Op};
use super::registry::{
    KNOWN_SINKS, ModelProfile, Phase, Port, SinkScope, Stage, intrinsic_model_gated,
    intrinsic_stages,
};
use crate::types::{DType, Shape, ValueType};

/// Which bit a channel's first in-pass op needs (§7.1's fire-time structural
/// predicate): `take`/`read` need **full**; a leading `put` needs **empty**
/// (back-pressure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    NeedsFull,
    NeedsEmpty,
}

/// One row of the readiness table: channel × the phase owning its first
/// in-pass op × the required bit. Emitted per pass; thrust 2's wait-word
/// machinery consumes it (contract C2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessEntry {
    pub chan: u32,
    pub phase: Phase,
    pub dir: Direction,
}

/// §7.1 lowering class, computed at registration.
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
    pub container: TraceContainer,
    pub profile: ModelProfile,
    /// C3 identity: FNV-1a over the canonical container bytes.
    pub hash: u64,
    /// Program-side channel element types (`ACT` materialized).
    pub channel_types: Vec<ValueType>,
    /// Per stage (container order): the body's SSA type table.
    pub stage_types: Vec<Vec<ValueType>>,
    /// The first-op direction table (one entry per touched channel).
    pub readiness: Vec<ReadinessEntry>,
    /// §7.1 class per channel (declaration order).
    pub classes: Vec<ChannelClass>,
}

/// A bind failure.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidateError {
    /// SSA/shape/dtype error inside a stage body.
    Body {
        stage: Stage,
        err: BodyError,
    },
    /// At most one program per stage; stages sorted by tag.
    DuplicateStage(Stage),
    StagesUnsorted,
    /// Ports sorted by tag, unique.
    DuplicatePort(Port),
    PortsUnsorted,
    PortChannelOutOfRange {
        port: Port,
        chan: u32,
    },
    /// Const payload length must equal `numel × elem_size`.
    PortConstPayload {
        port: Port,
    },
    /// Channel capacity must be ≥ 1 (trace-known constructor arg).
    ZeroCapacity {
        chan: u32,
    },
    /// T2 SPSC: the host writes this channel — no stage may put.
    SecondProducer {
        chan: u32,
        stage: Stage,
    },
    /// T2 SPSC: the host reads this channel — no stage may take/read, and
    /// no port may bind it.
    SecondConsumer {
        chan: u32,
        stage: Stage,
    },
    /// T11: a sink at a stage that does not precede its consumption point
    /// (pass-wide ⇒ prologue-only; attention ⇒ prologue or attn-proj).
    SinkMisplaced {
        name_index: u16,
        stage: Stage,
    },
    /// A `SinkCall` names something the profile knows as a value kernel, or
    /// a `KernelCall` names a sink.
    SinkKernelKindMismatch {
        name_index: u16,
    },
    /// Bind-time availability (overview §4): the backend lacks this
    /// second-party name.
    KernelUnavailable {
        name_index: u16,
    },
    /// T10: the named kernel returns a time-/load-varying value.
    NotReplayable {
        name_index: u16,
    },
    /// Stage-scoped intrinsic used outside its stages (overview §5.3).
    IntrinsicWrongStage {
        intr: IntrinsicId,
        stage: Stage,
    },
    /// Model-gated intrinsic the profile lacks (overview §4).
    IntrinsicUnavailable {
        intr: IntrinsicId,
    },
    /// Declared intrinsic type violates the registry rule (e.g. `logits`
    /// must be `[n_out, vocab]` F32 for the bound model).
    IntrinsicTypeRule {
        intr: IntrinsicId,
        stage: Stage,
    },
    /// v1.1: extern table not sorted by channel / duplicate channel.
    ExternsUnsortedOrDup,
    /// v1.1: an extern channel must be device-role (`host_role = None`) and
    /// unseeded (the producing instance fills it).
    ExternDeclConflict {
        chan: u32,
    },
    /// v1.1: extern name index outside the name table.
    ExternNameOutOfRange {
        chan: u32,
    },
    /// v1.1 SPSC across the pair: a stage op (or port) on the wrong side of
    /// the extern direction (put on an Import; take/read/port on an Export).
    ExternDirViolation {
        chan: u32,
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
            DuplicatePort(p) => write!(f, "duplicate binding for port {}", p.name()),
            PortsUnsorted => f.write_str("port bindings must be sorted by port tag"),
            PortChannelOutOfRange { port, chan } => {
                write!(f, "port {}: channel index {chan} out of range", port.name())
            }
            PortConstPayload { port } => {
                write!(f, "port {}: const payload length mismatch", port.name())
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
            SinkMisplaced { name_index, stage } => write!(
                f,
                "sink (name #{name_index}) at stage {} does not precede its consumption point",
                stage.name()
            ),
            SinkKernelKindMismatch { name_index } => {
                write!(
                    f,
                    "name #{name_index}: sink/kernel kind mismatch with the profile"
                )
            }
            KernelUnavailable { name_index } => {
                write!(
                    f,
                    "name #{name_index}: backend does not provide this kernel/sink"
                )
            }
            NotReplayable { name_index } => write!(
                f,
                "name #{name_index}: time-/load-varying return — a register read in disguise (T10)"
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
pub fn channel_value_type(decl: &ChannelDecl) -> ValueType {
    ValueType::new(decl.shape, decl.dtype.program_dtype())
}

/// Validate a container against a profile; returns the typed, bound trace.
pub fn bind(container: TraceContainer, profile: ModelProfile) -> Result<BoundTrace, ValidateError> {
    // ── container-level structure ────────────────────────────────────────
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
            return Err(ValidateError::ZeroCapacity { chan: i as u32 });
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
                let expect = shape.numel() as usize * super::container::const_elem_size(*dtype);
                if data.len() != expect {
                    return Err(ValidateError::PortConstPayload { port: p.port });
                }
            }
        }
    }

    // ── v1.1 extern channels: structure + decl constraints ──────────────
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

    // ── per-stage bodies: SSA + shape/dtype ──────────────────────────────
    let channel_types: Vec<ValueType> = container.channels.iter().map(channel_value_type).collect();
    let ctx = BodyCtx {
        channel_types: &channel_types,
        n_names: container.names.len() as u16,
    };
    let mut stage_types = Vec::with_capacity(container.stages.len());
    for sp in &container.stages {
        let types = body_types(&sp.ops, &ctx).map_err(|err| ValidateError::Body {
            stage: sp.stage,
            err,
        })?;
        stage_types.push(types);
    }

    // ── intrinsics: stage scope, model gating, registry type rules ──────
    for sp in &container.stages {
        for op in &sp.ops {
            if let Op::IntrinsicVal { intr, shape, dtype } = *op {
                if !intrinsic_stages(intr).contains(&sp.stage) {
                    return Err(ValidateError::IntrinsicWrongStage {
                        intr,
                        stage: sp.stage,
                    });
                }
                if intrinsic_model_gated(intr) {
                    let ok = match intr {
                        IntrinsicId::MtpLogits => profile.has_mtp_logits,
                        IntrinsicId::MtpDrafts => profile.has_mtp_drafts,
                        IntrinsicId::ValueHead => profile.has_value_head,
                        _ => true,
                    };
                    if !ok {
                        return Err(ValidateError::IntrinsicUnavailable { intr });
                    }
                }
                if !intrinsic_type_ok(intr, shape, dtype, &profile) {
                    return Err(ValidateError::IntrinsicTypeRule {
                        intr,
                        stage: sp.stage,
                    });
                }
            }
        }
    }

    // ── second-party names: availability, T10, sink/kernel kind + T11 ───
    for sp in &container.stages {
        for op in &sp.ops {
            match op {
                Op::KernelCall { name, .. } => {
                    let n = resolve_name(&container, *name);
                    let info = profile
                        .kernel(n)
                        .ok_or(ValidateError::KernelUnavailable { name_index: *name })?;
                    if info.sink_scope.is_some() {
                        return Err(ValidateError::SinkKernelKindMismatch { name_index: *name });
                    }
                    if !info.replayable {
                        return Err(ValidateError::NotReplayable { name_index: *name });
                    }
                }
                Op::SinkCall { name, .. } => {
                    let n = resolve_name(&container, *name);
                    // First-party sinks have spec-owned scopes; second-party
                    // sinks come from the profile.
                    let scope = KNOWN_SINKS
                        .iter()
                        .find(|(k, _)| *k == n)
                        .map(|(_, s)| *s)
                        .or_else(|| profile.kernel(n).and_then(|i| i.sink_scope));
                    let scope = match scope {
                        Some(s) => s,
                        None => {
                            // Unknown name: available? (bind-time rule)
                            let info = profile
                                .kernel(n)
                                .ok_or(ValidateError::KernelUnavailable { name_index: *name })?;
                            match info.sink_scope {
                                Some(s) => s,
                                None => {
                                    return Err(ValidateError::SinkKernelKindMismatch {
                                        name_index: *name,
                                    });
                                }
                            }
                        }
                    };
                    // T11: the call must precede the consumption point.
                    let ok = match scope {
                        SinkScope::PassWide => sp.stage == Stage::Prologue,
                        SinkScope::Attention => {
                            matches!(sp.stage, Stage::Prologue | Stage::OnAttnProj)
                        }
                    };
                    if !ok {
                        return Err(ValidateError::SinkMisplaced {
                            name_index: *name,
                            stage: sp.stage,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // ── SPSC endpoints (T2; v1.1 extends across the extern pair) ─────────
    let extern_dir = |chan: u32| -> Option<ExternDir> {
        container
            .externs
            .iter()
            .find(|e| e.chan == chan)
            .map(|e| e.dir)
    };
    for sp in &container.stages {
        for op in &sp.ops {
            match *op {
                Op::ChanPut { chan, .. } => {
                    if container.channels[chan as usize].host_role == HostRole::Writer {
                        return Err(ValidateError::SecondProducer {
                            chan,
                            stage: sp.stage,
                        });
                    }
                    if extern_dir(chan) == Some(ExternDir::Import) {
                        // The peer instance is the producer — a local put is a
                        // second producer endpoint.
                        return Err(ValidateError::ExternDirViolation {
                            chan,
                            stage: sp.stage,
                        });
                    }
                }
                Op::ChanTake(chan) | Op::ChanRead(chan) => {
                    if container.channels[chan as usize].host_role == HostRole::Reader {
                        return Err(ValidateError::SecondConsumer {
                            chan,
                            stage: sp.stage,
                        });
                    }
                    if extern_dir(chan) == Some(ExternDir::Export) {
                        // The peer instance is the consumer — a local take/read
                        // is a second consumer endpoint.
                        return Err(ValidateError::ExternDirViolation {
                            chan,
                            stage: sp.stage,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    // Ports consume too: an Export channel bound to a port would make this
    // pass a second consumer of its own export.
    for p in &container.ports {
        if let PortSource::Channel(c) = p.source {
            if extern_dir(c) == Some(ExternDir::Export) {
                return Err(ValidateError::ExternDirViolation {
                    chan: c,
                    stage: Stage::Prologue,
                });
            }
        }
    }
    // Ports are pass-side consumers too.
    for p in &container.ports {
        if let PortSource::Channel(c) = p.source {
            if container.channels[c as usize].host_role == HostRole::Reader {
                // Attribute to the descriptor; report with the epilogue tag
                // absent a stage — use Prologue? Keep a dedicated message via
                // SecondConsumer with the earliest stage marker.
                return Err(ValidateError::SecondConsumer {
                    chan: c,
                    stage: Stage::Prologue,
                });
            }
        }
    }

    // ── readiness: per-channel first-op direction (§7.1) ─────────────────
    let readiness = readiness_table(&container);

    // ── §7.1 channel classes ─────────────────────────────────────────────
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
    }
}

/// Walk the pass's phases in execution order (prologue → descriptor →
/// attn-proj → attn → epilogue) and record, per channel, the FIRST op's
/// required bit. Take/read ⇒ full; put ⇒ empty (§7.1).
pub fn readiness_table(c: &TraceContainer) -> Vec<ReadinessEntry> {
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
                        match *op {
                            Op::ChanTake(chan) | Op::ChanRead(chan) => {
                                visit(chan, phase, Direction::NeedsFull, &mut seen)
                            }
                            Op::ChanPut { chan, .. } => {
                                visit(chan, phase, Direction::NeedsEmpty, &mut seen)
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    out
}

/// §7.1 in-place classification, computed at registration:
/// host-visible ⇒ full ring; device-private **linear** `take`→`put` (the
/// taken value flows into exactly one put on the same channel) ⇒ in-place —
/// without undo when no fallible stage follows the mutating stage, with
/// row-granularity undo otherwise; anything else ⇒ full ring (always safe).
///
/// A stage is *fallible* here when it owns the first op of a host-coupled
/// channel (a late host edge — the only readiness fire time cannot settle;
/// §7.1). The descriptor phase inherits fallibility from host-fed ports.
pub fn classify_channels(c: &TraceContainer, readiness: &[ReadinessEntry]) -> Vec<ChannelClass> {
    // Fallible phases: first-use of a host-coupled OR extern channel (an
    // extern edge crosses pipelines — fire time cannot settle it, like a
    // late host edge; §7.1).
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
        let ci = ci as u32;
        if decl.host_role != HostRole::None || is_extern(ci) {
            // Host-visible and extern channels always keep the full ring
            // (§7.1: the other endpoint peeks/commits on its own clock).
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
                match *op {
                    Op::ChanTake(ch) if ch == ci => {
                        if take.is_some() {
                            extra = true;
                        }
                        take = Some((si, oi, next_id));
                    }
                    Op::ChanRead(ch) if ch == ci => extra = true,
                    Op::ChanPut { chan, value } if chan == ci => {
                        if put.is_some() {
                            extra = true;
                        }
                        put = Some((si, oi, value));
                    }
                    _ => {}
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
        // same channel" — §1; other non-put readers are fine, cf. §6.3's
        // `hist`, whose taken value also feeds the similarity matmul).
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
    use super::*;
    use crate::container::{ChanDType, ChannelDecl, PortBinding, StageProgram};
    use crate::types::Literal;
    use alloc::string::ToString;
    use alloc::vec;

    fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role,
            seeded,
        }
    }

    /// The overview §3 shape: tok (loop), out (host-read), mask (host-fed,
    /// bool), len (counter), rng (state) + greedy-gumbel epilogue.
    fn section3() -> TraceContainer {
        let vocab = 32u32;
        let channels = vec![
            chan(Shape::vector(1), DType::I32, HostRole::None, true), // 0 tok
            chan(Shape::vector(1), DType::I32, HostRole::Reader, false), // 1 out
            chan(Shape::vector(vocab), DType::Bool, HostRole::Writer, false), // 2 mask
            chan(Shape::vector(1), DType::U32, HostRole::None, true), // 3 len
            chan(Shape::vector(2), DType::U32, HostRole::None, true), // 4 rng
        ];
        let mut ops: Vec<Op> = vec![
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, vocab),
                dtype: DType::F32,
            }, // 0
            Op::Reshape {
                value: 0,
                shape: Shape::vector(vocab),
            }, // 1
            Op::ChanTake(4), // 2 r = rng.take()
            Op::ChanTake(2), // 3 m = mask.take()
        ];
        let g = crate::expand::gumbel(&mut ops, 2, Shape::vector(vocab)); // 4
        let masked = crate::expand::mask_apply(&mut ops, 1, 3); // 5,6
        let sum = crate::expand::next_id(&ops);
        ops.push(Op::Add(masked, g)); // sum
        ops.push(Op::ReduceArgmax(sum)); // t = sum+1
        let t = sum + 1;
        // rng.put(add(r, CTR1)) — CTR1 = [0,1] not expressible as a scalar
        // const; use iota(2) (=[0,1]) as the counter increment.
        ops.push(Op::Iota { len: 2 }); // t+1
        ops.push(Op::Cast {
            value: t + 1,
            dtype: DType::U32,
        }); // t+2 (identity; keeps ids readable)
        ops.push(Op::Add(2, t + 2)); // t+3
        ops.push(Op::ChanPut {
            chan: 4,
            value: t + 3,
        });
        // tok.put(t) — argmax over [vocab] gives scalar; reshape to [1].
        ops.push(Op::Reshape {
            value: t,
            shape: Shape::vector(1),
        }); // t+4
        ops.push(Op::ChanPut {
            chan: 0,
            value: t + 4,
        });
        // len.put(len.take() + 1)
        ops.push(Op::ChanTake(3)); // t+5
        ops.push(Op::Const(Literal::U32(1))); // t+6
        ops.push(Op::Add(t + 5, t + 6)); // t+7
        ops.push(Op::ChanPut {
            chan: 3,
            value: t + 7,
        });
        // out.put(t)
        ops.push(Op::ChanPut {
            chan: 1,
            value: t + 4,
        });

        TraceContainer {
            names: vec![],
            channels,
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    },
                },
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(3),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
            externs: Vec::new(),
        }
    }

    #[test]
    fn section3_binds_and_hashes_stably() {
        let c = section3();
        let h1 = c.hash();
        let bound = bind(c.clone(), ModelProfile::dummy()).expect("bind");
        assert_eq!(bound.hash, h1);
        assert_eq!(bind(c, ModelProfile::dummy()).unwrap().hash, h1);
    }

    #[test]
    fn section3_readiness_table() {
        let c = section3();
        let b = bind(c, ModelProfile::dummy()).unwrap();
        // tok: first op = descriptor take (embed) → NeedsFull @ Descriptor.
        // out: first op = epilogue put → NeedsEmpty (back-pressure).
        // mask: epilogue take → NeedsFull. len: descriptor peek → NeedsFull.
        // rng: epilogue take → NeedsFull.
        let get = |ch: u32| b.readiness.iter().find(|e| e.chan == ch).copied().unwrap();
        assert_eq!(get(0).phase, Phase::Descriptor);
        assert_eq!(get(0).dir, Direction::NeedsFull);
        assert_eq!(get(1).dir, Direction::NeedsEmpty);
        assert_eq!(get(1).phase, Phase::Epilogue);
        assert_eq!(get(2).dir, Direction::NeedsFull);
        assert_eq!(get(3).phase, Phase::Descriptor);
        assert_eq!(get(4).dir, Direction::NeedsFull);
    }

    #[test]
    fn section3_channel_classes() {
        let c = section3();
        let b = bind(c, ModelProfile::dummy()).unwrap();
        // tok: taken by descriptor (embed) + put by epilogue → not linear in
        // one stage → FullRing. out/mask host-visible → FullRing.
        // len: descriptor peek + epilogue take→put → extra consumer → FullRing.
        // rng: pure epilogue take→put ping-pong, epilogue is last stage and
        // the epilogue itself is the only fallible stage (mask) → InPlace.
        assert_eq!(b.classes[0], ChannelClass::FullRing);
        assert_eq!(b.classes[1], ChannelClass::FullRing);
        assert_eq!(b.classes[2], ChannelClass::FullRing);
        assert_eq!(b.classes[3], ChannelClass::FullRing);
        assert_eq!(b.classes[4], ChannelClass::InPlace);
    }

    #[test]
    fn spsc_second_producer_rejected() {
        let mut c = section3();
        // Host writes `mask` (chan 2); a stage put to it is a bind error.
        c.stages[0].ops.push(Op::Const(Literal::Bool(true)));
        let id = crate::expand::next_id(&c.stages[0].ops) - 1;
        c.stages[0].ops.push(Op::Broadcast {
            value: id,
            shape: Shape::vector(32),
        });
        c.stages[0].ops.push(Op::ChanPut {
            chan: 2,
            value: id + 1,
        });
        assert!(matches!(
            bind(c, ModelProfile::dummy()),
            Err(ValidateError::SecondProducer { chan: 2, .. })
        ));
    }

    #[test]
    fn spsc_second_consumer_rejected() {
        let mut c = section3();
        // Host reads `out` (chan 1); a stage read of it is a bind error.
        c.stages[0].ops.push(Op::ChanRead(1));
        assert!(matches!(
            bind(c, ModelProfile::dummy()),
            Err(ValidateError::SecondConsumer { chan: 1, .. })
        ));
    }

    #[test]
    fn sink_precedence_t11() {
        let mut profile = ModelProfile::dummy();
        profile.kernels.push(crate::registry::KernelInfo {
            name: "lora".to_string(),
            sink_scope: Some(SinkScope::PassWide),
            replayable: true,
        });
        // lora (pass-wide) in the prologue: OK.
        let mk = |stage: Stage| TraceContainer {
            names: vec!["lora".to_string()],
            channels: vec![chan(Shape::vector(4), DType::F32, HostRole::None, true)],
            ports: vec![],
            stages: vec![StageProgram {
                stage,
                ops: vec![
                    Op::ChanRead(0),
                    Op::SinkCall {
                        name: 0,
                        args: vec![0],
                    },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        assert!(bind(mk(Stage::Prologue), profile.clone()).is_ok());
        // lora at the epilogue: nothing after it consumes → T11 error.
        assert!(matches!(
            bind(mk(Stage::Epilogue), profile),
            Err(ValidateError::SinkMisplaced { .. })
        ));
        // attn_page_mask allowed at attn-proj (that layer)…
        let apm = TraceContainer {
            names: vec!["attn_page_mask".to_string()],
            channels: vec![chan(Shape::vector(4), DType::F32, HostRole::None, true)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::OnAttnProj,
                ops: vec![
                    Op::ChanRead(0),
                    Op::PivotThreshold {
                        input: 0,
                        predicate: crate::types::Predicate::ProbGe(1),
                    },
                    Op::SinkCall {
                        name: 0,
                        args: vec![2],
                    },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        // needs a threshold operand: insert const before pivot — rebuild:
        let apm = {
            let mut c = apm;
            c.stages[0].ops = vec![
                Op::ChanRead(0),              // 0
                Op::Const(Literal::F32(0.5)), // 1
                Op::PivotThreshold {
                    input: 0,
                    predicate: crate::types::Predicate::ProbGe(1),
                }, // 2
                Op::SinkCall {
                    name: 0,
                    args: vec![2],
                },
            ];
            c
        };
        assert!(bind(apm.clone(), ModelProfile::dummy()).is_ok());
        // …but not at on_attn (post-attention).
        let mut late = apm;
        late.stages[0].stage = Stage::OnAttn;
        assert!(matches!(
            bind(late, ModelProfile::dummy()),
            Err(ValidateError::SinkMisplaced { .. })
        ));
    }

    #[test]
    fn t10_non_replayable_kernel_rejected() {
        let mut profile = ModelProfile::dummy();
        profile.kernels.push(crate::registry::KernelInfo {
            name: "gpu_load".to_string(),
            sink_scope: None,
            replayable: false,
        });
        let c = TraceContainer {
            names: vec!["gpu_load".to_string()],
            channels: vec![chan(Shape::vector(1), DType::F32, HostRole::None, true)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::KernelCall {
                        name: 0,
                        args: vec![],
                        shape: Shape::vector(1),
                        dtype: DType::F32,
                    },
                    Op::ChanTake(0),
                    Op::Add(0, 1),
                    Op::ChanPut { chan: 0, value: 2 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        assert!(matches!(
            bind(c, profile),
            Err(ValidateError::NotReplayable { name_index: 0 })
        ));
    }

    #[test]
    fn model_gated_intrinsic_rejected_when_absent() {
        let mut profile = ModelProfile::dummy();
        profile.has_mtp_logits = false;
        let c = TraceContainer {
            names: vec![],
            channels: vec![chan(Shape::vector(4), DType::I32, HostRole::Reader, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::MtpLogits,
                        shape: Shape::matrix(4, 32),
                        dtype: DType::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        assert!(matches!(
            bind(c, profile),
            Err(ValidateError::IntrinsicUnavailable {
                intr: IntrinsicId::MtpLogits
            })
        ));
    }

    #[test]
    fn intrinsic_stage_scope_enforced() {
        // logits at the prologue is out of scope.
        let c = TraceContainer {
            names: vec![],
            channels: vec![chan(Shape::vector(1), DType::I32, HostRole::Reader, false)],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape: Shape::matrix(1, 32),
                        dtype: DType::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            externs: alloc::vec::Vec::new(),
        };
        assert!(matches!(
            bind(c, ModelProfile::dummy()),
            Err(ValidateError::IntrinsicWrongStage {
                intr: IntrinsicId::Logits,
                stage: Stage::Prologue
            })
        ));
    }
}
