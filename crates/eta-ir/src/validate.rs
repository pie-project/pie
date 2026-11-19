use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::container::{ChannelDecl, ExternDir, HostRole, PortSource, TraceContainer};
use super::infer::{BodyCtx, BodyError, BodyErrorKind, body_types};
use super::op::{ChannelUse, Op};
use super::registry::{Axis, EngineCaps, KNOWN_SINKS, OutputDecl, Phase, Port, Stage};
use crate::types::{Dtype, Shape, ValueType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Direction {
    NeedsFull,
    NeedsEmpty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessEntry {
    pub chan: u32,
    pub phase: Phase,
    pub dir: Direction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelClass {
    FullRing,
    InPlace,
    InPlaceUndo,
}

#[derive(Clone, Debug)]
pub struct BoundTrace {
    pub container: TraceContainer,
    pub lanes: Vec<Vec<OutputDecl>>,
    pub hash: u64,
    pub channel_types: Vec<ValueType>,
    pub stage_types: Vec<Vec<ValueType>>,
    pub readiness: Vec<ReadinessEntry>,
    pub classes: Vec<ChannelClass>,
    /// What the bind held the program to ([`bind_within`]); `None` for a
    /// [`bind`] that states none (an engine's own rebind of a program the
    /// runtime already held to it, or a test's), which nothing evaluates on
    /// the host.
    pub budget: Option<Budget>,
}

/// What a bind may take of the device and load serving it
/// ([`bind_within`]), derived from what they state: nothing here is
/// picked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    /// The bytes one instance's state room holds on the device serving the
    /// program: its pool's state chunks less what the engine holds back of
    /// them. A stage's values are laid in that room on the device, and the
    /// host's evaluation of a stage (`eta_compiler::eval::pareval`) holds
    /// every value of it at once: a stage whose values together pass it
    /// can bind nowhere, and is refused at its registration.
    pub room_bytes: u64,
    /// The device's stated bound on one submission and the rates its load
    /// measured, where it states one: a stage the rates cost past the bound
    /// would run long enough for the driver to fault the submission, and
    /// every tenant's fire in it, so it is refused at its registration.
    /// `None` where the device states no bound.
    pub submit: Option<SubmitBound>,
}

/// What a guest stage's time is held to ([`Budget::submit`]): the device's
/// stated bound and the rates the load measured, none of it picked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubmitBound {
    /// The device's stated bound on one submission, ns.
    pub bound_ns: u64,
    /// What the load measured one launch takes, ns.
    pub launch_ns: f64,
    /// What the load measured an elementwise pass moves, bytes a ns.
    pub elementwise_bytes_per_ns: f64,
}

impl SubmitBound {
    /// What a stage of `launches` launches touching `work` bytes takes at
    /// these rates, ns ([`StageFootprint`]): its launches' fixed cost and
    /// its bytes at the elementwise rate.
    #[must_use]
    pub fn cost_ns(&self, launches: u64, work: u64) -> f64 {
        launches as f64 * self.launch_ns + work as f64 / self.elementwise_bytes_per_ns
    }
}

/// What one stage of a program takes as it declares itself: the bytes of
/// every value it defines, as the host's evaluation holds them all at
/// once; and what it does on a device, the launches its ops take and the
/// bytes they touch ([`op_work`]), which a device's rates cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageFootprint {
    pub stage: Stage,
    pub bytes: u64,
    pub launches: u64,
    pub work: u64,
}

/// Whether `op` takes a launch of its own on a device that steps a stage
/// op by op (the closed set's steps, which the engines stating a submit
/// bound run): every op but a channel's or an output's read, a constant
/// and a reshape, which the host stages or aliases.
fn launches(op: &Op) -> bool {
    !matches!(
        op,
        Op::ChanTake(_)
            | Op::ChanRead(_)
            | Op::ChanPut { .. }
            | Op::Output { .. }
            | Op::Const(_)
            | Op::Reshape { .. }
            | Op::SinkCall { .. }
    )
}

/// The bytes `op` touches as the closed set's steps run it (the WGSL/SPIR-V
/// runtime of the engines stating a submit bound), its operands of `types`
/// and its results `results`: each operand read and each result written
/// once, but for the ops whose steps loop past that: a matmul reads a row
/// and a column for every result (`m·k·n`), a scatter visits every index
/// for every result element (`n × slots`), and a sort passes its row
/// `⌈log2 width⌉` times. Saturates at `u64::MAX`, which costs past any
/// bound.
#[must_use]
pub fn op_work(op: &Op, types: &[ValueType], results: &[ValueType]) -> u64 {
    let bytes = |ty: &ValueType| {
        ty.shape
            .numel()
            .saturating_mul(super::container::const_elem_size(ty.dtype) as u64)
    };
    let of = |id: u32| types.get(id as usize).copied();
    let linear = op
        .operands()
        .into_iter()
        .filter_map(of)
        .chain(results.iter().copied())
        .map(|ty| bytes(&ty))
        .fold(0u64, u64::saturating_add);
    match *op {
        Op::MatMul(a, b) => match (of(a), of(b)) {
            (Some(a), Some(b)) => {
                let k = a.shape.last_len().unwrap_or(1) as u64;
                let elem = super::container::const_elem_size(a.dtype) as u64;
                let mn = a.shape.rows().saturating_mul(b.shape.last_len().unwrap_or(1) as u64);
                mn.saturating_mul(k).saturating_mul(elem).max(linear)
            }
            _ => linear,
        },
        Op::ScatterAdd { idx, .. } | Op::ScatterSet { idx, .. } => {
            let slots = of(idx).map_or(1, |ty| ty.shape.numel());
            results
                .iter()
                .map(|ty| bytes(ty).saturating_mul(slots))
                .fold(linear, u64::saturating_add)
        }
        Op::SortDesc(x) => {
            let width = of(x).and_then(|ty| ty.shape.last_len()).unwrap_or(1).max(2);
            let passes = u64::from(u32::BITS - (width - 1).leading_zeros());
            linear.saturating_mul(passes)
        }
        _ => linear,
    }
}

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ValidateError {
    Body {
        stage: Stage,
        err: BodyError,
    },
    DuplicateStage(Stage),
    StagesUnsorted,
    NamesUnsortedOrDuplicate,
    DuplicatePort(Port),
    PortsUnsorted,
    PortChannelOutOfRange {
        port: Port,
        chan: u32,
    },
    PortConstPayload {
        port: Port,
    },
    ZeroCapacity {
        chan: u32,
    },
    SecondProducer {
        chan: u32,
        stage: Stage,
    },
    SecondConsumer {
        chan: u32,
        stage: Stage,
    },
    SinkMisplaced {
        name_index: u16,
        name: String,
        stage: Stage,
    },
    KernelUnavailable {
        name_index: u16,
        name: String,
    },
    NotReplayable {
        name_index: u16,
        name: String,
    },
    OutputWrongStage {
        name: String,
        stage: Stage,
    },
    OutputUnavailable {
        name: String,
        lane: u32,
    },
    OutputTypeRule {
        name: String,
    },
    OutputLane {
        name: String,
        lane: u32,
        lanes: u32,
    },
    InputsUnsortedOrDup,
    InputChannelOutOfRange {
        chan: u32,
    },
    ExternsUnsortedOrDup,
    ExternDeclConflict {
        chan: u32,
    },
    ExternNameOutOfRange {
        chan: u32,
    },
    ExternDirViolation {
        chan: u32,
        stage: Stage,
    },
    /// A stage's values take more bytes than an instance's state room
    /// holds ([`Budget::room_bytes`]).
    StagePastRoom {
        stage: Stage,
        bytes: u64,
        room: u64,
    },
    /// A stage's values' bytes pass a u64.
    StageBytesOverflow {
        stage: Stage,
    },
    /// A stage the device's measured rates cost past its stated bound on
    /// one submission ([`Budget::submit`]).
    StagePastSubmit {
        stage: Stage,
        cost_ns: u64,
        bound_ns: u64,
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
                "sink `{name}` (name #{name_index}) at stage {}; sinks run at the prologue",
                stage.name()
            ),
            KernelUnavailable { name_index, name } => {
                write!(
                    f,
                    "`{name}` (name #{name_index}): the engine serving this model runs \
                     no such kernel or sink, so binding it would either fail mid-fire \
                     or run as a silent no-op"
                )
            }
            NotReplayable { name_index, name } => write!(
                f,
                "`{name}` (name #{name_index}): time-/load-varying return — a \
                 register read in disguise"
            ),
            OutputWrongStage { name, stage } => write!(
                f,
                "output `{name}` read at stage {}; outputs are read at the epilogue",
                stage.name()
            ),
            OutputUnavailable { name, lane } => write!(
                f,
                "output `{name}` is not one lane {lane} of this pass reads out"
            ),
            OutputLane { name, lane, lanes } => write!(
                f,
                "output `{name}` read for lane {lane} of a pass of {lanes}"
            ),
            OutputTypeRule { name } => write!(
                f,
                "output `{name}`: the type read is not the one the model declares"
            ),
            InputsUnsortedOrDup => f.write_str("input table must be sorted by channel, unique"),
            InputChannelOutOfRange { chan } => {
                write!(f, "input: channel index {chan} out of range")
            }
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
            StagePastRoom { stage, bytes, room } => write!(
                f,
                "stage {}: its values take {bytes} bytes and an instance's state room \
                 on the device serving this model holds {room}; it can bind nowhere",
                stage.name()
            ),
            StagePastSubmit {
                stage,
                cost_ns,
                bound_ns,
            } => write!(
                f,
                "stage {}: the rates this device's load measured cost it at least \
                 {cost_ns} ns, past the {bound_ns} ns the device states one submission \
                 may take; its fire would fault every tenant's",
                stage.name()
            ),
            StageBytesOverflow { stage } => write!(
                f,
                "stage {}: its values' bytes pass a u64",
                stage.name()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ValidateError {}

impl BoundTrace {
    /// How wide a float row of the readout rows is, 0 where no lane of the
    /// pass reads one out.
    #[must_use]
    pub fn vocab(&self) -> u32 {
        self.lanes
            .iter()
            .flatten()
            .find(|decl| decl.readout && decl.dtype == Dtype::F32)
            .map_or(0, |decl| decl.width)
    }

    /// The decl of the output `ops`' name `index` names, as `lane` reads it.
    #[must_use]
    pub fn output(&self, names: &[String], index: u16, lane: u32) -> Option<&OutputDecl> {
        let name = names.get(index as usize)?;
        self.lanes
            .get(lane as usize)?
            .iter()
            .find(|decl| &decl.name == name)
    }
}

pub(crate) fn channel_value_type(decl: &ChannelDecl) -> ValueType {
    ValueType::new(decl.shape, decl.dtype.program_dtype())
}

/// `lanes` holds one entry a lane, in lane order.
pub fn bind(
    container: TraceContainer,
    lanes: Vec<Vec<OutputDecl>>,
    caps: EngineCaps,
) -> Result<BoundTrace, ValidateError> {
    check_structure(&container)?;
    check_externs(&container)?;
    let channel_types: Vec<ValueType> = container.channels.iter().map(channel_value_type).collect();
    let stage_types = check_bodies(&container, &channel_types)?;
    check_outputs(&container, &lanes)?;
    check_second_party_names(&container, caps)?;
    check_spsc_endpoints(&container)?;

    let readiness = readiness_table(&container);
    let classes = classify_channels(&container, &readiness);

    let hash = container.hash();
    Ok(BoundTrace {
        container,
        lanes,
        hash,
        channel_types,
        stage_types,
        readiness,
        classes,
        budget: None,
    })
}

/// [`bind`], then every stage held to `budget` ([`check_budget`]): what a
/// registration binds, refused by name past it.
pub fn bind_within(
    container: TraceContainer,
    lanes: Vec<Vec<OutputDecl>>,
    caps: EngineCaps,
    budget: Budget,
) -> Result<BoundTrace, ValidateError> {
    let mut bound = bind(container, lanes, caps)?;
    check_budget(&bound, budget)?;
    bound.budget = Some(budget);
    Ok(bound)
}

/// The bytes the values `types` take, each its elements times its
/// dtype's; `None` past a u64.
#[must_use]
pub fn values_bytes(types: &[ValueType]) -> Option<u64> {
    types.iter().try_fold(0u64, |sum, ty| {
        let elem = super::container::const_elem_size(ty.dtype) as u64;
        sum.checked_add(ty.shape.numel().checked_mul(elem)?)
    })
}

/// Every stage's footprint ([`StageFootprint`]) as `bound` declares it.
pub fn footprints(bound: &BoundTrace) -> Result<Vec<StageFootprint>, ValidateError> {
    bound
        .container
        .stages
        .iter()
        .zip(&bound.stage_types)
        .map(|(program, types)| {
            let stage = program.stage;
            let bytes = values_bytes(types).ok_or(ValidateError::StageBytesOverflow { stage })?;
            let (mut launched, mut work, mut next) = (0u64, 0u64, 0usize);
            for op in &program.ops {
                let count = op.result_count() as usize;
                let results = types.get(next..next + count).unwrap_or(&[]);
                next += count;
                if launches(op) {
                    launched += 1;
                    work = work.saturating_add(op_work(op, types, results));
                }
            }
            Ok(StageFootprint {
                stage,
                bytes,
                launches: launched,
                work,
            })
        })
        .collect()
}

/// Every stage of `bound` within `budget`: its values within an
/// instance's state room, and where the device states a bound on one
/// submission, its cost at the load's measured rates within it.
pub fn check_budget(bound: &BoundTrace, budget: Budget) -> Result<(), ValidateError> {
    for footprint in footprints(bound)? {
        if footprint.bytes > budget.room_bytes {
            return Err(ValidateError::StagePastRoom {
                stage: footprint.stage,
                bytes: footprint.bytes,
                room: budget.room_bytes,
            });
        }
        if let Some(submit) = budget.submit {
            let cost = submit.cost_ns(footprint.launches, footprint.work);
            if !(cost <= submit.bound_ns as f64) {
                return Err(ValidateError::StagePastSubmit {
                    stage: footprint.stage,
                    cost_ns: if cost.is_finite() { cost as u64 } else { u64::MAX },
                    bound_ns: submit.bound_ns,
                });
            }
        }
    }
    Ok(())
}

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
                let elem_size = super::container::const_elem_size(*dtype) as u64;
                let expect = shape.numel().checked_mul(elem_size);
                if expect != Some(data.len() as u64) {
                    return Err(ValidateError::PortConstPayload { port: p.port });
                }
            }
        }
    }
    if container.inputs.windows(2).any(|w| w[0] >= w[1]) {
        return Err(ValidateError::InputsUnsortedOrDup);
    }
    if let Some(&chan) = container
        .inputs
        .iter()
        .find(|&&chan| chan as usize >= container.channels.len())
    {
        return Err(ValidateError::InputChannelOutOfRange { chan });
    }
    Ok(())
}

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

/// Rechecked per pass: a cached program bound for one pass may not fit the next.
pub fn check_outputs(
    container: &TraceContainer,
    lanes: &[Vec<OutputDecl>],
) -> Result<(), ValidateError> {
    for sp in &container.stages {
        for op in &sp.ops {
            let Op::Output {
                index,
                lane,
                shape,
                dtype,
            } = *op
            else {
                continue;
            };
            let name = String::from(resolve_name(container, index));
            if sp.stage != Stage::Epilogue {
                return Err(ValidateError::OutputWrongStage {
                    name,
                    stage: sp.stage,
                });
            }
            let Some(outputs) = lanes.get(lane as usize) else {
                return Err(ValidateError::OutputLane {
                    name,
                    lane,
                    lanes: u32::try_from(lanes.len()).unwrap_or(u32::MAX),
                });
            };
            let Some(decl) = outputs.iter().find(|decl| decl.name == name) else {
                return Err(ValidateError::OutputUnavailable { name, lane });
            };
            if !output_type_ok(decl, shape, dtype) {
                return Err(ValidateError::OutputTypeRule { name });
            }
        }
    }
    Ok(())
}

/// A lane reads `[rows, width]` of a row or voxel output, `[1, width]` of a
/// lane output, and `[layers, width]` of one written per layer.
fn output_type_ok(decl: &OutputDecl, shape: Shape, dtype: Dtype) -> bool {
    let rows_ok = |rows: u32| match decl.axis {
        Axis::Lane => rows == decl.layers.max(1),
        Axis::Row | Axis::Voxel => rows >= 1,
    };
    dtype == decl.dtype
        && shape.rank() == 2
        && rows_ok(shape.dims()[0])
        && shape.dims()[1] == decl.width
}

fn check_second_party_names(
    container: &TraceContainer,
    caps: EngineCaps,
) -> Result<(), ValidateError> {
    for sp in &container.stages {
        for op in &sp.ops {
            let unavailable = |name: &u16| ValidateError::KernelUnavailable {
                name_index: *name,
                name: resolve_name(container, *name).into(),
            };
            match op {
                Op::KernelCall { name, .. } => return Err(unavailable(name)),
                Op::SinkCall { name, .. } => {
                    let n = resolve_name(container, *name);
                    let served = match n {
                        "attn_page_mask" => caps.attn_page_mask,
                        "lora" => caps.lora,
                        _ => true,
                    };
                    if !served || !KNOWN_SINKS.contains(&n) {
                        return Err(unavailable(name));
                    }
                    if sp.stage != Stage::Prologue {
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
            let (local_role, peer_dir) = match use_ {
                ChannelUse::Put => (HostRole::Writer, ExternDir::Import),
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
    let descriptor_reads = container
        .ports
        .iter()
        .filter_map(|p| match p.source {
            PortSource::Channel(c) => Some(c),
            PortSource::Const { .. } => None,
        })
        .chain(container.inputs.iter().copied());
    for c in descriptor_reads {
        if extern_dir(c) == Some(ExternDir::Export) {
            return Err(ValidateError::ExternDirViolation {
                chan: c,
                stage: Stage::Prologue,
            });
        }
        if channel_decl(container, c, Stage::Prologue, 0)?.host_role == HostRole::Reader {
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
                for &chan in &c.inputs {
                    visit(chan, phase, Direction::NeedsFull, &mut seen);
                }
            }
            _ => {
                let stage = match phase {
                    Phase::Prologue => Stage::Prologue,
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

pub(crate) fn classify_channels(
    c: &TraceContainer,
    readiness: &[ReadinessEntry],
) -> Vec<ChannelClass> {
    let is_extern = |chan: u32| c.externs.iter().any(|e| e.chan == chan);
    let mut fallible = [false; Phase::ORDER.len()];
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
            classes.push(ChannelClass::FullRing);
            continue;
        }
        let mut take: Option<(usize, usize, u32)> = None;
        let mut put: Option<(usize, usize, u32)> = None;
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
        for p in &c.ports {
            if matches!(p.source, PortSource::Channel(ch) if ch == ci) {
                extra = true;
            }
        }
        if c.inputs.contains(&ci) {
            extra = true;
        }
        let (Some((tsi, toi, tid)), Some((psi, poi, pval)), false) = (take, put, extra) else {
            classes.push(ChannelClass::FullRing);
            continue;
        };
        if tsi != psi || toi >= poi {
            classes.push(ChannelClass::FullRing);
            continue;
        }
        let ops = &c.stages[tsi].ops;
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
