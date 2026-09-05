//! The launch package: a program in the shape an engine executes it — the
//! value table, channels/ports to allocate and bind, per-stage op DAGs,
//! and the per-stage plan the emitted kernels were generated from.

use serde::{Deserialize, Serialize};

use eta_ir::container::{ChanDType, ExternDir, HostRole, PortSource};
use eta_ir::op::{ChannelIndex, IntrinsicId, intrinsic_tags, tags};
use eta_ir::registry::{Port, Stage};
use eta_ir::types::{Dtype, RngKind, ValueType, from_wire};
use eta_ir::validate::{BoundTrace, Direction};

use crate::plan::{
    CompiledStage, Dimension, NodeIndex, Region, RegionKind, RegionPartition, SymbolicExtent,
    SymbolicType, stage_identity,
};

use crate::codegen::fault::FaultClass;
use crate::codegen::op_view::OpView;

// ──────────────────────────── the trace half ────────────────────────────

/// Where an SSA value in the trace comes from. Not `eta_ir::op::ValueSource`,
/// which classifies a value by what decides it (device/channel/operands);
/// this one names the trace construct that defined it — the wire numbering
/// an engine reads positionally.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueOrigin {
    /// A trace-time constant; the bits are in `literal_bits`.
    Const = 0,
    /// A first-party value intrinsic; which one is in `intrinsic`.
    Intrinsic = 1,
    /// A `chan_take`; the channel is in `channel`.
    ChannelTake = 2,
    /// A `chan_read`; the channel is in `channel`.
    ChannelRead = 3,
    /// An op's result.
    #[default]
    OpResult = 4,
}

/// One row of the SSA value table, in the global numbering the stage bodies
/// use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchValue {
    /// The value's global id: stage `i`'s local `v` is `Σ|stage_types[..i]| + v`.
    pub id: u32,
    /// Where it comes from.
    pub source: ValueOrigin,
    /// Its element type.
    pub dtype: Dtype,
    /// Which intrinsic, when [`ValueOrigin::Intrinsic`]. `Option`, not a
    /// bare `IntrinsicId`: wire id 0 is `IntrinsicId::Logits`, so a bare
    /// field would misread "no intrinsic" as "logits" for every other value.
    pub intrinsic: Option<IntrinsicId>,
    /// Which channel, when [`ValueOrigin::ChannelTake`] or
    /// [`ValueOrigin::ChannelRead`].
    pub channel: ChannelIndex,
    /// The literal's raw bits, when [`ValueOrigin::Const`].
    pub literal_bits: u32,
    /// Its shape, as dims.
    pub shape: Vec<u32>,
}

impl Default for LaunchValue {
    fn default() -> LaunchValue {
        LaunchValue {
            id: 0,
            source: ValueOrigin::OpResult,
            dtype: Dtype::F32,
            intrinsic: None,
            channel: 0,
            literal_bits: 0,
            shape: Vec::new(),
        }
    }
}

/// One op of a stage body, or of a stage plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchOp {
    /// The ETA wire tag — `eta_ir::op::tags`, the one place an op tag is
    /// spelled as a number.
    pub tag: u8,
    /// How many SSA ids this op defines.
    pub result_count: u16,
    /// Its first result's id.
    pub result_id: u32,
    /// Which intrinsic, for `intrinsic_val`. See [`LaunchValue::intrinsic`] on
    /// why it is an `Option`.
    pub intrinsic: Option<IntrinsicId>,
    /// A `const`'s literal dtype.
    pub lit_dtype: Dtype,
    /// The result's element type.
    pub dtype: Dtype,
    /// A `pivot_threshold`'s predicate tag — `eta_ir::wire::predicate_tags`.
    /// Raw, not `Predicate`, since this field is only the discriminant.
    pub pred_tag: u8,
    /// A `pivot_threshold`'s predicate payload — a value id, remapped like
    /// an operand rather than left as an immediate.
    pub pred_payload: u32,
    /// An `rng`'s distribution.
    pub rng_kind: RngKind,
    /// A `const`'s raw bits, read per `lit_dtype`.
    pub lit_bits: u32,
    /// The channel this op touches, when it touches one.
    pub channel: Option<ChannelIndex>,
    /// Index into the package's name table, for `kernel_call` / `sink_call`.
    pub name_index: u32,
    /// First trace-known immediate.
    pub imm: u32,
    /// Second immediate.
    pub imm2: u32,
    /// Third immediate.
    pub imm3: u32,
    /// Value-id operands, as the container encodes them.
    pub args: Vec<u32>,
    /// The result's shape, as dims.
    pub shape: Vec<u32>,
}

impl Default for LaunchOp {
    fn default() -> LaunchOp {
        LaunchOp {
            tag: 0,
            result_count: 0,
            result_id: 0,
            intrinsic: None,
            lit_dtype: Dtype::F32,
            dtype: Dtype::F32,
            pred_tag: 0,
            pred_payload: 0,
            rng_kind: RngKind::Uniform,
            lit_bits: 0,
            channel: None,
            name_index: 0,
            imm: 0,
            imm2: 0,
            imm3: 0,
            args: Vec::new(),
            shape: Vec::new(),
        }
    }
}

/// One channel the engine allocates and binds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannel {
    /// Its index in declaration order.
    pub id: ChannelIndex,
    /// How many cells the ring holds.
    pub capacity: u32,
    /// The cell's element type. [`ChanDType::Act`] is the late-bound
    /// activation type; the engine allocates from
    /// [`ChanDType::program_dtype`].
    pub dtype: ChanDType,
    /// Whether the channel arrives seeded.
    pub seeded: bool,
    /// Which end the host holds, if any.
    pub host_role: HostRole,
    /// Whether this channel crosses to another instance, and which way.
    pub extern_dir: Option<ExternDir>,
    /// Which bit the channel's first in-pass op needs; `None` if no stage
    /// touches it.
    pub readiness: Option<Direction>,
    /// The cell's shape, as dims.
    pub shape: Vec<u32>,
    /// The extern binding's name, when `extern_dir` is `Some`.
    pub extern_name: Vec<u8>,
}

impl Default for LaunchChannel {
    fn default() -> LaunchChannel {
        LaunchChannel {
            id: 0,
            capacity: 0,
            dtype: ChanDType::Concrete(Dtype::F32),
            seeded: false,
            host_role: HostRole::None,
            extern_dir: None,
            readiness: None,
            shape: Vec::new(),
            extern_name: Vec::new(),
        }
    }
}

/// One descriptor-port binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPort {
    /// Which port — [`eta_ir::registry::Port`].
    pub port: Port,
    /// Whether the binding is a trace-time constant rather than a channel.
    pub is_const: bool,
    /// The constant's element type, when `is_const`.
    pub const_dtype: Dtype,
    /// The channel bound, when not `is_const`.
    pub channel: ChannelIndex,
    /// The constant's shape, when `is_const`.
    pub const_shape: Vec<u32>,
    /// The constant's payload, when `is_const`.
    pub const_data: Vec<u8>,
}

impl Default for LaunchPort {
    fn default() -> LaunchPort {
        LaunchPort {
            port: Port::EmbedTokens,
            is_const: false,
            const_dtype: Dtype::F32,
            channel: 0,
            const_shape: Vec::new(),
            const_data: Vec::new(),
        }
    }
}

/// One `chan_put`: a value committed to a channel at pass end.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPut {
    /// The channel written.
    pub channel: ChannelIndex,
    /// The value written.
    pub value: u32,
}

/// One stage body: the ops it launches and the channel effects it commits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStage {
    /// Which attachment stage this is.
    pub stage: Stage,
    /// The ops, in program order.
    pub ops: Vec<LaunchOp>,
    /// The puts, committed at pass end.
    pub puts: Vec<LaunchPut>,
    /// The channels this stage takes from.
    pub takes: Vec<ChannelIndex>,
    /// The channels this stage reads.
    pub reads: Vec<ChannelIndex>,
}

impl Default for LaunchStage {
    fn default() -> LaunchStage {
        LaunchStage {
            stage: Stage::Prologue,
            ops: Vec::new(),
            puts: Vec::new(),
            takes: Vec::new(),
            reads: Vec::new(),
        }
    }
}

// ──────────────────────────── the plan half ─────────────────────────────

/// One region of a compiled stage: a contiguous run of ops the backend
/// launches together.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRegion {
    /// Emitted, or a library call. [`RegionKind`] is the partitioner's own
    /// enum, shipped as itself.
    pub kind: RegionKind,
    /// The backend's schedule tag for the region — [`crate::plan::ScheduleTemplate`]
    /// as a byte; carried through to the device lane table unchanged.
    pub schedule: u8,
    /// The ops it covers, by index into the plan's op list.
    pub nodes: Vec<u32>,
    /// The values it reads.
    pub inputs: Vec<u32>,
    /// The values it defines.
    pub outputs: Vec<u32>,
    /// The puts it commits.
    pub sinks: Vec<LaunchPut>,
    /// A multi-row value naming the region's row geometry — the plan's
    /// `Region::row_value`. A backend that launches one block per row
    /// resolves the row count from this value's descriptor; `None` is one
    /// block per lane.
    #[serde(default)]
    pub row_value: Option<u32>,
    /// The plan's `Region::row_alias`: the static row count standing for
    /// the geometry's symbolic rows, if any.
    #[serde(default)]
    pub row_alias: Option<u64>,
}

/// One value's type in a stage plan: an element type and a list of axes, as
/// [`Dimension`]/[`SymbolicExtent`] (the planner's own types, shipped as
/// themselves).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    /// The element type.
    pub dtype: Dtype,
    /// The axes, outermost first.
    pub axes: Vec<Dimension>,
}

impl Default for LaunchPlanValue {
    fn default() -> LaunchPlanValue {
        LaunchPlanValue {
            dtype: Dtype::F32,
            axes: Vec::new(),
        }
    }
}

/// One channel-binding rule of a grouped launch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannelRule {
    /// The value bound.
    pub value: u32,
    /// Its local slot in the group.
    pub local: u32,
}

/// What a stage's grouped launch depends on.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageNeeds {
    /// Reads the projected query intrinsic.
    pub query: bool,
    /// Reads the layer-index intrinsic.
    pub layer: bool,
    /// Reads the attention-score intrinsic.
    pub attn_score: bool,
    /// Calls a second-party kernel.
    pub kernel_call: bool,
    /// Reads the attention page mask.
    pub page_mask: bool,
    /// Reads the MTP draft rows.
    pub mtp_rows: bool,
    /// Reads the draft head's token ids (`mtp_drafts`).
    pub mtp_drafts: bool,
    /// Reads a LoRA sink.
    pub lora: bool,
    /// The grouped launch path can cover this stage at all.
    pub grouped_valid: bool,
}

/// One compiled stage, as the emitted kernels were generated from it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStagePlan {
    /// The stage's structural signature — what makes two stages the same
    /// shape.
    pub signature_hash: u64,
    /// The stage's graph-cache identity: signature plus everything a compiled
    /// artifact depends on.
    pub identity: u64,
    /// What its grouped launch depends on.
    pub needs: StageNeeds,
    /// How many MTP draft rows it reads.
    pub mtp_rows: u32,
    /// How many ids its `mtp_drafts` readers declare (the longest), so a
    /// stage that reads the draft plane WITHOUT reading `logits` still says
    /// how many readout rows it spans. Zero when none does.
    #[serde(default)]
    pub drafts_len: u32,
    /// Its normalized ops, in stage-local numbering.
    pub ops: Vec<LaunchOp>,
    /// For each normalized op, which source ops it came from.
    pub source_ops: Vec<Vec<u32>>,
    /// Its value table's types.
    pub value_types: Vec<LaunchPlanValue>,
    /// **This stage's ORIGINAL value id -> its normalized one**, `u32::MAX`
    /// where normalization removed the value. Its length is the stage's
    /// original value count, which is also the width of the stage's block in
    /// the package's global numbering.
    pub value_map: Vec<u32>,
    /// Which channel each bound value binds through.
    pub channel_bindings: Vec<u32>,
    /// The names its `kernel_call`s and `sink_call`s index.
    pub names: Vec<String>,
    /// Its singleton-partition regions.
    pub singleton: Vec<LaunchRegion>,
    /// Its fused-partition regions.
    pub fused: Vec<LaunchRegion>,
    /// Which runtime extents it depends on.
    pub used_extents: Vec<SymbolicExtent>,
    /// Its grouped channel rules.
    pub channel_rules: Vec<LaunchChannelRule>,
    /// Why the grouped plan could not be derived, if it could not. Empty means
    /// it could.
    pub error: String,
}

/// A whole program, in the shape an engine executes it. Deliberately not
/// ETA: an engine reading this never sees a container, a wire format, or an
/// identity to re-check.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPackage {
    /// The SSA value table, in global numbering.
    pub values: Vec<LaunchValue>,
    /// The channels to allocate.
    pub channels: Vec<LaunchChannel>,
    /// The descriptor ports to bind.
    pub ports: Vec<LaunchPort>,
    /// The name table `kernel_call` / `sink_call` / extern names index.
    pub names: Vec<String>,
    /// The stage bodies.
    pub stages: Vec<LaunchStage>,
    /// The compiled stage plans, parallel to `stages`.
    pub plans: Vec<LaunchStagePlan>,
    /// The fault-code table the emitted kernels were written against.
    /// Carried with the program rather than shared vocabulary, since a
    /// code's meaning is decided by the emitter that compiled the kernel
    /// that raised it.
    pub fault_classes: Vec<FaultClass>,
}

impl LaunchPackage {
    /// **WHERE STAGE `at`'S BLOCK STARTS IN THE GLOBAL VALUE NUMBERING.**
    ///
    /// [`lower_values`] walks the stages in order and numbers each stage's
    /// ORIGINAL values `base..base + n`, so a stage's block is as wide as its
    /// original value table — which is [`LaunchStagePlan::value_map`]'s
    /// length, not its `value_types`' length. The two differ whenever
    /// normalization folded anything, which is nearly always.
    ///
    /// `None` when `at` is past the package's plans.
    #[must_use]
    pub fn stage_base(&self, at: usize) -> Option<u32> {
        if at >= self.plans.len() {
            return None;
        }
        Some(
            self.plans[..at]
                .iter()
                .map(|plan| plan.value_map.len() as u32)
                .sum(),
        )
    }

    /// **A GLOBAL VALUE ID AS THE PLAN'S OWN NUMBERING SPELLS IT**: the stage
    /// it belongs to, and its id in that stage's normalized value table — the
    /// table the emitted kernel's descriptors and scratch offsets are indexed
    /// by.
    ///
    /// Two hops, and neither is arithmetic on its own. The first takes the
    /// global id to the stage's original id by subtracting the block base; the
    /// second takes that through [`LaunchStagePlan::value_map`], because
    /// normalization renumbers densely after folding CSE, aliases and dead
    /// values away.
    ///
    /// `None` when no stage's block holds it, and `Some((stage, None))` when
    /// the stage holds it but normalization removed it — a value the device
    /// never allocates, which a caller staging roots must skip rather than
    /// treat as an error.
    #[must_use]
    pub fn plan_local(&self, global: u32) -> Option<(usize, Option<u32>)> {
        let mut base = 0u32;
        for (at, plan) in self.plans.iter().enumerate() {
            let width = plan.value_map.len() as u32;
            if global < base + width {
                let original = (global - base) as usize;
                let mapped = plan.value_map[original];
                return Some((at, (mapped != u32::MAX).then_some(mapped)));
            }
            base += width;
        }
        None
    }
}

/// Build the launch package for a bound trace and its compiled stages.
/// `stages` is `crate::plan::compile_bound(bound)`, in container order.
pub fn build(bound: &BoundTrace, stages: &[CompiledStage]) -> LaunchPackage {
    LaunchPackage {
        values: lower_values(bound),
        channels: lower_channels(bound),
        ports: lower_ports(bound),
        names: bound.container.names.clone(),
        stages: lower_stages(bound),
        plans: stages.iter().map(lower_plan).collect(),
        fault_classes: crate::codegen::fault::classes(),
    }
}

// ── the trace half: what to allocate, bind, and launch ──

/// The SSA value table, in the same global numbering the stage bodies use:
/// stage `i`'s local value `v` is global `Σ|stage_types[..i]| + v`.
fn lower_values(bound: &BoundTrace) -> Vec<LaunchValue> {
    let mut values = Vec::new();
    let mut base = 0u32;
    for (stage_index, program) in bound.container.stages.iter().enumerate() {
        let types = &bound.stage_types[stage_index];
        let mut local = 0u32;
        let mut push = |local: u32, mut value: LaunchValue| {
            let value_type = types
                .get(local as usize)
                .copied()
                .unwrap_or(ValueType::scalar(Dtype::F32));
            value.id = base + local;
            value.dtype = value_type.dtype;
            value.shape = value_type.shape.dims().to_vec();
            values.push(value);
        };
        for op in &program.ops {
            let view = OpView::of(op);
            match view.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => {
                    push(
                        local,
                        LaunchValue {
                            source: if view.tag == tags::CHAN_TAKE {
                                ValueOrigin::ChannelTake
                            } else {
                                ValueOrigin::ChannelRead
                            },
                            channel: view.chan as u32,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::CONST => {
                    push(
                        local,
                        LaunchValue {
                            source: ValueOrigin::Const,
                            literal_bits: view.lit_bits,
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::INTRINSIC_VAL => {
                    push(
                        local,
                        LaunchValue {
                            source: ValueOrigin::Intrinsic,
                            intrinsic: IntrinsicId::from_u16(view.intr),
                            ..LaunchValue::default()
                        },
                    );
                    local += 1;
                }
                tags::CHAN_PUT | tags::SINK_CALL => {}
                _ => {
                    for result in 0..view.results {
                        push(
                            local + result,
                            LaunchValue {
                                source: ValueOrigin::OpResult,
                                ..LaunchValue::default()
                            },
                        );
                    }
                    local += view.results;
                }
            }
        }
        base += types.len() as u32;
    }
    values
}

fn lower_channels(bound: &BoundTrace) -> Vec<LaunchChannel> {
    bound
        .container
        .channels
        .iter()
        .enumerate()
        .map(|(index, decl)| {
            let extern_decl = bound
                .container
                .externs
                .iter()
                .find(|entry| entry.chan as usize == index);
            // First-touch, in pass order: an `InPlace` channel is both taken
            // and put, so this says which comes first.
            let readiness = bound
                .readiness
                .iter()
                .find(|entry| entry.chan as usize == index)
                .map(|entry| entry.dir);
            LaunchChannel {
                id: index as u32,
                capacity: decl.capacity,
                // The program-side element type, with a late-bound activation
                // dtype already materialized — the engine never sees `ACT`.
                dtype: ChanDType::Concrete(bound.channel_types[index].dtype),
                seeded: decl.seeded,
                host_role: decl.host_role,
                extern_dir: extern_decl.map(|entry| entry.dir),
                readiness,
                shape: decl.shape.dims().to_vec(),
                extern_name: extern_decl
                    .and_then(|entry| bound.container.names.get(entry.name as usize))
                    .map(|name| name.as_bytes().to_vec())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

fn lower_ports(bound: &BoundTrace) -> Vec<LaunchPort> {
    bound
        .container
        .ports
        .iter()
        .map(|binding| match binding.source {
            PortSource::Channel(chan) => LaunchPort {
                port: binding.port,
                is_const: false,
                channel: chan,
                ..LaunchPort::default()
            },
            PortSource::Const {
                dtype,
                ref shape,
                ref data,
            } => LaunchPort {
                port: binding.port,
                is_const: true,
                const_dtype: dtype,
                const_shape: shape.dims().to_vec(),
                const_data: data.clone(),
                ..LaunchPort::default()
            },
        })
        .collect()
}

/// The stage bodies, with every operand remapped from stage-local to the
/// global numbering of [`lower_values`]. Ops that define no value
/// (`chan_put`) become stage effects instead of ops: a put is committed at
/// pass end, not launched.
fn lower_stages(bound: &BoundTrace) -> Vec<LaunchStage> {
    let mut stages = Vec::new();
    let mut base = 0u32;
    for (stage_index, program) in bound.container.stages.iter().enumerate() {
        let types = &bound.stage_types[stage_index];
        let global = |local: u32| base + local;
        let result_type = |local: u32| {
            types
                .get(local as usize)
                .copied()
                .unwrap_or(ValueType::scalar(Dtype::F32))
        };

        let mut stage = LaunchStage {
            stage: program.stage,
            ..LaunchStage::default()
        };
        let mut local = 0u32;
        for op in &program.ops {
            let view = OpView::of(op);
            match view.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => {
                    if view.tag == tags::CHAN_TAKE {
                        stage.takes.push(view.chan as u32);
                    } else {
                        stage.reads.push(view.chan as u32);
                    }
                    local += 1;
                }
                tags::CONST | tags::INTRINSIC_VAL => local += 1,
                tags::CHAN_PUT => stage.puts.push(LaunchPut {
                    channel: view.chan as u32,
                    value: global(view.args[0]),
                }),
                _ => {
                    let value_type = result_type(local);
                    stage.ops.push(LaunchOp {
                        tag: view.tag,
                        result_count: view.results as u16,
                        result_id: global(local),
                        // A stage body carries no `intrinsic_val` op — those
                        // became `LaunchValue`s above.
                        intrinsic: None,
                        lit_dtype: dtype(view.lit_dtype),
                        dtype: value_type.dtype,
                        pred_tag: view.pred_tag,
                        rng_kind: rng(view.kind),
                        lit_bits: view.lit_bits,
                        // Every predicate tag carries a value id, so it is
                        // remapped like any other operand.
                        pred_payload: global(view.pred_payload),
                        channel: None,
                        name_index: u32::from(view.name_idx),
                        imm: view.imm,
                        imm2: view.imm2,
                        imm3: view.imm3,
                        args: view.args.iter().map(|arg| global(*arg)).collect(),
                        shape: value_type.shape.dims().to_vec(),
                    });
                    local += view.results;
                }
            }
        }
        stages.push(stage);
        base += types.len() as u32;
    }
    stages
}

// ── the plan half: what the emitted kernels were generated from ──

fn lower_plan(stage: &CompiledStage) -> LaunchStagePlan {
    let normalized = &stage.normalized;
    let ops: Vec<OpView> = OpView::of_all(&normalized.ops);
    let grouped = GroupedPlan::derive(
        &ops,
        &normalized.value_types,
        &normalized.names,
        normalized.channel_bindings.len(),
    );

    LaunchStagePlan {
        signature_hash: stage.signature.hash,
        identity: stage_identity(stage),
        needs: grouped.needs,
        mtp_rows: grouped.mtp_rows,
        drafts_len: grouped.drafts_len,
        ops: ops.iter().map(lower_plan_op).collect(),
        source_ops: normalized.source_ops.clone(),
        value_map: normalized.value_map.clone(),
        value_types: normalized
            .value_types
            .iter()
            .map(lower_plan_value)
            .collect(),
        channel_bindings: normalized.channel_bindings.clone(),
        names: normalized.names.clone(),
        singleton: lower_partition(&stage.singleton),
        fused: lower_partition(&stage.fused),
        used_extents: grouped.used_extents,
        channel_rules: grouped.channel_rules,
        error: grouped.error,
    }
}

/// A normalized op keeps its stage-local numbering: the plan's ops index the
/// plan's own value table, which is what the emitted kernels bind against.
fn lower_plan_op(view: &OpView) -> LaunchOp {
    LaunchOp {
        tag: view.tag,
        result_count: view.results as u16,
        result_id: 0,
        // `intr` is set by the `intrinsic_val` arm of the wire projection and
        // by nothing else, so gating on the tag is what makes `None` mean
        // "no intrinsic" rather than "intrinsic zero".
        intrinsic: (view.tag == tags::INTRINSIC_VAL)
            .then(|| IntrinsicId::from_u16(view.intr))
            .flatten(),
        lit_dtype: dtype(view.lit_dtype),
        dtype: dtype(view.dtype),
        pred_tag: view.pred_tag,
        rng_kind: rng(view.kind),
        lit_bits: view.lit_bits,
        pred_payload: view.pred_payload,
        channel: u32::try_from(view.chan).ok(),
        name_index: u32::from(view.name_idx),
        imm: view.imm,
        imm2: view.imm2,
        imm3: view.imm3,
        args: view.args.clone(),
        shape: view.shape.clone(),
    }
}

/// A planned value's type, as the launch package spells it — the same
/// spelling as [`crate::plan::Dimension`], cloned rather than translated.
fn lower_plan_value(value_type: &SymbolicType) -> LaunchPlanValue {
    LaunchPlanValue {
        dtype: value_type.dtype,
        axes: value_type.dims.clone(),
    }
}

/// The element type a wire byte names. `F32` for a byte no dtype claims — the
/// same fallback the `as u8` round trip this replaced had, made explicit.
fn dtype(byte: u8) -> Dtype {
    from_wire(byte).unwrap_or(Dtype::F32)
}

/// The distribution a wire byte names. Uniform is tag 0, which is what every
/// non-`rng` op carries.
fn rng(byte: u8) -> RngKind {
    if byte == RngKind::Gumbel as u8 {
        RngKind::Gumbel
    } else {
        RngKind::Uniform
    }
}

fn lower_partition(partition: &RegionPartition) -> Vec<LaunchRegion> {
    partition.regions.iter().map(lower_region).collect()
}

fn lower_region(region: &Region) -> LaunchRegion {
    LaunchRegion {
        kind: region.kind,
        schedule: region.schedule as u8,
        // `LaunchRegion` is the engine ABI, which has one integer space;
        // the node tags stop here.
        nodes: region.nodes.iter().copied().map(NodeIndex::get).collect(),
        inputs: region.inputs.clone(),
        outputs: region.outputs.clone(),
        row_value: region.row_value,
        row_alias: region.row_alias,
        sinks: region
            .sinks
            .iter()
            .map(|sink| LaunchPut {
                channel: sink.channel_slot.get(),
                value: sink.value,
            })
            .collect(),
    }
}

/// The grouped launch path's view of a stage: which runtime extents it
/// depends on, which values bind through a channel, which intrinsics it
/// reads, and whether the path can cover it at all. A decision about the
/// program, made once here rather than in each engine's launch path.
#[derive(Default)]
struct GroupedPlan {
    needs: StageNeeds,
    mtp_rows: u32,
    /// The longest `mtp_drafts` declaration in the stage; see
    /// [`LaunchStagePlan::drafts_len`].
    drafts_len: u32,
    used_extents: Vec<SymbolicExtent>,
    channel_rules: Vec<LaunchChannelRule>,
    error: String,
}

impl GroupedPlan {
    fn derive(
        ops: &[OpView],
        value_types: &[SymbolicType],
        names: &[String],
        channel_count: usize,
    ) -> Self {
        let mut plan = GroupedPlan {
            needs: StageNeeds {
                grouped_valid: true,
                ..StageNeeds::default()
            },
            ..GroupedPlan::default()
        };
        // `SymbolicExtent` is an enum, so an unsupported runtime extent is
        // unrepresentable — no bounds check needed on the tag.
        let mut seen = [false; SymbolicExtent::ALL.len()];
        for value_type in value_types {
            for dimension in &value_type.dims {
                let Dimension::Symbolic(extent) = *dimension else {
                    continue;
                };
                if !seen[extent as usize] {
                    seen[extent as usize] = true;
                    plan.used_extents.push(extent);
                }
            }
        }

        let mut value_bases = Vec::with_capacity(ops.len());
        let mut next_value = 0u32;
        for op in ops {
            value_bases.push(next_value);
            next_value += op.results;
        }

        for (node, op) in ops.iter().enumerate() {
            // A second-party kernel is not grouped-coverable — only the
            // fused path can launch one — but this plan is the shared
            // lane-binding metadata the fused path resolves through, so it
            // describes the op instead of rejecting it. `query` is set
            // because the kernel reads the lane's post-rope query row.
            if op.tag == tags::KERNEL_CALL {
                plan.needs.query = true;
                plan.needs.kernel_call = true;
                continue;
            }
            // Sinks are grouped-walkable but only fused-executable, described
            // here for the same reason. `lora` configures the whole forward;
            // every other sink sets the page mask.
            if op.tag == tags::SINK_CALL {
                if names.get(op.name_idx as usize).map(String::as_str) == Some("lora") {
                    plan.needs.lora = true;
                } else {
                    plan.needs.page_mask = true;
                }
            }
            if !grouped_supported_tag(op.tag) {
                return plan.invalid("stage contains an unsupported grouped op");
            }
            if op.tag == tags::INTRINSIC_VAL {
                match op.intr {
                    intrinsic_tags::QUERY => plan.needs.query = true,
                    intrinsic_tags::LAYER => plan.needs.layer = true,
                    intrinsic_tags::ATTN_SCORE => plan.needs.attn_score = true,
                    intrinsic_tags::MTP_LOGITS => {
                        let value = value_bases[node] as usize;
                        let rows = match value_types.get(value).map(|ty| ty.dims.as_slice()) {
                            Some([Dimension::Static(rows), _]) => *rows,
                            _ => return plan.invalid("MtpLogits has no static draft-row layout"),
                        };
                        if plan.needs.mtp_rows && plan.mtp_rows != rows {
                            return plan.invalid(
                                "MtpLogits stages declare incompatible draft-row layouts",
                            );
                        }
                        plan.needs.mtp_rows = true;
                        plan.mtp_rows = rows;
                    }
                    intrinsic_tags::MTP_DRAFTS => {
                        plan.needs.mtp_drafts = true;
                        let value = value_bases[node] as usize;
                        if let Some([Dimension::Static(len)]) =
                            value_types.get(value).map(|ty| ty.dims.as_slice())
                        {
                            plan.drafts_len = plan.drafts_len.max(*len);
                        }
                    }
                    intrinsic_tags::LOGITS => {}
                    _ => return plan.invalid("stage uses an unsupported intrinsic"),
                }
            }

            let value = match op.tag {
                tags::CHAN_TAKE | tags::CHAN_READ => value_bases[node],
                tags::CHAN_PUT if !op.args.is_empty() => op.args[0],
                _ => continue,
            };
            if value as usize >= value_types.len()
                || op.chan < 0
                || op.chan as usize >= channel_count
            {
                return plan.invalid("channel value is outside the grouped plan");
            }
            plan.channel_rules.push(LaunchChannelRule {
                value,
                local: op.chan as u32,
            });
        }
        plan
    }

    /// The stage cannot take the grouped path. What was derived so far still
    /// describes it — the fused path reads it — so only the validity flag and
    /// the reason change.
    fn invalid(mut self, reason: &str) -> Self {
        self.needs.grouped_valid = false;
        self.error = reason.to_string();
        self
    }
}

/// Whether the grouped runtime has a body for this tag, answered from the
/// runtime source itself rather than a raw tag range or an exception list
/// (both of which silently misclassify a new tag).
fn grouped_supported_tag(tag: u8) -> bool {
    eta_ir::op::spec(tag).is_some()
}
