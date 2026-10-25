use serde::{Deserialize, Serialize};

use eta_ir::container::{ChanDType, ExternDir, HostRole, PortSource};
use eta_ir::op::{ChannelIndex, tags};
use eta_ir::registry::{Port, Stage};
use eta_ir::types::{Dtype, RngKind, ValueType, from_wire};
use eta_ir::validate::{BoundTrace, Direction};

use crate::plan::{
    CompiledStage, Dimension, NodeIndex, Region, RegionKind, RegionPartition, SymbolicExtent,
    SymbolicType, stage_identity,
};

use crate::codegen::fault::FaultClass;
use crate::codegen::op_view::OpView;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ValueOrigin {
    Const = 0,
    Output = 1,
    ChannelTake = 2,
    ChannelRead = 3,
    #[default]
    OpResult = 4,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchValue {
    pub id: u32,
    pub source: ValueOrigin,
    pub dtype: Dtype,
    /// The output it reads, by name.
    pub output: Option<String>,
    #[serde(default)]
    pub lane: u32,
    pub channel: ChannelIndex,
    pub literal_bits: u32,
    pub shape: Vec<u32>,
}

impl Default for LaunchValue {
    fn default() -> LaunchValue {
        LaunchValue {
            id: 0,
            source: ValueOrigin::OpResult,
            dtype: Dtype::F32,
            output: None,
            lane: 0,
            channel: 0,
            literal_bits: 0,
            shape: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchOp {
    pub tag: u8,
    pub result_count: u16,
    pub result_id: u32,
    pub lit_dtype: Dtype,
    pub dtype: Dtype,
    pub pred_tag: u8,
    pub pred_payload: u32,
    pub rng_kind: RngKind,
    pub lit_bits: u32,
    pub channel: Option<ChannelIndex>,
    pub name_index: u32,
    pub imm: u32,
    pub imm2: u32,
    pub imm3: u32,
    pub args: Vec<u32>,
    pub shape: Vec<u32>,
}

impl LaunchOp {
    /// Every value the op reads: its arguments, and a pivot's predicate
    /// operand, kept beside them in `pred_payload`.
    pub fn reads(&self) -> impl Iterator<Item = u32> + '_ {
        self.args
            .iter()
            .copied()
            .chain((self.tag == tags::PIVOT_THRESHOLD).then_some(self.pred_payload))
    }
}

impl Default for LaunchOp {
    fn default() -> LaunchOp {
        LaunchOp {
            tag: 0,
            result_count: 0,
            result_id: 0,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannel {
    pub id: ChannelIndex,
    pub capacity: u32,
    pub dtype: ChanDType,
    pub seeded: bool,
    pub host_role: HostRole,
    pub extern_dir: Option<ExternDir>,
    pub readiness: Option<Direction>,
    pub shape: Vec<u32>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPort {
    pub port: Port,
    pub is_const: bool,
    pub const_dtype: Dtype,
    pub channel: ChannelIndex,
    pub const_shape: Vec<u32>,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPut {
    pub channel: ChannelIndex,
    pub value: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStage {
    pub stage: Stage,
    pub ops: Vec<LaunchOp>,
    pub puts: Vec<LaunchPut>,
    pub takes: Vec<ChannelIndex>,
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRegion {
    pub kind: RegionKind,
    pub schedule: u8,
    pub nodes: Vec<u32>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub sinks: Vec<LaunchPut>,
    #[serde(default)]
    pub row_value: Option<u32>,
    #[serde(default)]
    pub spent: Vec<u32>,
    #[serde(default)]
    pub row_alias: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    pub dtype: Dtype,
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannelRule {
    pub value: u32,
    pub local: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageNeeds {
    pub kernel_call: bool,
    pub page_mask: bool,
    pub lora: bool,
    pub grouped_valid: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStagePlan {
    pub signature_hash: u64,
    pub identity: u64,
    pub needs: StageNeeds,
    /// The slots one lane takes in the stage's output table, and all of them.
    pub output_slots: u32,
    pub output_pitch: u32,
    pub ops: Vec<LaunchOp>,
    pub source_ops: Vec<Vec<u32>>,
    pub value_types: Vec<LaunchPlanValue>,
    pub value_map: Vec<u32>,
    pub channel_bindings: Vec<u32>,
    pub names: Vec<String>,
    pub singleton: Vec<LaunchRegion>,
    pub fused: Vec<LaunchRegion>,
    pub used_extents: Vec<SymbolicExtent>,
    pub channel_rules: Vec<LaunchChannelRule>,
    pub error: String,
}

/// The values one fold of the canonical reduction takes in (the
/// interpreter's `canonical_reduce`, a warp's lanes on a device).
pub const REDUCE_CHUNK: u64 = 32;

/// The values the first level of a `width`-wide row's reduction leaves.
#[must_use]
pub const fn reduction_chunks(width: u64) -> u64 {
    width.div_ceil(REDUCE_CHUNK)
}

/// The floats an f32 reduction of a `width`-wide row keeps in a block's
/// temporary arena: its first level's values, then room as large again for
/// the levels above it.
#[must_use]
pub const fn reduction_floats(width: u64) -> u64 {
    2 * reduction_chunks(width)
}

/// The bytes a block of the generated region `region` keeps in its
/// temporary arena over rows at most `widest` wide: [`reduction_floats`] for
/// each f32 reduction it holds, which a row stream may hold all at once,
/// each at its own slot.
#[must_use]
pub fn generated_temporary_bytes(
    plan: &LaunchStagePlan,
    region: &LaunchRegion,
    widest: u64,
) -> u64 {
    let reductions = region
        .nodes
        .iter()
        .filter_map(|&node| plan.ops.get(node as usize))
        .filter(|op| {
            matches!(
                op.tag,
                tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN
            ) && op
                .args
                .first()
                .and_then(|&value| plan.value_types.get(value as usize))
                .is_some_and(|value| value.dtype == Dtype::F32)
        })
        .count() as u64;
    4 * reduction_floats(widest) * reductions.max(1)
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPackage {
    pub values: Vec<LaunchValue>,
    pub channels: Vec<LaunchChannel>,
    pub ports: Vec<LaunchPort>,
    pub inputs: Vec<ChannelIndex>,
    pub names: Vec<String>,
    pub stages: Vec<LaunchStage>,
    pub plans: Vec<LaunchStagePlan>,
    pub fault_classes: Vec<FaultClass>,
}

impl LaunchPackage {
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

pub fn build(bound: &BoundTrace, stages: &[CompiledStage]) -> LaunchPackage {
    LaunchPackage {
        values: lower_values(bound),
        channels: lower_channels(bound),
        ports: lower_ports(bound),
        inputs: bound.container.inputs.clone(),
        names: bound.container.names.clone(),
        stages: lower_stages(bound),
        plans: stages.iter().map(lower_plan).collect(),
        fault_classes: crate::codegen::fault::classes(),
    }
}

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
                tags::OUTPUT => {
                    push(
                        local,
                        LaunchValue {
                            source: ValueOrigin::Output,
                            output: bound
                                .container
                                .names
                                .get(usize::from(view.name_idx))
                                .cloned(),
                            lane: view.imm,
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
            let readiness = bound
                .readiness
                .iter()
                .find(|entry| entry.chan as usize == index)
                .map(|entry| entry.dir);
            LaunchChannel {
                id: index as u32,
                capacity: decl.capacity,
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
                tags::CONST | tags::OUTPUT => local += 1,
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
                        lit_dtype: dtype(view.lit_dtype),
                        dtype: value_type.dtype,
                        pred_tag: view.pred_tag,
                        rng_kind: rng(view.kind),
                        lit_bits: view.lit_bits,
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
        output_slots: eta_ir::op::output_table(&normalized.ops).0,
        output_pitch: eta_ir::op::output_pitch(&normalized.ops),
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
        singleton: lower_partition(stage, &stage.singleton),
        fused: lower_partition(stage, &stage.fused),
        used_extents: grouped.used_extents,
        channel_rules: grouped.channel_rules,
        error: grouped.error,
    }
}

fn lower_plan_op(view: &OpView) -> LaunchOp {
    LaunchOp {
        tag: view.tag,
        result_count: view.results as u16,
        result_id: 0,
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

fn lower_plan_value(value_type: &SymbolicType) -> LaunchPlanValue {
    LaunchPlanValue {
        dtype: value_type.dtype,
        axes: value_type.dims.clone(),
    }
}

fn dtype(byte: u8) -> Dtype {
    from_wire(byte).unwrap_or(Dtype::F32)
}

fn rng(byte: u8) -> RngKind {
    if byte == RngKind::Gumbel as u8 {
        RngKind::Gumbel
    } else if byte == RngKind::Normal as u8 {
        RngKind::Normal
    } else {
        RngKind::Uniform
    }
}

fn lower_partition(stage: &CompiledStage, partition: &RegionPartition) -> Vec<LaunchRegion> {
    partition
        .regions
        .iter()
        .map(|region| lower_region(stage, region))
        .collect()
}

fn lower_region(stage: &CompiledStage, region: &Region) -> LaunchRegion {
    LaunchRegion {
        spent: crate::codegen::cuda::spent_values(stage, region),
        kind: region.kind,
        schedule: region.schedule as u8,
        nodes: region.nodes.iter().copied().map(NodeIndex::get).collect(),
        inputs: region.inputs.clone(),
        outputs: region.outputs.clone(),
        row_value: region.row_value,
        row_alias: region.row_alias.map(|(_, count)| count),
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

#[derive(Default)]
struct GroupedPlan {
    needs: StageNeeds,
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
            if op.tag == tags::KERNEL_CALL {
                plan.needs.kernel_call = true;
                continue;
            }
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

    fn invalid(mut self, reason: &str) -> Self {
        self.needs.grouped_valid = false;
        self.error = reason.to_string();
        self
    }
}

fn grouped_supported_tag(tag: u8) -> bool {
    eta_ir::op::spec(tag).is_some()
}
