//! Backend-neutral PTIR compiler planning.
//!
//! Rust owns normalization, stage signatures, value-domain analysis, region
//! partitioning, and the lane-table ABI. Drivers consume the serialized plan
//! and provide backend code generation and library implementations.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::container::{PortSource, encode_op, put_u16, put_u32};
use crate::op::{IntrinsicId, Op};
use crate::registry::{Port, Stage};
use crate::types::{DType, Literal, MAX_RANK, Predicate, RngKind, Shape, ValueType};
use crate::validate::BoundTrace;

pub const COMPILER_VERSION: u16 = 3;
pub const REGION_PLAN_VERSION: u16 = 4;
pub const LANE_TABLE_ABI_VERSION: u32 = 3;

const SIGNATURE_MAGIC: [u8; 4] = *b"PTSG";
const PLAN_MAGIC: [u8; 4] = *b"PTRP";

/// Runtime-varying dimensions represented symbolically in compiler types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SymbolicExtent {
    KvLen = 0,
    PageCount = 1,
    RowCount = 2,
    TokenCount = 3,
    SampledRows = 4,
    QueryLen = 5,
    KeyLen = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    Static(u32),
    Symbolic(SymbolicExtent),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolicType {
    pub dtype: DType,
    pub dims: Vec<Dimension>,
}

impl SymbolicType {
    fn static_type(value_type: ValueType) -> Self {
        Self {
            dtype: value_type.dtype,
            dims: value_type
                .shape
                .dims()
                .iter()
                .copied()
                .map(Dimension::Static)
                .collect(),
        }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
            || self
                .dims
                .iter()
                .all(|dimension| *dimension == Dimension::Static(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueDomain {
    Scalar = 0,
    PerRow = 1,
    Vocabulary = 2,
    GeneratedIndex = 3,
    Mask = 4,
    PageDescriptor = 5,
    LibraryResult = 6,
    EffectToken = 7,
}

/// A normalized stage with local channel/name numbering.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedStage {
    pub stage: Stage,
    pub source_op_count: u32,
    pub ops: Vec<Op>,
    pub value_types: Vec<SymbolicType>,
    pub value_domains: Vec<ValueDomain>,
    /// Original PTIR op positions represented by each normalized op.
    pub source_ops: Vec<Vec<u32>>,
    /// Local channel slot -> program-global dense channel index.
    pub channel_bindings: Vec<u32>,
    /// Local name slot -> canonical second-party name.
    pub names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageSignature {
    pub hash: u64,
    pub canonical_bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ScheduleTemplate {
    Effects = 0,
    OneCtaPerRow = 1,
    HierarchicalRow = 2,
    Library = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LibraryOp {
    NucleusSample = 0,
    TopK = 1,
    Sort = 2,
    Scan = 3,
    MatMul = 4,
    SecondParty = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Generated,
    Library(LibraryOp),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSink {
    pub channel_slot: u32,
    pub value: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub kind: RegionKind,
    pub schedule: ScheduleTemplate,
    pub nodes: Vec<u32>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub sinks: Vec<ChannelSink>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LibraryMatch {
    library: LibraryOp,
    nodes: Vec<u32>,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PartitionKind {
    Singleton = 0,
    Fused = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionPartition {
    pub kind: PartitionKind,
    pub regions: Vec<Region>,
    /// Legacy wire bit retained for decoder compatibility. Revision 6 plans
    /// never request whole-stage fallback.
    pub whole_stage_fallback: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompiledStage {
    pub normalized: NormalizedStage,
    pub signature: StageSignature,
    pub singleton: RegionPartition,
    pub fused: RegionPartition,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanMetrics {
    pub source_ops: u32,
    pub normalized_ops: u32,
    pub singleton_regions: u32,
    pub fused_regions: u32,
    pub library_regions: u32,
    pub static_scratch_bytes: u64,
    pub direct_channel_sink_bytes: u64,
}

impl CompiledStage {
    pub fn metrics(&self) -> PlanMetrics {
        let static_bytes = |value_type: &SymbolicType| {
            let mut elements = 1u64;
            for dimension in &value_type.dims {
                let Dimension::Static(dimension) = dimension else {
                    return 0;
                };
                elements = elements.saturating_mul(*dimension as u64);
            }
            elements.saturating_mul(match value_type.dtype {
                DType::Bool => 1,
                _ => 4,
            })
        };
        let direct_values: BTreeSet<u32> = self
            .fused
            .regions
            .iter()
            .flat_map(|region| region.sinks.iter().map(|sink| sink.value))
            .collect();
        let direct_channel_sink_bytes = direct_values
            .iter()
            .filter_map(|value| self.normalized.value_types.get(*value as usize))
            .map(static_bytes)
            .sum();
        let static_scratch_bytes = self
            .normalized
            .value_types
            .iter()
            .enumerate()
            .filter(|(value, _)| !direct_values.contains(&(*value as u32)))
            .map(|(_, value_type)| static_bytes(value_type))
            .sum();
        PlanMetrics {
            source_ops: self.normalized.source_op_count,
            normalized_ops: self.normalized.ops.len() as u32,
            singleton_regions: self.singleton.regions.len() as u32,
            fused_regions: self.fused.regions.len() as u32,
            library_regions: self
                .fused
                .regions
                .iter()
                .filter(|region| matches!(region.kind, RegionKind::Library(_)))
                .count() as u32,
            static_scratch_bytes,
            direct_channel_sink_bytes,
        }
    }
}

/// Runtime extents never enter a stage signature.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeExtents {
    pub kv_len: u32,
    pub page_count: u32,
    pub row_count: u32,
    pub token_count: u32,
    pub sampled_rows: u32,
    pub query_len: u32,
    pub key_len: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScheduleBucket {
    pub row_bucket: u8,
    pub lane_bucket: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum BackendKind {
    Cuda = 0,
    Metal = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum SemanticMode {
    Exact = 0,
}

/// Complete executable-cache identity. `device_arch` is the backend's stable
/// architecture identifier (for example CUDA SM or Metal GPU family).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExecutableCacheKey {
    pub backend: BackendKind,
    pub device_arch: u64,
    pub compiler_version: u16,
    pub stage_signature: u64,
    pub schedule_bucket: ScheduleBucket,
    pub semantic_mode: SemanticMode,
}

impl ExecutableCacheKey {
    pub fn encode(self) -> [u8; 22] {
        let mut bytes = [0u8; 22];
        bytes[0] = self.backend as u8;
        bytes[1..9].copy_from_slice(&self.device_arch.to_le_bytes());
        bytes[9..11].copy_from_slice(&self.compiler_version.to_le_bytes());
        bytes[11..19].copy_from_slice(&self.stage_signature.to_le_bytes());
        bytes[19] = self.schedule_bucket.row_bucket;
        bytes[20] = self.schedule_bucket.lane_bucket;
        bytes[21] = self.semantic_mode as u8;
        bytes
    }
}

impl ScheduleBucket {
    pub fn for_dispatch(lane_count: u32, extents: RuntimeExtents) -> Self {
        let rows = extents.sampled_rows.max(extents.row_count).max(1);
        Self {
            row_bucket: power_of_two_bucket(rows),
            lane_bucket: power_of_two_bucket(lane_count.max(1)),
        }
    }
}

fn power_of_two_bucket(value: u32) -> u8 {
    (u32::BITS - value.saturating_sub(1).leading_zeros()) as u8
}

/// Stable grouped-dispatch header. Address fields in the lane records are
/// device virtual addresses represented as `u64` on both supported backends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneTableHeader {
    pub abi_version: u32,
    pub lane_count: u32,
    pub channel_slots_per_lane: u32,
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneRecord {
    pub logits_base: u64,
    pub logits_row_offset: u32,
    pub logits_row_count: u32,
    pub kv_len: u32,
    pub page_count: u32,
    pub row_count: u32,
    pub token_count: u32,
    pub sampled_rows: u32,
    pub query_len: u32,
    pub key_len: u32,
    pub channel_slot_offset: u32,
    pub rng_state: u64,
    pub commit_slot: u64,
    /// Optional device bitset for active rows in a ragged lane; zero means all
    /// `logits_row_count` rows are active.
    pub active_row_mask: u64,
    /// Stage-local channel bits whose puts publish the sampled token value.
    pub sample_output_channel_mask: u64,
    /// Optional device byte mask for model rows. `row_valid_offset` selects the
    /// first row belonging to this program.
    pub row_valid: u64,
    pub row_valid_offset: u32,
    pub reserved0: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct LaneChannelSlot {
    pub committed_cell: u64,
    pub pending_cell: u64,
    pub expected_head: u64,
    pub expected_tail: u64,
}

/// Compile every stage in container order.
pub fn compile_bound(bound: &BoundTrace) -> Vec<CompiledStage> {
    (0..bound.container.stages.len())
        .map(|stage_index| compile_stage_at(bound, stage_index))
        .collect()
}

pub fn compile_stage(bound: &BoundTrace, stage: Stage) -> Option<CompiledStage> {
    let stage_index = bound
        .container
        .stages
        .iter()
        .position(|program| program.stage == stage)?;
    Some(compile_stage_at(bound, stage_index))
}

pub fn compile_stage_at(bound: &BoundTrace, stage_index: usize) -> CompiledStage {
    let mut normalized = normalize_stage(bound, stage_index);
    localize_stage(bound, &mut normalized);
    let signature = stage_signature(bound, &normalized);
    let singleton = singleton_partition(&normalized);
    let fused = fused_partition(&normalized, &recognize_library_dataflows(&normalized));
    CompiledStage {
        normalized,
        signature,
        singleton,
        fused,
    }
}

/// Human-readable normalized DAG and partition dump for diagnostics without a
/// backend or GPU.
pub fn debug_stage_plan(stage: &CompiledStage) -> String {
    use core::fmt::Write;

    let mut output = String::new();
    let _ = writeln!(
        output,
        "{} signature={:016x} ops={} values={}",
        stage.normalized.stage.name(),
        stage.signature.hash,
        stage.normalized.ops.len(),
        stage.normalized.value_types.len()
    );
    let (bases, _) = result_layout(&stage.normalized.ops);
    for (node, op) in stage.normalized.ops.iter().enumerate() {
        let _ = writeln!(
            output,
            "  n{node} v{} +{} {:?} <- {:?} source={:?}",
            bases[node],
            op.result_count(),
            op,
            op.operands(),
            stage.normalized.source_ops[node]
        );
    }
    for partition in [&stage.singleton, &stage.fused] {
        let _ = writeln!(
            output,
            "  {:?} fallback={} regions={}",
            partition.kind,
            partition.whole_stage_fallback,
            partition.regions.len()
        );
        for (index, region) in partition.regions.iter().enumerate() {
            let _ = writeln!(
                output,
                "    r{index} {:?}/{:?} nodes={:?} in={:?} out={:?} sinks={:?}",
                region.kind,
                region.schedule,
                region.nodes,
                region.inputs,
                region.outputs,
                region.sinks
            );
        }
    }
    output
}

fn normalize_stage(bound: &BoundTrace, stage_index: usize) -> NormalizedStage {
    let stage_program = &bound.container.stages[stage_index];
    let original_types = &bound.stage_types[stage_index];
    let (result_bases, producer) = result_layout(&stage_program.ops);
    let keep = live_ops(stage_program, &result_bases, &producer);

    let mut value_map = vec![u32::MAX; original_types.len()];
    let mut normalized_ops: Vec<Op> = Vec::new();
    let mut normalized_types: Vec<SymbolicType> = Vec::new();
    let mut normalized_domains: Vec<ValueDomain> = Vec::new();
    let mut source_ops: Vec<Vec<u32>> = Vec::new();
    let mut literals: Vec<Option<Literal>> = Vec::new();
    let mut cse: BTreeMap<Vec<u8>, (u32, u32)> = BTreeMap::new();

    for (op_index, original_op) in stage_program.ops.iter().enumerate() {
        let base = result_bases[op_index] as usize;
        let result_count = original_op.result_count() as usize;
        if !keep[op_index] {
            continue;
        }

        let mut op = original_op.clone();
        op.map_operands(|value| {
            let mapped = value_map[value as usize];
            debug_assert_ne!(mapped, u32::MAX, "live op references removed value");
            mapped
        });

        let result_types: Vec<SymbolicType> = (0..result_count)
            .map(|result| {
                symbolic_result_type(
                    bound,
                    original_op,
                    original_types[base + result],
                    &op,
                    original_types,
                    &normalized_types,
                )
            })
            .collect();

        if result_count == 1 {
            if let Some(alias) = simplify_alias(&op, &result_types[0], &literals) {
                value_map[base] = alias;
                continue;
            }
            if let Some(literal) = fold_scalar(&op, &literals) {
                op = Op::Const(literal);
            }
        }

        canonicalize_commutative(&mut op, result_types.first());

        let cse_key = if cse_candidate(&op) {
            Some(cse_key(&op, &result_types))
        } else {
            None
        };
        if let Some(key) = cse_key.as_ref() {
            if let Some(&(existing_base, existing_op)) = cse.get(key) {
                for result in 0..result_count {
                    value_map[base + result] = existing_base + result as u32;
                }
                source_ops[existing_op as usize].push(op_index as u32);
                continue;
            }
        }

        let new_base = normalized_types.len() as u32;
        let normalized_op_index = normalized_ops.len() as u32;
        for (result, symbolic_type) in result_types.into_iter().enumerate() {
            value_map[base + result] = new_base + result as u32;
            normalized_domains.push(value_domain(bound, &op, &symbolic_type));
            let literal = match &op {
                Op::Const(literal) => Some(*literal),
                _ => None,
            };
            literals.push(literal);
            normalized_types.push(symbolic_type);
        }
        normalized_ops.push(op);
        source_ops.push(vec![op_index as u32]);
        if let Some(key) = cse_key {
            cse.insert(key, (new_base, normalized_op_index));
        }
    }

    NormalizedStage {
        stage: stage_program.stage,
        source_op_count: stage_program.ops.len() as u32,
        ops: normalized_ops,
        value_types: normalized_types,
        value_domains: normalized_domains,
        source_ops,
        channel_bindings: Vec::new(),
        names: Vec::new(),
    }
}

fn result_layout(ops: &[Op]) -> (Vec<u32>, Vec<usize>) {
    let mut bases = Vec::with_capacity(ops.len());
    let mut producer = Vec::new();
    let mut next = 0u32;
    for (op_index, op) in ops.iter().enumerate() {
        bases.push(next);
        for _ in 0..op.result_count() {
            producer.push(op_index);
            next += 1;
        }
    }
    (bases, producer)
}

fn live_ops(
    stage_program: &crate::container::StageProgram,
    result_bases: &[u32],
    producer: &[usize],
) -> Vec<bool> {
    let mut keep = vec![false; stage_program.ops.len()];
    let mut values = Vec::new();
    for (op_index, op) in stage_program.ops.iter().enumerate() {
        if matches!(
            op,
            Op::ChanTake(_)
                | Op::ChanRead(_)
                | Op::ChanPut { .. }
                | Op::KernelCall { .. }
                | Op::SinkCall { .. }
        ) {
            keep[op_index] = true;
            values.extend(op.operands());
        }
    }
    while let Some(value) = values.pop() {
        let op_index = producer[value as usize];
        if !keep[op_index] {
            keep[op_index] = true;
            values.extend(stage_program.ops[op_index].operands());
        }
    }

    // A kept multi-result producer keeps all of its positional results. The
    // bases are intentionally consumed here to assert the layout remains valid.
    debug_assert_eq!(
        result_bases.last().copied().unwrap_or(0)
            + stage_program.ops.last().map(Op::result_count).unwrap_or(0),
        producer.len() as u32
    );
    keep
}

fn symbolic_result_type(
    bound: &BoundTrace,
    original_op: &Op,
    value_type: ValueType,
    mapped_op: &Op,
    original_types: &[ValueType],
    normalized_types: &[SymbolicType],
) -> SymbolicType {
    match original_op {
        Op::ChanTake(channel) | Op::ChanRead(channel) => {
            symbolic_channel_type(bound, *channel, value_type)
        }
        Op::IntrinsicVal { intr, .. } => symbolic_intrinsic_type(bound, *intr, value_type),
        Op::ReduceSum(value)
        | Op::ReduceMax(value)
        | Op::ReduceMin(value)
        | Op::ReduceArgmax(value) => {
            let mapped = mapped_value(mapped_op, *value);
            let mut ty = normalized_types[mapped as usize].clone();
            ty.dims.pop();
            ty.dtype = value_type.dtype;
            ty
        }
        Op::Transpose(value) => {
            let mapped = mapped_value(mapped_op, *value);
            let mut ty = normalized_types[mapped as usize].clone();
            if ty.dims.len() == 2 {
                ty.dims.swap(0, 1);
            }
            ty.dtype = value_type.dtype;
            ty
        }
        Op::Gather { .. } => {
            let operands = mapped_op.operands();
            let src = &normalized_types[operands[0] as usize];
            let index = &normalized_types[operands[1] as usize];
            let mut dims = index.dims.clone();
            dims.extend_from_slice(&src.dims[1..]);
            SymbolicType {
                dtype: value_type.dtype,
                dims,
            }
        }
        Op::GatherRow { .. } => {
            let index = mapped_op.operands()[1];
            SymbolicType {
                dtype: value_type.dtype,
                dims: normalized_types[index as usize].dims.clone(),
            }
        }
        Op::ScatterAdd { .. } | Op::ScatterSet { .. } => {
            let base = mapped_op.operands()[0];
            let mut ty = normalized_types[base as usize].clone();
            ty.dtype = value_type.dtype;
            ty
        }
        Op::MaskApply { .. } => {
            let logits = mapped_op.operands()[0];
            normalized_types[logits as usize].clone()
        }
        Op::SortDesc(_) | Op::CumSum(_) | Op::CumProd(_) => {
            let input = mapped_op.operands()[0];
            let mut ty = normalized_types[input as usize].clone();
            ty.dtype = value_type.dtype;
            ty
        }
        Op::TopK { k, .. } => {
            let input = mapped_op.operands()[0];
            let mut ty = normalized_types[input as usize].clone();
            if let Some(last) = ty.dims.last_mut() {
                *last = Dimension::Static(*k);
            }
            ty.dtype = value_type.dtype;
            ty
        }
        Op::MatMul(_, _) => {
            let operands = mapped_op.operands();
            let left = &normalized_types[operands[0] as usize];
            let right = &normalized_types[operands[1] as usize];
            SymbolicType {
                dtype: value_type.dtype,
                dims: vec![left.dims[0], *right.dims.last().expect("matmul right rank")],
            }
        }
        Op::CausalMask { positions, .. }
        | Op::SlidingWindowMask { positions, .. }
        | Op::SinkWindowMask { positions, .. } => {
            let mapped = mapped_op.operands()[0];
            let source = &normalized_types[mapped as usize];
            let mut ty = SymbolicType::static_type(value_type);
            propagate_preserved_dimensions(
                &mut ty,
                source,
                original_types[*positions as usize],
                value_type,
            );
            ty
        }
        Op::Broadcast { value, .. } => {
            let mapped = mapped_value(mapped_op, *value);
            let source = &normalized_types[mapped as usize];
            let mut ty = SymbolicType::static_type(value_type);
            propagate_preserved_dimensions(
                &mut ty,
                source,
                original_types[*value as usize],
                value_type,
            );
            ty
        }
        Op::Reshape { value, .. } => {
            let mapped = mapped_value(mapped_op, *value);
            let source = &normalized_types[mapped as usize];
            let mut ty = SymbolicType::static_type(value_type);
            propagate_preserved_dimensions(
                &mut ty,
                source,
                original_types[*value as usize],
                value_type,
            );
            ty
        }
        _ => {
            let mut ty = SymbolicType::static_type(value_type);
            if let Some((original, mapped)) = original_op
                .operands()
                .into_iter()
                .zip(mapped_op.operands())
                .find(|(_, mapped)| {
                    normalized_types
                        .get(*mapped as usize)
                        .is_some_and(|source| source.rank() == ty.rank())
                })
            {
                propagate_preserved_dimensions(
                    &mut ty,
                    &normalized_types[mapped as usize],
                    original_types[original as usize],
                    value_type,
                );
            }
            ty
        }
    }
}

fn propagate_preserved_dimensions(
    target: &mut SymbolicType,
    source: &SymbolicType,
    source_static: ValueType,
    target_static: ValueType,
) {
    for (index, source_dimension) in source.dims.iter().enumerate() {
        if index < target.dims.len()
            && matches!(source_dimension, Dimension::Symbolic(_))
            && source_static.shape.dims().get(index) == target_static.shape.dims().get(index)
        {
            target.dims[index] = *source_dimension;
        }
    }
}

fn mapped_value(mapped_op: &Op, original_value: u32) -> u32 {
    let original_operands = match mapped_op {
        Op::ReduceSum(value)
        | Op::ReduceMax(value)
        | Op::ReduceMin(value)
        | Op::ReduceArgmax(value)
        | Op::Transpose(value)
        | Op::Broadcast { value, .. }
        | Op::Reshape { value, .. } => *value,
        _ => original_value,
    };
    original_operands
}

fn symbolic_channel_type(
    _bound: &BoundTrace,
    _channel: u32,
    value_type: ValueType,
) -> SymbolicType {
    SymbolicType::static_type(value_type)
}

fn symbolic_port_type(port: Port, value_type: ValueType) -> SymbolicType {
    let mut ty = SymbolicType::static_type(value_type);
    match port {
        Port::EmbedTokens | Port::Positions | Port::WSlot | Port::WOff => {
            set_first_symbolic(&mut ty, SymbolicExtent::TokenCount)
        }
        Port::Pages => set_first_symbolic(&mut ty, SymbolicExtent::PageCount),
        Port::PageIndptr => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
        Port::KvLen => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
        Port::Readout => set_first_symbolic(&mut ty, SymbolicExtent::SampledRows),
        Port::AttnMask => {
            if !ty.dims.is_empty() {
                ty.dims[0] = Dimension::Symbolic(SymbolicExtent::QueryLen);
            }
            if ty.dims.len() > 1 {
                let last = ty.dims.len() - 1;
                ty.dims[last] = Dimension::Symbolic(SymbolicExtent::KeyLen);
            }
        }
        Port::EmbedIndptr => set_first_symbolic(&mut ty, SymbolicExtent::RowCount),
    }
    ty
}

fn set_first_symbolic(ty: &mut SymbolicType, extent: SymbolicExtent) {
    if let Some(first) = ty.dims.first_mut() {
        *first = Dimension::Symbolic(extent);
    }
}

fn symbolic_intrinsic_type(
    bound: &BoundTrace,
    intrinsic: IntrinsicId,
    value_type: ValueType,
) -> SymbolicType {
    let mut ty = SymbolicType::static_type(value_type);
    match intrinsic {
        IntrinsicId::Logits => {
            if ty.rank() >= 2 && value_type.shape.last_len() == Some(bound.profile.vocab) {
                ty.dims[0] = Dimension::Symbolic(SymbolicExtent::SampledRows);
            }
        }
        _ => {}
    }
    ty
}

fn simplify_alias(
    op: &Op,
    result_type: &SymbolicType,
    literals: &[Option<Literal>],
) -> Option<u32> {
    if result_type
        .dims
        .iter()
        .all(|dim| *dim == Dimension::Static(1))
    {
        match *op {
            Op::CumSum(value) | Op::CumProd(value) => return Some(value),
            _ => {}
        }
    }
    if result_type.dtype == DType::F32 {
        return None;
    }
    let literal = |value: u32| literals.get(value as usize).copied().flatten();
    match *op {
        Op::Add(a, b) if is_zero(literal(b)) => Some(a),
        Op::Add(a, b) if is_zero(literal(a)) => Some(b),
        Op::Sub(a, b) if is_zero(literal(b)) => Some(a),
        Op::Mul(a, b) if is_one(literal(b)) => Some(a),
        Op::Mul(a, b) if is_one(literal(a)) => Some(b),
        Op::Div(a, b) if is_one(literal(b)) => Some(a),
        Op::And(a, b) if literal(b) == Some(Literal::Bool(true)) => Some(a),
        Op::And(a, b) if literal(a) == Some(Literal::Bool(true)) => Some(b),
        Op::Or(a, b) if literal(b) == Some(Literal::Bool(false)) => Some(a),
        Op::Or(a, b) if literal(a) == Some(Literal::Bool(false)) => Some(b),
        _ => None,
    }
}

fn is_zero(literal: Option<Literal>) -> bool {
    matches!(literal, Some(Literal::I32(0) | Literal::U32(0)))
}

fn is_one(literal: Option<Literal>) -> bool {
    matches!(literal, Some(Literal::I32(1) | Literal::U32(1)))
}

fn fold_scalar(op: &Op, literals: &[Option<Literal>]) -> Option<Literal> {
    let get = |value: u32| literals.get(value as usize).copied().flatten();
    match *op {
        Op::Neg(value) => match get(value)? {
            Literal::I32(value) => Some(Literal::I32(value.wrapping_neg())),
            Literal::U32(value) => Some(Literal::U32(value.wrapping_neg())),
            _ => None,
        },
        Op::Abs(value) => match get(value)? {
            Literal::I32(value) => Some(Literal::I32(value.wrapping_abs())),
            Literal::U32(value) => Some(Literal::U32(value)),
            _ => None,
        },
        Op::Sign(value) => match get(value)? {
            Literal::I32(value) => Some(Literal::I32(value.signum())),
            Literal::U32(value) => Some(Literal::U32(u32::from(value != 0))),
            _ => None,
        },
        Op::Not(value) => match get(value)? {
            Literal::Bool(value) => Some(Literal::Bool(!value)),
            _ => None,
        },
        Op::Add(a, b) => fold_int_binary(get(a)?, get(b)?, i32::wrapping_add, u32::wrapping_add),
        Op::Sub(a, b) => fold_int_binary(get(a)?, get(b)?, i32::wrapping_sub, u32::wrapping_sub),
        Op::Mul(a, b) => fold_int_binary(get(a)?, get(b)?, i32::wrapping_mul, u32::wrapping_mul),
        Op::Div(a, b) => match (get(a)?, get(b)?) {
            (Literal::I32(a), Literal::I32(b)) => {
                Some(Literal::I32(if b == 0 { 0 } else { a.wrapping_div(b) }))
            }
            (Literal::U32(a), Literal::U32(b)) => {
                Some(Literal::U32(if b == 0 { 0 } else { a / b }))
            }
            _ => None,
        },
        Op::Rem(a, b) => match (get(a)?, get(b)?) {
            (Literal::I32(a), Literal::I32(b)) => {
                Some(Literal::I32(if b == 0 { 0 } else { a.wrapping_rem(b) }))
            }
            (Literal::U32(a), Literal::U32(b)) => {
                Some(Literal::U32(if b == 0 { 0 } else { a % b }))
            }
            _ => None,
        },
        Op::MaxElem(a, b) => fold_ordered(get(a)?, get(b)?, true),
        Op::MinElem(a, b) => fold_ordered(get(a)?, get(b)?, false),
        Op::Eq(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering == 0),
        Op::Ne(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering != 0),
        Op::Lt(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering < 0),
        Op::Le(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering <= 0),
        Op::Gt(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering > 0),
        Op::Ge(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering >= 0),
        Op::And(a, b) => match (get(a)?, get(b)?) {
            (Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(a && b)),
            _ => None,
        },
        Op::Or(a, b) => match (get(a)?, get(b)?) {
            (Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(a || b)),
            _ => None,
        },
        _ => None,
    }
}

fn fold_int_binary(
    a: Literal,
    b: Literal,
    signed: fn(i32, i32) -> i32,
    unsigned: fn(u32, u32) -> u32,
) -> Option<Literal> {
    match (a, b) {
        (Literal::I32(a), Literal::I32(b)) => Some(Literal::I32(signed(a, b))),
        (Literal::U32(a), Literal::U32(b)) => Some(Literal::U32(unsigned(a, b))),
        _ => None,
    }
}

fn fold_ordered(a: Literal, b: Literal, maximum: bool) -> Option<Literal> {
    match (a, b) {
        (Literal::I32(a), Literal::I32(b)) => {
            Some(Literal::I32(if maximum { a.max(b) } else { a.min(b) }))
        }
        (Literal::U32(a), Literal::U32(b)) => {
            Some(Literal::U32(if maximum { a.max(b) } else { a.min(b) }))
        }
        _ => None,
    }
}

fn fold_compare(a: Literal, b: Literal, predicate: impl FnOnce(i8) -> bool) -> Option<Literal> {
    let ordering = match (a, b) {
        (Literal::I32(a), Literal::I32(b)) => a.cmp(&b),
        (Literal::U32(a), Literal::U32(b)) => a.cmp(&b),
        (Literal::Bool(a), Literal::Bool(b)) => a.cmp(&b),
        _ => return None,
    };
    let ordering = match ordering {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    };
    Some(Literal::Bool(predicate(ordering)))
}

fn canonicalize_commutative(op: &mut Op, result_type: Option<&SymbolicType>) {
    if result_type.is_some_and(|result_type| result_type.dtype == DType::F32) {
        return;
    }
    match op {
        Op::Add(a, b)
        | Op::Mul(a, b)
        | Op::MaxElem(a, b)
        | Op::MinElem(a, b)
        | Op::Eq(a, b)
        | Op::Ne(a, b)
        | Op::And(a, b)
        | Op::Or(a, b) => {
            if *a > *b {
                core::mem::swap(a, b);
            }
        }
        _ => {}
    }
}

fn cse_candidate(op: &Op) -> bool {
    !matches!(
        op,
        Op::ChanTake(_)
            | Op::ChanRead(_)
            | Op::ChanPut { .. }
            | Op::KernelCall { .. }
            | Op::SinkCall { .. }
    )
}

fn cse_key(op: &Op, result_types: &[SymbolicType]) -> Vec<u8> {
    let mut bytes = Vec::new();
    encode_op(&mut bytes, op);
    put_u32(&mut bytes, result_types.len() as u32);
    for value_type in result_types {
        encode_symbolic_type(&mut bytes, value_type);
    }
    bytes
}

fn value_domain(bound: &BoundTrace, op: &Op, value_type: &SymbolicType) -> ValueDomain {
    if value_type.is_scalar() {
        return ValueDomain::Scalar;
    }
    if matches!(op, Op::Iota { .. }) {
        return ValueDomain::GeneratedIndex;
    }
    if value_type.dtype == DType::Bool {
        return ValueDomain::Mask;
    }
    if value_type
        .dims
        .last()
        .is_some_and(|dimension| *dimension == Dimension::Static(bound.profile.vocab))
    {
        return ValueDomain::Vocabulary;
    }
    if matches!(
        op,
        Op::TopK { .. } | Op::SortDesc(_) | Op::MatMul(_, _) | Op::KernelCall { .. }
    ) {
        return ValueDomain::LibraryResult;
    }
    if matches!(
        op,
        Op::ReduceSum(_) | Op::ReduceMax(_) | Op::ReduceMin(_) | Op::ReduceArgmax(_)
    ) {
        return ValueDomain::PerRow;
    }
    ValueDomain::PerRow
}

fn localize_stage(bound: &BoundTrace, stage: &mut NormalizedStage) {
    let mut channels = Vec::new();
    let mut names = Vec::new();
    for op in &mut stage.ops {
        match op {
            Op::ChanTake(channel) | Op::ChanRead(channel) => {
                *channel = local_channel(&mut channels, *channel)
            }
            Op::ChanPut { chan, .. } => *chan = local_channel(&mut channels, *chan),
            Op::KernelCall { name, .. } | Op::SinkCall { name, .. } => {
                *name = local_name(&bound.container.names, &mut names, *name)
            }
            _ => {}
        }
    }
    for port in signature_ports(bound, stage.stage) {
        if let PortSource::Channel(channel) = port.source {
            local_channel(&mut channels, channel);
        }
    }

    stage.channel_bindings = channels;
    stage.names = names;
}

fn signature_ports(
    bound: &BoundTrace,
    stage: Stage,
) -> impl Iterator<Item = &crate::container::PortBinding> {
    bound
        .container
        .ports
        .iter()
        .filter(move |_| stage != Stage::Epilogue)
}

fn local_channel(channels: &mut Vec<u32>, global: u32) -> u32 {
    if let Some(local) = channels.iter().position(|channel| *channel == global) {
        local as u32
    } else {
        channels.push(global);
        (channels.len() - 1) as u32
    }
}

fn local_name(global_names: &[String], names: &mut Vec<String>, global: u16) -> u16 {
    let name = &global_names[global as usize];
    if let Some(local) = names.iter().position(|candidate| candidate == name) {
        local as u16
    } else {
        names.push(name.clone());
        (names.len() - 1) as u16
    }
}

fn stage_signature(bound: &BoundTrace, stage: &NormalizedStage) -> StageSignature {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SIGNATURE_MAGIC);
    put_u16(&mut bytes, COMPILER_VERSION);
    bytes.push(stage.stage as u8);

    put_u32(&mut bytes, stage.channel_bindings.len() as u32);
    for &global in &stage.channel_bindings {
        let declaration = &bound.container.channels[global as usize];
        let value_type = symbolic_channel_type(bound, global, bound.channel_types[global as usize]);
        encode_symbolic_type(&mut bytes, &value_type);
        put_u32(&mut bytes, declaration.capacity);
        bytes.push(declaration.host_role as u8);
        bytes.push(u8::from(declaration.seeded));
        let extern_direction = bound
            .container
            .externs
            .iter()
            .find(|external| external.chan == global)
            .map(|external| external.dir as u8 + 1)
            .unwrap_or(0);
        bytes.push(extern_direction);
    }

    let ports: Vec<_> = signature_ports(bound, stage.stage).collect();
    put_u32(&mut bytes, ports.len() as u32);
    for binding in ports {
        bytes.push(binding.port as u8);
        match &binding.source {
            PortSource::Channel(global) => {
                bytes.push(0);
                put_u32(
                    &mut bytes,
                    stage
                        .channel_bindings
                        .iter()
                        .position(|channel| channel == global)
                        .expect("port channel localized") as u32,
                );
                let port_type =
                    symbolic_port_type(binding.port, bound.channel_types[*global as usize]);
                encode_symbolic_type(&mut bytes, &port_type);
            }
            PortSource::Const { dtype, shape, data } => {
                bytes.push(1);
                bytes.push(*dtype as u8);
                encode_static_shape(&mut bytes, *shape);
                put_u32(&mut bytes, data.len() as u32);
                bytes.extend_from_slice(data);
            }
        }
    }

    put_u32(&mut bytes, stage.names.len() as u32);
    for name in &stage.names {
        put_u16(&mut bytes, name.len() as u16);
        bytes.extend_from_slice(name.as_bytes());
    }

    put_u32(&mut bytes, stage.ops.len() as u32);
    let mut next_value = 0usize;
    for op in &stage.ops {
        encode_planned_op(&mut bytes, op, stage.value_types.get(next_value));
        next_value += op.result_count() as usize;
    }
    put_u32(&mut bytes, stage.value_types.len() as u32);
    for (value_type, domain) in stage.value_types.iter().zip(&stage.value_domains) {
        encode_symbolic_type(&mut bytes, value_type);
        bytes.push(*domain as u8);
    }
    StageSignature {
        hash: crate::container_hash(&bytes),
        canonical_bytes: bytes,
    }
}

fn encode_static_shape(bytes: &mut Vec<u8>, shape: Shape) {
    bytes.push(shape.rank() as u8);
    for &dimension in shape.dims() {
        put_u32(bytes, dimension);
    }
}

fn encode_symbolic_type(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
    bytes.push(value_type.dtype as u8);
    bytes.push(value_type.dims.len() as u8);
    for dimension in &value_type.dims {
        match dimension {
            Dimension::Static(value) => {
                bytes.push(0);
                put_u32(bytes, *value);
            }
            Dimension::Symbolic(role) => {
                bytes.push(1);
                bytes.push(*role as u8);
            }
        }
    }
}

fn encode_symbolic_shape(bytes: &mut Vec<u8>, value_type: &SymbolicType) {
    bytes.push(value_type.dims.len() as u8);
    for dimension in &value_type.dims {
        put_u32(
            bytes,
            match dimension {
                Dimension::Static(value) => *value,
                Dimension::Symbolic(_) => 0,
            },
        );
    }
}

/// Plan op encoding reuses container op records, with zero dimensions denoting
/// symbolic extents whose roles and runtime values live in the adjacent plan
/// type table and lane record.
fn encode_planned_op(bytes: &mut Vec<u8>, op: &Op, result_type: Option<&SymbolicType>) {
    let result_type = || result_type.expect("shape-bearing op defines a value");
    match op {
        Op::Broadcast { value, .. } | Op::Reshape { value, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *value);
            encode_symbolic_shape(bytes, result_type());
        }
        Op::Rng { stream, kind, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *stream);
            encode_symbolic_shape(bytes, result_type());
            bytes.push(*kind as u8);
        }
        Op::RngKeyed { state, kind, .. } => {
            bytes.push(op.tag());
            put_u32(bytes, *state);
            encode_symbolic_shape(bytes, result_type());
            bytes.push(*kind as u8);
        }
        Op::IntrinsicVal { intr, dtype, .. } => {
            bytes.push(op.tag());
            put_u16(bytes, *intr as u16);
            bytes.push(*dtype as u8);
            encode_symbolic_shape(bytes, result_type());
        }
        Op::KernelCall {
            name, args, dtype, ..
        } => {
            bytes.push(op.tag());
            put_u16(bytes, *name);
            bytes.push(*dtype as u8);
            encode_symbolic_shape(bytes, result_type());
            bytes.push(args.len() as u8);
            for &argument in args {
                put_u32(bytes, argument);
            }
        }
        _ => encode_op(bytes, op),
    }
}

fn singleton_partition(stage: &NormalizedStage) -> RegionPartition {
    let regions = (0..stage.ops.len())
        .map(|node| build_region(stage, vec![node as u32], region_kind_for_node(stage, node)))
        .collect();
    RegionPartition {
        kind: PartitionKind::Singleton,
        regions,
        whole_stage_fallback: false,
    }
}

fn recognize_library_dataflows(stage: &NormalizedStage) -> Vec<LibraryMatch> {
    let (bases, producer) = result_layout(&stage.ops);
    let mut consumers = vec![Vec::new(); stage.value_types.len()];
    for (node, op) in stage.ops.iter().enumerate() {
        for operand in op.operands() {
            consumers[operand as usize].push(node as u32);
        }
    }

    let mut claimed = BTreeSet::new();
    let mut matches = Vec::new();
    for final_node in 0..stage.ops.len() {
        let Some(candidate) =
            match_nucleus_dataflow(stage, final_node, &bases, &producer, &consumers)
        else {
            continue;
        };
        if candidate.nodes.iter().any(|node| claimed.contains(node)) {
            continue;
        }
        claimed.extend(candidate.nodes.iter().copied());
        matches.push(candidate);
    }
    matches
}

fn match_nucleus_dataflow(
    stage: &NormalizedStage,
    final_node: usize,
    bases: &[u32],
    producer: &[usize],
    consumers: &[Vec<u32>],
) -> Option<LibraryMatch> {
    let Op::ReduceArgmax(perturbed) = stage.ops.get(final_node)? else {
        return None;
    };
    let add_node = *producer.get(*perturbed as usize)?;
    let Op::Add(left, right) = stage.ops.get(add_node)? else {
        return None;
    };

    match_nucleus_add_order(
        stage, final_node, add_node, *left, *right, bases, producer, consumers,
    )
    .or_else(|| {
        match_nucleus_add_order(
            stage, final_node, add_node, *right, *left, bases, producer, consumers,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn match_nucleus_add_order(
    stage: &NormalizedStage,
    final_node: usize,
    add_node: usize,
    masked: u32,
    noise: u32,
    bases: &[u32],
    producer: &[usize],
    consumers: &[Vec<u32>],
) -> Option<LibraryMatch> {
    let select_node = *producer.get(masked as usize)?;
    let Op::Select {
        cond: keep,
        a: logits,
        b: negative_infinity,
    } = stage.ops.get(select_node)?
    else {
        return None;
    };
    let (keep, logits, negative_infinity) = (*keep, *logits, *negative_infinity);

    let rng_node = *producer.get(noise as usize)?;
    let Op::RngKeyed {
        state: rng_state,
        shape: rng_shape,
        kind: RngKind::Gumbel,
    } = stage.ops.get(rng_node)?
    else {
        return None;
    };
    let rng_state = *rng_state;
    let rng_shape = *rng_shape;

    let negative_infinity_node = *producer.get(negative_infinity as usize)?;
    let Op::Const(Literal::F32(value)) = stage.ops.get(negative_infinity_node)? else {
        return None;
    };
    if value.to_bits() != f32::NEG_INFINITY.to_bits() {
        return None;
    }

    let pivot_node = *producer.get(keep as usize)?;
    let Op::PivotThreshold {
        input: probabilities,
        predicate: Predicate::CummassLe(top_p),
    } = stage.ops.get(pivot_node)?
    else {
        return None;
    };
    let (probabilities, top_p) = (*probabilities, *top_p);

    let div_node = *producer.get(probabilities as usize)?;
    let Op::Div(exponentials, sum_broadcast) = stage.ops.get(div_node)? else {
        return None;
    };
    let (exponentials, sum_broadcast) = (*exponentials, *sum_broadcast);

    let exponential_node = *producer.get(exponentials as usize)?;
    let Op::Exp(centered) = stage.ops.get(exponential_node)? else {
        return None;
    };
    let centered = *centered;

    let centered_node = *producer.get(centered as usize)?;
    let Op::Sub(centered_logits, maximum_broadcast) = stage.ops.get(centered_node)? else {
        return None;
    };
    if *centered_logits != logits {
        return None;
    }
    let maximum_broadcast = *maximum_broadcast;

    let maximum_broadcast_node = *producer.get(maximum_broadcast as usize)?;
    let Op::Broadcast {
        value: maximum,
        shape: maximum_shape,
    } = stage.ops.get(maximum_broadcast_node)?
    else {
        return None;
    };
    let (maximum, maximum_shape) = (*maximum, *maximum_shape);

    let maximum_node = *producer.get(maximum as usize)?;
    let Op::ReduceMax(maximum_logits) = stage.ops.get(maximum_node)? else {
        return None;
    };
    if *maximum_logits != logits {
        return None;
    }

    let sum_broadcast_node = *producer.get(sum_broadcast as usize)?;
    let Op::Broadcast {
        value: sum,
        shape: sum_shape,
    } = stage.ops.get(sum_broadcast_node)?
    else {
        return None;
    };
    let (sum, sum_shape) = (*sum, *sum_shape);

    let sum_node = *producer.get(sum as usize)?;
    let Op::ReduceSum(sum_exponentials) = stage.ops.get(sum_node)? else {
        return None;
    };
    if *sum_exponentials != exponentials || maximum_shape != sum_shape || maximum_shape != rng_shape
    {
        return None;
    }

    let token = *bases.get(final_node)?;
    let nodes = [
        maximum_node,
        maximum_broadcast_node,
        centered_node,
        exponential_node,
        sum_node,
        sum_broadcast_node,
        div_node,
        pivot_node,
        negative_infinity_node,
        select_node,
        rng_node,
        add_node,
        final_node,
    ];
    let mut ordered_nodes = nodes.map(|node| node as u32).to_vec();
    let mut library_inputs = vec![logits, top_p, rng_state];
    let mut scaled_input = None;
    if let Some(&scale_node) = producer.get(logits as usize)
        && let Some(Op::Div(raw_logits, divisor)) = stage.ops.get(scale_node)
    {
        let mut actual = consumers.get(logits as usize)?.clone();
        actual.sort_unstable();
        let mut expected = vec![
            maximum_node as u32,
            centered_node as u32,
            select_node as u32,
        ];
        expected.sort_unstable();
        if actual == expected {
            let mut library_logits = *raw_logits;
            if let Some(&reshape_node) = producer.get(*raw_logits as usize)
                && let Some(Op::Reshape { value, .. }) = stage.ops.get(reshape_node)
            {
                library_logits = *value;
            }
            library_inputs = vec![library_logits, *divisor, logits, top_p, rng_state];
            scaled_input = Some((library_logits, *divisor));
        }
    }
    ordered_nodes.sort_unstable();
    ordered_nodes.dedup();
    if ordered_nodes.len() != nodes.len() {
        return None;
    }
    let node_set: BTreeSet<u32> = ordered_nodes.iter().copied().collect();
    if library_inputs.iter().copied().any(|input| {
        producer
            .get(input as usize)
            .is_some_and(|node| node_set.contains(&(*node as u32)))
    }) {
        return None;
    }

    let exact_consumers = [
        (maximum, vec![maximum_broadcast_node as u32]),
        (maximum_broadcast, vec![centered_node as u32]),
        (centered, vec![exponential_node as u32]),
        (exponentials, vec![sum_node as u32, div_node as u32]),
        (sum, vec![sum_broadcast_node as u32]),
        (sum_broadcast, vec![div_node as u32]),
        (probabilities, vec![pivot_node as u32]),
        (keep, vec![select_node as u32]),
        (negative_infinity, vec![select_node as u32]),
        (masked, vec![add_node as u32]),
        (noise, vec![add_node as u32]),
        (
            *stage.ops[final_node].operands().first()?,
            vec![final_node as u32],
        ),
    ];
    for (value, mut expected) in exact_consumers {
        let mut actual = consumers.get(value as usize)?.clone();
        actual.sort_unstable();
        expected.sort_unstable();
        if actual != expected {
            return None;
        }
    }
    if consumers
        .get(token as usize)?
        .iter()
        .all(|consumer| node_set.contains(consumer))
    {
        return None;
    }

    let value_type = |value: u32| stage.value_types.get(value as usize);
    let logits_type = value_type(logits)?;
    if logits_type.dtype != DType::F32
        || !(1..=2).contains(&logits_type.rank())
        || !symbolic_shape_matches_static(logits_type, maximum_shape)
    {
        return None;
    }
    let row_dims = &logits_type.dims[..logits_type.dims.len() - 1];
    if let Some((raw_logits, divisor)) = scaled_input {
        let raw_type = value_type(raw_logits)?;
        if raw_type.dtype != DType::F32
            || !(1..=2).contains(&raw_type.rank())
            || raw_type.dims.last() != logits_type.dims.last()
        {
            return None;
        }
        let divisor_type = value_type(divisor)?;
        if divisor_type.dtype != DType::F32
            || (!divisor_type.is_scalar() && divisor_type.dims.as_slice() != row_dims)
        {
            return None;
        }
    }
    let top_p_type = value_type(top_p)?;
    if top_p_type.dtype != DType::F32
        || (!top_p_type.is_scalar()
            && !symbolic_dims_match_expected(
                &top_p_type.dims,
                row_dims,
                &maximum_shape.dims()[..maximum_shape.rank() - 1],
            ))
    {
        return None;
    }
    let rng_state_type = value_type(rng_state)?;
    if rng_state_type.dtype != DType::U32
        || rng_state_type.dims.as_slice() != [Dimension::Static(2)]
    {
        return None;
    }
    let token_type = value_type(token)?;
    if token_type.dtype != DType::I32 || token_type.dims.as_slice() != row_dims {
        return None;
    }

    for value in [
        maximum_broadcast,
        centered,
        exponentials,
        sum_broadcast,
        probabilities,
        masked,
        *stage.ops[final_node].operands().first()?,
    ] {
        if value_type(value)? != logits_type {
            return None;
        }
    }
    let noise_type = value_type(noise)?;
    if noise_type.dtype != DType::F32 || !symbolic_shape_matches_static(noise_type, rng_shape) {
        return None;
    }
    for value in [maximum, sum] {
        let ty = value_type(value)?;
        if ty.dtype != DType::F32 || ty.dims.as_slice() != row_dims {
            return None;
        }
    }
    let keep_type = value_type(keep)?;
    if keep_type.dtype != DType::Bool || keep_type.dims != logits_type.dims {
        return None;
    }
    let negative_infinity_type = value_type(negative_infinity)?;
    if negative_infinity_type.dtype != DType::F32 || !negative_infinity_type.dims.is_empty() {
        return None;
    }

    Some(LibraryMatch {
        library: LibraryOp::NucleusSample,
        nodes: ordered_nodes,
        inputs: library_inputs,
        outputs: vec![token],
    })
}

fn symbolic_shape_matches_static(value_type: &SymbolicType, shape: Shape) -> bool {
    symbolic_dims_match_static(&value_type.dims, shape.dims())
}

fn symbolic_dims_match_static(symbolic: &[Dimension], concrete: &[u32]) -> bool {
    symbolic.len() == concrete.len()
        && symbolic.iter().zip(concrete).all(|(symbolic, concrete)| {
            matches!(symbolic, Dimension::Symbolic(_)) || *symbolic == Dimension::Static(*concrete)
        })
}

fn symbolic_dims_match_expected(
    actual: &[Dimension],
    expected: &[Dimension],
    concrete: &[u32],
) -> bool {
    actual.len() == expected.len()
        && actual.len() == concrete.len()
        && actual
            .iter()
            .zip(expected)
            .zip(concrete)
            .all(|((actual, expected), concrete)| {
                actual == expected
                    || matches!(
                        (actual, expected),
                        (Dimension::Static(actual), Dimension::Symbolic(_))
                            if actual == concrete
                    )
            })
}

fn fused_partition(stage: &NormalizedStage, library_matches: &[LibraryMatch]) -> RegionPartition {
    let matched_nodes: BTreeSet<u32> = library_matches
        .iter()
        .flat_map(|candidate| candidate.nodes.iter().copied())
        .collect();
    let matches_by_end: BTreeMap<u32, &LibraryMatch> = library_matches
        .iter()
        .map(|candidate| {
            (
                *candidate.nodes.last().expect("library match has nodes"),
                candidate,
            )
        })
        .collect();
    let mut regions = Vec::new();
    let mut generated = Vec::new();
    for node in 0..stage.ops.len() as u32 {
        if matched_nodes.contains(&node) {
            flush_generated_region(stage, &mut regions, &mut generated);
            if let Some(candidate) = matches_by_end.get(&node) {
                regions.push(build_library_match_region(stage, candidate));
            }
            continue;
        }

        let kind = region_kind_for_node(stage, node as usize);
        if matches!(kind, RegionKind::Library(_)) {
            flush_generated_region(stage, &mut regions, &mut generated);
            regions.push(build_region(stage, vec![node], kind));
            continue;
        }

        if generated.first().is_some_and(|first| {
            !compatible_schedule(&stage.ops[*first as usize], &stage.ops[node as usize])
        }) {
            flush_generated_region(stage, &mut regions, &mut generated);
        }
        generated.push(node);
    }
    flush_generated_region(stage, &mut regions, &mut generated);
    RegionPartition {
        kind: PartitionKind::Fused,
        regions,
        whole_stage_fallback: false,
    }
}

fn flush_generated_region(
    stage: &NormalizedStage,
    regions: &mut Vec<Region>,
    nodes: &mut Vec<u32>,
) {
    if !nodes.is_empty() {
        regions.push(build_region(
            stage,
            core::mem::take(nodes),
            RegionKind::Generated,
        ));
    }
}

fn build_library_match_region(stage: &NormalizedStage, candidate: &LibraryMatch) -> Region {
    let mut region = build_region(
        stage,
        candidate.nodes.clone(),
        RegionKind::Library(candidate.library),
    );
    region.inputs = candidate.inputs.clone();
    region.outputs = candidate.outputs.clone();
    region
}

fn region_kind_for_node(stage: &NormalizedStage, node: usize) -> RegionKind {
    match stage.ops[node] {
        Op::TopK { .. } => RegionKind::Library(LibraryOp::TopK),
        Op::SortDesc(_) => RegionKind::Library(LibraryOp::Sort),
        Op::CumSum(_) | Op::CumProd(_) => RegionKind::Library(LibraryOp::Scan),
        Op::MatMul(_, _) => RegionKind::Library(LibraryOp::MatMul),
        Op::KernelCall { .. } | Op::SinkCall { .. } => RegionKind::Library(LibraryOp::SecondParty),
        _ => RegionKind::Generated,
    }
}

fn compatible_schedule(first: &Op, next: &Op) -> bool {
    !matches!(
        (first, next),
        (
            Op::CumSum(_) | Op::CumProd(_) | Op::SortDesc(_) | Op::TopK { .. } | Op::MatMul(_, _),
            _
        ) | (
            _,
            Op::CumSum(_) | Op::CumProd(_) | Op::SortDesc(_) | Op::TopK { .. } | Op::MatMul(_, _)
        )
    )
}

fn build_region(stage: &NormalizedStage, nodes: Vec<u32>, kind: RegionKind) -> Region {
    let node_set: BTreeSet<u32> = nodes.iter().copied().collect();
    let (bases, producer) = result_layout(&stage.ops);
    let mut consumers: Vec<Vec<u32>> = vec![Vec::new(); stage.value_types.len()];
    for (node, op) in stage.ops.iter().enumerate() {
        for operand in op.operands() {
            consumers[operand as usize].push(node as u32);
        }
    }

    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut sinks = Vec::new();
    for &node in &nodes {
        let op = &stage.ops[node as usize];
        for operand in op.operands() {
            if !node_set.contains(&(producer[operand as usize] as u32)) {
                inputs.insert(operand);
            }
        }
        if let Op::ChanPut { chan, value } = *op {
            sinks.push(ChannelSink {
                channel_slot: chan,
                value,
            });
        }
        let base = bases[node as usize];
        for result in 0..op.result_count() {
            let value = base + result;
            if consumers[value as usize]
                .iter()
                .any(|consumer| !node_set.contains(consumer))
            {
                outputs.insert(value);
            }
        }
    }

    let schedule = match kind {
        RegionKind::Library(_) => ScheduleTemplate::Library,
        RegionKind::Generated => {
            let has_compute = nodes.iter().any(|node| {
                !matches!(
                    stage.ops[*node as usize],
                    Op::ChanTake(_) | Op::ChanRead(_) | Op::ChanPut { .. } | Op::SinkCall { .. }
                )
            });
            let hierarchical = nodes.iter().any(|node| {
                let op = &stage.ops[*node as usize];
                if !matches!(
                    op,
                    Op::ReduceSum(_) | Op::ReduceMax(_) | Op::ReduceMin(_) | Op::ReduceArgmax(_)
                ) {
                    return false;
                }
                op.operands()
                    .first()
                    .and_then(|value| stage.value_types.get(*value as usize))
                    .and_then(|value_type| value_type.dims.last())
                    .is_some_and(|dimension| {
                        matches!(dimension, Dimension::Static(length) if *length > 32_768)
                    })
            });
            if !has_compute {
                ScheduleTemplate::Effects
            } else if hierarchical {
                ScheduleTemplate::HierarchicalRow
            } else {
                ScheduleTemplate::OneCtaPerRow
            }
        }
    };

    Region {
        kind,
        schedule,
        nodes,
        inputs: inputs.into_iter().collect(),
        outputs: outputs.into_iter().collect(),
        sinks,
    }
}

/// Serialize a complete stage plan. Every variable-sized record is
/// length-delimited so backend readers can reject unknown versions cleanly.
pub fn encode_stage_plan(stage: &CompiledStage) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&PLAN_MAGIC);
    put_u16(&mut bytes, REGION_PLAN_VERSION);
    put_u16(&mut bytes, COMPILER_VERSION);
    bytes.push(stage.normalized.stage as u8);
    bytes.extend_from_slice(&stage.signature.hash.to_le_bytes());
    put_u32(&mut bytes, stage.signature.canonical_bytes.len() as u32);
    bytes.extend_from_slice(&stage.signature.canonical_bytes);

    put_u32(&mut bytes, stage.normalized.channel_bindings.len() as u32);
    for &channel in &stage.normalized.channel_bindings {
        put_u32(&mut bytes, channel);
    }
    put_u32(&mut bytes, stage.normalized.names.len() as u32);
    for name in &stage.normalized.names {
        put_u16(&mut bytes, name.len() as u16);
        bytes.extend_from_slice(name.as_bytes());
    }

    put_u32(&mut bytes, stage.normalized.ops.len() as u32);
    for (op_index, op) in stage.normalized.ops.iter().enumerate() {
        let mut encoded = Vec::new();
        let result_base = stage.normalized.ops[..op_index]
            .iter()
            .map(Op::result_count)
            .sum::<u32>() as usize;
        encode_planned_op(
            &mut encoded,
            op,
            stage.normalized.value_types.get(result_base),
        );
        put_u32(&mut bytes, encoded.len() as u32);
        bytes.extend_from_slice(&encoded);
        put_u32(
            &mut bytes,
            stage.normalized.source_ops[op_index].len() as u32,
        );
        for &source in &stage.normalized.source_ops[op_index] {
            put_u32(&mut bytes, source);
        }
    }
    put_u32(&mut bytes, stage.normalized.value_types.len() as u32);
    for (value_type, domain) in stage
        .normalized
        .value_types
        .iter()
        .zip(&stage.normalized.value_domains)
    {
        encode_symbolic_type(&mut bytes, value_type);
        bytes.push(*domain as u8);
    }
    encode_partition(&mut bytes, &stage.singleton);
    encode_partition(&mut bytes, &stage.fused);
    bytes
}

fn encode_partition(bytes: &mut Vec<u8>, partition: &RegionPartition) {
    bytes.push(partition.kind as u8);
    bytes.push(u8::from(partition.whole_stage_fallback));
    put_u32(bytes, partition.regions.len() as u32);
    for region in &partition.regions {
        match region.kind {
            RegionKind::Generated => {
                bytes.push(0);
                bytes.push(0);
            }
            RegionKind::Library(library) => {
                bytes.push(1);
                bytes.push(library as u8);
            }
        }
        bytes.push(region.schedule as u8);
        encode_u32_slice(bytes, &region.nodes);
        encode_u32_slice(bytes, &region.inputs);
        encode_u32_slice(bytes, &region.outputs);
        put_u32(bytes, region.sinks.len() as u32);
        for sink in &region.sinks {
            put_u32(bytes, sink.channel_slot);
            put_u32(bytes, sink.value);
        }
    }
}

fn encode_u32_slice(bytes: &mut Vec<u8>, values: &[u32]) {
    put_u32(bytes, values.len() as u32);
    for &value in values {
        put_u32(bytes, value);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodedPlanHeader {
    pub stage: Stage,
    pub signature_hash: u64,
    pub singleton_regions: u32,
    pub fused_regions: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanDecodeError {
    Truncated,
    BadMagic,
    UnsupportedVersion,
    InvalidStage,
    InvalidRecord,
    CountTooLarge(&'static str),
}

#[derive(Clone, Copy)]
struct PlanReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PlanReader<'a> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], PlanDecodeError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(PlanDecodeError::Truncated)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(PlanDecodeError::Truncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, PlanDecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, PlanDecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32, PlanDecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, PlanDecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn bounded_count(
        &self,
        raw_count: u32,
        minimum_record_bytes: usize,
        structural_maximum: usize,
        table: &'static str,
    ) -> Result<usize, PlanDecodeError> {
        let count =
            usize::try_from(raw_count).map_err(|_| PlanDecodeError::CountTooLarge(table))?;
        let minimum_bytes = count
            .checked_mul(minimum_record_bytes)
            .ok_or(PlanDecodeError::CountTooLarge(table))?;
        if minimum_record_bytes == 0
            || count > structural_maximum
            || minimum_bytes > self.remaining()
        {
            return Err(PlanDecodeError::CountTooLarge(table));
        }
        Ok(count)
    }

    fn length_with_tail(
        &self,
        raw_length: u32,
        required_tail: usize,
        record: &'static str,
    ) -> Result<usize, PlanDecodeError> {
        let length =
            usize::try_from(raw_length).map_err(|_| PlanDecodeError::CountTooLarge(record))?;
        let required = length
            .checked_add(required_tail)
            .ok_or(PlanDecodeError::CountTooLarge(record))?;
        if required > self.remaining() {
            return Err(PlanDecodeError::CountTooLarge(record));
        }
        Ok(length)
    }
}

fn scan_plan_shape(reader: &mut PlanReader<'_>) -> Result<(), PlanDecodeError> {
    let rank = reader.u8()?;
    let rank = reader.bounded_count(
        rank as u32,
        4,
        MAX_RANK,
        "planned operation shape dimensions",
    )?;
    let bytes = rank.checked_mul(4).ok_or(PlanDecodeError::CountTooLarge(
        "planned operation shape dimensions",
    ))?;
    reader.take(bytes)?;
    Ok(())
}

fn scan_planned_op(bytes: &[u8]) -> Result<u32, PlanDecodeError> {
    let mut reader = PlanReader { bytes, offset: 0 };
    let tag = reader.u8()?;
    let results = match tag {
        0x01..=0x06 | 0x1E | 0x30..=0x33 | 0x3A | 0x40 | 0x41 | 0x50 | 0x64 | 0x90 | 0x91 => {
            reader.take(4)?;
            if tag == 0x50 { 2 } else { 1 }
        }
        0x07 => {
            reader.take(4)?;
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            1
        }
        0x10..=0x1D | 0x1F | 0x51 | 0x55 | 0x60 | 0x61 | 0x65 | 0x66 => {
            reader.take(8)?;
            if tag == 0x51 { 2 } else { 1 }
        }
        0x20 | 0x62 | 0x63 | 0x67 => {
            reader.take(12)?;
            1
        }
        0x68 => {
            reader.take(16)?;
            1
        }
        0x38 | 0x39 => {
            reader.take(4)?;
            scan_plan_shape(&mut reader)?;
            1
        }
        0x58 => {
            reader.take(4)?;
            if reader.u8()? > 2 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            reader.take(4)?;
            1
        }
        0x70 | 0x71 => {
            reader.take(4)?;
            scan_plan_shape(&mut reader)?;
            if reader.u8()? > 1 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            1
        }
        0x81 => {
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            reader.take(4)?;
            1
        }
        0x92 => {
            reader.take(8)?;
            0
        }
        0xA0 => {
            if reader.u16()? > IntrinsicId::MtpDrafts as u16 || reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            scan_plan_shape(&mut reader)?;
            1
        }
        0xA1 => {
            reader.u16()?;
            if reader.u8()? > DType::Bool as u8 {
                return Err(PlanDecodeError::InvalidRecord);
            }
            scan_plan_shape(&mut reader)?;
            let arguments = reader.u8()? as u32;
            let arguments =
                reader.bounded_count(arguments, 4, u8::MAX as usize, "kernel argument vector")?;
            reader.take(
                arguments
                    .checked_mul(4)
                    .ok_or(PlanDecodeError::CountTooLarge("kernel argument vector"))?,
            )?;
            1
        }
        0xA2 => {
            reader.u16()?;
            let arguments = reader.u8()? as u32;
            let arguments =
                reader.bounded_count(arguments, 4, u8::MAX as usize, "sink argument vector")?;
            reader.take(
                arguments
                    .checked_mul(4)
                    .ok_or(PlanDecodeError::CountTooLarge("sink argument vector"))?,
            )?;
            0
        }
        _ => return Err(PlanDecodeError::InvalidRecord),
    };
    if reader.offset != bytes.len() {
        return Err(PlanDecodeError::InvalidRecord);
    }
    Ok(results)
}

fn scan_index_vector(
    reader: &mut PlanReader<'_>,
    structural_maximum: usize,
    upper_bound: usize,
    ordered: bool,
    table: &'static str,
) -> Result<usize, PlanDecodeError> {
    let raw_count = reader.u32()?;
    let count = reader.bounded_count(raw_count, 4, structural_maximum, table)?;
    let byte_count = count
        .checked_mul(4)
        .ok_or(PlanDecodeError::CountTooLarge(table))?;
    let bytes = reader.take(byte_count)?;
    let mut previous = None;
    for value in bytes.chunks_exact(4) {
        let value = usize::try_from(u32::from_le_bytes(value.try_into().unwrap()))
            .map_err(|_| PlanDecodeError::CountTooLarge(table))?;
        if value >= upper_bound || (ordered && previous.is_some_and(|old| old >= value)) {
            return Err(PlanDecodeError::InvalidRecord);
        }
        previous = Some(value);
    }
    Ok(count)
}

fn scan_partition(
    reader: &mut PlanReader<'_>,
    expected_kind: PartitionKind,
    operation_count: usize,
    value_count: usize,
    channel_count: usize,
) -> Result<u32, PlanDecodeError> {
    let kind = match reader.u8()? {
        0 => PartitionKind::Singleton,
        1 => PartitionKind::Fused,
        _ => return Err(PlanDecodeError::InvalidRecord),
    };
    if kind != expected_kind || reader.u8()? > 1 {
        return Err(PlanDecodeError::InvalidRecord);
    }
    let raw_regions = reader.u32()?;
    let region_count = reader.bounded_count(raw_regions, 19, operation_count, "region table")?;
    for _ in 0..region_count {
        let region_kind = reader.u8()?;
        let library = reader.u8()?;
        let schedule = reader.u8()?;
        if region_kind > 1
            || (region_kind == 1 && library > LibraryOp::SecondParty as u8)
            || schedule > ScheduleTemplate::Library as u8
        {
            return Err(PlanDecodeError::InvalidRecord);
        }
        let nodes = scan_index_vector(
            reader,
            operation_count,
            operation_count,
            true,
            "region node vector",
        )?;
        let inputs = scan_index_vector(
            reader,
            value_count,
            value_count,
            false,
            "region input vector",
        )?;
        let outputs = scan_index_vector(
            reader,
            value_count,
            value_count,
            false,
            "region output vector",
        )?;
        let raw_sinks = reader.u32()?;
        let sinks = reader.bounded_count(raw_sinks, 8, nodes, "region sink vector")?;
        let sink_bytes = sinks
            .checked_mul(8)
            .ok_or(PlanDecodeError::CountTooLarge("region sink vector"))?;
        for sink in reader.take(sink_bytes)?.chunks_exact(8) {
            let channel = usize::try_from(u32::from_le_bytes(sink[..4].try_into().unwrap()))
                .map_err(|_| PlanDecodeError::CountTooLarge("region sink vector"))?;
            let value = usize::try_from(u32::from_le_bytes(sink[4..].try_into().unwrap()))
                .map_err(|_| PlanDecodeError::CountTooLarge("region sink vector"))?;
            if channel >= channel_count || value >= value_count {
                return Err(PlanDecodeError::InvalidRecord);
            }
        }
        if region_kind == 1
            && library == LibraryOp::NucleusSample as u8
            && (nodes != 13 || inputs != 3 || outputs != 1 || sinks != 0)
        {
            return Err(PlanDecodeError::InvalidRecord);
        }
    }
    Ok(raw_regions)
}

/// Allocation-free structural decoder used by registration tests and backend
/// preflight. Backend codegen applies the same limits before materializing a
/// plan.
pub fn decode_plan_header(bytes: &[u8]) -> Result<EncodedPlanHeader, PlanDecodeError> {
    let mut reader = PlanReader { bytes, offset: 0 };
    if reader.take(4)? != PLAN_MAGIC {
        return Err(PlanDecodeError::BadMagic);
    }
    if reader.u16()? != REGION_PLAN_VERSION || reader.u16()? != COMPILER_VERSION {
        return Err(PlanDecodeError::UnsupportedVersion);
    }
    let stage = Stage::from_u8(reader.u8()?).ok_or(PlanDecodeError::InvalidStage)?;
    let signature_hash = reader.u64()?;
    let signature_len = reader.u32()?;
    let signature_len = reader.length_with_tail(signature_len, 0, "stage signature")?;
    let signature = reader.take(signature_len)?;
    if crate::container_hash(signature) != signature_hash {
        return Err(PlanDecodeError::InvalidRecord);
    }

    let channels = reader.u32()?;
    let channel_count = reader.bounded_count(channels, 4, usize::MAX, "plan channel table")?;
    reader.take(
        channel_count
            .checked_mul(4)
            .ok_or(PlanDecodeError::CountTooLarge("plan channel table"))?,
    )?;

    let names = reader.u32()?;
    let name_count = reader.bounded_count(names, 2, u16::MAX as usize + 1, "plan name table")?;
    for _ in 0..name_count {
        let length = reader.u16()? as usize;
        reader.take(length)?;
    }

    let operations = reader.u32()?;
    let operation_count =
        reader.bounded_count(operations, 12, usize::MAX, "normalized operation table")?;
    let mut result_count = 0u32;
    for _ in 0..operation_count {
        let raw_op_len = reader.u32()?;
        let op_len = reader.length_with_tail(raw_op_len, 4, "normalized operation payload")?;
        result_count = result_count
            .checked_add(scan_planned_op(reader.take(op_len)?)?)
            .ok_or(PlanDecodeError::CountTooLarge("plan value table"))?;
        let sources = reader.u32()?;
        let source_count = reader.bounded_count(sources, 4, usize::MAX, "operation source map")?;
        reader.take(
            source_count
                .checked_mul(4)
                .ok_or(PlanDecodeError::CountTooLarge("operation source map"))?,
        )?;
    }

    let values = reader.u32()?;
    let structural_values = usize::try_from(result_count)
        .map_err(|_| PlanDecodeError::CountTooLarge("plan value table"))?;
    let value_count = reader.bounded_count(values, 3, structural_values, "plan value table")?;
    if value_count != structural_values {
        return Err(PlanDecodeError::InvalidRecord);
    }
    for _ in 0..value_count {
        if reader.u8()? > DType::Bool as u8 {
            return Err(PlanDecodeError::InvalidRecord);
        }
        let rank = reader.u8()?;
        let rank = reader.bounded_count(rank as u32, 2, MAX_RANK, "symbolic type dimensions")?;
        for _ in 0..rank {
            match reader.u8()? {
                0 => {
                    reader.u32()?;
                }
                1 => {
                    if reader.u8()? > SymbolicExtent::KeyLen as u8 {
                        return Err(PlanDecodeError::InvalidRecord);
                    }
                }
                _ => return Err(PlanDecodeError::InvalidRecord),
            }
        }
        if reader.u8()? > ValueDomain::EffectToken as u8 {
            return Err(PlanDecodeError::InvalidRecord);
        }
    }

    let singleton_regions = scan_partition(
        &mut reader,
        PartitionKind::Singleton,
        operation_count,
        value_count,
        channel_count,
    )?;
    let fused_regions = scan_partition(
        &mut reader,
        PartitionKind::Fused,
        operation_count,
        value_count,
        channel_count,
    )?;
    if reader.offset != bytes.len() {
        return Err(PlanDecodeError::InvalidRecord);
    }
    Ok(EncodedPlanHeader {
        stage,
        signature_hash,
        singleton_regions,
        fused_regions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, StageProgram, TraceContainer,
    };
    use crate::registry::{KernelInfo, ModelProfile};
    use crate::validate::bind;

    fn channel(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

    #[test]
    fn dce_preserves_faulting_kernel_calls_without_consumers() {
        let container = TraceContainer {
            names: vec!["observable".into()],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::KernelCall {
                    name: 0,
                    args: vec![],
                    shape: Shape::SCALAR,
                    dtype: DType::F32,
                }],
            }],
            ..TraceContainer::default()
        };
        let mut profile = ModelProfile::dummy();
        profile.kernels.push(KernelInfo {
            name: "observable".into(),
            sink_scope: None,
            replayable: true,
        });
        let bound = bind(container, profile).unwrap();
        let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
        assert!(matches!(
            compiled.normalized.ops.as_slice(),
            [Op::KernelCall { .. }]
        ));
    }

    #[test]
    fn symbolic_propagation_preserves_explicit_static_shape_changes() {
        let vocab = 32;
        let container = TraceContainer {
            channels: vec![
                channel(Shape::matrix(8, vocab), DType::F32, HostRole::Reader, false),
                channel(Shape::matrix(vocab, 1), DType::F32, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape: Shape::matrix(1, vocab),
                        dtype: DType::F32,
                    },
                    Op::ReduceMax(0),
                    Op::Broadcast {
                        value: 1,
                        shape: Shape::matrix(8, vocab),
                    },
                    Op::ChanPut { chan: 0, value: 2 },
                    Op::Reshape {
                        value: 0,
                        shape: Shape::matrix(vocab, 1),
                    },
                    Op::ChanPut { chan: 1, value: 3 },
                ],
            }],
            ..TraceContainer::default()
        };
        let mut profile = ModelProfile::dummy();
        profile.vocab = vocab;
        let bound = bind(container, profile).unwrap();
        let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
        assert_eq!(
            compiled.normalized.value_types[2].dims,
            vec![Dimension::Static(8), Dimension::Static(vocab)]
        );
        assert_eq!(
            compiled.normalized.value_types[3].dims,
            vec![Dimension::Static(vocab), Dimension::Static(1)]
        );
    }

    #[test]
    fn structured_masks_append_static_axis_without_dropping_symbolic_prefix() {
        let bound = program(0, 1);
        let original_types = [ValueType::vector(1, DType::U32)];
        let normalized_types = [SymbolicType {
            dtype: DType::U32,
            dims: vec![Dimension::Symbolic(SymbolicExtent::QueryLen)],
        }];
        let op = Op::CausalMask {
            positions: 0,
            len: 8,
        };
        let result = symbolic_result_type(
            &bound,
            &op,
            ValueType::new(Shape::matrix(1, 8), DType::Bool),
            &op,
            &original_types,
            &normalized_types,
        );
        assert_eq!(
            result.dims,
            vec![
                Dimension::Symbolic(SymbolicExtent::QueryLen),
                Dimension::Static(8),
            ]
        );
    }

    #[test]
    fn structured_masks_use_remapped_positions_after_dce() {
        let mask_ops = [
            Op::CausalMask {
                positions: 1,
                len: 8,
            },
            Op::SlidingWindowMask {
                positions: 1,
                len: 8,
                window: 3,
            },
            Op::SinkWindowMask {
                positions: 1,
                len: 8,
                sink: 2,
                window: 3,
            },
        ];
        for mask_op in mask_ops {
            let container = TraceContainer {
                channels: vec![
                    channel(Shape::vector(1), DType::U32, HostRole::None, true),
                    channel(Shape::matrix(1, 8), DType::Bool, HostRole::Reader, false),
                ],
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops: vec![
                        Op::Const(Literal::U32(99)),
                        Op::ChanTake(0),
                        mask_op,
                        Op::ChanPut { chan: 1, value: 2 },
                    ],
                }],
                ..TraceContainer::default()
            };
            let bound = bind(container, ModelProfile::dummy()).unwrap();
            let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
            assert_eq!(compiled.normalized.ops.len(), 3);
            assert_eq!(compiled.normalized.ops[1].operands(), vec![0]);
            assert_eq!(
                compiled.normalized.value_types[1].dims,
                vec![Dimension::Static(1), Dimension::Static(8)]
            );
        }
    }

    fn program(prefix_constant: u32, global_channel_offset: usize) -> BoundTrace {
        let vocab = 32;
        let mut channels = Vec::new();
        for _ in 0..global_channel_offset {
            channels.push(channel(Shape::SCALAR, DType::U32, HostRole::None, true));
        }

        let token = channels.len() as u32;
        channels.push(channel(Shape::vector(1), DType::I32, HostRole::None, true));
        let output = channels.len() as u32;
        channels.push(channel(
            Shape::vector(1),
            DType::I32,
            HostRole::Reader,
            false,
        ));
        let kv_len = channels.len() as u32;
        channels.push(channel(Shape::vector(1), DType::U32, HostRole::None, true));
        let stages = vec![
            StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::Const(Literal::U32(prefix_constant)),
                    Op::ChanPut {
                        chan: token.saturating_sub(1),
                        value: 0,
                    },
                ],
            },
            StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape: Shape::matrix(1, vocab),
                        dtype: DType::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::Reshape {
                        value: 1,
                        shape: Shape::vector(1),
                    },
                    Op::ChanPut {
                        chan: output,
                        value: 2,
                    },
                ],
            },
        ];
        let container = TraceContainer {
            names: vec![],
            channels,
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(token),
                },
                PortBinding {
                    port: Port::Positions,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::Pages,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::PageIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].into_iter().flat_map(u32::to_le_bytes).collect(),
                    },
                },
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Channel(kv_len),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
                PortBinding {
                    port: Port::WOff,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 0u32.to_le_bytes().to_vec(),
                    },
                },
            ],
            stages,
            externs: vec![],
        };
        let mut profile = ModelProfile::dummy();
        profile.vocab = vocab;
        bind(container, profile).unwrap()
    }

    fn top_k_program(global_channel_offset: usize) -> BoundTrace {
        let mut channels = (0..global_channel_offset)
            .map(|_| channel(Shape::SCALAR, DType::U32, HostRole::None, true))
            .collect::<Vec<_>>();
        let input = channels.len() as u32;
        channels.push(channel(
            Shape::matrix(2, 8),
            DType::F32,
            HostRole::None,
            true,
        ));
        let values = channels.len() as u32;
        channels.push(channel(
            Shape::matrix(2, 2),
            DType::F32,
            HostRole::Reader,
            false,
        ));
        let indices = channels.len() as u32;
        channels.push(channel(
            Shape::matrix(2, 2),
            DType::U32,
            HostRole::Reader,
            false,
        ));
        bind(
            TraceContainer {
                channels,
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops: vec![
                        Op::ChanTake(input),
                        Op::TopK { input: 0, k: 2 },
                        Op::ChanPut {
                            chan: values,
                            value: 1,
                        },
                        Op::ChanPut {
                            chan: indices,
                            value: 2,
                        },
                    ],
                }],
                ..TraceContainer::default()
            },
            ModelProfile::dummy(),
        )
        .unwrap()
    }

    fn softmax_program(rows: u32, vocab: u32) -> BoundTrace {
        let shape = Shape::matrix(rows, vocab);
        let container = TraceContainer {
            channels: vec![channel(Shape::SCALAR, DType::F32, HostRole::Reader, false)],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape,
                        dtype: DType::F32,
                    },
                    Op::ReduceMax(0),
                    Op::Broadcast { value: 1, shape },
                    Op::Sub(0, 2),
                    Op::Exp(3),
                    Op::ReduceSum(4),
                    Op::Broadcast { value: 5, shape },
                    Op::Div(4, 6),
                    Op::ReduceMax(7),
                    Op::ReduceMax(8),
                    Op::ChanPut { chan: 0, value: 9 },
                ],
            }],
            ..TraceContainer::default()
        };
        let mut profile = ModelProfile::dummy();
        profile.vocab = vocab;
        bind(container, profile).unwrap()
    }

    #[test]
    fn singleton_scans_normalize_to_aliases() {
        let bound = bind(
            TraceContainer {
                channels: vec![
                    channel(Shape::vector(1), DType::F32, HostRole::None, true),
                    channel(Shape::vector(1), DType::F32, HostRole::Reader, false),
                ],
                stages: vec![StageProgram {
                    stage: Stage::Prologue,
                    ops: vec![
                        Op::ChanTake(0),
                        Op::CumSum(0),
                        Op::ChanPut { chan: 1, value: 1 },
                    ],
                }],
                ..TraceContainer::default()
            },
            ModelProfile::dummy(),
        )
        .unwrap();
        let compiled = compile_stage(&bound, Stage::Prologue).unwrap();
        assert!(
            !compiled
                .normalized
                .ops
                .iter()
                .any(|op| matches!(op, Op::CumSum(_) | Op::CumProd(_)))
        );
    }

    #[derive(Clone, Copy)]
    enum NucleusMutation {
        Exact,
        CommutedAdd,
        WrongPredicate,
        WrongSelectSource,
        WrongCenteredSource,
        FiniteMaskFill,
        UniformRng,
        PartialArgmax,
        WrongSumOperand,
        EscapedIntermediate,
    }

    fn nucleus_program(mutation: NucleusMutation) -> BoundTrace {
        let shape = Shape::matrix(2, 8);
        let predicate = if matches!(mutation, NucleusMutation::WrongPredicate) {
            Predicate::ProbGe(1)
        } else {
            Predicate::CummassLe(1)
        };
        let select_source = if matches!(mutation, NucleusMutation::WrongSelectSource) {
            9
        } else {
            2
        };
        let centered_source = if matches!(mutation, NucleusMutation::WrongCenteredSource) {
            4
        } else {
            2
        };
        let mask_fill = if matches!(mutation, NucleusMutation::FiniteMaskFill) {
            Literal::F32(f32::MIN)
        } else {
            Literal::F32(f32::NEG_INFINITY)
        };
        let rng_kind = if matches!(mutation, NucleusMutation::UniformRng) {
            RngKind::Uniform
        } else {
            RngKind::Gumbel
        };
        let sum_input = if matches!(mutation, NucleusMutation::WrongSumOperand) {
            5
        } else {
            6
        };
        let add = if matches!(mutation, NucleusMutation::CommutedAdd) {
            Op::Add(13, 12)
        } else {
            Op::Add(12, 13)
        };
        let argmax_input = if matches!(mutation, NucleusMutation::PartialArgmax) {
            12
        } else {
            14
        };
        let mut channels = vec![
            channel(Shape::vector(2), DType::U32, HostRole::None, true),
            channel(Shape::vector(2), DType::F32, HostRole::None, true),
            channel(shape, DType::F32, HostRole::None, true),
            channel(Shape::vector(2), DType::I32, HostRole::Reader, false),
        ];
        let mut ops = vec![
            Op::ChanRead(0), // rng state: v0
            Op::ChanRead(1), // top-p: v1
            Op::ChanTake(2), // logits: v2
            Op::ReduceMax(2),
            Op::Broadcast { value: 3, shape },
            Op::Sub(centered_source, 4),
            Op::Exp(5),
            Op::ReduceSum(sum_input),
            Op::Broadcast { value: 7, shape },
            Op::Div(6, 8),
            Op::PivotThreshold {
                input: 9,
                predicate,
            },
            Op::Const(mask_fill),
            Op::Select {
                cond: 10,
                a: select_source,
                b: 11,
            },
            Op::RngKeyed {
                state: 0,
                shape,
                kind: rng_kind,
            },
            add,
            Op::ReduceArgmax(argmax_input),
            Op::ChanPut { chan: 3, value: 15 },
        ];
        if matches!(mutation, NucleusMutation::EscapedIntermediate) {
            channels.push(channel(shape, DType::F32, HostRole::Reader, false));
            ops.push(Op::ChanPut { chan: 4, value: 6 });
        }
        let container = TraceContainer {
            channels,
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
            ..TraceContainer::default()
        };
        bind(container, ModelProfile::dummy()).unwrap()
    }

    fn scaled_nucleus_program() -> BoundTrace {
        let shape = Shape::matrix(2, 8);
        let container = TraceContainer {
            channels: vec![
                channel(Shape::vector(2), DType::U32, HostRole::None, true),
                channel(Shape::vector(2), DType::F32, HostRole::None, true),
                channel(shape, DType::F32, HostRole::None, true),
                channel(Shape::vector(2), DType::I32, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanRead(0),
                    Op::ChanRead(1),
                    Op::ChanTake(2),
                    Op::Reshape { value: 2, shape },
                    Op::Const(Literal::F32(0.8)),
                    Op::Div(3, 4),
                    Op::ReduceMax(5),
                    Op::Broadcast { value: 6, shape },
                    Op::Sub(5, 7),
                    Op::Exp(8),
                    Op::ReduceSum(9),
                    Op::Broadcast { value: 10, shape },
                    Op::Div(9, 11),
                    Op::PivotThreshold {
                        input: 12,
                        predicate: Predicate::CummassLe(1),
                    },
                    Op::Const(Literal::F32(f32::NEG_INFINITY)),
                    Op::Select {
                        cond: 13,
                        a: 5,
                        b: 14,
                    },
                    Op::RngKeyed {
                        state: 0,
                        shape,
                        kind: RngKind::Gumbel,
                    },
                    Op::Add(15, 16),
                    Op::ReduceArgmax(17),
                    Op::ChanPut { chan: 3, value: 18 },
                ],
            }],
            ..TraceContainer::default()
        };
        bind(container, ModelProfile::dummy()).unwrap()
    }

    fn interleaved_nucleus_program() -> BoundTrace {
        let shape = Shape::matrix(2, 8);
        bind(
            TraceContainer {
                channels: vec![
                    channel(Shape::vector(2), DType::U32, HostRole::None, true),
                    channel(Shape::vector(2), DType::F32, HostRole::None, true),
                    channel(shape, DType::F32, HostRole::None, true),
                    channel(Shape::SCALAR, DType::U32, HostRole::None, true),
                    channel(Shape::vector(2), DType::I32, HostRole::Reader, false),
                ],
                stages: vec![StageProgram {
                    stage: Stage::Epilogue,
                    ops: vec![
                        Op::ChanRead(0),  // rng state: v0
                        Op::ChanRead(1),  // top-p: v1
                        Op::ChanTake(2),  // logits: v2
                        Op::ReduceMax(2), // matched n3, v3
                        Op::ChanRead(3),  // unrelated n4, v4
                        Op::Broadcast { value: 3, shape },
                        Op::Sub(2, 5),
                        Op::Exp(6),
                        Op::ReduceSum(7),
                        Op::Broadcast { value: 8, shape },
                        Op::Div(7, 9),
                        Op::PivotThreshold {
                            input: 10,
                            predicate: Predicate::CummassLe(1),
                        },
                        Op::Const(Literal::F32(f32::NEG_INFINITY)),
                        Op::Select {
                            cond: 11,
                            a: 2,
                            b: 12,
                        },
                        Op::RngKeyed {
                            state: 0,
                            shape,
                            kind: RngKind::Gumbel,
                        },
                        Op::Add(13, 14),
                        Op::ReduceArgmax(15),
                        Op::ChanPut { chan: 4, value: 16 },
                    ],
                }],
                ..TraceContainer::default()
            },
            ModelProfile::dummy(),
        )
        .unwrap()
    }

    #[test]
    fn identical_epilogues_share_signature_across_programs() {
        let first = program(1, 1);
        let second = program(2, 2);
        let first = compile_stage(&first, Stage::Epilogue).unwrap();
        let second = compile_stage(&second, Stage::Epilogue).unwrap();
        assert_eq!(first.signature, second.signature);
        assert_ne!(
            first.normalized.channel_bindings,
            second.normalized.channel_bindings
        );
    }

    #[test]
    fn top_k_has_one_canonical_signature_and_library_kind() {
        let generic = top_k_program(0);
        let beam_style = top_k_program(3);
        let generic = compile_stage(&generic, Stage::Epilogue).unwrap();
        let beam_style = compile_stage(&beam_style, Stage::Epilogue).unwrap();
        assert_eq!(generic.signature, beam_style.signature);
        assert_eq!(
            generic
                .fused
                .regions
                .iter()
                .filter(|region| region.kind == RegionKind::Library(LibraryOp::TopK))
                .count(),
            1
        );
    }

    #[test]
    fn normalized_nucleus_dataflow_has_role_ordered_library_abi() {
        let compiled =
            compile_stage(&nucleus_program(NucleusMutation::Exact), Stage::Epilogue).unwrap();
        let nucleus = compiled
            .fused
            .regions
            .iter()
            .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
            .expect("nucleus library region");
        assert_eq!(nucleus.nodes, (3..=15).collect::<Vec<_>>());
        assert_eq!(nucleus.inputs, vec![2, 1, 0]);
        assert_eq!(nucleus.outputs, vec![15]);
        assert!(nucleus.nodes.windows(2).all(|nodes| nodes[0] < nodes[1]));
        assert!(
            !compiled
                .singleton
                .regions
                .iter()
                .any(|region| { region.kind == RegionKind::Library(LibraryOp::NucleusSample) })
        );
    }

    #[test]
    fn scaled_nucleus_absorbs_temperature_and_peels_reshape() {
        let compiled = compile_stage(&scaled_nucleus_program(), Stage::Epilogue).unwrap();
        let nucleus = compiled
            .fused
            .regions
            .iter()
            .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
            .expect("scaled nucleus library region");
        assert_eq!(nucleus.nodes, (6..=18).collect::<Vec<_>>());
        assert_eq!(nucleus.inputs, vec![2, 4, 5, 1, 0]);
        assert_eq!(nucleus.outputs, vec![18]);
    }

    #[test]
    fn byte_identical_nucleus_dags_share_signature_and_library_plan() {
        let first = nucleus_program(NucleusMutation::Exact);
        let second = bind(first.container.clone(), first.profile.clone()).unwrap();
        assert_eq!(
            crate::container::encode(&first.container),
            crate::container::encode(&second.container)
        );
        let first = compile_stage(&first, Stage::Epilogue).unwrap();
        let second = compile_stage(&second, Stage::Epilogue).unwrap();
        assert_eq!(first.signature, second.signature);
        assert_eq!(first.fused, second.fused);
        assert_eq!(encode_stage_plan(&first), encode_stage_plan(&second));
    }

    #[test]
    fn nucleus_matching_uses_connectivity_not_contiguous_source_ranges() {
        let compiled = compile_stage(&interleaved_nucleus_program(), Stage::Epilogue).unwrap();
        let region = compiled
            .fused
            .regions
            .iter()
            .find(|region| region.kind == RegionKind::Library(LibraryOp::NucleusSample))
            .expect("interleaved nucleus region");
        assert_eq!(
            region.nodes,
            core::iter::once(3).chain(5..=16).collect::<Vec<_>>()
        );
        assert_eq!(region.inputs, vec![2, 1, 0]);
        assert_eq!(region.outputs, vec![16]);
    }

    #[test]
    fn nucleus_lookalikes_remain_generic() {
        let exact =
            compile_stage(&nucleus_program(NucleusMutation::Exact), Stage::Epilogue).unwrap();
        let commuted = compile_stage(
            &nucleus_program(NucleusMutation::CommutedAdd),
            Stage::Epilogue,
        )
        .unwrap();
        assert!(
            commuted
                .fused
                .regions
                .iter()
                .any(|region| { region.kind == RegionKind::Library(LibraryOp::NucleusSample) })
        );

        for mutation in [
            NucleusMutation::WrongPredicate,
            NucleusMutation::WrongSelectSource,
            NucleusMutation::WrongCenteredSource,
            NucleusMutation::FiniteMaskFill,
            NucleusMutation::UniformRng,
            NucleusMutation::PartialArgmax,
            NucleusMutation::WrongSumOperand,
            NucleusMutation::EscapedIntermediate,
        ] {
            let near = compile_stage(&nucleus_program(mutation), Stage::Epilogue).unwrap();
            assert_ne!(near.signature, exact.signature);
            assert!(
                !near
                    .fused
                    .regions
                    .iter()
                    .any(|region| { region.kind == RegionKind::Library(LibraryOp::NucleusSample) })
            );
        }
    }

    #[test]
    fn epilogue_signature_ignores_unrelated_descriptor_schema() {
        let first = program(1, 1);
        let mut container = first.container.clone();
        container.ports.push(PortBinding {
            port: Port::AttnMask,
            source: PortSource::Const {
                dtype: DType::Bool,
                shape: Shape::vector(1),
                data: vec![1],
            },
        });
        let second = bind(container, first.profile.clone()).unwrap();
        assert_eq!(
            compile_stage(&first, Stage::Epilogue).unwrap().signature,
            compile_stage(&second, Stage::Epilogue).unwrap().signature
        );
    }

    #[test]
    fn symbolic_row_shapes_share_signature_but_vocab_does_not() {
        let one = softmax_program(1, 32);
        let eight = softmax_program(8, 32);
        let other_vocab = softmax_program(1, 64);
        let signature =
            |bound: &BoundTrace| compile_stage(bound, Stage::Epilogue).unwrap().signature;
        assert_eq!(signature(&one), signature(&eight));
        assert_ne!(signature(&one), signature(&other_vocab));
    }

    #[test]
    fn gather_prefix_dimensions_come_from_indices() {
        let vocab = 32;
        let container = TraceContainer {
            channels: vec![
                channel(Shape::matrix(2, vocab), DType::F32, HostRole::None, true),
                channel(Shape::vector(3), DType::U32, HostRole::None, true),
                channel(Shape::matrix(3, vocab), DType::F32, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::ChanTake(0),
                    Op::ChanTake(1),
                    Op::Gather { src: 0, idx: 1 },
                    Op::ChanPut { chan: 2, value: 2 },
                ],
            }],
            ..TraceContainer::default()
        };
        let bound = bind(container, ModelProfile::dummy()).unwrap();
        let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
        assert_eq!(
            compiled.normalized.value_types[2].dims,
            vec![Dimension::Static(3), Dimension::Static(vocab)]
        );
    }

    #[test]
    fn channel_storage_stays_static_while_lane_carries_logical_extents() {
        let container = TraceContainer {
            channels: vec![channel(
                Shape::matrix(4, 8),
                DType::Bool,
                HostRole::None,
                true,
            )],
            ports: vec![PortBinding {
                port: Port::AttnMask,
                source: PortSource::Channel(0),
            }],
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![Op::ChanRead(0)],
            }],
            ..TraceContainer::default()
        };
        let bound = bind(container, ModelProfile::dummy()).unwrap();
        let compiled = compile_stage(&bound, Stage::Prologue).unwrap();
        assert_eq!(
            compiled.normalized.value_types[0].dims,
            vec![Dimension::Static(4), Dimension::Static(8),]
        );
        let lane = LaneRecord {
            query_len: 4,
            key_len: 8,
            ..LaneRecord::default()
        };
        assert_eq!((lane.query_len, lane.key_len), (4, 8));
    }

    #[test]
    fn mtp_row_count_remains_distinct_and_static() {
        let vocab = 32;
        let container = TraceContainer {
            channels: vec![channel(
                Shape::vector(3),
                DType::I32,
                HostRole::Reader,
                false,
            )],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::MtpLogits,
                        shape: Shape::matrix(3, vocab),
                        dtype: DType::F32,
                    },
                    Op::ReduceArgmax(0),
                    Op::ChanPut { chan: 0, value: 1 },
                ],
            }],
            ..TraceContainer::default()
        };
        let mut profile = ModelProfile::dummy();
        profile.vocab = vocab;
        profile.has_mtp_logits = true;
        let bound = bind(container, profile).unwrap();
        let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
        assert_eq!(
            compiled.normalized.value_types[0].dims,
            vec![Dimension::Static(3), Dimension::Static(vocab)]
        );
    }

    #[test]
    fn explicit_candidate_batch_does_not_inherit_sampled_rows() {
        let vocab = 32;
        let container = TraceContainer {
            channels: vec![
                channel(Shape::vector(2), DType::U32, HostRole::None, true),
                channel(Shape::matrix(4, vocab), DType::F32, HostRole::Reader, false),
                channel(Shape::matrix(1, vocab), DType::F32, HostRole::Reader, false),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Logits,
                        shape: Shape::matrix(1, vocab),
                        dtype: DType::F32,
                    },
                    Op::ChanTake(0),
                    Op::RngKeyed {
                        state: 1,
                        shape: Shape::matrix(4, vocab),
                        kind: crate::RngKind::Gumbel,
                    },
                    Op::ChanPut { chan: 1, value: 2 },
                    Op::ChanPut { chan: 2, value: 0 },
                ],
            }],
            ..TraceContainer::default()
        };
        let mut profile = ModelProfile::dummy();
        profile.vocab = vocab;
        let bound = bind(container, profile).unwrap();
        let compiled = compile_stage(&bound, Stage::Epilogue).unwrap();
        assert_eq!(
            compiled.normalized.value_types[0].dims,
            vec![
                Dimension::Symbolic(SymbolicExtent::SampledRows),
                Dimension::Static(vocab),
            ]
        );
        assert_eq!(
            compiled.normalized.value_types[2].dims,
            vec![Dimension::Static(4), Dimension::Static(vocab)]
        );
    }

    #[test]
    fn runtime_extents_do_not_change_signature() {
        let bound = program(1, 1);
        let stage = compile_stage(&bound, Stage::Epilogue).unwrap();
        let signature = stage.signature.clone();
        let first = ScheduleBucket::for_dispatch(
            2,
            RuntimeExtents {
                kv_len: 8,
                page_count: 1,
                row_count: 1,
                token_count: 1,
                sampled_rows: 1,
                query_len: 1,
                key_len: 8,
            },
        );
        let second = ScheduleBucket::for_dispatch(
            8,
            RuntimeExtents {
                kv_len: 4096,
                page_count: 64,
                row_count: 8,
                token_count: 32,
                sampled_rows: 8,
                query_len: 32,
                key_len: 4096,
            },
        );
        assert_ne!(first, second);
        assert_eq!(stage.signature, signature);
    }

    #[test]
    fn plan_encoding_is_deterministic_and_self_describing() {
        let bound = program(1, 1);
        let stage = compile_stage(&bound, Stage::Epilogue).unwrap();
        let first = encode_stage_plan(&stage);
        let second = encode_stage_plan(&stage);
        assert_eq!(first, second);
        let header = decode_plan_header(&first).unwrap();
        assert_eq!(header.stage, Stage::Epilogue);
        assert_eq!(header.signature_hash, stage.signature.hash);
        assert_eq!(
            header.singleton_regions as usize,
            stage.singleton.regions.len()
        );
        assert_eq!(header.fused_regions as usize, stage.fused.regions.len());
        let debug = debug_stage_plan(&stage);
        assert!(debug.contains("epilogue signature="));
        assert!(debug.contains("Fused"));
        assert_eq!(
            stage.metrics().normalized_ops,
            stage.normalized.ops.len() as u32
        );
        let mut corrupted = first;
        corrupted[21] ^= 1;
        assert_eq!(
            decode_plan_header(&corrupted),
            Err(PlanDecodeError::InvalidRecord)
        );
    }

    #[test]
    fn lane_layout_is_stable() {
        assert_eq!(core::mem::size_of::<LaneTableHeader>(), 16);
        assert_eq!(core::mem::size_of::<LaneRecord>(), 96);
        assert_eq!(core::mem::size_of::<LaneChannelSlot>(), 32);
        let key = ExecutableCacheKey {
            backend: BackendKind::Cuda,
            device_arch: 89,
            compiler_version: COMPILER_VERSION,
            stage_signature: 7,
            schedule_bucket: ScheduleBucket {
                row_bucket: 2,
                lane_bucket: 3,
            },
            semantic_mode: SemanticMode::Exact,
        };
        assert_ne!(
            key.encode(),
            ExecutableCacheKey {
                backend: BackendKind::Metal,
                ..key
            }
            .encode()
        );
    }
}
