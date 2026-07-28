//! Backend-neutral PTIR compiler planning.
//!
//! Rust owns normalization, stage signatures, value-domain analysis, region
//! partitioning, and the lane-table ABI. Drivers consume the serialized plan
//! and provide backend code generation and library implementations.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use pie_ir::Fnv1a;
use pie_ir::registry::Stage;
use pie_ir::validate::BoundTrace;

mod decode;
mod encode;
mod fold;
mod lane;
mod normalize;
mod region;
mod signature;
mod symbolic;

pub use decode::*;
pub use encode::*;
pub use lane::*;
pub use normalize::*;
pub use region::*;
pub use signature::*;
pub use symbolic::*;

pub const COMPILER_VERSION: u16 = 3;
pub const REGION_PLAN_VERSION: u16 = 4;
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
            elements.saturating_mul(pie_ir::container::const_elem_size(value_type.dtype) as u64)
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
    let index = StageIndex::of(&normalized);
    let singleton = singleton_partition(&normalized, &index);
    let matches = recognize_library_dataflows(&normalized, &index);
    let fused = fused_partition(&normalized, &index, &matches);
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

/// The graph-cache identity of one compiled stage.
///
/// The CUDA driver computes exactly this in `program_identity.hpp` after
/// decoding the plan, and keys its CUDA-graph cache on it. It is a decision
/// about the program, so under `ptir-refactor.md`'s north star it belongs to
/// the host: the driver should be told, not re-derive.
///
/// The walk is byte-order-locked to the C++ one — the same fields, in the same
/// order, through the same FNV-1a 64. `stage_identity_matches_the_driver` in
/// `compiler/tests` pins the two together, and while both exist the driver
/// compares its own value against this one and counts any divergence
/// (`ProgramRuntimeStats::host_stage_identities` /
/// `divergent_stage_identities`), which is what makes deleting the C++ copy an
/// evidenced step rather than a leap.
pub fn stage_identity(stage: &CompiledStage) -> u64 {
    let mut hash = Fnv1a::new();
    hash.byte(stage.normalized.stage as u8);
    hash.u32_le(stage.signature.canonical_bytes.len() as u32);
    hash.bytes(&stage.signature.canonical_bytes);
    hash.byte(stage.fused.kind as u8);
    hash.byte(u8::from(stage.fused.whole_stage_fallback));
    hash.u32_le(stage.fused.regions.len() as u32);
    for region in &stage.fused.regions {
        hash.byte(region.schedule as u8);
        let (library, library_op) = match region.kind {
            RegionKind::Generated => (0u8, 0u8),
            RegionKind::Library(op) => (1u8, op as u8),
        };
        hash.byte(library);
        hash.byte(library_op);
        hash.u32_le(region.nodes.len() as u32);
        for &node in &region.nodes {
            hash.u32_le(node.get());
        }
        hash.u32_le(region.inputs.len() as u32);
        for &input in &region.inputs {
            hash.u32_le(input);
        }
        hash.u32_le(region.outputs.len() as u32);
        for &output in &region.outputs {
            hash.u32_le(output);
        }
        hash.u32_le(region.sinks.len() as u32);
        for sink in &region.sinks {
            hash.u32_le(sink.channel_slot);
            hash.u32_le(sink.value);
        }
    }
    let hash = hash.finish();
    if hash == 0 { 1 } else { hash }
}

#[cfg(test)]
mod tests;
