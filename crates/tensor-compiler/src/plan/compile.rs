//! Backend-neutral PTIR compiler planning.
//!
//! Rust owns normalization, stage signatures, value-domain analysis, region
//! partitioning, and the lane-table ABI. Engines consume the serialized plan
//! and provide backend code generation and library implementations.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use tensor_ir::Fnv1a;
use tensor_ir::registry::Stage;
use tensor_ir::validate::BoundTrace;

mod canonical;
mod fold;
mod normalize;
mod nucleus;
mod region;
mod signature;
mod symbolic;

pub use normalize::*;
use nucleus::recognize_library_dataflows;
pub use region::*;
pub use signature::*;
pub use symbolic::*;

/// Cache-identity tokens, not wire-format versions.
///
/// Both engines fold these into the key of their compiled-module caches
/// (`crates/driver-cuda/csrc/.../module_cache.hpp`, `crates/engine-metal/csrc/.../m1_runtime.cpp`), so
/// bumping one is how a change in this crate's planning semantics invalidates
/// everything a device already built. Nothing parses a byte stream stamped with
/// them.
pub const COMPILER_VERSION: u16 = 3;
/// Bumped when region partitioning changes shape. See [`COMPILER_VERSION`].
pub const REGION_PLAN_VERSION: u16 = 4;

/// The complete plan for one stage, handed to `tensor-compiler` as a value.
///
/// Produced by [`compile_stage_at`]. Both partitions are carried so a backend
/// can emit the [`fused`](Self::fused) form yet fall back to the
/// always-correct [`singleton`](Self::singleton) one without re-planning.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledStage {
    /// The normalized op DAG every other field was derived from.
    pub normalized: NormalizedStage,
    /// The canonical signature; two stages sharing it may share an executable.
    pub signature: StageSignature,
    /// One region per op — the always-correct fallback partition.
    pub singleton: RegionPartition,
    /// Ops grouped by schedule with recognized library dataflows lifted out.
    pub fused: RegionPartition,
}

/// Static, backend-independent counts summarizing a [`CompiledStage`].
///
/// Diagnostic only — no planning decision reads these. The byte totals count
/// only fully-static values; anything with a symbolic extent contributes zero,
/// because its size is unknown until launch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanMetrics {
    /// PTIR ops in the stage before normalization.
    pub source_ops: u32,
    /// Ops left after normalization dropped, folded and merged them.
    pub normalized_ops: u32,
    /// Regions in the singleton partition — one per normalized op.
    pub singleton_regions: u32,
    /// Regions in the fused partition.
    pub fused_regions: u32,
    /// Fused regions dispatched to a library rather than generated inline.
    pub library_regions: u32,
    /// Bytes of every statically-shaped value that is not a direct channel
    /// sink — the stage's scratch footprint.
    pub static_scratch_bytes: u64,
    /// Bytes of statically-shaped values a fused region writes straight to a
    /// channel.
    pub direct_channel_sink_bytes: u64,
}

impl CompiledStage {
    /// Summarizes this stage as [`PlanMetrics`] for diagnostics.
    pub fn metrics(&self) -> PlanMetrics {
        let static_bytes = |value_type: &SymbolicType| {
            let mut elements = 1u64;
            for dimension in &value_type.dims {
                let Dimension::Static(dimension) = dimension else {
                    return 0;
                };
                elements = elements.saturating_mul(*dimension as u64);
            }
            elements.saturating_mul(tensor_ir::container::const_elem_size(value_type.dtype) as u64)
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

/// Compiles the stage of the given [`Stage`] kind, or `None` when the
/// container has no stage of that kind.
///
/// A thin wrapper over [`compile_stage_at`] that first locates the stage.
pub fn compile_stage(bound: &BoundTrace, stage: Stage) -> Option<CompiledStage> {
    let stage_index = bound
        .container
        .stages
        .iter()
        .position(|program| program.stage == stage)?;
    Some(compile_stage_at(bound, stage_index))
}

/// Compiles the stage at `stage_index` in container order.
///
/// The core of the crate: it normalizes the stage body, localizes its channels
/// and names, signs it, and builds both the singleton and fused partitions.
///
/// # Panics
///
/// Panics if `stage_index` is out of range for the container's stage list.
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
/// An engine keys its graph cache on this value, so it is a decision about the
/// program and belongs to the host. Every engine is handed the bytes rather
/// than deriving them — the Metal runtime reads them out of
/// `interface/driver/src/plan.rs`'s `identity` — because an engine-side copy of
/// this derivation is a second definition of a cache key, and two definitions
/// of a cache key that drift apart share graphs between programs that are not
/// the same program.
///
/// Because the key is published, moving it is an operational event and not a
/// refactor: a stale key is silent, reusing a graph built by a different
/// planner. `stage_identity_is_pinned` in `compiler/tests` catches the move,
/// and [`COMPILER_VERSION`] must be bumped alongside it.
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
            hash.u32_le(sink.channel_slot.get());
            hash.u32_le(sink.value);
        }
    }
    let hash = hash.finish();
    if hash == 0 { 1 } else { hash }
}

#[cfg(test)]
mod tests;
