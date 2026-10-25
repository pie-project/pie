use crate::codegen::error::{EmitError, RegionForm, ValueLayoutSite};
use crate::codegen::wellformed::{op_valid, region_ranges_valid, value_types_valid};
use alloc::string::String;
use alloc::vec::Vec;

use crate::plan::{
    CompiledStage, Dimension, LibraryOp, PartitionKind, Region, RegionKind, RegionPartition,
    ScheduleTemplate,
};
use eta_ir::op::{intrinsic_tags, tags};
use eta_ir::types::Dtype;

use super::M1OpMeta;
use crate::codegen::op_view::OpView;

pub(crate) fn library_op_byte(region: &Region) -> u8 {
    match region.kind {
        RegionKind::Library(op) => op as u8,
        RegionKind::Generated => 0,
    }
}

pub(crate) fn is_library(region: &Region) -> bool {
    matches!(region.kind, RegionKind::Library(_))
}

pub fn metal_intrinsic_supported(intr: u16) -> bool {
    matches!(
        intr,
        intrinsic_tags::LOGITS
            | intrinsic_tags::VELOCITY
            | intrinsic_tags::HIDDEN
            | intrinsic_tags::MTP_LOGITS
            | intrinsic_tags::MTP_DRAFTS
            | intrinsic_tags::ATTN_SCORE
            | intrinsic_tags::PIXELS
    )
}

pub fn intrinsics_bindable(ops: &[OpView], region: &Region) -> Result<(), EmitError> {
    unbindable_intrinsic(ops, region, metal_intrinsic_supported)
}

pub fn grouped_intrinsics_bindable(ops: &[OpView], region: &Region) -> Result<(), EmitError> {
    unbindable_intrinsic(ops, region, super::intrinsics::m3_intrinsic_bindable)
}

fn unbindable_intrinsic(
    ops: &[OpView],
    region: &Region,
    bindable: fn(u16) -> bool,
) -> Result<(), EmitError> {
    for &node in &region.nodes {
        let Some(op) = ops.get(node.index()) else {
            continue;
        };
        if op.tag == tags::INTRINSIC_VAL && !bindable(op.intr) {
            return Err(EmitError::UnbindableIntrinsic { intrinsic: op.intr });
        }
    }
    Ok(())
}

pub fn nucleus_library_region_valid(stage: &CompiledStage, region: &Region) -> bool {
    let value_types = &stage.normalized.value_types;
    let scaled = region.inputs.len() == 5;
    if !is_library(region)
        || library_op_byte(region) != LibraryOp::NucleusSample as u8
        || region.schedule != ScheduleTemplate::Library
        || region.nodes.len() != 13
        || !(region.inputs.len() == 3 || scaled)
        || region.outputs.len() != 1
        || !region.sinks.is_empty()
        || region
            .inputs
            .iter()
            .any(|value| *value as usize >= value_types.len())
        || region.outputs[0] as usize >= value_types.len()
    {
        return false;
    }
    let raw_logits_type = &value_types[region.inputs[0] as usize];
    let scale_type = &value_types[region.inputs[if scaled { 1 } else { 0 }] as usize];
    let logits_type = &value_types[region.inputs[if scaled { 2 } else { 0 }] as usize];
    let top_p_type = &value_types[region.inputs[if scaled { 3 } else { 1 }] as usize];
    let state_type = &value_types[region.inputs[if scaled { 4 } else { 2 }] as usize];
    let output_type = &value_types[region.outputs[0] as usize];
    if logits_type.dtype != Dtype::F32 || logits_type.dims.is_empty() || logits_type.dims.len() > 2
    {
        return false;
    }
    if raw_logits_type.dtype != Dtype::F32
        || raw_logits_type.dims.is_empty()
        || raw_logits_type.dims.last() != logits_type.dims.last()
    {
        return false;
    }
    let row_dims = &logits_type.dims[..logits_type.dims.len() - 1];
    top_p_type.dtype == Dtype::F32
        && (top_p_type.dims.is_empty() || top_p_type.dims.len() == row_dims.len())
        && (!scaled
            || (scale_type.dtype == Dtype::F32
                && (scale_type.dims.is_empty() || scale_type.dims.len() == row_dims.len())))
        && state_type.dtype == Dtype::U32
        && state_type.dims.len() == 1
        && state_type.dims[0] == Dimension::Static(2)
        && output_type.dtype == Dtype::I32
        && output_type.dims == row_dims
}

pub fn library_region_valid(stage: &CompiledStage, region: &Region) -> bool {
    if !is_library(region) {
        return true;
    }
    if library_op_byte(region) == LibraryOp::NucleusSample as u8 {
        return nucleus_library_region_valid(stage, region);
    }
    // A top-k sampler is the nucleus chain with a rank cut: Metal runs its
    // thirteen nodes as a generated region.
    if library_op_byte(region) == LibraryOp::LaneCut as u8 {
        return region.nodes.len() == 1 && region.outputs.len() == 1 && region.sinks.is_empty();
    }
    if library_op_byte(region) == LibraryOp::RankSample as u8 {
        return region.nodes.len() == 13 && region.outputs.len() == 1 && region.sinks.is_empty();
    }
    let ops = &stage.normalized.ops;
    if region.nodes.len() != 1 || region.nodes[0].index() >= ops.len() {
        return false;
    }
    let tag = ops[region.nodes[0].index()].tag();
    match region.kind {
        RegionKind::Library(LibraryOp::TopK) => tag == tags::TOP_K,
        RegionKind::Library(LibraryOp::Sort) => tag == tags::SORT_DESC,
        RegionKind::Library(LibraryOp::Scan) => tag == tags::CUMSUM || tag == tags::CUMPROD,
        RegionKind::Library(LibraryOp::MatMul) => tag == tags::MATMUL,
        RegionKind::Library(LibraryOp::SecondParty) => {
            tag == tags::KERNEL_CALL || tag == tags::SINK_CALL
        }
        _ => false,
    }
}

pub fn used_channel_slots(ops: &[OpView]) -> usize {
    let mut count = 0usize;
    for op in ops {
        if op.chan >= 0 {
            count = count.max(op.chan as usize + 1);
        }
    }
    count
}

fn partition_valid(stage: &CompiledStage, partition: &RegionPartition) -> Result<(), EmitError> {
    for region in &partition.regions {
        region_ranges_valid(stage, region, RegionForm::Unnamed)?;
        if !library_region_valid(stage, region) {
            return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::Unnamed));
        }
    }
    Ok(())
}

pub fn validate_singleton_plan(stage: &CompiledStage) -> Result<Vec<M1OpMeta>, EmitError> {
    let (operations, result) = validate_singleton_plan_partial(stage);
    result.map(|()| operations)
}

pub fn validate_singleton_plan_partial(
    stage: &CompiledStage,
) -> (Vec<M1OpMeta>, Result<(), EmitError>) {
    let mut operations = Vec::new();
    let result = validate_into(stage, &mut operations);
    (operations, result)
}

fn validate_into(stage: &CompiledStage, operations: &mut Vec<M1OpMeta>) -> Result<(), EmitError> {
    let normalized = &stage.normalized;
    let value_types = &normalized.value_types;
    let names = &normalized.names;

    if stage.signature.hash == 0
        || eta_ir::fnv1a64(&stage.signature.canonical_bytes) != stage.signature.hash
        || stage.singleton.kind != PartitionKind::Singleton
    {
        return Err(EmitError::SingletonPlanIdentityInvalid);
    }
    value_types_valid(stage)?;
    partition_valid(stage, &stage.singleton)?;
    partition_valid(stage, &stage.fused)?;

    let ops = OpView::of_all(&normalized.ops);
    if stage.singleton.regions.len() != ops.len() {
        return Err(EmitError::SingletonPartitionArityMismatch);
    }
    operations.reserve(ops.len());
    let mut result_base: u32 = 0;
    for (node, op) in ops.iter().enumerate() {
        let region = &stage.singleton.regions[node];
        if region.nodes.len() != 1 || region.nodes[0].index() != node {
            return Err(EmitError::SingletonRegionOrderingMismatch);
        }
        if op.tag == tags::KERNEL_CALL {
            let identity = names
                .get(op.name_idx as usize)
                .is_some_and(|name| name == "metal.identity")
                && op.args.len() == 1
                && (result_base as usize) < value_types.len()
                && value_types
                    .get(op.args[0] as usize)
                    .is_some_and(|argument| *argument == value_types[result_base as usize]);
            if !identity {
                return Err(EmitError::UnsupportedKernelBoundary);
            }
        } else if op.tag == tags::SINK_CALL
            && names.get(op.name_idx as usize).map(String::as_str) != Some("metal.discard")
        {
            return Err(EmitError::UnsupportedSinkBoundary);
        } else if op.tag == tags::INTRINSIC_VAL && !metal_intrinsic_supported(op.intr) {
            return Err(EmitError::UnbindableIntrinsic { intrinsic: op.intr });
        }
        op_valid(op, result_base, stage)?;
        operations.push(M1OpMeta {
            node: node as u32,
            result_base,
            op: op.clone(),
        });
        result_base += op.results;
    }
    if stage.singleton.whole_stage_fallback {
        return Err(EmitError::WholeStageFallbackWithoutCause);
    }
    if result_base as usize != value_types.len() {
        return Err(EmitError::NormalizedValueLayoutMismatch(
            ValueLayoutSite::MetalNormalized,
        ));
    }
    Ok(())
}
