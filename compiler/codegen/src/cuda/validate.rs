//! The region ABI checks the CUDA fused emitter gates on.
//!
//! Ported from `fused_codegen.hpp`; `validate_generated_region` is what decides
//! whether a region can be emitted at all, and `second_party_region_supported`
//! names the one second-party kernel this backend launches.

use alloc::string::{String, ToString};

use pie_ir::op::Op;
use pie_ir::registry::Stage;
use pie_ir::types::DType;
use pie_plan::{CompiledStage, Region, RegionKind, ScheduleTemplate};

/// `second_party_region_supported` — `envelope_dot` is the only second-party
/// kernel this backend launches, so anything else fails at bind rather than
/// reaching a runtime throw mid-fire.
///
/// It is deliberately not type-checked the way `cuda.identity` is: its argument
/// is the query and its result is a per-page score, shapes unrelated by
/// construction, so arity is the only structural claim that holds.
pub fn second_party_region_supported(stage: &CompiledStage, region: &Region) -> bool {
    if region.nodes.len() != 1 {
        return false;
    }
    let node = region.nodes[0] as usize;
    let Some(op) = stage.normalized.ops.get(node) else {
        return false;
    };
    let value_types = &stage.normalized.value_types;

    // `attn_page_mask(mask)` — a configuration sink. One argument, no result.
    // The mask is a per-page vector over the request's page list, so the only
    // structural claim that holds is rank 1; its extent is the program's own
    // page ceiling, which the runtime checks against the lane's page count.
    if let Op::SinkCall { name, args } = op {
        let Some(sink) = stage.normalized.names.get(*name as usize) else {
            return false;
        };
        if sink != "attn_page_mask" || args.len() != 1 {
            return false;
        }
        if !region.outputs.is_empty() || region.inputs.len() != 1 {
            return false;
        }
        let Some(mask) = value_types.get(region.inputs[0] as usize) else {
            return false;
        };
        return mask.dims.len() == 1 && stage.normalized.stage == Stage::OnAttnProj;
    }

    let Op::KernelCall { name, args, .. } = op else {
        return false;
    };
    let Some(kernel) = stage.normalized.names.get(*name as usize) else {
        return false;
    };
    if kernel != "envelope_dot" || args.len() != 1 || region.outputs.len() != 1 {
        return false;
    }
    // The score is a per-page f32 vector. A different rank or dtype means the
    // program disagrees with the kernel's ABI.
    let Some(result) = value_types.get(region.outputs[0] as usize) else {
        return false;
    };
    if result.dtype != DType::F32 || result.dims.len() != 1 {
        return false;
    }
    matches!(stage.normalized.stage, Stage::OnAttnProj | Stage::OnAttn)
}

/// `validate_generated_region`.
pub fn validate_generated_region(stage: &CompiledStage, region: &Region) -> Result<(), String> {
    if matches!(region.kind, RegionKind::Library(_))
        || region.schedule == ScheduleTemplate::Library
        || region.nodes.is_empty()
    {
        return Err("fused CUDA emitter requires a non-library generated region".to_string());
    }
    let mut previous = 0u32;
    let mut have_previous = false;
    for &node in &region.nodes {
        if node as usize >= stage.normalized.ops.len() || (have_previous && node <= previous) {
            return Err("generated region nodes are invalid or unordered".to_string());
        }
        let op = &stage.normalized.ops[node as usize];
        if matches!(op, Op::KernelCall { .. } | Op::SinkCall { .. }) {
            return Err("generated region contains a non-generated boundary".to_string());
        }
        previous = node;
        have_previous = true;
    }
    Ok(())
}
