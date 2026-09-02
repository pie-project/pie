//! The region ABI checks the CUDA fused emitter gates on.
//!
//! `validate_generated_region` is what decides whether a region can be emitted
//! at all, and `second_party_region_supported` names the one second-party
//! kernel this backend launches.

use crate::codegen::error::{EmitError, RegionForm, ValueLayoutSite};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};

use crate::plan::{CompiledStage, LibraryOp, Region, RegionKind, library_op_for_tag};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::Dtype;

/// `envelope_dot` is the only second-party kernel this backend launches, so
/// anything else fails at bind rather than reaching a runtime throw mid-fire.
/// Not type-checked the way `cuda.identity` is: arity is the only structural
/// claim that holds, since argument and result shapes are unrelated by
/// construction.
pub fn second_party_region_supported(stage: &CompiledStage, region: &Region) -> bool {
    if region.nodes.len() != 1 {
        return false;
    }
    let node = region.nodes[0].index();
    let Some(op) = stage.normalized.ops.get(node) else {
        return false;
    };
    let value_types = &stage.normalized.value_types;

    // Configuration sinks: `attn_page_mask(mask)` (one arg, rank-1 mask) and
    // `lora(a, b, sites)` (3 or 2 args, prologue only). Everything else is
    // refused here rather than mid-fire.
    if let Op::SinkCall { name, args } = op {
        let Some(sink) = stage.normalized.names.get(*name as usize) else {
            return false;
        };
        if !region.outputs.is_empty() || region.inputs.len() != args.len() {
            return false;
        }
        return match sink.as_str() {
            "attn_page_mask" => {
                if args.len() != 1 {
                    return false;
                }
                let Some(mask) = value_types.get(region.inputs[0] as usize) else {
                    return false;
                };
                mask.dims.len() == 1 && stage.normalized.stage == Stage::OnAttnProj
            }
            // 3 args = low-rank (A, B, SITES); 2 args = the scale form (L, SITES).
            "lora" => {
                (args.len() == 3 || args.len() == 2) && stage.normalized.stage == Stage::Prologue
            }
            _ => false,
        };
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
    if result.dtype != Dtype::F32 || result.dims.len() != 1 {
        return false;
    }
    matches!(stage.normalized.stage, Stage::OnAttnProj | Stage::OnAttn)
}

/// Also the gate `region_analysis` asks before declaring a region bindable,
/// so everything emission refuses has to be refused here too — a region that
/// passes analysis and then fails emission reaches the engine as an error
/// kernel instead of a tier-0 fallback.
///
/// A `RegionKind::Library` region is not refused for being one: `Library`
/// says the plan recognized a dataflow a backend *may* have a kernel for, not
/// that every backend has it. What decides emittability is the per-node
/// check below: everything the fused emitter refuses must still be refused
/// here, but `top_k`/`sort_desc`/`cumsum`/`cumprod` regions route to their
/// own kernel via [`super::emit_region`] before reaching the fused emitter,
/// so they emit anyway even though this gate still refuses them.
pub fn validate_generated_region(stage: &CompiledStage, region: &Region) -> Result<(), EmitError> {
    if region.nodes.is_empty() {
        return Err(EmitError::FusedRequiresGeneratedRegion);
    }
    // A library claim still has to be true: nothing downstream re-derives it.
    // `NucleusSample` is the one multi-op lift; its arity is what
    // `compile::nucleus` builds.
    if let RegionKind::Library(claimed) = region.kind {
        let honest = if claimed == LibraryOp::NucleusSample {
            region.nodes.len() == 13
                && (region.inputs.len() == 3 || region.inputs.len() == 5)
                && region.outputs.len() == 1
        } else {
            region.nodes.len() == 1
                && library_op_for_tag(stage.normalized.ops[region.nodes[0].index()].tag())
                    == Some(claimed)
        };
        if !honest {
            return Err(EmitError::FusedRequiresGeneratedRegion);
        }
    }
    value_types_valid(stage)?;
    ops_valid(stage, ValueLayoutSite::CudaFusedStage)?;
    region_ranges_valid(stage, region, RegionForm::Fused)?;
    for &node in &region.nodes {
        let op = &stage.normalized.ops[node.index()];
        // Same classifier `region_kind_for_node` used to build the region,
        // so this check cannot disagree with the decision it's checking.
        if let Some(library) = library_op_for_tag(op.tag()) {
            return Err(EmitError::GeneratedRegionHasBoundary {
                library_op: library.name(),
            });
        }
    }
    Ok(())
}

