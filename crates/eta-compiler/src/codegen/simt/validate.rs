use crate::codegen::error::{EmitError, RegionForm, ValueLayoutSite};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};

use crate::plan::{CompiledStage, LibraryOp, Region, RegionKind, library_op_for_tag};
use eta_ir::op::Op;
use eta_ir::registry::Stage;

pub fn second_party_region_supported(stage: &CompiledStage, region: &Region) -> bool {
    if region.nodes.len() != 1 {
        return false;
    }
    let node = region.nodes[0].index();
    let Some(op) = stage.normalized.ops.get(node) else {
        return false;
    };
    let value_types = &stage.normalized.value_types;

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
                mask.dims.len() == 1 && stage.normalized.stage == Stage::Prologue
            }
            "lora" => {
                (args.len() == 3 || args.len() == 2) && stage.normalized.stage == Stage::Prologue
            }
            _ => false,
        };
    }

    false
}

pub fn validate_generated_region(stage: &CompiledStage, region: &Region) -> Result<(), EmitError> {
    if region.nodes.is_empty() {
        return Err(EmitError::FusedRequiresGeneratedRegion);
    }
    if let RegionKind::Library(claimed) = region.kind {
        let honest = if claimed == LibraryOp::LaneCut {
            region.nodes.len() == 1 && region.inputs.len() == 2 && region.outputs.len() == 1
        } else if matches!(claimed, LibraryOp::NucleusSample | LibraryOp::RankSample) {
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
        if let Some(library) = library_op_for_tag(op.tag()) {
            return Err(EmitError::GeneratedRegionHasBoundary {
                library_op: library.name(),
            });
        }
    }
    Ok(())
}
