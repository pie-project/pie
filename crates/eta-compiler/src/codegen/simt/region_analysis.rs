use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use eta_ir::op::IntrinsicId;

use crate::plan::CompiledStage;

use super::fused::analyze_direct_argmax;
use super::validate::{second_party_region_supported, validate_generated_region};
use crate::codegen::op_view::{OpView, result_bases};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectArgmax {
    pub node: u32,
    pub source_value: u32,
    pub intrinsic: IntrinsicId,
    pub requires_single_row: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionAnalysis {
    pub stage_index: u32,
    pub region_index: u32,
    pub second_party_supported: bool,
    pub generated_valid: bool,
    pub direct_argmax: Vec<DirectArgmax>,
    pub skipped: Vec<u32>,
}

pub fn analyze_program(stages: &[CompiledStage]) -> Vec<RegionAnalysis> {
    let mut out = Vec::new();
    for (stage_index, stage) in stages.iter().enumerate() {
        let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
        let bases = result_bases(&ops);
        for (region_index, region) in stage.fused.regions.iter().enumerate() {
            let direct = analyze_direct_argmax(stage, region, &bases);
            let mut records = Vec::new();
            for node in 0..direct.intrinsic.len() {
                if let Some(intrinsic) = IntrinsicId::from_u16(direct.intrinsic[node]) {
                    records.push(DirectArgmax {
                        node: node as u32,
                        source_value: direct.source_value[node],
                        intrinsic,
                        requires_single_row: direct.requires_single_row[node] != 0,
                    });
                }
            }
            let skipped: Vec<u32> = direct
                .skipped
                .iter()
                .enumerate()
                .filter(|&(_, &flag)| flag != 0)
                .map(|(node, _)| node as u32)
                .collect();

            out.push(RegionAnalysis {
                stage_index: stage_index as u32,
                region_index: region_index as u32,
                second_party_supported: second_party_region_supported(stage, region),
                generated_valid: validate_generated_region(stage, region).is_ok(),
                direct_argmax: records,
                skipped,
            });
        }
    }
    out
}
