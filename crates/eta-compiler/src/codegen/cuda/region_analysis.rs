//! Per-region decisions the host makes and the CUDA engine currently
//! re-derives (CUDA-only; no Metal path reads a [`RegionAnalysis`]). Shipped
//! alongside the engine's own derivation so the two can be compared and the
//! C++ copy retired once they agree.

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use eta_ir::op::IntrinsicId;

use crate::plan::CompiledStage;

use super::fused::analyze_direct_argmax;
use super::validate::{second_party_region_supported, validate_generated_region};
use crate::codegen::op_view::{OpView, result_bases};

/// An engine-side fast path a region admits: an `argmax` the backend can
/// answer without running the region's generated source, by reading a logits
/// intrinsic's device buffer straight.
///
/// Sparse on purpose: [`super::fused::ArgmaxScan`] is four arrays as long as
/// the stage's op list, and almost every entry is the "does not apply"
/// sentinel. What a packer consumes is this handful of records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectArgmax {
    /// The `argmax` node, by index into the plan's ops.
    pub node: u32,
    /// The value id of the intrinsic buffer it reads instead.
    pub source_value: u32,
    /// Which intrinsic that buffer is — `Logits` or `MtpLogits`. An id no
    /// intrinsic claims is dropped rather than shipped.
    pub intrinsic: IntrinsicId,
    /// Whether the path is legal only for a single-row fire.
    pub requires_single_row: bool,
}

/// Every decision about one region that the engine derives for itself today.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionAnalysis {
    /// Index of the stage this region belongs to, matching `emit_program`'s
    /// numbering so the two tables join on `(stage_index, region_index)`.
    pub stage_index: u32,
    /// Index of this region within its stage, matching `emit_program`'s.
    pub region_index: u32,
    /// The region can be bound as a second-party (non-generated) region —
    /// today `envelope_dot` and the `attn_page_mask` sink.
    pub second_party_supported: bool,
    /// The region is a well-formed generated region, so the fused emitter
    /// accepts it. Not the same question as "can this backend run the
    /// region" — a `top_k`/`sort_desc`/scan library region is `false` here
    /// but is still emitted, by [`super::order`] or [`super::scan`].
    pub generated_valid: bool,
    /// The direct-argmax fast paths this region admits; empty when none
    /// qualify.
    pub direct_argmax: Vec<DirectArgmax>,
    /// Nodes made redundant by the rewrites above, ascending. The engine's
    /// dense `skipped` array is exactly this set.
    pub skipped: Vec<u32>,
}

/// Analyse every fused region of every stage, in stage then region order.
///
/// Indices match `emit_program`'s, so an engine can join the two tables on
/// `(stage_index, region_index)` without knowing what a plan is.
pub fn analyze_program(stages: &[CompiledStage]) -> Vec<RegionAnalysis> {
    let mut out = Vec::new();
    for (stage_index, stage) in stages.iter().enumerate() {
        let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
        let bases = result_bases(&ops);
        for (region_index, region) in stage.fused.regions.iter().enumerate() {
            let direct = analyze_direct_argmax(stage, region, &bases);
            let mut records = Vec::new();
            for node in 0..direct.intrinsic.len() {
                // `u16::MAX` names no intrinsic, so `from_u16` skips those rows.
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
