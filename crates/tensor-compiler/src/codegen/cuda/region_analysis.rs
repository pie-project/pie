//! Per-region decisions the host makes and the CUDA engine currently
//! re-derives.
//!
//! CUDA-only, and filed here to say so. These are the other half of *this*
//! backend's contract — which regions bind, and how the generated kernel's
//! intrinsic side tables are laid out — so they mean nothing to an engine
//! running someone else's kernels. The runtime already gates on that
//! (`crates/runtime/src/engine/backend.rs`, `codegen_backend == Some("cuda")`)
//! and no Metal path reads a [`RegionAnalysis`].
//!
//! Two per-program analyses the launch package has yet to absorb still live
//! in one C++ file, `crates/driver-cuda/csrc/src/pipeline/region_support.hpp`:
//!
//! * the **bind-time region gates** — `second_party_region_supported` and
//!   `validate_generated_region`, which decide whether a region can be bound
//!   and whether it can be emitted;
//! * the **intrinsic side-table analysis** — `analyze_direct_argmax`, which
//!   decides which `argmax` reads a logits intrinsic's device buffer straight
//!   and which nodes that makes redundant.
//!
//! All three already exist here, because the emitter needs the same answers to
//! generate the kernel. That is the problem: the emitted kernel and the host
//! packer that fills its side tables are two ends of one contract, and today
//! they are decided by two implementations in two languages. A disagreement
//! does not fail to compile — it produces a kernel that reads a slot the packer
//! never wrote.
//!
//! So the answers ship. The engine keeps its derivation while both exist,
//! compares, and counts divergence; the copy goes when the counter is zero
//! *and* the host-supplied counter is not (`e50769003` — a comparison that
//! never ran reports the same zero as one that always agreed).

use alloc::vec::Vec;

use crate::plan::CompiledStage;

use super::fused::analyze_direct_argmax;
use super::validate::{second_party_region_supported, validate_generated_region};
use crate::codegen::op_view::{OpView, result_bases};

/// The region can be bound as a second-party (non-generated) region — today
/// `envelope_dot` and the `attn_page_mask` sink.
pub const REGION_SECOND_PARTY_SUPPORTED: u32 = 1 << 0;
/// The region is a well-formed generated region, so the fused emitter accepts
/// it. Mutually exclusive with the flag above in practice, but derived
/// independently, so recorded independently.
///
/// NOT the same question as "can this backend run the region". A `top_k`,
/// `sort_desc` or scan library region answers `false` here — none is a
/// generated region and the fused emitter does not accept any — and each is
/// still emitted, by [`super::order`] or [`super::scan`]. What reaches the
/// engine is the kernel table; [`super::emit_region`] is the function that
/// decides it.
pub const REGION_GENERATED_VALID: u32 = 1 << 1;

/// One `argmax` that reads a logits intrinsic's buffer directly.
///
/// Sparse on purpose: the engine's form is four arrays as long as the stage's
/// op list, and almost every entry is the "does not apply" sentinel. What the
/// packer actually consumes is this handful of records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectArgmaxRecord {
    /// The `argmax` node, in stage op order.
    pub node: u32,
    /// The value id of the intrinsic buffer it reads instead.
    pub source_value: u32,
    /// `IntrinsicId` of that buffer — `Logits` or `MtpLogits`.
    pub intrinsic: u16,
    /// The rewrite is only valid if the source has exactly one runtime row,
    /// which is a per-fire check the engine makes against the lane's
    /// descriptors.
    pub requires_single_row: u8,
}

/// Every decision about one region that the engine derives for itself today.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionAnalysis {
    /// Index of the stage this region belongs to, matching `emit_program`'s
    /// numbering so the two tables join on `(stage_index, region_index)`.
    pub stage_index: u32,
    /// Index of this region within its stage, matching `emit_program`'s.
    pub region_index: u32,
    /// `REGION_*` bits.
    pub flags: u32,
    /// The [`DirectArgmaxRecord`]s for this region's `argmax` nodes that read a
    /// logits intrinsic's buffer directly; empty when none qualify.
    pub direct_argmax: Vec<DirectArgmaxRecord>,
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
            let mut flags = 0u32;
            if second_party_region_supported(stage, region) {
                flags |= REGION_SECOND_PARTY_SUPPORTED;
            }
            if validate_generated_region(stage, region).is_ok() {
                flags |= REGION_GENERATED_VALID;
            }

            let direct = analyze_direct_argmax(stage, region, &bases);
            let mut records = Vec::new();
            for node in 0..direct.intrinsic.len() {
                if direct.intrinsic[node] != u16::MAX {
                    records.push(DirectArgmaxRecord {
                        node: node as u32,
                        source_value: direct.source_value[node],
                        intrinsic: direct.intrinsic[node],
                        requires_single_row: direct.requires_single_row[node],
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
                flags,
                direct_argmax: records,
                skipped,
            });
        }
    }
    out
}
