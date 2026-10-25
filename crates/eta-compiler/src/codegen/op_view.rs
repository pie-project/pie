pub use eta_ir::wire::{OpWire as OpView, result_bases};

use eta_ir::op::tags;

use crate::plan::CompiledStage;

/// Where output op `op` sits in its lane's table.
pub(crate) fn output_slot(stage: &CompiledStage, op: &OpView) -> u16 {
    u16::try_from(eta_ir::op::output_slot(
        &stage.normalized.ops,
        op.name_idx,
        op.imm,
    ))
    .unwrap_or(u16::MAX)
}

/// Whether `op`, its result at `base`, reads a float output's readout rows:
/// the plane a fused argmax or top-k scans in place.
pub(crate) fn reads_readout_plane(stage: &CompiledStage, op: &OpView, base: u32) -> bool {
    op.tag == tags::OUTPUT
        && stage
            .normalized
            .value_types
            .get(base as usize)
            .is_some_and(crate::plan::readout_plane)
}
