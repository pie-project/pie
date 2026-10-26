use crate::codegen::op_view::{OpView, output_slot, reads_readout_plane};
use crate::plan::CompiledStage;

pub const M2_LOGITS_BUFFER: usize = 6;

pub const M2_INTRINSIC_TOP_BUFFER: usize = 30;

/// The single-lane form's output table: one `M1OutputSeat` a slot, [`eta_ir::op::output_pitch`] of them.
pub const M2_INTRINSIC_TABLE_BUFFER: usize = M2_INTRINSIC_TOP_BUFFER - 3;

/// The grouped and streamed forms' output tables, one an instance, by dispatch lane.
pub const M3_INTRINSIC_TABLE_BUFFER: usize = 12;

/// The address-table slot of output op `op` (its result at `base`); `None`
/// for a float output's readout rows, read through the logits argument.
#[must_use]
pub fn addressed_slot(stage: &CompiledStage, op: &OpView, base: u32) -> Option<u32> {
    (!reads_readout_plane(stage, op, base)).then(|| u32::from(output_slot(stage, op)))
}

/// The argument output op `op` (its result at `base`) is read through.
#[must_use]
pub fn m2_output_buffer(stage: &CompiledStage, op: &OpView, base: u32) -> usize {
    if reads_readout_plane(stage, op, base) {
        M2_LOGITS_BUFFER
    } else {
        M2_INTRINSIC_TABLE_BUFFER
    }
}

/// The intrinsic argument indices a region binds sit above the channels.
#[must_use]
pub fn fused_channel_ceiling(used: &[usize]) -> usize {
    let lowest = used
        .iter()
        .copied()
        .filter(|&at| at > M2_LOGITS_BUFFER)
        .min();
    match lowest {
        Some(lowest) => lowest.saturating_sub(7) / 2,
        None => super::METAL_M2_MAX_FUSED_CHANNELS,
    }
}
