pub mod effects;
pub mod fused;
pub mod intrinsics;
pub mod nucleus;
pub mod preamble;
pub mod singleton;
pub mod streamed;
pub mod streamed_topk;
pub mod topk;
pub mod validate;

pub use crate::codegen::op_view::OpView;
pub use effects::{
    channel_effects, emit_commit, emit_grouped_commit, emit_grouped_readiness, emit_readiness,
};
pub use fused::{emit_fused_region, emit_grouped_fused_region};
pub use intrinsics::{
    M2_INTRINSIC_TABLE_BUFFER, M2_INTRINSIC_TOP_BUFFER, M2_LOGITS_BUFFER,
    M3_INTRINSIC_TABLE_BUFFER, addressed_slot, fused_channel_ceiling, m2_output_buffer,
};
pub use nucleus::emit_grouped_nucleus;
pub use preamble::RUNTIME_TEMPLATE;
pub use singleton::emit_singleton_region;
pub use streamed::{
    STEP_NODE, STEP_NODES, StepKind, emit_streamed_region, reduce_dispatch_levels, reduce_levels,
    step_kind, step_node_table, step_value, streamed_step,
};
pub use streamed_topk::emit_streamed_topk;
pub use topk::emit_grouped_topk;
pub use validate::validate_singleton_plan;

pub const METAL_M1_EMITTER_VERSION: u16 = 64;

/// The pieces a lane's scratch lies in: the grouped kernels read lane L's
/// base for piece P at `scratch_bases[L * M3_SCRATCH_PIECES + P]` (buffer
/// 4, GPU addresses), so a lane's scratch lies wherever its instance holds
/// it, not a stride from the batch's first. One piece until eta_exec's
/// chunk-aware layout states a program's pieces (no value crosses one).
pub const M3_SCRATCH_PIECES: u32 = 1;

/// The line every grouped kernel takes its lane's scratch by: piece 0's base.
pub const M3_LANE_SCRATCH: &str = "  device uchar* scratch = reinterpret_cast<device uchar*>(\
     scratch_bases[ulong(dispatch_lane) * ulong(M3_SCRATCH_PIECES)]);\n";

pub const METAL_M1_MAX_CHANNELS: usize = 29;

pub const METAL_M2_MAX_FUSED_CHANNELS: usize = 12;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct M1ChannelEffect {
    pub requires_full: bool,
    pub requires_empty: bool,
    pub take: bool,
    pub put: bool,
    pub capacity: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct M1OpMeta {
    pub node: u32,
    pub result_base: u32,
    pub op: OpView,
}
