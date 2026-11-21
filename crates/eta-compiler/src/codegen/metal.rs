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
    M4_STEP_CONSTANT, STEP_NODE, STEP_NODES, STEP_VIEW, StepKind, StepNodes, emit_streamed_region,
    is_streamed_select, reduce_dispatch_levels, reduce_levels, step_kind, step_node_table,
    step_value, streamed_step, streamed_temporary_need,
};
pub use streamed_topk::{TOPK_ROW_BYTES, emit_streamed_topk, streamed_topk_temporary_need};
pub use topk::emit_grouped_topk;
pub use validate::validate_singleton_plan;

/// The library forms Metal's emitters serve: the nucleus sampler over f32
/// rows, its rank cut, an f32 lane cut. The temperature-scaled Gumbel draw
/// and a bf16 row's lane cut are planned as generated regions (Metal's own
/// two-level draw, `m4_gumbel_*`), as before those forms existed.
pub const LIBRARY_FORMS: crate::plan::LibraryForms = crate::plan::LibraryForms {
    draw: false,
    bf16_cut: false,
};

pub const METAL_M1_EMITTER_VERSION: u16 = 80;

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
