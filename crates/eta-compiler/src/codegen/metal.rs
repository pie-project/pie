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
    M2_INTRINSIC_TOP_BUFFER, M2_LOGITS_BUFFER, fused_channel_ceiling, m2_intrinsic_buffer,
    m2_intrinsic_element_bytes, m3_intrinsic_bindable,
};
pub use nucleus::emit_grouped_nucleus;
pub use preamble::RUNTIME_TEMPLATE;
pub use singleton::emit_singleton_region;
pub use streamed::{
    StepKind, emit_streamed_region, reduce_dispatch_levels, reduce_levels, step_kind, step_value,
    streamed_step,
};
pub use streamed_topk::emit_streamed_topk;
pub use topk::emit_grouped_topk;
pub use validate::validate_singleton_plan;

pub const METAL_M1_EMITTER_VERSION: u16 = 54;

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
