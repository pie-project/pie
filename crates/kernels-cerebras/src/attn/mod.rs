//! Attention families. Paged attention lives in [`paged`] and is re-exported
//! here, matching kernels-wgpu's `attn::` root.

pub mod dense;
pub mod dynconv;
pub mod index;
pub mod paged;
pub mod ple;
pub mod ssm;

use crate::tensor::Tensor;

pub use paged::{
    cap_rows_per_phase, decode, decode_lse, decode_selected, kv_append, kv_append_shared, masked,
    masked_lse, plan_decode, plan_prefill, prefill, prefill_lse, prefill_selected,
};

/// What a paged decode reads besides q and the pool: the fire's per-token
/// positions and owning request, and the custom-mask plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    pub positions: Tensor,
    pub request_of_token: Tensor,
    pub mask: Tensor,
    pub mask_enabled: Tensor,
    pub mask_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    pub positions: Tensor,
    pub request_of_token: Tensor,
    pub mask: Tensor,
    pub mask_enabled: Tensor,
    pub mask_stride: u32,
}
