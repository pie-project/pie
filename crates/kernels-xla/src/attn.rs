pub mod arbiter;
pub mod dense;
pub mod index;
pub mod merge;
pub mod mla;
pub mod paged;
pub mod ple;
pub mod pool;
pub mod ragged;
pub mod rel;
pub mod score;
pub mod ssm;

#[allow(unused_imports)]
pub use paged::*;

use crate::tensor::Tensor;

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
