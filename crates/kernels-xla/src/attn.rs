pub mod arbiter;
pub mod dense;
pub mod index;
pub mod merge;
pub mod mla;
pub mod paged;
pub mod pool;
pub mod ragged;
pub mod replay;
pub mod rel;
pub mod score;
pub mod voxel;

#[allow(unused_imports)]
pub use paged::*;

/// What a paged decode reads besides q and the pool: the fire's per-token
/// positions and owning request, and the custom-mask plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan<T> {
    pub positions: T,

    pub request_of_token: T,

    pub mask: T,

    pub mask_enabled: T,

    pub mask_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan<T> {
    pub positions: T,

    pub request_of_token: T,

    pub mask: T,

    pub mask_enabled: T,

    pub mask_stride: u32,
}
