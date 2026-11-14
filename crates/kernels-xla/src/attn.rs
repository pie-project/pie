pub(crate) mod append;
pub mod arbiter;
pub mod dense;
pub mod flash;
pub mod index;
pub mod merge;
pub mod mla;
pub mod paged;
pub mod pages;
pub mod pool;
pub mod ragged;
pub mod rel;
pub mod replay;
pub mod score;
pub mod voxel;

#[allow(unused_imports)]
pub use paged::*;

/// What a paged decode reads besides q and the pool: the fire's per-token
/// positions and owning request, and the custom-mask plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodePlan<T> {
    /// Each lane's keys once this fire lands (`have + rows`), i32: where a
    /// wide row's keys end.
    pub kv_len: T,

    pub positions: T,

    pub request_of_token: T,

    pub mask: T,

    pub mask_enabled: T,

    pub mask_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefillPlan<T> {
    /// Each lane's keys once this fire lands (`have + rows`), i32: where a
    /// wide row's keys end.
    pub kv_len: T,

    pub positions: T,

    pub request_of_token: T,

    pub mask: T,

    pub mask_enabled: T,

    pub mask_stride: u32,
}
