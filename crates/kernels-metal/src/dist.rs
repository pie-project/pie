//! The `Dist` family: one point, and this plane has no answer for it.
//!
//! The file exists for the reason `kernels_cuda::attn::POOL_CLAIMS` is listed
//! empty rather than omitted — a family a plane implements and does not claim
//! is a MEASUREMENT, and a family a plane does not implement at all is a hole
//! in the table where a measurement should be.

use crate::routine::Ctx;

/// The `Dist` family, implemented and claiming nothing.
///
/// SEAM: `dist.all_reduce` sums one rectangle across the shards of a
/// tensor-parallel deployment, and this plane has no collective — no `.metal`
/// entrypoint, and no transport under it either. Cuda's answer is NCCL
/// reached from the body; the metal analogue is a decision about what a
/// multi-device Metal deployment even is, which is a driver question and not
/// a kernel one. Until it is answered, `-tp1` is the only degree this plane
/// serves and the point's default body is the honest row.
#[kernels_macros::claims]
impl kernels::points::Dist for Ctx<'_> {}
