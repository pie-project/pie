//! The size-only walk (`kernels_cuda::sizing`): a load walks the
//! compositions it is about to arm with launches, copies and commits
//! standing still, and holds for arming what that walk says arming will
//! take. The device layer's commits and spans answer it here.

pub use kernels_cuda::sizing::{Footprint, active, begin, end, stands_in};

/// The base an unbacked span is named by in a walk: never dereferenced.
pub const STAND_IN: u64 = kernels_cuda::sizing::STAND_IN as u64;
