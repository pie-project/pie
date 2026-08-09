//! Numeric formats, and nothing else.
//!
//! Every function here takes bytes and returns bytes or numbers. None of them
//! knows what a `LoadPlan` is, which is why they are addressable without one.
//!
//! # Why this is a module and not a region of the walker
//!
//! This code is the **reference implementation every device kernel is diffed
//! against**. `kernels/dequant_fp4.cu` and `quantize_fp8.cu` are checked
//! element for element against [`mxfp4`] and [`fp8`], and the load path's
//! guarantee — that a checkpoint produces the same weights whichever backing
//! ran the transforms — rests entirely on those two agreeing.
//!
//! It spent its life as six hundred lines in the middle of the plan walker,
//! reachable only by compiling a plan and executing it. A reference
//! implementation that can only be reached through the thing it is a
//! reference FOR is a reference in name: it cannot be benchmarked against the
//! kernel it mirrors, and a test of `f32_to_fp8_e4m3` had to build a
//! checkpoint on disk to call it.
//!
//! # How it is split
//!
//! By FORMAT, not by direction. [`fp8`] holds both the decode and the encode
//! of E4M3; [`mxfp4`] holds the nibble table, the group encoder and the AVX2
//! implementation of that same encoder. A round-trip that does not close, or
//! a vector path that disagrees with its scalar path, is then a disagreement
//! between two functions in one file rather than between two files.
//!
//! [`cast`] is the exception, because a cast is not a format: it is the
//! conversion between any dtype's bytes and `f64`, and it belongs to no one
//! scheme.

pub mod cast;
pub mod e8m0;
pub mod fp8;
pub mod int4;
pub mod mlx;
pub mod mxfp4;
pub mod rows;

use crate::error::Error;

/// The one error a codec raises: it was handed a dtype it does not implement.
///
/// `Contract` rather than `Internal` because reaching an unimplemented dtype
/// means a plan named one, and a plan is a contract's consequence.
fn invalid(message: impl Into<String>) -> Error {
    Error::Contract(message.into())
}
