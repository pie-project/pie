//! Numeric formats, and nothing else.
//!
//! Every function here takes bytes and returns bytes or numbers, and none knows
//! what a `LoadPlan` is. This is the **reference implementation every device
//! kernel is diffed against**: the guarantee that a checkpoint produces the
//! same weights whichever backing ran the transforms rests on `dequant_fp4.cu`
//! and `quantize_fp8.cu` agreeing element for element with [`mxfp4`]/[`fp8`].
//! Split by FORMAT, not direction; [`cast`] is the exception, belonging to no
//! one scheme.

pub mod cast;
pub mod e8m0;
pub mod fp8;
pub mod int4;
pub mod mlx;
pub mod mxfp4;
pub mod rows;

use crate::error::Error;

/// `Contract` rather than `Internal`: reaching an unimplemented dtype means a
/// plan named one, and a plan is a contract's consequence.
fn invalid(message: impl Into<String>) -> Error {
    Error::Contract(message.into())
}
