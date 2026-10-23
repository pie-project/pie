//! Attention and its KV cache, from `kernels/attn/`.

use crate::encode::{nonzero, refuse};
use crate::error::Error;

pub mod dense;
pub mod index;
pub mod kv_append;
pub mod merge;
pub mod mla;
pub mod pool;
pub mod score;

/// The heads a `width`-wide row of `head_dim`-wide heads holds.
pub(crate) fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the {width}-wide query row does not divide by the stated head width {head_dim}"
            ),
        ));
    }
    Ok(width / head_dim)
}

/// The bf16 words of a `head_dim`-wide head.
pub(crate) fn pairs(op: &'static str, head_dim: u32) -> Result<u32, Error> {
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the head width {head_dim} is odd; the bf16 kernels write whole pairs"),
        ));
    }
    Ok(head_dim / 2)
}

/// The bf16 words of a `width`-wide `what` plane, which must be even.
pub(crate) fn even_lanes(op: &'static str, what: &str, width: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "the {what} is {width} wide, and a bf16 plane moves as whole words (an even width)"
            ),
        ));
    }
    Ok(width / 2)
}
