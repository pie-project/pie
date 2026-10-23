pub mod gate;
pub mod hc;
pub mod modulate;
pub mod norm;
pub mod norm_residual;
pub mod norm_rope;
pub mod pointwise;
pub mod posenc;
pub mod rope;

use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Launch, nonzero, refuse};
use crate::tensor::Tensor;

const NORM_UNIT: &str = "elemwise/norm.hip";

const BLOCK: u32 = 256;

/// Lanes per wavefront on CDNA: a block is a whole number of them.
const WARP: u32 = 64;

const VEC_WIDTH: u32 = 8;

pub(crate) fn elementwise(op: &'static str, t: Tensor) -> Result<(Launch, u64), Error> {
    let n = t.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    Ok((Launch::flat(lanes, BLOCK), n))
}

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

const ROPE_UNIT: &str = "elemwise/rope.hip";

fn heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this rotation states", head_dim)?;
    if !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn positions_stream(op: &'static str, positions: Tensor, x: &Tensor) {
    debug_assert_eq!(
        positions.dtype,
        Dtype::I32,
        "`{op}` reads an i32 position stream"
    );
    debug_assert_eq!(
        positions.rows, x.rows,
        "the position stream is one entry per rotated row"
    );
}
