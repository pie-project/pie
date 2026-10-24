#![allow(clippy::too_many_arguments)]

use super::rope::Yarn;
use super::*;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::Tensor;

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_rope_partial_q(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    positions: Tensor,
    rotary_dim: u32,
    theta: f32,
    span: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "fused.norm_rope";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    positions_stream(OP, positions, &x);
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !y.width.is_multiple_of(head_dim)
        || !rotary_dim.is_multiple_of(2)
        || span < rotary_dim / 2
        || span + rotary_dim / 2 > head_dim
    {
        return Err(refuse(
            OP,
            format!(
                "wants whole even heads and an even rotary width whose pairs sit within them, \
                 and was handed width={}, head_dim={head_dim}, rotary_dim={rotary_dim}, \
                 span={span}",
                y.width
            ),
        ));
    }
    let heads = y.width / head_dim;
    if y.rows == 0 || heads == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            ROPE_UNIT,
            "::pie::elemwise::q_rmsnorm_rope_partial<128>",
        ))
        .apply(Launch::grid([y.rows, heads, 1], [128, 1, 1])),
        &[
            x.arg(),
            weight.arg(),
            positions.arg(),
            y.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, span)?.arg(),
            theta.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

/// A head's norm and its tail's rotation in one pass, which is CUDA's
/// (`fused.norm_rope` turning `Last`, outside ROCm's fused kernels).
#[allow(clippy::too_many_arguments)]
pub fn norm_rope_last(
    _ctx: &Ctx,
    _x: Tensor,
    _weight: Option<Tensor>,
    _head_dim: u32,
    _eps: f32,
    _positions: Tensor,
    _rotary_dim: u32,
    _theta: f32,
    _interleaved: bool,
    _inverse: bool,
    _yarn: Option<Yarn>,
    _y: &mut Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "fused.norm_rope",
    })
}
