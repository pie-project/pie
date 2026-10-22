//! Layer norms (`elemwise.layernorm`, with and without its affine) from
//! `kernels/elemwise/layernorm.slang`: one workgroup a row.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const LAYERNORM: &str = "elemwise/layernorm.slang";

/// The workgroup width every layernorm instantiation is built at.
const GROUP: u32 = 256;

fn grid(op: &'static str, x: Tensor) -> Result<Grid, Error> {
    nonzero(op, "width", x.width)?;
    let rows = nonzero(op, "rows", x.rows)?;
    if x.dtype == Dtype::Bf16 && !x.width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "a {}-wide bf16 row is odd; rows land two values a word",
                x.width
            ),
        ));
    }
    let lanes = rows
        .checked_mul(GROUP)
        .ok_or_else(|| refuse(op, format!("{rows} rows will not launch")))?;
    Ok(Grid::of([lanes, 1, 1], [GROUP, 1, 1]))
}

/// `y = (x - mean) / sqrt(var + eps) * weight + bias` over each row.
pub fn layernorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.layernorm";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_layernorm_bf16" });
    ctx.fire(
        Fire::at(LAYERNORM, entry).apply(grid(OP, x)?),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg_mut(),
            eps.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

/// Each row standardized, with no weight and no bias.
pub fn layernorm_no_scale(ctx: &Ctx<'_>, x: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.layernorm";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_layernorm_unscaled_bf16",
        F32 => "elemwise_layernorm_unscaled_f32",
    });
    ctx.fire(
        Fire::at(LAYERNORM, entry).apply(grid(OP, x)?),
        &[x.arg(), y.arg_mut(), eps.arg(), stated(OP, x.width)?.arg()],
    )
}
