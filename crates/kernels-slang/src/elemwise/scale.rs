//! `elemwise.mul_scalar` and `elemwise.scale` from
//! `kernels/elemwise/scale.slang`: a tensor times a scalar in place, the
//! scalar stated or read from a one-element tensor.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise};
use crate::error::Error;
use crate::tensor::Tensor;

const SCALE: &str = "elemwise/scale.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// `x *= s`, `s` rounded as `x` holds it.
pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.mul_scalar";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_scale_bf16",
        F32 => "elemwise_scale_f32",
    });
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(SCALE, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), s.arg(), n.arg()],
    )
}

/// `x *= s[0]`.
pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.scale";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_scale_read_bf16" });
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(SCALE, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), s.arg(), n.arg()],
    )
}
