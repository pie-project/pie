//! `elemwise.tanh` from `kernels/elemwise/tanh.slang`: `x = tanh(x)` in
//! place.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise};
use crate::error::Error;
use crate::tensor::Tensor;

const TANH: &str = "elemwise/tanh.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn tanh(ctx: &Ctx<'_>, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.tanh";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_tanh_bf16",
        F32 => "elemwise_tanh_f32",
    });
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(TANH, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), n.arg()],
    )
}
