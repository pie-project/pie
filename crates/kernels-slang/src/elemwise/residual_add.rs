//! `elemwise.residual_add` from `kernels/elemwise/residual_add.slang`:
//! `y += x` over the whole tensor, a pair of elements a lane.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const RESIDUAL_ADD: &str = "elemwise/residual_add.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.residual_add";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "elemwise_residual_add_bf16",
        F32 => "elemwise_residual_add_f32",
    });
    let [n, _, _] = elementwise(OP, y.width, y.rows)?;
    if y.dtype == dtype::Dtype::Bf16 && !n.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("{n} bf16 elements are odd; they land two a word"),
        ));
    }
    ctx.fire(
        Fire::at(RESIDUAL_ADD, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg(), y.arg_mut(), n.arg()],
    )
}
