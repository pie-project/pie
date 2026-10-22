//! `elemwise.silu_scaled` from `kernels/elemwise/silu_scaled.slang`:
//! `x = silu(x * s)` in place.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise};
use crate::error::Error;
use crate::tensor::Tensor;

const SILU_SCALED: &str = "elemwise/silu_scaled.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.silu_scaled";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_silu_scaled_bf16",
        F32 => "elemwise_silu_scaled_f32",
    });
    let [n, _, _] = elementwise(OP, x.width, x.rows)?;
    ctx.fire(
        Fire::at(SILU_SCALED, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg_mut(), s.arg(), n.arg()],
    )
}
