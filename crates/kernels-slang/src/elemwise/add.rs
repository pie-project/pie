//! `elemwise.add` from `kernels/elemwise/add.slang`: `z = x + y`, a pair of
//! elements a lane.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const ADD: &str = "elemwise/add.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn add(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.add";
    if x.dtype != z.dtype || y.dtype != z.dtype {
        return Err(refuse(
            OP,
            format!("{:?} + {:?} into {:?}", x.dtype, y.dtype, z.dtype),
        ));
    }
    let entry = dtype_dispatch!(OP, z.dtype, {
        Bf16 => "elemwise_add_bf16",
        F32 => "elemwise_add_f32",
    });
    let [n, _, _] = elementwise(OP, z.width, z.rows)?;
    ctx.fire(
        Fire::at(ADD, entry).apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg(), y.arg(), z.arg_mut(), n.arg()],
    )
}
