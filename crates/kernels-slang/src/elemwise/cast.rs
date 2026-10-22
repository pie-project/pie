//! `elemwise.cast` from `kernels/elemwise/cast.slang`: f32 values rounded
//! into bf16 ones.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, elementwise, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const CAST: &str = "elemwise/cast.slang";

/// The workgroup width the instantiation is built at.
const GROUP: u32 = 256;

pub fn cast_f32_to_bf16(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.cast";
    if x.dtype != Dtype::F32 || o.dtype != Dtype::Bf16 {
        return Err(refuse(OP, format!("{:?} into {:?}", x.dtype, o.dtype)));
    }
    let [n, _, _] = elementwise(OP, o.width, o.rows)?;
    ctx.fire(
        Fire::at(CAST, "elemwise_cast_f32_to_bf16")
            .apply(Grid::of([n.div_ceil(2), 1, 1], [GROUP, 1, 1])),
        &[x.arg(), o.arg_mut(), n.arg()],
    )
}
