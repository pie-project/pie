//! `elemwise.add_bias` from `kernels/elemwise/add_bias.slang`: a bias row
//! added to every row, a pair of columns a lane.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, elementwise_rows, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const ADD_BIAS: &str = "elemwise/add_bias.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.add_bias";
    let entry = match (out.dtype, bias.dtype) {
        (Dtype::Bf16, Dtype::Bf16) => "elemwise_add_bias_bf16",
        (Dtype::F32, Dtype::F32) => "elemwise_add_bias_f32",
        (Dtype::F32, Dtype::Bf16) => "elemwise_add_bias_f32_bf16",
        (row, bank) => return Err(refuse(OP, format!("a {bank:?} bias on a {row:?} row"))),
    };
    let [width, rows, _] = elementwise_rows(OP, out.width, out.rows)?;
    if out.dtype == Dtype::Bf16 && !width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {width}-wide bf16 row is odd; rows land two values a word"),
        ));
    }
    ctx.fire(
        Fire::at(ADD_BIAS, entry).apply(Grid::of([width.div_ceil(2), rows, 1], [GROUP, 1, 1])),
        &[out.arg_mut(), bias.arg(), stated(OP, width)?.arg()],
    )
}
