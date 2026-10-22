//! `elemwise.standardize` from `kernels/elemwise/standardize.slang`: each
//! column shifted by a bias plane and scaled by a scale plane, in place.

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const STANDARDIZE: &str = "elemwise/standardize.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

pub fn standardize(ctx: &Ctx<'_>, bias: Tensor, scale: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.standardize";
    let entry = dtype_dispatch!(OP, out.dtype, { Bf16 => "elemwise_standardize_bf16" });
    let [width, rows, _] = elementwise_rows(OP, out.width, out.rows)?;
    if !width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {width}-wide bf16 row is odd; rows land two values a word"),
        ));
    }
    for (what, plane) in [("bias", bias), ("scale", scale)] {
        if plane.dtype != out.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {:?} and the rows it standardizes are {:?}; \
                     both planes ride the activation's element",
                    plane.dtype, out.dtype
                ),
            ));
        }
        if u64::from(plane.rows) * u64::from(plane.width) != u64::from(width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is a {} x {} plane, and this row reads one \
                     scalar per column of a {width}-wide rectangle",
                    plane.rows, plane.width
                ),
            ));
        }
    }
    ctx.fire(
        Fire::at(STANDARDIZE, entry).apply(Grid::of([width.div_ceil(2), rows, 1], [GROUP, 1, 1])),
        &[
            out.arg_mut(),
            bias.arg(),
            scale.arg(),
            stated(OP, width)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}
