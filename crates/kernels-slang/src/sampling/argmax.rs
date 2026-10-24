//! `sampling.argmax` from `kernels/sampling/argmax.slang`: each row's first
//! largest value's index, one workgroup a row.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const ARGMAX: &str = "sampling/argmax.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// `y[r, column] = argmax(x[r])`, NaNs skipped.
pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "sampling.argmax";
    debug_assert_eq!(y.dtype, Dtype::I32, "`{OP}` writes i32 column indices");
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "sampling_argmax_bf16",
        F32 => "sampling_argmax_f32",
    });
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    debug_assert_eq!(x.rows, y.rows, "an argmax lands one entry per row");
    ctx.fire(
        Fire::at(ARGMAX, entry).apply(Grid::of([GROUP, rows, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, column)?.arg(),
        ],
    )
}
