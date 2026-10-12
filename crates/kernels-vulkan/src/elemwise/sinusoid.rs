use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, elementwise_rows, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "elemwise/sinusoid.slang";

const GROUP: [u32; 3] = [256, 1, 1];

pub fn sinusoid(
    ctx: &Ctx<'_>,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.sinusoid";
    if t.dtype != Dtype::F32 || y.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the timesteps are {:?} and the embedding {:?}; both are f32",
                t.dtype, y.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    if dim < 2 || y.width != dim || t.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "a {dim}-wide embedding into a {}-wide plane from {} timesteps for {rows} rows",
                y.width, t.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at(FILE, "sinusoid_f32").apply(Grid::of(elementwise_rows(OP, dim / 2, rows)?, GROUP)),
        &[
            t.arg(),
            y.arg_mut(),
            dim.arg(),
            max_period.arg(),
            u32::from(flip_sin_cos).arg(),
            scale.arg(),
        ],
    )
}
