//! Positional encodings (`elemwise.sinusoid`) from
//! `kernels/elemwise/posenc.slang`: f32 sinusoidal timestep embeddings.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, elementwise_rows, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const POSENC: &str = "elemwise/posenc.slang";

/// The workgroup width the instantiation is built at.
const GROUP: u32 = 256;

/// `y[r] = [sin(t[r] f) | cos(t[r] f)]` over `dim / 2` frequencies.
pub fn sinusoid(
    ctx: &Ctx<'_>,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.sinusoid";
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
        Fire::at(POSENC, "elemwise_sinusoid_f32").apply(Grid::of(
            elementwise_rows(OP, dim / 2, rows)?,
            [GROUP, 1, 1],
        )),
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
