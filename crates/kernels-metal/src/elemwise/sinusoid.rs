use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, elementwise_rows, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

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
                "the timestep plane is {:?} and the embedding {:?}; both are f32 — a \
                 schedule's sigma is a fraction, not a token index",
                t.dtype, y.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    if dim < 2 {
        return Err(refuse(
            OP,
            format!("a {dim}-wide embedding carries no frequency"),
        ));
    }
    if y.width != dim {
        return Err(refuse(
            OP,
            format!(
                "the embedding plane is {} wide and this call writes {dim}",
                y.width
            ),
        ));
    }
    if t.rows < rows {
        return Err(refuse(
            OP,
            format!("{} timesteps for {rows} embedded rows", t.rows),
        ));
    }
    ctx.fire(
        Fire::at("elemwise/sinusoid.metal", "sinusoid")
            .apply(Grid::of(elementwise_rows(OP, dim / 2, rows)?, [256, 1, 1])),
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
