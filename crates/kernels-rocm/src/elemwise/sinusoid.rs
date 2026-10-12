use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, nonzero, refuse, stated};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/sinusoid.cuh";

const BLOCK: u32 = 256;

pub fn sinusoid(
    ctx: &Ctx,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.sinusoid";
    let rows = nonzero(OP, "rows", o.rows)?;
    nonzero(OP, "the embedding width", dim)?;
    for (what, plane) in [("timestep", t), ("embedding", *o)] {
        if plane.dtype != Dtype::F32 {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {:?}, and this embedding is computed and \
                     published in f32",
                    plane.dtype
                ),
            ));
        }
    }
    if t.width != 1 || t.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the timestep stream is {} x {}, and this embedding reads one scalar for \
                 each of {rows} rows",
                t.rows, t.width
            ),
        ));
    }
    if o.width != dim {
        return Err(refuse(
            OP,
            format!(
                "the destination is {} wide and the embedding is {dim}",
                o.width
            ),
        ));
    }
    if max_period <= 0.0 || !max_period.is_finite() {
        return Err(refuse(
            OP,
            format!("the period is {max_period}, and its logarithm sets the ladder"),
        ));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::sinusoid").apply(Launch::per_row(rows, BLOCK)),
        &[
            t.arg(),
            o.arg(),
            stated(OP, dim)?.arg(),
            max_period.arg(),
            i32::from(flip_sin_cos).arg(),
            scale.arg(),
            ctx.stage(),
        ],
    )
}
