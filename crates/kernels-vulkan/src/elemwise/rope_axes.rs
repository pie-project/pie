use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "rope/axes.slang";

const GROUP: [u32; 3] = [256, 1, 1];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

impl RopeForm {
    const fn word(self) -> i32 {
        match self {
            Self::Interleaved => 0,
            Self::Neox => 1,
            Self::Split => 2,
            Self::SplitLadder => 3,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rope_axes(
    ctx: &Ctx<'_>,
    x: Tensor,
    positions: Tensor,
    dims: [u32; 4],
    thetas: [f32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_axes";
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("{:?} positions; a rotation reads f32", positions.dtype),
        ));
    }
    let rows = nonzero(OP, "rows", o.rows)?;
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("a {}-wide row of {head_dim}-wide heads", o.width),
        ));
    }
    let heads = o.width / head_dim;
    let axes = u32::try_from(dims.iter().take_while(|d| **d != 0).count()).unwrap_or(0);
    if axes == 0 || dims[..axes as usize].iter().any(|d| d % 2 != 0) {
        return Err(refuse(OP, format!("axis widths {dims:?}")));
    }
    let span: u32 = dims[..axes as usize].iter().sum();
    let fits = if form == RopeForm::SplitLadder {
        rotary_dim != 0
            && rotary_dim <= head_dim
            && span <= heads * rotary_dim
            && (heads * rotary_dim - span).is_multiple_of(2)
    } else {
        span == rotary_dim && rotary_dim != 0 && rotary_dim <= head_dim
    };
    if !fits || positions.rows < rows || positions.width < axes {
        return Err(refuse(
            OP,
            format!(
                "axes {dims:?} over {rotary_dim} of {head_dim} channels, {} x {} positions \
                 for {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    let angles = rotary_dim / 2;
    let tail = if x.buf == o.buf {
        0
    } else {
        head_dim - rotary_dim
    };
    let lanes = heads * (angles + tail);
    ctx.fire(
        Fire::at(
            FILE,
            dtype_dispatch!(OP, o.dtype, { Bf16 => "rope_axes_bf16", F32 => "rope_axes_f32" }),
        )
        .apply(Grid::of([lanes, rows, 1], GROUP)),
        &[
            x.arg(),
            positions.arg(),
            o.arg_mut(),
            dims[0].arg(),
            dims[1].arg(),
            dims[2].arg(),
            dims[3].arg(),
            thetas[0].arg(),
            thetas[1].arg(),
            thetas[2].arg(),
            thetas[3].arg(),
            axes.arg(),
            rotary_dim.arg(),
            head_dim.arg(),
            heads.arg(),
            form.word().arg(),
            u32::from(tail != 0).arg(),
        ],
    )
}
