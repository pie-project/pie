//! `elemwise.rope_axes` from `kernels/elemwise/rope_axes.slang`: rotary over
//! up to four position axes, in place, one rotation pair a lane.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const ROPE_AXES: &str = "elemwise/rope_axes.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// How an axis's channels pair up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    /// Adjacent channels.
    Interleaved,
    /// Halves of the rotated prefix.
    Neox,
    /// Halves of the axis's own block.
    Split,
    /// Axes cycling over every head's rotated channels after a centred pad.
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

/// `x` turned in place by `positions` (f32, a row of axes a token), each
/// axis `dims[a]` channels at `thetas[a]`; a zero width ends the axes.
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
) -> Result<(), Error> {
    const OP: &str = "elemwise.rope_axes";
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "elemwise_rope_axes_bf16",
        F32 => "elemwise_rope_axes_f32",
    });
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("{:?} positions; a rotation reads f32", positions.dtype),
        ));
    }
    let rows = nonzero(OP, "rows", x.rows)?;
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !x.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!("a {}-wide row of {head_dim}-wide heads", x.width),
        ));
    }
    let heads = x.width / head_dim;
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
    let lanes = heads * (rotary_dim / 2);
    ctx.fire(
        Fire::at(ROPE_AXES, entry).apply(Grid::of([lanes, rows, 1], [GROUP, 1, 1])),
        &[
            x.arg_mut(),
            positions.arg(),
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
        ],
    )
}
