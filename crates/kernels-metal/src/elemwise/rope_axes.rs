use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

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
    let entry = dtype_dispatch!(OP, o.dtype, {
        Bf16 => "rope_axes_bfloat16",
        F32 => "rope_axes_float32",
    });
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads f32 coordinates — \
                 LTX's are seconds and pixels, not token indices",
                positions.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows", o.rows)?;
    let head_dim = nonzero(OP, "the head width", head_dim)?;
    if !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "a {}-wide row is not a whole number of {head_dim}-wide heads",
                o.width
            ),
        ));
    }
    let heads = o.width / head_dim;
    let axes = dims.iter().take_while(|d| **d != 0).count();
    let axes = u32::try_from(axes).unwrap_or(0);
    if axes == 0 {
        return Err(refuse(OP, "no axis carries a channel"));
    }
    if let Some(odd) = dims[..axes as usize].iter().find(|d| **d % 2 != 0) {
        return Err(refuse(
            OP,
            format!("axis width {odd} is odd, and an angle turns a pair"),
        ));
    }
    let span: u32 = dims[..axes as usize].iter().sum();
    if form == RopeForm::SplitLadder {
        if rotary_dim == 0 || rotary_dim > head_dim {
            return Err(refuse(
                OP,
                format!("the ladder turns a {rotary_dim}-wide prefix of a {head_dim}-wide head"),
            ));
        }
        let row = heads.saturating_mul(rotary_dim);
        if span == 0 || span > row || !(row - span).is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!(
                    "the ladder's axes own {span} channels of a {row}-wide rotated row, which \
                     leaves no whole identity pad"
                ),
            ));
        }
        if dims[..axes as usize].iter().any(|d| *d != dims[0]) {
            return Err(refuse(
                OP,
                format!("one ladder hands its axes out round-robin, and {dims:?} is not flat"),
            ));
        }
    } else if span != rotary_dim || rotary_dim > head_dim || rotary_dim == 0 {
        return Err(refuse(
            OP,
            format!(
                "the axes span {span} channels, the rotation states {rotary_dim}, and the \
                 head is {head_dim} wide"
            ),
        ));
    }
    if positions.rows < rows || positions.width < axes {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {} x {} and this rotation reads {axes} coordinates \
                 for each of {rows} rows",
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
    let lanes = heads.checked_mul(angles + tail).ok_or_else(|| {
        refuse(
            OP,
            format!("{heads} heads x {angles} angles will not launch"),
        )
    })?;
    ctx.fire(
        Fire::at("elemwise/rope_axes.metal", entry).apply(Grid::of([lanes, rows, 1], [256, 1, 1])),
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
