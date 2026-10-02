use crate::error::Error;
use dtype::Dtype;

use crate::elemwise::rope::ROTATE_BLOCK;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/rope_axes.cuh";

pub const MAX_AXES: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

impl RopeForm {
    const fn stamp(self) -> &'static str {
        match self {
            RopeForm::Interleaved => "0",
            RopeForm::Neox => "1",
            RopeForm::Split => "2",
            RopeForm::SplitLadder => "3",
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rope_axes(
    ctx: &Ctx,
    x: Tensor,
    positions: Tensor,
    dims: [u32; MAX_AXES],
    thetas: [f32; MAX_AXES],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_axes";
    dtype_dispatch!(OP, x.dtype, { Bf16 => (), F16 => () });
    debug_assert!(
        x.rows == o.rows && x.width == o.width && x.dtype == o.dtype,
        "`{OP}` writes the rectangle it reads"
    );
    let rows = nonzero(OP, "rows", o.rows)?;
    nonzero(OP, "the head width this rotation states", head_dim)?;
    if !head_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {head_dim}-wide head has no whole number of rotation pairs"),
        ));
    }
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the rotated prefix is {rotary_dim} wide, and the head it sits at the front \
                 of is {head_dim}"
            ),
        ));
    }
    if o.width == 0 || !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide row is not a whole number of {head_dim}-wide heads",
                o.width
            ),
        ));
    }
    let heads = o.width / head_dim;

    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads f32 coordinates \
                 (they may be fractional)",
                positions.dtype
            ),
        ));
    }
    let axes = positions.width as usize;
    if axes == 0 || axes > MAX_AXES || positions.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {} x {}, and this rotation reads one coordinate per \
                 axis (at most {MAX_AXES}) for each of {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    if form == RopeForm::SplitLadder {
        if rotary_dim != head_dim {
            return Err(refuse(
                OP,
                format!(
                    "the ladder pairs (i, i + head_dim/2) and turns the whole head, and this \
                     call rotates {rotary_dim} of {head_dim}"
                ),
            ));
        }
        let span: u32 = dims[..axes].iter().sum();
        let row = heads * rotary_dim;
        if span == 0 || span > row || !(row - span).is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!(
                    "the ladder's axes own {span} channels of a {row}-wide rotated row, which \
                     leaves no whole identity pad"
                ),
            ));
        }
        if dims[..axes].iter().any(|d| *d != dims[0]) {
            return Err(refuse(
                OP,
                format!("one ladder hands its axes out round-robin, and {dims:?} is not flat"),
            ));
        }
        return fire(
            ctx, x, positions, form, dims, thetas, rotary_dim, head_dim, heads, rows, o,
        );
    }
    let mut spanned = 0u32;
    for (a, &d) in dims.iter().enumerate() {
        if a >= axes {
            if d != 0 {
                return Err(refuse(
                    OP,
                    format!(
                        "axis {a} owns {d} channels and the position stream carries {axes} \
                         axes; an axis with no coordinate is a text to fix"
                    ),
                ));
            }
            continue;
        }
        if d == 0 || !d.is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!("axis {a} owns {d} channels, and an axis turns whole pairs"),
            ));
        }
        spanned += d;
    }
    if spanned != rotary_dim {
        return Err(refuse(
            OP,
            format!(
                "the axes own {spanned} channels between them and the rotated prefix is \
                 {rotary_dim} wide; the blocks tile it in axis order"
            ),
        ));
    }

    fire(
        ctx, x, positions, form, dims, thetas, rotary_dim, head_dim, heads, rows, o,
    )
}

#[allow(clippy::too_many_arguments)]
fn fire(
    ctx: &Ctx,
    x: Tensor,
    positions: Tensor,
    form: RopeForm,
    dims: [u32; MAX_AXES],
    thetas: [f32; MAX_AXES],
    rotary_dim: u32,
    head_dim: u32,
    heads: u32,
    rows: u32,
    o: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_axes";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::rope_axes<{t}, {}>",
                form.stamp()
            )),
        )
        .apply(Launch::per_row(rows, ROTATE_BLOCK)),
        &[
            x.arg(),
            positions.arg(),
            o.arg(),
            stated(OP, positions.width)?.arg(),
            stated(OP, dims[0])?.arg(),
            stated(OP, dims[1])?.arg(),
            stated(OP, dims[2])?.arg(),
            stated(OP, dims[3])?.arg(),
            thetas[0].arg(),
            thetas[1].arg(),
            thetas[2].arg(),
            thetas[3].arg(),
            stated(OP, rotary_dim)?.arg(),
            stated(OP, head_dim)?.arg(),
            stated(OP, heads)?.arg(),
            ctx.stage(),
        ],
    )
}
