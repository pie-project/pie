//! Image input (`gather.pixels`, `gather.grid_taps`) from
//! `kernels/gather/pixels.slang`: raw RGB patches to normalized bf16 rows,
//! and the position-table taps of each patch.

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const PIXELS: &str = "gather/pixels.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn pixels(
    ctx: &Ctx<'_>,
    x: Tensor,
    patch: u32,
    mean: [f32; 3],
    std: [f32; 3],
    channel_major: bool,
    temporal: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.pixels";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "gather_pixels_bf16" });
    if x.dtype != Dtype::U8 {
        return Err(refuse(
            OP,
            format!("pixel rows are u8 bytes, not {:?}", x.dtype),
        ));
    }
    let in_width = 3 * patch * patch;
    if x.width != in_width || y.width != temporal * in_width || y.rows < x.rows {
        return Err(refuse(
            OP,
            format!(
                "{} x {} pixel rows of {patch} x {patch} patches do not land as {} x {} rows \
                 repeated {temporal} times",
                x.rows, x.width, y.rows, y.width
            ),
        ));
    }
    if std.contains(&0.0) {
        return Err(refuse(OP, "a channel's std is zero"));
    }
    if y.width == 0 || !y.width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the pixel row is {} wide, and a bf16 row lands as whole words",
                y.width
            ),
        ));
    }
    let words = y.width / 2;
    let order: i32 = if channel_major { 0 } else { 1 };
    ctx.fire(
        Fire::at(PIXELS, entry).apply(Grid::of(
            elementwise_rows(OP, words, x.rows)?,
            [GROUP, 1, 1],
        )),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, patch)?.arg(),
            stated(OP, y.width)?.arg(),
            order.arg(),
            mean[0].arg(),
            mean[1].arg(),
            mean[2].arg(),
            std[0].arg(),
            std[1].arg(),
            std[2].arg(),
            stated(OP, x.rows)?.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn grid_taps(
    ctx: &Ctx<'_>,
    positions: Tensor,
    grids: Tensor,
    segments: Tensor,
    bilinear: bool,
    side: u32,
    ids: Tensor,
    weights: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.grid_taps";
    let taps: u32 = if bilinear { 4 } else { 2 };
    if positions.dtype != Dtype::I32 || grids.dtype != Dtype::I32 || segments.dtype != Dtype::I32 {
        return Err(refuse(OP, "positions, grids and segments are i32"));
    }
    if positions.width != 3 || grids.width != 3 {
        return Err(refuse(OP, "positions and grids are three wide"));
    }
    if ids.width != taps || weights.width != taps || ids.rows < positions.rows {
        return Err(refuse(
            OP,
            format!(
                "{taps} taps a row, and the destinations are {} x {} and {} x {}",
                ids.rows, ids.width, weights.rows, weights.width
            ),
        ));
    }
    let images = segments.rows.saturating_sub(1);
    if images == 0 || positions.rows == 0 {
        return Ok(());
    }
    let kind: i32 = if bilinear { 0 } else { 1 };
    ctx.fire(
        Fire::at(PIXELS, "gather_grid_taps").apply(Grid::of([positions.rows, 1, 1], [GROUP, 1, 1])),
        &[
            positions.arg(),
            grids.arg(),
            segments.arg(),
            ids.arg_mut(),
            weights.arg_mut(),
            kind.arg(),
            stated(OP, side)?.arg(),
            stated(OP, images)?.arg(),
            stated(OP, positions.rows)?.arg(),
        ],
    )
}
