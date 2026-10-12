use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::Tensor;

const PIXELS: &str = "layout/pixels.cuh";

const GRID_TAPS: &str = "layout/grid_taps.cuh";

const BLOCK: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn pixels(
    ctx: &Ctx,
    x: Tensor,
    patch: u32,
    mean: [f32; 3],
    std: [f32; 3],
    channel_major: bool,
    temporal: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.pixels";
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16" });
    if x.dtype != Dtype::U8 {
        return Err(refuse(
            OP,
            format!("pixel rows are u8 bytes, not {:?}", x.dtype),
        ));
    }
    let in_width = patch
        .checked_mul(patch)
        .and_then(|p| p.checked_mul(3))
        .ok_or_else(|| refuse(OP, format!("a {patch}-wide patch's bytes do not fit a u32")))?;
    if x.width != in_width || Some(y.width) != temporal.checked_mul(in_width) || y.rows < x.rows {
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
    if x.rows == 0 || y.width == 0 {
        return Ok(());
    }
    let order: i32 = if channel_major { 0 } else { 1 };
    ctx.fire(
        OP,
        Fire::at(PIXELS, symbol(&format!("::pie::layout::pixels<{t}>")))
            .apply(Launch::per_row(x.rows, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            stated(OP, patch)?.arg(),
            stated(OP, y.width)?.arg(),
            order.arg(),
            mean[0].arg(),
            mean[1].arg(),
            mean[2].arg(),
            std[0].arg(),
            std[1].arg(),
            std[2].arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn grid_taps(
    ctx: &Ctx,
    positions: Tensor,
    grids: Tensor,
    segments: Tensor,
    bilinear: bool,
    side: u32,
    ids: &mut Tensor,
    weights: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.grid_taps";
    let taps: u32 = if bilinear { 4 } else { 2 };
    if positions.dtype != Dtype::I32 || grids.dtype != Dtype::I32 || segments.dtype != Dtype::I32 {
        return Err(refuse(OP, "positions, grids and segments are i32"));
    }
    if ids.dtype != Dtype::I32 || weights.dtype != Dtype::F32 {
        return Err(refuse(OP, "the taps are i32 and their weights f32"));
    }
    if positions.width != 3 || grids.width != 3 {
        return Err(refuse(OP, "positions and grids are three wide"));
    }
    if ids.width != taps
        || weights.width != taps
        || ids.rows < positions.rows
        || weights.rows < positions.rows
    {
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
    if side == 0 {
        return Err(refuse(OP, "the position table's side is zero"));
    }
    let kind: i32 = if bilinear { 0 } else { 1 };
    ctx.fire(
        OP,
        Fire::at(GRID_TAPS, symbol("::pie::layout::grid_taps"))
            .apply(Launch::flat(positions.rows, BLOCK)),
        &[
            positions.arg(),
            grids.arg(),
            segments.arg(),
            ids.arg(),
            weights.arg(),
            kind.arg(),
            stated(OP, side)?.arg(),
            stated(OP, images)?.arg(),
            stated(OP, positions.rows)?.arg(),
        ],
    )
}
