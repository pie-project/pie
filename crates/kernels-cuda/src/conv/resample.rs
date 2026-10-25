use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::Tensor;
use crate::voxels::{flat_elements, lane_pair};

const FILE: &str = "conv/resample.cu";

const BLOCK: u32 = 256;

fn element(op: &'static str, dtype: dtype::Dtype) -> Result<&'static str, Error> {
    Ok(dtype_dispatch!(op, dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" }))
}

/// `bases` (here and in the other resamplers): the rows `grid` and `o_grid`
/// address `x`'s and `o`'s row 0 at (`[x, o]`), nonzero in a window of a
/// segmented fire whose grids hold fire-wide rows.
#[allow(clippy::too_many_arguments)]
pub fn upsample_nearest(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    keep_first_frame: bool,
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.upsample";
    let t = element(OP, x.dtype)?;
    debug_assert!(
        o.width == x.width && o.dtype == x.dtype,
        "`{OP}` keeps the channel width"
    );
    let lanes = lane_pair(OP, grid, o_grid)?;
    let c = count(OP, "the channel count", x.width)?;
    let [ft, fh, fw] = [
        count(OP, "the time factor", factor[0])?,
        count(OP, "the height factor", factor[1])?,
        count(OP, "the width factor", factor[2])?,
    ];
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::conv::upsample_nearest<{t}>")))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            ft.arg(),
            fh.arg(),
            fw.arg(),
            i32::from(keep_first_frame).arg(),
            stated(OP, bases[0])?.arg(),
            stated(OP, bases[1])?.arg(),
            total.arg(),
        ],
    )
}

fn block_of(op: &'static str, r: [u32; 3]) -> Result<([i32; 3], u32), Error> {
    let r = [
        count(op, "r1", r[0])?,
        count(op, "r2", r[1])?,
        count(op, "r3", r[2])?,
    ];
    let volume = r[0]
        .unsigned_abs()
        .checked_mul(r[1].unsigned_abs())
        .and_then(|v| v.checked_mul(r[2].unsigned_abs()))
        .ok_or_else(|| refuse(op, "the block volume overflows"))?;
    Ok((r, volume))
}

#[allow(clippy::too_many_arguments)]
pub fn pixel_shuffle(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    trim_t: u32,
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.pixel_shuffle";
    let t = element(OP, x.dtype)?;
    debug_assert_eq!(o.dtype, x.dtype, "`{OP}` keeps the element");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let (r, volume) = block_of(OP, r)?;
    if !x.width.is_multiple_of(volume) || o.width != x.width / volume {
        return Err(refuse(
            OP,
            format!(
                "{} channels do not unpack as {} x {volume} output channels",
                x.width, o.width
            ),
        ));
    }
    let c = count(OP, "the output channel count", o.width)?;
    let trim = stated(OP, trim_t)?;
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::conv::pixel_shuffle<{t}>")))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            trim.arg(),
            stated(OP, bases[0])?.arg(),
            stated(OP, bases[1])?.arg(),
            total.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn pixel_unshuffle(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.pixel_unshuffle";
    let t = element(OP, x.dtype)?;
    debug_assert_eq!(o.dtype, x.dtype, "`{OP}` keeps the element");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let (r, volume) = block_of(OP, r)?;
    if o.width != x.width.saturating_mul(volume) {
        return Err(refuse(
            OP,
            format!(
                "{} channels do not pack as {} = C x {volume} output channels",
                x.width, o.width
            ),
        ));
    }
    let c = count(OP, "the input channel count", x.width)?;
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::conv::pixel_unshuffle<{t}>")))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            stated(OP, bases[0])?.arg(),
            stated(OP, bases[1])?.arg(),
            total.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn avg_down(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    group: u32,
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.avg_down";
    let t = element(OP, x.dtype)?;
    debug_assert_eq!(o.dtype, x.dtype, "`{OP}` keeps the element");
    let lanes = lane_pair(OP, grid, o_grid)?;
    let (r, volume) = block_of(OP, r)?;
    let widened = x.width.saturating_mul(volume);
    if group == 0 || !widened.is_multiple_of(group) || o.width != widened / group {
        return Err(refuse(
            OP,
            format!(
                "{} channels widen to {widened} and do not fold into {} groups of {group}",
                x.width, o.width
            ),
        ));
    }
    let c = count(OP, "the input channel count", x.width)?;
    let g = count(OP, "the group size", group)?;
    let (blocks, total) = flat_elements(OP, *o, BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::conv::avg_down<{t}>")))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            o.arg(),
            o_grid.arg(),
            c.arg(),
            lanes.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            g.arg(),
            stated(OP, bases[0])?.arg(),
            stated(OP, bases[1])?.arg(),
            total.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn patchify(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    pixel_unshuffle(ctx, x, grid, p, o, o_grid, bases)
}

#[allow(clippy::too_many_arguments)]
pub fn unpatchify(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    o: &mut Tensor,
    o_grid: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    pixel_shuffle(ctx, x, grid, p, 0, o, o_grid, bases)
}
