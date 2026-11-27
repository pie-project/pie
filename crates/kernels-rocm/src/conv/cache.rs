use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{RecurrentPool, StateRows, Tensor};
use crate::voxels::{flat_elements, lanes_of};
use dtype::Dtype;

const FILE: &str = "conv/cache.hip";

const BLOCK: u32 = 256;

#[must_use]
pub fn cache_rows(grid: &[i32], frames: u32) -> u64 {
    grid.as_chunks::<4>()
        .0
        .iter()
        .map(|clip| u64::from(frames) * clip[1].max(0) as u64 * clip[2].max(0) as u64)
        .sum()
}

/// The lanes, frames, channels, a row's elements, the cache's elements and
/// blocks, and where the rows live.
type Checked = (i32, i32, i32, i64, i64, u32, StateRows);

/// The frame slab's rows (`pool.slab`, `[slots, frames * plane * C]`)
/// as every recurrent op reads them, through the pool's lane table
/// (`pool.slot_ids`, its rows window-local): clip `l` is lane
/// `clip_lanes[l] - lane_base` of it. They are cut along their
/// frame-position rows: `pool.segments` lays a lane's row
/// out in pieces of whole rows of `C`, `pool.slot_stride_elems` apart (0: a
/// row apart, the slab's own rows).
fn check(
    op: &'static str,
    pool: &RecurrentPool,
    clip_lanes: Tensor,
    grid: Tensor,
    cache: Tensor,
    frames: u32,
) -> Result<Checked, Error> {
    let slab = pool.slab;
    dtype_dispatch!(op, cache.dtype, { Bf16 => () });
    if slab.dtype != Dtype::Bf16 {
        return Err(refuse(op, "the slab is not bf16"));
    }
    let lanes = lanes_of(op, "input", grid)?;
    if clip_lanes.dtype != Dtype::I32 || clip_lanes.elements() < lanes as u64 {
        return Err(refuse(
            op,
            format!(
                "the clips' lane table is {}x{} {:?}; one i32 per clip is needed",
                clip_lanes.rows, clip_lanes.width, clip_lanes.dtype
            ),
        ));
    }
    let frames = count(op, "the cached frames", frames)?;
    let c = count(op, "the channel count", cache.width)?;
    let row_elems = i64::from(slab.width);
    let chunk = if pool.slot_stride_elems == 0 {
        row_elems
    } else {
        pool.slot_stride_elems
    };
    let rows = pool.segments.rows(
        op,
        slab,
        pool.slot_ids,
        chunk,
        slab.width.div_ceil(cache.width.max(1)),
    )?;
    let (blocks, total) = flat_elements(op, cache, BLOCK)?;
    Ok((lanes, frames, c, row_elems, total, blocks, rows))
}

pub fn cache_gather(
    ctx: &Ctx,
    pool: &RecurrentPool,
    clip_lanes: Tensor,
    lane_base: u32,
    grid: Tensor,
    frames: u32,
    cache: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.cache_gather";
    let (lanes, frames, c, stride, total, blocks, rows) =
        check(OP, pool, clip_lanes, grid, *cache, frames)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::conv::cache_gather<::pie::bf16>"))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            rows.arg(),
            grid.arg(),
            clip_lanes.arg(),
            stated(OP, lane_base)?.arg(),
            cache.arg(),
            lanes.arg(),
            frames.arg(),
            c.arg(),
            stride.arg(),
            total.arg(),
        ],
    )
}

/// `x_base`: the row `grid` addresses `x`'s row 0 at.
#[allow(clippy::too_many_arguments)]
pub fn cache_store(
    ctx: &Ctx,
    x: Tensor,
    cache: Tensor,
    pool: &RecurrentPool,
    clip_lanes: Tensor,
    lane_base: u32,
    grid: Tensor,
    frames: u32,
    x_base: u32,
) -> Result<(), Error> {
    const OP: &str = "conv.cache_store";
    let (lanes, frames, c, stride, total, blocks, rows) =
        check(OP, pool, clip_lanes, grid, cache, frames)?;
    if x.dtype != Dtype::Bf16 || x.width != cache.width {
        return Err(refuse(OP, "the input is not `[rows, C_in]` bf16"));
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::conv::cache_store<::pie::bf16>"))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            x.arg(),
            cache.arg(),
            rows.arg(),
            grid.arg(),
            clip_lanes.arg(),
            stated(OP, lane_base)?.arg(),
            lanes.arg(),
            frames.arg(),
            c.arg(),
            stride.arg(),
            stated(OP, x_base)?.arg(),
            total.arg(),
        ],
    )
}
