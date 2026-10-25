use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::Tensor;
use crate::voxels::{flat_elements, lanes_of};
use dtype::Dtype;

const FILE: &str = "conv/cache.cu";

const BLOCK: u32 = 256;

#[must_use]
pub fn cache_rows(grid: &[i32], frames: u32) -> u64 {
    grid.as_chunks::<4>()
        .0
        .iter()
        .map(|clip| u64::from(frames) * clip[1].max(0) as u64 * clip[2].max(0) as u64)
        .sum()
}

fn check(
    op: &'static str,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    cache: Tensor,
    frames: u32,
) -> Result<(i32, i32, i32, i64, i64, u32), Error> {
    dtype_dispatch!(op, cache.dtype, { Bf16 => () });
    if slab.dtype != Dtype::Bf16 {
        return Err(refuse(op, "the slab is not bf16"));
    }
    let lanes = lanes_of(op, "input", grid)?;
    if slot_ids.dtype != Dtype::I32 || slot_ids.elements() < lanes as u64 {
        return Err(refuse(
            op,
            format!(
                "the slot table is {}x{} {:?}; one i32 per lane is needed",
                slot_ids.rows, slot_ids.width, slot_ids.dtype
            ),
        ));
    }
    let frames = count(op, "the cached frames", frames)?;
    let c = count(op, "the channel count", cache.width)?;
    let stride = i64::from(slab.width);
    let (blocks, total) = flat_elements(op, cache, BLOCK)?;
    Ok((lanes, frames, c, stride, total, blocks))
}

pub fn cache_gather(
    ctx: &Ctx,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    cache: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.cache_gather";
    let (lanes, frames, c, stride, total, blocks) =
        check(OP, slab, slot_ids, grid, *cache, frames)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::conv::cache_gather<::pie::bf16>"))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            slab.arg(),
            slot_ids.arg(),
            grid.arg(),
            cache.arg(),
            lanes.arg(),
            frames.arg(),
            c.arg(),
            stride.arg(),
            total.arg(),
        ],
    )
}

/// `x_base`: the row `grid` addresses `x`'s row 0 at, nonzero in a window of
/// a segmented fire whose grid holds fire-wide rows.
#[allow(clippy::too_many_arguments)]
pub fn cache_store(
    ctx: &Ctx,
    x: Tensor,
    cache: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    slab: &mut Tensor,
    x_base: u32,
) -> Result<(), Error> {
    const OP: &str = "conv.cache_store";
    let (lanes, frames, c, stride, total, blocks) =
        check(OP, *slab, slot_ids, grid, cache, frames)?;
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
            slot_ids.arg(),
            grid.arg(),
            slab.arg(),
            lanes.arg(),
            frames.arg(),
            c.arg(),
            stride.arg(),
            stated(OP, x_base)?.arg(),
            total.arg(),
        ],
    )
}
