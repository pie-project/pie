use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "spatial/resample.metal";

fn grid_of(op: &'static str, columns: u32, rows: u32) -> Result<Grid, Error> {
    nonzero(op, "the row's width", columns)?;
    Ok(Grid::of([columns, rows, 1], [256.min(columns), 1, 1]))
}

pub fn upsample_nearest(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    keep_first_frame: bool,
    o_grid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.upsample_nearest";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "spatial_upsample_nearest_bfloat16",
        F32 => "spatial_upsample_nearest_float32",
    });
    let clips = super::clip_pair(OP, grid, o_grid)?;
    if x.width != y.width {
        return Err(refuse(
            OP,
            format!("an upsample does not reshape: {} into {}", x.width, y.width),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(grid_of(OP, y.width, y.rows)?),
        &[
            x.arg(),
            grid.arg(),
            y.arg_mut(),
            o_grid.arg(),
            y.width.arg(),
            clips.arg(),
            factor[0].arg(),
            factor[1].arg(),
            factor[2].arg(),
            u32::from(keep_first_frame).arg(),
        ],
    )
}

pub fn pixel_shuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    trim_t: u32,
    o_grid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.pixel_shuffle";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "spatial_pixel_shuffle_bfloat16",
        F32 => "spatial_pixel_shuffle_float32",
    });
    let clips = super::clip_pair(OP, grid, o_grid)?;
    let block = r[0] * r[1] * r[2];
    if block == 0 || x.width != y.width * block {
        return Err(refuse(
            OP,
            format!(
                "a shuffle by {r:?} takes {} channels into {} and was handed {} into {}",
                y.width * block,
                y.width,
                x.width,
                y.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(grid_of(OP, y.width, y.rows)?),
        &[
            x.arg(),
            grid.arg(),
            y.arg_mut(),
            o_grid.arg(),
            y.width.arg(),
            clips.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            trim_t.arg(),
        ],
    )
}

pub fn pixel_unshuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    o_grid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.pixel_unshuffle";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "spatial_pixel_unshuffle_bfloat16",
        F32 => "spatial_pixel_unshuffle_float32",
    });
    let clips = super::clip_pair(OP, grid, o_grid)?;
    let block = r[0] * r[1] * r[2];
    if block == 0 || y.width != x.width * block {
        return Err(refuse(
            OP,
            format!(
                "an unshuffle by {r:?} takes {} channels into {} and was handed {} into {}",
                x.width,
                x.width * block,
                x.width,
                y.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, entry).apply(grid_of(OP, y.width, y.rows)?),
        &[
            x.arg(),
            grid.arg(),
            y.arg_mut(),
            o_grid.arg(),
            x.width.arg(),
            clips.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
        ],
    )
}

pub fn avg_down(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    group: u32,
    o_grid: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.avg_down";
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    let clips = super::clip_pair(OP, grid, o_grid)?;
    let block = factor[0] * factor[1] * factor[2];
    let group = nonzero(OP, "the averaged run", group)?;
    if block == 0 || !block.is_multiple_of(group) || y.width != x.width * block / group {
        return Err(refuse(
            OP,
            format!(
                "an avg-down by {factor:?} in runs of {group} takes {} channels into {} \
                 and was handed {} into {}",
                x.width,
                x.width * block / group.max(1),
                x.width,
                y.width
            ),
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        Fire::at(FILE, "spatial_avg_down_bfloat16").apply(grid_of(OP, y.width, y.rows)?),
        &[
            x.arg(),
            grid.arg(),
            y.arg_mut(),
            o_grid.arg(),
            x.width.arg(),
            clips.arg(),
            factor[0].arg(),
            factor[1].arg(),
            factor[2].arg(),
            group.arg(),
        ],
    )
}

pub fn cache_gather(
    ctx: &Ctx<'_>,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    cache: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.cache_gather";
    for t in [slab, cache] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
    }
    let clips = super::clips_of(OP, grid)?;
    let frames = nonzero(OP, "cached frames", frames)?;
    let channels = nonzero(OP, "channels", cache.width)?;
    if slab.width != channels {
        return Err(refuse(
            OP,
            format!(
                "the gathered rectangle is {channels} wide and the slab {}",
                slab.width
            ),
        ));
    }
    if slot_ids.dtype != Dtype::I32
        || u64::from(slot_ids.rows) * u64::from(slot_ids.width) < u64::from(clips)
    {
        return Err(refuse(
            OP,
            format!(
                "the slot table is {} x {} {:?}, and this gather names one slot per clip \
                 of {clips}",
                slot_ids.rows, slot_ids.width, slot_ids.dtype
            ),
        ));
    }
    if cache.rows == 0 {
        return Ok(());
    }
    let stride = nonzero(OP, "the slot's rows", slab.rows)? * channels;
    ctx.fire(
        Fire::at(FILE, "spatial_cache_gather_bfloat16").apply(grid_of(OP, channels, cache.rows)?),
        &[
            slab.arg(),
            slot_ids.arg(),
            grid.arg(),
            cache.arg_mut(),
            clips.arg(),
            frames.arg(),
            channels.arg(),
            stride.arg(),
        ],
    )
}

pub fn cache_store(
    ctx: &Ctx<'_>,
    x: Tensor,
    cache: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    slab: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.cache_store";
    for t in [x, cache, slab] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
    }
    let clips = super::clips_of(OP, grid)?;
    let frames = nonzero(OP, "cached frames", frames)?;
    let channels = nonzero(OP, "channels", x.width)?;
    if cache.width != channels || slab.width != channels {
        return Err(refuse(
            OP,
            format!(
                "the input is {channels} wide and the cache/slab are {}/{}",
                cache.width, slab.width
            ),
        ));
    }
    if slot_ids.dtype != Dtype::I32
        || u64::from(slot_ids.rows) * u64::from(slot_ids.width) < u64::from(clips)
    {
        return Err(refuse(
            OP,
            format!(
                "the slot table is {} x {} {:?}, and this store names one slot per clip \
                 of {clips}",
                slot_ids.rows, slot_ids.width, slot_ids.dtype
            ),
        ));
    }
    if cache.rows == 0 {
        return Ok(());
    }
    let stride = nonzero(OP, "the slot's rows", slab.rows)? * channels;
    ctx.fire(
        Fire::at(FILE, "spatial_cache_store_bfloat16").apply(grid_of(OP, channels, cache.rows)?),
        &[
            x.arg(),
            cache.arg(),
            slot_ids.arg(),
            grid.arg(),
            slab.arg_mut(),
            clips.arg(),
            frames.arg(),
            channels.arg(),
            stride.arg(),
        ],
    )
}
