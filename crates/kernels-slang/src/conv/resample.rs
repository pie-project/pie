//! Resampling over clip tables (`conv.upsample`, `conv.pixel_shuffle`,
//! `conv.pixel_unshuffle`, `conv.avg_down`), from
//! `kernels/conv/resample.slang`.

use dtype::Dtype;

use super::{clip_pair, even_row, row_grid};
use crate::encode::{Arg, ArgValue, Ctx, Fire, dtype_dispatch, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const RESAMPLE: &str = "conv/resample.slang";

/// Fire `entry` over `y`'s rows, a pair of its columns a lane; `scalars`
/// follow the four planes, the row count and span after them, then `bases`:
/// the rows `grid` and `o_grid` address `x`'s and `y`'s row 0 at, nonzero in
/// a window of a segmented fire whose grids hold fire-wide rows.
fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    [x, grid, o_grid, y]: [Tensor; 4],
    scalars: &[ArgValue],
    bases: [u32; 2],
) -> Result<(), Error> {
    if y.rows == 0 {
        return Ok(());
    }
    nonzero(op, "the row's width", y.width)?;
    if y.dtype == Dtype::Bf16 {
        even_row(op, "output row", y.width)?;
    }
    let (shape, span) = row_grid(y.width.div_ceil(2), y.rows, 256);
    let mut args = vec![x.arg(), grid.arg(), y.arg_mut(), o_grid.arg()];
    args.extend_from_slice(scalars);
    args.push(y.rows.arg());
    args.push(span.arg());
    args.push(bases[0].arg());
    args.push(bases[1].arg());
    ctx.fire(Fire::at(RESAMPLE, entry).apply(shape), &args)
}

/// Nearest upsampling by `factor` in `(t, h, w)`; with
/// `keep_first_frame`, the first frame is not repeated. `bases`: see
/// [`fire`].
#[allow(clippy::too_many_arguments)]
pub fn upsample_nearest(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    keep_first_frame: bool,
    o_grid: Tensor,
    y: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.upsample";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "conv_resample_upsample_bf16",
        F32 => "conv_resample_upsample_f32",
    });
    let clips = clip_pair(OP, grid, o_grid)?;
    if x.width != y.width || x.dtype != y.dtype {
        return Err(refuse(
            OP,
            format!(
                "an upsample does not reshape: {} {:?} into {} {:?}",
                x.width, x.dtype, y.width, y.dtype
            ),
        ));
    }
    fire(
        ctx,
        OP,
        entry,
        [x, grid, o_grid, y],
        &[
            y.width.arg(),
            clips.arg(),
            factor[0].arg(),
            factor[1].arg(),
            factor[2].arg(),
            u32::from(keep_first_frame).arg(),
        ],
        bases,
    )
}

/// Channel blocks of `r[0] r[1] r[2]` spread into time and space, the
/// first `trim_t` frames dropped. `bases`: see [`fire`].
#[allow(clippy::too_many_arguments)]
pub fn pixel_shuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    trim_t: u32,
    o_grid: Tensor,
    y: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.pixel_shuffle";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "conv_resample_shuffle_bf16",
        F32 => "conv_resample_shuffle_f32",
    });
    let clips = clip_pair(OP, grid, o_grid)?;
    let block = r[0] * r[1] * r[2];
    if block == 0 || x.width != y.width * block || x.dtype != y.dtype {
        return Err(refuse(
            OP,
            format!(
                "a shuffle by {r:?} takes {} channels into {} and was handed {} {:?} into {} {:?}",
                y.width * block,
                y.width,
                x.width,
                x.dtype,
                y.width,
                y.dtype
            ),
        ));
    }
    fire(
        ctx,
        OP,
        entry,
        [x, grid, o_grid, y],
        &[
            y.width.arg(),
            clips.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
            trim_t.arg(),
        ],
        bases,
    )
}

/// Time and space folded into channel blocks of `r[0] r[1] r[2]`; `bases`
/// as [`pixel_shuffle`]'s.
pub fn pixel_unshuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    o_grid: Tensor,
    y: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.pixel_unshuffle";
    let entry = dtype_dispatch!(OP, y.dtype, {
        Bf16 => "conv_resample_unshuffle_bf16",
        F32 => "conv_resample_unshuffle_f32",
    });
    let clips = clip_pair(OP, grid, o_grid)?;
    let block = r[0] * r[1] * r[2];
    if block == 0 || y.width != x.width * block || x.dtype != y.dtype {
        return Err(refuse(
            OP,
            format!(
                "an unshuffle by {r:?} takes {} channels into {} and was handed {} {:?} into {} {:?}",
                x.width,
                x.width * block,
                x.width,
                x.dtype,
                y.width,
                y.dtype
            ),
        ));
    }
    fire(
        ctx,
        OP,
        entry,
        [x, grid, o_grid, y],
        &[
            x.width.arg(),
            clips.arg(),
            r[0].arg(),
            r[1].arg(),
            r[2].arg(),
        ],
        bases,
    )
}

/// The unshuffle by `factor`, each output channel the mean of a run of
/// `group` unshuffled ones; time is front-padded to a multiple of
/// `factor[0]`. `bases`: see [`fire`].
#[allow(clippy::too_many_arguments)]
pub fn avg_down(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    group: u32,
    o_grid: Tensor,
    y: Tensor,
    bases: [u32; 2],
) -> Result<(), Error> {
    const OP: &str = "conv.avg_down";
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    let clips = clip_pair(OP, grid, o_grid)?;
    let block = factor[0] * factor[1] * factor[2];
    let group = nonzero(OP, "the averaged run", group)?;
    if block == 0 || !block.is_multiple_of(group) || y.width != x.width * block / group {
        return Err(refuse(
            OP,
            format!(
                "an avg-down by {factor:?} in runs of {group} takes {} channels into {} \
                 and was handed {} into {}",
                x.width,
                x.width * block / group,
                x.width,
                y.width
            ),
        ));
    }
    fire(
        ctx,
        OP,
        "conv_resample_avg_down_bf16",
        [x, grid, o_grid, y],
        &[
            x.width.arg(),
            clips.arg(),
            factor[0].arg(),
            factor[1].arg(),
            factor[2].arg(),
            group.arg(),
        ],
        bases,
    )
}
