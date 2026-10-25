//! Group norm over clip tables (`conv.group_norm`), from
//! `kernels/conv/group_norm.slang`: moments a split of a clip's rows, the
//! splits folded into each clip's group statistics, then the normalized
//! rows.

use dtype::Dtype;

use super::{clips_of, even_row, row_grid};
use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const GROUP_NORM: &str = "conv/group_norm.slang";

const OP: &str = "conv.group_norm";

const STATS_THREADS: u32 = 256;

/// The splits a clip's rows are cut into for their moments.
pub const SPLITS: u32 = 32;

/// The f32s the scratch planes hold: `partials` (a float4 a split, clip and
/// group) and then `stats` (a float2 a clip and group).
#[must_use]
pub fn scratch_floats(clips: u32, groups: u32) -> u64 {
    u64::from(clips) * u64::from(SPLITS) * u64::from(groups) * 4
        + u64::from(clips) * u64::from(groups) * 2
}

/// `y` the group norm of `x` (both bf16 `[rows, channels]`) over each clip
/// of `grid`, by `groups` channel groups, then `weight` and `bias` (f32 a
/// channel) and, with `silu`, a SiLU. `partials` and `stats` are the
/// scratch planes `scratch_floats` sizes. `base`: `x`'s row 0 in `grid`'s fire-wide rows.
#[allow(clippy::too_many_arguments)]
pub fn group_norm(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    groups: u32,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    silu: bool,
    partials: Tensor,
    stats: Tensor,
    y: Tensor,
    base: u32,
) -> Result<(), Error> {
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    let clips = clips_of(OP, grid)?;
    let channels = nonzero(OP, "channels", x.width)?;
    even_row(OP, "channel row", channels)?;
    let groups = nonzero(OP, "groups", groups)?;
    if !channels.is_multiple_of(groups) {
        return Err(refuse(
            OP,
            format!("{channels} channels do not divide into {groups} group(s)"),
        ));
    }
    for (what, t) in [("weight", weight), ("bias", bias)] {
        if t.dtype != Dtype::F32 || u64::from(t.rows) * u64::from(t.width) < u64::from(channels) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {} x {} {:?}, and this norm reads one f32 per \
                     channel of {channels}",
                    t.rows, t.width, t.dtype
                ),
            ));
        }
    }
    if x.rows != y.rows || x.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the input is {} x {} and the answer {} x {}",
                x.rows, x.width, y.rows, y.width
            ),
        ));
    }
    if x.rows == 0 {
        return Ok(());
    }

    ctx.fire(
        Fire::at(GROUP_NORM, "conv_group_norm_stats_bf16").apply(Grid::of(
            [SPLITS * STATS_THREADS, clips, groups],
            [STATS_THREADS, 1, 1],
        )),
        &[
            x.arg(),
            grid.arg(),
            partials.arg_mut(),
            channels.arg(),
            groups.arg(),
            SPLITS.arg(),
            base.arg(),
        ],
    )?;
    ctx.fire(
        Fire::at(GROUP_NORM, "conv_group_norm_finalize")
            .apply(Grid::of([groups * 32, clips, 1], [32, 1, 1])),
        &[
            partials.arg(),
            stats.arg_mut(),
            groups.arg(),
            SPLITS.arg(),
            eps.arg(),
        ],
    )?;
    let (apply, span) = row_grid(channels / 2, x.rows, 256);
    ctx.fire(
        Fire::at(GROUP_NORM, "conv_group_norm_apply_bf16").apply(apply),
        &[
            x.arg(),
            grid.arg(),
            stats.arg(),
            weight.arg(),
            bias.arg(),
            y.arg_mut(),
            channels.arg(),
            groups.arg(),
            clips.arg(),
            u32::from(silu).arg(),
            x.rows.arg(),
            span.arg(),
            base.arg(),
        ],
    )
}
