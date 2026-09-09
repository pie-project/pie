use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "spatial/norm.metal";

const OP: &str = "spatial.group_norm";

const STATS_THREADS: u32 = 256;

pub const SPLITS: u32 = 32;

#[must_use]
pub fn scratch_floats(clips: u32, groups: u32) -> u64 {
    u64::from(clips) * u64::from(SPLITS) * u64::from(groups) * 4
        + u64::from(clips) * u64::from(groups) * 2
}

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
) -> Result<(), Error> {
    if x.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    let clips = super::clips_of(OP, grid)?;
    let channels = nonzero(OP, "channels", x.width)?;
    let groups = nonzero(OP, "groups", groups)?;
    if channels % groups != 0 {
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
        Fire::at(FILE, "spatial_group_norm_stats_bfloat16").apply(Grid::of(
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
        ],
    )?;
    ctx.fire(
        Fire::at(FILE, "spatial_group_norm_finalize")
            .apply(Grid::of([groups * 32, clips, 1], [32, 1, 1])),
        &[
            partials.arg(),
            stats.arg_mut(),
            groups.arg(),
            SPLITS.arg(),
            eps.arg(),
        ],
    )?;
    ctx.fire(
        Fire::at(FILE, "spatial_group_norm_apply_bfloat16")
            .apply(Grid::of([channels, x.rows, 1], [256.min(channels), 1, 1])),
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
        ],
    )
}
