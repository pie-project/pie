//! Attention over a clip's voxels (`attn.voxel`), one head the whole
//! channel row, from `kernels/attn/voxel.slang`.

use dtype::Dtype;

use crate::conv::{clips_of, even_row, row_grid};
use crate::encode::{Arg, Ctx, Fire, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const VOXEL: &str = "attn/voxel.slang";

const OP: &str = "attn.voxel";

const THREADS: u32 = 128;

/// The keys a query row attends to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// Its whole clip.
    Clip,
    /// Its clip's run of this many frames.
    Frames(u32),
}

/// The instantiation whose accumulators hold a `channels`-wide head.
fn entry(channels: u32) -> Result<&'static str, Error> {
    match channels {
        1..=256 => Ok("attn_voxel_bf16_c_256"),
        257..=512 => Ok("attn_voxel_bf16_c_512"),
        513..=1024 => Ok("attn_voxel_bf16_c_1024"),
        other => Err(refuse(
            OP,
            format!(
                "no attention is instantiated for a {other}-wide head; this one holds 256, \
                 512 and 1024"
            ),
        )),
    }
}

/// `y` each row of `q` (bf16, the clips of `grid`) attended over the keys
/// and values of its segment. `base`: the row `grid` addresses the
/// rectangles' row 0 at, nonzero in a window of a segmented fire whose grid
/// holds fire-wide rows.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid: Tensor,
    segment: Segment,
    sm_scale: f32,
    y: Tensor,
    base: u32,
) -> Result<(), Error> {
    for t in [q, k, v, y] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
    }
    let clips = clips_of(OP, grid)?;
    let channels = q.width;
    if k.width != channels || v.width != channels || y.width != channels {
        return Err(refuse(
            OP,
            format!(
                "the query is {channels} wide and the key/value/answer are {}/{}/{}; a \
                 voxel head is the whole channel row",
                k.width, v.width, y.width
            ),
        ));
    }
    even_row(OP, "channel row", channels)?;
    if y.rows != q.rows {
        return Err(refuse(
            OP,
            format!("the query has {} rows and the answer {}", q.rows, y.rows),
        ));
    }
    let frames = match segment {
        Segment::Clip => 0i32,
        Segment::Frames(0) => {
            return Err(refuse(
                OP,
                "a segment of zero frames names no block; `Clip` is what one block per \
                 clip is spelled",
            ));
        }
        Segment::Frames(n) => i32::try_from(n).unwrap_or(i32::MAX),
    };
    if q.rows == 0 {
        return Ok(());
    }
    let entry = entry(channels)?;
    let (shape, span) = row_grid(THREADS, q.rows, THREADS);
    ctx.fire(
        Fire::at(VOXEL, entry).apply(shape),
        &[
            q.arg(),
            k.arg(),
            v.arg(),
            y.arg_mut(),
            grid.arg(),
            channels.arg(),
            clips.arg(),
            frames.arg(),
            sm_scale.arg(),
            q.rows.arg(),
            span.arg(),
            stated(OP, base)?.arg(),
        ],
    )
}
