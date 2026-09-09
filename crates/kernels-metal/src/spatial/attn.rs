use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "spatial/attn.metal";

const OP: &str = "spatial.attention";

const THREADS: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    Clip,
    Frames(u32),
}

fn stamp(channels: u32) -> Result<&'static str, Error> {
    match channels {
        1..=256 => Ok("256"),
        257..=512 => Ok("512"),
        513..=1024 => Ok("1024"),
        other => Err(refuse(
            OP,
            format!(
                "no point is stamped for a {other}-wide head; this plane holds 256, 512 \
                 and 1024, which is every VAE mid block under study"
            ),
        )),
    }
}

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
) -> Result<(), Error> {
    for t in [q, k, v, y] {
        if t.dtype != Dtype::Bf16 {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: t.dtype,
            });
        }
    }
    let clips = super::clips_of(OP, grid)?;
    let channels = q.width;
    if k.width != channels || v.width != channels || y.width != channels {
        return Err(refuse(
            OP,
            format!(
                "the query is {channels} wide and the key/value/answer are {}/{}/{}; a \
                 VAE's head is the whole channel row",
                k.width, v.width, y.width
            ),
        ));
    }
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
    let width = stamp(channels)?;
    let entry = match width {
        "256" => "spatial_attention_bfloat16_c_256",
        "512" => "spatial_attention_bfloat16_c_512",
        _ => "spatial_attention_bfloat16_c_1024",
    };
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([q.rows * THREADS, 1, 1], [THREADS, 1, 1])),
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
        ],
    )
}
