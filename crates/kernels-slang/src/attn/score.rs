//! Attention mass capture (`attn.score_capture`) from
//! `kernels/attn/score.slang`: the mean causal softmax of a request's last
//! query rows over its paged keys, per key, into an f32 row per lane.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const SCORE: &str = "attn/score.slang";

const STRIPS: u32 = 8;

const THREADS: u32 = STRIPS * 32;

const STAMPS: [u32; 3] = [64, 128, 256];

const CAPTURE: [&str; 3] = [
    "attn_score_capture_bf16_d_64",
    "attn_score_capture_bf16_d_128",
    "attn_score_capture_bf16_d_256",
];

fn stamp_for(head_dim: u32) -> Option<usize> {
    STAMPS.iter().position(|stamp| head_dim <= *stamp)
}

/// The kv heads the pool row's strides spell, which must be the stated ones.
fn kv_heads_agree(
    op: &'static str,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
) -> Result<(), Error> {
    nonzero(op, "the head width this attention states", head_dim)?;
    if pool.head_stride != u64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the stated head width {head_dim} is not the pool row's head stride {}",
                pool.head_stride
            ),
        ));
    }
    if pool.seq_stride == 0 || !pool.seq_stride.is_multiple_of(pool.head_stride) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not a whole number of {head_dim}-wide kv heads",
                pool.seq_stride
            ),
        ));
    }
    let spelled = pool.seq_stride / pool.head_stride;
    if u64::from(kv_heads) != spelled {
        return Err(refuse(
            op,
            format!(
                "the stated kv head count {kv_heads} is not the {spelled} the pool row's strides spell"
            ),
        ));
    }
    Ok(())
}

pub fn capture(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    positions: Tensor,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    observe: u32,
    lane_offset: u32,
    kv_max: u32,
    requests: u32,
    scores: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.score_capture";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    if pool.keys.dtype != Dtype::Bf16 {
        return Err(refuse(
            OP,
            format!(
                "the pool's key pages are stored as {:?}, and this capture reads keys \
                 straight out of the pages: it dequantizes nothing, so a quantized or \
                 fp8 pool has no scores to give",
                pool.keys.dtype
            ),
        ));
    }
    if scores.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the score rows are {:?}, and a per-key mass is an f32 rectangle",
                scores.dtype
            ),
        ));
    }

    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the fire's position table is {:?}, and this capture reads the causal \
                 bound as an i32 absolute position per token",
                positions.dtype
            ),
        ));
    }
    if q.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the window's boundary vector is {:?}, and this capture walks an i32 \
                 qo indptr",
                q.indptr.dtype
            ),
        ));
    }
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let Some(at) = stamp_for(head_dim) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_dim}-wide head is wider than the {}-wide dot this kernel is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = super::row_heads(OP, q.data.width, head_dim)?;
    if !num_q_heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {kv_heads} kv heads"),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    nonzero(OP, "the slab's per-row kv ceiling", kv_max)?;
    nonzero(OP, "the requests this capture answers", requests)?;
    let page_size = u32::try_from(pool.page_size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or_else(|| refuse(OP, "the kv page size is zero"))?;

    if scores.width != num_q_heads.saturating_mul(kv_max)
        || scores.rows < lane_offset.saturating_add(requests)
    {
        return Err(refuse(
            OP,
            format!(
                "the score rows are {} by {}, and {requests} lanes from lane {lane_offset} \
                 capture {num_q_heads} heads of {kv_max} keys",
                scores.rows, scores.width
            ),
        ));
    }

    let lanes = requests.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            OP,
            format!(
                "the grid will not launch: {requests} requests, one {THREADS}-thread group \
                 each"
            ),
        )
    })?;

    ctx.fire(
        Fire::at(SCORE, CAPTURE[at]).apply(Grid::of([lanes, num_q_heads, 1], [THREADS, 1, 1])),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            positions.arg(),
            scores.arg_mut(),
            stated(OP, page_size)?.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, kv_heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, lane_offset)?.arg(),
            stated(OP, kv_max)?.arg(),
        ],
    )
}
