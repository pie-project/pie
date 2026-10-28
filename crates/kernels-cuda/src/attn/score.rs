use dtype::Dtype;

use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const SCORE: &str = "attn/score.cu";

const WARPS: u32 = 8;

const BLOCK: u32 = WARPS * 32;

const STAMPS: [u32; 4] = [64, 128, 256, 512];

fn row_heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

fn stamp_for(head_dim: u32) -> Option<u32> {
    STAMPS.into_iter().find(|stamp| head_dim <= *stamp)
}

#[allow(clippy::too_many_arguments)]
pub fn capture(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    observe: u32,
    lane_offset: u32,
    kv_max: u32,
    scores: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.score_capture";
    plan.accepts(OP, head_dim, Some(kv_heads), None)?;
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    if !kv::native_bf16(pool) {
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
    let head_dim = count(OP, "the head width this capture states", head_dim)?;
    let head_width = head_dim.unsigned_abs();
    let Some(stamp) = stamp_for(head_width) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_width}-wide head is wider than the {}-wide dot this kernel is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = row_heads(OP, "query", q.data.width, head_width)?;
    let num_kv_heads = count(OP, "the kv head count this capture states", kv_heads)?.unsigned_abs();
    if num_q_heads % num_kv_heads != 0 {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    let kv_ceiling = count(OP, "the slab's per-row kv ceiling", kv_max)?;
    let page_size = count(OP, "the pool's page size", pool.page_size.unsigned_abs())?;

    let requests = kv::lanes_of(OP, q.indptr)?.unsigned_abs();
    debug_assert_eq!(
        num_q_heads, plan.shape.num_q_heads,
        "`{OP}` captures one row per query head of the plan it was carved with"
    );
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

    let floats = head_width.saturating_add(2 * WARPS);
    let smem = floats
        .checked_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("a {head_width}-wide head over {WARPS} warps overflows its shared plane"),
            )
        })?;

    let hnd = if pool.layout == 0 {
        "::pie::false_type::value"
    } else {
        "::pie::true_type::value"
    };

    ctx.fire(
        OP,
        Fire::at(kernel(
            SCORE,
            &format!("::pie::attn::score_capture<{stamp}, {WARPS}, {hnd}>"),
        ))
        .apply(Launch::grid([requests, num_q_heads, 1], [BLOCK, 1, 1]).smem(smem)),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            scores.arg(),
            page_size.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, lane_offset)?.arg(),
            kv_ceiling.arg(),
            ctx.stage(),
        ],
    )
}
