use super::{DecodePlan, Paged};
use crate::encode::{Arg, Ctx, Fire, Grid, stated};
use crate::{Error, KvPool, Tensor};
use dtype::Dtype;
use std::sync::OnceLock;

pub fn enabled(pool: &KvPool, dim: u32) -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("PIE_METAL_KV_Q8").is_ok_and(|v| v == "1"))
        && matches!(dim, 64 | 128 | 256 | 512)
        && pool.keys.dtype == Dtype::Bf16
        && pool.values.dtype == Dtype::Bf16
        && pool.head_stride == u64::from(dim)
        && pool.keys.width == pool.values.width
        && u64::from(pool.keys.width) == pool.seq_stride
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &DecodePlan,
    window: Option<u32>,
    causal: bool,
    dim: u32,
    scale: f32,
    out: Tensor,
    lse: Option<Tensor>,
    allow_mpp: bool,
) -> Result<(), Error> {
    let shape = Paged::of(op, q, pool, window, causal, dim)?;
    let mpp = allow_mpp
        && causal
        && lse.is_none()
        && pool.page_size == 32
        && matches!(dim, 128 | 256)
        && q.rows >= 32;
    let paired = !mpp
        && dim == 256
        && shape.rows < 32
        && shape.gqa.is_multiple_of(2)
        && pool.page_size == 32
        && causal
        && window.is_none()
        && lse.is_none();
    let entry = if paired {
        "q8_vector_gqa_2"
    } else {
        match (mpp, lse.is_some(), dim) {
            (true, _, 128) => "q8_mpp_d128",
            (true, _, 256) if q.rows >= 512 => "q8_visit_direct",
            (true, _, 256) => "q8_mpp_d256",
            (_, false, 64) => "q8_decode_bfloat16_d_64",
            (_, false, 128) => "q8_decode_bfloat16_d_128",
            (_, false, 256) => "q8_decode_bfloat16_d_256",
            (_, false, 512) => "q8_decode_bfloat16_d_512",
            (_, true, 64) => "q8_decode_lse_bfloat16_d_64",
            (_, true, 128) => "q8_decode_lse_bfloat16_d_128",
            (_, true, 256) => "q8_decode_lse_bfloat16_d_256",
            (_, true, 512) => "q8_decode_lse_bfloat16_d_512",
            _ => unreachable!("enabled validates Q8 head widths"),
        }
    };
    let mut args = vec![
        q.arg(),
        pool.keys.arg(),
        pool.values.arg(),
        out.arg_mut(),
        stated(op, shape.gqa)?.arg(),
        plan.positions.arg(),
        plan.request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        scale.arg(),
        plan.mask.arg(),
        plan.mask_stride.arg(),
        plan.mask_enabled.arg(),
        shape.window.arg(),
        ctx.absent()?,
    ];
    if mpp {
        args.push(stated(op, q.rows)?.arg());
    }
    if let Some(lse) = lse {
        super::lse_plane(op, lse, &shape);
        args.push(lse.arg_mut());
    }
    let threads = if mpp { 256 } else { 1024 };
    let grid = if mpp {
        [shape.q_heads * threads, q.rows.div_ceil(32), 1]
    } else {
        super::vector_grid(op, shape.q_heads / if paired { 2 } else { 1 }, q.rows)?
    };
    ctx.fire(
        Fire::at(
            if mpp {
                "attn/q8_mpp.metal"
            } else {
                "attn/q8_decode.metal"
            },
            entry,
        )
        .apply(Grid::of(grid, [threads, 1, 1])),
        &args,
    )
}
