use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};
use crate::tuning::Tuning;

use super::{DecodePlan, Paged, PrefillPlan, SDPA_TILE, kv_heads_agree, lse_plane, tiled, vector};

const MMA_HEAD_DIM: u32 = 64;

const MMA_THREADS: u32 = 128;

const MMA_ENTRY: &str = "sdpa_paged_mma_bfloat16_d_64";

const MMA_LSE_ENTRY: &str = "sdpa_paged_mma_lse_bfloat16_d_64";

/// The wide heads' simdgroup-matrix kernels, a 128-wide slice of the head a
/// simdgroup and four row groups: as many threads as the head is wide.
const MMA_WIDE: [(u32, &str, &str); 2] = [
    (
        256,
        "sdpa_paged_mma_wide_bfloat16_d_256",
        "sdpa_paged_mma_wide_lse_bfloat16_d_256",
    ),
    (
        512,
        "sdpa_paged_mma_wide_bfloat16_d_512",
        "sdpa_paged_mma_wide_lse_bfloat16_d_512",
    ),
];

const MMA_FILE: &str = "attn/sdpa_paged_mma.metal";

#[must_use]
pub fn should_tile(rows: u32, requests: u32, tuning: &Tuning) -> bool {
    rows / requests.max(1) >= tuning.sdpa_tile_rows_per_request()
}

#[must_use]
pub fn should_mma(head_dim: u32, tuning: &Tuning) -> bool {
    // The plain and the lse kernels are stamped at 64, 256 and 512.
    tuning.sdpa_mma() && (head_dim == MMA_HEAD_DIM || MMA_WIDE.iter().any(|w| w.0 == head_dim))
}

fn as_decode(plan: &PrefillPlan, mask: Tensor) -> DecodePlan {
    DecodePlan {
        positions: plan.positions,
        request_of_token: plan.request_of_token,
        mask,
        mask_enabled: plan.mask_enabled,
        mask_stride: plan.mask_stride,
    }
}

#[allow(clippy::too_many_arguments)]
fn mma(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
) -> Result<(), Error> {
    let shape = Paged::of(op, q, pool, window, true, head_dim)?;
    let (threads, plain, with_lse) = match MMA_WIDE.iter().find(|w| w.0 == head_dim) {
        Some(&(width, plain, with_lse)) => (width, plain, with_lse),
        None => (MMA_THREADS, MMA_ENTRY, MMA_LSE_ENTRY),
    };
    let lanes = shape.q_heads.checked_mul(threads).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the grid will not launch: {} query heads, one {threads}-thread group each",
                shape.q_heads
            ),
        )
    })?;
    let mut args = vec![
        q.arg(),
        pool.keys.arg(),
        pool.values.arg(),
        o.arg_mut(),
        stated(op, shape.gqa)?.arg(),
        plan.positions.arg(),
        plan.request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        pool.page_size.arg(),
        stated(op, shape.kv_heads)?.arg(),
        sm_scale.arg(),
        mask.arg(),
        plan.mask_stride.arg(),
        plan.mask_enabled.arg(),
        shape.window.arg(),
        ctx.absent()?,
        stated(op, shape.rows)?.arg(),
    ];
    let entry = match lse {
        None => plain,
        Some(lse) => {
            lse_plane(op, lse, &shape);
            args.push(lse.arg_mut());
            with_lse
        }
    };
    ctx.fire(
        Fire::at(MMA_FILE, entry).apply(Grid::of(
            [lanes, shape.rows.div_ceil(SDPA_TILE).max(1), 1],
            [threads, 1, 1],
        )),
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
fn arbitrate(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    requests: u32,
    tuning: &Tuning,
) -> Result<(), Error> {
    if super::q8::enabled(pool, head_dim) {
        return super::q8::attention(
            ctx,
            op,
            q,
            pool,
            &as_decode(plan, mask),
            window,
            causal,
            head_dim,
            sm_scale,
            o,
            lse,
            tuning.sdpa_mpp() && should_tile(q.rows, requests, tuning),
            tuning.sdpa_mpp_rows(),
        );
    }
    if !should_tile(q.rows, requests, tuning) {
        return vector(
            ctx,
            op,
            q,
            pool,
            &as_decode(plan, mask),
            window,
            causal,
            head_dim,
            sm_scale,
            o,
            lse,
            tuning,
        );
    }
    if tuning.sdpa_mpp()
        && super::q8::mpp_fits(
            q,
            pool,
            head_dim,
            causal,
            lse.is_some(),
            tuning.sdpa_mpp_rows(),
        )
    {
        return super::q8::attention(
            ctx,
            op,
            q,
            pool,
            &as_decode(plan, mask),
            window,
            causal,
            head_dim,
            sm_scale,
            o,
            lse,
            true,
            tuning.sdpa_mpp_rows(),
        );
    }
    if causal && should_mma(head_dim, tuning) {
        return mma(
            ctx, op, q, pool, plan, mask, window, head_dim, sm_scale, o, lse,
        );
    }
    tiled(
        ctx, op, q, pool, plan, mask, window, causal, head_dim, sm_scale, o, lse, tuning,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    tuning: &Tuning,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    arbitrate(
        ctx, OP, q.data, pool, plan, plan.mask, window, true, head_dim, sm_scale, o, None,
        requests, tuning,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
    requests: u32,
    tuning: &Tuning,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    arbitrate(
        ctx,
        OP,
        q.data,
        pool,
        plan,
        plan.mask,
        window,
        true,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
        tuning,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    tuning: &Tuning,
) -> Result<(), Error> {
    const OP: &str = "attention.masked";
    if mask.dtype != dtype::Dtype::U8 {
        return Err(refuse(
            OP,
            format!(
                "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                mask.dtype
            ),
        ));
    }
    arbitrate(
        ctx, OP, q.data, pool, plan, mask, window, causal, head_dim, sm_scale, o, None, requests,
        tuning,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn masked_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
    requests: u32,
    tuning: &Tuning,
) -> Result<(), Error> {
    const OP: &str = "attention.masked_lse";
    if mask.dtype != dtype::Dtype::U8 {
        return Err(refuse(
            OP,
            format!(
                "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                mask.dtype
            ),
        ));
    }
    arbitrate(
        ctx,
        OP,
        q.data,
        pool,
        plan,
        mask,
        window,
        causal,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
        tuning,
    )
}

/// Decode rows (a speculative verify or a drafter's block) through the split where
/// `kernel.sdpa.split_bf16` says, as Splash attends its verify and draft rows: one
/// threadgroup a KV head and history slice, each page read once for every grouped query
/// head of the tile's eight rows. Anything the split does not seat takes the vector decode.
#[allow(clippy::too_many_arguments)]
pub fn decode_with_scratch(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    partials: &dyn Fn(u32, u32) -> Option<Tensor>,
) -> Result<(), Error> {
    if tuning.sdpa_split_bf16()
        && q.rows >= 8
        && super::split::try_tables(
            ctx,
            q,
            pool,
            &super::split::Tables {
                positions: plan.positions,
                request_of_token: plan.request_of_token,
                mask: plan.mask,
                mask_enabled: plan.mask_enabled,
                mask_stride: plan.mask_stride,
            },
            window,
            true,
            head_dim,
            sm_scale,
            o,
            requests,
            partials,
            tuning.sdpa_split(),
            true,
        )?
    {
        return Ok(());
    }
    super::decode(ctx, tuning, q, plan, pool, window, head_dim, sm_scale, o)
}

#[allow(clippy::too_many_arguments)]
pub fn prefill_with_scratch(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    tuning: &Tuning,
    partials: &dyn Fn(u32, u32) -> Option<Tensor>,
) -> Result<(), Error> {
    if (tuning.sdpa_mpp() || tuning.sdpa_split_bf16())
        && super::split::try_paged(
            ctx,
            q.data,
            pool,
            plan,
            plan.mask,
            window,
            true,
            head_dim,
            sm_scale,
            o,
            requests,
            partials,
            tuning.sdpa_split(),
            tuning.sdpa_split_bf16(),
        )?
    {
        return Ok(());
    }

    prefill(
        ctx, q, plan, pool, window, head_dim, kv_heads, sm_scale, o, requests, tuning,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn masked_with_scratch(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    tuning: &Tuning,
    partials: &dyn Fn(u32, u32) -> Option<Tensor>,
) -> Result<(), Error> {
    if (tuning.sdpa_mpp() || tuning.sdpa_split_bf16())
        && super::split::try_paged(
            ctx,
            q.data,
            pool,
            plan,
            mask,
            window,
            causal,
            head_dim,
            sm_scale,
            o,
            requests,
            partials,
            tuning.sdpa_split(),
            tuning.sdpa_split_bf16(),
        )?
    {
        return Ok(());
    }

    masked(
        ctx, q, plan, mask, pool, window, causal, head_dim, sm_scale, o, requests, tuning,
    )
}

#[cfg(test)]
mod tests {

    #[test]
    fn the_fragment_map_gives_each_row_four_lanes_and_one_writer() {
        let map = |lid: u32| -> (u32, u32) {
            let qid = lid / 4;
            ((qid & 4) + ((lid / 2) % 4), (qid & 2) * 2 + (lid % 2) * 2)
        };
        let mut rows: [Vec<(u32, u32)>; 8] = Default::default();
        for lid in 0..32u32 {
            let (fm, col) = map(lid);
            assert!(fm < 8, "lane {lid} claims fragment row {fm}");
            rows[fm as usize].push((lid, col));
        }
        for (fm, lanes) in rows.iter().enumerate() {
            assert_eq!(lanes.len(), 4, "row {fm} is not four lanes wide");

            for &(lid, _) in lanes {
                assert!(lanes.iter().any(|&(o, _)| o == lid ^ 1), "row {fm}: xor 1");
                assert!(lanes.iter().any(|&(o, _)| o == lid ^ 8), "row {fm}: xor 8");
            }

            let writers = lanes.iter().filter(|&&(_, col)| col == 0).count();
            assert_eq!(writers, 1, "row {fm} publishes its lse {writers} times");
            let mut cols: Vec<u32> = lanes.iter().flat_map(|&(_, c)| [c, c + 1]).collect();
            cols.sort_unstable();
            assert_eq!(cols, (0..8).collect::<Vec<_>>(), "row {fm} is not covered");
        }
    }
}
