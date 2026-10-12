#![allow(clippy::too_many_arguments)]

use crate::encode::{Ctx, refuse};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};
use crate::tuning::Tuning;

use super::{DecodePlan, PrefillPlan, kv_heads_agree, tiled, vector};

#[must_use]
pub fn should_tile(rows: u32, requests: u32, tuning: &Tuning) -> bool {
    rows / requests.max(1) >= tuning.sdpa_tile_rows_per_request()
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

fn arbitrate(
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
    requests: u32,
    tuning: &Tuning,
) -> Result<(), Error> {
    if !should_tile(q.rows, requests, tuning) {
        return vector(
            ctx,
            tuning,
            op,
            q,
            pool,
            &as_decode(plan, mask),
            window,
            head_dim,
            sm_scale,
            o,
            lse,
        );
    }
    tiled(
        ctx, tuning, op, q, pool, plan, mask, window, head_dim, sm_scale, o, lse,
    )
}

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
        ctx, OP, q.data, pool, plan, plan.mask, window, head_dim, sm_scale, o, None, requests,
        tuning,
    )
}

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
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
        tuning,
    )
}

pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
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
        ctx, OP, q.data, pool, plan, mask, window, head_dim, sm_scale, o, None, requests, tuning,
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
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
        tuning,
    )
}
