//! The prefill entries engine dispatches call (kernels-wgpu `attn::arbiter`),
//! with Metal's `causal` flag on the masked forms. `requests` picks the walk:
//! at most two rows per request gather per row, more walk row blocks. The GPU
//! tuning argument is dropped.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::PrefillPlan;
use super::paged::{Paged, Tables, Walk, attend, kv_heads_agree};
use crate::cx::Ctx;
use crate::error::{Error, refuse};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

fn walk(rows: u32, requests: u32) -> Walk {
    if rows <= 2 * requests.max(1) {
        Walk::PerRow
    } else {
        Walk::Blocked
    }
}

fn call(
    op: &'static str,
    q: RaggedTensor,
    tables: Tables,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    requests: u32,
) -> Paged {
    Paged {
        op,
        q: q.data,
        tables,
        masked: true,
        window,
        causal,
        head_dim,
        sm_scale,
        o,
        lse,
        walk: walk(q.data.rows, requests),
        selection: None,
        rel: None,
    }
}

fn u8_mask(op: &'static str, mask: Tensor) -> Result<(), Error> {
    if !matches!(mask.dtype, Dtype::U8 | Dtype::Bool) {
        return Err(refuse(
            op,
            format!(
                "the mask this op states is {:?}; a mask plane is u8",
                mask.dtype
            ),
        ));
    }
    Ok(())
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
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let c = call(
        OP,
        q,
        Tables::prefill(plan),
        window,
        true,
        head_dim,
        sm_scale,
        o,
        None,
        requests,
    );
    attend(ctx, c, pool)
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
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let c = call(
        OP,
        q,
        Tables::prefill(plan),
        window,
        true,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
    );
    attend(ctx, c, pool)
}

/// A prefill under the custom `mask` (gated per row by the plan's enable
/// flags). `causal: false` reads every key the row's lane holds (Metal's
/// bidirectional arm); so does a causal row whose enable flag is 2.
/// Reference: kernels-metal `attn::arbiter::masked`.
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
) -> Result<(), Error> {
    const OP: &str = "attention.masked";
    u8_mask(OP, mask)?;
    let tables = Tables::prefill(plan).with_mask(mask);
    let c = call(
        OP, q, tables, window, causal, head_dim, sm_scale, o, None, requests,
    );
    attend(ctx, c, pool)
}

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
) -> Result<(), Error> {
    const OP: &str = "attention.masked_lse";
    u8_mask(OP, mask)?;
    let tables = Tables::prefill(plan).with_mask(mask);
    let c = call(
        OP,
        q,
        tables,
        window,
        causal,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
    );
    attend(ctx, c, pool)
}
