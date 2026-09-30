//! Paged attention with a per-row relative-position bias (T5-style buckets
//! gathered upstream) and optional log-length logit scaling.
//! Reference: kernels-cuda `attn::{decode_rel, prefill_rel}` and its
//! `VariantRelBias`.

#![allow(clippy::too_many_arguments)]

use super::paged::{Paged, Tables, Walk, attend, kv_heads_agree};
use super::{DecodePlan, PrefillPlan};
use crate::cx::Ctx;
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};

/// `bias` is `[rows, q_heads · extent]` f32: row `r`, head `h`, distance
/// `d = q_pos − k_pos` in `[0, extent)` adds `bias[r, h·extent + d]` to the
/// scaled logit. When `log_alpha != 0` and `log_floor != 0`, the biased
/// logit is then scaled by `1 + log_alpha · ln((q_pos + 1) / log_floor)`
/// wherever that ratio exceeds one.
#[derive(Clone, Copy, Debug)]
pub struct RelBias {
    pub bias: Tensor,
    pub extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
}

/// Causal paged attention with the relative bias; the fire's custom mask is
/// not read (as on CUDA).
pub fn decode_rel(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    rel: RelBias,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    let call = rel_call(
        "attention.decode_rel",
        q,
        Tables::decode(plan),
        rel,
        window,
        head_dim,
        sm_scale,
        o,
        Walk::PerRow,
    );
    attend(ctx, call, pool)
}

pub fn prefill_rel(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    rel: RelBias,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_rel";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let call = rel_call(
        OP,
        q.data,
        Tables::prefill(plan),
        rel,
        window,
        head_dim,
        sm_scale,
        o,
        Walk::Auto,
    );
    attend(ctx, call, pool)
}

fn rel_call(
    op: &'static str,
    q: Tensor,
    tables: Tables,
    rel: RelBias,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    walk: Walk,
) -> Paged {
    Paged {
        op,
        q,
        tables,
        masked: false,
        window,
        causal: true,
        head_dim,
        sm_scale,
        o,
        lse: None,
        walk,
        selection: None,
        rel: Some((rel.bias, rel.extent, rel.log_floor, rel.log_alpha)),
    }
}
