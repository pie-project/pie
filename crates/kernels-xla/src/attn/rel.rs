//! Paged attention with a per-row relative-position bias (T5-style buckets
//! gathered upstream) and optional log-length logit scaling.
//! Reference: kernels-cuda `attn::{decode_rel, prefill_rel}` and its
//! `VariantRelBias`.

#![allow(clippy::too_many_arguments)]

use super::paged::{Paged, Tables, Walk, kv_heads_agree};
use super::{DecodePlan, PrefillPlan};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Stored};

/// `bias` is `[rows, q_heads · extent]` f32: row `r`, head `h`, distance
/// `d = q_pos − k_pos` in `[0, extent)` adds `bias[r, h·extent + d]` to the
/// scaled logit. When `log_alpha != 0` and `log_floor != 0`, the biased
/// logit is then scaled by `1 + log_alpha · ln((q_pos + 1) / log_floor)`
/// wherever that ratio exceeds one.
#[derive(Clone, Copy, Debug)]
pub struct RelBias<T> {
    pub bias: T,
    pub extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
}

/// Causal paged attention with the relative bias; the fire's custom mask is
/// not read (as on CUDA).
pub fn decode_rel<T: Stored>(
    q: T,
    plan: &DecodePlan<T>,
    rel: RelBias<T>,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: T,
) -> Result<Paged<T>, Error> {
    let call = rel_call(
        "attn.decode_rel",
        q,
        Tables::decode(plan),
        rel,
        window,
        head_dim,
        sm_scale,
        o,
        Walk::PerRow,
    );
    Ok(call)
}

pub fn prefill_rel<T: Stored>(
    q: RaggedTensor<T>,
    plan: &PrefillPlan<T>,
    pool: &KvPool<T>,
    rel: RelBias<T>,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: T,
) -> Result<Paged<T>, Error> {
    const OP: &str = "attn.prefill_rel";
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
    Ok(call)
}

fn rel_call<T>(
    op: &'static str,
    q: T,
    tables: Tables<T>,
    rel: RelBias<T>,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: T,
    walk: Walk,
) -> Paged<T> {
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
