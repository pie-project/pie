//! The prefill entries engine dispatches call (kernels-slang `attn::sdpa`),
//! with Metal's `causal` flag on the masked forms. The paged driver picks
//! the walk (`Walk::Auto`); `requests` and the GPU tuning argument are
//! dropped.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::PrefillPlan;
use super::paged::{Paged, Tables, Walk, kv_heads_agree};
use crate::error::{Error, refuse};
use crate::tensor::{KvPool, RaggedTensor, Stored};

fn call<T>(
    op: &'static str,
    q: RaggedTensor<T>,
    tables: Tables<T>,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: T,
    lse: Option<T>,
) -> Paged<T> {
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
        walk: Walk::Auto,
        selection: None,
        rel: None,
    }
}

fn u8_mask(op: &'static str, mask: impl Stored) -> Result<(), Error> {
    if !matches!(mask.dtype(), Dtype::U8 | Dtype::Bool) {
        return Err(refuse(
            op,
            format!(
                "the mask this op states is {:?}; a mask plane is u8",
                mask.dtype()
            ),
        ));
    }
    Ok(())
}

pub fn prefill<T: Stored>(
    q: RaggedTensor<T>,
    plan: &PrefillPlan<T>,
    pool: &KvPool<T>,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: T,
    _requests: u32,
) -> Result<Paged<T>, Error> {
    const OP: &str = "attn.prefill";
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
    );
    Ok(c)
}

pub fn prefill_lse<T: Stored>(
    q: RaggedTensor<T>,
    plan: &PrefillPlan<T>,
    pool: &KvPool<T>,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: T,
    lse: T,
    _requests: u32,
) -> Result<Paged<T>, Error> {
    const OP: &str = "attn.prefill";
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
    );
    Ok(c)
}

/// A prefill under the custom `mask` (gated per row by the plan's enable
/// flags). `causal: false` reads every key the row's lane holds (Metal's
/// bidirectional arm); so does a causal row whose enable flag is 2.
/// Reference: kernels-metal `attn::arbiter::masked`.
pub fn masked<T: Stored>(
    q: RaggedTensor<T>,
    plan: &PrefillPlan<T>,
    mask: T,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: T,
    _requests: u32,
) -> Result<Paged<T>, Error> {
    const OP: &str = "attn.prefill";
    u8_mask(OP, mask)?;
    let tables = Tables::prefill(plan).with_mask(mask);
    let c = call(OP, q, tables, window, causal, head_dim, sm_scale, o, None);
    Ok(c)
}

pub fn masked_lse<T: Stored>(
    q: RaggedTensor<T>,
    plan: &PrefillPlan<T>,
    mask: T,
    window: Option<u32>,
    causal: bool,
    head_dim: u32,
    sm_scale: f32,
    o: T,
    lse: T,
    _requests: u32,
) -> Result<Paged<T>, Error> {
    const OP: &str = "attn.prefill";
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
    );
    Ok(c)
}
