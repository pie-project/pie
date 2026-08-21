//! GENERATED — do not edit. One named `pub fn` per traced `#[routine]` in
//! `crates/kernels-cuda/src`, named by its trace name, in sorted-file then
//! source order (design-no-ask §10, B4-gen).
//!
//! The generator is `tests/generator/mod.rs`;
//! `cargo test -p model-dsl --test wrappers_are_current` refuses a stale
//! file and `UPDATE_WRAPPERS=1` rewrites it.
//!
//! Every mark is one argument — runtime streams and views included; a
//! wrapper here mints NOTHING in secret. A result whose routine states an
//! `out(..)` rule is derived at trace time through
//! [`model_ir::kernels::out_shape`]; an `Unstated` result stays a
//! `(Shape, DType)` argument. Trailing `layer` and `state` are the
//! statement's tags, uniformly. Recording goes through
//! [`crate::fire::fire`], so the symbol and the run arities come off the
//! routine's own marker.

// The prelude is fixed while the surface below is generated from another
// crate's tree, so any one regeneration may leave part of it unused.
#![allow(unused_imports)]

use kernels::{OutRule, OutWidth};
use model_ir::trace::{DType, Shape, StateRef, ValueId};

use super::ruled_out;
use crate::fire::{Call, fire};
use crate::{Trace, Val};

/// Generated for `attn::dispatch_attention_flashinfer_decode` from the
/// routine's own signature
/// (`kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dispatch_attention_flashinfer_decode(
    q: &Val,
    plan: &Val,
    o: (Shape, DType),
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    kvc: &Val,
    lse: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse = lse.is_some();
    let run_params = vec![
        window_left as u32,
        logits_soft_cap.to_bits(),
        sm_scale.to_bits(),
    ];
    let run_inputs = vec![q.id, plan.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    if let Some(o) = lse {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::dispatch_attention_flashinfer_decode` states `o`");
    let lse = has_lse.then(|| made.next().expect("`attn::dispatch_attention_flashinfer_decode` states `lse`"));
    (o, lse)
}

/// Generated for `attn::dispatch_attention_flashinfer_decode_lse` from the
/// routine's own signature
/// (`kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode_lse`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn dispatch_attention_flashinfer_decode_lse(
    q: &Val,
    plan: &Val,
    o: (Shape, DType),
    _lse: (Shape, DType),
    _window_left: i32,
    _logits_soft_cap: f32,
    _sm_scale: f32,
    kvc: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        _window_left as u32,
        _logits_soft_cap.to_bits(),
        _sm_scale.to_bits(),
    ];
    let run_inputs = vec![q.id, plan.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![o, _lse];
    let made = fire::<kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode_lse>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::dispatch_attention_flashinfer_decode_lse` states `o`");
    let _lse = made.next().expect("`attn::dispatch_attention_flashinfer_decode_lse` states `_lse`");
    (o, _lse)
}

/// Generated for `attn::dispatch_attention_flashinfer_decode_capture` from
/// the routine's own signature
/// (`kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode_capture`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn dispatch_attention_flashinfer_decode_capture(
    q: &Val,
    plan: &Val,
    o: (Shape, DType),
    window_left: i32,
    logits_soft_cap: f32,
    sm_scale: f32,
    kvc: &Val,
    score_out: (Shape, DType),
    score: &Val,
    lse: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse = lse.is_some();
    let run_params = vec![
        window_left as u32,
        logits_soft_cap.to_bits(),
        sm_scale.to_bits(),
    ];
    let run_inputs = vec![q.id, plan.id, kvc.id, score.id];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    run_outs.push(score_out);
    if let Some(o) = lse {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::fa2::dispatch_attention_flashinfer_decode_capture>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::dispatch_attention_flashinfer_decode_capture` states `o`");
    let score_out = made.next().expect("`attn::dispatch_attention_flashinfer_decode_capture` states `score_out`");
    let lse = has_lse.then(|| made.next().expect("`attn::dispatch_attention_flashinfer_decode_capture` states `lse`"));
    (o, score_out, lse)
}

/// Generated for `attn::dispatch_attention_flashinfer_prefill_bf16` from
/// the routine's own signature
/// (`kernels_cuda::attn::fa2::dispatch_attention_flashinfer_prefill_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn dispatch_attention_flashinfer_prefill_bf16(
    q: &Val,
    plan: &Val,
    o: (Shape, DType),
    logits_soft_cap: f32,
    sm_scale: f32,
    qo_indptr: &Val,
    kvc: &Val,
    lse: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse = lse.is_some();
    let run_params = vec![logits_soft_cap.to_bits(), sm_scale.to_bits()];
    let run_inputs = vec![q.id, plan.id, qo_indptr.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    if let Some(o) = lse {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::fa2::dispatch_attention_flashinfer_prefill_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::dispatch_attention_flashinfer_prefill_bf16` states `o`");
    let lse = has_lse.then(|| made.next().expect("`attn::dispatch_attention_flashinfer_prefill_bf16` states `lse`"));
    (o, lse)
}

/// Generated for `attn::dispatch_attention_flashinfer_prefill_capture_bf16`
/// from the routine's own signature
/// (`kernels_cuda::attn::fa2::dispatch_attention_flashinfer_prefill_capture_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn dispatch_attention_flashinfer_prefill_capture_bf16(
    q: &Val,
    plan: &Val,
    o: (Shape, DType),
    logits_soft_cap: f32,
    sm_scale: f32,
    kvc: &Val,
    qo_indptr: &Val,
    score_out: (Shape, DType),
    score: &Val,
    lse: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse = lse.is_some();
    let run_params = vec![logits_soft_cap.to_bits(), sm_scale.to_bits()];
    let run_inputs = vec![q.id, plan.id, kvc.id, qo_indptr.id, score.id];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    run_outs.push(score_out);
    if let Some(o) = lse {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::fa2::dispatch_attention_flashinfer_prefill_capture_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::dispatch_attention_flashinfer_prefill_capture_bf16` states `o`");
    let score_out = made.next().expect("`attn::dispatch_attention_flashinfer_prefill_capture_bf16` states `score_out`");
    let lse = has_lse.then(|| made.next().expect("`attn::dispatch_attention_flashinfer_prefill_capture_bf16` states `lse`"));
    (o, score_out, lse)
}

/// Generated for `attn::dispatch_attention_flashinfer_prefill_custom` from
/// the routine's own signature
/// (`kernels_cuda::attn::fa2::dispatch_attention_flashinfer_prefill_custom`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn dispatch_attention_flashinfer_prefill_custom(
    q: &Val,
    plan: &Val,
    o: (Shape, DType),
    logits_soft_cap: f32,
    sm_scale: f32,
    maskv: &Val,
    kvc: &Val,
    qo_indptr: &Val,
    lse: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse = lse.is_some();
    let run_params = vec![logits_soft_cap.to_bits(), sm_scale.to_bits()];
    let run_inputs = vec![q.id, plan.id, maskv.id, kvc.id, qo_indptr.id];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    if let Some(o) = lse {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::fa2::dispatch_attention_flashinfer_prefill_custom>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::dispatch_attention_flashinfer_prefill_custom` states `o`");
    let lse = has_lse.then(|| made.next().expect("`attn::dispatch_attention_flashinfer_prefill_custom` states `lse`"));
    (o, lse)
}

/// Generated for `attn::attention_flashinfer_prefill` from the routine's
/// own signature (`kernels_cuda::attn::fa2::attention_flashinfer_prefill`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn attention_flashinfer_prefill(
    q: &Val,
    o: (Shape, DType),
    logits_soft_cap: f32,
    sm_scale: f32,
    kvc: &Val,
    qo_indptr: &Val,
    head_dim: i32,
    plan_cache: &Val,
    qo_indptr_host: &Val,
    kv_page_indptr_host: &Val,
    kv_num_heads: i32,
    window_left: i32,
    lse: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse = lse.is_some();
    let run_params = vec![
        logits_soft_cap.to_bits(),
        sm_scale.to_bits(),
        head_dim as u32,
        kv_num_heads as u32,
        window_left as u32,
    ];
    let run_inputs = vec![
        q.id,
        kvc.id,
        qo_indptr.id,
        plan_cache.id,
        qo_indptr_host.id,
        kv_page_indptr_host.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    if let Some(o) = lse {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::fa2::attention_flashinfer_prefill>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::attention_flashinfer_prefill` states `o`");
    let lse = has_lse.then(|| made.next().expect("`attn::attention_flashinfer_prefill` states `lse`"));
    (o, lse)
}

/// Generated for `attn::attention_flashinfer_prefill_lse` from the
/// routine's own signature
/// (`kernels_cuda::attn::fa2::attention_flashinfer_prefill_lse`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn attention_flashinfer_prefill_lse(
    q: &Val,
    o: (Shape, DType),
    _lse: (Shape, DType),
    attn_logits_soft_cap: f32,
    sm_scale: f32,
    kvc: &Val,
    qo_indptr: &Val,
    head_dim: i32,
    plan_cache: &Val,
    qo_indptr_host: &Val,
    kv_page_indptr_host: &Val,
    kv_num_heads: i32,
    window_left: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        attn_logits_soft_cap.to_bits(),
        sm_scale.to_bits(),
        head_dim as u32,
        kv_num_heads as u32,
        window_left as u32,
    ];
    let run_inputs = vec![
        q.id,
        kvc.id,
        qo_indptr.id,
        plan_cache.id,
        qo_indptr_host.id,
        kv_page_indptr_host.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![o, _lse];
    let made = fire::<kernels_cuda::attn::fa2::attention_flashinfer_prefill_lse>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::attention_flashinfer_prefill_lse` states `o`");
    let _lse = made.next().expect("`attn::attention_flashinfer_prefill_lse` states `_lse`");
    (o, _lse)
}

/// Generated for `attn::mtp_shift_hidden` from the routine's own signature
/// (`kernels_cuda::attn::mtp_shift_hidden`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn mtp_shift_hidden(
    target_hidden: &Val,
    pending_hidden: &Val,
    out: (Shape, DType),
    qo_indptr: &Val,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = target_hidden.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![
        target_hidden.id,
        pending_hidden.id,
        qo_indptr.id,
        rsv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::attn::mtp_shift_hidden>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::mtp_shift_hidden` states `out`")
}

/// Generated for `attn::mtp_update_pending_hidden` from the routine's own
/// signature (`kernels_cuda::attn::mtp_update_pending_hidden`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn mtp_update_pending_hidden(
    target_hidden: &Val,
    qo_indptr: &Val,
    rsv: &Val,
    pending: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = target_hidden.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![target_hidden.id, qo_indptr.id, rsv.id, pending.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::attn::mtp_update_pending_hidden>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`attn::mtp_update_pending_hidden` states no result");
}

/// Generated for `attn::dsv4_boundary_meta_decode` from the routine's own
/// signature (`kernels_cuda::attn::dsv4_boundary_meta_decode`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dsv4_boundary_meta_decode(
    positions: &Val,
    out_pos: (Shape, DType),
    out_req: (Shape, DType),
    out_rope: (Shape, DType),
    ratio: i32,
    row_valid: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = positions.t.clone();
    let run_params = vec![ratio as u32];
    let run_inputs = vec![positions.id, row_valid.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out_pos, out_req, out_rope];
    let made = fire::<kernels_cuda::attn::dsv4_boundary_meta_decode>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let out_pos = made.next().expect("`attn::dsv4_boundary_meta_decode` states `out_pos`");
    let out_req = made.next().expect("`attn::dsv4_boundary_meta_decode` states `out_req`");
    let out_rope = made.next().expect("`attn::dsv4_boundary_meta_decode` states `out_rope`");
    (out_pos, out_req, out_rope)
}

/// Generated for `attn::dsv4_boundary_meta_paged` from the routine's own
/// signature (`kernels_cuda::attn::dsv4_boundary_meta_paged`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dsv4_boundary_meta_paged(
    positions: &Val,
    out_pos: (Shape, DType),
    out_req: (Shape, DType),
    out_rope: (Shape, DType),
    ratio: i32,
    row_valid: &Val,
    qo_indptr: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = positions.t.clone();
    let run_params = vec![ratio as u32];
    let run_inputs = vec![positions.id, row_valid.id, qo_indptr.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out_pos, out_req, out_rope];
    let made = fire::<kernels_cuda::attn::dsv4_boundary_meta_paged>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let out_pos = made.next().expect("`attn::dsv4_boundary_meta_paged` states `out_pos`");
    let out_req = made.next().expect("`attn::dsv4_boundary_meta_paged` states `out_req`");
    let out_rope = made.next().expect("`attn::dsv4_boundary_meta_paged` states `out_rope`");
    (out_pos, out_req, out_rope)
}

/// Generated for `attn::attention_compressed_paged_bf16` from the routine's
/// own signature (`kernels_cuda::attn::attention_compressed_paged_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn attention_compressed_paged_bf16(
    q: &Val,
    o: (Shape, DType),
    lse_out: (Shape, DType),
    ratio: i32,
    num_q_heads: i32,
    head_dim: i32,
    kvc: &Val,
    sm_scale: f32,
    positions: &Val,
    request_of_token: &Val,
    comp_kv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        ratio as u32,
        num_q_heads as u32,
        head_dim as u32,
        sm_scale.to_bits(),
    ];
    let run_inputs = vec![
        q.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        comp_kv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![o, lse_out];
    let made = fire::<kernels_cuda::attn::attention_compressed_paged_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::attention_compressed_paged_bf16` states `o`");
    let lse_out = made.next().expect("`attn::attention_compressed_paged_bf16` states `lse_out`");
    (o, lse_out)
}

/// Generated for `attn::dsa_index_knorm_rope` from the routine's own
/// signature (`kernels_cuda::attn::dsa_index_knorm_rope`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dsa_index_knorm_rope(
    idx_k: &Val,
    idx_k_out: (Shape, DType),
    k_norm_weight: &str,
    k_norm_bias: &str,
    rope_dim: i32,
    theta: f32,
    eps: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = idx_k.t.clone();
    let run_params = vec![rope_dim as u32, theta.to_bits(), eps.to_bits()];
    let run_inputs = vec![idx_k.id, positions.id];
    let run_weights = vec![
        k_norm_weight.to_string(),
        k_norm_bias.to_string(),
    ];
    let run_outs = vec![idx_k_out];
    let made = fire::<kernels_cuda::attn::dsa_index_knorm_rope>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::dsa_index_knorm_rope` states `idx_k`")
}

/// Generated for `attn::dsa_index_q_rope` from the routine's own signature
/// (`kernels_cuda::attn::dsa_index_q_rope`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dsa_index_q_rope(
    idx_q: &Val,
    idx_q_out: (Shape, DType),
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = idx_q.t.clone();
    let run_params = vec![
        n_heads as u32,
        head_dim as u32,
        rope_dim as u32,
        theta.to_bits(),
    ];
    let run_inputs = vec![idx_q.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![idx_q_out];
    let made = fire::<kernels_cuda::attn::dsa_index_q_rope>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::dsa_index_q_rope` states `idx_q`")
}

/// Generated for `attn::dsa_index_topk_mask` from the routine's own
/// signature (`kernels_cuda::attn::dsa_index_topk_mask`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dsa_index_topk_mask(
    idx_q: &Val,
    idx_k: &Val,
    idx_w: &Val,
    mask: (Shape, DType),
    n_heads: i32,
    head_dim: i32,
    topk: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = idx_q.t.clone();
    let run_params = vec![n_heads as u32, head_dim as u32, topk as u32];
    let run_inputs = vec![idx_q.id, idx_k.id, idx_w.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![mask];
    let made = fire::<kernels_cuda::attn::dsa_index_topk_mask>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::dsa_index_topk_mask` states `mask`")
}

/// Generated for `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` from
/// the routine's own signature
/// (`kernels_cuda::attn::qkv_fused::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
    packed: &Val,
    q_out: (Shape, DType),
    q_weight: &str,
    k_weight: &str,
    num_kv_heads: i32,
    head_dim: i32,
    kvc: &Val,
    theta: f32,
    eps: f32,
    positions: &Val,
    row_valid: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![
        num_kv_heads as u32,
        head_dim as u32,
        theta.to_bits(),
        eps.to_bits(),
    ];
    let run_inputs = vec![packed.id, kvc.id, positions.id, row_valid.id];
    let run_weights = vec![q_weight.to_string(), k_weight.to_string()];
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::attn::qkv_fused::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` states `q_out`")
}

/// Generated for `attn::qkv_decode_qk_norm_rope_write_kv_bf16` from the
/// routine's own signature
/// (`kernels_cuda::attn::qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn qkv_decode_qk_norm_rope_write_kv_bf16(
    packed: &Val,
    q_out: (Shape, DType),
    q_weight: &str,
    k_weight: &str,
    rope_table: Option<&Val>,
    num_kv_heads: i32,
    head_dim: i32,
    kvc: &Val,
    theta: f32,
    eps: f32,
    positions: &Val,
    row_valid: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![
        num_kv_heads as u32,
        head_dim as u32,
        theta.to_bits(),
        eps.to_bits(),
    ];
    let mut run_inputs = Vec::new();
    run_inputs.push(packed.id);
    if let Some(v) = rope_table {
        run_inputs.push(v.id);
    }
    run_inputs.push(kvc.id);
    run_inputs.push(positions.id);
    run_inputs.push(row_valid.id);
    let run_weights = vec![q_weight.to_string(), k_weight.to_string()];
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::attn::qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::qkv_decode_qk_norm_rope_write_kv_bf16` states `q_out`")
}

/// Generated for `attn::dsv4_compress_gather_paged_bf16` from the routine's
/// own signature
/// (`kernels_cuda::attn::dsv4_compress::dsv4_compress_gather_paged_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn dsv4_compress_gather_paged_bf16(
    boundary_pos: &Val,
    boundary_req: &Val,
    out: (Shape, DType),
    ratio: i32,
    coff: i32,
    kvc: &Val,
    state_kv: &Val,
    state_score: &Val,
    ape: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = boundary_pos.t.clone();
    let run_params = vec![ratio as u32, coff as u32];
    let run_inputs = vec![
        boundary_pos.id,
        boundary_req.id,
        kvc.id,
        state_kv.id,
        state_score.id,
        ape.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::attn::dsv4_compress::dsv4_compress_gather_paged_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::dsv4_compress_gather_paged_bf16` states `out`")
}

/// Generated for `attn::dsv4_store_comp_entries_bf16` from the routine's
/// own signature
/// (`kernels_cuda::attn::dsv4_compress::dsv4_store_comp_entries_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn dsv4_store_comp_entries_bf16(
    entries: &Val,
    boundary_pos: &Val,
    boundary_req: &Val,
    kvc: &Val,
    comp_kv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = entries.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![
        entries.id,
        boundary_pos.id,
        boundary_req.id,
        kvc.id,
        comp_kv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::attn::dsv4_compress::dsv4_store_comp_entries_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`attn::dsv4_store_comp_entries_bf16` states no result");
}

/// Generated for `attn::write_kv_explicit_bf16` from the routine's own
/// signature (`kernels_cuda::attn::kv_paged::write_kv_explicit_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn write_kv_explicit_bf16(
    k_curr: &Val,
    v_curr: &Val,
    kvc: &Val,
    num_kv_heads: i32,
    head_dim: i32,
    row_valid: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = k_curr.t.clone();
    let run_params = vec![num_kv_heads as u32, head_dim as u32];
    let run_inputs = vec![k_curr.id, v_curr.id, kvc.id, row_valid.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::attn::kv_paged::write_kv_explicit_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`attn::write_kv_explicit_bf16` states no result");
}

/// Generated for `attn::write_kv_explicit_bf16_devwin` from the routine's
/// own signature
/// (`kernels_cuda::attn::kv_paged::write_kv_explicit_bf16_devwin`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn write_kv_explicit_bf16_devwin(
    k_curr: &Val,
    v_curr: &Val,
    kvc: &Val,
    num_kv_heads: i32,
    head_dim: i32,
    row_valid: &Val,
    n_max: i32,
    win_start: i32,
    win_len: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = k_curr.t.clone();
    let run_params = vec![
        num_kv_heads as u32,
        head_dim as u32,
        n_max as u32,
        win_start as u32,
        win_len as u32,
    ];
    let run_inputs = vec![k_curr.id, v_curr.id, kvc.id, row_valid.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::attn::kv_paged::write_kv_explicit_bf16_devwin>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`attn::write_kv_explicit_bf16_devwin` states no result");
}

/// Generated for `attn::write_kv_to_pages_bf16` from the routine's own
/// signature (`kernels_cuda::attn::kv_paged::write_kv_to_pages_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn write_kv_to_pages_bf16(
    k_curr: &Val,
    v_curr: &Val,
    kvc: &Val,
    num_kv_heads: i32,
    head_dim: i32,
    qo_indptr: &Val,
    row_valid: &Val,
    first_token: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = k_curr.t.clone();
    let run_params = vec![num_kv_heads as u32, head_dim as u32];
    let run_inputs = vec![
        k_curr.id,
        v_curr.id,
        kvc.id,
        qo_indptr.id,
        row_valid.id,
        first_token.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::attn::kv_paged::write_kv_to_pages_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`attn::write_kv_to_pages_bf16` states no result");
}

/// Generated for `attn::write_kv_to_pages_quantised` from the routine's own
/// signature (`kernels_cuda::attn::kv_paged::write_kv_to_pages_quantised`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
pub fn write_kv_to_pages_quantised(
    k_curr: &Val,
    v_curr: &Val,
    kvc: &Val,
    num_kv_heads: i32,
    head_dim: i32,
    first_token: &Val,
    qo_indptr: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = k_curr.t.clone();
    let run_params = vec![num_kv_heads as u32, head_dim as u32];
    let run_inputs = vec![
        k_curr.id,
        v_curr.id,
        kvc.id,
        first_token.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::attn::kv_paged::write_kv_to_pages_quantised>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`attn::write_kv_to_pages_quantised` states no result");
}

/// Generated for `attn::lse_log2_to_ln` from the routine's own signature
/// (`kernels_cuda::attn::lse_log2_to_ln`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn lse_log2_to_ln(
    lse: &Val,
    lse_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = lse.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![lse.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![lse_out];
    let made = fire::<kernels_cuda::attn::lse_log2_to_ln>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::lse_log2_to_ln` states `lse`")
}

/// Generated for `attn::attention_sink_rescale` from the routine's own
/// signature (`kernels_cuda::attn::attention_sink_rescale`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn attention_sink_rescale(
    o: &Val,
    o_out: (Shape, DType),
    lse: &Val,
    sinks: &str,
    num_q_heads: i32,
    head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = o.t.clone();
    let run_params = vec![num_q_heads as u32, head_dim as u32];
    let run_inputs = vec![o.id, lse.id];
    let run_weights = vec![sinks.to_string()];
    let run_outs = vec![o_out];
    let made = fire::<kernels_cuda::attn::attention_sink_rescale>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::attention_sink_rescale` states `o`")
}

/// Generated for `attn::split_qkv_bf16_devwin` from the routine's own
/// signature (`kernels_cuda::attn::split_qkv_bf16_devwin`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn split_qkv_bf16_devwin(
    packed: &Val,
    q_out: (Shape, DType),
    k_out: (Shape, DType),
    v_out: (Shape, DType),
    n_max: i32,
    win_start: i32,
    win_len: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = packed.t.clone();
    let run_params = vec![n_max as u32, win_start as u32, win_len as u32];
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, k_out, v_out];
    let made = fire::<kernels_cuda::attn::split_qkv_bf16_devwin>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q_out = made.next().expect("`attn::split_qkv_bf16_devwin` states `q_out`");
    let k_out = made.next().expect("`attn::split_qkv_bf16_devwin` states `k_out`");
    let v_out = made.next().expect("`attn::split_qkv_bf16_devwin` states `v_out`");
    (q_out, k_out, v_out)
}

/// Generated for `attn::attention_naive_paged` from the routine's own
/// signature (`kernels_cuda::attn::attention_naive_paged`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn attention_naive_paged(
    q: &Val,
    o: (Shape, DType),
    kvc: &Val,
    head_dim: i32,
    num_kv_heads: i32,
    window_left: i32,
    sm_scale: f32,
    logits_soft_cap: f32,
    qo_indptr: &Val,
    lse_out: Option<(Shape, DType)>,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Option<Val>) {
    let t = q.t.clone();
    let has_lse_out = lse_out.is_some();
    let run_params = vec![
        head_dim as u32,
        num_kv_heads as u32,
        window_left as u32,
        sm_scale.to_bits(),
        logits_soft_cap.to_bits(),
    ];
    let run_inputs = vec![q.id, kvc.id, qo_indptr.id];
    let run_weights: Vec<String> = Vec::new();
    let mut run_outs = Vec::new();
    run_outs.push(o);
    if let Some(o) = lse_out {
        run_outs.push(o);
    }
    let made = fire::<kernels_cuda::attn::attention_naive_paged>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o = made.next().expect("`attn::attention_naive_paged` states `o`");
    let lse_out = has_lse_out.then(|| made.next().expect("`attn::attention_naive_paged` states `lse_out`"));
    (o, lse_out)
}

/// Generated for `attn::attn_res_blend` from the routine's own signature
/// (`kernels_cuda::attn::attn_res_blend`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn attn_res_blend(
    prefix: &Val,
    blocks: &Val,
    norm_weight: &str,
    proj_weight: &str,
    out: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = prefix.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![prefix.id, blocks.id];
    let run_weights = vec![norm_weight.to_string(), proj_weight.to_string()];
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::attn::attn_res_blend>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::attn_res_blend` states `out`")
}

/// Generated for `attn::pad_head_dim` from the routine's own signature
/// (`kernels_cuda::attn::pad_head_dim`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn pad_head_dim(
    packed: &Val,
    padded: (Shape, DType),
    head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![head_dim as u32];
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![padded];
    let made = fire::<kernels_cuda::attn::pad_head_dim>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::pad_head_dim` states `padded`")
}

/// Generated for `attn::strip_head_dim` from the routine's own signature
/// (`kernels_cuda::attn::strip_head_dim`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn strip_head_dim(
    padded: &Val,
    packed: (Shape, DType),
    head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = padded.t.clone();
    let run_params = vec![head_dim as u32];
    let run_inputs = vec![padded.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![packed];
    let made = fire::<kernels_cuda::attn::strip_head_dim>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::strip_head_dim` states `packed`")
}

/// Generated for `attn::logit_softcap` from the routine's own signature
/// (`kernels_cuda::attn::logit_softcap`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn logit_softcap(
    x: &Val,
    x_out: (Shape, DType),
    cap: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![cap.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![x_out];
    let made = fire::<kernels_cuda::attn::logit_softcap>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::logit_softcap` states `x`")
}

/// Generated for `attn::kimi_split_q_b` from the routine's own signature
/// (`kernels_cuda::attn::kimi_split_q_b`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn kimi_split_q_b(
    q_b: &Val,
    q_nope: (Shape, DType),
    q_pe: (Shape, DType),
    heads: i32,
    nope: i32,
    rope: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q_b.t.clone();
    let run_params = vec![heads as u32, nope as u32, rope as u32];
    let run_inputs = vec![q_b.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_nope, q_pe];
    let made = fire::<kernels_cuda::attn::kimi_split_q_b>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q_nope = made.next().expect("`attn::kimi_split_q_b` states `q_nope`");
    let q_pe = made.next().expect("`attn::kimi_split_q_b` states `q_pe`");
    (q_nope, q_pe)
}

/// Generated for `attn::kimi_split_kv_a_norm` from the routine's own
/// signature (`kernels_cuda::attn::kimi_split_kv_a_norm`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn kimi_split_kv_a_norm(
    kv_a: &Val,
    norm_weight: &str,
    kv_c: (Shape, DType),
    k_pe: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = kv_a.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![kv_a.id];
    let run_weights = vec![norm_weight.to_string()];
    let run_outs = vec![kv_c, k_pe];
    let made = fire::<kernels_cuda::attn::kimi_split_kv_a_norm>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let kv_c = made.next().expect("`attn::kimi_split_kv_a_norm` states `kv_c`");
    let k_pe = made.next().expect("`attn::kimi_split_kv_a_norm` states `k_pe`");
    (kv_c, k_pe)
}

/// Generated for `attn::combine_attn_outputs` from the routine's own
/// signature (`kernels_cuda::attn::combine_attn_outputs`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn combine_attn_outputs(
    o1: &Val,
    lse1: &Val,
    o2: &Val,
    lse2: &Val,
    o_out: (Shape, DType),
    lse_out: (Shape, DType),
    num_heads: i32,
    head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = o1.t.clone();
    let run_params = vec![num_heads as u32, head_dim as u32];
    let run_inputs = vec![o1.id, lse1.id, o2.id, lse2.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![o_out, lse_out];
    let made = fire::<kernels_cuda::attn::combine_attn_outputs>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let o_out = made.next().expect("`attn::combine_attn_outputs` states `o_out`");
    let lse_out = made.next().expect("`attn::combine_attn_outputs` states `lse_out`");
    (o_out, lse_out)
}

/// Generated for `attn::attention_xqa_decode_bf16_prepared` from the
/// routine's own signature
/// (`kernels_cuda::attn::xqa::attention_xqa_decode_bf16_prepared`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn attention_xqa_decode_bf16_prepared(
    q: &Val,
    o: (Shape, DType),
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    kvc: &Val,
    sm_scale: f32,
    num_requests: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q.t.clone();
    let run_params = vec![
        num_q_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        sm_scale.to_bits(),
        num_requests as u32,
    ];
    let run_inputs = vec![q.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![o];
    let made = fire::<kernels_cuda::attn::xqa::attention_xqa_decode_bf16_prepared>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`attn::attention_xqa_decode_bf16_prepared` states `o`")
}

/// Generated for `dist::all_reduce_bf16` from the routine's own signature
/// (`kernels_cuda::dist::all_reduce_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn all_reduce_bf16(
    buf: &Val,
    buf_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = buf.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![buf.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![buf_out];
    let made = fire::<kernels_cuda::dist::all_reduce_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`dist::all_reduce_bf16` states `buf`")
}

/// Generated for `dist::all_reduce_bf16_out` from the routine's own
/// signature (`kernels_cuda::dist::all_reduce_bf16_out`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn all_reduce_bf16_out(
    src: &Val,
    dst: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![dst];
    let made = fire::<kernels_cuda::dist::all_reduce_bf16_out>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`dist::all_reduce_bf16_out` states `dst`")
}

/// Generated for `dist::all_gather_bf16` from the routine's own signature
/// (`kernels_cuda::dist::all_gather_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn all_gather_bf16(
    src: &Val,
    dst: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![dst];
    let made = fire::<kernels_cuda::dist::all_gather_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`dist::all_gather_bf16` states `dst`")
}

/// Generated for `attn::split_qkv_bf16` from the routine's own signature
/// (`kernels_cuda::driver_internal::split_qkv_bf16`); the statement records
/// through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn split_qkv_bf16(
    packed: &Val,
    q_out: (Shape, DType),
    k_out: (Shape, DType),
    v_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = packed.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, k_out, v_out];
    let made = fire::<kernels_cuda::driver_internal::split_qkv_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q_out = made.next().expect("`attn::split_qkv_bf16` states `q_out`");
    let k_out = made.next().expect("`attn::split_qkv_bf16` states `k_out`");
    let v_out = made.next().expect("`attn::split_qkv_bf16` states `v_out`");
    (q_out, k_out, v_out)
}

/// Generated for `ssm::qwen_gdn_post_conv_prep_bf16` from the routine's own
/// signature
/// (`kernels_cuda::driver_internal::qwen_gdn_post_conv_prep_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn qwen_gdn_post_conv_prep_bf16(
    qkv_post: &Val,
    a: &Val,
    b: &Val,
    a_log: &str,
    dt_bias: &str,
    q_norm_kh: (Shape, DType),
    k_norm_kh: (Shape, DType),
    v_fp32: (Shape, DType),
    g_log_out: (Shape, DType),
    beta_out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val, Val, Val) {
    let t = qkv_post.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        conv_dim as u32,
    ];
    let run_inputs = vec![qkv_post.id, a.id, b.id];
    let run_weights = vec![a_log.to_string(), dt_bias.to_string()];
    let run_outs = vec![q_norm_kh, k_norm_kh, v_fp32, g_log_out, beta_out];
    let made = fire::<kernels_cuda::driver_internal::qwen_gdn_post_conv_prep_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q_norm_kh = made.next().expect("`ssm::qwen_gdn_post_conv_prep_bf16` states `q_norm_kh`");
    let k_norm_kh = made.next().expect("`ssm::qwen_gdn_post_conv_prep_bf16` states `k_norm_kh`");
    let v_fp32 = made.next().expect("`ssm::qwen_gdn_post_conv_prep_bf16` states `v_fp32`");
    let g_log_out = made.next().expect("`ssm::qwen_gdn_post_conv_prep_bf16` states `g_log_out`");
    let beta_out = made.next().expect("`ssm::qwen_gdn_post_conv_prep_bf16` states `beta_out`");
    (q_norm_kh, k_norm_kh, v_fp32, g_log_out, beta_out)
}

/// Generated for `layout::split_q_gate_bf16` from the routine's own
/// signature (`kernels_cuda::driver_internal::split_q_gate_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn split_q_gate_bf16(
    packed: &Val,
    q_out: (Shape, DType),
    gate_out: (Shape, DType),
    head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = packed.t.clone();
    let run_params = vec![head_dim as u32];
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, gate_out];
    let made = fire::<kernels_cuda::driver_internal::split_q_gate_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q_out = made.next().expect("`layout::split_q_gate_bf16` states `q_out`");
    let gate_out = made.next().expect("`layout::split_q_gate_bf16` states `gate_out`");
    (q_out, gate_out)
}

/// Generated for `mlp::sigmoid_gate_inplace_bf16` from the routine's own
/// signature (`kernels_cuda::driver_internal::sigmoid_gate_inplace_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn sigmoid_gate_inplace_bf16(
    x: &Val,
    x_out: (Shape, DType),
    gate: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id, gate.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![x_out];
    let made = fire::<kernels_cuda::driver_internal::sigmoid_gate_inplace_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::sigmoid_gate_inplace_bf16` states `x`")
}

/// Generated for `gemm::act_x_wt_bf16` from the routine's own signature
/// (`kernels_cuda::gemm::act_x_wt_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn act_x_wt_bf16(
    act: &Val,
    w: &str,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = act.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![act.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::gemm::act_x_wt_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`gemm::act_x_wt_bf16` states `y`")
}

/// Generated for `gemm::act_x_w` from the routine's own signature
/// (`kernels_cuda::gemm::act_x_w`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn act_x_w(
    act: &Val,
    w: &str,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = act.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![act.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::gemm::act_x_w>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`gemm::act_x_w` states `y`")
}

/// Generated for `gemm::act_x_w_acc` from the routine's own signature
/// (`kernels_cuda::gemm::act_x_w_acc`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn act_x_w_acc(
    act: &Val,
    w: &str,
    y: &Val,
    y_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = act.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![act.id, y.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![y_out];
    let made = fire::<kernels_cuda::gemm::act_x_w_acc>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`gemm::act_x_w_acc` states `y`")
}

/// Generated for `gemm::act_x_wt_bf16_out_fp32` from the routine's own
/// signature (`kernels_cuda::gemm::act_x_wt_bf16_out_fp32`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn act_x_wt_bf16_out_fp32(
    act: &Val,
    w: &str,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = act.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![act.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::gemm::act_x_wt_bf16_out_fp32>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`gemm::act_x_wt_bf16_out_fp32` states `y`")
}

/// Generated for `gemm::grouped_act_x_wt_bf16` from the routine's own
/// signature (`kernels_cuda::gemm::grouped_act_x_wt_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
pub fn grouped_act_x_wt_bf16(
    group_count: i32,
    beta: f32,
    n: i32,
    k: i32,
    groups: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = groups.t.clone();
    let run_params = vec![
        group_count as u32,
        beta.to_bits(),
        n as u32,
        k as u32,
    ];
    let run_inputs = vec![groups.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::gemm::grouped_act_x_wt_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`gemm::grouped_act_x_wt_bf16` states no result");
}

/// Generated for `gemm::act_x_wt_bias_bf16` from the routine's own
/// signature (`kernels_cuda::gemm::act_x_wt_bias_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn act_x_wt_bias_bf16(
    act: &Val,
    w: &str,
    bias: &str,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = act.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![act.id];
    let run_weights = vec![w.to_string(), bias.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::gemm::act_x_wt_bias_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`gemm::act_x_wt_bias_bf16` states `y`")
}

/// Generated for `layout::split_bf16_rows` from the routine's own signature
/// (`kernels_cuda::layout::split_bf16_rows`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn split_bf16_rows(
    src: &Val,
    left: (Shape, DType),
    right: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = src.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![left, right];
    let made = fire::<kernels_cuda::layout::split_bf16_rows>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let left = made.next().expect("`layout::split_bf16_rows` states `left`");
    let right = made.next().expect("`layout::split_bf16_rows` states `right`");
    (left, right)
}

/// Generated for `layout::split_qwen_gdn_ba` from the routine's own
/// signature (`kernels_cuda::layout::split_qwen_gdn_ba`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn split_qwen_gdn_ba(
    ba: &Val,
    b_out: (Shape, DType),
    a_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = ba.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![ba.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![b_out, a_out];
    let made = fire::<kernels_cuda::layout::split_qwen_gdn_ba>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let b_out = made.next().expect("`layout::split_qwen_gdn_ba` states `b_out`");
    let a_out = made.next().expect("`layout::split_qwen_gdn_ba` states `a_out`");
    (b_out, a_out)
}

/// Generated for `layout::gather_bf16_rows` from the routine's own
/// signature (`kernels_cuda::layout::gather_bf16_rows`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn gather_bf16_rows(
    src: &Val,
    dst: (Shape, DType),
    sampling_indices: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src.id, sampling_indices.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![dst];
    let made = fire::<kernels_cuda::layout::gather_bf16_rows>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`layout::gather_bf16_rows` states `dst`")
}

/// Generated for `layout::transpose_bf16_nld_to_lnd` from the routine's own
/// signature (`kernels_cuda::layout::transpose_bf16_nld_to_lnd`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn transpose_bf16_nld_to_lnd(
    src: &Val,
    dst: (Shape, DType),
    dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src.t.clone();
    let run_params = vec![dim as u32];
    let run_inputs = vec![src.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![dst];
    let made = fire::<kernels_cuda::layout::transpose_bf16_nld_to_lnd>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`layout::transpose_bf16_nld_to_lnd` states `dst`")
}

/// Generated for `layout::embed_bf16` from the routine's own signature
/// (`kernels_cuda::layout::embed_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn embed_bf16(
    weight: &str,
    y: (Shape, DType),
    token_ids: &Val,
    vocab: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = token_ids.t.clone();
    let run_params = vec![vocab as u32];
    let run_inputs = vec![token_ids.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::layout::embed_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`layout::embed_bf16` states `y`")
}

/// Generated for `mlp::swiglu` from the routine's own signature
/// (`kernels_cuda::mlp::swiglu`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn swiglu(
    gate: &Val,
    up: &Val,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::swiglu>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::swiglu` states `y`")
}

/// Generated for `mlp::swiglu_clamp` from the routine's own signature
/// (`kernels_cuda::mlp::swiglu_clamp`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn swiglu_clamp(
    gate: &Val,
    up: &Val,
    y: (Shape, DType),
    limit: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![limit.to_bits()];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::swiglu_clamp>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::swiglu_clamp` states `y`")
}

/// Generated for `mlp::situ` from the routine's own signature
/// (`kernels_cuda::mlp::situ`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn situ(
    gate: &Val,
    up: &Val,
    y: (Shape, DType),
    beta: f32,
    linear_beta: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![beta.to_bits(), linear_beta.to_bits()];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::situ>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::situ` states `y`")
}

/// Generated for `mlp::geglu_tanh` from the routine's own signature
/// (`kernels_cuda::mlp::geglu_tanh`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn geglu_tanh(
    gate: &Val,
    up: &Val,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::geglu_tanh>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::geglu_tanh` states `y`")
}

/// Generated for `mlp::relu2` from the routine's own signature
/// (`kernels_cuda::mlp::relu2`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn relu2(
    x: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "mlp::relu2", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let made = fire::<kernels_cuda::mlp::relu2>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::relu2` states `y`")
}

/// Generated for `mlp::gpt_oss_glu` from the routine's own signature
/// (`kernels_cuda::mlp::gpt_oss_glu`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn gpt_oss_glu(
    gate: &Val,
    up: &Val,
    y: (Shape, DType),
    limit: f32,
    alpha: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![limit.to_bits(), alpha.to_bits()];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::gpt_oss_glu>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::gpt_oss_glu` states `y`")
}

/// Generated for `mlp::chunked_swiglu` from the routine's own signature
/// (`kernels_cuda::mlp::chunked_swiglu`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunked_swiglu(
    packed: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "mlp::chunked_swiglu", OutRule::Shaped { rows_of: 0, width: OutWidth::Half { of: 0 } }, &run_inputs, &run_params),
    ];
    let made = fire::<kernels_cuda::mlp::chunked_swiglu>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::chunked_swiglu` states `y`")
}

/// Generated for `mlp::chunked_swiglu_into` from the routine's own
/// signature (`kernels_cuda::mlp::chunked_swiglu_into`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunked_swiglu_into(
    packed: &Val,
    y: &Val,
    y_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![packed.id, y.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y_out];
    let made = fire::<kernels_cuda::mlp::chunked_swiglu_into>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::chunked_swiglu_into` states `y`")
}

/// Generated for `mlp::chunked_swiglu_clamp` from the routine's own
/// signature (`kernels_cuda::mlp::chunked_swiglu_clamp`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunked_swiglu_clamp(
    packed: &Val,
    y: (Shape, DType),
    limit: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![limit.to_bits()];
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::chunked_swiglu_clamp>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::chunked_swiglu_clamp` states `y`")
}

/// Generated for `mlp::chunked_situ` from the routine's own signature
/// (`kernels_cuda::mlp::chunked_situ`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunked_situ(
    packed: &Val,
    y: (Shape, DType),
    beta: f32,
    linear_beta: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![beta.to_bits(), linear_beta.to_bits()];
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::chunked_situ>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::chunked_situ` states `y`")
}

/// Generated for `mlp::chunked_geglu_tanh` from the routine's own signature
/// (`kernels_cuda::mlp::chunked_geglu_tanh`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunked_geglu_tanh(
    packed: &Val,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::mlp::chunked_geglu_tanh>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::chunked_geglu_tanh` states `y`")
}

/// Generated for `mlp::sigmoid_dot_scalar_gate_add` from the routine's own
/// signature (`kernels_cuda::mlp::sigmoid_dot_scalar_gate_add`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn sigmoid_dot_scalar_gate_add(
    x: &Val,
    gate_w: &str,
    out: &Val,
    out_out: (Shape, DType),
    y: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id, out.id, y.id];
    let run_weights = vec![gate_w.to_string()];
    let run_outs = vec![out_out];
    let made = fire::<kernels_cuda::mlp::sigmoid_dot_scalar_gate_add>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::sigmoid_dot_scalar_gate_add` states `out`")
}

/// Generated for `mlp::gaussian_topk` from the routine's own signature
/// (`kernels_cuda::mlp::gaussian_topk`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn gaussian_topk(
    x: &Val,
    x_out: (Shape, DType),
    std_multiplier: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![std_multiplier.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![x_out];
    let made = fire::<kernels_cuda::mlp::gaussian_topk>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`mlp::gaussian_topk` states `x`")
}

/// Generated for `moe::topk_sigmoid` from the routine's own signature
/// (`kernels_cuda::moe::topk_sigmoid`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn topk_sigmoid(
    logits: &Val,
    topk_idx: (Shape, DType),
    topk_w: (Shape, DType),
    correction_bias: Option<&str>,
    renormalize: bool,
    routed_scaling_factor: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params = vec![
        u32::from(renormalize),
        routed_scaling_factor.to_bits(),
    ];
    let run_inputs = vec![logits.id];
    let mut run_weights = Vec::new();
    if let Some(w) = correction_bias {
        run_weights.push(w.to_string());
    }
    let run_outs = vec![topk_idx, topk_w];
    let made = fire::<kernels_cuda::moe::topk_sigmoid>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let topk_idx = made.next().expect("`moe::topk_sigmoid` states `topk_idx`");
    let topk_w = made.next().expect("`moe::topk_sigmoid` states `topk_w`");
    (topk_idx, topk_w)
}

/// Generated for `moe::topk_sqrtsoftplus` from the routine's own signature
/// (`kernels_cuda::moe::topk_sqrtsoftplus`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn topk_sqrtsoftplus(
    logits: &Val,
    topk_idx: (Shape, DType),
    topk_w: (Shape, DType),
    correction_bias: Option<&str>,
    renormalize: bool,
    routed_scaling_factor: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params = vec![
        u32::from(renormalize),
        routed_scaling_factor.to_bits(),
    ];
    let run_inputs = vec![logits.id];
    let mut run_weights = Vec::new();
    if let Some(w) = correction_bias {
        run_weights.push(w.to_string());
    }
    let run_outs = vec![topk_idx, topk_w];
    let made = fire::<kernels_cuda::moe::topk_sqrtsoftplus>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let topk_idx = made.next().expect("`moe::topk_sqrtsoftplus` states `topk_idx`");
    let topk_w = made.next().expect("`moe::topk_sqrtsoftplus` states `topk_w`");
    (topk_idx, topk_w)
}

/// Generated for `moe::hash_route_lookup` from the routine's own signature
/// (`kernels_cuda::moe::hash_route_lookup`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn hash_route_lookup(
    token_ids: &Val,
    tid2eid: &str,
    logits: &Val,
    topk_idx: (Shape, DType),
    topk_w: (Shape, DType),
    vocab_size: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = token_ids.t.clone();
    let run_params = vec![
        vocab_size as u32,
        u32::from(renormalize),
        routed_scaling_factor.to_bits(),
    ];
    let run_inputs = vec![token_ids.id, logits.id];
    let run_weights = vec![tid2eid.to_string()];
    let run_outs = vec![topk_idx, topk_w];
    let made = fire::<kernels_cuda::moe::hash_route_lookup>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let topk_idx = made.next().expect("`moe::hash_route_lookup` states `topk_idx`");
    let topk_w = made.next().expect("`moe::hash_route_lookup` states `topk_w`");
    (topk_idx, topk_w)
}

/// Generated for `moe::topk_softmax` from the routine's own signature
/// (`kernels_cuda::moe::topk_softmax`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn topk_softmax(
    logits: &Val,
    topk_idx: (Shape, DType),
    topk_w: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![logits.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![topk_idx, topk_w];
    let made = fire::<kernels_cuda::moe::topk_softmax>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let topk_idx = made.next().expect("`moe::topk_softmax` states `topk_idx`");
    let topk_w = made.next().expect("`moe::topk_softmax` states `topk_w`");
    (topk_idx, topk_w)
}

/// Generated for `moe::topk_sigmoid_bias_fp32` from the routine's own
/// signature (`kernels_cuda::moe::topk_sigmoid_bias_fp32`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn topk_sigmoid_bias_fp32(
    logits: &Val,
    correction_bias: &str,
    topk_idx: (Shape, DType),
    topk_w: (Shape, DType),
    normalize: bool,
    routed_scaling_factor: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params = vec![
        u32::from(normalize),
        routed_scaling_factor.to_bits(),
    ];
    let run_inputs = vec![logits.id];
    let run_weights = vec![correction_bias.to_string()];
    let run_outs = vec![topk_idx, topk_w];
    let made = fire::<kernels_cuda::moe::topk_sigmoid_bias_fp32>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let topk_idx = made.next().expect("`moe::topk_sigmoid_bias_fp32` states `topk_idx`");
    let topk_w = made.next().expect("`moe::topk_sigmoid_bias_fp32` states `topk_w`");
    (topk_idx, topk_w)
}

/// Generated for `moe::apply_per_expert_scale` from the routine's own
/// signature (`kernels_cuda::moe::apply_per_expert_scale`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn apply_per_expert_scale(
    topk_idx: &Val,
    topk_w: &Val,
    topk_w_out: (Shape, DType),
    per_expert_scale: &str,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = topk_idx.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![topk_idx.id, topk_w.id];
    let run_weights = vec![per_expert_scale.to_string()];
    let run_outs = vec![topk_w_out];
    let made = fire::<kernels_cuda::moe::apply_per_expert_scale>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::apply_per_expert_scale` states `topk_w`")
}

/// Generated for `moe::moe_gate_up_decode_gemv` from the routine's own
/// signature (`kernels_cuda::moe::moe_gate_up_decode_gemv`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn moe_gate_up_decode_gemv(
    topk_idx: &Val,
    norm_x: &Val,
    gate_up_base: &str,
    expert_gate_up: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = topk_idx.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![topk_idx.id, norm_x.id];
    let run_weights = vec![gate_up_base.to_string()];
    let run_outs = vec![expert_gate_up];
    let made = fire::<kernels_cuda::moe::moe_gate_up_decode_gemv>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::moe_gate_up_decode_gemv` states `expert_gate_up`")
}

/// Generated for `moe::moe_down_decode_gemv` from the routine's own
/// signature (`kernels_cuda::moe::moe_down_decode_gemv`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn moe_down_decode_gemv(
    topk_idx: &Val,
    expert_act: &Val,
    down_base: &str,
    expert_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = topk_idx.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![topk_idx.id, expert_act.id];
    let run_weights = vec![down_base.to_string()];
    let run_outs = vec![expert_out];
    let made = fire::<kernels_cuda::moe::moe_down_decode_gemv>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::moe_down_decode_gemv` states `expert_out`")
}

/// Generated for `moe::transpose_expert_scales_u8` from the routine's own
/// signature (`kernels_cuda::moe::transpose_expert_scales_u8`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn transpose_expert_scales_u8(
    t: &Trace,
    src: &str,
    dst: (Shape, DType),
    num_experts: i32,
    n: i32,
    k_groups: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = t.clone();
    let run_params = vec![num_experts as u32, n as u32, k_groups as u32];
    let run_inputs: Vec<ValueId> = Vec::new();
    let run_weights = vec![src.to_string()];
    let run_outs = vec![dst];
    let made = fire::<kernels_cuda::moe::transpose_expert_scales_u8>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::transpose_expert_scales_u8` states `dst`")
}

/// Generated for `moe::reorder_moe_aligned_output` from the routine's own
/// signature (`kernels_cuda::moe::reorder_moe_aligned_output`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn reorder_moe_aligned_output(
    aligned_out: &Val,
    sorted_route_ids: &Val,
    route_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = aligned_out.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![aligned_out.id, sorted_route_ids.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![route_out];
    let made = fire::<kernels_cuda::moe::reorder_moe_aligned_output>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::reorder_moe_aligned_output` states `route_out`")
}

/// Generated for `moe::moe_align_decode` from the routine's own signature
/// (`kernels_cuda::moe::moe_align_decode`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn moe_align_decode(
    topk_idx: &Val,
    sorted_route_ids: (Shape, DType),
    expert_ids: (Shape, DType),
    route_to_aligned_row: (Shape, DType),
    num_experts: i32,
    block_size: i32,
    max_blocks: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = topk_idx.t.clone();
    let run_params = vec![
        num_experts as u32,
        block_size as u32,
        max_blocks as u32,
    ];
    let run_inputs = vec![topk_idx.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![sorted_route_ids, expert_ids, route_to_aligned_row];
    let made = fire::<kernels_cuda::moe::moe_align_decode>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let sorted_route_ids = made.next().expect("`moe::moe_align_decode` states `sorted_route_ids`");
    let expert_ids = made.next().expect("`moe::moe_align_decode` states `expert_ids`");
    let route_to_aligned_row = made.next().expect("`moe::moe_align_decode` states `route_to_aligned_row`");
    (sorted_route_ids, expert_ids, route_to_aligned_row)
}

/// Generated for `moe::moe_bucket_exact` from the routine's own signature
/// (`kernels_cuda::moe::moe_bucket_exact`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn moe_bucket_exact(
    topk_idx: &Val,
    sorted_route_ids: (Shape, DType),
    route_to_sorted_row: (Shape, DType),
    counts_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = topk_idx.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![topk_idx.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![sorted_route_ids, route_to_sorted_row, counts_out];
    let made = fire::<kernels_cuda::moe::moe_bucket_exact>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let sorted_route_ids = made.next().expect("`moe::moe_bucket_exact` states `sorted_route_ids`");
    let route_to_sorted_row = made.next().expect("`moe::moe_bucket_exact` states `route_to_sorted_row`");
    let counts_out = made.next().expect("`moe::moe_bucket_exact` states `counts_out`");
    (sorted_route_ids, route_to_sorted_row, counts_out)
}

/// Generated for `moe::gather_moe_aligned_inputs` from the routine's own
/// signature (`kernels_cuda::moe::gather_moe_aligned_inputs`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn gather_moe_aligned_inputs(
    norm_x: &Val,
    sorted_route_ids: &Val,
    aligned_in: (Shape, DType),
    top_k: i32,
    tokens: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = norm_x.t.clone();
    let run_params = vec![top_k as u32, tokens as u32];
    let run_inputs = vec![norm_x.id, sorted_route_ids.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![aligned_in];
    let made = fire::<kernels_cuda::moe::gather_moe_aligned_inputs>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::gather_moe_aligned_inputs` states `aligned_in`")
}

/// Generated for `moe::token_batched_weighted_sum` from the routine's own
/// signature (`kernels_cuda::moe::token_batched_weighted_sum`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn token_batched_weighted_sum(
    out: (Shape, DType),
    src: &Val,
    weights: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src.id, weights.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::moe::token_batched_weighted_sum>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::token_batched_weighted_sum` states `out`")
}

/// Generated for `moe::token_batched_weighted_sum_add` from the routine's
/// own signature (`kernels_cuda::moe::token_batched_weighted_sum_add`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn token_batched_weighted_sum_add(
    src: &Val,
    weights: &Val,
    out: &Val,
    out_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src.id, weights.id, out.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out_out];
    let made = fire::<kernels_cuda::moe::token_batched_weighted_sum_add>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::token_batched_weighted_sum_add` states `out`")
}

/// Generated for `moe::add_moe_route_bias` from the routine's own signature
/// (`kernels_cuda::moe::add_moe_route_bias`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn add_moe_route_bias(
    out: &Val,
    out_out: (Shape, DType),
    bias: &str,
    topk_idx: &Val,
    out_stride: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = out.t.clone();
    let run_params = vec![out_stride as u32];
    let run_inputs = vec![out.id, topk_idx.id];
    let run_weights = vec![bias.to_string()];
    let run_outs = vec![out_out];
    let made = fire::<kernels_cuda::moe::add_moe_route_bias>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`moe::add_moe_route_bias` states `out`")
}

/// Generated for `norm::rmsnorm_strided_bf16` from the routine's own
/// signature (`kernels_cuda::norm::rmsnorm_strided_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_strided_bf16(
    x: &Val,
    weight: &str,
    y: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::norm::rmsnorm_strided_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm_strided_bf16` states `y`")
}

/// Generated for `norm::rmsnorm_bf16_with_fp16` from the routine's own
/// signature (`kernels_cuda::norm::rmsnorm_bf16_with_fp16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_bf16_with_fp16(
    x: &Val,
    weight: &str,
    y: (Shape, DType),
    y_fp16: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y, y_fp16];
    let made = fire::<kernels_cuda::norm::rmsnorm_bf16_with_fp16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let y = made.next().expect("`norm::rmsnorm_bf16_with_fp16` states `y`");
    let y_fp16 = made.next().expect("`norm::rmsnorm_bf16_with_fp16` states `y_fp16`");
    (y, y_fp16)
}

/// Generated for `norm::rmsnorm` from the routine's own signature
/// (`kernels_cuda::norm::rmsnorm`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm(
    x: &Val,
    weight: &str,
    y: (Shape, DType),
    per_head_dim: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![per_head_dim as u32, eps.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::norm::rmsnorm>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm` states `y`")
}

/// Generated for `norm::rmsnorm_gemma` from the routine's own signature
/// (`kernels_cuda::norm::rmsnorm_gemma`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_gemma(
    x: &Val,
    weight: &str,
    y: (Shape, DType),
    per_head_dim: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![per_head_dim as u32, eps.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::norm::rmsnorm_gemma>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm_gemma` states `y`")
}

/// Generated for `norm::rmsnorm_no_scale` from the routine's own signature
/// (`kernels_cuda::norm::rmsnorm_no_scale`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_no_scale(
    x: &Val,
    y: (Shape, DType),
    per_head_dim: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![per_head_dim as u32, eps.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::norm::rmsnorm_no_scale>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm_no_scale` states `y`")
}

/// Generated for `norm::rmsnorm_gated` from the routine's own signature
/// (`kernels_cuda::norm::rmsnorm_gated`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_gated(
    x: &Val,
    gate: &Val,
    weight: &str,
    y: (Shape, DType),
    per_head_dim: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![per_head_dim as u32, eps.to_bits()];
    let run_inputs = vec![x.id, gate.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::norm::rmsnorm_gated>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm_gated` states `y`")
}

/// Generated for `norm::rmsnorm_gated_fp32_in` from the routine's own
/// signature (`kernels_cuda::norm::rmsnorm_gated_fp32_in`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_gated_fp32_in(
    x: &Val,
    gate: &Val,
    weight: &str,
    y: (Shape, DType),
    eps: f32,
    per_head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits(), per_head_dim as u32];
    let run_inputs = vec![x.id, gate.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::norm::rmsnorm_gated_fp32_in>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm_gated_fp32_in` states `y`")
}

/// Generated for `norm::residual_add_rmsnorm` from the routine's own
/// signature (`kernels_cuda::norm::residual_add_rmsnorm`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn residual_add_rmsnorm(
    hidden: &Val,
    residual: &Val,
    weight: &str,
    norm_out: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = hidden.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![hidden.id, residual.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![norm_out];
    let made = fire::<kernels_cuda::norm::residual_add_rmsnorm>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::residual_add_rmsnorm` states `norm_out`")
}

/// Generated for `norm::rmsnorm_residual_add` from the routine's own
/// signature (`kernels_cuda::norm::rmsnorm_residual_add`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_residual_add(
    x: &Val,
    weight: &str,
    hidden: &Val,
    hidden_out: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![x.id, hidden.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![hidden_out];
    let made = fire::<kernels_cuda::norm::rmsnorm_residual_add>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::rmsnorm_residual_add` states `hidden`")
}

/// Generated for `norm::rmsnorm_residual_add_scale_rmsnorm_bf16` from the
/// routine's own signature
/// (`kernels_cuda::norm::rmsnorm_residual_add_scale_rmsnorm_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rmsnorm_residual_add_scale_rmsnorm_bf16(
    x: &Val,
    weight: &str,
    hidden: &Val,
    hidden_out: (Shape, DType),
    scale: f32,
    next_weight: &str,
    norm_out: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = x.t.clone();
    let run_params = vec![scale.to_bits(), eps.to_bits()];
    let run_inputs = vec![x.id, hidden.id];
    let run_weights = vec![weight.to_string(), next_weight.to_string()];
    let run_outs = vec![hidden_out, norm_out];
    let made = fire::<kernels_cuda::norm::rmsnorm_residual_add_scale_rmsnorm_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let hidden = made.next().expect("`norm::rmsnorm_residual_add_scale_rmsnorm_bf16` states `hidden`");
    let norm_out = made.next().expect("`norm::rmsnorm_residual_add_scale_rmsnorm_bf16` states `norm_out`");
    (hidden, norm_out)
}

/// Generated for `norm::add_bias` from the routine's own signature
/// (`kernels_cuda::norm::add_bias`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn add_bias(
    out: &Val,
    out_out: (Shape, DType),
    bias: &str,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = out.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![out.id];
    let run_weights = vec![bias.to_string()];
    let run_outs = vec![out_out];
    let made = fire::<kernels_cuda::norm::add_bias>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::add_bias` states `out`")
}

/// Generated for `norm::altup_predict` from the routine's own signature
/// (`kernels_cuda::norm::altup_predict`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn altup_predict(
    streams: &Val,
    coefs: &Val,
    predictions: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = streams.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![streams.id, coefs.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![predictions];
    let made = fire::<kernels_cuda::norm::altup_predict>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::altup_predict` states `predictions`")
}

/// Generated for `norm::altup_correct` from the routine's own signature
/// (`kernels_cuda::norm::altup_correct`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn altup_correct(
    predictions: &Val,
    activated: &Val,
    correction_coefs_plus_one: &Val,
    corrected: (Shape, DType),
    active_idx: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = predictions.t.clone();
    let run_params = vec![active_idx as u32];
    let run_inputs = vec![
        predictions.id,
        activated.id,
        correction_coefs_plus_one.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![corrected];
    let made = fire::<kernels_cuda::norm::altup_correct>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::altup_correct` states `corrected`")
}

/// Generated for `norm::compute_rms` from the routine's own signature
/// (`kernels_cuda::norm::compute_rms`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn compute_rms(
    reference: &Val,
    out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = reference.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![reference.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::norm::compute_rms>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::compute_rms` states `out`")
}

/// Generated for `norm::magnitude_rescale` from the routine's own signature
/// (`kernels_cuda::norm::magnitude_rescale`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn magnitude_rescale(
    x: &Val,
    x_out: (Shape, DType),
    target_rms: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id, target_rms.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![x_out];
    let made = fire::<kernels_cuda::norm::magnitude_rescale>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::magnitude_rescale` states `x`")
}

/// Generated for `norm::mean_streams` from the routine's own signature
/// (`kernels_cuda::norm::mean_streams`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn mean_streams(
    streams: &Val,
    out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = streams.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![streams.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::norm::mean_streams>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::mean_streams` states `out`")
}

/// Generated for `norm::altup_unpack_predict_coefs` from the routine's own
/// signature (`kernels_cuda::norm::altup_unpack_predict_coefs`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn altup_unpack_predict_coefs(
    in_bf16: &Val,
    out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = in_bf16.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![in_bf16.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::norm::altup_unpack_predict_coefs>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::altup_unpack_predict_coefs` states `out`")
}

/// Generated for `norm::altup_unpack_correct_coefs` from the routine's own
/// signature (`kernels_cuda::norm::altup_unpack_correct_coefs`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn altup_unpack_correct_coefs(
    in_bf16: &Val,
    out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = in_bf16.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![in_bf16.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::norm::altup_unpack_correct_coefs>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::altup_unpack_correct_coefs` states `out`")
}

/// Generated for `norm::tanh` from the routine's own signature
/// (`kernels_cuda::norm::tanh`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn tanh(
    x: &Val,
    x_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![x_out];
    let made = fire::<kernels_cuda::norm::tanh>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::tanh` states `x`")
}

/// Generated for `norm::residual_add` from the routine's own signature
/// (`kernels_cuda::norm::residual_add`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn residual_add(
    y: &Val,
    y_out: (Shape, DType),
    x: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = y.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![y.id, x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y_out];
    let made = fire::<kernels_cuda::norm::residual_add>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::residual_add` states `y`")
}

/// Generated for `norm::scalar_mul` from the routine's own signature
/// (`kernels_cuda::norm::scalar_mul`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn scalar_mul(
    x: &Val,
    x_out: (Shape, DType),
    s: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![s.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![x_out];
    let made = fire::<kernels_cuda::norm::scalar_mul>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::scalar_mul` states `x`")
}

/// Generated for `norm::hc_pre_postprocess` from the routine's own
/// signature (`kernels_cuda::norm::hc_pre_postprocess`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn hc_pre_postprocess(
    mixes: &Val,
    scale: &str,
    base: &str,
    residual: &Val,
    post_mix: (Shape, DType),
    comb_mix: (Shape, DType),
    layer_input: (Shape, DType),
    hc_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_iters: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = mixes.t.clone();
    let run_params = vec![
        hc_eps.to_bits(),
        hc_post_alpha.to_bits(),
        sinkhorn_iters as u32,
    ];
    let run_inputs = vec![mixes.id, residual.id];
    let run_weights = vec![scale.to_string(), base.to_string()];
    let run_outs = vec![post_mix, comb_mix, layer_input];
    let made = fire::<kernels_cuda::norm::hc_pre_postprocess>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let post_mix = made.next().expect("`norm::hc_pre_postprocess` states `post_mix`");
    let comb_mix = made.next().expect("`norm::hc_pre_postprocess` states `comb_mix`");
    let layer_input = made.next().expect("`norm::hc_pre_postprocess` states `layer_input`");
    (post_mix, comb_mix, layer_input)
}

/// Generated for `norm::hc_post` from the routine's own signature
/// (`kernels_cuda::norm::hc_post`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn hc_post(
    x: &Val,
    residual: &Val,
    post_mix: &Val,
    comb_mix: &Val,
    out_residual: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id, residual.id, post_mix.id, comb_mix.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out_residual];
    let made = fire::<kernels_cuda::norm::hc_post>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::hc_post` states `out_residual`")
}

/// Generated for `norm::hc_head_postprocess` from the routine's own
/// signature (`kernels_cuda::norm::hc_head_postprocess`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn hc_head_postprocess(
    mixes: &Val,
    scale: &str,
    base: &str,
    residual: &Val,
    out: (Shape, DType),
    hc_eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = mixes.t.clone();
    let run_params = vec![hc_eps.to_bits()];
    let run_inputs = vec![mixes.id, residual.id];
    let run_weights = vec![scale.to_string(), base.to_string()];
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::norm::hc_head_postprocess>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::hc_head_postprocess` states `out`")
}

/// Generated for `norm::hc_expand` from the routine's own signature
/// (`kernels_cuda::norm::hc_expand`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn hc_expand(
    input: &Val,
    output: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = input.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![input.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![output];
    let made = fire::<kernels_cuda::norm::hc_expand>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::hc_expand` states `output`")
}

/// Generated for `norm::hc_rmsnorm_to_f32` from the routine's own signature
/// (`kernels_cuda::norm::hc_rmsnorm_to_f32`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn hc_rmsnorm_to_f32(
    input: &Val,
    output: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = input.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![input.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![output];
    let made = fire::<kernels_cuda::norm::hc_rmsnorm_to_f32>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::hc_rmsnorm_to_f32` states `output`")
}

/// Generated for `norm::attn_sink_correction` from the routine's own
/// signature (`kernels_cuda::norm::attn_sink_correction`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn attn_sink_correction(
    out: &Val,
    out_out: (Shape, DType),
    lse: &Val,
    sink: &str,
    head_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = out.t.clone();
    let run_params = vec![head_dim as u32];
    let run_inputs = vec![out.id, lse.id];
    let run_weights = vec![sink.to_string()];
    let run_outs = vec![out_out];
    let made = fire::<kernels_cuda::norm::attn_sink_correction>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::attn_sink_correction` states `out`")
}

/// Generated for `norm::per_head_rmsnorm` from the routine's own signature
/// (`kernels_cuda::norm::per_head_rmsnorm`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn per_head_rmsnorm(
    q: &Val,
    q_out: (Shape, DType),
    head_dim: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q.t.clone();
    let run_params = vec![head_dim as u32, eps.to_bits()];
    let run_inputs = vec![q.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::norm::per_head_rmsnorm>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`norm::per_head_rmsnorm` states `q`")
}

/// Generated for `quant::cast_fp32_to` from the routine's own signature
/// (`kernels_cuda::quant::cast_fp32_to`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn cast_fp32_to(
    src_fp32: &Val,
    dst_bf16: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = src_fp32.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![src_fp32.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![dst_bf16];
    let made = fire::<kernels_cuda::quant::cast_fp32_to>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::cast_fp32_to` states `dst_bf16`")
}

/// Generated for `quant::scale_rows` from the routine's own signature
/// (`kernels_cuda::quant::scale_rows`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn scale_rows(
    buf_bf16: &Val,
    buf_bf16_out: (Shape, DType),
    l_bf16: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = buf_bf16.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![buf_bf16.id, l_bf16.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![buf_bf16_out];
    let made = fire::<kernels_cuda::quant::scale_rows>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::scale_rows` states `buf_bf16`")
}

/// Generated for `quant::bf16_to_fp16` from the routine's own signature
/// (`kernels_cuda::quant::bf16_to_fp16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn bf16_to_fp16(
    in_bf16: &Val,
    out_fp16: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = in_bf16.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![in_bf16.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out_fp16];
    let made = fire::<kernels_cuda::quant::bf16_to_fp16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::bf16_to_fp16` states `out_fp16`")
}

/// Generated for `quant::dequant_fp8_e4m3_to` from the routine's own
/// signature (`kernels_cuda::quant::dequant_fp8_e4m3_to`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dequant_fp8_e4m3_to(
    fp8_in: &Val,
    bf16_out: (Shape, DType),
    scale: f32,
    rows: i32,
    cols: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = fp8_in.t.clone();
    let run_params = vec![scale.to_bits(), rows as u32, cols as u32];
    let run_inputs = vec![fp8_in.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![bf16_out];
    let made = fire::<kernels_cuda::quant::dequant_fp8_e4m3_to>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::dequant_fp8_e4m3_to` states `bf16_out`")
}

/// Generated for `quant::dequant_fp8_e4m3_to_bf16_per_channel` from the
/// routine's own signature
/// (`kernels_cuda::quant::dequant_fp8_e4m3_to_bf16_per_channel`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dequant_fp8_e4m3_to_bf16_per_channel(
    fp8_in: &Val,
    bf16_out: (Shape, DType),
    scale_inv: &Val,
    rows: i32,
    cols: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = fp8_in.t.clone();
    let run_params = vec![rows as u32, cols as u32];
    let run_inputs = vec![fp8_in.id, scale_inv.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![bf16_out];
    let made = fire::<kernels_cuda::quant::dequant_fp8_e4m3_to_bf16_per_channel>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::dequant_fp8_e4m3_to_bf16_per_channel` states `bf16_out`")
}

/// Generated for `quant::dequant_fp8_e4m3_to_bf16_per_group` from the
/// routine's own signature
/// (`kernels_cuda::quant::dequant_fp8_e4m3_to_bf16_per_group`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dequant_fp8_e4m3_to_bf16_per_group(
    fp8_in: &Val,
    bf16_out: (Shape, DType),
    scales: &Val,
    group_size: i32,
    rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = fp8_in.t.clone();
    let run_params = vec![group_size as u32, rows as u32];
    let run_inputs = vec![fp8_in.id, scales.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![bf16_out];
    let made = fire::<kernels_cuda::quant::dequant_fp8_e4m3_to_bf16_per_group>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::dequant_fp8_e4m3_to_bf16_per_group` states `bf16_out`")
}

/// Generated for `quant::dequant_mxfp4_to` from the routine's own signature
/// (`kernels_cuda::quant::dequant_mxfp4_to`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dequant_mxfp4_to(
    packed: &Val,
    block_scale: &Val,
    out: (Shape, DType),
    out_dim: i32,
    in_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![out_dim as u32, in_dim as u32];
    let run_inputs = vec![packed.id, block_scale.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::quant::dequant_mxfp4_to>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::dequant_mxfp4_to` states `out`")
}

/// Generated for `quant::dequant_wna16_int4b8_to` from the routine's own
/// signature (`kernels_cuda::quant::dequant_wna16_int4b8_to`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn dequant_wna16_int4b8_to(
    packed: &Val,
    scale: &Val,
    out: (Shape, DType),
    group_size: i32,
    out_dim: i32,
    in_dim: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = packed.t.clone();
    let run_params = vec![group_size as u32, out_dim as u32, in_dim as u32];
    let run_inputs = vec![packed.id, scale.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::quant::dequant_wna16_int4b8_to>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::dequant_wna16_int4b8_to` states `out`")
}

/// Generated for `quant::mxfp4_scales_to_marlin_e8m0` from the routine's
/// own signature (`kernels_cuda::quant::mxfp4_scales_to_marlin_e8m0`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn mxfp4_scales_to_marlin_e8m0(
    raw: &Val,
    out: (Shape, DType),
    source_rows: i32,
    source_row_offset: i32,
    valid_rows: i32,
    source_stride_groups: i32,
    source_group_offset: i32,
    source_groups: i32,
    row_select: i32,
    selected_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = raw.t.clone();
    let run_params = vec![
        source_rows as u32,
        source_row_offset as u32,
        valid_rows as u32,
        source_stride_groups as u32,
        source_group_offset as u32,
        source_groups as u32,
        row_select as u32,
        selected_rows as u32,
    ];
    let run_inputs = vec![raw.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::quant::mxfp4_scales_to_marlin_e8m0>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::mxfp4_scales_to_marlin_e8m0` states `out`")
}

/// Generated for `quant::mxfp4_moe_gate_up_decode_bf16` from the routine's
/// own signature (`kernels_cuda::quant::mxfp4_moe_gate_up_decode_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn mxfp4_moe_gate_up_decode_bf16(
    topk_idx: &Val,
    act: &Val,
    _packed_bank: &str,
    gate_out: (Shape, DType),
    up_out: (Shape, DType),
    glu_limit: f32,
    glu_alpha: f32,
    ew: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = topk_idx.t.clone();
    let run_params = vec![glu_limit.to_bits(), glu_alpha.to_bits()];
    let run_inputs = vec![topk_idx.id, act.id, ew.id];
    let run_weights = vec![_packed_bank.to_string()];
    let run_outs = vec![gate_out, up_out];
    let made = fire::<kernels_cuda::quant::mxfp4_moe_gate_up_decode_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let gate_out = made.next().expect("`quant::mxfp4_moe_gate_up_decode_bf16` states `gate_out`");
    let up_out = made.next().expect("`quant::mxfp4_moe_gate_up_decode_bf16` states `up_out`");
    (gate_out, up_out)
}

/// Generated for `quant::mxfp4_moe_down_decode_bf16` from the routine's own
/// signature (`kernels_cuda::quant::mxfp4_moe_down_decode_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn mxfp4_moe_down_decode_bf16(
    topk_idx: &Val,
    act: &Val,
    _packed_bank: &str,
    out: (Shape, DType),
    ew: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = topk_idx.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![topk_idx.id, act.id, ew.id];
    let run_weights = vec![_packed_bank.to_string()];
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::quant::mxfp4_moe_down_decode_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::mxfp4_moe_down_decode_bf16` states `out`")
}

/// Generated for `quant::wna16_gate_up_decode_bf16` from the routine's own
/// signature (`kernels_cuda::quant::wna16_gate_up_decode_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn wna16_gate_up_decode_bf16(
    act: &Val,
    topk_idx: &Val,
    gate_packed_ptrs: &str,
    gate_scale_ptrs: &str,
    up_packed_ptrs: &str,
    up_scale_ptrs: &str,
    gate_out: (Shape, DType),
    up_out: (Shape, DType),
    group_size: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = act.t.clone();
    let run_params = vec![group_size as u32];
    let run_inputs = vec![act.id, topk_idx.id];
    let run_weights = vec![
        gate_packed_ptrs.to_string(),
        gate_scale_ptrs.to_string(),
        up_packed_ptrs.to_string(),
        up_scale_ptrs.to_string(),
    ];
    let run_outs = vec![gate_out, up_out];
    let made = fire::<kernels_cuda::quant::wna16_gate_up_decode_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let gate_out = made.next().expect("`quant::wna16_gate_up_decode_bf16` states `gate_out`");
    let up_out = made.next().expect("`quant::wna16_gate_up_decode_bf16` states `up_out`");
    (gate_out, up_out)
}

/// Generated for `quant::wna16_down_decode_bf16` from the routine's own
/// signature (`kernels_cuda::quant::wna16_down_decode_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn wna16_down_decode_bf16(
    act: &Val,
    topk_idx: &Val,
    down_packed_ptrs: &str,
    down_scale_ptrs: &str,
    out: (Shape, DType),
    group_size: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = act.t.clone();
    let run_params = vec![group_size as u32];
    let run_inputs = vec![act.id, topk_idx.id];
    let run_weights = vec![
        down_packed_ptrs.to_string(),
        down_scale_ptrs.to_string(),
    ];
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::quant::wna16_down_decode_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`quant::wna16_down_decode_bf16` states `out`")
}

/// Generated for `rope::rope_standard_table` from the routine's own
/// signature (`kernels_cuda::rope::rope_standard_table`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_standard_table(
    table: (Shape, DType),
    head_dim: i32,
    theta: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = positions.t.clone();
    let run_params = vec![head_dim as u32, theta.to_bits()];
    let run_inputs = vec![positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![table];
    let made = fire::<kernels_cuda::rope::rope_standard_table>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`rope::rope_standard_table` states `table`")
}

/// Generated for `rope::rope_bf16` from the routine's own signature
/// (`kernels_cuda::rope::rope_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    interleaved: bool,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        num_q_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        theta.to_bits(),
        u32::from(interleaved),
    ];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::rope_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::rope_bf16` states `q`");
    let k = made.next().expect("`rope::rope_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::rope_write_kv_bf16` from the routine's own
/// signature (`kernels_cuda::rope::rope_write_kv_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_write_kv_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    v: &Val,
    interleaved: bool,
    kvc: &Val,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    qo_indptr: &Val,
    row_valid: &Val,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q.t.clone();
    let run_params = vec![
        u32::from(interleaved),
        num_q_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        theta.to_bits(),
    ];
    let run_inputs = vec![
        q.id,
        k.id,
        v.id,
        kvc.id,
        qo_indptr.id,
        row_valid.id,
        positions.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::rope::rope_write_kv_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`rope::rope_write_kv_bf16` states `q`")
}

/// Generated for `rope::qk_rmsnorm_rope_bf16` from the routine's own
/// signature (`kernels_cuda::rope::qk_rmsnorm_rope_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn qk_rmsnorm_rope_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    q_weight: &str,
    k_weight: &str,
    head_dim: i32,
    theta: f32,
    eps: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![head_dim as u32, theta.to_bits(), eps.to_bits()];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights = vec![q_weight.to_string(), k_weight.to_string()];
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::qk_rmsnorm_rope_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::qk_rmsnorm_rope_bf16` states `q`");
    let k = made.next().expect("`rope::qk_rmsnorm_rope_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::qk_rmsnorm_rope_bf16_devwin` from the routine's own
/// signature (`kernels_cuda::rope::qk_rmsnorm_rope_bf16_devwin`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn qk_rmsnorm_rope_bf16_devwin(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    q_weight: &str,
    k_weight: &str,
    head_dim: i32,
    theta: f32,
    eps: f32,
    n_max: i32,
    positions: &Val,
    win_start: i32,
    win_len: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        head_dim as u32,
        theta.to_bits(),
        eps.to_bits(),
        n_max as u32,
        win_start as u32,
        win_len as u32,
    ];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights = vec![q_weight.to_string(), k_weight.to_string()];
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::qk_rmsnorm_rope_bf16_devwin>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::qk_rmsnorm_rope_bf16_devwin` states `q`");
    let k = made.next().expect("`rope::qk_rmsnorm_rope_bf16_devwin` states `k`");
    (q, k)
}

/// Generated for `rope::qk_rmsnorm_mrope_bf16` from the routine's own
/// signature (`kernels_cuda::rope::qk_rmsnorm_mrope_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn qk_rmsnorm_mrope_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    q_weight: &str,
    k_weight: &str,
    mrope_section_t: i32,
    mrope_section_h: i32,
    mrope_section_w: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        mrope_section_t as u32,
        mrope_section_h as u32,
        mrope_section_w as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        theta.to_bits(),
        eps.to_bits(),
    ];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights = vec![q_weight.to_string(), k_weight.to_string()];
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::qk_rmsnorm_mrope_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::qk_rmsnorm_mrope_bf16` states `q`");
    let k = made.next().expect("`rope::qk_rmsnorm_mrope_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::qk_rmsnorm_rope_bf16_rounded` from the routine's
/// own signature (`kernels_cuda::rope::qk_rmsnorm_rope_bf16_rounded`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn qk_rmsnorm_rope_bf16_rounded(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    q_weight: &str,
    k_weight: &str,
    head_dim: i32,
    theta: f32,
    eps: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![head_dim as u32, theta.to_bits(), eps.to_bits()];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights = vec![q_weight.to_string(), k_weight.to_string()];
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::qk_rmsnorm_rope_bf16_rounded>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::qk_rmsnorm_rope_bf16_rounded` states `q`");
    let k = made.next().expect("`rope::qk_rmsnorm_rope_bf16_rounded` states `k`");
    (q, k)
}

/// Generated for `rope::q_rmsnorm_rope_bf16_rounded` from the routine's own
/// signature (`kernels_cuda::rope::q_rmsnorm_rope_bf16_rounded`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn q_rmsnorm_rope_bf16_rounded(
    q: &Val,
    q_out: (Shape, DType),
    q_weight: &str,
    head_dim: i32,
    theta: f32,
    eps: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q.t.clone();
    let run_params = vec![head_dim as u32, theta.to_bits(), eps.to_bits()];
    let run_inputs = vec![q.id, positions.id];
    let run_weights = vec![q_weight.to_string()];
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::rope::q_rmsnorm_rope_bf16_rounded>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`rope::q_rmsnorm_rope_bf16_rounded` states `q`")
}

/// Generated for `rope::rope_yarn_bf16` from the routine's own signature
/// (`kernels_cuda::rope::rope_yarn_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_yarn_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        factor.to_bits(),
        low_freq_factor.to_bits(),
        high_freq_factor.to_bits(),
        original_max_position as u32,
        num_q_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        theta.to_bits(),
    ];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::rope_yarn_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::rope_yarn_bf16` states `q`");
    let k = made.next().expect("`rope::rope_yarn_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::rope_yarn_original_bf16` from the routine's own
/// signature (`kernels_cuda::rope::rope_yarn_original_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_yarn_original_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    head_dim: i32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: i32,
    interleaved: bool,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        head_dim as u32,
        theta.to_bits(),
        factor.to_bits(),
        beta_fast.to_bits(),
        beta_slow.to_bits(),
        attention_factor.to_bits(),
        original_max_position as u32,
        u32::from(interleaved),
    ];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::rope_yarn_original_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::rope_yarn_original_bf16` states `q`");
    let k = made.next().expect("`rope::rope_yarn_original_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::rope_partial_bf16` from the routine's own signature
/// (`kernels_cuda::rope::rope_partial_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_partial_bf16(
    q: &Val,
    q_out: (Shape, DType),
    k: &Val,
    k_out: (Shape, DType),
    rotary_dim: i32,
    head_dim: i32,
    theta: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = q.t.clone();
    let run_params = vec![
        rotary_dim as u32,
        head_dim as u32,
        theta.to_bits(),
    ];
    let run_inputs = vec![q.id, k.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, k_out];
    let made = fire::<kernels_cuda::rope::rope_partial_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::rope_partial_bf16` states `q`");
    let k = made.next().expect("`rope::rope_partial_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::rope_partial_q_bf16` from the routine's own
/// signature (`kernels_cuda::rope::rope_partial_q_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_partial_q_bf16(
    q: &Val,
    q_out: (Shape, DType),
    rotary_dim: i32,
    head_dim: i32,
    theta: f32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q.t.clone();
    let run_params = vec![
        rotary_dim as u32,
        head_dim as u32,
        theta.to_bits(),
    ];
    let run_inputs = vec![q.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::rope::rope_partial_q_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`rope::rope_partial_q_bf16` states `q`")
}

/// Generated for `rope::rope_partial_last_bf16` from the routine's own
/// signature (`kernels_cuda::rope::rope_partial_last_bf16`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_partial_last_bf16(
    q: (Shape, DType),
    k: (Shape, DType),
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    interleaved: bool,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_max_position: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = positions.t.clone();
    let run_params = vec![
        head_dim as u32,
        rotary_dim as u32,
        theta.to_bits(),
        u32::from(interleaved),
        yarn_factor.to_bits(),
        yarn_beta_fast.to_bits(),
        yarn_beta_slow.to_bits(),
        yarn_original_max_position as u32,
    ];
    let run_inputs = vec![positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q, k];
    let made = fire::<kernels_cuda::rope::rope_partial_last_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`rope::rope_partial_last_bf16` states `q`");
    let k = made.next().expect("`rope::rope_partial_last_bf16` states `k`");
    (q, k)
}

/// Generated for `rope::rope_partial_last_q_bf16` from the routine's own
/// signature (`kernels_cuda::rope::rope_partial_last_q_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn rope_partial_last_q_bf16(
    q: &Val,
    q_out: (Shape, DType),
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    interleaved: bool,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_max_position: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q.t.clone();
    let run_params = vec![
        head_dim as u32,
        rotary_dim as u32,
        theta.to_bits(),
        u32::from(interleaved),
        yarn_factor.to_bits(),
        yarn_beta_fast.to_bits(),
        yarn_beta_slow.to_bits(),
        yarn_original_max_position as u32,
    ];
    let run_inputs = vec![q.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out];
    let made = fire::<kernels_cuda::rope::rope_partial_last_q_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`rope::rope_partial_last_q_bf16` states `q`")
}

/// Generated for `sample::lm_head_gemv_argmax_int8` from the routine's own
/// signature (`kernels_cuda::sample::lm_head_gemv_argmax_int8`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn lm_head_gemv_argmax_int8(
    hidden_states: &Val,
    lm_head_weight: &str,
    scale_inv: &str,
    token_ids: (Shape, DType),
    vocab: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = hidden_states.t.clone();
    let run_params = vec![vocab as u32];
    let run_inputs = vec![hidden_states.id];
    let run_weights = vec![lm_head_weight.to_string(), scale_inv.to_string()];
    let run_outs = vec![token_ids];
    let made = fire::<kernels_cuda::sample::lm_head_gemv_argmax_int8>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`sample::lm_head_gemv_argmax_int8` states `token_ids`")
}

/// Generated for `ssm::causal_conv1d_update_batched` from the routine's own
/// signature (`kernels_cuda::ssm::causal_conv1d_update_batched`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn causal_conv1d_update_batched(
    x: &Val,
    weight: &str,
    bias: Option<&str>,
    y: (Shape, DType),
    c: i32,
    k: i32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![c as u32, k as u32];
    let run_inputs = vec![x.id, rsv.id];
    let mut run_weights = Vec::new();
    run_weights.push(weight.to_string());
    if let Some(w) = bias {
        run_weights.push(w.to_string());
    }
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::causal_conv1d_update_batched>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::causal_conv1d_update_batched` states `y`")
}

/// Generated for `ssm::causal_conv1d_prefill_batched` from the routine's
/// own signature (`kernels_cuda::ssm::causal_conv1d_prefill_batched`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn causal_conv1d_prefill_batched(
    x: &Val,
    weight: &str,
    bias: Option<&str>,
    y: (Shape, DType),
    c: i32,
    k: i32,
    rsv: &Val,
    write_state: bool,
    qo_indptr: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![c as u32, k as u32, u32::from(write_state)];
    let run_inputs = vec![x.id, rsv.id, qo_indptr.id];
    let mut run_weights = Vec::new();
    run_weights.push(weight.to_string());
    if let Some(w) = bias {
        run_weights.push(w.to_string());
    }
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::causal_conv1d_prefill_batched>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::causal_conv1d_prefill_batched` states `y`")
}

/// Generated for `ssm::bf16_to_fp32` from the routine's own signature
/// (`kernels_cuda::ssm::bf16_to_fp32`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn bf16_to_fp32(
    x: &Val,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::bf16_to_fp32>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::bf16_to_fp32` states `y`")
}

/// Generated for `ssm::fp32_to_bf16` from the routine's own signature
/// (`kernels_cuda::ssm::fp32_to_bf16`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn fp32_to_bf16(
    x: &Val,
    y: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::fp32_to_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::fp32_to_bf16` states `y`")
}

/// Generated for `ssm::repeat_interleave_heads_fp32` from the routine's own
/// signature (`kernels_cuda::ssm::repeat_interleave_heads_fp32`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn repeat_interleave_heads_fp32(
    in_: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    d: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = in_.t.clone();
    let run_params = vec![k_h as u32, v_h as u32, d as u32];
    let run_inputs = vec![in_.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::repeat_interleave_heads_fp32>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::repeat_interleave_heads_fp32` states `out`")
}

/// Generated for `ssm::l2norm_scale_bf16_to_fp32` from the routine's own
/// signature (`kernels_cuda::ssm::l2norm_scale_bf16_to_fp32`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn l2norm_scale_bf16_to_fp32(
    x: &Val,
    y: (Shape, DType),
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits()];
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::l2norm_scale_bf16_to_fp32>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::l2norm_scale_bf16_to_fp32` states `y`")
}

/// Generated for `ssm::kda_gate_beta` from the routine's own signature
/// (`kernels_cuda::ssm::kda_gate_beta`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn kda_gate_beta(
    raw_g: &Val,
    raw_beta: &Val,
    a_log: &str,
    dt_bias: &str,
    gate_out: (Shape, DType),
    beta_out: (Shape, DType),
    d: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = raw_g.t.clone();
    let run_params = vec![d as u32];
    let run_inputs = vec![raw_g.id, raw_beta.id];
    let run_weights = vec![a_log.to_string(), dt_bias.to_string()];
    let run_outs = vec![gate_out, beta_out];
    let made = fire::<kernels_cuda::ssm::kda_gate_beta>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let gate_out = made.next().expect("`ssm::kda_gate_beta` states `gate_out`");
    let beta_out = made.next().expect("`ssm::kda_gate_beta` states `beta_out`");
    (gate_out, beta_out)
}

/// Generated for `ssm::kda_o_norm_gated` from the routine's own signature
/// (`kernels_cuda::ssm::kda_o_norm_gated`); the statement records through
/// [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn kda_o_norm_gated(
    o: &Val,
    g: &Val,
    weight: &str,
    out: (Shape, DType),
    h: i32,
    d: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = o.t.clone();
    let run_params = vec![h as u32, d as u32, eps.to_bits()];
    let run_inputs = vec![o.id, g.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::kda_o_norm_gated>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::kda_o_norm_gated` states `out`")
}

/// Generated for `ssm::kda_recurrent_step_batched` from the routine's own
/// signature (`kernels_cuda::ssm::kda_recurrent_step_batched`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn kda_recurrent_step_batched(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    gate: &Val,
    beta: &Val,
    out: (Shape, DType),
    h: i32,
    d: i32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![h as u32, d as u32];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        gate.id,
        beta.id,
        rsv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::kda_recurrent_step_batched>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::kda_recurrent_step_batched` states `out`")
}

/// Generated for `ssm::kda_prefill_batched` from the routine's own
/// signature (`kernels_cuda::ssm::kda_prefill_batched`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn kda_prefill_batched(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    gate: &Val,
    beta: &Val,
    out: (Shape, DType),
    h: i32,
    d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![h as u32, d as u32];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        gate.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::kda_prefill_batched>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::kda_prefill_batched` states `out`")
}

/// Generated for `ssm::nemotron_prepare_mamba_params` from the routine's
/// own signature (`kernels_cuda::ssm::nemotron_prepare_mamba_params`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn nemotron_prepare_mamba_params(
    t: &Trace,
    a_log: &str,
    d: &str,
    dt_bias: &str,
    a: (Shape, DType),
    d_f32: (Shape, DType),
    dt_bias_f32: (Shape, DType),
    num_heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = t.clone();
    let run_params = vec![num_heads as u32];
    let run_inputs: Vec<ValueId> = Vec::new();
    let run_weights = vec![
        a_log.to_string(),
        d.to_string(),
        dt_bias.to_string(),
    ];
    let run_outs = vec![a, d_f32, dt_bias_f32];
    let made = fire::<kernels_cuda::ssm::nemotron_prepare_mamba_params>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let a = made.next().expect("`ssm::nemotron_prepare_mamba_params` states `a`");
    let d_f32 = made.next().expect("`ssm::nemotron_prepare_mamba_params` states `d_f32`");
    let dt_bias_f32 = made.next().expect("`ssm::nemotron_prepare_mamba_params` states `dt_bias_f32`");
    (a, d_f32, dt_bias_f32)
}

/// Generated for `ssm::nemotron_prepare_mamba_dt_da` from the routine's own
/// signature (`kernels_cuda::ssm::nemotron_prepare_mamba_dt_da`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn nemotron_prepare_mamba_dt_da(
    dt: &Val,
    a: &Val,
    dt_bias: &Val,
    dt_out: (Shape, DType),
    da_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = dt.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![dt.id, a.id, dt_bias.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![dt_out, da_out];
    let made = fire::<kernels_cuda::ssm::nemotron_prepare_mamba_dt_da>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let dt_out = made.next().expect("`ssm::nemotron_prepare_mamba_dt_da` states `dt_out`");
    let da_out = made.next().expect("`ssm::nemotron_prepare_mamba_dt_da` states `da_out`");
    (dt_out, da_out)
}

/// Generated for `ssm::zamba_rmsnorm_gated` from the routine's own
/// signature (`kernels_cuda::ssm::zamba_rmsnorm_gated`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn zamba_rmsnorm_gated(
    x: &Val,
    gate: &Val,
    weight: &str,
    y: (Shape, DType),
    n_groups: i32,
    eps: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![n_groups as u32, eps.to_bits()];
    let run_inputs = vec![x.id, gate.id];
    let run_weights = vec![weight.to_string()];
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::zamba_rmsnorm_gated>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::zamba_rmsnorm_gated` states `y`")
}

/// Generated for `ssm::nemotron_mamba_split_bf16` from the routine's own
/// signature (`kernels_cuda::ssm::nemotron_mamba_split_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn nemotron_mamba_split_bf16(
    projected: &Val,
    gate: (Shape, DType),
    conv_in: (Shape, DType),
    dt: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = projected.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![projected.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![gate, conv_in, dt];
    let made = fire::<kernels_cuda::ssm::nemotron_mamba_split_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    let mut made = made.into_iter();
    let gate = made.next().expect("`ssm::nemotron_mamba_split_bf16` states `gate`");
    let conv_in = made.next().expect("`ssm::nemotron_mamba_split_bf16` states `conv_in`");
    let dt = made.next().expect("`ssm::nemotron_mamba_split_bf16` states `dt`");
    (gate, conv_in, dt)
}

/// Generated for `ssm::nemotron_mamba_ssm_batched_bf16` from the routine's
/// own signature (`kernels_cuda::ssm::nemotron_mamba_ssm_batched_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn nemotron_mamba_ssm_batched_bf16(
    conv_out: &Val,
    dt_precomputed: &Val,
    dt: &Val,
    a: &Val,
    d: &Val,
    dt_bias: &Val,
    da_precomputed: &Val,
    y: (Shape, DType),
    num_heads: i32,
    head_dim: i32,
    state_size: i32,
    n_groups: i32,
    conv_dim: i32,
    rsv: &Val,
    qo_indptr: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = conv_out.t.clone();
    let run_params = vec![
        num_heads as u32,
        head_dim as u32,
        state_size as u32,
        n_groups as u32,
        conv_dim as u32,
    ];
    let run_inputs = vec![
        conv_out.id,
        dt_precomputed.id,
        dt.id,
        a.id,
        d.id,
        dt_bias.id,
        da_precomputed.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let made = fire::<kernels_cuda::ssm::nemotron_mamba_ssm_batched_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::nemotron_mamba_ssm_batched_bf16` states `y`")
}

/// Generated for `ssm::build_nemotron_moe_ptrs_decode_batched_bf16` from
/// the routine's own signature
/// (`kernels_cuda::ssm::build_nemotron_moe_ptrs_decode_batched_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn build_nemotron_moe_ptrs_decode_batched_bf16(
    topk_idx: &Val,
    topk_w: &Val,
    norm_x: &Val,
    top_k: i32,
    hidden: i32,
    intermediate: i32,
    banks: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = topk_idx.t.clone();
    let run_params = vec![top_k as u32, hidden as u32, intermediate as u32];
    let run_inputs = vec![topk_idx.id, topk_w.id, norm_x.id, banks.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::ssm::build_nemotron_moe_ptrs_decode_batched_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`ssm::build_nemotron_moe_ptrs_decode_batched_bf16` states no result");
}

/// Generated for `ssm::build_nemotron_moe_ptrs_aligned_bf16` from the
/// routine's own signature
/// (`kernels_cuda::ssm::build_nemotron_moe_ptrs_aligned_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
pub fn build_nemotron_moe_ptrs_aligned_bf16(
    expert_ids: &Val,
    aligned_in: &Val,
    max_blocks: i32,
    block_size: i32,
    hidden: i32,
    intermediate: i32,
    banks: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = expert_ids.t.clone();
    let run_params = vec![
        max_blocks as u32,
        block_size as u32,
        hidden as u32,
        intermediate as u32,
    ];
    let run_inputs = vec![expert_ids.id, aligned_in.id, banks.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let made = fire::<kernels_cuda::ssm::build_nemotron_moe_ptrs_aligned_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    assert!(made.is_empty(), "`ssm::build_nemotron_moe_ptrs_aligned_bf16` states no result");
}

/// Generated for `ssm::chunk_gated_delta_prefill_batched` from the
/// routine's own signature
/// (`kernels_cuda::ssm::chunk_gated_delta_prefill_batched`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunk_gated_delta_prefill_batched(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    write_state: bool,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        u32::from(write_state),
    ];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::chunk_gated_delta_prefill_batched>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::chunk_gated_delta_prefill_batched` states `out`")
}

/// Generated for `ssm::chunk_gated_delta_prefill_batched_state_bf16` from
/// the routine's own signature
/// (`kernels_cuda::ssm::chunk_gated_delta_prefill_batched_state_bf16`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunk_gated_delta_prefill_batched_state_bf16(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    write_state: bool,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        u32::from(write_state),
    ];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::chunk_gated_delta_prefill_batched_state_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::chunk_gated_delta_prefill_batched_state_bf16` states `out`")
}

/// Generated for `ssm::chunk_gated_delta_prefill_batched_cached` from the
/// routine's own signature
/// (`kernels_cuda::ssm::chunk_gated_delta_prefill_batched_cached`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn chunk_gated_delta_prefill_batched_cached(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    v_h: i32,
    k_d: i32,
    v_d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    write_state: bool,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![
        v_h as u32,
        k_d as u32,
        v_d as u32,
        u32::from(write_state),
    ];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::chunk_gated_delta_prefill_batched_cached>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::chunk_gated_delta_prefill_batched_cached` states `out`")
}

/// Generated for `ssm::chunk_gated_delta_prefill_batched_cached_state_bf16`
/// from the routine's own signature
/// (`kernels_cuda::ssm::chunk_gated_delta_prefill_batched_cached_state_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    v_h: i32,
    k_d: i32,
    v_d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    write_state: bool,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![
        v_h as u32,
        k_d as u32,
        v_d as u32,
        u32::from(write_state),
    ];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::chunk_gated_delta_prefill_batched_cached_state_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::chunk_gated_delta_prefill_batched_cached_state_bf16` states `out`")
}

/// Generated for `ssm::recurrent_gated_delta_step_batched_gqa_state_bf16`
/// from the routine's own signature
/// (`kernels_cuda::ssm::recurrent_gated_delta_step_batched_gqa_state_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    q_norm_kh: &Val,
    k_norm_kh: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    r: i32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm_kh.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        r as u32,
    ];
    let run_inputs = vec![
        q_norm_kh.id,
        k_norm_kh.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::recurrent_gated_delta_step_batched_gqa_state_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::recurrent_gated_delta_step_batched_gqa_state_bf16` states `out`")
}

/// Generated for `ssm::recurrent_gated_delta_step_batched` from the
/// routine's own signature
/// (`kernels_cuda::ssm::recurrent_gated_delta_step_batched`); the statement
/// records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn recurrent_gated_delta_step_batched(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    v_h: i32,
    k_d: i32,
    v_d: i32,
    r: i32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![v_h as u32, k_d as u32, v_d as u32, r as u32];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::recurrent_gated_delta_step_batched>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::recurrent_gated_delta_step_batched` states `out`")
}

/// Generated for `ssm::recurrent_gated_delta_step_batched_state_bf16` from
/// the routine's own signature
/// (`kernels_cuda::ssm::recurrent_gated_delta_step_batched_state_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn recurrent_gated_delta_step_batched_state_bf16(
    q_norm: &Val,
    k_norm: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    v_h: i32,
    k_d: i32,
    v_d: i32,
    r: i32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm.t.clone();
    let run_params = vec![v_h as u32, k_d as u32, v_d as u32, r as u32];
    let run_inputs = vec![
        q_norm.id,
        k_norm.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::recurrent_gated_delta_step_batched_state_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::recurrent_gated_delta_step_batched_state_bf16` states `out`")
}

/// Generated for `ssm::recurrent_gated_delta_step_batched_gqa` from the
/// routine's own signature
/// (`kernels_cuda::ssm::recurrent_gated_delta_step_batched_gqa`); the
/// statement records through [`crate::fire::fire`], one argument per mark.
#[must_use]
pub fn recurrent_gated_delta_step_batched_gqa(
    q_norm_kh: &Val,
    k_norm_kh: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    r: i32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm_kh.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        r as u32,
    ];
    let run_inputs = vec![
        q_norm_kh.id,
        k_norm_kh.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::recurrent_gated_delta_step_batched_gqa>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::recurrent_gated_delta_step_batched_gqa` states `out`")
}

/// Generated for `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa`
/// from the routine's own signature
/// (`kernels_cuda::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    q_norm_kh: &Val,
    k_norm_kh: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    write_state: bool,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm_kh.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        u32::from(write_state),
    ];
    let run_inputs = vec![
        q_norm_kh.id,
        k_norm_kh.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa` states `out`")
}

/// Generated for
/// `ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16` from
/// the routine's own signature
/// (`kernels_cuda::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16`);
/// the statement records through [`crate::fire::fire`], one argument per
/// mark.
#[must_use]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    q_norm_kh: &Val,
    k_norm_kh: &Val,
    v: &Val,
    g_log: &Val,
    beta: &Val,
    out: (Shape, DType),
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    rsv: &Val,
    qo_indptr: &Val,
    write_state: bool,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = q_norm_kh.t.clone();
    let run_params = vec![
        k_h as u32,
        v_h as u32,
        k_d as u32,
        v_d as u32,
        u32::from(write_state),
    ];
    let run_inputs = vec![
        q_norm_kh.id,
        k_norm_kh.id,
        v.id,
        g_log.id,
        beta.id,
        rsv.id,
        qo_indptr.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let made = fire::<kernels_cuda::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16>(&t, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: Vec::new(),
    });
    made.into_iter().next().expect("`ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16` states `out`")
}
