//! GENERATED — do not edit. One named `pub fn` per traced `#[routine]` in
//! `crates/kernels-metal/src`, named by the routine's own name, in
//! sorted-file then source order (design-no-ask §10, B4-gen step 6 — the
//! metal half; vulkan and wgpu execute the same statements off their
//! pinned-equal tables, so this is the one generated surface for all
//! three shader planes).
//!
//! The generator is `tests/generator/mod.rs`;
//! `cargo test -p model-dsl --test wrappers_are_current` refuses a stale
//! file and `UPDATE_WRAPPERS=1` rewrites it.
//!
//! Every mark is one argument — runtime streams and views included; a
//! wrapper here mints NOTHING in secret. Two readings are this plane's
//! own. The SYMBOL is the instantiated entrypoint the drivers resolve by
//! census stem: where the routine's body names one literal the wrapper
//! states it verbatim, and where the body composes an instantiation
//! point the symbol is the caller's first argument, checked against the
//! routine by [`crate::fire::fire_at`]. And a trailing `Const<i32>`
//! named `rows` is the FIRE's row extent by this plane's convention: the
//! wrapper takes no argument for it, writes the zero placeholder, and
//! splices the first operand's row axis (`rows_of`) — exactly what every
//! hand statement recorded. A result whose routine states an `out(..)`
//! rule is derived at trace time through
//! [`model_ir::kernels::out_shape`]; an `Unstated` result stays a
//! `(Shape, DType)` argument. Trailing `layer` and `state` are the
//! statement's tags, uniformly.
#![cfg_attr(rustfmt, rustfmt::skip)]

// The prelude is fixed while the surface below is generated from another
// crate's tree, so any one regeneration may leave part of it unused.
#![allow(unused_imports)]

use ::kernels::{OutRule, OutWidth};
use model_ir::trace::{DType, Shape, StateRef, ValueId};

use super::{ruled_out, rows_of};
use crate::fire::{Call, fire_at};
use crate::{Trace, Val};

/// Generated for `split_qkv_bf16` from the routine's own signature
/// (`kernels_metal::attn::split_qkv_bf16`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn split_qkv_bf16(
    packed: &Val,
    q_width: u32,
    kv_width: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = packed.t.clone();
    let run_params = vec![q_width, kv_width, 0];
    let run_inputs = vec![packed.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "split_qkv_bf16", OutRule::Shaped { rows_of: 0, width: OutWidth::Param { of: 0 } }, &run_inputs, &run_params),
        ruled_out(&t, "split_qkv_bf16", OutRule::Shaped { rows_of: 0, width: OutWidth::Param { of: 1 } }, &run_inputs, &run_params),
        ruled_out(&t, "split_qkv_bf16", OutRule::Shaped { rows_of: 0, width: OutWidth::Param { of: 1 } }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(2, Shape(vec![rows_of(packed)]))];
    let made = fire_at::<kernels_metal::attn::split_qkv_bf16>(&t, "split_qkv_bf16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let q = made.next().expect("`split_qkv_bf16` states `q`");
    let k = made.next().expect("`split_qkv_bf16` states `k`");
    let v = made.next().expect("`split_qkv_bf16` states `v`");
    (q, k, v)
}

/// Generated for `gate_bfloat16` from the routine's own signature
/// (`kernels_metal::attn::gate`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gate(
    attn: &Val,
    gate: &Val,
    row_stride: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = attn.t.clone();
    let run_params = vec![row_stride as u32, 0];
    let run_inputs = vec![attn.id, gate.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "gate_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(1, Shape(vec![rows_of(attn)]))];
    let made = fire_at::<kernels_metal::attn::gate>(&t, "gate_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gate_bfloat16` states `attn`")
}

/// Generated for `q_gate_split_bfloat16` from the routine's own signature
/// (`kernels_metal::attn::q_gate_split`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn q_gate_split(
    qg: &Val,
    q_out: (Shape, DType),
    gate_out: (Shape, DType),
    head_dim: i32,
    qg_row_stride: i32,
    out_row_stride: i32,
    q_heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = qg.t.clone();
    let run_params = vec![
        head_dim as u32,
        qg_row_stride as u32,
        out_row_stride as u32,
        q_heads as u32,
        0,
    ];
    let run_inputs = vec![qg.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![q_out, gate_out];
    let run_extents = vec![(4, Shape(vec![rows_of(qg)]))];
    let made = fire_at::<kernels_metal::attn::q_gate_split>(&t, "q_gate_split_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let q_out = made.next().expect("`q_gate_split_bfloat16` states `q_out`");
    let gate_out = made.next().expect("`q_gate_split_bfloat16` states `gate_out`");
    (q_out, gate_out)
}

/// Generated for `kv_append_bfloat16` from the routine's own signature
/// (`kernels_metal::attn::kv_append`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
pub fn kv_append(
    k_new: &Val,
    v_new: &Val,
    head_dim: i32,
    heads: i32,
    kvc: &Val,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = k_new.t.clone();
    let run_params = vec![head_dim as u32, heads as u32];
    let run_inputs = vec![k_new.id, v_new.id, kvc.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::kv_append>(&t, "kv_append_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    assert!(made.is_empty(), "`kv_append_bfloat16` states no result");
}

/// Generated for `kv_append_paged_bfloat16` from the routine's own
/// signature (`kernels_metal::attn::kv_append_paged`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
pub fn kv_append_paged(
    k_new: &Val,
    v_new: &Val,
    head_dim: i32,
    n_kv_heads: i32,
    kvc: &Val,
    tokens: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) {
    let t = k_new.t.clone();
    let run_params = vec![head_dim as u32, n_kv_heads as u32, tokens as u32];
    let run_inputs = vec![k_new.id, v_new.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs: Vec<(Shape, DType)> = Vec::new();
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::kv_append_paged>(&t, "kv_append_paged_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    assert!(made.is_empty(), "`kv_append_paged_bfloat16` states no result");
}

/// Generated for `logit_softcap_bfloat16` from the routine's own signature
/// (`kernels_metal::attn::logit_softcap`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn logit_softcap(
    logits: &Val,
    cap: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = logits.t.clone();
    let run_params = vec![cap.to_bits()];
    let run_inputs = vec![logits.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "logit_softcap_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::logit_softcap>(&t, "logit_softcap_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`logit_softcap_bfloat16` states `out`")
}

/// Generated for `sdpa_paged_decode`'s instantiations from the routine's
/// own signature (`kernels_metal::attn::sdpa_paged_decode`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_decode(
    symbol: &str,
    queries: &Val,
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    split: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        0,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
        split.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(queries)]))];
    let made = fire_at::<kernels_metal::attn::sdpa_paged_decode>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_decode` states `out`")
}

/// Generated for `sdpa_paged_decode_sink`'s instantiations from the
/// routine's own signature (`kernels_metal::attn::sdpa_paged_decode_sink`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_decode_sink(
    symbol: &str,
    queries: &Val,
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    sinks: &str,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    split: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        0,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
        split.id,
    ];
    let run_weights = vec![sinks.to_string()];
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(queries)]))];
    let made = fire_at::<kernels_metal::attn::sdpa_paged_decode_sink>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_decode_sink` states `out`")
}

/// Generated for `sdpa_paged_tiled`'s instantiations from the routine's own
/// signature (`kernels_metal::attn::sdpa_paged_tiled`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_tiled(
    symbol: &str,
    queries: &Val,
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    n_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        n_rows as u32,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::sdpa_paged_tiled>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_tiled` states `out`")
}

/// Generated for `sdpa_paged_tiled_sink`'s instantiations from the
/// routine's own signature (`kernels_metal::attn::sdpa_paged_tiled_sink`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_tiled_sink(
    symbol: &str,
    queries: &Val,
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    sinks: &str,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    n_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        n_rows as u32,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
    ];
    let run_weights = vec![sinks.to_string()];
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::sdpa_paged_tiled_sink>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_tiled_sink` states `out`")
}

/// Generated for `sdpa_paged_tiled_strided`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::attn::sdpa_paged_tiled_strided`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_tiled_strided(
    symbol: &str,
    queries: &Val,
    out: (Shape, DType),
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    n_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        n_rows as u32,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::sdpa_paged_tiled_strided>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_tiled_strided` states `out`")
}

/// Generated for `sdpa_paged_mma`'s instantiations from the routine's own
/// signature (`kernels_metal::attn::sdpa_paged_mma`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_mma(
    symbol: &str,
    queries: &Val,
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    n_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        n_rows as u32,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
    ];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::sdpa_paged_mma>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_mma` states `out`")
}

/// Generated for `sdpa_paged_mma_sink`'s instantiations from the routine's
/// own signature (`kernels_metal::attn::sdpa_paged_mma_sink`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_paged_mma_sink(
    symbol: &str,
    queries: &Val,
    n_kv_heads: i32,
    scale: f32,
    window: i32,
    sinks: &str,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    positions: &Val,
    request_of_token: &Val,
    maskv: &Val,
    n_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        n_kv_heads as u32,
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        n_rows as u32,
    ];
    let run_inputs = vec![
        queries.id,
        kvc.id,
        positions.id,
        request_of_token.id,
        maskv.id,
    ];
    let run_weights = vec![sinks.to_string()];
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::attn::sdpa_paged_mma_sink>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_paged_mma_sink` states `out`")
}

/// Generated for `sdpa_vector_decode`'s instantiations from the routine's
/// own signature (`kernels_metal::attn::sdpa_vector_decode`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_vector_decode(
    symbol: &str,
    queries: &Val,
    scale: f32,
    head_dim: i32,
    q_heads: i32,
    kvc: &Val,
    n_kv_heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        scale.to_bits(),
        head_dim as u32,
        q_heads as u32,
        n_kv_heads as u32,
        0,
    ];
    let run_inputs = vec![queries.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, symbol, OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(4, Shape(vec![rows_of(queries)]))];
    let made = fire_at::<kernels_metal::attn::sdpa_vector_decode>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_vector_decode` states `out`")
}

/// Generated for `sdpa_vector_decode_swa`'s instantiations from the
/// routine's own signature (`kernels_metal::attn::sdpa_vector_decode_swa`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_vector_decode_swa(
    symbol: &str,
    queries: &Val,
    out: (Shape, DType),
    scale: f32,
    window: i32,
    head_dim: i32,
    q_heads: i32,
    q_row_stride: i32,
    o_row_stride: i32,
    kvc: &Val,
    n_kv_heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        q_row_stride as u32,
        o_row_stride as u32,
        n_kv_heads as u32,
        0,
    ];
    let run_inputs = vec![queries.id, kvc.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents = vec![(7, Shape(vec![rows_of(queries)]))];
    let made = fire_at::<kernels_metal::attn::sdpa_vector_decode_swa>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_vector_decode_swa` states `out`")
}

/// Generated for `sdpa_vector_decode_sink`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::attn::sdpa_vector_decode_sink`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn sdpa_vector_decode_sink(
    symbol: &str,
    queries: &Val,
    out: (Shape, DType),
    scale: f32,
    window: i32,
    sinks: &str,
    head_dim: i32,
    q_heads: i32,
    q_row_stride: i32,
    o_row_stride: i32,
    kvc: &Val,
    n_kv_heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = queries.t.clone();
    let run_params = vec![
        scale.to_bits(),
        window as u32,
        head_dim as u32,
        q_heads as u32,
        q_row_stride as u32,
        o_row_stride as u32,
        n_kv_heads as u32,
        0,
    ];
    let run_inputs = vec![queries.id, kvc.id];
    let run_weights = vec![sinks.to_string()];
    let run_outs = vec![out];
    let run_extents = vec![(7, Shape(vec![rows_of(queries)]))];
    let made = fire_at::<kernels_metal::attn::sdpa_vector_decode_sink>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`sdpa_vector_decode_sink` states `out`")
}

/// Generated for `embed_gather_4bit`'s instantiations from the routine's
/// own signature (`kernels_metal::layout::embed_gather_4bit`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn embed_gather_4bit(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    out: (Shape, DType),
    group: i32,
    bits: i32,
    token_ids: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = token_ids.t.clone();
    let run_params = vec![group as u32, bits as u32];
    let run_inputs = vec![token_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::layout::embed_gather_4bit>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`embed_gather_4bit` states `out`")
}

/// Generated for `embed_gather_mb_4bit`'s instantiations from the routine's
/// own signature (`kernels_metal::layout::embed_gather_mb_4bit`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn embed_gather_mb_4bit(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    out: (Shape, DType),
    group: i32,
    bits: i32,
    token_ids: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = token_ids.t.clone();
    let run_params = vec![group as u32, bits as u32, 0];
    let run_inputs = vec![token_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents = vec![(2, Shape(vec![rows_of(token_ids)]))];
    let made = fire_at::<kernels_metal::layout::embed_gather_mb_4bit>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`embed_gather_mb_4bit` states `out`")
}

/// Generated for `embed_gather_scaled_4bit`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::layout::embed_gather_scaled_4bit`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn embed_gather_scaled_4bit(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    out: (Shape, DType),
    embed_scale: f32,
    group: i32,
    bits: i32,
    token_ids: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = token_ids.t.clone();
    let run_params = vec![embed_scale.to_bits(), group as u32, bits as u32];
    let run_inputs = vec![token_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::layout::embed_gather_scaled_4bit>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`embed_gather_scaled_4bit` states `out`")
}

/// Generated for `embed_gather_scaled_mb_4bit`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::layout::embed_gather_scaled_mb_4bit`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn embed_gather_scaled_mb_4bit(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    out: (Shape, DType),
    embed_scale: f32,
    group: i32,
    bits: i32,
    token_ids: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = token_ids.t.clone();
    let run_params = vec![
        embed_scale.to_bits(),
        group as u32,
        bits as u32,
        0,
    ];
    let run_inputs = vec![token_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents = vec![(3, Shape(vec![rows_of(token_ids)]))];
    let made = fire_at::<kernels_metal::layout::embed_gather_scaled_mb_4bit>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`embed_gather_scaled_mb_4bit` states `out`")
}

/// Generated for `ple_combine_bfloat16` from the routine's own signature
/// (`kernels_metal::layout::ple_combine`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn ple_combine(
    proj: &Val,
    token: &Val,
    inv_sqrt2: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = proj.t.clone();
    let run_params = vec![inv_sqrt2.to_bits(), 0];
    let run_inputs = vec![proj.id, token.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "ple_combine_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(1, Shape(vec![rows_of(proj)]))];
    let made = fire_at::<kernels_metal::layout::ple_combine>(&t, "ple_combine_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`ple_combine_bfloat16` states `out`")
}

/// Generated for `row_gather_bfloat16` from the routine's own signature
/// (`kernels_metal::layout::row_gather`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn row_gather(
    input: &Val,
    out: (Shape, DType),
    width: u32,
    sampling_indices: &Val,
    count: u32,
    row_count: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = input.t.clone();
    let run_params = vec![width, count, row_count as u32];
    let run_inputs = vec![input.id, sampling_indices.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::layout::row_gather>(&t, "row_gather_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`row_gather_bfloat16` states `out`")
}

/// Generated for `geglu_tanh_bfloat16` from the routine's own signature
/// (`kernels_metal::mlp::geglu_tanh`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn geglu_tanh(
    gate: &Val,
    up: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "geglu_tanh_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(0, Shape(vec![rows_of(gate)]))];
    let made = fire_at::<kernels_metal::mlp::geglu_tanh>(&t, "geglu_tanh_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`geglu_tanh_bfloat16` states `out`")
}

/// Generated for `geglu_tanh_strided_bfloat16` from the routine's own
/// signature (`kernels_metal::mlp::geglu_tanh_strided`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn geglu_tanh_strided(
    gate: &Val,
    up: &Val,
    stated_width: u32,
    stated_rows: u32,
    gate_pitch: u32,
    up_pitch: u32,
    out_pitch: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![
        stated_width,
        stated_rows,
        gate_pitch,
        up_pitch,
        out_pitch,
        0,
    ];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "geglu_tanh_strided_bfloat16", OutRule::Shaped { rows_of: 0, width: OutWidth::Param { of: 0 } }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(gate)]))];
    let made = fire_at::<kernels_metal::mlp::geglu_tanh_strided>(&t, "geglu_tanh_strided_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`geglu_tanh_strided_bfloat16` states `out`")
}

/// Generated for `gptoss_swiglu_bfloat16` from the routine's own signature
/// (`kernels_metal::mlp::gptoss_swiglu`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gptoss_swiglu(
    gate: &Val,
    up: &Val,
    _stated_elements: u32,
    limit: f32,
    alpha: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![
        _stated_elements,
        limit.to_bits(),
        alpha.to_bits(),
        0,
    ];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "gptoss_swiglu_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(3, Shape(vec![rows_of(gate)]))];
    let made = fire_at::<kernels_metal::mlp::gptoss_swiglu>(&t, "gptoss_swiglu_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gptoss_swiglu_bfloat16` states `out`")
}

/// Generated for `silu_mul_bfloat16` from the routine's own signature
/// (`kernels_metal::mlp::silu_mul`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn silu_mul(
    gate: &Val,
    up: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = gate.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![gate.id, up.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "silu_mul_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(0, Shape(vec![rows_of(gate)]))];
    let made = fire_at::<kernels_metal::mlp::silu_mul>(&t, "silu_mul_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`silu_mul_bfloat16` states `out`")
}

/// Generated for `router_topk_bfloat16` from the routine's own signature
/// (`kernels_metal::moe::router_topk`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn router_topk(
    logits: &Val,
    expert_ids: (Shape, DType),
    n_experts: u32,
    experts_per_token: u32,
    softmax_over_all: u32,
    logits_pitch: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params = vec![
        n_experts,
        experts_per_token,
        softmax_over_all,
        logits_pitch,
        0,
    ];
    let run_inputs = vec![logits.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        expert_ids,
        ruled_out(&t, "router_topk_bfloat16", OutRule::Shaped { rows_of: 0, width: OutWidth::Param { of: 1 } }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(4, Shape(vec![rows_of(logits)]))];
    let made = fire_at::<kernels_metal::moe::router_topk>(&t, "router_topk_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let expert_ids = made.next().expect("`router_topk_bfloat16` states `expert_ids`");
    let expert_weights = made.next().expect("`router_topk_bfloat16` states `expert_weights`");
    (expert_ids, expert_weights)
}

/// Generated for `router_topk_scaled_bfloat16` from the routine's own
/// signature (`kernels_metal::moe::router_topk_scaled`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn router_topk_scaled(
    logits: &Val,
    expert_ids: (Shape, DType),
    per_expert_scale: &str,
    n_experts: u32,
    experts_per_token: u32,
    softmax_over_all: u32,
    logits_pitch: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params = vec![
        n_experts,
        experts_per_token,
        softmax_over_all,
        logits_pitch,
        0,
    ];
    let run_inputs = vec![logits.id];
    let run_weights = vec![per_expert_scale.to_string()];
    let run_outs = vec![
        expert_ids,
        ruled_out(&t, "router_topk_scaled_bfloat16", OutRule::Shaped { rows_of: 0, width: OutWidth::Param { of: 1 } }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(4, Shape(vec![rows_of(logits)]))];
    let made = fire_at::<kernels_metal::moe::router_topk_scaled>(&t, "router_topk_scaled_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let expert_ids = made.next().expect("`router_topk_scaled_bfloat16` states `expert_ids`");
    let expert_weights = made.next().expect("`router_topk_scaled_bfloat16` states `expert_weights`");
    (expert_ids, expert_weights)
}

/// Generated for `route_sort` from the routine's own signature
/// (`kernels_metal::moe::route_sort`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn route_sort(
    expert_ids: &Val,
    perm: (Shape, DType),
    row_expert: (Shape, DType),
    tile_expert: (Shape, DType),
    inv: (Shape, DType),
    n: u32,
    n_experts: u32,
    experts_per_token: u32,
    tile_rows: u32,
    padded: u32,
    width: u32,
    x_pitch: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val, Val) {
    let t = expert_ids.t.clone();
    let run_params = vec![
        n,
        n_experts,
        experts_per_token,
        tile_rows,
        padded,
        width,
        x_pitch,
    ];
    let run_inputs = vec![expert_ids.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![perm, row_expert, tile_expert, inv];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::moe::route_sort>(&t, "route_sort", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let perm = made.next().expect("`route_sort` states `perm`");
    let row_expert = made.next().expect("`route_sort` states `row_expert`");
    let tile_expert = made.next().expect("`route_sort` states `tile_expert`");
    let inv = made.next().expect("`route_sort` states `inv`");
    (perm, row_expert, tile_expert, inv)
}

/// Generated for `route_gather` from the routine's own signature
/// (`kernels_metal::moe::route_gather`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn route_gather(
    x: &Val,
    out: (Shape, DType),
    perm: &Val,
    n: u32,
    n_experts: u32,
    experts_per_token: u32,
    tile_rows: u32,
    padded: u32,
    width: u32,
    x_pitch: u32,
    padded_rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        n,
        n_experts,
        experts_per_token,
        tile_rows,
        padded,
        width,
        x_pitch,
        padded_rows as u32,
    ];
    let run_inputs = vec![x.id, perm.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::moe::route_gather>(&t, "route_gather", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`route_gather` states `out`")
}

/// Generated for `combine_sorted` from the routine's own signature
/// (`kernels_metal::moe::combine_sorted`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn combine_sorted(
    y: &Val,
    expert_weights: &Val,
    inv: &Val,
    width: u32,
    experts_per_token: u32,
    out_pitch: u32,
    tokens: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = y.t.clone();
    let run_params = vec![width, experts_per_token, out_pitch, tokens as u32];
    let run_inputs = vec![y.id, expert_weights.id, inv.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "combine_sorted", OutRule::Shaped { rows_of: 1, width: OutWidth::Param { of: 0 } }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::moe::combine_sorted>(&t, "combine_sorted", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`combine_sorted` states `out`")
}

/// Generated for `shared_expert_combine` from the routine's own signature
/// (`kernels_metal::moe::shared_expert_combine`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn shared_expert_combine(
    routed: &Val,
    shared: &Val,
    gate: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = routed.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![routed.id, shared.id, gate.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "shared_expert_combine", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(0, Shape(vec![rows_of(routed)]))];
    let made = fire_at::<kernels_metal::moe::shared_expert_combine>(&t, "shared_expert_combine", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`shared_expert_combine` states `out`")
}

/// Generated for `shared_expert_combine_strided` from the routine's own
/// signature (`kernels_metal::moe::shared_expert_combine_strided`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn shared_expert_combine_strided(
    routed: &Val,
    shared: &Val,
    gate: &Val,
    out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = routed.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![routed.id, shared.id, gate.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents = vec![(0, Shape(vec![rows_of(routed)]))];
    let made = fire_at::<kernels_metal::moe::shared_expert_combine_strided>(&t, "shared_expert_combine_strided", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`shared_expert_combine_strided` states `out`")
}

/// Generated for `affine_qmv_routed_bfloat16_gs_64_b_4` from the routine's
/// own signature (`kernels_metal::moe::qmv_routed`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// `rows` is the FIRE's token count, but the first operand is the
/// SORTED STACK (`MoeAlignedRoutes`), so the rows convention would
/// splice the wrong axis. It stays a stated argument here; the hand
/// `routed_qmv` keeper splices `Dim::Tokens` and remains the fireable
/// form.
#[must_use]
pub fn qmv_routed(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
    expert_ids: &Val,
    rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        x_slot_stride as u32,
        x_row_stride as u32,
        slots_per_row as u32,
        rows as u32,
    ];
    let run_inputs = vec![x.id, expert_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::moe::qmv_routed>(&t, "affine_qmv_routed_bfloat16_gs_64_b_4", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmv_routed_bfloat16_gs_64_b_4` states `y`")
}

/// Generated for `affine_qmv_routed_bias_bfloat16_gs_64_b_4` from the
/// routine's own signature (`kernels_metal::moe::qmv_routed_bias`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// as `qmv_routed`: the fire's token count over a stack-rowed first
/// operand; the hand keeper splices `Dim::Tokens`.
#[must_use]
pub fn qmv_routed_bias(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    bias: &str,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
    expert_ids: &Val,
    rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        x_slot_stride as u32,
        x_row_stride as u32,
        slots_per_row as u32,
        rows as u32,
    ];
    let run_inputs = vec![x.id, expert_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
        bias.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::moe::qmv_routed_bias>(&t, "affine_qmv_routed_bias_bfloat16_gs_64_b_4", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmv_routed_bias_bfloat16_gs_64_b_4` states `y`")
}

/// Generated for `mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4` from the
/// routine's own signature (`kernels_metal::moe::mxfp4_qmv_routed_bias`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// as `qmv_routed`: the fire's token count over a stack-rowed first
/// operand; the hand keeper splices `Dim::Tokens`.
#[must_use]
pub fn mxfp4_qmv_routed_bias(
    w: &str,
    scales: &str,
    x: &Val,
    y: (Shape, DType),
    bias: &str,
    x_slot_stride: i32,
    x_row_stride: i32,
    slots_per_row: i32,
    expert_ids: &Val,
    rows: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        x_slot_stride as u32,
        x_row_stride as u32,
        slots_per_row as u32,
        rows as u32,
    ];
    let run_inputs = vec![x.id, expert_ids.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        bias.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::moe::mxfp4_qmv_routed_bias>(&t, "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4` states `y`")
}

/// Generated for `qmm_t_routed`'s instantiations from the routine's own
/// signature (`kernels_metal::moe::qmm_t_routed`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_routed(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    pad: &Val,
    tile_expert: &Val,
    group: i32,
    bits: i32,
    tile_m: i32,
    tile_n: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        group as u32,
        bits as u32,
        tile_m as u32,
        tile_n as u32,
        0,
    ];
    let run_inputs = vec![x.id, pad.id, tile_expert.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents = vec![(4, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::moe::qmm_t_routed>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_routed` states `y`")
}

/// Generated for `qmm_t_routed_fp16`'s instantiations from the routine's
/// own signature (`kernels_metal::moe::qmm_t_routed_fp16`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_routed_fp16(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    pad: &Val,
    tile_expert: &Val,
    tile_m: i32,
    tile_n: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![tile_m as u32, tile_n as u32, 0];
    let run_inputs = vec![x.id, pad.id, tile_expert.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents = vec![(2, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::moe::qmm_t_routed_fp16>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_routed_fp16` states `y`")
}

/// Generated for `mxfp4_qmm_t_routed_bias`'s instantiations from the
/// routine's own signature (`kernels_metal::moe::mxfp4_qmm_t_routed_bias`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn mxfp4_qmm_t_routed_bias(
    symbol: &str,
    w: &str,
    exponents: &str,
    x: &Val,
    pad: &Val,
    y: (Shape, DType),
    bias: &str,
    tile_expert: &Val,
    tile_m: i32,
    tile_n: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![tile_m as u32, tile_n as u32, 0];
    let run_inputs = vec![x.id, pad.id, tile_expert.id];
    let run_weights = vec![
        w.to_string(),
        exponents.to_string(),
        bias.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents = vec![(2, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::moe::mxfp4_qmm_t_routed_bias>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`mxfp4_qmm_t_routed_bias` states `y`")
}

/// Generated for `rms_single_row_bfloat16` from the routine's own signature
/// (`kernels_metal::norm::rms_single_row`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn rms_single_row(
    x: &Val,
    w: &str,
    eps: f32,
    axis: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        eps.to_bits(),
        axis as u32,
        w_stride,
        plus_one,
        gain.to_bits(),
        0,
    ];
    let run_inputs = vec![x.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![
        ruled_out(&t, "rms_single_row_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::rms_single_row>(&t, "rms_single_row_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`rms_single_row_bfloat16` states `out`")
}

/// Generated for `rms_strided_row_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::rms_strided_row`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn rms_strided_row(
    x: &Val,
    w: &str,
    out: (Shape, DType),
    eps: f32,
    axis: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        eps.to_bits(),
        axis as u32,
        w_stride,
        plus_one,
        gain.to_bits(),
        0,
    ];
    let run_inputs = vec![x.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![out];
    let run_extents = vec![(5, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::rms_strided_row>(&t, "rms_strided_row_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`rms_strided_row_bfloat16` states `out`")
}

/// Generated for `rms_strided_head_row_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::rms_strided_head_row`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn rms_strided_head_row(
    x: &Val,
    w: &str,
    out: (Shape, DType),
    eps: f32,
    axis: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        eps.to_bits(),
        axis as u32,
        w_stride,
        plus_one,
        gain.to_bits(),
        0,
    ];
    let run_inputs = vec![x.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![out];
    let run_extents = vec![(5, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::rms_strided_head_row>(&t, "rms_strided_head_row_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`rms_strided_head_row_bfloat16` states `out`")
}

/// Generated for `rms_residual_bfloat16` from the routine's own signature
/// (`kernels_metal::norm::rms_residual`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn rms_residual(
    x: &Val,
    w: &str,
    r: &Val,
    eps: f32,
    axis_size: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        eps.to_bits(),
        axis_size as u32,
        w_stride,
        plus_one,
        gain.to_bits(),
        0,
    ];
    let run_inputs = vec![x.id, r.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![
        ruled_out(&t, "rms_residual_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::rms_residual>(&t, "rms_residual_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`rms_residual_bfloat16` states `out`")
}

/// Generated for `rms_residual_scaled_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::rms_residual_scaled`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn rms_residual_scaled(
    x: &Val,
    w: &str,
    r: &Val,
    s: &Val,
    eps: f32,
    axis_size: i32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        eps.to_bits(),
        axis_size as u32,
        w_stride,
        plus_one,
        gain.to_bits(),
        0,
    ];
    let run_inputs = vec![x.id, r.id, s.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![
        ruled_out(&t, "rms_residual_scaled_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::rms_residual_scaled>(&t, "rms_residual_scaled_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`rms_residual_scaled_bfloat16` states `out`")
}

/// Generated for `vnorm_single_row_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::vnorm_single_row`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn vnorm_single_row(
    x: &Val,
    eps: f32,
    axis: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits(), axis as u32, 0];
    let run_inputs = vec![x.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "vnorm_single_row_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(2, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::vnorm_single_row>(&t, "vnorm_single_row_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`vnorm_single_row_bfloat16` states `out`")
}

/// Generated for `gated_rms_bfloat16` from the routine's own signature
/// (`kernels_metal::norm::gated_rms`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gated_rms(
    x: &Val,
    z: &Val,
    w: &str,
    out: (Shape, DType),
    eps: f32,
    vd: i32,
    heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits(), vd as u32, heads as u32, 0];
    let run_inputs = vec![x.id, z.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![out];
    let run_extents = vec![(3, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::gated_rms>(&t, "gated_rms_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gated_rms_bfloat16` states `out`")
}

/// Generated for `gated_rms_strided_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::gated_rms_strided`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gated_rms_strided(
    x: &Val,
    z: &Val,
    w: &str,
    out: (Shape, DType),
    eps: f32,
    vd: i32,
    heads: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![eps.to_bits(), vd as u32, heads as u32, 0];
    let run_inputs = vec![x.id, z.id];
    let run_weights = vec![w.to_string()];
    let run_outs = vec![out];
    let run_extents = vec![(3, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::gated_rms_strided>(&t, "gated_rms_strided_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gated_rms_strided_bfloat16` states `out`")
}

/// Generated for `layer_scalar_mul_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::layer_scalar_mul`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn layer_scalar_mul(
    x: &Val,
    scalar: &str,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![x.id];
    let run_weights = vec![scalar.to_string()];
    let run_outs = vec![
        ruled_out(&t, "layer_scalar_mul_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(0, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::layer_scalar_mul>(&t, "layer_scalar_mul_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`layer_scalar_mul_bfloat16` states `out`")
}

/// Generated for `residual_add_bfloat16` from the routine's own signature
/// (`kernels_metal::norm::residual_add`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn residual_add(
    x: &Val,
    residual: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![x.id, residual.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "residual_add_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(0, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::residual_add>(&t, "residual_add_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`residual_add_bfloat16` states `out`")
}

/// Generated for `residual_add_strided_bfloat16` from the routine's own
/// signature (`kernels_metal::norm::residual_add_strided`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn residual_add_strided(
    x: &Val,
    residual: &Val,
    out: (Shape, DType),
    row_pitch: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![row_pitch as u32, 0];
    let run_inputs = vec![x.id, residual.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents = vec![(1, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::norm::residual_add_strided>(&t, "residual_add_strided_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`residual_add_strided_bfloat16` states `out`")
}

/// Generated for `add_bias_bfloat16` from the routine's own signature
/// (`kernels_metal::norm::add_bias`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn add_bias(
    out: &Val,
    bias: &str,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = out.t.clone();
    let run_params = vec![0];
    let run_inputs = vec![out.id];
    let run_weights = vec![bias.to_string()];
    let run_outs = vec![
        ruled_out(&t, "add_bias_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(0, Shape(vec![rows_of(out)]))];
    let made = fire_at::<kernels_metal::norm::add_bias>(&t, "add_bias_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`add_bias_bfloat16` states `out`")
}

/// Generated for `copy_logits_bf16` from the routine's own signature
/// (`kernels_metal::ptir::copy_logits_bf16`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn copy_logits_bf16(
    source: &Val,
    destination: (Shape, DType),
    records: &Val,
    rows: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = source.t.clone();
    let run_params = vec![rows];
    let run_inputs = vec![source.id, records.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![destination];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::ptir::copy_logits_bf16>(&t, "copy_logits_bf16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`copy_logits_bf16` states `destination`")
}

/// Generated for `qmm_t`'s instantiations from the routine's own signature
/// (`kernels_metal::quant::qmm_t`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        group as u32,
        bits as u32,
        bm as u32,
        bn as u32,
        m as u32,
    ];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t` states `y`")
}

/// Generated for `qmm_t_bias`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmm_t_bias`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_bias(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    bias: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        group as u32,
        bits as u32,
        bm as u32,
        bn as u32,
        m as u32,
    ];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
        bias.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bias>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_bias` states `y`")
}

/// Generated for `qmm_t_residual`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmm_t_residual`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_residual(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    residual: &Val,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        group as u32,
        bits as u32,
        bm as u32,
        bn as u32,
        m as u32,
    ];
    let run_inputs = vec![x.id, residual.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_residual>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_residual` states `y`")
}

/// Generated for `qmm_t_fp16_precast`'s instantiations from the routine's
/// own signature (`kernels_metal::quant::qmm_t_fp16_precast`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_fp16_precast(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    y: (Shape, DType),
    half_in: &Val,
    bm: i32,
    bn: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, bn as u32, m as u32];
    let run_inputs = vec![half_in.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_fp16_precast>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_fp16_precast` states `y`")
}

/// Generated for `qmm_t_bias_fp16_precast`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::quant::qmm_t_bias_fp16_precast`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_bias_fp16_precast(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    y: (Shape, DType),
    bias: &str,
    half_in: &Val,
    bm: i32,
    bn: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, bn as u32, m as u32];
    let run_inputs = vec![half_in.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
        bias.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bias_fp16_precast>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_bias_fp16_precast` states `y`")
}

/// Generated for `qmm_t_residual_fp16_precast`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::quant::qmm_t_residual_fp16_precast`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_residual_fp16_precast(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    y: (Shape, DType),
    half_in: &Val,
    residual: &Val,
    bm: i32,
    bn: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, bn as u32, m as u32];
    let run_inputs = vec![half_in.id, residual.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_residual_fp16_precast>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_residual_fp16_precast` states `y`")
}

/// Generated for `qmm_t_splitk`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmm_t_splitk`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_splitk(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    out: (Shape, DType),
    group: i32,
    bits: i32,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![group as u32, bits as u32, bm as u32, m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_splitk>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_splitk` states `out`")
}

/// Generated for `qmm_t_splitk_f32`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmm_t_splitk_f32`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_splitk_f32(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    out: (Shape, DType),
    group: i32,
    bits: i32,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![group as u32, bits as u32, bm as u32, m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_splitk_f32>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_splitk_f32` states `out`")
}

/// Generated for `qmm_t_splitk_fp16_precast`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::quant::qmm_t_splitk_fp16_precast`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_splitk_fp16_precast(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    out: (Shape, DType),
    half_in: &Val,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, m as u32];
    let run_inputs = vec![half_in.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_splitk_fp16_precast>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_splitk_fp16_precast` states `out`")
}

/// Generated for `qmm_t_splitk_fp16_precast_f32`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::quant::qmm_t_splitk_fp16_precast_f32`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_splitk_fp16_precast_f32(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    out: (Shape, DType),
    half_in: &Val,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, m as u32];
    let run_inputs = vec![half_in.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_splitk_fp16_precast_f32>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_splitk_fp16_precast_f32` states `out`")
}

/// Generated for `qmm_t_strided`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmm_t_strided`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_strided(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    group: i32,
    bits: i32,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![group as u32, bits as u32, bm as u32, m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_strided>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_strided` states `y`")
}

/// Generated for `qmm_t_strided_residual`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::quant::qmm_t_strided_residual`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_strided_residual(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    residual: &Val,
    group: i32,
    bits: i32,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![group as u32, bits as u32, bm as u32, m as u32];
    let run_inputs = vec![x.id, residual.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_strided_residual>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_strided_residual` states `y`")
}

/// Generated for `qmm_t_strided_fp16_precast`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::quant::qmm_t_strided_fp16_precast`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_strided_fp16_precast(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    y: (Shape, DType),
    half_in: &Val,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, m as u32];
    let run_inputs = vec![half_in.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_strided_fp16_precast>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_strided_fp16_precast` states `y`")
}

/// Generated for `qmm_t_strided_fp16_precast_residual`'s instantiations
/// from the routine's own signature
/// (`kernels_metal::quant::qmm_t_strided_fp16_precast_residual`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmm_t_strided_fp16_precast_residual(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    y: (Shape, DType),
    half_in: &Val,
    residual: &Val,
    bm: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = half_in.t.clone();
    let run_params = vec![bm as u32, m as u32];
    let run_inputs = vec![half_in.id, residual.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_strided_fp16_precast_residual>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_t_strided_fp16_precast_residual` states `y`")
}

/// Generated for `qmm_splitk_reduce_bfloat16` from the routine's own
/// signature (`kernels_metal::quant::qmm_splitk_reduce`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn qmm_splitk_reduce(
    y: (Shape, DType),
    partial: &Val,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = partial.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![partial.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_splitk_reduce>(&t, "qmm_splitk_reduce_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_splitk_reduce_bfloat16` states `y`")
}

/// Generated for `qmm_splitk_reduce_f32_bfloat16` from the routine's own
/// signature (`kernels_metal::quant::qmm_splitk_reduce_f32`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn qmm_splitk_reduce_f32(
    y: (Shape, DType),
    partial: &Val,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = partial.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![partial.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_splitk_reduce_f32>(&t, "qmm_splitk_reduce_f32_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmm_splitk_reduce_f32_bfloat16` states `y`")
}

/// Generated for `cast_qmm_input_bfloat16_to_float16` from the routine's
/// own signature
/// (`kernels_metal::quant::cast_qmm_input_bfloat16_to_float16`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn cast_qmm_input_bfloat16_to_float16(
    cast_in: &Val,
    half_out: (Shape, DType),
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = cast_in.t.clone();
    let run_params: Vec<u32> = Vec::new();
    let run_inputs = vec![cast_in.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![half_out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::cast_qmm_input_bfloat16_to_float16>(&t, "cast_qmm_input_bfloat16_to_float16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`cast_qmm_input_bfloat16_to_float16` states `half_out`")
}

/// Generated for `cast_qmm_input_strided_bfloat16_to_float16` from the
/// routine's own signature
/// (`kernels_metal::quant::cast_qmm_input_strided_bfloat16_to_float16`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn cast_qmm_input_strided_bfloat16_to_float16(
    cast_in: &Val,
    half_out: (Shape, DType),
    row_stride: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = cast_in.t.clone();
    let run_params = vec![row_stride as u32, 0];
    let run_inputs = vec![cast_in.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![half_out];
    let run_extents = vec![(1, Shape(vec![rows_of(cast_in)]))];
    let made = fire_at::<kernels_metal::quant::cast_qmm_input_strided_bfloat16_to_float16>(&t, "cast_qmm_input_strided_bfloat16_to_float16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`cast_qmm_input_strided_bfloat16_to_float16` states `half_out`")
}

/// Generated for `qmv_fast`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmv_fast`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmv_fast(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    group: i32,
    bits: i32,
    vecs: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![group as u32, bits as u32, vecs as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmv_fast>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmv_fast` states `y`")
}

/// Generated for `qmv_fast_residual`'s instantiations from the routine's
/// own signature (`kernels_metal::quant::qmv_fast_residual`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmv_fast_residual(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    residual: &Val,
    group: i32,
    bits: i32,
    vecs: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![group as u32, bits as u32, vecs as u32];
    let run_inputs = vec![x.id, residual.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmv_fast_residual>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmv_fast_residual` states `y`")
}

/// Generated for `qmv_tail`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmv_tail`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmv_tail(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    bits: i32,
    vecs: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![bits as u32, vecs as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmv_tail>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmv_tail` states `y`")
}

/// Generated for `qmv_tail_bias`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmv_tail_bias`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmv_tail_bias(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    bias: &str,
    bits: i32,
    vecs: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![bits as u32, vecs as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
        bias.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmv_tail_bias>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmv_tail_bias` states `y`")
}

/// Generated for `qmv_wide_strided`'s instantiations from the routine's own
/// signature (`kernels_metal::quant::qmv_wide_strided`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn qmv_wide_strided(
    symbol: &str,
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    bits: i32,
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![bits as u32, m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmv_wide_strided>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`qmv_wide_strided` states `y`")
}

/// Generated for `affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4` from
/// the routine's own signature
/// (`kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4>(&t, "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4` states `y`")
}

/// Generated for `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2`
/// from the routine's own signature
/// (`kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2>(&t, "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2` states `y`")
}

/// Generated for `affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2`
/// from the routine's own signature
/// (`kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2>(&t, "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2` states `y`")
}

/// Generated for `affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1`
/// from the routine's own signature
/// (`kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1`);
/// the statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1>(&t, "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1` states `y`")
}

/// Generated for `affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4` from
/// the routine's own signature
/// (`kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4(
    w: &str,
    scales: &str,
    biases: &str,
    x: &Val,
    y: (Shape, DType),
    m: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![m as u32];
    let run_inputs = vec![x.id];
    let run_weights = vec![
        w.to_string(),
        scales.to_string(),
        biases.to_string(),
    ];
    let run_outs = vec![y];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4>(&t, "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4` states `y`")
}

/// Generated for `affine_encode_u4_bf16` from the routine's own signature
/// (`kernels_metal::quant::encode_u4_bf16`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn encode_u4_bf16(
    input: &Val,
    codes: (Shape, DType),
    scales: (Shape, DType),
    biases: (Shape, DType),
    group_size: i32,
    groups: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = input.t.clone();
    let run_params = vec![group_size as u32, groups as u32];
    let run_inputs = vec![input.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![codes, scales, biases];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::encode_u4_bf16>(&t, "affine_encode_u4_bf16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let codes = made.next().expect("`affine_encode_u4_bf16` states `codes`");
    let scales = made.next().expect("`affine_encode_u4_bf16` states `scales`");
    let biases = made.next().expect("`affine_encode_u4_bf16` states `biases`");
    (codes, scales, biases)
}

/// Generated for `affine_encode_u4_f32` from the routine's own signature
/// (`kernels_metal::quant::encode_u4_f32`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn encode_u4_f32(
    input: &Val,
    codes: (Shape, DType),
    scales: (Shape, DType),
    biases: (Shape, DType),
    group_size: i32,
    groups: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = input.t.clone();
    let run_params = vec![group_size as u32, groups as u32];
    let run_inputs = vec![input.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![codes, scales, biases];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::encode_u4_f32>(&t, "affine_encode_u4_f32", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let codes = made.next().expect("`affine_encode_u4_f32` states `codes`");
    let scales = made.next().expect("`affine_encode_u4_f32` states `scales`");
    let biases = made.next().expect("`affine_encode_u4_f32` states `biases`");
    (codes, scales, biases)
}

/// Generated for `mxfp4_dequant_bf16` from the routine's own signature
/// (`kernels_metal::quant::mxfp4_dequant_bf16`); the statement records
/// through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn mxfp4_dequant_bf16(
    payload: &Val,
    exponents: &Val,
    out: (Shape, DType),
    block_size: i32,
    blocks: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = payload.t.clone();
    let run_params = vec![block_size as u32, blocks as u32];
    let run_inputs = vec![payload.id, exponents.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::quant::mxfp4_dequant_bf16>(&t, "mxfp4_dequant_bf16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`mxfp4_dequant_bf16` states `out`")
}

/// Generated for `neox_decode_bfloat16` from the routine's own signature
/// (`kernels_metal::rope::neox_decode`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_decode(
    x: &Val,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        base.to_bits(),
        head_dim as u32,
        rotary as u32,
    ];
    let run_inputs = vec![x.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_decode_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::rope::neox_decode>(&t, "neox_decode_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_decode_bfloat16` states `x`")
}

/// Generated for `neox_mb_bfloat16` from the routine's own signature
/// (`kernels_metal::rope::neox_mb`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_mb(
    x: &Val,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        base.to_bits(),
        head_dim as u32,
        rotary as u32,
        0,
    ];
    let run_inputs = vec![x.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_mb_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(4, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::rope::neox_mb>(&t, "neox_mb_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_mb_bfloat16` states `x`")
}

/// Generated for `neox_freqs_decode_bfloat16` from the routine's own
/// signature (`kernels_metal::rope::neox_freqs_decode`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_freqs_decode(
    x: &Val,
    scale: f32,
    head_dim: i32,
    mscale: f32,
    rotary: i32,
    rope_freqs: &Val,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        head_dim as u32,
        mscale.to_bits(),
        rotary as u32,
    ];
    let run_inputs = vec![x.id, rope_freqs.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_freqs_decode_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::rope::neox_freqs_decode>(&t, "neox_freqs_decode_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_freqs_decode_bfloat16` states `x`")
}

/// Generated for `neox_freqs_mb_bfloat16` from the routine's own signature
/// (`kernels_metal::rope::neox_freqs_mb`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_freqs_mb(
    x: &Val,
    scale: f32,
    head_dim: i32,
    mscale: f32,
    rotary: i32,
    rope_freqs: &Val,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        head_dim as u32,
        mscale.to_bits(),
        rotary as u32,
        0,
    ];
    let run_inputs = vec![x.id, rope_freqs.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_freqs_mb_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(4, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::rope::neox_freqs_mb>(&t, "neox_freqs_mb_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_freqs_mb_bfloat16` states `x`")
}

/// Generated for `neox_prop_decode_bfloat16` from the routine's own
/// signature (`kernels_metal::rope::neox_prop_decode`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_prop_decode(
    x: &Val,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        base.to_bits(),
        head_dim as u32,
        rotary as u32,
    ];
    let run_inputs = vec![x.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_prop_decode_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::rope::neox_prop_decode>(&t, "neox_prop_decode_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_prop_decode_bfloat16` states `x`")
}

/// Generated for `neox_prop_mb_bfloat16` from the routine's own signature
/// (`kernels_metal::rope::neox_prop_mb`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_prop_mb(
    x: &Val,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        base.to_bits(),
        head_dim as u32,
        rotary as u32,
        0,
    ];
    let run_inputs = vec![x.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_prop_mb_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(4, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::rope::neox_prop_mb>(&t, "neox_prop_mb_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_prop_mb_bfloat16` states `x`")
}

/// Generated for `neox_strided_bfloat16` from the routine's own signature
/// (`kernels_metal::rope::neox_strided`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn neox_strided(
    x: &Val,
    scale: f32,
    base: f32,
    head_dim: i32,
    rotary: i32,
    row_pitch: i32,
    positions: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = x.t.clone();
    let run_params = vec![
        scale.to_bits(),
        base.to_bits(),
        head_dim as u32,
        rotary as u32,
        row_pitch as u32,
        0,
    ];
    let run_inputs = vec![x.id, positions.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![
        ruled_out(&t, "neox_strided_bfloat16", OutRule::Like { of: 0 }, &run_inputs, &run_params),
    ];
    let run_extents = vec![(5, Shape(vec![rows_of(x)]))];
    let made = fire_at::<kernels_metal::rope::neox_strided>(&t, "neox_strided_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`neox_strided_bfloat16` states `x`")
}

/// Generated for `argmax_logits_bfloat16` from the routine's own signature
/// (`kernels_metal::sample::argmax_logits`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn argmax_logits(
    logits: &Val,
    next_token: (Shape, DType),
    params: &Val,
    eos_flag: (Shape, DType),
    rows: u32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val) {
    let t = logits.t.clone();
    let run_params = vec![rows];
    let run_inputs = vec![logits.id, params.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![next_token, eos_flag];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::sample::argmax_logits>(&t, "argmax_logits_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let next_token = made.next().expect("`argmax_logits_bfloat16` states `next_token`");
    let eos_flag = made.next().expect("`argmax_logits_bfloat16` states `eos_flag`");
    (next_token, eos_flag)
}

/// Generated for `gdn_core_bfloat16` from the routine's own signature
/// (`kernels_metal::ssm::gdn_core`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gdn_core(
    mixed: &Val,
    core_out: (Shape, DType),
    conv_w: &str,
    conv_b: &str,
    a_log: &str,
    dt_bias: &str,
    a_gate: &Val,
    b_gate: &Val,
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        0,
    ];
    let run_inputs = vec![mixed.id, a_gate.id, b_gate.id, rsv.id];
    let run_weights = vec![
        conv_w.to_string(),
        conv_b.to_string(),
        a_log.to_string(),
        dt_bias.to_string(),
    ];
    let run_outs = vec![core_out];
    let run_extents = vec![(11, Shape(vec![rows_of(mixed)]))];
    let made = fire_at::<kernels_metal::ssm::gdn_core>(&t, "gdn_core_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gdn_core_bfloat16` states `core_out`")
}

/// Generated for `gdn_core_slotted_bfloat16` from the routine's own
/// signature (`kernels_metal::ssm::gdn_core_slotted`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gdn_core_slotted(
    mixed: &Val,
    core_out: (Shape, DType),
    conv_w: &str,
    conv_b: &str,
    a_log: &str,
    dt_bias: &str,
    a_gate: &Val,
    b_gate: &Val,
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        0,
    ];
    let run_inputs = vec![mixed.id, a_gate.id, b_gate.id, rsv.id];
    let run_weights = vec![
        conv_w.to_string(),
        conv_b.to_string(),
        a_log.to_string(),
        dt_bias.to_string(),
    ];
    let run_outs = vec![core_out];
    let run_extents = vec![(11, Shape(vec![rows_of(mixed)]))];
    let made = fire_at::<kernels_metal::ssm::gdn_core_slotted>(&t, "gdn_core_slotted_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gdn_core_slotted_bfloat16` states `core_out`")
}

/// Generated for `gdn_prep_bfloat16` from the routine's own signature
/// (`kernels_metal::ssm::gdn_prep`); the statement records through
/// [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gdn_prep(
    mixed: &Val,
    conv_w: &str,
    conv_b: &str,
    a_log: &str,
    dt_bias: &str,
    a_gate: &Val,
    b_gate: &Val,
    pre_q: (Shape, DType),
    pre_k: (Shape, DType),
    pre_gate: (Shape, DType),
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        0,
    ];
    let run_inputs = vec![mixed.id, a_gate.id, b_gate.id, rsv.id];
    let run_weights = vec![
        conv_w.to_string(),
        conv_b.to_string(),
        a_log.to_string(),
        dt_bias.to_string(),
    ];
    let run_outs = vec![pre_q, pre_k, pre_gate];
    let run_extents = vec![(11, Shape(vec![rows_of(mixed)]))];
    let made = fire_at::<kernels_metal::ssm::gdn_prep>(&t, "gdn_prep_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let pre_q = made.next().expect("`gdn_prep_bfloat16` states `pre_q`");
    let pre_k = made.next().expect("`gdn_prep_bfloat16` states `pre_k`");
    let pre_gate = made.next().expect("`gdn_prep_bfloat16` states `pre_gate`");
    (pre_q, pre_k, pre_gate)
}

/// Generated for `gdn_prep_slotted_bfloat16` from the routine's own
/// signature (`kernels_metal::ssm::gdn_prep_slotted`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gdn_prep_slotted(
    mixed: &Val,
    conv_w: &str,
    conv_b: &str,
    a_log: &str,
    dt_bias: &str,
    a_gate: &Val,
    b_gate: &Val,
    pre_q: (Shape, DType),
    pre_k: (Shape, DType),
    pre_gate: (Shape, DType),
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        0,
    ];
    let run_inputs = vec![mixed.id, a_gate.id, b_gate.id, rsv.id];
    let run_weights = vec![
        conv_w.to_string(),
        conv_b.to_string(),
        a_log.to_string(),
        dt_bias.to_string(),
    ];
    let run_outs = vec![pre_q, pre_k, pre_gate];
    let run_extents = vec![(11, Shape(vec![rows_of(mixed)]))];
    let made = fire_at::<kernels_metal::ssm::gdn_prep_slotted>(&t, "gdn_prep_slotted_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let pre_q = made.next().expect("`gdn_prep_slotted_bfloat16` states `pre_q`");
    let pre_k = made.next().expect("`gdn_prep_slotted_bfloat16` states `pre_k`");
    let pre_gate = made.next().expect("`gdn_prep_slotted_bfloat16` states `pre_gate`");
    (pre_q, pre_k, pre_gate)
}

/// Generated for `gdn_core_recurrent_bfloat16` from the routine's own
/// signature (`kernels_metal::ssm::gdn_core_recurrent`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gdn_core_recurrent(
    mixed: &Val,
    core_out: (Shape, DType),
    conv_w: &str,
    conv_b: &str,
    pre_q: &Val,
    pre_k: &Val,
    pre_gate: &Val,
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        0,
    ];
    let run_inputs = vec![mixed.id, pre_q.id, pre_k.id, pre_gate.id, rsv.id];
    let run_weights = vec![conv_w.to_string(), conv_b.to_string()];
    let run_outs = vec![core_out];
    let run_extents = vec![(11, Shape(vec![rows_of(mixed)]))];
    let made = fire_at::<kernels_metal::ssm::gdn_core_recurrent>(&t, "gdn_core_recurrent_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gdn_core_recurrent_bfloat16` states `core_out`")
}

/// Generated for `gdn_core_recurrent_slotted_bfloat16` from the routine's
/// own signature (`kernels_metal::ssm::gdn_core_recurrent_slotted`); the
/// statement records through [`crate::fire::fire_at`], one argument per
/// mark.
#[must_use]
pub fn gdn_core_recurrent_slotted(
    mixed: &Val,
    core_out: (Shape, DType),
    conv_w: &str,
    conv_b: &str,
    pre_q: &Val,
    pre_k: &Val,
    pre_gate: &Val,
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        0,
    ];
    let run_inputs = vec![mixed.id, pre_q.id, pre_k.id, pre_gate.id, rsv.id];
    let run_weights = vec![conv_w.to_string(), conv_b.to_string()];
    let run_outs = vec![core_out];
    let run_extents = vec![(11, Shape(vec![rows_of(mixed)]))];
    let made = fire_at::<kernels_metal::ssm::gdn_core_recurrent_slotted>(&t, "gdn_core_recurrent_slotted_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gdn_core_recurrent_slotted_bfloat16` states `core_out`")
}

/// Generated for `gdn_prep_prefill_bfloat16` from the routine's own
/// signature (`kernels_metal::ssm::gdn_prep_prefill`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
#[must_use]
pub fn gdn_prep_prefill(
    mixed: &Val,
    conv_w: &str,
    conv_b: &str,
    a_log: &str,
    dt_bias: &str,
    a_gate: &Val,
    b_gate: &Val,
    pre_q: (Shape, DType),
    pre_k: (Shape, DType),
    pre_gate: (Shape, DType),
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    rsv: &Val,
    n_scan: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> (Val, Val, Val) {
    let t = mixed.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        n_scan as u32,
    ];
    let run_inputs = vec![mixed.id, a_gate.id, b_gate.id, rsv.id];
    let run_weights = vec![
        conv_w.to_string(),
        conv_b.to_string(),
        a_log.to_string(),
        dt_bias.to_string(),
    ];
    let run_outs = vec![pre_q, pre_k, pre_gate];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::ssm::gdn_prep_prefill>(&t, "gdn_prep_prefill_bfloat16", Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    let mut made = made.into_iter();
    let pre_q = made.next().expect("`gdn_prep_prefill_bfloat16` states `pre_q`");
    let pre_k = made.next().expect("`gdn_prep_prefill_bfloat16` states `pre_k`");
    let pre_gate = made.next().expect("`gdn_prep_prefill_bfloat16` states `pre_gate`");
    (pre_q, pre_k, pre_gate)
}

/// Generated for `gdn_core_recurrent_prefill`'s instantiations from the
/// routine's own signature
/// (`kernels_metal::ssm::gdn_core_recurrent_prefill`); the statement
/// records through [`crate::fire::fire_at`], one argument per mark.
///
/// The routine's entrypoint is COMPOSED from an instantiation point, so
/// the SYMBOL is this wrapper's first argument; `fire_at` refuses one
/// the census does not resolve to this routine.
#[must_use]
pub fn gdn_core_recurrent_prefill(
    symbol: &str,
    pad: &Val,
    core_out: (Shape, DType),
    pre_q: &Val,
    pre_k: &Val,
    pre_gate: &Val,
    k_dim: i32,
    v_dim: i32,
    k_heads: i32,
    v_heads: i32,
    conv_dim: i32,
    conv_k: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
    lanes: i32,
    vrows: i32,
    rsv: &Val,
    n_scan: i32,
    layer: Option<u32>,
    state: Option<StateRef>,
) -> Val {
    let t = pad.t.clone();
    let run_params = vec![
        k_dim as u32,
        v_dim as u32,
        k_heads as u32,
        v_heads as u32,
        conv_dim as u32,
        conv_k as u32,
        q_off as u32,
        k_off as u32,
        v_off as u32,
        eps.to_bits(),
        inv_sqrt_dk.to_bits(),
        lanes as u32,
        vrows as u32,
        n_scan as u32,
    ];
    let run_inputs = vec![pad.id, pre_q.id, pre_k.id, pre_gate.id, rsv.id];
    let run_weights: Vec<String> = Vec::new();
    let run_outs = vec![core_out];
    let run_extents: Vec<(u8, Shape)> = Vec::new();
    let made = fire_at::<kernels_metal::ssm::gdn_core_recurrent_prefill>(&t, symbol, Call {
        inputs: run_inputs,
        weights: run_weights,
        params: run_params,
        outs: run_outs,
        state,
        layer,
        extents: run_extents,
    });
    made.into_iter().next().expect("`gdn_core_recurrent_prefill` states `core_out`")
}
