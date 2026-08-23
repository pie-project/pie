//! CUDA launch statements for model DSL traces.
//!
//! Functions state semantic operands; driver-owned stream, dims, workspace,
//! and prepare work are bound by the driver contract.

use crate::{Kv, MatW, NormW, RaggedVal, Rs, Trace, Val};
use model_ir::trace::{DType, Dim, Shape, StateRef, StateStore};

fn record_many(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    inputs: Vec<model_ir::trace::ValueId>,
    outs: Vec<(Shape, DType)>,
) -> Vec<Val> {
    let n = outs.len();
    let ids = t.with(layer, |b| b.launch(kernel, weights, None, inputs, outs));
    assert_eq!(
        ids.len(),
        n,
        "the tape recorded a different arity than stated"
    );
    ids.into_iter()
        .map(|id| Val {
            t: t.clone(),
            id,
            layer,
        })
        .collect()
}

fn record(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch(kernel, weights, state, inputs, out.into_iter().collect())
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record`], plus symbol-defined params.
/// Signed values use two's complement.
fn record_with_params(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_with_params(
            kernel,
            weights,
            state,
            params,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record_with_params`], plus the scalars whose value is an extent the
/// FIRE decides — see [`model_ir::trace::OpKind::Launch`]'s `param_extents`.
/// Constants at those indices are placeholders and written as zero.
#[allow(clippy::too_many_arguments)]
fn record_with_extents(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_with_extents(
            kernel,
            weights,
            state,
            params,
            param_extents,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record_with_extents`], plus the peel-window slots the walk fills.
fn record_devwin(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    peel_slots: Option<(u8, u8)>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_devwin(
            kernel,
            weights,
            state,
            params,
            param_extents,
            peel_slots,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record_many_with_params`], plus fire-decided scalar extents.
#[allow(clippy::too_many_arguments)]
fn record_many_with_extents(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    inputs: Vec<model_ir::trace::ValueId>,
    outs: Vec<(Shape, DType)>,
) -> Vec<Val> {
    let n = outs.len();
    let ids = t.with(layer, |b| {
        b.launch_with_extents(kernel, weights, state, params, param_extents, inputs, outs)
    });
    assert_eq!(
        ids.len(),
        n,
        "the tape recorded a different arity than stated"
    );
    ids.into_iter()
        .map(|id| Val {
            t: t.clone(),
            id,
            layer,
        })
        .collect()
}

/// Mint one runtime OBJECT operand — a driver-owned view out of the
/// `kernels` runtime vocabulary (`"kv_cache"`, `"fa2.prefill"`, …) — and
/// return its value id for an `inputs` run.
fn rt_object(t: &Trace, name: &str, layer: Option<u32>) -> model_ir::trace::ValueId {
    t.with(layer, |b| b.runtime_object(name, layer))
}

/// Mint one per-fire `[Tokens]` i32 stream by its vocabulary name
/// (`"positions"`, `"row_valid"`, …).
fn rt_tokens(t: &Trace, name: &str) -> model_ir::trace::ValueId {
    t.with(None, |b| {
        b.runtime_tensor(name, None, Shape(vec![Dim::Tokens]), DType::I32)
    })
}

/// Mint one per-fire `[Requests]` i32 stream by its vocabulary name
/// (`"qo_indptr"`, `"attn.score_indptr"`, …). CSR streams state
/// `[Requests]`; the driver stages the `+1` row the convention implies.
fn rt_requests(t: &Trace, name: &str) -> model_ir::trace::ValueId {
    t.with(None, |b| {
        b.runtime_tensor(name, None, Shape(vec![Dim::Requests]), DType::I32)
    })
}

/// The extent pair for a `num_requests`/`r` scalar the fire decides:
/// zero placeholder at `at`, spliced with the fire's request count.
fn requests_extent(at: u8) -> (u8, Shape) {
    (at, Shape(vec![Dim::Requests]))
}

/// The extent pair for a `rows`/`n_max` scalar: the fire's token rows.
fn tokens_extent(at: u8) -> (u8, Shape) {
    (at, Shape(vec![Dim::Tokens]))
}

// `ruled_out` moved to the crate root when the METAL plane's generated
// wrappers arrived: both planes derive ruled results through the same
// evaluation, and this re-export keeps `generated.rs`'s `use super::ruled_out`
// path stable so the move regenerated nothing.
pub(crate) use crate::ruled_out;

fn kv_state(kv: &Kv) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::KvCache,
        layer: kv.l,
    })
}

/// State mark for GDN ops.
fn rs_state(rs: &Rs) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::RecurrentState,
        layer: rs.l,
    })
}

mod attn;
mod base;
mod deepseek_v4;
mod gemma;
/// GENERATED named wrappers, one per traced `#[routine]` in
/// `crates/kernels-cuda/src` — see design-no-ask §10 (B4-gen). Deliberately
/// NOT glob-re-exported: callers opt in with `dsl::cuda::generated::`, so no
/// generated name can shadow a hand-written one while both exist.
pub mod generated;
mod mla;
mod moe;
mod qwen_3_5;
mod tp;

pub use attn::*;
pub use base::*;
pub use deepseek_v4::*;
pub use gemma::*;
pub use mla::*;
pub use moe::*;
pub use qwen_3_5::*;
pub use tp::*;
