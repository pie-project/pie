//! THE CUDA LAUNCHER SURFACE.
//!
//! The CUDA launchers a lowered declaration may state, one function per
//! kernel, PARAMETERS = the launcher's semantic operands (tensors,
//! weights, state, tables). Mechanical parameters — stream, dims,
//! workspace scratch, plan caches — are the driver's binding, not a
//! choice, and do not appear. Each records one [`OpKind::Launch`](model_ir::trace::OpKind::Launch) (or
//! the exact launch pair the hand-written arm makes); the doc comment is
//! the contract naming the C++ symbol.
//!
//! Prepare-phase host work (decode-plan builds, XQA's fire-wide prepare)
//! is NOT stated here: the trace states the BODY's launches, and a
//! stated kernel obligates the driver to whatever prepare its contract
//! needs — the same prepare/body seam the graph work built.

// NOT `use super::*;`, which is what every other file in this crate does.
// A glob here would collide with the `pub use <file>::*` re-exports below:
// `rmsnorm`, `swiglu` and `weighted_sum` are spelled BOTH by the neutral
// vocabulary and by this surface, and two globs binding one name in one
// module is an ambiguity rather than a shadowing. When `cuda` was an inline
// module the local `pub fn` won outright; a re-export does not, so the
// import that used to be free has to be explicit.
//
// It is only TYPES. A statement here never calls a neutral op -- it records a
// launch -- so nothing below is a function, and that is what keeps the list
// from growing back into a glob.
use crate::{ConvW, Kv, MatW, NormW, Rs, Trace, Val};
use model_ir::trace::{DType, Dim, NormVariant, Shape, StateRef, StateStore};


/// A launch that produces MORE THAN ONE value.
///
/// `TraceBuilder::launch` always returned a `Vec`; [`record`] narrowed it
/// to the first, which was right for every statement until MLA. Its
/// prepare splits a latent KV row into four -- `kv_c`, `k_pe`, `q_nope`,
/// `q_pe` -- and a statement returning one of them would leave the other
/// three unnamed on the tape, which is exactly the silent dataflow gap
/// the trace exists to make visible.
/// [`record_many`], plus the scalar arguments — [`record_with_params`]
/// for a statement with more than one result.
fn record_many_with_params(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    params: Vec<u32>,
    inputs: Vec<model_ir::trace::ValueId>,
    outs: Vec<(Shape, DType)>,
) -> Vec<Val> {
    let n = outs.len();
    let ids = t.with(layer, |b| {
        b.launch_with_params(kernel, weights, None, params, inputs, outs)
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

/// [`record`], plus the SCALAR ARGUMENTS the symbol takes that no
/// operand shape gives ([`model_ir::trace::OpKind::Launch`]'s params).
///
/// Signed values ride as their two's complement: `window_left = -1`
/// is `0xFFFFFFFF`, and the executor casts back. The channel is
/// untyped on purpose -- what each slot means is the SYMBOL's
/// contract, exactly as `aux_names`' slots are.
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

fn kv_state(kv: &Kv) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::KvCache,
        layer: kv.l,
    })
}

/// The GDN ops' state mark, [`kv_state`]-style: the layer's
/// per-request conv/recurrent slabs.
fn rs_state(rs: &Rs) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::RecurrentState,
        layer: rs.l,
    })
}

/// `kernels::rope::rope_standard_table`: build the fire's cos/sin
/// table, once. A value, not a latch — the fused-QKV kernel consumes
/// it as an operand.
pub fn rope_standard_table(t: &Trace, head_dim: u32) -> Val {
    record(
        t,
        None,
        "rope::rope_standard_table",
        vec![],
        None,
        vec![],
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::F32)),
    )
    .expect("table launch produces a value")
}

/// `kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16`: the fused
/// decode-QKV epilogue — split + per-head Plain q/k norms + Standard
/// rope + KV append, one launch. Packed GEMM output in, roped Q out;
/// K/V go straight to the cache and never exist as values. The
/// general arm (`split_qkv` + `rmsnorm`×2 + `rope` + `Kv::append`) is
/// this call's semantics, and the parity harness holds it there.
pub fn qkv_decode_qk_norm_rope_write_kv(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    table: Option<&Val>,
    q_width: u32,
) -> Val {
    let mut inputs = vec![packed.id];
    inputs.extend(table.map(|t| t.id));
    record(
        &packed.t,
        Some(kv.l),
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
        inputs,
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("fused post produces q")
}


// ── The CUDA surface, by subject ───────────────────────────────────────
//
// `cuda.rs` was 4,511 lines and 171 statements in one module, ordered by
// WHEN each generation needed something rather than by what it is. The
// files below are the subjects; the recording helpers above stay here
// because every one of them calls these.
//
// Flat `pub use`, because a declaration spells `dsl::cuda::rope_partial`
// and which file a statement lives in is not a declaration's business.
mod attn;
mod base;
mod deepseek_v4;
mod gemma;
mod mla;
mod moe;
mod qwen_3_5;
mod rope;
mod ssm;
mod tp;

pub use attn::*;
pub use base::*;
pub use deepseek_v4::*;
pub use gemma::*;
pub use mla::*;
pub use moe::*;
pub use qwen_3_5::*;
pub use rope::*;
pub use ssm::*;
pub use tp::*;
