//! Round-trip tests for the POD boundary.
//!
//! Same discipline as `loader/src/ffi/tests.rs`: build the Rust form, flatten
//! it, and read the result back through the raw pointers a C++ consumer would
//! walk. The assertions target the *synthesized* fields — the name table, the
//! flat operand array, the param packing — because those are the only places
//! the flattening can silently disagree with the trace it came from.

use super::arena::{self, view};
use super::entry::{
    PieForwardLlamaLikeFacts, PieForwardStatus, pie_forward_release, pie_forward_trace_llama_like,
};
use super::types::*;
use crate::facts::LlamaLikeFacts;
use crate::family::llama_like;
use crate::trace::OpKind;

/// The qwen3 parity facts, as a C caller would state them.
fn c_facts_qwen3() -> PieForwardLlamaLikeFacts {
    let facts = LlamaLikeFacts::qwen3_0_6b();
    PieForwardLlamaLikeFacts {
        hidden: facts.hidden,
        layers: facts.layers,
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        intermediate: facts.intermediate,
        vocab: facts.vocab,
        rope: PieForwardRopeKind::from(facts.rope) as u32,
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
        qk_norm: u8::from(facts.qk_norm),
        fused_qkv: u8::from(facts.fused_qkv),
        tied_embeddings: u8::from(facts.tied_embeddings),
    }
}

/// The kind tag each Rust op must flatten to; a second statement of the
/// mapping in `arena::flatten_kind`, so a drift is a failure here rather
/// than a mis-dispatched op driver-side.
fn expect_kind(kind: &OpKind) -> PieForwardOpKind {
    match kind {
        OpKind::Embed { .. } => PieForwardOpKind::Embed,
        OpKind::Matmul { .. } => PieForwardOpKind::Matmul,
        OpKind::Rmsnorm { .. } => PieForwardOpKind::Rmsnorm,
        OpKind::RmsnormPerHead { .. } => PieForwardOpKind::RmsnormPerHead,
        OpKind::SplitQkv { .. } => PieForwardOpKind::SplitQkv,
        OpKind::Rope { .. } => PieForwardOpKind::Rope,
        OpKind::KvAppend { .. } => PieForwardOpKind::KvAppend,
        OpKind::Attention { .. } => PieForwardOpKind::Attention,
        OpKind::Swiglu { .. } => PieForwardOpKind::Swiglu,
        OpKind::LmHead { .. } => PieForwardOpKind::LmHead,
    }
}

/// The weight name each Rust op carries, if any.
fn expect_weight(kind: &OpKind) -> Option<&str> {
    match kind {
        OpKind::Embed { weight }
        | OpKind::Matmul { weight, .. }
        | OpKind::Rmsnorm { weight, .. }
        | OpKind::RmsnormPerHead { weight, .. }
        | OpKind::LmHead { weight } => Some(weight),
        _ => None,
    }
}

/// A traced qwen3 plan survives the arena: op count, kinds, layers, operand
/// dataflow and weight names all read back exactly.
#[test]
fn qwen3_round_trips_through_the_arena() {
    let plan = llama_like(&LlamaLikeFacts::qwen3_0_6b());
    let mut pod = arena::build(&plan);

    assert_eq!(view::name(&pod, pod.family), "llama_like");
    assert_eq!(pod.compiler_version, crate::ffi::compiler_version());

    let ops = view::ops(&pod);
    assert_eq!(ops.len(), plan.ops.len());
    assert_eq!(view::values(&pod).len(), plan.values.len());

    for (rust, c) in plan.ops.iter().zip(ops) {
        assert_eq!(c.kind, expect_kind(&rust.kind), "kind of {rust:?}");
        assert_eq!(
            c.layer,
            rust.layer.map_or(PIE_FORWARD_NO_LAYER, |l| l as i32),
            "layer of {rust:?}"
        );
        assert_eq!(view::ids(&pod, c.inputs), &rust.inputs[..]);
        assert_eq!(view::ids(&pod, c.outputs), &rust.outputs[..]);
        match expect_weight(&rust.kind) {
            Some(weight) => assert_eq!(view::name(&pod, c.weight_name), weight),
            None => assert_eq!(c.weight_name, PIE_FORWARD_NO_NAME),
        }
    }

    // Spot-check the names the driver will bind: the prologue's table, one
    // per-layer weight, and the tied lm_head resolving to `embed` again —
    // through the *same* interned entry, which is what makes tying visible
    // as identity rather than as string comparison.
    assert_eq!(view::name(&pod, ops[0].weight_name), "embed");
    let qkv = ops
        .iter()
        .find(|op| op.layer == 3 && op.kind == PieForwardOpKind::Matmul)
        .expect("layer 3 has a matmul");
    assert_eq!(view::name(&pod, qkv.weight_name), "layer.3.qkv");
    let lm_head = ops.last().unwrap();
    assert_eq!(lm_head.kind, PieForwardOpKind::LmHead);
    assert_eq!(lm_head.weight_name, ops[0].weight_name);

    unsafe { arena::release(&mut pod) };
    assert!(pod.owner.is_null());
}

/// Shapes and params survive: the logits value reads back as
/// `[Requests, Const(vocab)] f32`, and SplitQkv carries its widths.
#[test]
fn qwen3_values_and_params_round_trip() {
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let plan = llama_like(&facts);
    let mut pod = arena::build(&plan);

    let ops = view::ops(&pod);
    let logits_id = view::ids(&pod, ops.last().unwrap().outputs)[0];
    let logits = view::values(&pod)[logits_id as usize];
    assert_eq!(logits.dtype, PieForwardDType::F32);
    assert_eq!(logits.rank, 2);
    assert_eq!(logits.dims[0].kind, PieForwardDimKind::Requests);
    assert_eq!(logits.dims[1].kind, PieForwardDimKind::Const);
    assert_eq!(logits.dims[1].value, facts.vocab);
    // Slots past the rank rest at the impossible zero-extent constant.
    assert_eq!(logits.dims[2], PieForwardDim::default());

    let split = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::SplitQkv)
        .expect("fused binding traces a split");
    assert_eq!(split.param0, facts.q_width());
    assert_eq!(split.param1, facts.kv_width());
    assert_eq!(view::ids(&pod, split.outputs).len(), 3);

    unsafe { arena::release(&mut pod) };
}

/// The entry point end to end: C facts in, POD plan out, release twice.
#[test]
fn entry_traces_and_releases() {
    let facts = c_facts_qwen3();
    let mut out = PieForwardPlan::default();
    let status = unsafe { pie_forward_trace_llama_like(&facts, &mut out) };
    assert_eq!(status, PieForwardStatus::Ok);
    // 13 ops per layer + embed + final norm + lm_head, the count
    // `family::tests::qwen3_full_plan_shape` pins on the Rust side.
    assert_eq!(out.ops.len, 13 * facts.layers as usize + 3);
    assert!(!out.owner.is_null());

    unsafe { pie_forward_release(&mut out) };
    assert!(out.owner.is_null());
    // Idempotent: the first release emptied the header.
    unsafe { pie_forward_release(&mut out) };
    unsafe { pie_forward_release(std::ptr::null_mut()) };
}

/// Null pointers are a malformed request, not a crash.
#[test]
fn entry_rejects_null_pointers() {
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(std::ptr::null(), &mut out) },
        PieForwardStatus::InvalidArgument
    );
    assert!(out.owner.is_null(), "failed call must leave the slot empty");

    let facts = c_facts_qwen3();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&facts, std::ptr::null_mut()) },
        PieForwardStatus::InvalidArgument
    );
}

/// Out-of-range enum values are rejected before any Rust enum is formed.
#[test]
fn entry_rejects_out_of_range_enums() {
    let mut out = PieForwardPlan::default();

    let mut facts = c_facts_qwen3();
    facts.rope = 99;
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&facts, &mut out) },
        PieForwardStatus::InvalidArgument
    );

    let mut facts = c_facts_qwen3();
    facts.norm_variant = 7;
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&facts, &mut out) },
        PieForwardStatus::InvalidArgument
    );
    assert!(out.owner.is_null());
}

/// The unfused binding crosses the boundary too: three per-layer projection
/// matmuls and no SplitQkv, mirroring
/// `family::tests::unfused_binding_traces_three_matmuls`.
#[test]
fn entry_honours_the_unfused_binding() {
    let mut facts = c_facts_qwen3();
    facts.fused_qkv = 0;
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&facts, &mut out) },
        PieForwardStatus::Ok
    );
    let ops = view::ops(&out);
    assert!(!ops.iter().any(|op| op.kind == PieForwardOpKind::SplitQkv));
    let layer0_matmuls = ops
        .iter()
        .filter(|op| op.layer == 0 && op.kind == PieForwardOpKind::Matmul)
        .count();
    assert_eq!(layer0_matmuls, 6);
    unsafe { pie_forward_release(&mut out) };
}
