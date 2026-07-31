//! Round-trip tests for the POD boundary.
//!
//! Same discipline as `loader/src/ffi/tests.rs`: build the Rust form, flatten
//! it, and read the result back through the raw pointers a C++ consumer would
//! walk. The assertions target the *synthesized* fields — the name table, the
//! flat operand array, the param packing — because those are the only places
//! the flattening can silently disagree with the trace it came from.

use super::arena::{self, view};
use super::entry::{
    PieForwardLlamaLikeFacts, PieForwardQwen35MoeMlpFacts, PieForwardStatus, pie_forward_release,
    pie_forward_trace_llama_like, pie_forward_trace_qwen3_5_moe_mlp,
};
use super::types::*;
use crate::facts::{LlamaLikeFacts, Qwen35MoeMlpFacts};
use crate::family::{llama_like, qwen3_5_moe_mlp_block};
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
        norm_placement: PieForwardNormPlacement::from(facts.norm_placement) as u32,
        qk_norm: PieForwardQkNorm::from(facts.qk_norm) as u32,
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
        OpKind::ResidualAdd => PieForwardOpKind::ResidualAdd,
        OpKind::TopK { .. } => PieForwardOpKind::TopK,
        OpKind::WeightedSum { .. } => PieForwardOpKind::WeightedSum,
        OpKind::SigmoidGateAdd => PieForwardOpKind::SigmoidGateAdd,
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

    let mut facts = c_facts_qwen3();
    facts.norm_placement = 2;
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&facts, &mut out) },
        PieForwardStatus::InvalidArgument
    );

    let mut facts = c_facts_qwen3();
    facts.qk_norm = 3;
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&facts, &mut out) },
        PieForwardStatus::InvalidArgument
    );
    assert!(out.owner.is_null());
}

/// The olmo2 facts cross the boundary: post-norm placement and the global
/// qk-norm reach the tracer as enum wire values, and the traced form that
/// comes back is the post-norm walk — ResidualAdd ops present, no beta=1
/// accumulates, no RmsnormPerHead (the global convention is a plain
/// Rmsnorm), 16 ops per layer.
#[test]
fn entry_honours_post_norm_and_global_qk_norm() {
    let facts = LlamaLikeFacts::olmo2_1b();
    let c_facts = PieForwardLlamaLikeFacts {
        hidden: facts.hidden,
        layers: facts.layers,
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        intermediate: facts.intermediate,
        vocab: facts.vocab,
        rope: PieForwardRopeKind::from(facts.rope) as u32,
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
        norm_placement: PieForwardNormPlacement::from(facts.norm_placement) as u32,
        qk_norm: PieForwardQkNorm::from(facts.qk_norm) as u32,
        fused_qkv: u8::from(facts.fused_qkv),
        tied_embeddings: u8::from(facts.tied_embeddings),
    };
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like(&c_facts, &mut out) },
        PieForwardStatus::Ok
    );
    let ops = view::ops(&out);
    assert_eq!(ops.len(), 16 * facts.layers as usize + 3);
    let layer0_adds = ops
        .iter()
        .filter(|op| op.layer == 0 && op.kind == PieForwardOpKind::ResidualAdd)
        .count();
    assert_eq!(layer0_adds, 2);
    assert!(
        !ops.iter()
            .any(|op| op.kind == PieForwardOpKind::Matmul && op.param0 != 0)
    );
    assert!(
        !ops.iter()
            .any(|op| op.kind == PieForwardOpKind::RmsnormPerHead)
    );
    // ResidualAdd references no weight; its two inputs (normed output,
    // residual stream) crossed intact.
    let add = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::ResidualAdd)
        .unwrap();
    assert_eq!(add.weight_name, PIE_FORWARD_NO_NAME);
    assert_eq!(add.inputs.len, 2);
    unsafe { pie_forward_release(&mut out) };
}

/// The MoE fragment round-trips: dyn op kinds, the selector field (set on
/// exactly the two expert matmuls, resting at [`PIE_FORWARD_NO_VALUE`]
/// everywhere else), the weight TEMPLATE crossing verbatim, the rank-3
/// route-expanded shapes, and the TopK params.
#[test]
fn moe_fragment_round_trips_through_the_arena() {
    let facts = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
    let plan = qwen3_5_moe_mlp_block(&facts);
    let mut pod = arena::build(&plan);

    assert_eq!(view::name(&pod, pod.family), "qwen3_5_moe_mlp_block");
    let ops = view::ops(&pod);
    assert_eq!(ops.len(), plan.ops.len());
    for (rust, c) in plan.ops.iter().zip(ops) {
        assert_eq!(c.kind, expect_kind(&rust.kind), "kind of {rust:?}");
        assert_eq!(view::ids(&pod, c.inputs), &rust.inputs[..]);
        assert_eq!(view::ids(&pod, c.outputs), &rust.outputs[..]);
        match &rust.kind {
            OpKind::Matmul {
                selector: Some(selector),
                ..
            } => assert_eq!(c.selector, *selector),
            _ => assert_eq!(c.selector, PIE_FORWARD_NO_VALUE, "selector of {rust:?}"),
        }
    }

    let topk = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::TopK)
        .unwrap();
    assert_eq!(topk.param0, facts.top_k);
    assert_eq!(topk.weight_name, PIE_FORWARD_NO_NAME);
    let idx_id = view::ids(&pod, topk.outputs)[0];
    let idx = view::values(&pod)[idx_id as usize];
    assert_eq!(idx.dtype, PieForwardDType::I32);
    assert_eq!(idx.rank, 2);
    assert_eq!(idx.dims[0].kind, PieForwardDimKind::Tokens);
    assert_eq!(idx.dims[1].value, facts.top_k);

    let grouped: Vec<_> = ops
        .iter()
        .filter(|op| op.selector != PIE_FORWARD_NO_VALUE)
        .collect();
    assert_eq!(grouped.len(), 2);
    for op in &grouped {
        assert_eq!(op.kind, PieForwardOpKind::Matmul);
        assert_eq!(op.selector, idx_id);
        // The selector is also the last input — the field states which
        // input selects, it does not add an operand.
        assert_eq!(*view::ids(&pod, op.inputs).last().unwrap(), idx_id);
    }
    assert_eq!(
        view::name(&pod, grouped[0].weight_name),
        "layer.0.expert.{e}.gate_up"
    );
    // Rank-3 route-expanded output: [Tokens, k, 2 * Im].
    let gu_out = view::values(&pod)[view::ids(&pod, grouped[0].outputs)[0] as usize];
    assert_eq!(gu_out.rank, 3);
    assert_eq!(gu_out.dims[0].kind, PieForwardDimKind::Tokens);
    assert_eq!(gu_out.dims[1].value, facts.top_k);
    assert_eq!(gu_out.dims[2].value, 2 * facts.moe_intermediate);

    let combine = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::WeightedSum)
        .unwrap();
    assert_eq!(combine.param0, facts.top_k);

    unsafe { arena::release(&mut pod) };
}

/// The MoE entry point end to end: C facts in, POD plan out; malformed
/// requests (bad enum, zero experts/k) answer InvalidArgument.
#[test]
fn moe_entry_traces_and_validates() {
    let facts = Qwen35MoeMlpFacts::qwen3_5_35b_a3b();
    let c_facts = PieForwardQwen35MoeMlpFacts {
        hidden: facts.hidden,
        num_experts: facts.num_experts,
        top_k: facts.top_k,
        moe_intermediate: facts.moe_intermediate,
        shared_expert_intermediate: facts.shared_expert_intermediate,
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
    };
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_moe_mlp(&c_facts, &mut out) },
        PieForwardStatus::Ok
    );
    // The 13-op block `family::tests::moe_block_op_sequence` pins.
    assert_eq!(out.ops.len, 13);
    unsafe { pie_forward_release(&mut out) };

    for bad in [
        PieForwardQwen35MoeMlpFacts {
            norm_variant: 9,
            ..c_facts
        },
        PieForwardQwen35MoeMlpFacts {
            num_experts: 0,
            ..c_facts
        },
        PieForwardQwen35MoeMlpFacts { top_k: 0, ..c_facts },
    ] {
        assert_eq!(
            unsafe { pie_forward_trace_qwen3_5_moe_mlp(&bad, &mut out) },
            PieForwardStatus::InvalidArgument
        );
        assert!(out.owner.is_null());
    }
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_moe_mlp(std::ptr::null(), &mut out) },
        PieForwardStatus::InvalidArgument
    );
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
