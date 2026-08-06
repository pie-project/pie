//! Round-trip tests for the POD boundary.
//!
//! Same discipline as `loader/src/ffi/tests.rs`: build the Rust form, flatten
//! it, and read the result back through the raw pointers a C++ consumer would
//! walk. The assertions target the *synthesized* fields — the name table, the
//! flat operand array, the param packing — because those are the only places
//! the flattening can silently disagree with the trace it came from.

use super::arena::{self, view};
use super::entry::{
    PieForwardLlamaLikeCudaFacts, PieForwardLlamaLikeFacts, PieForwardQwen35CudaFacts,
    PieForwardQwen35FullAttnFacts, PieForwardQwen35GdnFacts, PieForwardQwen35HybridFacts,
    PieForwardQwen35MoeMlpFacts, PieForwardStatus, pie_forward_lower, pie_forward_release,
    pie_forward_trace_llama_like, pie_forward_trace_llama_like_cuda,
    pie_forward_trace_qwen3_5_full_attn, pie_forward_trace_qwen3_5_gdn,
    pie_forward_trace_qwen3_5_hybrid, pie_forward_trace_qwen3_5_hybrid_cuda,
    pie_forward_trace_qwen3_5_moe_mlp,
};
use super::types::*;
use crate::facts::{
    LlamaLikeFacts, Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MoeMlpFacts,
};
use crate::family::{llama_like, qwen3_5_full_attn_block, qwen3_5_gdn_block, qwen3_5_moe_mlp_block};
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
        qkv_bias: u8::from(facts.qkv_bias),
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
        OpKind::SplitGdn { .. } => PieForwardOpKind::SplitGdn,
        OpKind::CausalConv1d { .. } => PieForwardOpKind::CausalConv1d,
        OpKind::GdnPrep { .. } => PieForwardOpKind::GdnPrep,
        OpKind::GatedDelta { .. } => PieForwardOpKind::GatedDelta,
        OpKind::RmsnormGated { .. } => PieForwardOpKind::RmsnormGated,
        OpKind::SplitQGate { .. } => PieForwardOpKind::SplitQGate,
        OpKind::SigmoidGateMul => PieForwardOpKind::SigmoidGateMul,
        OpKind::Launch { .. } => PieForwardOpKind::Launch,
        OpKind::Guard { .. } => PieForwardOpKind::Guard,
        OpKind::HookSite { .. } => PieForwardOpKind::HookSite,
        OpKind::Peel { .. } => PieForwardOpKind::Peel,
        OpKind::AddBias { .. } => PieForwardOpKind::AddBias,
    }
}

/// The (primary) weight name each Rust op carries, if any. GdnPrep's
/// second name (dt_bias) crosses as a param0 name index and is asserted
/// where the GDN fragment is round-tripped.
fn expect_weight(kind: &OpKind) -> Option<&str> {
    match kind {
        OpKind::Embed { weight }
        | OpKind::Matmul { weight, .. }
        | OpKind::Rmsnorm { weight, .. }
        | OpKind::RmsnormPerHead { weight, .. }
        | OpKind::CausalConv1d { weight, .. }
        | OpKind::RmsnormGated { weight }
        | OpKind::LmHead { weight } => Some(weight),
        OpKind::GdnPrep { a_log, .. } => Some(a_log),
        // The Launch "weight" slot carries the KERNEL name.
        OpKind::Launch { kernel, .. } => Some(kernel),
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
        qkv_bias: u8::from(facts.qkv_bias),
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

/// The GDN fragment round-trips: the five appended op kinds, the state
/// layer + kernel params, GdnPrep's TWO names (a_log in the weight slot,
/// dt_bias as the param0 name index), the rank-3 f32 prep/core values, and
/// the five-output prep operand run.
#[test]
fn gdn_fragment_round_trips_through_the_arena() {
    let facts = Qwen35GdnFacts::qwen3_5_0_8b();
    let plan = qwen3_5_gdn_block(&facts);
    let mut pod = arena::build(&plan);

    assert_eq!(view::name(&pod, pod.family), "qwen3_5_gdn_block");
    let ops = view::ops(&pod);
    assert_eq!(ops.len(), plan.ops.len());
    for (rust, c) in plan.ops.iter().zip(ops) {
        assert_eq!(c.kind, expect_kind(&rust.kind), "kind of {rust:?}");
        assert_eq!(view::ids(&pod, c.inputs), &rust.inputs[..]);
        assert_eq!(view::ids(&pod, c.outputs), &rust.outputs[..]);
        assert_eq!(c.selector, PIE_FORWARD_NO_VALUE, "selector of {rust:?}");
        match expect_weight(&rust.kind) {
            Some(weight) => assert_eq!(view::name(&pod, c.weight_name), weight),
            None => assert_eq!(c.weight_name, PIE_FORWARD_NO_NAME),
        }
    }

    let conv = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::CausalConv1d)
        .unwrap();
    assert_eq!(view::name(&pod, conv.weight_name), "layer.0.conv");
    assert_eq!(conv.param0, 0); // state layer
    assert_eq!(conv.param1, facts.conv_kernel);

    // GdnPrep: a_log in the weight slot, dt_bias as a param0 NAME index.
    let prep = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::GdnPrep)
        .unwrap();
    assert_eq!(view::name(&pod, prep.weight_name), "layer.0.a_log");
    assert_eq!(view::name(&pod, prep.param0), "layer.0.dt_bias");
    let prep_outs = view::ids(&pod, prep.outputs);
    assert_eq!(prep_outs.len(), 5);
    let q = view::values(&pod)[prep_outs[0] as usize];
    assert_eq!(q.dtype, PieForwardDType::F32);
    assert_eq!(q.rank, 3);
    assert_eq!(q.dims[0].kind, PieForwardDimKind::Tokens);
    assert_eq!(q.dims[1].value, facts.key_heads);
    assert_eq!(q.dims[2].value, facts.key_head_dim);

    let delta = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::GatedDelta)
        .unwrap();
    assert_eq!(delta.param0, 0); // state layer
    assert_eq!(delta.weight_name, PIE_FORWARD_NO_NAME);
    assert_eq!(view::ids(&pod, delta.inputs), prep_outs);
    let core = view::values(&pod)[view::ids(&pod, delta.outputs)[0] as usize];
    assert_eq!(core.dtype, PieForwardDType::F32);
    assert_eq!(core.rank, 3);
    assert_eq!(core.dims[1].value, facts.value_heads);
    assert_eq!(core.dims[2].value, facts.value_head_dim);

    let gated = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::RmsnormGated)
        .unwrap();
    assert_eq!(view::name(&pod, gated.weight_name), "layer.0.gate_norm");
    let out = view::values(&pod)[view::ids(&pod, gated.outputs)[0] as usize];
    assert_eq!(out.dtype, PieForwardDType::BF16);
    assert_eq!(out.rank, 2);
    assert_eq!(out.dims[1].value, facts.value_width());

    unsafe { arena::release(&mut pod) };
}

/// The fused in-proj binding crosses too: two SplitGdn ops carrying their
/// widths, three matmuls, same 10-op count.
#[test]
fn gdn_fragment_fused_binding_round_trips() {
    let facts = Qwen35GdnFacts {
        fused_in_proj: true,
        ..Qwen35GdnFacts::qwen3_5_0_8b()
    };
    let plan = qwen3_5_gdn_block(&facts);
    let mut pod = arena::build(&plan);
    let ops = view::ops(&pod);
    assert_eq!(ops.len(), 10);
    let splits: Vec<_> = ops
        .iter()
        .filter(|op| op.kind == PieForwardOpKind::SplitGdn)
        .collect();
    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].param0, facts.conv_dim());
    assert_eq!(splits[0].param1, facts.value_width());
    assert_eq!(splits[1].param0, facts.value_heads);
    assert_eq!(splits[1].param1, facts.value_heads);
    assert_eq!(splits[0].weight_name, PIE_FORWARD_NO_NAME);
    unsafe { arena::release(&mut pod) };
}

/// The GDN entry point end to end: C facts in, POD plan out; malformed
/// requests (bad enum, zero heads/dims/kernel, a GQA share that does not
/// divide) answer InvalidArgument.
#[test]
fn gdn_entry_traces_and_validates() {
    let facts = Qwen35GdnFacts::qwen3_5_0_8b();
    let c_facts = PieForwardQwen35GdnFacts {
        hidden: facts.hidden,
        key_heads: facts.key_heads,
        value_heads: facts.value_heads,
        key_head_dim: facts.key_head_dim,
        value_head_dim: facts.value_head_dim,
        conv_kernel: facts.conv_kernel,
        fused_in_proj: u8::from(facts.fused_in_proj),
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
    };
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_gdn(&c_facts, &mut out) },
        PieForwardStatus::Ok
    );
    // The 10-op block `family::tests::gdn_block_op_sequence` pins.
    assert_eq!(out.ops.len, 10);
    unsafe { pie_forward_release(&mut out) };

    for bad in [
        PieForwardQwen35GdnFacts {
            norm_variant: 9,
            ..c_facts
        },
        PieForwardQwen35GdnFacts {
            key_heads: 0,
            ..c_facts
        },
        PieForwardQwen35GdnFacts {
            value_head_dim: 0,
            ..c_facts
        },
        PieForwardQwen35GdnFacts {
            conv_kernel: 0,
            ..c_facts
        },
        // 16 value heads cannot GQA-share 3 key heads.
        PieForwardQwen35GdnFacts {
            key_heads: 3,
            ..c_facts
        },
    ] {
        assert_eq!(
            unsafe { pie_forward_trace_qwen3_5_gdn(&bad, &mut out) },
            PieForwardStatus::InvalidArgument
        );
        assert!(out.owner.is_null());
    }
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_gdn(std::ptr::null(), &mut out) },
        PieForwardStatus::InvalidArgument
    );
}

/// The 0.8B full-attention facts, as a C caller would state them.
fn c_facts_full_attn() -> PieForwardQwen35FullAttnFacts {
    let facts = Qwen35FullAttnFacts::qwen3_5_0_8b();
    PieForwardQwen35FullAttnFacts {
        hidden: facts.hidden,
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dim: facts.rotary_dim,
        fused_qkv: u8::from(facts.fused_qkv),
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
    }
}

/// The full-attention fragment round-trips: the two appended op kinds
/// (SplitQGate's head geometry in its params, SigmoidGateMul's paired
/// operands), the partial rope crossing as Rope's param1, and the per-head
/// norm's Gemma variant crossing as RmsnormPerHead's param1 — the two
/// appended-param-additive fields.
#[test]
fn full_attn_fragment_round_trips_through_the_arena() {
    let facts = Qwen35FullAttnFacts::qwen3_5_0_8b();
    let plan = qwen3_5_full_attn_block(&facts);
    let mut pod = arena::build(&plan);

    assert_eq!(view::name(&pod, pod.family), "qwen3_5_full_attn_block");
    let ops = view::ops(&pod);
    assert_eq!(ops.len(), plan.ops.len());
    for (rust, c) in plan.ops.iter().zip(ops) {
        assert_eq!(c.kind, expect_kind(&rust.kind), "kind of {rust:?}");
        assert_eq!(view::ids(&pod, c.inputs), &rust.inputs[..]);
        assert_eq!(view::ids(&pod, c.outputs), &rust.outputs[..]);
        assert_eq!(c.selector, PIE_FORWARD_NO_VALUE, "selector of {rust:?}");
        match expect_weight(&rust.kind) {
            Some(weight) => assert_eq!(view::name(&pod, c.weight_name), weight),
            None => assert_eq!(c.weight_name, PIE_FORWARD_NO_NAME),
        }
    }

    let split = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::SplitQGate)
        .unwrap();
    assert_eq!(split.param0, facts.q_heads);
    assert_eq!(split.param1, facts.head_dim);
    assert_eq!(split.weight_name, PIE_FORWARD_NO_NAME);
    let split_outs = view::ids(&pod, split.outputs);
    assert_eq!(split_outs.len(), 2);
    for &id in split_outs {
        let v = view::values(&pod)[id as usize];
        assert_eq!(v.rank, 2);
        assert_eq!(v.dims[1].value, facts.q_width());
    }

    // Per-head norm: head_dim in param0, the GEMMA variant in param1 —
    // non-zero for the first time, which is what pins the appended param.
    let per_head = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::RmsnormPerHead)
        .unwrap();
    assert_eq!(per_head.param0, facts.head_dim);
    assert_eq!(
        per_head.param1,
        PieForwardNormVariant::Gemma as u32
    );

    // Partial rope: kind in param0, the rotary width in param1.
    let rope = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::Rope)
        .unwrap();
    assert_eq!(rope.param0, PieForwardRopeKind::Standard as u32);
    assert_eq!(rope.param1, facts.rotary_dim);

    // The output gate consumes attention's output and the gate leg.
    let attn = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::Attention)
        .unwrap();
    let gate_mul = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::SigmoidGateMul)
        .unwrap();
    assert_eq!(gate_mul.weight_name, PIE_FORWARD_NO_NAME);
    let gm_in = view::ids(&pod, gate_mul.inputs);
    assert_eq!(gm_in[0], view::ids(&pod, attn.outputs)[0]);
    assert_eq!(gm_in[1], split_outs[1]);

    unsafe { arena::release(&mut pod) };
}

/// The full-attention entry point end to end: C facts in, POD plan out;
/// malformed requests (bad enum, zero heads, a rotary width of zero or
/// wider than the head, a GQA share that does not divide) answer
/// InvalidArgument.
#[test]
fn full_attn_entry_traces_and_validates() {
    let c_facts = c_facts_full_attn();
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_full_attn(&c_facts, &mut out) },
        PieForwardStatus::Ok
    );
    // The 12-op block `family::tests::full_attn_block_op_sequence` pins.
    assert_eq!(out.ops.len, 12);
    unsafe { pie_forward_release(&mut out) };

    for bad in [
        PieForwardQwen35FullAttnFacts {
            norm_variant: 9,
            ..c_facts
        },
        PieForwardQwen35FullAttnFacts {
            q_heads: 0,
            ..c_facts
        },
        PieForwardQwen35FullAttnFacts {
            rotary_dim: 0,
            ..c_facts
        },
        // Wider than the head: rotating channels that do not exist.
        PieForwardQwen35FullAttnFacts {
            rotary_dim: 512,
            ..c_facts
        },
        // 3 query heads cannot GQA-share 2 kv heads.
        PieForwardQwen35FullAttnFacts {
            q_heads: 3,
            ..c_facts
        },
    ] {
        assert_eq!(
            unsafe { pie_forward_trace_qwen3_5_full_attn(&bad, &mut out) },
            PieForwardStatus::InvalidArgument
        );
        assert!(out.owner.is_null());
    }
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_full_attn(std::ptr::null(), &mut out) },
        PieForwardStatus::InvalidArgument
    );
}

/// The hybrid entry point end to end: the flattened C facts (nested
/// sub-facts, the mlp_is_moe tag) trace the 351-op 0.8B model; malformed
/// requests — zero layers/interval, a dense MLP with no width, sub-facts
/// disagreeing on hidden — answer InvalidArgument.
#[test]
fn hybrid_entry_traces_and_validates() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let gdn = &facts.gdn;
    let c_gdn = PieForwardQwen35GdnFacts {
        hidden: gdn.hidden,
        key_heads: gdn.key_heads,
        value_heads: gdn.value_heads,
        key_head_dim: gdn.key_head_dim,
        value_head_dim: gdn.value_head_dim,
        conv_kernel: gdn.conv_kernel,
        fused_in_proj: u8::from(gdn.fused_in_proj),
        norm_variant: PieForwardNormVariant::from(gdn.norm_variant) as u32,
    };
    let c_moe_unused = PieForwardQwen35MoeMlpFacts {
        hidden: 0,
        num_experts: 0,
        top_k: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        norm_variant: 0,
    };
    let c_facts = PieForwardQwen35HybridFacts {
        layers: facts.layers,
        full_attn_interval: facts.full_attn_interval,
        vocab: facts.vocab,
        tied_embeddings: u8::from(facts.tied_embeddings),
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
        attn: c_facts_full_attn(),
        gdn: c_gdn,
        mlp_is_moe: 0,
        dense_intermediate: 3584,
        // Ignored under mlp_is_moe == 0 — and deliberately garbage, so the
        // test also pins that a dense request never reads the moe leg.
        moe: c_moe_unused,
    };
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_hybrid(&c_facts, &mut out) },
        PieForwardStatus::Ok
    );
    // The count `family::tests::hybrid_full_plan_shape` pins.
    assert_eq!(out.ops.len, 18 * 14 + 6 * 16 + 3);
    let ops = view::ops(&out);
    assert_eq!(view::name(&out, out.family), "qwen3_5_hybrid");
    // Layer 3 is the first full-attention layer; its rope is partial.
    let rope3 = ops
        .iter()
        .find(|op| op.layer == 3 && op.kind == PieForwardOpKind::Rope)
        .unwrap();
    assert_eq!(rope3.param1, 64);
    // Layer 0 is GDN: its conv addresses the layer-0 recurrent slab.
    let conv0 = ops
        .iter()
        .find(|op| op.layer == 0 && op.kind == PieForwardOpKind::CausalConv1d)
        .unwrap();
    assert_eq!(view::name(&out, conv0.weight_name), "layer.0.conv");
    unsafe { pie_forward_release(&mut out) };

    for bad in [
        PieForwardQwen35HybridFacts { layers: 0, ..c_facts },
        PieForwardQwen35HybridFacts {
            full_attn_interval: 0,
            ..c_facts
        },
        PieForwardQwen35HybridFacts {
            dense_intermediate: 0,
            ..c_facts
        },
        // The gdn sub-facts disagree with attn on hidden.
        PieForwardQwen35HybridFacts {
            gdn: PieForwardQwen35GdnFacts {
                hidden: 2048,
                ..c_gdn
            },
            ..c_facts
        },
        // Under mlp_is_moe the (garbage) moe leg IS read, and refused.
        PieForwardQwen35HybridFacts {
            mlp_is_moe: 1,
            ..c_facts
        },
    ] {
        assert_eq!(
            unsafe { pie_forward_trace_qwen3_5_hybrid(&bad, &mut out) },
            PieForwardStatus::InvalidArgument
        );
        assert!(out.owner.is_null());
    }
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_hybrid(std::ptr::null(), &mut out) },
        PieForwardStatus::InvalidArgument
    );
}

/// The 0.8B hybrid facts, as a C caller would state them — shared by the
/// lowered-trace round-trip tests.
fn c_facts_hybrid_0_8b() -> PieForwardQwen35HybridFacts {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let gdn = &facts.gdn;
    PieForwardQwen35HybridFacts {
        layers: facts.layers,
        full_attn_interval: facts.full_attn_interval,
        vocab: facts.vocab,
        tied_embeddings: u8::from(facts.tied_embeddings),
        norm_variant: PieForwardNormVariant::from(facts.norm_variant) as u32,
        attn: c_facts_full_attn(),
        gdn: PieForwardQwen35GdnFacts {
            hidden: gdn.hidden,
            key_heads: gdn.key_heads,
            value_heads: gdn.value_heads,
            key_head_dim: gdn.key_head_dim,
            value_head_dim: gdn.value_head_dim,
            conv_kernel: gdn.conv_kernel,
            fused_in_proj: u8::from(gdn.fused_in_proj),
            norm_variant: PieForwardNormVariant::from(gdn.norm_variant) as u32,
        },
        mlp_is_moe: 0,
        dense_intermediate: 3584,
        moe: PieForwardQwen35MoeMlpFacts {
            hidden: 0,
            num_experts: 0,
            top_k: 0,
            moe_intermediate: 0,
            shared_expert_intermediate: 0,
            norm_variant: 0,
        },
    }
}

/// The synthetic 0.8B CUDA facts, as a C caller would state them.
fn c_cuda_facts_synthetic() -> PieForwardQwen35CudaFacts {
    PieForwardQwen35CudaFacts {
        state_bf16: 1,
        warp_tiled: 1,
        warp_tiled_max: 64,
        cached_max: 4096,
        verify_stash: 1,
    }
}

/// The lowered qwen3_5 decode trace crosses the boundary intact,
/// `lowered_trace_round_trips_through_the_arena`-style: the GDN Launch
/// wire form — kernel symbol in the weight slot, the conv weight in
/// `aux_names`, the RecurrentState mark as param0=2 with the state layer
/// in param1 — plus the full-attention layers' KV-write Guard and
/// FlashInfer decode dispatch, with the semantic conv/recurrence/append/
/// attention kinds absent. An out-of-range class (4) answers
/// InvalidArgument.
#[test]
fn lowered_qwen3_5_trace_round_trips_through_the_arena() {
    let c_facts = c_facts_hybrid_0_8b();
    let cuda = c_cuda_facts_synthetic();
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe {
            pie_forward_trace_qwen3_5_hybrid_cuda(&c_facts, &cuda, /*Decode=*/ 0, &mut out)
        },
        PieForwardStatus::Ok
    );
    assert_eq!(view::name(&out, out.family), "qwen3_5_hybrid.cuda.decode");
    let ops = view::ops(&out);

    // Layer 0 is GDN: the conv update Launch — kernel symbol in the
    // weight slot, the conv weight name in aux_names, the RecurrentState
    // mark as param0=2 with the state layer in param1.
    let conv = ops
        .iter()
        .find(|op| {
            op.layer == 0
                && op.kind == PieForwardOpKind::Launch
                && view::name(&out, op.weight_name) == "launch_causal_conv1d_update_batched_bf16"
        })
        .expect("decode class states the batched conv update");
    let aux = view::ids(&out, conv.aux_names);
    assert_eq!(aux.len(), 1);
    assert_eq!(view::name(&out, aux[0]), "layer.0.conv");
    assert_eq!(conv.param0, 2); // recurrent-state store...
    assert_eq!(conv.param1, 0); // ...of layer 0

    // The decode recurrence step (the fixture's gqa=false, state_bf16
    // variant): five prep operands in, one core value out, same state
    // mark, no weights.
    let step = ops
        .iter()
        .find(|op| {
            op.layer == 0
                && op.kind == PieForwardOpKind::Launch
                && view::name(&out, op.weight_name)
                    == "launch_recurrent_gated_delta_step_batched_state_bf16"
        })
        .expect("decode class states the batched recurrence step");
    assert_eq!(step.param0, 2);
    assert_eq!(step.param1, 0);
    assert_eq!(step.aux_names.len, 0);
    assert_eq!(view::ids(&out, step.inputs).len(), 5);
    assert_eq!(view::ids(&out, step.outputs).len(), 1);

    // Layer 3 is full attention: the KV-write Guard (one HasWriteDesc
    // arm of one op, one else op) and the FlashInfer decode dispatch
    // marking the layer-3 KV cache (param0=1).
    let guard = ops
        .iter()
        .find(|op| op.kind == PieForwardOpKind::Guard)
        .expect("full-attention layers carry the KV-write guard");
    assert_eq!(guard.param0, 1); // one arm
    assert_eq!(view::ids(&out, guard.aux_names), &[0, 0, 1, 1]);
    let write = ops
        .iter()
        .find(|op| {
            op.layer == 3
                && op.kind == PieForwardOpKind::Launch
                && view::name(&out, op.weight_name) == "launch_write_kv_explicit_bf16"
        })
        .expect("the guard's then-region states the explicit write");
    assert_eq!(write.param0, 1); // kv-cache state...
    assert_eq!(write.param1, 3); // ...of layer 3
    let attn = ops
        .iter()
        .find(|op| {
            op.layer == 3
                && op.kind == PieForwardOpKind::Launch
                && view::name(&out, op.weight_name) == "dispatch_attention_flashinfer_decode"
        })
        .expect("decode class states the FlashInfer decode dispatch");
    assert_eq!(attn.param0, 1);
    assert_eq!(attn.param1, 3);

    // No semantic leftovers where the class arms stated kernels.
    assert!(!ops.iter().any(|op| matches!(
        op.kind,
        PieForwardOpKind::CausalConv1d
            | PieForwardOpKind::GatedDelta
            | PieForwardOpKind::KvAppend
            | PieForwardOpKind::Attention
    )));
    unsafe { pie_forward_release(&mut out) };

    // Past the last qwen3_5 class (FrozenVerify = 4 accepted since its
    // slice; the mask classes 5/6 are llama_like's until the qwen3_5
    // masked slice): malformed, not defaulted.
    let mut out2 = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_qwen3_5_hybrid_cuda(&c_facts, &cuda, 5, &mut out2) },
        PieForwardStatus::InvalidArgument
    );
    assert!(out2.owner.is_null());
}

/// The service classes cross the boundary (rung 4c-iv). CommitAdvance
/// (class 2): the stash PSEUDO-SYMBOL in the Launch's weight slot, no
/// inputs, THREE outputs (qkv/a/b), the RecurrentState mark as param0=2
/// with the state layer in param1 — and the 72-op shape (18 linear
/// layers x 4, no embed, no lm_head, nothing on the full-attention
/// layers). StateOnly (class 3): the prefill backbone that simply ends
/// after the last layer — no LmHead, no layer-less final norm.
#[test]
fn service_class_traces_round_trip_through_the_arena() {
    let c_facts = c_facts_hybrid_0_8b();
    let cuda = c_cuda_facts_synthetic();

    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe {
            pie_forward_trace_qwen3_5_hybrid_cuda(&c_facts, &cuda, /*CommitAdvance=*/ 2, &mut out)
        },
        PieForwardStatus::Ok
    );
    assert_eq!(
        view::name(&out, out.family),
        "qwen3_5_hybrid.cuda.commit_advance"
    );
    let ops = view::ops(&out);
    // Per linear layer: stash load, conv, prep, recurrence + the two
    // hook sites the hand-written replay passes through (A4).
    assert_eq!(ops.len(), 18 * 6);
    assert!(ops.iter().all(|op| op.kind == PieForwardOpKind::Launch
        || op.kind == PieForwardOpKind::GdnPrep
        || op.kind == PieForwardOpKind::HookSite));
    // Nothing on the full-attention layers (3, 7, ... are skipped).
    assert!(!ops.iter().any(|op| op.layer == 3));

    // The stash load: pseudo-symbol in the weight slot, no inputs, three
    // outputs, the layer-0 RecurrentState mark.
    let load = ops
        .iter()
        .find(|op| {
            op.layer == 0
                && op.kind == PieForwardOpKind::Launch
                && view::name(&out, op.weight_name) == "qwen35_verify_stash_load"
        })
        .expect("the stash-configured commit pass states the stash load");
    assert_eq!(load.inputs.len, 0);
    let load_outs = view::ids(&out, load.outputs);
    assert_eq!(load_outs.len(), 3);
    assert_eq!(load.param0, 2); // recurrent-state store...
    assert_eq!(load.param1, 0); // ...of layer 0
    // ...and gdn_prep consumes exactly that triple (dataflow complete).
    let prep = ops
        .iter()
        .find(|op| op.layer == 0 && op.kind == PieForwardOpKind::GdnPrep)
        .unwrap();
    let prep_ins = view::ids(&out, prep.inputs);
    assert_eq!(prep_ins[1], load_outs[1]); // a
    assert_eq!(prep_ins[2], load_outs[2]); // b
    // The GEMMs are skipped: no Matmul anywhere in the pass.
    assert!(!ops.iter().any(|op| op.kind == PieForwardOpKind::Matmul));
    unsafe { pie_forward_release(&mut out) };

    // StateOnly: the backbone ends with the last layer — no LmHead, and
    // every op carries a layer tag (the layer-less final norm is gone).
    assert_eq!(
        unsafe {
            pie_forward_trace_qwen3_5_hybrid_cuda(&c_facts, &cuda, /*StateOnly=*/ 3, &mut out)
        },
        PieForwardStatus::Ok
    );
    assert_eq!(
        view::name(&out, out.family),
        "qwen3_5_hybrid.cuda.state_only"
    );
    let ops = view::ops(&out);
    assert!(!ops.iter().any(|op| op.kind == PieForwardOpKind::LmHead));
    let last = ops.last().unwrap();
    assert_eq!(last.layer, 23, "state_only must end inside the last layer");
    unsafe { pie_forward_release(&mut out) };
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

/// The lowered trace crosses the boundary intact: the `Launch` wire form
/// — kernel symbol in the weight slot, consumed weight names in
/// `aux_names` (signature order), the state mark in the params — plus
/// the fused decode shape itself: 28 fused posts, ONE table build
/// consumed by all of them as an operand. Since A1 (the class-collapse
/// amendment) the Decode trace ALSO carries the HasCustomMask guard's
/// mask arm per layer — the general QKV sequence (SplitQkv appears as a
/// region op) and the custom-mask dispatch — beside the fused else-arm;
/// the launch count pins both arms.
#[test]
fn lowered_trace_round_trips_through_the_arena() {
    let facts = c_facts_qwen3();
    let cuda = PieForwardLlamaLikeCudaFacts {
        xqa_decode: 1,
        decode_fused_post: 1,
        rope_table: 1,
        force_prefill_path: 0,
        head_dim_padded: 0,
        gate_up_fused: 1,
    };
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like_cuda(&facts, &cuda, /*Decode=*/ 0, &mut out) },
        PieForwardStatus::Ok
    );
    let ops = view::ops(&out);

    let launches: Vec<_> = ops
        .iter()
        .filter(|op| op.kind == PieForwardOpKind::Launch)
        .collect();
    // 1 table build + per layer: the mask arm's 5 (fused qk-norm+rope,
    // the lora correction, the write guard's two regions, the custom
    // dispatch) + the lora arm's 5 (the general sequence + correction +
    // XQA) + the else arm's 5 — the Peel's prefix (the fused-post
    // region form) + tail (fused qk-norm+rope, the write guard's two
    // regions) — plus the plain XQA launch (this fixture is XQA-on; no
    // capture variant, so no WantsAttnScore guard) = 421, plus ONE
    // swiglu per layer now that the activation states its kernel from
    // the gate_up binding fact instead of leaving the choice to a
    // workspace read (+28) = 449. The trace's op TOTAL is unchanged:
    // every one of those was already a `Swiglu` statement.
    assert_eq!(launches.len(), 449);

    let table = launches[0];
    assert_eq!(view::name(&out, table.weight_name), "launch_rope_standard_table");
    assert_eq!(table.param0, 0); // no implicit state
    assert_eq!(table.aux_names.len, 0);
    let table_out = view::ids(&out, table.outputs)[0];

    let posts: Vec<_> = launches
        .iter()
        .filter(|op| {
            view::name(&out, op.weight_name) == "launch_qkv_decode_qk_norm_rope_write_kv_bf16"
        })
        .collect();
    assert_eq!(posts.len(), 28);
    let post = posts[0];
    let aux = view::ids(&out, post.aux_names);
    assert_eq!(view::name(&out, aux[0]), "layer.0.q_norm");
    assert_eq!(view::name(&out, aux[1]), "layer.0.k_norm");
    assert_eq!(post.param0, 1); // kv-cache state...
    assert_eq!(post.param1, 0); // ...of layer 0
    // Every fused post consumes THE table value as its second operand.
    for post in &posts {
        assert_eq!(view::ids(&out, post.inputs)[1], table_out);
    }

    let attn = launches
        .iter()
        .find(|op| {
            view::name(&out, op.weight_name) == "launch_attention_xqa_decode_bf16_prepared"
        })
        .expect("decode class states XQA");
    assert_eq!(attn.param0, 1);

    // The lowered general arm states its kernels: no semantic Rope or
    // KvAppend anywhere (SplitQkv is legitimately present — the mask
    // arm's region carries the general QKV sequence since A1).
    assert!(!ops.iter().any(|op| matches!(
        op.kind,
        PieForwardOpKind::Rope | PieForwardOpKind::KvAppend
    )));
    assert!(ops.iter().any(|op| op.kind == PieForwardOpKind::SplitQkv));
    // The per-layer guard chains: 28 outer HasCustomMask/HasLora
    // chains (value-producing) + 28×3 nested HasWriteDesc (mask arm,
    // lora arm, the Peel's tail) + 28×2 nested HasLora inside the two
    // general-arm bodies (no WantsAttnScore under XQA).
    let guards = ops
        .iter()
        .filter(|op| op.kind == PieForwardOpKind::Guard)
        .count();
    assert_eq!(guards, 168);
    // The one body's sites (A3): two per layer — argument no-ops on an
    // unhooked fire — and one Peel per layer splitting the fused prefix
    // from the hook-visible tail at fast_rows.
    // Sites in every arm that runs a body: the mask arm (masked+hooked
    // composes), the lora arm, and the plain else — 2 × 3 per layer.
    let sites = ops
        .iter()
        .filter(|op| op.kind == PieForwardOpKind::HookSite)
        .count();
    assert_eq!(sites, 168);
    let peels: Vec<_> = ops
        .iter()
        .filter(|op| op.kind == PieForwardOpKind::Peel)
        .collect();
    assert_eq!(peels.len(), 28);
    // Region lengths ride param0/param1: prefix = the fused region
    // launch alone; tail = SplitQkv + fused qk-norm+rope + the write
    // guard chain (guard op + two single-launch regions).
    assert_eq!(peels[0].param0, 1);
    assert_eq!(peels[0].param1, 5);
    unsafe { pie_forward_release(&mut out) };

    // An out-of-range class is a malformed request, not a default.
    let mut out2 = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like_cuda(&facts, &cuda, 2, &mut out2) },
        PieForwardStatus::InvalidArgument
    );
}


// ── The lowering across the C ABI (the shadow's Rust half) ─────────────

fn lowered_view(out: &PieForwardLowered) -> Vec<(String, u32, u32, u32)> {
    if out.launches.is_null() {
        return Vec::new();
    }
    let launches = unsafe { std::slice::from_raw_parts(out.launches, out.launches_len) };
    let names = unsafe { std::slice::from_raw_parts(out.kernel_names, out.kernel_names_len) };
    let bytes =
        unsafe { std::slice::from_raw_parts(out.kernel_name_bytes.ptr, out.kernel_name_bytes.len) };
    launches
        .iter()
        .map(|l| {
            let n = names[l.kernel_name as usize];
            let name = std::str::from_utf8(
                &bytes[n.offset as usize..n.offset as usize + n.len as usize],
            )
            .expect("kernel names are utf8")
            .to_string();
            (name, l.at_op, l.row_lo, l.row_hi)
        })
        .collect()
}

fn plain_c_rows(n: usize) -> Vec<PieForwardRow> {
    vec![
        PieForwardRow {
            samples: 1,
            depth_k: -1,
            ..PieForwardRow::default()
        };
        n
    ]
}

fn traced_cuda_decode() -> PieForwardPlan {
    let facts = c_facts_qwen3();
    let cuda = PieForwardLlamaLikeCudaFacts {
        xqa_decode: 0,
        decode_fused_post: 1,
        rope_table: 1,
        force_prefill_path: 0,
        head_dim_padded: 0,
        gate_up_fused: 1,
    };
    let mut out = PieForwardPlan::default();
    assert_eq!(
        unsafe { pie_forward_trace_llama_like_cuda(&facts, &cuda, /*Decode=*/ 0, &mut out) },
        PieForwardStatus::Ok
    );
    out
}

/// The flat launch list crosses the ABI, names its kernels, and covers
/// every row of a plain fire — the shadow comparison's input.
#[test]
fn the_lowering_crosses_the_abi() {
    let mut plan = traced_cuda_decode();
    let rows = plain_c_rows(4);
    let mut out = PieForwardLowered::default();
    assert_eq!(
        unsafe { pie_forward_lower(&mut plan, rows.as_ptr(), rows.len(), 0, &mut out) },
        PieForwardStatus::Ok
    );
    assert_eq!(out.uncovered, PieForwardUncovered::None);
    assert!(out.arena_bytes > 0);

    let view = lowered_view(&out);
    assert!(!view.is_empty());
    // The body's rectangles cover the whole fire; the epilogue's run in
    // the Requests row space, which for an all-sampled fire is the same
    // four rows.
    assert!(view.iter().all(|(_, _, lo, hi)| *lo == 0 && *hi == 4));
    // Both the stated kernels and the semantic statements' launchers are
    // named — the list is what the fire RUNS, not what it states.
    assert!(view.iter().any(|(k, ..)| k == "dispatch_attention_flashinfer_decode"));
    assert!(view.iter().any(|(k, ..)| k == "launch_chunked_swiglu_bf16"));
    assert!(view.iter().any(|(k, ..)| k == "gemm_act_x_w"));
    // Every rectangle points at a real statement.
    let ops = view::ops(&plan).len() as u32;
    assert!(view.iter().all(|(_, at, ..)| *at < ops));

    unsafe { pie_forward_release(&mut plan) };
}

/// A row order the seriation could not have produced is refused across
/// the ABI too — and refusing leaves an EMPTY list rather than a partial
/// one, so a caller that ignores the code cannot read half a fire.
#[test]
fn an_uncoverable_fire_crosses_as_a_reason() {
    let mut plan = traced_cuda_decode();
    let mut rows = plain_c_rows(8);
    rows[1].custom_mask = 1;
    rows[5].custom_mask = 1;
    let mut out = PieForwardLowered::default();
    assert_eq!(
        unsafe { pie_forward_lower(&mut plan, rows.as_ptr(), rows.len(), 0, &mut out) },
        PieForwardStatus::Ok
    );
    assert_eq!(out.uncovered, PieForwardUncovered::Discontiguous);
    assert_eq!(out.launches_len, 0);
    assert!(lowered_view(&out).is_empty());

    unsafe { pie_forward_release(&mut plan) };
}

/// The masked suffix reaches the driver as its own rectangles — the
/// thing the flat ABI buys, seen from C.
#[test]
fn the_mask_split_crosses_as_rectangles() {
    let mut plan = traced_cuda_decode();
    let mut rows = plain_c_rows(8);
    for r in &mut rows[6..] {
        r.custom_mask = 1;
    }
    let mut out = PieForwardLowered::default();
    assert_eq!(
        unsafe { pie_forward_lower(&mut plan, rows.as_ptr(), rows.len(), 0, &mut out) },
        PieForwardStatus::Ok
    );
    let view = lowered_view(&out);
    assert!(view.iter().any(|(_, _, lo, hi)| *lo == 6 && *hi == 8));
    assert!(view.iter().any(|(_, _, lo, hi)| *lo == 0 && *hi == 6));

    unsafe { pie_forward_release(&mut plan) };
}

/// A released plan is not a lowering source, and asking is a no-op
/// rather than a dereference of freed storage.
#[test]
fn a_released_plan_lowers_to_nothing() {
    let mut plan = traced_cuda_decode();
    unsafe { pie_forward_release(&mut plan) };
    let rows = plain_c_rows(2);
    let mut out = PieForwardLowered::default();
    assert_eq!(
        unsafe { pie_forward_lower(&mut plan, rows.as_ptr(), rows.len(), 0, &mut out) },
        PieForwardStatus::Ok
    );
    assert_eq!(out.launches_len, 0);
}
