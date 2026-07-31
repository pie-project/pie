//! Round-trip tests for the POD boundary.
//!
//! Same discipline as `loader/src/ffi/tests.rs`: build the Rust form, flatten
//! it, and read the result back through the raw pointers a C++ consumer would
//! walk. The assertions target the *synthesized* fields — the name table, the
//! flat operand array, the param packing — because those are the only places
//! the flattening can silently disagree with the trace it came from.

use super::arena::{self, view};
use super::entry::{
    PieForwardLlamaLikeFacts, PieForwardQwen35GdnFacts, PieForwardQwen35MoeMlpFacts,
    PieForwardStatus, pie_forward_release, pie_forward_trace_llama_like,
    pie_forward_trace_qwen3_5_gdn, pie_forward_trace_qwen3_5_moe_mlp,
};
use super::types::*;
use crate::facts::{LlamaLikeFacts, Qwen35GdnFacts, Qwen35MoeMlpFacts};
use crate::family::{llama_like, qwen3_5_gdn_block, qwen3_5_moe_mlp_block};
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
        OpKind::SplitGdn { .. } => PieForwardOpKind::SplitGdn,
        OpKind::CausalConv1d { .. } => PieForwardOpKind::CausalConv1d,
        OpKind::GdnPrep { .. } => PieForwardOpKind::GdnPrep,
        OpKind::GatedDelta { .. } => PieForwardOpKind::GatedDelta,
        OpKind::RmsnormGated { .. } => PieForwardOpKind::RmsnormGated,
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
