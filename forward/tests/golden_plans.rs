//! Golden traced forms.
//!
//! The unit tests argue that individual ops are right; this one checks the
//! thing a driver actually receives: the whole traced form, byte for byte,
//! for the family configurations the executor runs. Its value is not that
//! any particular op is *right* — it is that a change to the traced form
//! cannot happen quietly. The executor's op→kernel mapping is a
//! deterministic walk of this structure, so pinning the structure pins the
//! emitted kernel sequence too: this is the CI-enforceable half of the
//! parity story, the half that needs no GPU
//! (`driver/cuda/src/model/llama_like/declared_forward.cpp` is the other).
//!
//! Regenerate after an intended change:
//!
//! ```text
//! UPDATE_GOLDEN=1 cargo test -p pie-forward --test golden_plans
//! ```
//!
//! The pattern is `loader/tests/golden_plans.rs`, including the rule that a
//! regenerated golden is a diff a human reads and approves.

use std::path::PathBuf;

use pie_forward::family::{
    llama_like, llama_like_cuda, qwen3_5_full_attn_block, qwen3_5_gdn_block, qwen3_5_hybrid,
    qwen3_5_moe_mlp_block,
};
use pie_forward::{
    FireClass, ForwardPlan, LlamaLikeCudaFacts, LlamaLikeFacts, Qwen35FullAttnFacts,
    Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MoeMlpFacts,
};

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.json"))
}

fn check(name: &str, facts: &LlamaLikeFacts) {
    check_plan(name, &llama_like(facts));
}

fn check_plan(name: &str, plan: &ForwardPlan) {
    let fresh = serde_json::to_string_pretty(plan).expect("serialize plan");
    let path = golden_path(name);
    if std::env::var_os("UPDATE_GOLDEN").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir golden");
        std::fs::write(&path, &fresh).expect("write golden");
        return;
    }
    let stored = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "no golden at {}: {err}.\n\
             If this plan is new, regenerate with UPDATE_GOLDEN=1.",
            path.display()
        )
    });
    assert_eq!(
        stored, fresh,
        "traced form for `{name}` changed.\n\
         If the change is intended, regenerate with UPDATE_GOLDEN=1 and \
         review the diff — the executor emits kernels by walking exactly \
         this structure."
    );
}

/// The parity model: what `declared_forward.cpp` ran token-identical to the
/// hand-written pass (stage3 commit trail).
#[test]
fn qwen3_0_6b() {
    check("qwen3_0_6b", &LlamaLikeFacts::qwen3_0_6b());
}

/// The second declared configuration (stage 3 rung d): no qk-norm — the
/// RmsnormPerHead pair vanishes from every layer — an untied lm_head, and
/// the unfused QKV binding (the contract splits the checkpoint's fused
/// qkv_proj and the dense join cannot re-fuse the bands). The 96 → 128
/// head-dim pad, the 2047 sliding window and the null rope scaling are
/// backend cfg, invisible here by design.
#[test]
fn phi3_mini() {
    check("phi3_mini", &LlamaLikeFacts::phi3_mini());
}

/// The third declared configuration (Mistral-7B-Instruct-v0.3): the fused
/// QKV binding (the checkpoint's raw q/k/v re-fused by the dense join)
/// with no qk-norm — the branch combination qwen3 and phi3 between them
/// never traced. Untied lm_head; rope theta 1e6, null sliding window and
/// null rope scaling are backend cfg, invisible here by design.
#[test]
fn mistral_7b_v03() {
    check("mistral_7b_v03", &LlamaLikeFacts::mistral_7b_v03());
}

/// The fourth declared configuration (OLMo-2-0425-1B-Instruct), and the
/// first that extends the declaration itself: post-norm placement (each
/// sub-layer's matmul(beta=0) → rmsnorm → residual_add triplet replaces the
/// pre-norm accumulate GEMM) and the global qk-norm (a plain row Rmsnorm
/// over the flattened `[heads * head_dim]` q/k — the checkpoint's
/// q_norm/k_norm are `[2048]`, not `[128]`). Unfused QKV because
/// `bind_olmo3` binds the per-projection views, never the dense join's
/// fused bank; untied lm_head; rope theta 5e5 and `attention_bias: false`
/// are backend cfg / absent branches, invisible here by design.
#[test]
fn olmo2_1b() {
    check("olmo2_1b", &LlamaLikeFacts::olmo2_1b());
}

/// The unfused-binding variant: three projection matmuls, no SplitQkv. Kept
/// golden so the binding-driven divergence stays a reviewed artifact rather
/// than an emergent one.
#[test]
fn qwen3_0_6b_unfused_qkv() {
    check(
        "qwen3_0_6b_unfused_qkv",
        &LlamaLikeFacts {
            fused_qkv: false,
            ..LlamaLikeFacts::qwen3_0_6b()
        },
    );
}

/// The first `dyn` traced form, and the first FRAGMENT golden: one
/// qwen3_5_moe MoE MLP block (Qwen3.5-35B-A3B dims), router → topk →
/// grouped gate_up → swiglu → grouped down → weighted sum, plus the
/// shared-expert path behind its sigmoid scalar gate. Everything the dyn
/// vocabulary added — `selector` fields, `dyn_axis` markers, rank-3
/// route-expanded shapes, the `{e}` weight templates — appears here and,
/// per the serde-additive rule, NOWHERE in the dense goldens above, which
/// this change leaves byte-untouched.
#[test]
fn qwen3_5_moe_mlp_35b_a3b() {
    check_plan(
        "qwen3_5_moe_mlp_35b_a3b",
        &qwen3_5_moe_mlp_block(&Qwen35MoeMlpFacts::qwen3_5_35b_a3b()),
    );
}

/// The second fragment golden: one qwen3_5 GDN linear-attention block
/// (Qwen3.5-0.8B dims, the default unfused in-proj binding) — attn_norm →
/// four in-projections → causal conv → gdn prep → gated-delta recurrence →
/// z-gated norm → o_proj accumulate. The first traced form whose ops
/// address PER-REQUEST state (the conv/recurrent slabs behind
/// `CausalConv1d`/`GatedDelta`'s layer, plan §5.4); everything the GDN
/// vocabulary added appears here and nowhere in the goldens above, which
/// this change leaves byte-untouched.
#[test]
fn qwen3_5_gdn_0_8b() {
    check_plan(
        "qwen3_5_gdn_0_8b",
        &qwen3_5_gdn_block(&Qwen35GdnFacts::qwen3_5_0_8b()),
    );
}

/// The third fragment golden: one qwen3_5 full-attention block
/// (Qwen3.5-0.8B dims, the default unfused qgkv binding) — attn_norm →
/// 2×-wide q + k/v projections → per-head [query|gate] de-interleave →
/// Gemma-fold per-head q/k norms → PARTIAL rope (64 of 256 channels) →
/// kv-append/attention → sigmoid output gate → o_proj accumulate.
/// Everything the full-attention vocabulary added — `SplitQGate`,
/// `SigmoidGateMul`, `Rope.partial`, `RmsnormPerHead.variant` — appears
/// here and nowhere in the goldens above, which this change leaves
/// byte-untouched.
#[test]
fn qwen3_5_full_attn_0_8b() {
    check_plan(
        "qwen3_5_full_attn_0_8b",
        &qwen3_5_full_attn_block(&Qwen35FullAttnFacts::qwen3_5_0_8b()),
    );
}

/// The first whole-model golden beyond llama_like: the qwen3_5 HYBRID
/// (Qwen3.5-0.8B — 24 layers on the 3:1 linear:full schedule, dense MLP,
/// tied lm_head over the 248320 vocab). Each layer's attention ops are the
/// standalone fragments' by construction (one shared body each, pinned by
/// the family unit tests), so this golden pins the COMPOSITION: the layer
/// schedule, the per-layer norm/MLP bracketing, and the embed/final-norm/
/// lm_head frame, 351 ops in all.
#[test]
fn qwen3_5_hybrid_0_8b() {
    check_plan(
        "qwen3_5_hybrid_0_8b",
        &qwen3_5_hybrid(&Qwen35HybridFacts::qwen3_5_0_8b()),
    );
}

/// The first LOWERED goldens (north-star-dsl.md): the SAME llama_like
/// text, traced with the CUDA backend facts and a fire class in hand, so
/// the class arms run and the traced form states kernels. Decode: the
/// fused-QKV arm replaces SplitQkv + RmsnormPerHead×2 + Rope + KvAppend
/// with one `QkvDecodeFusedPost` per layer (layer 0 preceded by the
/// once-per-fire `RopeTableBuild` — the hand-written runtime latch, made
/// trace-time by the unrolled layer loop), and every Attention states
/// `XqaDecode`. This golden IS the decode launch list — the thing rung 2's
/// dumb interpreter walks and rung 3's emitter transliterates to C++.
#[test]
fn qwen3_0_6b_cuda_decode() {
    check_plan(
        "qwen3_0_6b.cuda.decode",
        &llama_like_cuda(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Decode,
        ),
    );
}

/// The prefill-class lowering of the same text: the general arm
/// throughout (no fused post — its predicate is decode-only), every
/// Attention stating `PrefillPlanned`. Structurally the semantic trace
/// plus kernel statements; the golden pins exactly that relationship.
#[test]
fn qwen3_0_6b_cuda_prefill() {
    check_plan(
        "qwen3_0_6b.cuda.prefill",
        &llama_like_cuda(
            &LlamaLikeFacts::qwen3_0_6b(),
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            FireClass::Prefill,
        ),
    );
}
