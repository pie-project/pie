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
    gemma4_cuda, qwen3_5_hybrid_cuda, qwen3_5_moe_mlp_block, qwen3_5_moe_mlp_block_cuda,
};
use pie_forward::{
    FireClass, Gemma4CudaFacts, Gemma4Facts, ForwardPlan, HookStage, LlamaLikeCudaFacts, LlamaLikeFacts, OpKind, Qwen35CudaFacts,
    Qwen35FullAttnFacts, Qwen35GdnFacts, Qwen35HybridFacts, Qwen35MoeMlpFacts,
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

/// The fifth declared configuration (Qwen2.5-1.5B-Instruct), and the
/// first with attention biases: three AddBias ops per layer land on the
/// raw q/k/v after the (lowered-only) lora guard and before rope — the
/// hand-written `maybe_add_bias` position. Fused QKV binding (the dense
/// join fuses WEIGHTS; biases stay separate tensors, added after the
/// split), no qk-norm, tied embeddings.
#[test]
fn qwen2_5_1_5b() {
    check("qwen2_5_1_5b", &LlamaLikeFacts::qwen2_5_1_5b());
}

/// The lowered qwen2_5 pins: the first force-prefill deployment through
/// the walk (GQA 6 is outside the flashinfer decode set and XQA is off
/// live) — the decode class states dequant + the flashinfer prefill
/// region, whose executor case falls back to the PLAN-LESS launcher when
/// prepare (deliberately) built no plan. The cuda facts here match the
/// live L40S derivation (xqa0/dfp0/rt1/fpp1); the digest holds the pair
/// together.
#[test]
fn qwen2_5_1_5b_cuda_decode() {
    check_plan(
        "qwen2_5_1_5b.cuda.decode",
        &llama_like_cuda(
            &LlamaLikeFacts::qwen2_5_1_5b(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: true,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            FireClass::Decode,
        ),
    );
}

#[test]
fn qwen2_5_1_5b_cuda_prefill() {
    check_plan(
        "qwen2_5_1_5b.cuda.prefill",
        &llama_like_cuda(
            &LlamaLikeFacts::qwen2_5_1_5b(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: true,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            FireClass::Prefill,
        ),
    );
}

/// The lowered phi3 pins: the first PADDED head dim through the emitter
/// (96 -> 128) — pad staging around the KV write, the softmax scale
/// override, the post-attention strip, all constants of the text. The
/// cuda facts match the live L40S derivation (xqa0/dfp0/rt1/fpp0/pad1).
#[test]
fn phi3_mini_cuda_decode() {
    check_plan(
        "phi3_mini.cuda.decode",
        &llama_like_cuda(
            &LlamaLikeFacts::phi3_mini(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: true,
                gate_up_fused: true,
            },
            FireClass::Decode,
        ),
    );
}

#[test]
fn phi3_mini_cuda_prefill() {
    check_plan(
        "phi3_mini.cuda.prefill",
        &llama_like_cuda(
            &LlamaLikeFacts::phi3_mini(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: true,
                gate_up_fused: true,
            },
            FireClass::Prefill,
        ),
    );
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

/// Qwen3.6-27B, the dense hybrid — the SAME traced form as 0.8B at a
/// different geometry, which is the claim worth pinning: this checkpoint
/// needs no new vocabulary, only its dims.
///
/// It is the first fixture whose GDN half is GQA (48 value heads over 16
/// key heads), so it is also the first golden where the `_gqa`
/// recurrence and the head-repeat are the stated form rather than a
/// branch nothing takes.
#[test]
fn qwen3_6_27b_cuda_decode() {
    check_plan(
        "qwen3_6_27b_cuda_decode",
        &qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_6_27b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            FireClass::Decode,
        ),
    );
}

#[test]
fn qwen3_6_27b_cuda_prefill() {
    check_plan(
        "qwen3_6_27b_cuda_prefill",
        &qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_6_27b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            FireClass::Prefill,
        ),
    );
}

/// The same fragment's CUDA reading: the fused CUTLASS leg, which is the
/// one the decode path takes and the only one of `run_moe_mlp`'s four
/// that is a single rectangle.
///
/// Read it against the semantic golden above and the difference IS the
/// argument: the selector's two `matmul_per_token`s, the routed swiglu
/// and the `WeightedSum` collapse into ONE launch that produces
/// `[Tokens, hidden]`, and the trailing `ResidualAdd` becomes an
/// explicit `launch_residual_add_bf16` because the fused runner
/// overwrites its output rather than accumulating.
#[test]
fn qwen3_5_moe_mlp_35b_a3b_cuda() {
    check_plan(
        "qwen3_5_moe_mlp_35b_a3b_cuda",
        &qwen3_5_moe_mlp_block_cuda(
            &Qwen35MoeMlpFacts::qwen3_5_35b_a3b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
        ),
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

/// The first LOWERED qwen3_5 goldens (north-star-dsl.md rung 4c-ii): the
/// SAME hybrid text, traced with the SYNTHETIC CUDA backend facts
/// ([`Qwen35CudaFacts::qwen3_5_0_8b_synthetic`] — these pin the golden
/// FORM only; the live derivation + digest validation is 4c-iii) and a
/// fire class in hand. Decode: every GDN layer states the conv update +
/// decode recurrence step as Launches (no Guard — the decode step has no
/// N-threshold), every full-attention layer the HasWriteDesc KV-write
/// guard + the FlashInfer decode dispatch.
#[test]
fn qwen3_5_hybrid_0_8b_cuda_decode() {
    check_plan(
        "qwen3_5_hybrid_0_8b.cuda.decode",
        &qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            FireClass::Decode,
        ),
    );
}

/// The prefill-class lowering of the same text: every GDN layer states
/// the prefill conv walk and the recurrence three-way as the first
/// VALUE-PRODUCING guard chain — TokensLE(64) warp-tiled, TokensLE(4096)
/// cached, else FLA, the guard's output being the core the gated norm
/// consumes — and every full-attention layer the KV-write guard + the
/// dequant-less planned prefill dispatch (qwen3_5's cache is bf16-gated,
/// unlike llama_like's dequant+dispatch pair).
#[test]
fn qwen3_5_hybrid_0_8b_cuda_prefill() {
    check_plan(
        "qwen3_5_hybrid_0_8b.cuda.prefill",
        &qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            FireClass::Prefill,
        ),
    );
}

/// The StateOnly service class (rung 4c-iv): the whole prefill-shaped
/// backbone with the epilogue class-matched away — structurally the
/// prefill trace minus EXACTLY the final-norm/lm_head pair (the pair the
/// hand-written `if (num_logit_rows < 0) return` skips), everything else
/// byte-identical, which the assertions below pin against the prefill
/// trace before the golden pins the form itself.
#[test]
fn qwen3_5_hybrid_0_8b_cuda_state_only() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    let plan = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::StateOnly);
    let prefill = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::Prefill);

    // Prefill minus 2 ops (final rmsnorm + lm_head), prefix-identical…
    assert_eq!(plan.ops.len(), prefill.ops.len() - 2);
    assert_eq!(plan.ops[..], prefill.ops[..prefill.ops.len() - 2]);
    // …and minus their 2 output values, likewise prefix-identical.
    assert_eq!(plan.values.len(), prefill.values.len() - 2);
    assert_eq!(plan.values[..], prefill.values[..prefill.values.len() - 2]);
    assert_eq!(plan.family, "qwen3_5_hybrid.cuda.state_only");

    check_plan("qwen3_5_hybrid_0_8b.cuda.state_only", &plan);
}

/// The CommitAdvance service class (rung 4c-iv): a genuinely different
/// pass — no embed (the root is a bare input placeholder), ONLY the 18
/// linear layers, each exactly [verify-stash load (pseudo-symbol, 3
/// outputs) → prefill conv walk → GdnPrep → FLA recurrence], nothing
/// after the loop: 72 ops. The synthetic cuda facts configure the verify
/// stash, so the in-proj GEMMs are skipped — no Matmul in the pass.
#[test]
fn qwen3_5_hybrid_0_8b_cuda_commit_advance() {
    let facts = Qwen35HybridFacts::qwen3_5_0_8b();
    let cuda = Qwen35CudaFacts::qwen3_5_0_8b_synthetic();
    let plan = qwen3_5_hybrid_cuda(&facts, &cuda, FireClass::CommitAdvance);
    assert_eq!(plan.family, "qwen3_5_hybrid.cuda.commit_advance");

    // 18 linear layers x (4 ops + the two hook sites the hand-written
    // replay passes through, A4); 1 input value + 18 x (3 + 1 + 5)
    // fresh (sites produce nothing).
    assert_eq!(plan.ops.len(), 18 * 6);
    assert_eq!(plan.values.len(), 1 + 18 * 9);
    // The root is a placeholder no op produces.
    assert!(!plan.ops.iter().any(|op| op.outputs.contains(&0)));

    for l in (0..facts.layers).filter(|&l| !facts.is_full_attn(l)) {
        let names: Vec<&str> = plan
            .layer_ops(l)
            .map(|op| match &op.kind {
                OpKind::Launch { kernel, .. } => kernel.as_str(),
                OpKind::GdnPrep { .. } => "GdnPrep",
                OpKind::HookSite { stage, .. } => match stage {
                    HookStage::OnAttnProj => "HookSite(OnAttnProj)",
                    HookStage::OnAttn => "HookSite(OnAttn)",
                },
                other => panic!("foreign op in the commit pass: {other:?}"),
            })
            .collect();
        assert_eq!(
            names,
            [
                "qwen35_verify_stash_load",
                "launch_causal_conv1d_prefill_batched_bf16",
                "GdnPrep",
                "HookSite(OnAttnProj)",
                "launch_chunk_gated_delta_prefill_batched_state_bf16",
                "HookSite(OnAttn)",
            ],
            "layer {l}"
        );
    }
    // Full-attention layers do not exist in this pass, and neither does
    // any GEMM (the stash replays the in-proj outputs).
    for l in (0..facts.layers).filter(|&l| facts.is_full_attn(l)) {
        assert_eq!(plan.layer_ops(l).count(), 0, "layer {l} must be skipped");
    }
    assert!(
        !plan
            .ops
            .iter()
            .any(|op| matches!(op.kind, OpKind::Matmul { .. }))
    );

    check_plan("qwen3_5_hybrid_0_8b.cuda.commit_advance", &plan);
}

/// The first LOWERED goldens (north-star-dsl.md): the SAME llama_like
/// text, traced with the CUDA backend facts and a fire class in hand, so
/// the class arms run and the traced form states kernels. Decode, since
/// A1 (the class-collapse amendment): each layer is a value-producing
/// HasCustomMask guard — the mask arm carries the whole general QKV
/// sequence (split, fused qk-norm+rope, the NESTED HasWriteDesc write
/// guard) ending in the custom-mask dispatch; the else-arm is the fused
/// decode-QKV launch (consuming the once-per-fire rope-table value,
/// hoisted unconditionally — a masked fire launches it unread) plus the
/// plain decode attention. This golden IS the decode launch list — the
/// thing rung 2's dumb interpreter walks (with a skip stack) and rung
/// 3's emitter transliterates to nested `if`s.
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
/// throughout (no fused post — its predicate is decode-only), then the
/// per-layer HasCustomMask guard (A1): custom dispatch in the mask arm
/// (no dequant — the custom dispatch takes the layer view whole),
/// dequant + planned prefill in the else-arm.
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

// (The masked-class goldens are gone with the classes themselves — A1,
// the class-collapse amendment: the custom mask is a HasCustomMask
// guard arm INSIDE the decode/prefill goldens above, which pin the
// arm's op-list delta — the general QKV sequence in the fused
// deployment's mask arm, the custom dispatch, no dequant.)

/// The frozen-verify class (the rung-5 geometry amendment): the prefill
/// body plus ONE stash store per linear layer — the cheap in-proj
/// activations cached for the later commit-advance replay, at the
/// hand-written launch position (after the splits, before the conv).
/// `write_state=false` is a runtime argument of the stated kernels, not
/// a trace difference, so the golden differs from prefill by exactly 18
/// store launches.
#[test]
fn qwen3_5_hybrid_cuda_frozen_verify() {
    check_plan(
        "qwen3_5_hybrid_0_8b.cuda.frozen_verify",
        &qwen3_5_hybrid_cuda(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            &Qwen35CudaFacts::qwen3_5_0_8b_synthetic(),
            FireClass::FrozenVerify,
        ),
    );
}

// (The hooked-class goldens are gone with the classes themselves — A2,
// the class-collapse amendment: attached stage hooks are a
// HasStageHooks guard arm INSIDE the decode/prefill goldens above —
// the general body, the two per-layer HookSites and the
// WantsAttnScore-guarded attention, all in the hooked arm's region.
// Which PROGRAM runs never appears: sites state WHERE and WHAT IS
// OBSERVABLE; programs are sideband data.)

/// The Off-norm lowered goldens (mistral shape): the first LOWERED pin of
/// a deployment whose general arm keeps the SEMANTIC rope (no per-head
/// qk-norm) and whose decode has no fused post — the branch combination
/// the 2026-08-03 hoist regression hid in (general QKV traced into the
/// mask arm after `guarded_value` opened; every unmasked fire skipped
/// QKV). The cuda facts are STRUCTURAL fixtures (xqa/fpp off): what
/// these goldens pin is the region layout — QKV/rope/write BEFORE the
/// attention chain's guard op, arms carrying attention only.
#[test]
fn mistral_7b_v03_cuda_decode() {
    check_plan(
        "mistral_7b_v03.cuda.decode",
        &llama_like_cuda(
            &LlamaLikeFacts::mistral_7b_v03(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            FireClass::Decode,
        ),
    );
}

#[test]
fn mistral_7b_v03_cuda_prefill() {
    check_plan(
        "mistral_7b_v03.cuda.prefill",
        &llama_like_cuda(
            &LlamaLikeFacts::mistral_7b_v03(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            FireClass::Prefill,
        ),
    );
}

/// gemma-4-E4B's decode reading — the third family's first golden.
///
/// Worth reading for three shapes no earlier golden has: the input norm
/// appears ONCE (layer 0's; every other layer's arrives fused into the
/// previous layer's PLE landing), the trailing 18 layers carry no k/v
/// projection or cache write at all, and the two layer kinds differ by
/// head WIDTH rather than by which statements run.
#[test]
fn gemma_4_e4b_cuda_decode() {
    check_plan(
        "gemma_4_e4b_cuda_decode",
        &gemma4_cuda(
            &Gemma4Facts::gemma_4_e4b(),
            &Gemma4CudaFacts::gemma_4_e4b_synthetic(),
            FireClass::Decode,
        ),
    );
}
