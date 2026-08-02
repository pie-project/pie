//! Regenerate the committed static-C++ forms of the lowered llama_like
//! class traces (north-star-dsl.md rung 3), one TU per deployment:
//!
//! ```text
//! cargo run -p pie-forward --bin emit-cuda
//! ```
//!
//! Writes `driver/cuda/src/model/llama_like/generated/{qwen3_0_6b,
//! olmo2_1b}.inc`. The facts are each deployment's LIVE-anchored set;
//! the driver runs a generated pair only when its own derived facts
//! digest matches the constant embedded in that file (drift →
//! interpreter, loudly — the mechanism that corrects any guessed fact
//! on first live run).

use pie_forward::emit_cuda::emit_llama_like_cuda_inc;
use pie_forward::emit_qwen35::emit_qwen35_cuda_inc;
use pie_forward::{LlamaLikeCudaFacts, LlamaLikeFacts, Qwen35CudaFacts, Qwen35HybridFacts};

fn write_inc_at(family: &str, name: &str, contents: &str) {
    let dir = format!(
        "{}/../driver/cuda/src/model/{family}/generated",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::create_dir_all(&dir).unwrap();
    let path = format!("{dir}/{name}.inc");
    std::fs::write(&path, contents).unwrap();
    println!("wrote {path}");
}

fn write_inc(name: &str, contents: &str) {
    write_inc_at("llama_like", name, contents);
}

fn main() {
    // Qwen3-0.6B on L40S (the parity-anchored deployment). The binding
    // unties the lm_head (live digest `te0`, measured 2026-08-02), so
    // emission overrides the fixture's config-level `tied_embeddings`.
    let qwen_facts = LlamaLikeFacts {
        tied_embeddings: false,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    write_inc(
        "qwen3_0_6b",
        &emit_llama_like_cuda_inc(
            &qwen_facts,
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            "qwen3_0_6b",
        ),
    );

    // OLMo-2-1B on L40S: post-norm placement + global qk-norm — the
    // second deployment, and the first post-norm static form. No fused
    // decode post (per-head qk-norm is a predicate term), no XQA; the
    // rope-table workspace exists but nothing consumes it. Digest-
    // verified live like every fact set.
    write_inc(
        "olmo2_1b",
        &emit_llama_like_cuda_inc(
            &LlamaLikeFacts::olmo2_1b(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                // TRUE as a deployment FACT (env on, native bf16,
                // head_dim == kernel — the build's derivation; the live
                // digest said dfp1 and corrected the dfp0 guess on first
                // boot, the mechanism's third catch). The TEXT still
                // never fires the fused arm here: its predicate also
                // wants per-head qk-norm and the fused binding, both
                // false for olmo2 — a fact can be true and unused.
                decode_fused_post: true,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            "olmo2_1b",
        ),
    );

    // Qwen2.5-1.5B on L40S: the first bias deployment (AddBias ops in
    // both class fns) and the first force-prefill one (GQA 6 is outside
    // the flashinfer decode set, XQA off live) — the decode class emits
    // the PLAN-LESS prefill launcher directly, the static mirror of the
    // hand-written final else. Facts guessed from the 2026-08-03
    // interpreter-leg run; the live digest judges them on first boot.
    write_inc(
        "qwen2_5_1_5b",
        &emit_llama_like_cuda_inc(
            &LlamaLikeFacts::qwen2_5_1_5b(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                // The derivation's !use_qkv_bias term forces this off —
                // the fused epilogue has no bias step.
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: true,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            "qwen2_5_1_5b",
        ),
    );

    // Mistral-7B-Instruct-v0.3 on L40S: the fused binding with no
    // qk-norm (the combination the lowered goldens pinned after the
    // hoist regression) at 7B scale. GQA 4 is inside the flashinfer
    // decode set (no force-prefill); XQA off live like every deployment
    // on this card; dfp TRUE as a fact and unused by the text (the
    // fused arm also wants per-head qk-norm — the olmo2 precedent).
    // Facts guessed from the config + binding rules; the live digest
    // judges them on first boot.
    write_inc(
        "mistral_7b_v03",
        &emit_llama_like_cuda_inc(
            &LlamaLikeFacts::mistral_7b_v03(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: true,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            "mistral_7b_v03",
        ),
    );

    // Phi-3-mini-4k on L40S: the padded head dim (96 -> 128) — the
    // generated form stages the zero-padded q/k/v copies around the KV
    // write, overrides the softmax scale to 1/sqrt(96), and strips the
    // attention output, all spelled statically (the interpreter's
    // head_dim_padded locals became this deployment's constants). MHA
    // ratio 1 is in the flashinfer decode set (no force-prefill); dfp
    // off (the derivation's head_dim == head_dim_kernel term). Facts
    // guessed; the live digest judges them on first boot.
    write_inc(
        "phi3_mini",
        &emit_llama_like_cuda_inc(
            &LlamaLikeFacts::phi3_mini(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: true,
                gate_up_fused: true,
            },
            "phi3_mini",
        ),
    );

    // Qwen3.5-0.8B hybrid on L40S (decode + prefill; the MTP service
    // classes stay on the interpreter walk). The cuda facts fixture is
    // the SYNTHETIC set — the live digest judges and corrects it on
    // first boot, the mechanism's standing contract.
    write_inc_at(
        "qwen3_5",
        "qwen3_5_0_8b",
        &emit_qwen35_cuda_inc(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            &Qwen35CudaFacts {
                // LIVE-anchored (digest-corrected on first boot — the
                // mechanism's fourth catch: the synthetic fixture said
                // wt1/cm4096, the live env defaults say wt0/cm0):
                // warp-tiled needs its state-persist env gate, the
                // cached family its max-tokens env, both off by default.
                state_bf16: true,
                warp_tiled: false,
                warp_tiled_max: 64,
                cached_max: 0,
                verify_stash: true,
                // 0.8B is DENSE — it reaches no MoE op, so these are the
                // "no fused leg" values and the emitted body is unchanged
                // by them. A MoE emission target would set them live.
                moe_cutlass_max_rows: 0,
                moe_residual_fold: false,
                moe_shared_gate_dot: false,
                moe_streamed_experts: false,
                moe_force_general: false,
                // 0.8B binds the packed bank; the emitted body states the
                // chunked activation rather than reading a workspace.
                gate_up_fused: true,
            },
            "qwen3_5_0_8b",
        ),
    );
}
