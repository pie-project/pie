//! The committed static-C++ emissions, as ONE list.
//!
//! `bin/emit-cuda.rs` used to hold each deployment's fact set inline and
//! `tests/generated_cuda.rs` held a hand-mirrored copy ("must mirror
//! `bin/emit-cuda.rs` exactly"). Two statements of one list is the
//! `workspace_bytes` shape: edit the bin, forget the test, and the test goes
//! on proving the committed `.inc`s match an emission nothing writes anymore.
//! Now the bin WRITES this list and the test CHECKS it, and there is no
//! second copy to forget.
//!
//! Each entry's fact set is LIVE-anchored — the provenance comments ride
//! along from the bin, because which digest corrected which guess is the
//! history that explains the values.

use crate::families::llama_like::forward::emit::emit_llama_like_cuda_inc;
use crate::families::llama_like::forward::facts::{LlamaLikeCudaFacts, LlamaLikeFacts};
use crate::qwen_3_5::forward::emit::emit_qwen35_cuda_inc;
use crate::qwen_3_5::forward::facts::{Qwen35CudaFacts, Qwen35HybridFacts};

/// One committed emission: which family directory it lands in, the file
/// stem, and the full generated text.
pub struct CudaEmission {
    /// The `model/<family>/generated/` directory the file lives under.
    pub family: &'static str,
    /// The file stem (`<name>.inc`).
    pub name: &'static str,
    /// The emitted translation-unit body.
    pub text: String,
}

impl CudaEmission {
    /// The committed path, relative to the workspace's `crates/` directory.
    pub fn rel_path(&self) -> String {
        format!(
            "driver-cuda/csrc/src/model/{}/generated/{}.inc",
            self.family, self.name
        )
    }
}

/// Every `.inc` the repository commits, in emission order.
pub fn committed_cuda_emissions() -> Vec<CudaEmission> {
    let mut out = Vec::new();
    let llama = |name: &'static str, text: String| CudaEmission { family: "llama_like", name, text };

    // Qwen3-0.6B on L40S (the parity-anchored deployment). The binding
    // unties the lm_head (live digest `te0`, measured 2026-08-02), so
    // emission overrides the fixture's config-level `tied_embeddings`.
    let qwen_facts = LlamaLikeFacts {
        tied_embeddings: false,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    out.push(llama(
        "qwen3_0_6b",
        emit_llama_like_cuda_inc(
            &qwen_facts,
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            "qwen3_0_6b",
        ),
    ));

    // OLMo-2-1B on L40S: post-norm placement + global qk-norm — the
    // second deployment, and the first post-norm static form. No fused
    // decode post (per-head qk-norm is a predicate term), no XQA; the
    // rope-table workspace exists but nothing consumes it. Digest-
    // verified live like every fact set.
    out.push(llama(
        "olmo2_1b",
        emit_llama_like_cuda_inc(
            &LlamaLikeFacts::olmo2_1b(),
            &LlamaLikeCudaFacts {
                head_dim_kernel: 0,
                proj_repr: model_compiler::dsl::WeightRepr::Bf16,
                tp_size: 1,
                window_left: Vec::new(),
                all_reduce_p2p_max_rows: 0,
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
    ));

    // Qwen2.5-1.5B on L40S: the first bias deployment (AddBias ops in
    // both class fns) and the first force-prefill one (GQA 6 is outside
    // the flashinfer decode set, XQA off live) — the decode class emits
    // the PLAN-LESS prefill launcher directly, the static mirror of the
    // hand-written final else. Facts guessed from the 2026-08-03
    // interpreter-leg run; the live digest judges them on first boot.
    out.push(llama(
        "qwen2_5_1_5b",
        emit_llama_like_cuda_inc(
            &LlamaLikeFacts::qwen2_5_1_5b(),
            &LlamaLikeCudaFacts {
                head_dim_kernel: 0,
                proj_repr: model_compiler::dsl::WeightRepr::Bf16,
                tp_size: 1,
                window_left: Vec::new(),
                all_reduce_p2p_max_rows: 0,
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
    ));

    // Mistral-7B-Instruct-v0.3 on L40S: the fused binding with no
    // qk-norm (the combination the lowered goldens pinned after the
    // hoist regression) at 7B scale. GQA 4 is inside the flashinfer
    // decode set (no force-prefill); XQA off live like every deployment
    // on this card; dfp TRUE as a fact and unused by the text (the
    // fused arm also wants per-head qk-norm — the olmo2 precedent).
    // Facts guessed from the config + binding rules; the live digest
    // judges them on first boot.
    out.push(llama(
        "mistral_7b_v03",
        emit_llama_like_cuda_inc(
            &LlamaLikeFacts::mistral_7b_v03(),
            &LlamaLikeCudaFacts {
                head_dim_kernel: 0,
                proj_repr: model_compiler::dsl::WeightRepr::Bf16,
                tp_size: 1,
                window_left: Vec::new(),
                all_reduce_p2p_max_rows: 0,
                xqa_decode: false,
                decode_fused_post: true,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            "mistral_7b_v03",
        ),
    ));

    // Phi-3-mini-4k on L40S: the padded head dim (96 -> 128) — the
    // generated form stages the zero-padded q/k/v copies around the KV
    // write, overrides the softmax scale to 1/sqrt(96), and strips the
    // attention output, all spelled statically (the interpreter's
    // head_dim_padded locals became this deployment's constants). MHA
    // ratio 1 is in the flashinfer decode set (no force-prefill); dfp
    // off (the derivation's head_dim == head_dim_kernel term). Facts
    // guessed; the live digest judges them on first boot.
    out.push(llama(
        "phi3_mini",
        emit_llama_like_cuda_inc(
            &LlamaLikeFacts::phi3_mini(),
            &LlamaLikeCudaFacts {
                head_dim_kernel: 128,
                proj_repr: model_compiler::dsl::WeightRepr::Bf16,
                tp_size: 1,
                window_left: Vec::new(),
                all_reduce_p2p_max_rows: 0,
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: false,
                head_dim_padded: true,
                gate_up_fused: true,
            },
            "phi3_mini",
        ),
    ));

    // Qwen3.5-0.8B hybrid on L40S (decode + prefill; the MTP service
    // classes stay on the interpreter walk). The cuda facts fixture is
    // the SYNTHETIC set — the live digest judges and corrects it on
    // first boot, the mechanism's standing contract.
    out.push(CudaEmission {
        family: "qwen3_5",
        name: "qwen3_5_0_8b",
        text: emit_qwen35_cuda_inc(
            &Qwen35HybridFacts::qwen3_5_0_8b(),
            &Qwen35CudaFacts {
                // Attends the whole context.
                window_left: Vec::new(),
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
                // The live default: PIE_QWEN35_PREFILL_DECODE is on
                // unless set to 0, and 0.8B's cache is native bf16.
                prefill_decode: true,
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
                // 0.8B is a BF16 checkpoint.
                proj_repr: model_compiler::dsl::WeightRepr::Bf16,
            },
            "qwen3_5_0_8b",
        ),
    });

    out
}
