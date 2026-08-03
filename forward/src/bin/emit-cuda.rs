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
            },
            "qwen2_5_1_5b",
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
            },
            "qwen3_5_0_8b",
        ),
    );
}
