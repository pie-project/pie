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
use pie_forward::{LlamaLikeCudaFacts, LlamaLikeFacts};

fn write_inc(name: &str, contents: &str) {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../driver/cuda/src/model/llama_like/generated"
    );
    std::fs::create_dir_all(dir).unwrap();
    let path = format!("{dir}/{name}.inc");
    std::fs::write(&path, contents).unwrap();
    println!("wrote {path}");
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
}
