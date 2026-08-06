//! The committed generated `.inc`s match what the emitter produces —
//! the cbindgen-header rule applied to rung 3's artifacts: a drift between
//! the declaration (or the emitter) and the committed static C++ cannot
//! happen quietly. Regenerate with `cargo run -p pie-forward --bin
//! emit-cuda` and review the diff; then re-run the three-way parity gate.

use pie_forward::emit_cuda::emit_llama_like_cuda_inc;
use pie_forward::{LlamaLikeCudaFacts, LlamaLikeFacts};

fn check(name: &str, fresh: &str) {
    let path = format!(
        "{}/../driver/cuda/src/model/llama_like/generated/{name}.inc",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read_to_string(&path).expect("committed generated .inc");
    assert_eq!(
        committed, fresh,
        "generated {name}.inc drifted from the emitter; regenerate with \
         `cargo run -p pie-forward --bin emit-cuda`, review the diff, and \
         re-run the three-way parity gate"
    );
}

#[test]
fn committed_incs_are_regeneration_clean() {
    // The LIVE deployments' facts — the same sets emit-cuda writes (see
    // its comments for provenance).
    let qwen_facts = LlamaLikeFacts {
        tied_embeddings: false,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    check(
        "qwen3_0_6b",
        &emit_llama_like_cuda_inc(
            &qwen_facts,
            &LlamaLikeCudaFacts::qwen3_0_6b_l40s(),
            "qwen3_0_6b",
        ),
    );
    check(
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
    check(
        "qwen2_5_1_5b",
        &emit_llama_like_cuda_inc(
            &LlamaLikeFacts::qwen2_5_1_5b(),
            &LlamaLikeCudaFacts {
                xqa_decode: false,
                decode_fused_post: false,
                rope_table: true,
                force_prefill_path: true,
                head_dim_padded: false,
                gate_up_fused: true,
            },
            "qwen2_5_1_5b",
        ),
    );
    check(
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
    check(
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
    check_q35(
        "qwen3_5_0_8b",
        &pie_forward::emit_qwen35::emit_qwen35_cuda_inc(
            &pie_forward::Qwen35HybridFacts::qwen3_5_0_8b(),
            &pie_forward::Qwen35CudaFacts {
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

fn check_q35(name: &str, fresh: &str) {
    let path = format!(
        "{}/../driver/cuda/src/model/qwen3_5/generated/{name}.inc",
        env!("CARGO_MANIFEST_DIR")
    );
    let committed = std::fs::read_to_string(&path).expect("committed generated .inc");
    assert_eq!(
        committed, fresh,
        "generated {name}.inc drifted from the emitter; regenerate with \
         `cargo run -p pie-forward --bin emit-cuda`"
    );
}
