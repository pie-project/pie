//! The committed generated `.inc` matches what the emitter produces —
//! the cbindgen-header rule applied to rung 3's artifact: a drift between
//! the declaration (or the emitter) and the committed static C++ cannot
//! happen quietly. Regenerate with `cargo run -p pie-forward --bin
//! emit-cuda` and review the diff; then re-run the three-way parity gate.

use pie_forward::emit_cuda::emit_llama_like_cuda_inc;
use pie_forward::{LlamaLikeCudaFacts, LlamaLikeFacts};

#[test]
fn committed_inc_is_regeneration_clean() {
    // The LIVE deployment's facts — the same override emit-cuda applies
    // (see its comment for the te0 provenance).
    let facts = LlamaLikeFacts {
        tied_embeddings: false,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    let fresh = emit_llama_like_cuda_inc(&facts, &LlamaLikeCudaFacts::qwen3_0_6b_l40s());
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../driver/cuda/src/model/llama_like/generated/qwen3_0_6b.inc"
    );
    let committed = std::fs::read_to_string(path).expect("committed generated .inc");
    assert_eq!(
        committed, fresh,
        "generated .inc drifted from the emitter; regenerate with \
         `cargo run -p pie-forward --bin emit-cuda`, review the diff, and \
         re-run the three-way parity gate"
    );
}
