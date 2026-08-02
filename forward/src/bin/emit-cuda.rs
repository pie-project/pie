//! Regenerate the committed static-C++ form of the lowered llama_like
//! class traces (north-star-dsl.md rung 3):
//!
//! ```text
//! cargo run -p pie-forward --bin emit-cuda
//! ```
//!
//! Writes `driver/cuda/src/model/llama_like/generated/qwen3_0_6b.inc`.
//! The facts here are the parity-anchored deployment (Qwen3-0.6B on
//! L40S, default env — `LlamaLikeCudaFacts::qwen3_0_6b_l40s`'s
//! provenance); the driver runs the generated code only when its own
//! derived facts digest matches the constant embedded in the file.

use pie_forward::emit_cuda::emit_llama_like_cuda_inc;
use pie_forward::{LlamaLikeCudaFacts, LlamaLikeFacts};

fn main() {
    // The LIVE deployment's facts, not the config-level fixture: the
    // binding unties the lm_head (`w.lm_head != w.embed` — live digest
    // `te0`, measured 2026-08-02), so emission overrides the fixture's
    // config-level `tied_embeddings`. The digest match at model load is
    // what makes this override safe: drift → interpreter, loudly.
    let facts = LlamaLikeFacts {
        tied_embeddings: false,
        ..LlamaLikeFacts::qwen3_0_6b()
    };
    let out = emit_llama_like_cuda_inc(&facts, &LlamaLikeCudaFacts::qwen3_0_6b_l40s());
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../driver/cuda/src/model/llama_like/generated/qwen3_0_6b.inc"
    );
    std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
    std::fs::write(path, out).unwrap();
    println!("wrote {path}");
}
