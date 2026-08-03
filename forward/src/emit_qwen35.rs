//! Static C++ emission for the qwen3_5 HYBRID — rung 3's second family.
//!
//! This module starts with the DIGEST (the llama mechanism's port: one
//! format, two printers — this and `declared_facts.cpp` — held together
//! by the live static-form gate, which corrects any guessed emission
//! fact on first boot). The class-function emitter follows incrementally
//! (decode first), transliterating `qwen3_5/declared_forward.cpp`'s walk
//! arms exactly as `emit_cuda` transliterated llama's.

use crate::facts::{Qwen35CudaFacts, Qwen35HybridFacts, Qwen35MlpKind};
use crate::trace::NormVariant;

/// The digest naming what a generated qwen3_5 TU is emitted FROM.
/// Field-for-field the C++ printer in `declared_facts.cpp`.
pub fn facts_digest(facts: &Qwen35HybridFacts, cuda: &Qwen35CudaFacts) -> String {
    let nv = |v: NormVariant| match v {
        NormVariant::Plain => 0,
        NormVariant::Gemma => 1,
    };
    let (moe, dense_intermediate) = match &facts.mlp {
        Qwen35MlpKind::Dense { intermediate } => (0, *intermediate),
        Qwen35MlpKind::Moe(_) => (1, 0),
    };
    format!(
        "qwen3_5/l{}/int{}/v{}/te{}/nv{}/ah{}/aqh{}/akvh{}/ahd{}/arot{}/afq{}\
         /gkh{}/gvh{}/gkd{}/gvd{}/gck{}/gfi{}/moe{}/di{}/sb{}/wt{}/wtm{}/cm{}/vs{}",
        facts.layers,
        facts.full_attn_interval,
        facts.vocab,
        u8::from(facts.tied_embeddings),
        nv(facts.norm_variant),
        facts.attn.hidden,
        facts.attn.q_heads,
        facts.attn.kv_heads,
        facts.attn.head_dim,
        facts.attn.rotary_dim,
        u8::from(facts.attn.fused_qkv),
        facts.gdn.key_heads,
        facts.gdn.value_heads,
        facts.gdn.key_head_dim,
        facts.gdn.value_head_dim,
        facts.gdn.conv_kernel,
        u8::from(facts.gdn.fused_in_proj),
        moe,
        dense_intermediate,
        u8::from(cuda.state_bf16),
        u8::from(cuda.warp_tiled),
        cuda.warp_tiled_max,
        cuda.cached_max,
        u8::from(cuda.verify_stash),
    )
}
