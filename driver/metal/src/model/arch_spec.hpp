#pragma once

// ArchSpec — per-arch structural feature switches read by the graph builders.
// Mirrors driver/portable's arch_spec.hpp, pared to the flags the metal
// driver's first wave of architectures (Llama-like + Qwen3, Gemma2/3) needs.
//
// Built once per forward from `arch_spec_for(arch, config)`. The shared
// Llama-like builder and the Gemma builder branch on these flags so every
// arch routes through one skeleton; the flags are the *only* place arch
// quirks live (no compile-time arch coupling in the graph bodies).

#include <cstdint>
#include <string>

#include "config.hpp"

namespace pie_metal_driver::model {

struct ArchSpec {
    // ── Attention block quirks ──
    bool  has_qkv_bias  = false;  // qwen2: additive Q/K/V bias post-projection
    bool  has_qk_norm   = false;  // qwen3 / gemma3: per-head RMSNorm on Q,K
    // gemma2/3: extra norms around attention / FFN (norm sandwich).
    bool  has_post_attn_norm = false;
    bool  has_pre_ffn_norm   = false;
    bool  has_post_ffn_norm  = false;

    // ── Embedding / activation ──
    bool  scale_embed_by_sqrt_d = false;  // gemma: embed *= sqrt(hidden_size)
    bool  ffn_use_gelu          = false;  // gemma GeGLU; else SwiGLU (SiLU)
    // gemma RMSNorm stores weights centered at 0 and applies (1 + w).
    bool  norm_weight_plus_one  = false;

    // ── Softcaps (gemma2) ──
    float attn_softcap  = 0.0f;   // 0 = none
    float final_softcap = 0.0f;   // 0 = none

    // ── Attention scale ──
    // gemma2/3 use 1/sqrt(query_pre_attn_scalar) instead of 1/sqrt(head_dim).
    // 0 = use head_dim default.
    float query_pre_attn_scalar = 0.0f;

    // ── Sliding-window attention ──
    // Per-layer pattern: 'g' (full) / 's' (sliding). Empty = all-global.
    std::string  layer_pattern;
    std::int32_t sliding_window = 0;   // 0 = no SWA

    // ── YaRN RoPE (Ministral 3, llama-3.1 NTK lives in freq_factors) ──
    std::int32_t yarn_n_ctx_orig  = 0;     // 0 = YaRN off
    float        yarn_freq_scale  = 1.0f;
    float        yarn_attn_factor = 1.0f;
    float        yarn_beta_fast   = 32.0f;
    float        yarn_beta_slow   = 1.0f;

    // ── MoE (qwen3-moe, mixtral) ──
    std::int32_t n_experts         = 0;    // 0 = dense MLP
    std::int32_t n_experts_per_tok = 0;
    bool         moe_norm_topk     = true;

    // Helpers.
    bool is_sliding_layer(std::int32_t il) const {
        return !layer_pattern.empty()
            && static_cast<std::size_t>(il) < layer_pattern.size()
            && layer_pattern[il] == 's';
    }
};

ArchSpec arch_spec_for(PieArch arch, const ModelConfig& c);

}  // namespace pie_metal_driver::model
