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

    // ── MoE (qwen3-moe, mixtral, gemma4, qwen3.6) ──
    std::int32_t n_experts         = 0;    // 0 = dense MLP
    std::int32_t n_experts_per_tok = 0;
    bool         moe_norm_topk     = true;
    float        moe_routed_scale  = 1.0f; // router gain (gemma4/qwen3.6)
    std::int32_t moe_n_group       = 0;    // grouped routing (qwen3.6); 0 = off
    std::int32_t moe_topk_group    = 0;
    std::int32_t n_shared_experts  = 0;    // sigmoid-gated shared expert (qwen3.6)
    std::int32_t first_k_dense     = 0;    // layers < this stay dense

    // ── gemma4 extras ──
    bool         gemma4_parallel_moe = false;  // dense MLP + parallel MoE block
    std::int32_t per_layer_emb_dim   = 0;      // PLE width; 0 = no PLE
    std::int32_t num_kv_shared_layers = 0;     // cross-layer KV sharing

    // ── qwen3.6 (Qwen3.5 hybrid) extras ──
    bool  attn_output_gate     = false;  // sigmoid-gated 2x-wide q_proj
    float partial_rotary_factor = 1.0f;  // <1 = partial RoPE
    // Linear-attn (Gated DeltaNet) dims; >0 enables linear layers.
    std::int32_t linear_num_value_heads = 0;
    std::int32_t linear_num_key_heads   = 0;
    std::int32_t linear_key_head_dim    = 0;
    std::int32_t linear_value_head_dim  = 0;
    std::int32_t linear_conv_kernel_dim = 0;

    // Helpers.
    bool is_sliding_layer(std::int32_t il) const {
        return !layer_pattern.empty()
            && static_cast<std::size_t>(il) < layer_pattern.size()
            && layer_pattern[il] == 's';
    }
    // 'l' marks a Gated-DeltaNet linear-attention layer (qwen3.6).
    bool is_linear_attn_layer(std::int32_t il) const {
        return !layer_pattern.empty()
            && static_cast<std::size_t>(il) < layer_pattern.size()
            && layer_pattern[il] == 'l';
    }

    // ── gemma4 cross-layer KV-share schedule ──────────────────────────────
    // The last `num_kv_shared_layers` decoder layers don't compute their own
    // K/V; they re-attend through the most-recent earlier layer of the SAME
    // attention type (sliding vs full). These pure helpers drive both the
    // graph (which layers skip k/v_proj + append, and which source layer to
    // read) and delta's per-layer PagedKvCache spec (`n_pages = 0` on shared
    // layers). `n_layers` is `cfg.num_hidden_layers`.
    bool gemma4_is_kv_shared(std::int32_t il, std::int32_t n_layers) const {
        return num_kv_shared_layers > 0
            && il >= n_layers - num_kv_shared_layers;
    }
    // KV source layer for `il`: itself when not shared; otherwise the most
    // recent earlier non-shared layer of the same attention type. Returns -1
    // when none exists (caller treats as a config error).
    std::int32_t gemma4_kv_source(std::int32_t il, std::int32_t n_layers) const {
        if (!gemma4_is_kv_shared(il, n_layers)) return il;
        const std::int32_t first_shared = n_layers - num_kv_shared_layers;
        const bool want_sliding = is_sliding_layer(il);
        for (std::int32_t j = first_shared - 1; j >= 0; --j) {
            if (is_sliding_layer(j) == want_sliding) return j;
        }
        return -1;
    }
};

ArchSpec arch_spec_for(PieArch arch, const ModelConfig& c);

}  // namespace pie_metal_driver::model
