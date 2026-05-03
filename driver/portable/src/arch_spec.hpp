#pragma once

// ArchSpec: per-arch feature switches read by the graph builder.
//
// Each entry below corresponds to a structural quirk of one or more
// supported HF architectures (qwen2 biases, qwen3 QK-norm, gemma2 softcaps,
// gemma4 PLE, etc.). The dense graph builder branches on these flags so
// every arch routes through the same skeleton; gemma4 has its own builder
// because per-layer head_dim + KV-share don't fit the shared layout.
//
// Built once per `compute_()` call from `arch_spec_for(arch, hparams)`.

#include <cstdint>
#include <string>

#include "hf_config.hpp"

namespace pie_portable_driver {

struct ArchSpec {
    // Phi-3-small blocksparse attention. block_size > 0 enables it; the
    // graph builder dispatches per-layer mask choice based on
    // phi3small_dense_attention_every_n_layers.
    std::int32_t phi3small_block_size                  = 0;
    std::int32_t phi3small_num_local_blocks            = 0;
    std::int32_t phi3small_vert_stride                 = 0;
    std::int32_t phi3small_dense_attention_every_n_layers = 0;
    bool   has_qkv_bias       = false;  // qwen2, phi3 (sometimes)
    bool   has_qk_norm        = false;  // qwen3, gemma3, olmo3
    // Qwen3 / Gemma3 store Q/K-norm weights as [head_dim] — applied per-
    // head after reshape. Olmo3 stores them as [hidden_size] / [kv_dim]
    // and normalizes the flat Q/K vectors before reshape (so the RMS
    // statistic is global, not per-head).
    bool   qk_norm_full       = false;
    bool   has_pre_ffn_norm   = false;  // gemma2/3/4 (extra norm before FFN)
    bool   has_post_attn_norm = false;  // gemma2/3/4 (extra norm after attn)
    bool   has_post_ffn_norm  = false;  // gemma2/3/4 (extra norm after FFN)
    bool   scale_embed_by_sqrt_d = false;  // gemma family
    float  attn_softcap       = 0.0f;   // gemma2 (50.0 typical), 0 = none
    float  final_softcap      = 0.0f;   // gemma2 (30.0 typical), 0 = none
    // Per-layer attention pattern. Empty = all-global (causal). Non-empty
    // = vector of size n_layers; entries 'g' (global) or 's' (sliding).
    // For mistral all layers are sliding. For gemma3, every Nth is 'g'.
    std::string layer_pattern;
    std::int32_t sliding_window = 0;     // 0 = no SWA
    // Custom Q scaling (gemma2/3 use 1/sqrt(query_pre_attn_scalar) instead
    // of 1/sqrt(head_dim)). 0 = use default head_dim.
    float  query_pre_attn_scalar = 0.0f;
    // MLP activation. SiLU (SwiGLU) for L4MA family; GeLU (GeGLU) for Gemma.
    bool   ffn_use_gelu = false;
    // Gemma family: RMSNorm uses `(1 + weight)` instead of `weight`. The
    // weights are stored centered at 0 around 1, so we add 1 before
    // multiplying the normalized activation.
    bool   norm_weight_plus_one = false;

    // ── MoE ──
    // n_experts == 0 means dense MLP (the standard SwiGLU/GeGLU path).
    std::int32_t n_experts        = 0;
    std::int32_t n_experts_per_tok = 0;
    bool         moe_norm_topk    = true;   // renormalize selected weights

    // ── Gemma 4 ──
    // Gemma 4 stores RMSNorm weights centered at 1 (init=ones), unlike
    // Gemma 2/3 which center at 0 and apply (1+w). norm_weight_plus_one
    // would be wrong here; this flag suppresses it for gemma4 paths.
    bool gemma4_norm_weight_direct = false;
    // V-norm: pure RMSNorm (no learnable weight) on V before KV write.
    bool gemma4_v_norm = false;
    // Per-layer scalar applied to the layer's output.
    bool gemma4_layer_scalar = false;
    // Per-Layer Embeddings injected after MLP every layer.
    bool gemma4_ple_enabled = false;
    // sm_scale = 1.0 instead of 1/sqrt(head_dim). Q/K-norm absorbs the scale.
    bool gemma4_unit_sm_scale = false;
    // Dual head_dim per layer + dual rope_theta per layer-type.
    std::int32_t gemma4_head_dim_global = 0;
    float gemma4_rope_theta_full = 0.0f;
    float gemma4_rope_theta_sliding = 0.0f;
    float gemma4_partial_full = 1.0f;
    float gemma4_partial_sliding = 1.0f;
    std::int32_t gemma4_first_shared = 0;  // index of first KV-shared layer
    std::int32_t gemma4_ple_dim = 0;

    // ── YaRN RoPE (olmo3, Ministral 3, gpt-oss) ──
    // ggml_rope_ext switches into YaRN mode when ext_factor > 0; we then
    // also pass freq_scale = 1/factor, attn_factor = mscale, n_ctx_orig =
    // original_max_position_embeddings, beta_fast/beta_slow (smooth-ramp
    // bounds). yarn_n_ctx_orig == 0 means YaRN is off (fall back to plain
    // θ-only RoPE; freq_factors path remains usable for llama3 NTK).
    std::int32_t yarn_n_ctx_orig = 0;
    float        yarn_freq_scale = 1.0f;     // 1.0 / rope_scaling_factor
    float        yarn_attn_factor = 1.0f;    // HF attention_factor (mscale)
    float        yarn_beta_fast  = 32.0f;
    float        yarn_beta_slow  = 1.0f;
};

ArchSpec arch_spec_for(PieArch a, const Hparams& h);

}  // namespace pie_portable_driver
