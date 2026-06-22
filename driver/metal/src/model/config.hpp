#pragma once

// ModelConfig — the subset of an HF `config.json` the metal driver's graph
// builders read. Mirrors driver/portable's `Hparams`, pared to the fields the
// Llama-like + Qwen3 + Gemma2/3 forward passes actually consume. Optional
// fields use std::optional so a builder can branch on presence (e.g. SWA,
// softcaps, query_pre_attn_scalar).
//
// Ownership note: the loader (delta) is responsible for *populating* this from
// config.json; the graph builders (this dir) only read it. Kept in model/ so
// the per-arch builders and arch_spec are self-contained; if delta prefers a
// canonical config struct in the loader, this becomes a thin re-export — the
// field set is the contract. Reconcile on integration (see #mac).

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "arch.hpp"

namespace pie_metal_driver::model {

struct ModelConfig {
    PieArch     arch = PieArch::Unknown;
    std::string hf_model_type;   // raw "qwen3"
    std::string torch_dtype;     // "bfloat16" / "float16" / "float32"

    // ── Core transformer dims ──
    std::int32_t num_hidden_layers     = 0;
    std::int32_t num_attention_heads   = 0;
    std::int32_t num_key_value_heads   = 0;
    std::int32_t hidden_size           = 0;
    std::int32_t intermediate_size     = 0;
    std::int32_t head_dim              = 0;   // computed (hidden/heads) if absent
    std::int32_t vocab_size            = 0;
    std::int32_t max_position_embeddings = 0;

    // ── Norm / RoPE ──
    float rms_norm_eps        = 1e-6f;
    float rope_theta          = 1e6f;
    // Gemma3: separate base frequency for sliding-window layers (0 = none).
    float rope_local_base_freq = 0.0f;

    // Tied embeddings (qwen3 default true; llama3 typically false).
    bool tie_word_embeddings = true;

    // ── Sliding-window attention ──
    std::optional<std::int32_t> sliding_window;  // present when SWA configured

    // ── RoPE scaling (LLaMA-3.1 NTK-by-parts, YaRN, linear) ──
    bool         has_rope_scaling = false;
    std::string  rope_scaling_type;            // "llama3" / "yarn" / "linear" / ""
    float        rope_scaling_factor = 1.0f;
    float        rope_scaling_low_freq_factor  = 1.0f;
    float        rope_scaling_high_freq_factor = 4.0f;
    std::int32_t rope_scaling_original_max_position = 0;
    // YaRN extras (Ministral 3, olmo-style). Defaults match HF.
    float        rope_yarn_attention_factor = 0.0f;  // 0 = 0.1*ln(factor)+1
    float        rope_yarn_beta_fast = 32.0f;
    float        rope_yarn_beta_slow = 1.0f;

    // Per-layer attention type: 's' (sliding) / 'g' (full). Empty = uniform.
    std::vector<char> layer_types;

    // ── Gemma family ──
    std::optional<float> attn_logit_softcapping;    // gemma2 (~50.0)
    std::optional<float> final_logit_softcapping;   // gemma2 (~30.0)
    // gemma2/3: 1/sqrt(query_pre_attn_scalar) Q scale instead of 1/sqrt(head_dim).
    std::optional<float> query_pre_attn_scalar;

    // ── Mixture-of-Experts (Qwen3-MoE, Mixtral) ──
    std::int32_t num_experts            = 0;  // 0 = dense
    std::int32_t num_experts_per_tok    = 0;  // top-k routing
    std::int32_t moe_intermediate_size  = 0;  // per-expert FFN hidden
    bool         norm_topk_prob         = true;

    // Convenience: KV embedding width (kv_heads * head_dim).
    std::int32_t n_embd_gqa() const { return num_key_value_heads * head_dim; }
};

}  // namespace pie_metal_driver::model
