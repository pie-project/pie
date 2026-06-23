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
    // gemma4: full-attention layers use a wider head_dim than sliding layers.
    std::int32_t global_head_dim       = 0;   // 0 = same as head_dim
    std::int32_t num_global_kv_heads   = 0;   // 0 = same as num_key_value_heads
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

    // ── Mixture-of-Experts (Qwen3-MoE, Mixtral, gemma4, qwen3.6) ──
    std::int32_t num_experts            = 0;  // 0 = dense
    std::int32_t num_experts_per_tok    = 0;  // top-k routing
    std::int32_t moe_intermediate_size  = 0;  // per-expert FFN hidden
    bool         norm_topk_prob         = true;
    float        routed_scaling_factor  = 1.0f;  // gemma4/qwen3.6 router gain
    // Shared (always-on) expert — qwen3.6 MoE, sigmoid-gated.
    std::int32_t n_shared_experts            = 0;  // 0 = none
    std::int32_t shared_expert_intermediate_size = 0;
    // Grouped routing (qwen3.6): route within n_group, keep topk_group groups.
    std::int32_t n_group     = 0;  // 0 = ungrouped
    std::int32_t topk_group  = 0;
    // Layers [0, first_k_dense_replace) stay dense before MoE kicks in.
    std::int32_t first_k_dense_replace = 0;

    // ── gemma4 ──
    // Flips dense-MLP layers to run a dense MLP + parallel MoE block.
    bool         gemma4_enable_moe = false;
    // Per-layer-embedding (PLE) feature width; 0 = no PLE.
    std::int32_t per_layer_emb_dim = 0;
    // Cross-layer KV sharing: last N layers reuse K/V from a prior layer.
    std::int32_t num_kv_shared_layers = 0;

    // ── qwen3.6 (Qwen3.5 hybrid linear-attention family) ──
    // Gated DeltaNet linear-attention dims; >0 on linear-attn layers.
    std::int32_t linear_num_value_heads = 0;
    std::int32_t linear_num_key_heads   = 0;
    std::int32_t linear_key_head_dim    = 0;
    std::int32_t linear_value_head_dim  = 0;
    std::int32_t linear_conv_kernel_dim = 0;
    // Sigmoid-gated output gate on full-attn layers (2x-wide q_proj).
    bool         attn_output_gate = false;
    // Partial rotary: only first rotary_dim = 2*floor(0.5*f*head_dim) dims
    // are rotated. 1.0 = full rotation.
    float        partial_rotary_factor = 1.0f;
    // Per-layer attention kind: "full_attention" / "linear_attention" /
    // "sliding_attention". Empty = uniform full attention.
    std::vector<std::string> layer_attn_types;

    // Convenience: KV embedding width (kv_heads * head_dim).
    std::int32_t n_embd_gqa() const { return num_key_value_heads * head_dim; }
};

}  // namespace pie_metal_driver::model
