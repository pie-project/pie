#include "arch_spec.hpp"

#include <cmath>

namespace pie_metal_driver::model {

namespace {

// YaRN smooth-ramp parameters (Ministral 3 etc.). Plain theta-only RoPE
// leaves yarn_n_ctx_orig at 0; the graph builder reads that as "YaRN off".
void apply_yarn(ArchSpec& s, const ModelConfig& c) {
    if (!c.has_rope_scaling || c.rope_scaling_type != "yarn") return;
    if (c.rope_scaling_original_max_position <= 0) return;
    if (c.rope_scaling_factor <= 0.0f) return;
    s.yarn_n_ctx_orig = c.rope_scaling_original_max_position;
    s.yarn_freq_scale = 1.0f / c.rope_scaling_factor;
    // HF default when attention_factor isn't given: 0.1*ln(factor) + 1.0.
    s.yarn_attn_factor =
        (c.rope_yarn_attention_factor > 0.0f)
            ? c.rope_yarn_attention_factor
            : (0.1f * std::log(c.rope_scaling_factor) + 1.0f);
    s.yarn_beta_fast = c.rope_yarn_beta_fast;
    s.yarn_beta_slow = c.rope_yarn_beta_slow;
}

// Gemma family: pre-FFN / post-attn / post-FFN norms, GeGLU, sqrt(d) embed.
void apply_gemma_norms(ArchSpec& s) {
    s.has_pre_ffn_norm      = true;
    s.has_post_attn_norm    = true;
    s.has_post_ffn_norm     = true;
    s.ffn_use_gelu          = true;
    s.scale_embed_by_sqrt_d = true;
}

void apply_moe_flags(ArchSpec& s, const ModelConfig& c) {
    s.n_experts         = c.num_experts;
    s.n_experts_per_tok = c.num_experts_per_tok;
    s.moe_norm_topk     = c.norm_topk_prob;
    s.moe_routed_scale  = c.routed_scaling_factor;
    s.moe_n_group       = c.n_group;
    s.moe_topk_group    = c.topk_group;
    s.n_shared_experts  = c.n_shared_experts;
    s.first_k_dense     = c.first_k_dense_replace;
}

}  // namespace

ArchSpec arch_spec_for(PieArch arch, const ModelConfig& c) {
    ArchSpec s;
    switch (arch) {
        case PieArch::Qwen3:
            s.has_qk_norm = true;
            break;

        case PieArch::Qwen2:
            s.has_qkv_bias = true;
            break;

        case PieArch::Llama3:
            // No structural flags; llama-3.1 NTK rope lives in freq_factors
            // synthesized at load time.
            break;

        case PieArch::Mistral3:
            // All layers sliding (when configured) + optional YaRN (Ministral 3).
            if (c.sliding_window) {
                s.sliding_window = *c.sliding_window;
                s.layer_pattern.assign(c.num_hidden_layers, 's');
            }
            apply_yarn(s, c);
            break;

        case PieArch::Qwen3Moe:
            apply_moe_flags(s, c);
            s.has_qk_norm = true;  // qwen3-moe keeps qwen3 per-head qk-norm
            break;

        case PieArch::Mixtral:
            apply_moe_flags(s, c);
            break;

        case PieArch::Gemma2:
            apply_gemma_norms(s);
            s.norm_weight_plus_one = true;
            if (c.query_pre_attn_scalar) s.query_pre_attn_scalar = *c.query_pre_attn_scalar;
            if (c.attn_logit_softcapping)  s.attn_softcap  = *c.attn_logit_softcapping;
            if (c.final_logit_softcapping) s.final_softcap = *c.final_logit_softcapping;
            // Gemma2 alternates even='s' / odd='g'.
            if (c.sliding_window) {
                s.sliding_window = *c.sliding_window;
                s.layer_pattern.resize(c.num_hidden_layers);
                for (std::int32_t i = 0; i < c.num_hidden_layers; ++i) {
                    s.layer_pattern[i] = (i % 2 == 0) ? 's' : 'g';
                }
            }
            break;

        case PieArch::Gemma3:
            apply_gemma_norms(s);
            s.norm_weight_plus_one = true;
            s.has_qk_norm = true;
            if (c.query_pre_attn_scalar) s.query_pre_attn_scalar = *c.query_pre_attn_scalar;
            // Gemma3: every 6th layer global, rest sliding.
            if (c.sliding_window) {
                s.sliding_window = *c.sliding_window;
                s.layer_pattern.resize(c.num_hidden_layers);
                for (std::int32_t i = 0; i < c.num_hidden_layers; ++i) {
                    s.layer_pattern[i] = ((i + 1) % 6 == 0) ? 'g' : 's';
                }
            }
            break;

        case PieArch::Gemma4:
            apply_gemma_norms(s);
            s.norm_weight_plus_one = true;
            s.has_qk_norm = true;
            if (c.query_pre_attn_scalar) s.query_pre_attn_scalar = *c.query_pre_attn_scalar;
            if (c.attn_logit_softcapping)  s.attn_softcap  = *c.attn_logit_softcapping;
            if (c.final_logit_softcapping) s.final_softcap = *c.final_logit_softcapping;
            // Per-attn-type sliding/full schedule (same every-6 default as
            // gemma3 unless layer_types is published). Sliding layers use the
            // local rope base (cfg.rope_local_base_freq) in the graph.
            if (c.sliding_window) {
                s.sliding_window = *c.sliding_window;
                s.layer_pattern.resize(c.num_hidden_layers);
                for (std::int32_t i = 0; i < c.num_hidden_layers; ++i) {
                    s.layer_pattern[i] = ((i + 1) % 6 == 0) ? 'g' : 's';
                }
            }
            // gemma4 structural extras (graph branches on these).
            s.gemma4_parallel_moe  = c.gemma4_enable_moe;
            s.per_layer_emb_dim    = c.per_layer_emb_dim;
            s.num_kv_shared_layers = c.num_kv_shared_layers;
            if (c.gemma4_enable_moe) apply_moe_flags(s, c);
            break;

        case PieArch::Qwen36:
            // Qwen3.5/3.6 hybrid: qk-norm + optional output-gate + partial rope,
            // Gemma-style (1+w) final norm, MoE w/ sigmoid shared expert.
            s.has_qk_norm          = true;
            s.norm_weight_plus_one = true;
            s.attn_output_gate     = c.attn_output_gate;
            s.partial_rotary_factor = c.partial_rotary_factor;
            s.linear_num_value_heads = c.linear_num_value_heads;
            s.linear_num_key_heads   = c.linear_num_key_heads;
            s.linear_key_head_dim    = c.linear_key_head_dim;
            s.linear_value_head_dim  = c.linear_value_head_dim;
            s.linear_conv_kernel_dim = c.linear_conv_kernel_dim;
            // Per-layer kind: 'l' = Gated-DeltaNet linear-attn, 'g' = full.
            if (!c.layer_attn_types.empty()) {
                s.layer_pattern.resize(c.layer_attn_types.size());
                for (std::size_t i = 0; i < c.layer_attn_types.size(); ++i) {
                    s.layer_pattern[i] =
                        (c.layer_attn_types[i] == "linear_attention") ? 'l' : 'g';
                }
            }
            if (c.num_experts > 0) apply_moe_flags(s, c);
            apply_yarn(s, c);
            break;

        default:
            break;
    }
    return s;
}

}  // namespace pie_metal_driver::model
