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

        default:
            break;
    }
    return s;
}

}  // namespace pie_metal_driver::model
