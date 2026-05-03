#include "arch_spec.hpp"

#include <cmath>

namespace pie_portable_driver {

namespace {

// YaRN RoPE smooth-ramp parameters (olmo3, Ministral 3, gpt-oss). Plain
// θ-only RoPE leaves yarn_n_ctx_orig at 0; the graph builder reads that
// as "YaRN off."
inline void apply_yarn_(ArchSpec& s, const Hparams& h) {
    if (!h.has_rope_scaling || h.rope_scaling_type != "yarn") return;
    if (h.rope_scaling_original_max_position <= 0) return;
    if (h.rope_scaling_factor <= 0.0f) return;
    s.yarn_n_ctx_orig  = h.rope_scaling_original_max_position;
    s.yarn_freq_scale  = 1.0f / h.rope_scaling_factor;
    // HF default when attention_factor isn't given: 0.1 * ln(factor) + 1.0
    // (matches transformers' _compute_yarn_parameters and DeepSeek mscale).
    s.yarn_attn_factor =
        (h.rope_yarn_attention_factor > 0.0f)
            ? h.rope_yarn_attention_factor
            : (0.1f * std::log(h.rope_scaling_factor) + 1.0f);
    s.yarn_beta_fast   = h.rope_yarn_beta_fast;
    s.yarn_beta_slow   = h.rope_yarn_beta_slow;
}

// Gemma family base: extra RMSNorms (pre-FFN, post-attn, post-FFN), GeGLU,
// embed scaled by sqrt(hidden_size). Gemma2/3 layer their own
// `norm_weight_plus_one`; gemma4 uses `gemma4_norm_weight_direct` instead.
inline void apply_gemma_norms_(ArchSpec& s) {
    s.has_pre_ffn_norm      = true;
    s.has_post_attn_norm    = true;
    s.has_post_ffn_norm     = true;
    s.ffn_use_gelu          = true;
    s.scale_embed_by_sqrt_d = true;
}

// Per-layer attention pattern. Mistral3 is "all sliding"; gemma2 alternates
// even/odd; gemma3 makes every 6th layer global; olmo3/gpt-oss/gemma4 read
// the pattern from `hparams.layer_types` directly.
enum class SwaPattern {
    None,            // no SWA
    AllSliding,      // mistral3
    AltGemma2,       // gemma2: even='s', odd='g'
    Every6Gemma3,    // gemma3: every 6th layer 'g', rest 's'
    FromLayerTypes,  // olmo3, gpt-oss, gemma4 — copy h.layer_types
};

inline void apply_swa_pattern_(ArchSpec& s, const Hparams& h, SwaPattern kind) {
    if (kind == SwaPattern::None) return;
    if (h.sliding_window) s.sliding_window = *h.sliding_window;

    switch (kind) {
        case SwaPattern::None:
            break;
        case SwaPattern::AllSliding:
            if (h.sliding_window) {
                s.layer_pattern.assign(h.num_hidden_layers, 's');
            }
            break;
        case SwaPattern::AltGemma2:
            if (h.sliding_window) {
                s.layer_pattern.resize(h.num_hidden_layers);
                for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
                    s.layer_pattern[i] = (i % 2 == 0) ? 's' : 'g';
                }
            }
            break;
        case SwaPattern::Every6Gemma3:
            if (h.sliding_window) {
                s.layer_pattern.resize(h.num_hidden_layers);
                for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
                    // Every 6th layer is global; the rest are sliding.
                    s.layer_pattern[i] = ((i + 1) % 6 == 0) ? 'g' : 's';
                }
            }
            break;
        case SwaPattern::FromLayerTypes:
            if (!h.layer_types.empty()) {
                s.layer_pattern.assign(h.layer_types.begin(),
                                       h.layer_types.end());
            }
            break;
    }
}

// MoE flags shared by Mixtral / GptOss / Qwen3-MoE.
inline void apply_moe_flags_(ArchSpec& s, const Hparams& h) {
    s.n_experts         = h.num_experts;
    s.n_experts_per_tok = h.num_experts_per_tok;
    s.moe_norm_topk     = h.norm_topk_prob;
}

}  // namespace

ArchSpec arch_spec_for(PieArch a, const Hparams& h) {
    ArchSpec s;
    switch (a) {
        case PieArch::Qwen3:
            s.has_qk_norm = true;
            break;
        case PieArch::Qwen2:
            s.has_qkv_bias = true;
            break;
        case PieArch::Llama3:
        case PieArch::Phi3:
            // No per-arch flags. Llama3 carries its NTK-by-parts via
            // `weights.freq_factors` (synthesized at load time); Phi3's
            // fused QKV is handled at load time too.
            break;
        case PieArch::Phi3Small:
            // Per-arch dispatch lives in graph_phi3small.cpp; ArchSpec
            // is unused for this arch beyond defaults.
            break;
        case PieArch::PhiMoe:
            // Per-arch dispatch lives in graph_phimoe.cpp.
            apply_moe_flags_(s, h);
            break;
        case PieArch::Mistral3:
            // SWA on every layer (when configured) plus optional YaRN
            // (Ministral 3). Mistral 7B v0.3 has no YaRN; apply_yarn_
            // no-ops in that case.
            apply_swa_pattern_(s, h, SwaPattern::AllSliding);
            apply_yarn_(s, h);
            break;
        case PieArch::Gemma2:
            apply_gemma_norms_(s);
            s.norm_weight_plus_one = true;
            if (h.query_pre_attn_scalar) {
                s.query_pre_attn_scalar = *h.query_pre_attn_scalar;
            }
            if (h.attn_logit_softcapping)  s.attn_softcap  = *h.attn_logit_softcapping;
            if (h.final_logit_softcapping) s.final_softcap = *h.final_logit_softcapping;
            apply_swa_pattern_(s, h, SwaPattern::AltGemma2);
            break;
        case PieArch::Gemma3:
            apply_gemma_norms_(s);
            s.norm_weight_plus_one = true;
            s.has_qk_norm = true;
            if (h.query_pre_attn_scalar) {
                s.query_pre_attn_scalar = *h.query_pre_attn_scalar;
            }
            apply_swa_pattern_(s, h, SwaPattern::Every6Gemma3);
            break;
        case PieArch::Gemma3n:
            // Gemma 3-style pre/post norm sandwich + GeGLU + per-head
            // qk_norm + per-layer-type SWA pattern (`layer_types`). Unlike
            // Gemma 2/3, Gemma 3n's RMSNorm stores weights centered at 1
            // (init=ones) and applies a direct `x * w` (NOT `x * (1+w)`),
            // so `norm_weight_plus_one` stays FALSE here even though the
            // overall norm-sandwich layout matches Gemma 3.
            // Attention scale is 1.0 (Q is pre-normalized via q_norm — the
            // 1/sqrt(head_dim) factor is absorbed into the q_norm weights).
            // Final softcap (30.0) preserved. AltUp / PLE / Laurel /
            // activation sparsity are layered on top inside the graph builder.
            apply_gemma_norms_(s);
            s.has_qk_norm = true;
            s.gemma4_unit_sm_scale = true;  // attention scale = 1.0
            s.gemma4_v_norm        = true;  // pure RMS-norm on V (no scale)
            // KV-share: last `num_kv_shared_layers` (=10 on E2B/E4B) reuse
            // upstream non-shared layer's K/V matched by attention type.
            // The graph builder reads `gemma4_first_shared` to know where
            // the cutoff is (mirrors gemma4 wiring; same field name reused).
            s.gemma4_first_shared = h.gemma4_num_kv_shared_layers > 0
                ? h.num_hidden_layers - h.gemma4_num_kv_shared_layers
                : h.num_hidden_layers;
            apply_swa_pattern_(s, h, SwaPattern::FromLayerTypes);
            if (h.final_logit_softcapping) s.final_softcap = *h.final_logit_softcapping;
            break;
        case PieArch::Mixtral:
        case PieArch::GptOss:
        case PieArch::Qwen3Moe:
            apply_moe_flags_(s, h);
            // Qwen3-MoE retains the qwen3 attention block (per-head Q/K
            // RMSNorm). Mixtral and gpt-oss leave L.q_norm / L.k_norm null
            // at load time, so this flag is a no-op for them.
            if (a == PieArch::Qwen3Moe) s.has_qk_norm = true;
            // gpt-oss: YaRN + per-layer mixed sliding from `layer_types` +
            // Q/K/V biases (and o_proj bias, applied unconditionally when
            // L.o_proj_b is non-null) + attention sinks.
            if (a == PieArch::GptOss) {
                s.has_qkv_bias = true;
                apply_swa_pattern_(s, h, SwaPattern::FromLayerTypes);
                apply_yarn_(s, h);
            }
            break;
        case PieArch::Gemma4:
            // Gemma 4 reuses gemma2/3's pre/post norm sandwich + GeGLU +
            // sqrt(d) embed, but stores RMSNorm weights centered at 1
            // (init=ones) instead of 0 — `gemma4_norm_weight_direct=true`
            // suppresses the (1+w) wrapping. Adds: V-norm, per-layer
            // scalar, PLE, dual head_dim, dual rope_theta per layer-type.
            apply_gemma_norms_(s);
            s.has_qk_norm               = true;
            s.gemma4_norm_weight_direct = true;
            s.gemma4_v_norm             = true;
            s.gemma4_layer_scalar       = true;
            s.gemma4_unit_sm_scale      = true;
            s.gemma4_ple_enabled        = h.gemma4_ple_dim > 0;
            s.gemma4_ple_dim            = h.gemma4_ple_dim;
            s.gemma4_head_dim_global    =
                h.gemma4_head_dim_global > 0 ? h.gemma4_head_dim_global : h.head_dim;
            s.gemma4_rope_theta_full    = h.gemma4_rope_theta_full;
            s.gemma4_rope_theta_sliding = h.gemma4_rope_theta_sliding;
            s.gemma4_partial_full       = h.gemma4_rope_partial_factor_full;
            s.gemma4_partial_sliding    = h.gemma4_rope_partial_factor_sliding;
            s.gemma4_first_shared       = h.gemma4_num_kv_shared_layers > 0
                ? h.num_hidden_layers - h.gemma4_num_kv_shared_layers
                : h.num_hidden_layers;
            apply_swa_pattern_(s, h, SwaPattern::FromLayerTypes);
            if (h.final_logit_softcapping) s.final_softcap = *h.final_logit_softcapping;
            break;
        case PieArch::Qwen3_5:
            // Qwen 3.5 / 3.6 has its own graph builder (graph_qwen3_5.cpp);
            // most ArchSpec flags don't apply (per-layer-type dispatch
            // happens inside that builder). Set the bare minimum so other
            // parts of the engine see a consistent picture.
            s.has_qk_norm = false;  // q/k_norm tensors only on full_attention layers
            break;
        case PieArch::Olmo3:
            // Post-norm-only (input/pre-FFN norms absent — `attn_norm` /
            // `ffn_norm` are null at load time and skipped at graph time).
            // QK-norm with full-vector ([hidden_size] / [kv_dim]) weights.
            // YaRN + per-layer mixed sliding from layer_types.
            s.has_qk_norm        = true;
            s.qk_norm_full       = true;
            s.has_post_attn_norm = true;
            s.has_post_ffn_norm  = true;
            apply_swa_pattern_(s, h, SwaPattern::FromLayerTypes);
            apply_yarn_(s, h);
            break;
        default:
            break;
    }
    return s;
}

}  // namespace pie_portable_driver
