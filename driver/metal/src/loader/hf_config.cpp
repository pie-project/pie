#include "hf_config.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "../model/arch.hpp"

namespace pie_metal_driver::loader {

namespace {

using nlohmann::json;

template <typename T>
T get_or(const json& j, const char* key, T fallback) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return fallback;
    try {
        return it->get<T>();
    } catch (...) {
        return fallback;
    }
}

// Newer HF checkpoints nest RoPE hyperparameters (rope_theta,
// partial_rotary_factor, ...) under a `rope_parameters` object instead of the
// top level. Two shapes occur:
//   * flat       (Qwen3.5 / qwen3_next): rope_parameters.rope_theta
//   * per-type   (gemma4):               rope_parameters.full_attention.rope_theta
//                                        rope_parameters.sliding_attention.rope_theta
// `rope_params_primary` returns the sub-object holding the FULL-attention rope
// params (the canonical rope_theta / partial_rotary_factor): the per-type
// `full_attention` object when present, else the flat `rope_parameters` object.
const json* rope_params_primary(const json& j) {
    auto it = j.find("rope_parameters");
    if (it == j.end() || !it->is_object()) return nullptr;
    if (auto fit = it->find("full_attention");
        fit != it->end() && fit->is_object()) {
        return &(*fit);  // gemma4 per-attention-type nesting
    }
    return &(*it);       // flat nesting (qwen3.5)
}

// Read a RoPE param: prefer the (possibly per-type) `rope_parameters` object,
// then the top-level key, then `fallback`.
template <typename T>
T get_rope_param(const json& j, const char* key, T fallback) {
    if (const json* rp = rope_params_primary(j)) {
        auto nit = rp->find(key);
        if (nit != rp->end() && !nit->is_null()) {
            try {
                return nit->get<T>();
            } catch (...) {
            }
        }
    }
    return get_or<T>(j, key, fallback);
}

// gemma4's sliding-attention layers use a separate rope base, nested at
// rope_parameters.sliding_attention.rope_theta (older gemma keeps it top-level
// as `rope_local_base_freq`). 0 = no distinct local base (graph uses rope_theta).
float get_rope_local_base(const json& j) {
    if (auto it = j.find("rope_parameters"); it != j.end() && it->is_object()) {
        if (auto sit = it->find("sliding_attention");
            sit != it->end() && sit->is_object()) {
            auto tit = sit->find("rope_theta");
            if (tit != sit->end() && tit->is_number()) {
                return tit->get<float>();
            }
        }
    }
    return get_or<float>(j, "rope_local_base_freq", 0.0f);
}

// gemma4's full-attention layers use `rope_type=proportional` (mlx-lm's
// ProportionalRoPE: rotated_dims = 0.25*head_dim but with the freq exponent
// over head_dim, not rotated_dims) — empirically near-exact as FULL rope, not
// the driver's standard partial rope. So only honor a nested
// `partial_rotary_factor` when the primary rope is a STANDARD type
// (default/linear/empty): qwen3.6 (rope_type=default) keeps 0.25; gemma4
// (rope_type=proportional) falls back to full rope (1.0). Top-level
// partial_rotary_factor (legacy phi-style) is always honored.
float get_partial_rotary_factor(const json& j) {
    if (const json* rp = rope_params_primary(j)) {
        const std::string rt =
            get_or<std::string>(*rp, "rope_type", get_or<std::string>(*rp, "type", ""));
        const bool standard = rt.empty() || rt == "default" || rt == "linear";
        if (standard) {
            auto nit = rp->find("partial_rotary_factor");
            if (nit != rp->end() && nit->is_number()) {
                return nit->get<float>();
            }
        } else {
            // Non-standard nested rope (gemma4 proportional) → full rope,
            // unless a legacy top-level factor is explicitly present.
            return get_or<float>(j, "partial_rotary_factor", 1.0f);
        }
    }
    return get_or<float>(j, "partial_rotary_factor", 1.0f);
}

template <typename T>
T require(const json& j, const char* key, const std::string& where) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        throw std::runtime_error("config.json (" + where + "): missing required key '" +
                                 std::string(key) + "'");
    }
    return it->get<T>();
}

// Gemma-4 (and other multimodal Gemma checkpoints) nest the text tower's
// hyperparameters under `text_config`, keeping a stub model_type at the top
// level. Return the sub-object that actually carries the transformer hparams.
const json& hparams_view(const json& root) {
    auto it = root.find("text_config");
    if (it != root.end() && it->is_object()) return *it;
    return root;
}

std::string resolve_model_type(const json& root, const json& view) {
    if (auto it = view.find("model_type"); it != view.end() && it->is_string()) {
        return it->get<std::string>();
    }
    if (auto it = root.find("model_type"); it != root.end() && it->is_string()) {
        return it->get<std::string>();
    }
    return "";
}

// Parse the (possibly nested) `rope_scaling` object into the metal fields.
void parse_rope_scaling(const json& j, model::ModelConfig& cfg) {
    auto it = j.find("rope_scaling");
    if (it == j.end() || !it->is_object()) return;
    const json& s = *it;

    const std::string rope_type =
        get_or<std::string>(s, "rope_type", get_or<std::string>(s, "type", ""));
    const bool has_llama3_keys = s.contains("low_freq_factor") || s.contains("high_freq_factor");

    if (rope_type == "llama3" || has_llama3_keys) {
        cfg.has_rope_scaling   = true;
        cfg.rope_scaling_type  = "llama3";
        cfg.rope_scaling_factor = get_or<float>(s, "factor", 8.0f);
        cfg.rope_scaling_low_freq_factor  = get_or<float>(s, "low_freq_factor", 1.0f);
        cfg.rope_scaling_high_freq_factor = get_or<float>(s, "high_freq_factor", 4.0f);
        cfg.rope_scaling_original_max_position =
            get_or<int>(s, "original_max_position_embeddings", cfg.max_position_embeddings);
    } else if (rope_type == "yarn") {
        cfg.has_rope_scaling  = true;
        cfg.rope_scaling_type = "yarn";
        cfg.rope_scaling_factor = get_or<float>(s, "factor", 1.0f);
        cfg.rope_yarn_beta_fast = get_or<float>(s, "beta_fast", 32.0f);
        cfg.rope_yarn_beta_slow = get_or<float>(s, "beta_slow", 1.0f);
        const float mscale_all_dim = get_or<float>(s, "mscale_all_dim", 0.0f);
        const float default_factor =
            mscale_all_dim > 0.f
                ? mscale_all_dim
                : (cfg.rope_scaling_factor > 1.f ? 0.1f * std::log(cfg.rope_scaling_factor) + 1.0f
                                                 : 1.0f);
        cfg.rope_yarn_attention_factor = get_or<float>(s, "attention_factor", default_factor);
        cfg.rope_scaling_original_max_position =
            get_or<int>(s, "original_max_position_embeddings", cfg.max_position_embeddings);
    } else if (rope_type == "linear") {
        cfg.has_rope_scaling   = true;
        cfg.rope_scaling_type  = "linear";
        cfg.rope_scaling_factor = get_or<float>(s, "factor", 1.0f);
    }
}

model::ModelConfig parse_doc(const json& root, const std::string& where) {
    const json& j = hparams_view(root);

    model::ModelConfig cfg;
    cfg.hf_model_type = resolve_model_type(root, j);
    cfg.arch          = model::hf_model_type_to_pie_arch(cfg.hf_model_type);
    cfg.torch_dtype   = get_or<std::string>(root, "torch_dtype",
                                            get_or<std::string>(j, "torch_dtype", "bfloat16"));

    // ── Core dims ──
    cfg.hidden_size         = require<int>(j, "hidden_size", where);
    cfg.num_hidden_layers   = require<int>(j, "num_hidden_layers", where);
    cfg.num_attention_heads = require<int>(j, "num_attention_heads", where);
    cfg.num_key_value_heads = get_or<int>(j, "num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim            = get_or<int>(j, "head_dim",
                                          cfg.num_attention_heads > 0
                                              ? cfg.hidden_size / cfg.num_attention_heads
                                              : 0);
    cfg.vocab_size              = require<int>(j, "vocab_size", where);
    cfg.max_position_embeddings = get_or<int>(j, "max_position_embeddings", 0);

    // `intermediate_size` is normally scalar; Gemma-3n stores a per-layer list.
    if (auto it = j.find("intermediate_size"); it != j.end() && it->is_array()) {
        if (!it->empty()) cfg.intermediate_size = it->front().get<int>();
    } else {
        cfg.intermediate_size = get_or<int>(j, "intermediate_size", 0);
    }

    // ── Norm / RoPE ──
    cfg.rms_norm_eps = get_or<float>(j, "rms_norm_eps",
                                     get_or<float>(j, "layer_norm_epsilon",
                                                   get_or<float>(j, "norm_eps", 1e-5f)));
    cfg.rope_theta          = get_rope_param<float>(j, "rope_theta", 10000.0f);
    cfg.rope_local_base_freq = get_rope_local_base(j);
    parse_rope_scaling(j, cfg);

    // ── Tied embeddings (Gemma family + Qwen3 default true; Llama false) ──
    const bool tie_default = cfg.hf_model_type == "qwen3" || cfg.hf_model_type == "qwen3_5" ||
                             cfg.hf_model_type.rfind("gemma", 0) == 0;
    cfg.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", tie_default);

    // ── 4-bit affine quantization (mlx-community style) ──
    // Top-level `quantization: {group_size, bits}`. When present, per-tensor
    // routing is by `.scales`-sibling presence (the binder's index-as-quant-map);
    // these scalars only carry the group_size/bits the quantized_matmul needs.
    if (auto it = root.find("quantization"); it != root.end() && it->is_object()) {
        cfg.quant_bits       = get_or<int>(*it, "bits", 4);
        cfg.quant_group_size = get_or<int>(*it, "group_size", 64);
    }

    // ── Sliding-window attention (present only when actually enabled) ──
    // Qwen2 ships `sliding_window` with `use_sliding_window:false` — the
    // window value is inert unless the flag (default true elsewhere) is set.
    const int sw = get_or<int>(j, "sliding_window", -1);
    if (sw > 0 && get_or<bool>(j, "use_sliding_window", true)) {
        cfg.sliding_window = sw;
    }

    // Explicit per-layer types when the checkpoint carries them ('s'/'g').
    if (auto it = j.find("layer_types"); it != j.end() && it->is_array()) {
        for (const auto& t : *it) {
            const std::string s = t.get<std::string>();
            cfg.layer_types.push_back(s.find("sliding") != std::string::npos ? 's' : 'g');
        }
    }

    // ── Gemma softcaps + query scaling ──
    if (auto it = j.find("attn_logit_softcapping"); it != j.end() && !it->is_null())
        cfg.attn_logit_softcapping = it->get<float>();
    if (auto it = j.find("final_logit_softcapping"); it != j.end() && !it->is_null())
        cfg.final_logit_softcapping = it->get<float>();
    if (auto it = j.find("query_pre_attn_scalar"); it != j.end() && !it->is_null())
        cfg.query_pre_attn_scalar = it->get<float>();

    // ── Mixture-of-Experts ──
    cfg.num_experts = get_or<int>(j, "num_local_experts",
                                  get_or<int>(j, "num_experts", 0));
    cfg.num_experts_per_tok   = get_or<int>(j, "num_experts_per_tok", 0);
    cfg.moe_intermediate_size = get_or<int>(j, "moe_intermediate_size", 0);
    cfg.norm_topk_prob        = get_or<bool>(j, "norm_topk_prob", true);
    cfg.routed_scaling_factor = get_or<float>(j, "routed_scaling_factor", 1.0f);
    cfg.n_shared_experts      = get_or<int>(j, "n_shared_experts", 0);
    cfg.shared_expert_intermediate_size =
        get_or<int>(j, "shared_expert_intermediate_size", 0);
    cfg.n_group               = get_or<int>(j, "n_group", 0);
    cfg.topk_group            = get_or<int>(j, "topk_group", 0);
    cfg.first_k_dense_replace = get_or<int>(j, "first_k_dense_replace", 0);

    // ── gemma4 ──
    cfg.gemma4_enable_moe    = get_or<bool>(j, "enable_moe_block", false);
    cfg.global_head_dim      = get_or<int>(j, "global_head_dim", 0);
    // num_global_key_value_heads is often JSON null → treat as 0 (= fall back to
    // num_key_value_heads). Only read when it's an actual number.
    if (auto it = j.find("num_global_key_value_heads");
        it != j.end() && it->is_number_integer()) {
        cfg.num_global_kv_heads = it->get<int>();
    }
    // PLE feature width: HF spells it hidden_size_per_layer_input (some dumps
    // gemma_hidden_size_per_layer_input).
    cfg.per_layer_emb_dim    = get_or<int>(
        j, "hidden_size_per_layer_input",
        get_or<int>(j, "gemma_hidden_size_per_layer_input", 0));
    cfg.num_kv_shared_layers = get_or<int>(j, "num_kv_shared_layers", 0);

    // ── qwen3.6 (Qwen3.5 hybrid linear-attention) ──
    cfg.linear_num_value_heads = get_or<int>(j, "linear_num_value_heads", 0);
    cfg.linear_num_key_heads   = get_or<int>(j, "linear_num_key_heads", 0);
    cfg.linear_key_head_dim    = get_or<int>(j, "linear_key_head_dim", 0);
    cfg.linear_value_head_dim  = get_or<int>(j, "linear_value_head_dim", 0);
    cfg.linear_conv_kernel_dim = get_or<int>(j, "linear_conv_kernel_dim", 0);
    cfg.attn_output_gate       = get_or<bool>(j, "attn_output_gate", false);
    cfg.partial_rotary_factor  = get_partial_rotary_factor(j);
    if (auto it = j.find("layer_types");
        it != j.end() && it->is_array() && cfg.layer_attn_types.empty()) {
        for (const auto& t : *it) cfg.layer_attn_types.push_back(t.get<std::string>());
    }

    return cfg;
}

}  // namespace

model::ModelConfig parse_hf_config(const std::string& hf_path) {
    const std::string file = hf_path + "/config.json";
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open config.json: " + file);
    json root;
    in >> root;
    return parse_doc(root, file);
}

model::ModelConfig parse_hf_config_json(const std::string& json_text) {
    return parse_doc(json::parse(json_text), "<memory>");
}

}  // namespace pie_metal_driver::loader
