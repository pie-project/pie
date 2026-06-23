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
    cfg.rope_theta          = get_or<float>(j, "rope_theta", 10000.0f);
    cfg.rope_local_base_freq = get_or<float>(j, "rope_local_base_freq", 0.0f);
    parse_rope_scaling(j, cfg);

    // ── Tied embeddings (Gemma family + Qwen3 default true; Llama false) ──
    const bool tie_default = cfg.hf_model_type == "qwen3" || cfg.hf_model_type == "qwen3_5" ||
                             cfg.hf_model_type.rfind("gemma", 0) == 0;
    cfg.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", tie_default);

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
