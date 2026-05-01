#include "hf_config.hpp"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace pie_portable_driver {

const char* pie_arch_name(PieArch a) {
    switch (a) {
        case PieArch::Qwen3:    return "qwen3";
        case PieArch::Qwen2:    return "qwen2";
        case PieArch::Llama3:   return "llama3";
        case PieArch::Gemma2:   return "gemma2";
        case PieArch::Gemma3:   return "gemma3";
        case PieArch::Gemma4:   return "gemma4";
        case PieArch::Mistral3: return "mistral3";
        case PieArch::Olmo3:    return "olmo3";
        case PieArch::GptOss:   return "gptoss";
        case PieArch::Phi3:     return "phi3";
        case PieArch::Mixtral:  return "mixtral";
        case PieArch::Qwen3_5:  return "qwen3_5";
    }
    return "?";
}

PieArch hf_model_type_to_pie_arch(const std::string& hf_model_type) {
    // Mirrors `pie/src/pie_driver/hf_utils.py::HF_TO_PIE_ARCH`. Source-of-truth
    // is that file; keep this table in sync.
    if (hf_model_type == "qwen3")       return PieArch::Qwen3;
    if (hf_model_type == "qwen2")       return PieArch::Qwen2;
    if (hf_model_type == "llama")       return PieArch::Llama3;
    if (hf_model_type == "gemma2")        return PieArch::Gemma2;
    if (hf_model_type == "gemma3" ||
        hf_model_type == "gemma3_text")   return PieArch::Gemma3;
    if (hf_model_type == "gemma4" ||
        hf_model_type == "gemma4_text")   return PieArch::Gemma4;
    if (hf_model_type == "mistral")     return PieArch::Mistral3;
    if (hf_model_type == "mistral3")    return PieArch::Mistral3;
    if (hf_model_type == "olmo3")       return PieArch::Olmo3;
    if (hf_model_type == "gpt_oss")     return PieArch::GptOss;
    if (hf_model_type == "gptoss")      return PieArch::GptOss;
    if (hf_model_type == "phi3")        return PieArch::Phi3;
    if (hf_model_type == "mixtral")     return PieArch::Mixtral;
    if (hf_model_type == "qwen3_5")     return PieArch::Qwen3_5;
    if (hf_model_type == "qwen3_moe")   return PieArch::Qwen3_5;
    throw std::runtime_error(
        "hf_config: unsupported model_type '" + hf_model_type + "'");
}

namespace {

template <typename T>
std::optional<T> get_opt(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    return it->get<T>();
}

template <typename T>
T get_or(const nlohmann::json& j, const char* key, T def) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    return it->get<T>();
}

}  // namespace

Hparams parse_hf_config(const std::filesystem::path& config_json_path) {
    std::ifstream f(config_json_path);
    if (!f) {
        throw std::runtime_error(
            "hf_config: cannot open " + config_json_path.string());
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(f);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "hf_config: parse failed for " + config_json_path.string() +
            ": " + e.what());
    }

    Hparams h;

    h.hf_model_type = j.at("model_type").get<std::string>();
    h.arch = hf_model_type_to_pie_arch(h.hf_model_type);
    h.torch_dtype = get_or<std::string>(j, "torch_dtype", "float16");

    h.num_hidden_layers = j.at("num_hidden_layers").get<std::int32_t>();
    h.num_attention_heads = j.at("num_attention_heads").get<std::int32_t>();
    // Some configs default num_key_value_heads to num_attention_heads.
    h.num_key_value_heads =
        get_or<std::int32_t>(j, "num_key_value_heads", h.num_attention_heads);
    h.hidden_size = j.at("hidden_size").get<std::int32_t>();
    h.intermediate_size = j.at("intermediate_size").get<std::int32_t>();
    h.vocab_size = j.at("vocab_size").get<std::int32_t>();
    h.max_position_embeddings =
        get_or<std::int32_t>(j, "max_position_embeddings", 4096);

    if (j.contains("head_dim") && !j["head_dim"].is_null()) {
        h.head_dim = j["head_dim"].get<std::int32_t>();
    } else {
        if (h.num_attention_heads <= 0) {
            throw std::runtime_error("hf_config: num_attention_heads is zero");
        }
        h.head_dim = h.hidden_size / h.num_attention_heads;
    }

    h.rms_norm_eps = get_or<float>(j, "rms_norm_eps", 1e-6f);
    h.rope_theta = get_or<float>(j, "rope_theta", 1e6f);
    h.rope_local_base_freq = get_or<float>(j, "rope_local_base_freq", 0.0f);
    // HF's PretrainedConfig defaults this to True; specific archs override
    // (Llama-3 base / Llama-3.1+ untie). When the config omits the flag,
    // assume tied.
    h.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", true);

    if (auto sw = get_opt<std::int32_t>(j, "sliding_window")) {
        h.sliding_window = sw;
    }
    h.use_sliding_window = get_or<bool>(j, "use_sliding_window", false);

    auto rs_it = j.find("rope_scaling");
    if (rs_it != j.end() && !rs_it->is_null()) {
        h.has_rope_scaling = true;
        const auto& rs = *rs_it;
        // Newer HF configs use `rope_type`; older ones use `type`.
        h.rope_scaling_type =
            get_or<std::string>(rs, "rope_type",
                                get_or<std::string>(rs, "type", ""));
        h.rope_scaling_factor = get_or<float>(rs, "factor", 1.0f);
        h.rope_scaling_low_freq_factor =
            get_or<float>(rs, "low_freq_factor", 1.0f);
        h.rope_scaling_high_freq_factor =
            get_or<float>(rs, "high_freq_factor", 4.0f);
        h.rope_scaling_original_max_position =
            get_or<std::int32_t>(rs, "original_max_position_embeddings", 0);
    }

    if (auto v = get_opt<float>(j, "attn_logit_softcapping")) {
        h.attn_logit_softcapping = v;
    }
    if (auto v = get_opt<float>(j, "final_logit_softcapping")) {
        h.final_logit_softcapping = v;
    }
    if (auto v = get_opt<float>(j, "query_pre_attn_scalar")) {
        h.query_pre_attn_scalar = v;
    }

    // ── MoE ──
    // Different repos use different field names: Mixtral uses
    // `num_local_experts`, Qwen-MoE / GPT-OSS use `num_experts`.
    h.num_experts =
        get_or<std::int32_t>(j, "num_experts",
            get_or<std::int32_t>(j, "num_local_experts", 0));
    h.num_experts_per_tok =
        get_or<std::int32_t>(j, "num_experts_per_tok", 0);
    h.moe_intermediate_size =
        get_or<std::int32_t>(j, "moe_intermediate_size",
            h.intermediate_size);  // fallback
    h.norm_topk_prob = get_or<bool>(j, "norm_topk_prob", true);

    return h;
}

}  // namespace pie_portable_driver
