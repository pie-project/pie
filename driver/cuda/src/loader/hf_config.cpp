#include "loader/hf_config.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace pie_cuda_driver {

namespace {

template <typename T>
T require(const nlohmann::json& j, const char* key, const std::string& path) {
    if (!j.contains(key)) {
        throw std::runtime_error("config.json (" + path + "): missing key '" + key + "'");
    }
    return j[key].get<T>();
}

template <typename T>
T optional(const nlohmann::json& j, const char* key, T default_value) {
    if (!j.contains(key) || j[key].is_null()) return default_value;
    return j[key].get<T>();
}

// Qwen3-specific signal: HF marks `use_qk_norm` implicitly via model_type.
// Some other archs use the same flag explicitly. Until we add per-arch
// metadata, derive it here.
bool infer_qk_norm(const std::string& model_type, const nlohmann::json& j) {
    if (j.contains("use_qk_norm")) return j["use_qk_norm"].get<bool>();
    return model_type == "qwen3" || model_type == "qwen3_5";
}

}  // namespace

HfConfig parse_hf_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config.json: " + path.string());

    nlohmann::json j;
    in >> j;
    const auto path_str = path.string();

    HfConfig cfg;

    if (!j.contains("architectures") || !j["architectures"].is_array() || j["architectures"].empty()) {
        throw std::runtime_error("config.json: missing or empty 'architectures'");
    }
    cfg.arch_name = j["architectures"][0].get<std::string>();
    cfg.model_type = optional<std::string>(j, "model_type", "");

    cfg.hidden_size              = require<int>(j, "hidden_size", path_str);
    cfg.intermediate_size        = require<int>(j, "intermediate_size", path_str);
    cfg.num_hidden_layers        = require<int>(j, "num_hidden_layers", path_str);
    cfg.num_attention_heads      = require<int>(j, "num_attention_heads", path_str);
    cfg.num_key_value_heads      = optional<int>(j, "num_key_value_heads", cfg.num_attention_heads);
    cfg.head_dim                 = optional<int>(j, "head_dim", cfg.hidden_size / cfg.num_attention_heads);
    cfg.vocab_size               = require<int>(j, "vocab_size", path_str);
    cfg.max_position_embeddings  = require<int>(j, "max_position_embeddings", path_str);

    cfg.rms_norm_eps = require<float>(j, "rms_norm_eps", path_str);
    cfg.hidden_act   = optional<std::string>(j, "hidden_act", "silu");

    cfg.rope_theta       = optional<float>(j, "rope_theta", 10000.0f);
    cfg.has_rope_scaling = j.contains("rope_scaling") && !j["rope_scaling"].is_null();

    cfg.tie_word_embeddings = optional<bool>(j, "tie_word_embeddings", false);
    cfg.attention_bias      = optional<bool>(j, "attention_bias", false);
    cfg.use_qk_norm         = infer_qk_norm(cfg.model_type, j);

    cfg.torch_dtype = optional<std::string>(j, "torch_dtype", "bfloat16");

    return cfg;
}

}  // namespace pie_cuda_driver
