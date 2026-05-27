#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace pie_driver_common {

inline const nlohmann::json& text_config_view(const nlohmann::json& root) {
    if (root.contains("text_config") && root["text_config"].is_object()) {
        return root["text_config"];
    }
    return root;
}

inline std::string json_model_type(const nlohmann::json& j) {
    if (j.contains("model_type") && !j["model_type"].is_null()) {
        return j["model_type"].get<std::string>();
    }
    return {};
}

struct HfConfigJsonView {
    const nlohmann::json& root;
    const nlohmann::json& text;
    std::string outer_model_type;
    std::string text_model_type;

    std::string text_or_outer_model_type() const {
        return !text_model_type.empty() ? text_model_type : outer_model_type;
    }

    std::string outer_or_text_model_type() const {
        return !outer_model_type.empty() ? outer_model_type : text_model_type;
    }
};

inline HfConfigJsonView hf_config_json_view(const nlohmann::json& root) {
    const auto& text = text_config_view(root);
    return HfConfigJsonView{
        root,
        text,
        json_model_type(root),
        json_model_type(text),
    };
}

inline bool is_nested_rope_parameters(const nlohmann::json& rope_parameters) {
    return rope_parameters.is_object() && !rope_parameters.empty() &&
           rope_parameters.begin().value().is_object();
}

inline const nlohmann::json* flat_rope_parameters_view(
    const nlohmann::json& text_config) {
    auto rp_it = text_config.find("rope_parameters");
    if (rp_it != text_config.end() && rp_it->is_object() &&
        !is_nested_rope_parameters(*rp_it)) {
        return &(*rp_it);
    }
    return nullptr;
}

inline const nlohmann::json* flat_rope_config_view(
    const nlohmann::json& text_config) {
    auto rs_it = text_config.find("rope_scaling");
    if (rs_it != text_config.end() && rs_it->is_object()) {
        return &(*rs_it);
    }
    return flat_rope_parameters_view(text_config);
}

template <typename T>
T json_require(const nlohmann::json& j,
               const char* key,
               const std::string& path) {
    if (!j.contains(key)) {
        throw std::runtime_error(
            "config.json (" + path + "): missing key '" + key + "'");
    }
    return j[key].get<T>();
}

template <typename T>
std::optional<T> json_get_opt(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return std::nullopt;
    return it->get<T>();
}

template <typename T>
T json_get_or(const nlohmann::json& j, const char* key, T default_value) {
    auto v = json_get_opt<T>(j, key);
    return v ? *v : default_value;
}

}  // namespace pie_driver_common
