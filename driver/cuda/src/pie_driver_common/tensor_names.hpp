#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace pie_driver_common {

inline bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

inline bool ends_with(std::string_view s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.substr(s.size() - suffix.size()) == suffix;
}

inline bool starts_with_any(std::string_view s,
                            const std::vector<std::string>& prefixes) {
    for (const auto& prefix : prefixes) {
        if (starts_with(s, prefix)) return true;
    }
    return false;
}

inline std::string strip_prefix(std::string_view s, std::string_view prefix) {
    if (prefix.empty() || !starts_with(s, prefix)) return std::string(s);
    return std::string(s.substr(prefix.size()));
}

inline std::string strip_suffix(std::string_view s, std::string_view suffix) {
    if (suffix.empty() || !ends_with(s, suffix)) return std::string(s);
    return std::string(s.substr(0, s.size() - suffix.size()));
}

inline std::string apply_tensor_prefix(std::string_view name,
                                       std::string_view tensor_prefix) {
    if (tensor_prefix.empty()) return std::string(name);
    constexpr std::string_view model_prefix = "model.";
    if (starts_with(name, model_prefix)) {
        return std::string(tensor_prefix) +
               std::string(name.substr(model_prefix.size()));
    }
    return std::string(tensor_prefix) + std::string(name);
}

}  // namespace pie_driver_common
