#pragma once

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <filesystem>

#include <iostream>


// ======================================================================================
// MARK: - Data Structures
// ======================================================================================

/**
 * @brief Configuration for the L4ma model architecture.
 */
struct L4maConfig {
    std::string type;
    int num_layers;
    int num_query_heads;
    int num_key_value_heads;
    int head_size;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    bool use_qkv_bias;
    float rms_norm_eps;
    float rope_factor;
    float rope_high_frequency_factor;
    float rope_low_frequency_factor;
    float rope_theta;
};

/**
 * @brief Configuration for the tokenizer.
 */
struct TokenizerInfo {
    std::string type;
    std::map<int, std::vector<uint8_t>> merge_table;
    std::string split_regex;
    std::map<int, std::string> special_tokens;
};

/**
 * @brief Holds the final, merged configuration for the application.
 */
struct AppConfig {
    // Network
    std::string host = "localhost";
    int port = 10123;
    std::string controller_host = "localhost";
    int controller_port = 9123;
    std::optional<std::string> internal_auth_token;

    // Model & Cache
    std::string model_name;
    std::filesystem::path cache_dir;

    // Engine Parameters
    int kv_page_size = 32;
    int dist_size = 32;
    int max_num_kv_pages = 1000;
    int max_num_embeds = 50000;

    // Hardware
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
};

/**
 * @brief Top-level structure holding the entire parsed model configuration.
 */
struct ModelMetadata {
    std::string name;
    std::string description;
    std::vector<std::string> parameters;
    L4maConfig architecture;
    TokenizerInfo tokenizer;
    std::string template_type;
    std::string template_content;
};



// ======================================================================================
// MARK: - Helper Functions for Printing
// ======================================================================================

namespace detail {

/**
 * @brief Helper to print a map with a given title. Marked inline to prevent linker errors.
 */
template<typename K, typename V>
inline void print_map(std::ostream& os, const std::string& title, const std::map<K, V>& m, const std::string& indent = "") {
    os << indent << title << " (" << m.size() << " entries):\n";
    if (m.empty()) {
        os << indent << "  <empty>\n";
        return;
    }
    for (const auto& [key, value] : m) {
        os << indent << "  - " << key << ": ";
        if constexpr (std::is_same_v<V, std::vector<uint8_t>>) {
            os << "[ ";
            for (const auto& byte : value) {
                os << "0x" << std::hex << static_cast<int>(byte) << std::dec << " ";
            }
            os << "]";
        } else {
            os << value;
        }
        os << "\n";
    }
}

/**
 * @brief Helper to print a vector with a given title. Marked inline to prevent linker errors.
 */
template<typename T>
inline void print_vector(std::ostream& os, const std::string& title, const std::vector<T>& v, const std::string& indent = "") {
    os << indent << title << " (" << v.size() << " items):\n";
    if (v.empty()) {
        os << indent << "  <empty>\n";
        return;
    }
    for (const auto& item : v) {
        os << indent << "  - " << item << "\n";
    }
}

/**
 * @brief Helper to print an optional value. Marked inline to prevent linker errors.
 */
template<typename T>
inline void print_optional(std::ostream& os, const std::string& name, const std::optional<T>& opt, const std::string& indent = "") {
    os << indent << name << ": ";
    if (opt) {
        os << *opt;
    } else {
        os << "<not set>";
    }
    os << "\n";
}

} // namespace detail

// ======================================================================================
// MARK: - Stream Operators
// ======================================================================================

/**
 * @brief Overloads the << operator for L4maConfig. Marked inline to prevent linker errors.
 */
inline std::ostream& operator<<(std::ostream& os, const L4maConfig& config) {
    std::string indent = "    ";
    os << "L4maConfig:\n"
       << indent << "Type: " << config.type << "\n"
       << indent << "Num Layers: " << config.num_layers << "\n"
       << indent << "Num Query Heads: " << config.num_query_heads << "\n"
       << indent << "Num Key/Value Heads: " << config.num_key_value_heads << "\n"
       << indent << "Head Size: " << config.head_size << "\n"
       << indent << "Hidden Size: " << config.hidden_size << "\n"
       << indent << "Intermediate Size: " << config.intermediate_size << "\n"
       << indent << "Vocab Size: " << config.vocab_size << "\n"
       << indent << "Use QKV Bias: " << (config.use_qkv_bias ? "true" : "false") << "\n"
       << indent << "RMS Norm Epsilon: " << config.rms_norm_eps << "\n"
       << indent << "RoPE Factor: " << config.rope_factor << "\n"
       << indent << "RoPE High Frequency Factor: " << config.rope_high_frequency_factor << "\n"
       << indent << "RoPE Low Frequency Factor: " << config.rope_low_frequency_factor << "\n"
       << indent << "RoPE Theta: " << config.rope_theta;
    return os;
}

/**
 * @brief Overloads the << operator for TokenizerInfo. Marked inline to prevent linker errors.
 */
inline std::ostream& operator<<(std::ostream& os, const TokenizerInfo& info) {
    std::string indent = "    ";
    os << "TokenizerInfo:\n"
       << indent << "Type: " << info.type << "\n"
       << indent << "Split Regex: '" << info.split_regex << "'\n";
    //detail::print_map(os, "Merge Table", info.merge_table, indent);
    //detail::print_map(os, "Special Tokens", info.special_tokens, indent);
    return os;
}

/**
 * @brief Overloads the << operator for AppConfig. Marked inline to prevent linker errors.
 */
inline std::ostream& operator<<(std::ostream& os, const AppConfig& config) {
    std::string indent = ""; // AppConfig is usually top-level
    os << "AppConfig:\n"
       << indent << "  Network:\n"
       << indent << "    Host: " << config.host << "\n"
       << indent << "    Port: " << config.port << "\n"
       << indent << "    Controller Host: " << config.controller_host << "\n"
       << indent << "    Controller Port: " << config.controller_port << "\n";
    detail::print_optional(os, "Auth Token", config.auth_token, indent + "    ");

    os << indent << "  Model & Cache:\n"
       << indent << "    Model Name: " << config.model_name << "\n"
       << indent << "    Cache Dir: " << config.cache_dir.string() << "\n"

       << indent << "  Engine Parameters:\n"
       << indent << "    KV Page Size: " << config.kv_page_size << "\n"
       << indent << "    Dist Size: " << config.dist_size << "\n"
       << indent << "    Max Num KV Pages: " << config.max_num_kv_pages << "\n"
       << indent << "    Max Num Embeds: " << config.max_num_embeds << "\n"

       << indent << "  Hardware:\n"
       << indent << "    Device: " << config.device << "\n"
       << indent << "    DType: " << config.dtype;
    return os;
}

/**
 * @brief Overloads the << operator for ModelMetadata. Marked inline to prevent linker errors.
 */
inline std::ostream& operator<<(std::ostream& os, const ModelMetadata& metadata) {
    std::string indent = "  ";
    os << "=================================================\n"
       << "Model Metadata\n"
       << "=================================================\n"
       << "Name: " << metadata.name << "\n"
       << "Description: " << metadata.description << "\n"
       << "Template Type: " << metadata.template_type << "\n"
       << "Template Content: \"" << metadata.template_content << "\"\n\n";

    detail::print_vector(os, "Parameters", metadata.parameters, "");
    os << "\n";

    // Print nested structs
    os << metadata.architecture << "\n\n";
    os << metadata.tokenizer;

    os << "=================================================\n";

    return os;
}