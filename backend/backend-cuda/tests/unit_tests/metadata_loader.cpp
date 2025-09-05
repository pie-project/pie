#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <toml++/toml.hpp>
#include "config.hpp"

static std::map<int, std::vector<uint8_t>> read_vocab_merge_rules(const std::filesystem::path& path) {
    std::map<int, std::vector<uint8_t>> merge_rules;
    std::ifstream file(path);
    if (!file.is_open()) return merge_rules; // allow empty in tests
    std::string line; int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos) continue;
        auto pos = line.find(' ');
        if (pos == std::string::npos) continue;
        std::string token_hex = line.substr(0, pos);
        std::string rank_str = line.substr(pos + 1);
        std::vector<uint8_t> bytes(token_hex.begin(), token_hex.end());
        try { merge_rules[std::stoi(rank_str)] = std::move(bytes); } catch (...) {}
    }
    return merge_rules;
}

ModelMetadata load_model_metadata_for_test(const AppConfig& cfg) {
    auto metadata_path = cfg.cache_dir / "models" / (cfg.model_name + ".toml");
    if (!std::filesystem::exists(metadata_path)) {
        throw std::runtime_error("Metadata file missing: " + metadata_path.string());
    }
    toml::table tbl;
    try { tbl = toml::parse_file(metadata_path.string()); } catch (const toml::parse_error& e) {
        throw std::runtime_error(std::string("Failed to parse metadata: ") + e.what());
    }

    auto req = [&](const toml::table& t, const char* key) -> toml::node_view<const toml::node> {
        auto n = t[key];
        if (!n) throw std::runtime_error(std::string("Missing key '") + key + "' in " + metadata_path.string());
        return n;
    };

    ModelMetadata meta{};
    meta.name = req(tbl, "name").value_or("");
    meta.description = req(tbl, "description").value_or("");
    if (auto params = tbl["parameters"].as_array()) {
        for (auto& el : *params) meta.parameters.push_back(el.value_or(""));
    }
    auto arch = req(tbl, "architecture").as_table();
    auto rope = req(*arch, "rope").as_table();
    meta.architecture = {
        .type = (*arch)["type"].value_or(""),
        .num_layers = (*arch)["num_layers"].value_or(0),
        .num_query_heads = (*arch)["num_query_heads"].value_or(0),
        .num_key_value_heads = (*arch)["num_key_value_heads"].value_or(0),
        .head_size = (*arch)["head_size"].value_or(0),
        .hidden_size = (*arch)["hidden_size"].value_or(0),
        .intermediate_size = (*arch)["intermediate_size"].value_or(0),
        .vocab_size = (*arch)["vocab_size"].value_or(0),
        .use_qkv_bias = false,
        .rms_norm_eps = 1e-5f,
        .rope_factor = (*rope)["factor"].value_or(0.0f),
        .rope_high_frequency_factor = (*rope)["high_frequency_factor"].value_or(0.0f),
        .rope_low_frequency_factor = (*rope)["low_frequency_factor"].value_or(0.0f),
        .rope_theta = (*rope)["theta"].value_or(0.0f)
    };
    if (auto tok = tbl["tokenizer"].as_table()) {
        meta.tokenizer.type = (*tok)["type"].value_or("");
        meta.tokenizer.split_regex = (*tok)["split_regex"].value_or("");
        std::string vocab_file = (*tok)["vocabulary_file"].value_or("");
        auto vocab_path = metadata_path.parent_path() / meta.name / vocab_file;
        meta.tokenizer.merge_table = read_vocab_merge_rules(vocab_path);
        if (auto st = (*tok)["special_tokens"].as_table()) {
            for (auto&& [k,v] : *st) {
                if (auto id = v.value<int>()) meta.tokenizer.special_tokens[*id] = std::string(k);
            }
        }
    }
    if (auto templ = tbl["template"].as_table()) {
        meta.template_type = (*templ)["type"].value_or("");
        meta.template_content = (*templ)["content"].value_or("");
    }
    return meta;
}
