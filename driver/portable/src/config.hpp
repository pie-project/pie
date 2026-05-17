#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

namespace pie_portable_driver {

inline constexpr std::uint32_t kKvPageSize = 32;

struct RuntimeConfig {
    bool verbose = false;
};

struct ModelConfig {
    // Path to a HuggingFace snapshot directory containing config.json and
    // model.safetensors (optionally model.safetensors.index.json for sharded
    // checkpoints). Resolved by the Python wrapper from `hf_repo`, or set
    // directly for local dev. Required.
    std::string hf_path;
    // Backend selector from `model.driver.device`. `auto` keeps ggml's
    // best-available behavior; `cpu` forces CPU even when GPU backends are
    // compiled in.
    std::string backend = "auto";
};

struct Config {
    ModelConfig   model;
    RuntimeConfig runtime;
};

inline Config load_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config not found: " + path.string());
    }

    auto tbl = toml::parse_file(path.string());
    Config c;

    if (auto m = tbl["model"].as_table()) {
        c.model.hf_path      = (*m)["hf_path"].value_or(std::string{});
        c.model.backend      = (*m)["backend"].value_or(c.model.backend);
    }
    if (auto b = tbl["batching"].as_table()) {
        for (const auto& [key, _] : *b) {
            const auto name = key.str();
            throw std::runtime_error(
                "config: [batching]." + std::string{name} +
                " is not accepted; portable derives capacity at startup");
        }
    }
    if (auto r = tbl["runtime"].as_table()) {
        c.runtime.verbose = (*r)["verbose"].value_or(c.runtime.verbose);
    }

    if (c.model.hf_path.empty()) {
        throw std::runtime_error("config: [model].hf_path is required");
    }
    return c;
}

}  // namespace pie_portable_driver
