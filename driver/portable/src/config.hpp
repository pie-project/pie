#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include <toml++/toml.hpp>

namespace pie_portable_driver {

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

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t total_pages = 1024;
    std::uint32_t max_forward_tokens = 10240;
    std::uint32_t max_forward_requests = 512;
    // Host-side swap pool capacity, in pages. 0 = no swap (M7 disabled).
    // The runtime sees `swap_pool_size = cpu_pages` in capabilities.
    std::uint32_t cpu_pages = 0;
};

struct Config {
    ModelConfig    model;
    BatchingConfig batching;
    RuntimeConfig  runtime;
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
        constexpr std::string_view allowed[] = {
            "kv_page_size",
            "total_pages",
            "max_forward_tokens",
            "max_forward_requests",
            "cpu_pages",
        };
        for (const auto& [key, _] : *b) {
            const auto name = key.str();
            bool ok = false;
            for (const auto candidate : allowed) {
                if (name == candidate) {
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                throw std::runtime_error(
                    "config: unknown [batching] key: " + std::string{name});
            }
        }
        c.batching.kv_page_size =
            (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        c.batching.total_pages =
            (*b)["total_pages"].value_or<int64_t>(c.batching.total_pages);
        c.batching.max_forward_tokens =
            (*b)["max_forward_tokens"].value_or<int64_t>(
                c.batching.max_forward_tokens);
        c.batching.max_forward_requests =
            (*b)["max_forward_requests"].value_or<int64_t>(
                c.batching.max_forward_requests);
        c.batching.cpu_pages =
            (*b)["cpu_pages"].value_or<int64_t>(c.batching.cpu_pages);
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
