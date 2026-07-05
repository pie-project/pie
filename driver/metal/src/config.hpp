#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include <toml++/toml.hpp>

namespace pie_metal_driver {

struct RuntimeConfig {
    bool verbose = false;
};

struct ModelConfig {
    // Path to a HuggingFace snapshot directory containing config.json and
    // model.safetensors. Resolved by the worker from the model's snapshot,
    // or set directly for local dev. Optional while the driver runs the
    // stub forward (no weights loaded); required once the loader lands.
    std::string hf_path;
    // Backend selector from `model.driver.device`, e.g. `metal:0`.
    std::string backend = "metal:0";
};

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t total_pages = 1024;
    std::uint32_t max_forward_tokens = 10240;
    std::uint32_t max_forward_requests = 512;
    // Host-side swap pool capacity, in pages. 0 = no swap.
    std::uint32_t cpu_pages = 0;
    std::string kv_cache_dtype = "auto";
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
        c.model.hf_path = (*m)["hf_path"].value_or(std::string{});
        c.model.backend = (*m)["backend"].value_or(c.model.backend);
    }
    if (auto b = tbl["batching"].as_table()) {
        constexpr std::string_view allowed[] = {
            "kv_page_size",
            "total_pages",
            "max_forward_tokens",
            "max_forward_requests",
            "cpu_pages",
            "kv_cache_dtype",
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
        c.batching.kv_cache_dtype =
            (*b)["kv_cache_dtype"].value_or(c.batching.kv_cache_dtype);
    }
    if (auto r = tbl["runtime"].as_table()) {
        c.runtime.verbose = (*r)["verbose"].value_or(c.runtime.verbose);
    }

    const auto& kv = c.batching.kv_cache_dtype;
    if (!(kv == "auto" || kv == "bf16" || kv == "bfloat16" ||
          kv == "fp8_e4m3" || kv == "fp8_e5m2" ||
          kv == "int8_per_token_head" || kv == "fp8_per_token_head" ||
          kv == "fp4_e2m1" || kv == "nvfp4")) {
        throw std::runtime_error(
            "config: invalid [batching].kv_cache_dtype '" + kv +
            "'; expected one of: auto, bf16, bfloat16, fp8_e4m3, fp8_e5m2, "
            "int8_per_token_head, fp8_per_token_head, fp4_e2m1, nvfp4");
    }
    return c;
}

}  // namespace pie_metal_driver
