#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

namespace pie_portable_driver {

struct ShmemConfig {
    std::string name = "/pie-fwd-0";
    std::size_t num_slots = 8;
    std::size_t req_buf = 4 * 1024 * 1024;
    std::size_t resp_buf = 1 * 1024 * 1024;
    std::uint64_t spin_us = 0;
};

struct AuxIpcConfig {
    // Unix-socket path for the cold-path command channel between the
    // wrapper and the binary. Empty = no aux server (M7 swap RPCs return
    // an error). Set by the Python wrapper at spawn time.
    std::string socket_path;
};

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
    std::uint32_t max_num_kv_pages = 1024;
    std::uint32_t max_batch_tokens = 10240;
    std::uint32_t max_batch_size = 512;
    // Host-side swap pool capacity, in pages. 0 = no swap (M7 disabled).
    // The runtime sees `swap_pool_size = cpu_pages` in capabilities.
    std::uint32_t cpu_pages = 0;
    // Portable ggml graphs keep native F16 cache tensors. Non-native modes
    // round-trip written rows through the selected qdq format after graph
    // compute so subsequent cache reads include quantization error.
    std::string kv_cache_dtype = "auto";
};

struct Config {
    ShmemConfig    shmem;
    ModelConfig    model;
    BatchingConfig batching;
    AuxIpcConfig   aux_ipc;
    RuntimeConfig  runtime;
};

inline Config load_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config not found: " + path.string());
    }

    auto tbl = toml::parse_file(path.string());
    Config c;

    if (auto s = tbl["shmem"].as_table()) {
        c.shmem.name      = (*s)["name"].value_or(c.shmem.name);
        c.shmem.num_slots = (*s)["num_slots"].value_or<int64_t>(c.shmem.num_slots);
        c.shmem.req_buf   = (*s)["req_buf"].value_or<int64_t>(c.shmem.req_buf);
        c.shmem.resp_buf  = (*s)["resp_buf"].value_or<int64_t>(c.shmem.resp_buf);
        c.shmem.spin_us   = (*s)["spin_us"].value_or<int64_t>(c.shmem.spin_us);
    }
    if (auto m = tbl["model"].as_table()) {
        c.model.hf_path      = (*m)["hf_path"].value_or(std::string{});
        c.model.backend      = (*m)["backend"].value_or(c.model.backend);
    }
    if (auto b = tbl["batching"].as_table()) {
        c.batching.kv_page_size     = (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        c.batching.max_num_kv_pages = (*b)["max_num_kv_pages"].value_or<int64_t>(c.batching.max_num_kv_pages);
        c.batching.max_batch_tokens = (*b)["max_batch_tokens"].value_or<int64_t>(c.batching.max_batch_tokens);
        c.batching.max_batch_size   = (*b)["max_batch_size"].value_or<int64_t>(c.batching.max_batch_size);
        c.batching.cpu_pages        = (*b)["cpu_pages"].value_or<int64_t>(c.batching.cpu_pages);
        c.batching.kv_cache_dtype   = (*b)["kv_cache_dtype"].value_or(c.batching.kv_cache_dtype);
    }
    if (auto a = tbl["aux_ipc"].as_table()) {
        c.aux_ipc.socket_path = (*a)["socket_path"].value_or(std::string{});
    }
    if (auto r = tbl["runtime"].as_table()) {
        c.runtime.verbose = (*r)["verbose"].value_or(c.runtime.verbose);
    }

    if (c.model.hf_path.empty()) {
        throw std::runtime_error("config: [model].hf_path is required");
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

}  // namespace pie_portable_driver
