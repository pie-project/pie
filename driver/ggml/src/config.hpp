#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

namespace pie_ggml_driver {

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

struct ModelConfig {
    // Path to a HuggingFace snapshot directory containing config.json and
    // model.safetensors (optionally model.safetensors.index.json for sharded
    // checkpoints). Resolved by the Python wrapper from `hf_repo`, or set
    // directly for local dev. Required.
    std::string hf_path;
    // 0 = CPU only, -1 = offload all layers to GPU, N = first N on GPU.
    // Backend is CPU-only in v1; this knob lands when CUDA/Metal backends
    // are wired up in M11.
    std::int32_t n_gpu_layers = 0;
    // Upper bound on max_position_embeddings the runtime may dispatch.
    // Informational — does not size the KV cache (total_pages does).
    std::uint32_t n_ctx = 4096;
};

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t max_num_kv_pages = 1024;
    std::uint32_t max_batch_tokens = 10240;
    std::uint32_t max_batch_size = 512;
    // Host-side swap pool capacity, in pages. 0 = no swap (M7 disabled).
    // The runtime sees `swap_pool_size = cpu_pages` in capabilities.
    std::uint32_t cpu_pages = 0;
};

struct Config {
    ShmemConfig    shmem;
    ModelConfig    model;
    BatchingConfig batching;
    AuxIpcConfig   aux_ipc;
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
        c.model.n_gpu_layers = (*m)["n_gpu_layers"].value_or<int64_t>(c.model.n_gpu_layers);
        c.model.n_ctx        = (*m)["n_ctx"].value_or<int64_t>(c.model.n_ctx);
    }
    if (auto b = tbl["batching"].as_table()) {
        c.batching.kv_page_size     = (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        c.batching.max_num_kv_pages = (*b)["max_num_kv_pages"].value_or<int64_t>(c.batching.max_num_kv_pages);
        c.batching.max_batch_tokens = (*b)["max_batch_tokens"].value_or<int64_t>(c.batching.max_batch_tokens);
        c.batching.max_batch_size   = (*b)["max_batch_size"].value_or<int64_t>(c.batching.max_batch_size);
        c.batching.cpu_pages        = (*b)["cpu_pages"].value_or<int64_t>(c.batching.cpu_pages);
    }
    if (auto a = tbl["aux_ipc"].as_table()) {
        c.aux_ipc.socket_path = (*a)["socket_path"].value_or(std::string{});
    }

    if (c.model.hf_path.empty()) {
        throw std::runtime_error("config: [model].hf_path is required");
    }
    return c;
}

}  // namespace pie_ggml_driver
