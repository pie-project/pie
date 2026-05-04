#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

namespace pie_cuda_driver {

struct ShmemConfig {
    std::string name = "/pie-fwd-0";
    std::size_t num_slots = 8;
    std::size_t req_buf = 4 * 1024 * 1024;
    std::size_t resp_buf = 1 * 1024 * 1024;
    std::uint64_t spin_us = 0;
};

struct ModelConfig {
    std::string hf_repo;
    std::string snapshot_dir;     // local path to weights + config.json
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
    // Runtime quantization mode applied after weight load. Empty (default)
    // = no quantization. Recognised values:
    //   * "fp8"  — per-tensor symmetric FP8_E4M3 on every projection
    //              weight (Q/K/V/O/gate/up/down). Norms, biases,
    //              embeddings, lm_head stay in their native dtype.
    // M3 will add `"int4"` for offline GPTQ/AWQ; M2 may add `"int8"`.
    std::string runtime_quant;
};

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t max_num_kv_pages = 1024;
    std::uint32_t max_batch_tokens = 10240;
    std::uint32_t max_batch_size = 512;
    // Pinned host KV slots for swap-out. 0 = swap disabled.
    std::uint32_t swap_pool_size = 0;
    // Cap for the linear-attention state cache slot count (Qwen3.5/3.6).
    // 0 = "follow max_batch_size" — fine on small/medium models. Bound it
    // explicitly on huge MoE × wide max_batch_size combos to avoid OOM:
    //   per-slot bytes ≈ num_linear_layers
    //                  * (V_h * K_d * V_d * 4   // recurrent_state fp32
    //                     + conv_K * conv_dim * 2)  // conv_state bf16
    // Qwen3.6-35B-A3B at max_batch_size=2048 → ~48 GB; 256 → ~6 GB.
    std::uint32_t linear_attn_max_slots = 0;
};

// Tensor-parallel group geometry. Default {1, 0, ""} = single-GPU; nothing
// in the forward path runs collectives. The wrapper writes these fields when
// it spawns N>1 ranks per group; rank 0 is also the shmem leader.
struct DistributedConfig {
    int tp_size = 1;
    int tp_rank = 0;
    // Hex-encoded ncclUniqueId (256 chars). Empty when tp_size == 1.
    std::string nccl_unique_id_hex;
    // Optional CPU-side rendezvous file prefix. Embedded TP launches use this
    // to keep follower ranks from posting their idle NCCL receive before the
    // leader has finished all startup allocations.
    std::string startup_barrier_path;
};

struct Config {
    ShmemConfig shmem;
    ModelConfig model;
    BatchingConfig batching;
    DistributedConfig distributed;
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
        c.model.hf_repo       = (*m)["hf_repo"].value_or(std::string{});
        c.model.snapshot_dir  = (*m)["snapshot_dir"].value_or(std::string{});
        c.model.device        = (*m)["device"].value_or(c.model.device);
        c.model.dtype         = (*m)["dtype"].value_or(c.model.dtype);
        c.model.runtime_quant = (*m)["runtime_quant"].value_or(std::string{});
    }
    if (auto b = tbl["batching"].as_table()) {
        c.batching.kv_page_size     = (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        c.batching.max_num_kv_pages = (*b)["max_num_kv_pages"].value_or<int64_t>(c.batching.max_num_kv_pages);
        c.batching.max_batch_tokens = (*b)["max_batch_tokens"].value_or<int64_t>(c.batching.max_batch_tokens);
        c.batching.max_batch_size   = (*b)["max_batch_size"].value_or<int64_t>(c.batching.max_batch_size);
        c.batching.swap_pool_size   = (*b)["swap_pool_size"].value_or<int64_t>(c.batching.swap_pool_size);
        c.batching.linear_attn_max_slots = (*b)["linear_attn_max_slots"]
            .value_or<int64_t>(c.batching.linear_attn_max_slots);
    }
    if (auto d = tbl["distributed"].as_table()) {
        c.distributed.tp_size = static_cast<int>(
            (*d)["tp_size"].value_or<int64_t>(c.distributed.tp_size));
        c.distributed.tp_rank = static_cast<int>(
            (*d)["tp_rank"].value_or<int64_t>(c.distributed.tp_rank));
        c.distributed.nccl_unique_id_hex =
            (*d)["nccl_unique_id_hex"].value_or(std::string{});
        c.distributed.startup_barrier_path =
            (*d)["startup_barrier_path"].value_or(std::string{});
    }

    if (c.model.hf_repo.empty()) {
        throw std::runtime_error("config: [model].hf_repo is required");
    }
    if (c.distributed.tp_size < 1) {
        throw std::runtime_error("config: [distributed].tp_size must be >= 1");
    }
    if (c.distributed.tp_rank < 0 || c.distributed.tp_rank >= c.distributed.tp_size) {
        throw std::runtime_error("config: [distributed].tp_rank out of range");
    }
    if (c.distributed.tp_size > 1 && c.distributed.nccl_unique_id_hex.empty()) {
        throw std::runtime_error(
            "config: [distributed].nccl_unique_id_hex required when tp_size > 1");
    }
    return c;
}

}  // namespace pie_cuda_driver
