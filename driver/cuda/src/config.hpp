#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <toml++/toml.hpp>

namespace pie_cuda_driver {

struct ModelConfig {
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
    double gpu_mem_utilization = 0.90;
    std::string memory_profile = "balanced";
    // Pinned host KV slots for swap-out. 0 = swap disabled.
    std::uint32_t swap_pool_size = 0;
};

// Tensor-parallel group geometry. Default {1, 0, ""} = single-GPU; nothing
// in the forward path runs collectives. Embedded TP launches set tp_size,
// tp_rank, and nccl_unique_id_hex per process. `nccl_unique_id_hex`
// also acts as the in-process rendezvous key for the startup barrier
// and per-fire CPU gate (see `entry.cpp::tp_startup_cpu_barrier`).
struct DistributedConfig {
    int tp_size = 1;
    int tp_rank = 0;
    // Hex-encoded ncclUniqueId (256 chars). Empty when tp_size == 1.
    std::string nccl_unique_id_hex;
};

struct RuntimeConfig {
    bool verbose = false;
};

struct Config {
    ModelConfig model;
    BatchingConfig batching;
    DistributedConfig distributed;
    RuntimeConfig runtime;
};

inline Config load_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config not found: " + path.string());
    }

    auto tbl = toml::parse_file(path.string());
    Config c;

    if (auto m = tbl["model"].as_table()) {
        c.model.snapshot_dir  = (*m)["snapshot_dir"].value_or(std::string{});
        c.model.device        = (*m)["device"].value_or(c.model.device);
        c.model.dtype         = (*m)["dtype"].value_or(c.model.dtype);
        c.model.runtime_quant = (*m)["runtime_quant"].value_or(std::string{});
    }
    if (auto b = tbl["batching"].as_table()) {
        constexpr std::string_view allowed[] = {
            "kv_page_size",
            "gpu_mem_utilization",
            "memory_profile",
            "swap_pool_size",
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
        const auto kv_page_size =
            (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        if (kv_page_size <= 0 ||
            kv_page_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "config: [batching].kv_page_size must be in [1, u32::MAX]");
        }
        c.batching.kv_page_size = static_cast<std::uint32_t>(kv_page_size);
        c.batching.gpu_mem_utilization =
            (*b)["gpu_mem_utilization"].value_or<double>(
                static_cast<double>(c.batching.gpu_mem_utilization));
        c.batching.memory_profile =
            (*b)["memory_profile"].value_or(c.batching.memory_profile);
        const auto swap_pool_size =
            (*b)["swap_pool_size"].value_or<int64_t>(c.batching.swap_pool_size);
        if (swap_pool_size < 0 ||
            swap_pool_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "config: [batching].swap_pool_size must be in [0, u32::MAX]");
        }
        c.batching.swap_pool_size =
            static_cast<std::uint32_t>(swap_pool_size);
    }
    if (auto d = tbl["distributed"].as_table()) {
        c.distributed.tp_size = static_cast<int>(
            (*d)["tp_size"].value_or<int64_t>(c.distributed.tp_size));
        c.distributed.tp_rank = static_cast<int>(
            (*d)["tp_rank"].value_or<int64_t>(c.distributed.tp_rank));
        c.distributed.nccl_unique_id_hex =
            (*d)["nccl_unique_id_hex"].value_or(std::string{});
    }
    if (auto r = tbl["runtime"].as_table()) {
        c.runtime.verbose = (*r)["verbose"].value_or(c.runtime.verbose);
    }

    if (c.model.snapshot_dir.empty()) {
        throw std::runtime_error("config: [model].snapshot_dir is required");
    }
    if (!(c.batching.gpu_mem_utilization > 0.0 &&
          c.batching.gpu_mem_utilization <= 1.0)) {
        throw std::runtime_error(
            "config: [batching].gpu_mem_utilization must be in (0.0, 1.0]");
    }
    if (c.batching.memory_profile != "latency" &&
        c.batching.memory_profile != "balanced" &&
        c.batching.memory_profile != "throughput" &&
        c.batching.memory_profile != "capacity") {
        throw std::runtime_error(
            "config: [batching].memory_profile must be one of latency, "
            "balanced, throughput, capacity");
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
