#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

// toml++ cannot be parsed by nvcc (its `table::for_each` generic lambdas trip
// the EDG frontend). Config parsing is host-only, so keep it out of device
// translation units; `.cu` files that include this header still get the plain
// config structs. Same convention as the model/loader adapter headers.
#ifndef __CUDACC__
#include <toml++/toml.hpp>
#endif

#include "store/kv_cache_format.hpp"

namespace pie_cuda_driver {

struct ModelConfig {
    std::string snapshot_dir;     // local path to weights + config.json

    // Path to a `pie.model/1` descriptor: HuggingFace's `config.json`
    // normalized once at import, so the driver reads a flat document with
    // every defaulting rule already resolved rather than re-deriving one at
    // every boot. The worker writes it beside this TOML.
    //
    // Empty for a snapshot that predates the artifact format, in which case
    // the driver parses `snapshot_dir/config.json` itself. That fallback is
    // what lets the descriptor land before a forward pass has run against it;
    // removing it is a separate, deliberate change.
    std::string descriptor;

    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
    int mtp_num_drafts = 3;

    // Page routed MoE experts through a bounded VRAM slab instead of keeping
    // all of them resident. Trades throughput for a resident set that is
    // bounded by the slab rather than by the model, which is what lets a large
    // MoE run on a GPU that cannot hold it.
    //
    // Off by default: for a model that fits, this is strictly slower.
    bool stream_routed_experts = false;

    // The slab, in GiB. 0 = derive one at startup from what is left after the
    // resident weights and the KV pool. A slab too small to hold the experts
    // one step touches thrashes, and no setting of this rescues that -- it is
    // a property of the model's top-k and the batch.
    double expert_cache_gb = 0.0;
    double expert_host_cache_gb = 0.0;
};

struct BatchingConfig {
    double gpu_mem_utilization = 0.90;
    std::string memory_profile = "auto";
    std::uint32_t kv_page_size = 32;
    // Pinned host KV slots for swap-out. 0 = swap disabled.
    std::uint32_t swap_pool_size = 0;
    // KV cache storage format. "auto" preserves the historical bf16 cache.
    std::string kv_cache_dtype = "auto";
    // Optional HARD cap on the runtime KV page count. 0 = derive from
    // gpu_mem_utilization (default). >0 clamps `min(derived, total_pages)` so a
    // tiny deterministic pool can be forced (contention/preempt tests + CI),
    // independent of the forward-layout budget floor. Mirrors metal's total_pages.
    std::int64_t total_pages = 0;
};

// Tensor-parallel group geometry. Default {1, 0, ""} = single-GPU; nothing
// in the forward path runs collectives. Embedded TP launches set tp_size,
// tp_rank, and nccl_unique_id_hex per process. `nccl_unique_id_hex`
// also acts as the in-process rendezvous key for the startup barrier
// and per-fire CPU gate (see `context.cpp::tp_startup_cpu_barrier`).
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

inline int parse_cuda_device_id(const std::string& device) {
    const auto colon = device.find(':');
    const std::string id_str =
        colon == std::string::npos ? device : device.substr(colon + 1);
    std::size_t consumed = 0;
    int id = 0;
    try {
        id = std::stoi(id_str, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "invalid CUDA device '" + device + "'; expected cuda:N or N");
    }
    if (consumed != id_str.size() || id < 0) {
        throw std::runtime_error(
            "invalid CUDA device '" + device + "'; expected cuda:N or N");
    }
    return id;
}

#ifndef __CUDACC__
inline Config load_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config not found: " + path.string());
    }

    auto tbl = toml::parse_file(path.string());
    Config c;

    if (auto m = tbl["model"].as_table()) {
        c.model.snapshot_dir  = (*m)["snapshot_dir"].value_or(std::string{});
        c.model.descriptor    = (*m)["descriptor"].value_or(std::string{});
        c.model.device        = (*m)["device"].value_or(c.model.device);
        c.model.dtype         = (*m)["dtype"].value_or(c.model.dtype);
        c.model.mtp_num_drafts = static_cast<int>(
            (*m)["mtp_num_drafts"].value_or<int64_t>(c.model.mtp_num_drafts));
        c.model.stream_routed_experts =
            (*m)["stream_routed_experts"].value_or(
                c.model.stream_routed_experts);
        c.model.expert_cache_gb =
            (*m)["expert_cache_gb"].value_or<double>(
                static_cast<double>(c.model.expert_cache_gb));
        if (c.model.expert_cache_gb < 0.0) {
            throw std::runtime_error(
                "config: [model].expert_cache_gb must not be negative "
                "(0 = derive one at startup)");
        }
        c.model.expert_host_cache_gb =
            (*m)["expert_host_cache_gb"].value_or<double>(
                static_cast<double>(c.model.expert_host_cache_gb));
        if (c.model.expert_host_cache_gb < 0.0) {
            throw std::runtime_error(
                "config: [model].expert_host_cache_gb must not be negative "
                "(0 = no host tier)");
        }
    }
    if (auto b = tbl["batching"].as_table()) {
        constexpr std::string_view allowed[] = {
            "gpu_mem_utilization",
            "memory_profile",
            "kv_page_size",
            "swap_pool_size",
            "kv_cache_dtype",
            "total_pages",
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
        c.batching.gpu_mem_utilization =
            (*b)["gpu_mem_utilization"].value_or<double>(
                static_cast<double>(c.batching.gpu_mem_utilization));
        c.batching.memory_profile =
            (*b)["memory_profile"].value_or(c.batching.memory_profile);
        const auto kv_page_size =
            (*b)["kv_page_size"].value_or<int64_t>(c.batching.kv_page_size);
        if (kv_page_size <= 0 ||
            kv_page_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "config: [batching].kv_page_size must be in [1, u32::MAX]");
        }
        c.batching.kv_page_size =
            static_cast<std::uint32_t>(kv_page_size);
        const auto swap_pool_size =
            (*b)["swap_pool_size"].value_or<int64_t>(c.batching.swap_pool_size);
        if (swap_pool_size < 0 ||
            swap_pool_size > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "config: [batching].swap_pool_size must be in [0, u32::MAX]");
        }
        c.batching.swap_pool_size =
            static_cast<std::uint32_t>(swap_pool_size);
        c.batching.kv_cache_dtype   = (*b)["kv_cache_dtype"].value_or(c.batching.kv_cache_dtype);
        const auto total_pages =
            (*b)["total_pages"].value_or<int64_t>(
                static_cast<std::int64_t>(c.batching.total_pages));
        if (total_pages < 0) {
            throw std::runtime_error(
                "config: [batching].total_pages must be >= 0 (0 = derive from util)");
        }
        c.batching.total_pages = total_pages;
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
    if (c.model.mtp_num_drafts < 0 || c.model.mtp_num_drafts > 32) {
        throw std::runtime_error(
            "config: [model].mtp_num_drafts must be in [0, 32]");
    }
    if (!(c.batching.gpu_mem_utilization > 0.0 &&
          c.batching.gpu_mem_utilization <= 1.0)) {
        throw std::runtime_error(
            "config: [batching].gpu_mem_utilization must be in (0.0, 1.0]");
    }
    if (c.batching.memory_profile != "auto" &&
        c.batching.memory_profile != "latency" &&
        c.batching.memory_profile != "balanced" &&
        c.batching.memory_profile != "throughput" &&
        c.batching.memory_profile != "capacity") {
        throw std::runtime_error(
            "config: [batching].memory_profile must be one of auto, "
            "latency, balanced, throughput, capacity");
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
    (void)kv_cache_format_from_string(c.batching.kv_cache_dtype, c.model.dtype);
    return c;
}
#endif  // __CUDACC__

}  // namespace pie_cuda_driver
