#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <toml++/toml.hpp>

#include "kv_cache_format.hpp"

namespace pie_cuda_driver {

struct ModelConfig {
    std::string snapshot_dir;     // local path to weights + config.json
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
    // Runtime quantization mode applied during load-plan materialization.
    // Empty (default) = no quantization. Recognised values:
    //   * "fp8"  — per-channel symmetric FP8_E4M3 for projection weights.
    //   * "int8" — per-channel symmetric INT8 for projection weights.
    //   * "fp4" / "mxfp4" — MXFP4 (E2M1 weight + E8M0 block scale) for the
    //                      target model's expert weights. Used by GLM-5.1 to
    //                      transcode the checkpoint's FP8 routed-expert
    //                      weights to MXFP4 at materialize time, halving
    //                      the per-rank expert footprint.
    // Norms, biases, embeddings, and lm_head stay in their native dtype.
    std::string runtime_quant;
    // GPT-OSS MXFP4 MoE load/runtime policy. "auto" selects native packed
    // MXFP4 expert GEMM on supported Blackwell-class GPUs/builds and uses the
    // routed-dequant fallback on legacy GPUs. Recognised values:
    //   * "routed_dequant" / "packed" — keep MXFP4 resident and dequantize
    //     only routed experts into bounded BF16 runtime scratch.
    //   * "bf16" / "dequant" — eagerly dequantize experts to BF16 at load.
    //   * "native" — require a true MXFP4 MoE GEMM backend.
    std::string mxfp4_moe = "auto";
    // SSD expert streaming (DeepSeek-V4 / GPT-OSS / Mixtral, tp_size=1). When
    // true, routed MoE expert weights are not materialized on the GPU at load
    // time; instead they are paged on demand from the safetensors shards into
    // a bounded LRU expert cache at forward time.
    bool stream_routed_experts = false;
    // Expert stream cache budget in GiB. 0 (default) = auto: half of the
    // free device memory after resident weights, capped at the full routed
    // expert set. Only meaningful when stream_routed_experts is true.
    double expert_cache_gb = 0.0;
    // Optional Gemma-4 native MTP assistant checkpoint. When set on a
    // Gemma-4 target, output_spec_flags requests draft from this assistant.
    std::string mtp_assistant_snapshot_dir;
    int mtp_num_drafts = 3;
    // Deployment opt-in for system speculation (MTP). Emitted to the runtime,
    // which OWNS the decision to drive drafts (the driver stays pure mechanism).
    // Default false: speculation is a latency-regime feature, off unless the
    // operator enables it (matches vLLM/SGLang's explicit-enable convention).
    bool enable_system_speculation = false;
};

struct BatchingConfig {
    double gpu_mem_utilization = 0.90;
    std::string memory_profile = "auto";
    std::uint32_t kv_page_size = 32;
    // Pinned host KV slots for swap-out. 0 = swap disabled.
    std::uint32_t swap_pool_size = 0;
    // KV cache storage format. "auto" preserves the historical bf16 cache.
    std::string kv_cache_dtype = "auto";
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
        c.model.mxfp4_moe     = (*m)["mxfp4_moe"].value_or(c.model.mxfp4_moe);
        c.model.stream_routed_experts =
            (*m)["stream_routed_experts"].value_or(c.model.stream_routed_experts);
        c.model.expert_cache_gb =
            (*m)["expert_cache_gb"].value_or(c.model.expert_cache_gb);
        c.model.mtp_assistant_snapshot_dir =
            (*m)["mtp_assistant_snapshot_dir"].value_or(std::string{});
        c.model.mtp_num_drafts = static_cast<int>(
            (*m)["mtp_num_drafts"].value_or<int64_t>(c.model.mtp_num_drafts));
        c.model.enable_system_speculation =
            (*m)["enable_system_speculation"].value_or(
                c.model.enable_system_speculation);
    }
    if (auto b = tbl["batching"].as_table()) {
        constexpr std::string_view allowed[] = {
            "gpu_mem_utilization",
            "memory_profile",
            "kv_page_size",
            "swap_pool_size",
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
    if (c.model.expert_cache_gb < 0.0) {
        throw std::runtime_error(
            "config: [model].expert_cache_gb must be >= 0");
    }
    if (c.model.stream_routed_experts && c.distributed.tp_size > 1) {
        throw std::runtime_error(
            "config: [model].stream_routed_experts requires tp_size=1");
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

}  // namespace pie_cuda_driver
