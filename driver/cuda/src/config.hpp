#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <toml++/toml.hpp>

namespace pie_cuda_driver {

struct ModelConfig {
    std::string snapshot_dir;     // local path to weights + config.json
    std::string device = "cuda:0";
    std::string dtype = "bfloat16";
    // Runtime quantization mode applied during load-plan materialization.
    // Empty (default) = no quantization. Recognised values:
    //   * "fp8"  — per-channel symmetric FP8_E4M3 for projection weights.
    //   * "int8" — per-channel symmetric INT8 for projection weights.
    // Norms, biases, embeddings, and lm_head stay in their native dtype.
    std::string runtime_quant;
    // GPT-OSS MXFP4 MoE load/runtime policy. "auto" selects the best
    // registered backend for this build. Recognised values:
    //   * "auto" / "routed_dequant" / "packed" — keep MXFP4 resident and
    //     dequantize only routed experts into bounded BF16 runtime scratch.
    //   * "bf16" / "dequant" — eagerly dequantize experts to BF16 at load.
    //   * "native" — require a true MXFP4 MoE GEMM backend.
    std::string mxfp4_moe = "auto";
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
        c.model.mxfp4_moe     = (*m)["mxfp4_moe"].value_or(c.model.mxfp4_moe);
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
    }
    if (auto r = tbl["runtime"].as_table()) {
        c.runtime.verbose = (*r)["verbose"].value_or(c.runtime.verbose);
    }

    if (c.model.snapshot_dir.empty()) {
        throw std::runtime_error("config: [model].snapshot_dir is required");
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
