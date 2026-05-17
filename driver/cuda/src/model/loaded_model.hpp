#pragma once

// LoadedModel — owns the loaded model. Built once at startup; queried from main
// to populate the READY capability JSON and (later milestones) handed to the
// shmem executor for forward-pass execution.

#include <memory>
#include <optional>
#include <utility>

#include "config.hpp"
#include "loader/hf_config.hpp"
#include "loader/model_schema.hpp"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

struct LoadedModelCapabilities {
    int total_pages = 0;          // populated when KV cache lands (M1.2.2/3)
    int kv_page_size = 0;
    int swap_pool_size = 0;
    int max_batch_tokens = 0;
    int max_batch_size = 0;
    std::string arch_name;
    int vocab_size = 0;
    int max_model_len = 0;
    std::string activation_dtype;
    std::string snapshot_dir;
};

class NcclComm;  // distributed.hpp

class LoadedModel {
public:
    /// Load weights + config from disk. Throws on missing files / wrong dtypes.
    /// Pass `tp_comm` when `boot_cfg.distributed.tp_size > 1` to enable
    /// TP-aware runtime quantization (cross-rank absmax all-reduce for
    /// row-parallel weights). For single-GPU (tp_size=1) this can be null.
    static LoadedModel load(const Config& boot_cfg, NcclComm* tp_comm = nullptr);

    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&) noexcept = default;
    LoadedModel& operator=(LoadedModel&&) noexcept = default;

    const HfConfig& hf_config() const noexcept { return hf_; }
    const DistributedConfig& distributed() const noexcept { return boot_.distributed; }
    const WeightStore& weight_store() const noexcept { return weights_; }
    Mxfp4MoeLowering mxfp4_moe_lowering() const noexcept {
        return mxfp4_moe_lowering_;
    }
    LoadedModelCapabilities capabilities() const;

    /// Number of weights physically resident on device.
    std::size_t num_loaded_tensors() const noexcept { return weights_.size(); }
    std::uint64_t total_weight_bytes() const noexcept;

    bool has(const std::string& name) const {
        return weights_.find(name) != weights_.end();
    }
    const DeviceTensor& get(const std::string& name) const;

    // Lookup quantization metadata for a weight. Returns std::nullopt if
    // the weight is plain bf16/fp16/fp32 (the common case).
    std::optional<QuantMeta> quant_meta(const std::string& name) const;

private:
    // Owns runtime-layout tensors produced by the schema/load-plan materializer.
    // Some names are non-owning views into packed backing tensors so older
    // forward paths can keep their unfused fallback pointers.
    Config boot_;
    HfConfig hf_;
    WeightStore weights_;
    Mxfp4MoeLowering mxfp4_moe_lowering_ = Mxfp4MoeLowering::Bf16Dequant;
};

}  // namespace pie_cuda_driver
