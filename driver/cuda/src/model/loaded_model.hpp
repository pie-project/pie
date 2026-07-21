#pragma once

// LoadedModel — owns the loaded model. Built once at startup; queried from main
// to populate the READY capability JSON and (later milestones) handed to the
// shmem executor for forward-pass execution.

#include <memory>
#include <optional>
#include <utility>

#include "config.hpp"
#include "expert_stream_cache.hpp"
#include "loader/backend_target.hpp"
#include "loader/hf_config.hpp"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

struct LoadedModelCapabilities {
    int total_pages = 0;          // populated when KV cache lands (M1.2.2/3)
    int kv_page_size = 0;
    int swap_pool_size = 0;
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

    /// Number of weights resident on device.
    std::size_t num_loaded_tensors() const noexcept { return weights_.size(); }
    std::uint64_t total_weight_bytes() const noexcept;

    bool has(const std::string& name) const {
        return weights_.find(name) != weights_.end();
    }
    const DeviceTensor& get(const std::string& name) const;
    std::size_t erase_runtime_weight(const std::string& name);

    // Lookup quantization metadata for a weight. Returns std::nullopt if
    // the weight is plain bf16/fp16/fp32 (the common case).
    std::optional<QuantMeta> quant_meta(const std::string& name) const;

    // Per-expert file extents from the compiler's deferred stream plan when
    // `[model].stream_routed_experts` is on (empty otherwise). Feeds the
    // ExpertStreamCache constructed by the engine after load.
    const StreamedExpertTable& streamed_expert_table() const noexcept {
        return streamed_experts_;
    }

private:
    // Owns runtime-layout tensors produced by the Rust storage-program loader.
    // Some names are non-owning views into packed backing tensors so older
    // forward paths can keep their unfused fallback pointers.
    Config boot_;
    HfConfig hf_;
    WeightStore weights_;
    Mxfp4MoeLowering mxfp4_moe_lowering_ = Mxfp4MoeLowering::Bf16Dequant;
    StreamedExpertTable streamed_experts_;
};

namespace ops { struct RuntimeQuantScratchSpec; }

// Derive the runtime-quant scratch spec by scanning the loaded model's
// quantized weights and recording the widest FP8/INT8 weight shape we'd
// need to dequantize on the fly. `max_tokens` is the row dimension for
// the on-the-fly dequant scratch.
ops::RuntimeQuantScratchSpec runtime_quant_scratch_spec(const LoadedModel& engine,
                                                       std::size_t max_tokens);

}  // namespace pie_cuda_driver
