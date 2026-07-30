#pragma once

// LoadedModel — owns the loaded model. Built once at startup; queried from main
// to populate the READY capability JSON and (later milestones) handed to the
// direct executor for forward-pass execution.

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <config.hpp>
#include "loader/group_stream_cache.hpp"
#include "loader/load_plan.hpp"
#include "model/config.hpp"
#include "model/contract.hpp"
#include "pie_loader/checkpoint_source.hpp"
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
    ///
    static LoadedModel load(const Config& boot_cfg,
                            NcclComm* tp_comm,
                            std::string_view runtime_quant,
                            model::Mxfp4MoeRequest mxfp4_moe,
                            model::Component component);

    LoadedModel() = default;
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&) noexcept = default;
    LoadedModel& operator=(LoadedModel&&) noexcept = default;

    const HfConfig& hf_config() const noexcept { return hf_; }
    const DistributedConfig& distributed() const noexcept { return boot_.distributed; }
    const WeightStore& weight_store() const noexcept { return weights_; }
    /// How MXFP4 experts are executed, as the *contract author* decided.
    ///
    /// Read back rather than re-decided. The caller's request is answered once,
    /// by the family that knows what its own alternatives are, and the plan
    /// then materialized the weights in the layout that answer implies -- so a
    /// second opinion here could only disagree with the bytes on the device.
    /// That is why there is no accessor for the raw request: a bind path that
    /// wants to know reads the weights it was handed.
    model::Mxfp4MoePolicy mxfp4_moe_policy() const noexcept {
        return mxfp4_moe_policy_;
    }
    /// The routed-expert slab, or null when the experts are resident.
    ///
    /// Null is the whole signal, the way `moe_gate_up_bf16` being null is:
    /// a forward pass asks whether there is a cache, not whether a flag was
    /// set, so a model whose contract declared no groups and a model with
    /// streaming turned off are the same model to it.
    GroupStreamCache* group_cache() const noexcept {
        return group_cache_.get();
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

private:
    // Owns runtime-layout tensors produced by LoadPlan execution.
    // Some names are non-owning views into packed backing tensors so older
    // forward paths can keep their unfused fallback pointers.
    Config boot_;
    HfConfig hf_;
    WeightStore weights_;
    model::Mxfp4MoePolicy mxfp4_moe_policy_ =
        model::Mxfp4MoePolicy::EagerBf16;

    // Streaming keeps the load alive past the load. A group is paged in by
    // running its plan, so the plan and the file handles it reads through
    // outlive `load()` instead of being torn down at the end of it -- and they
    // are held indirectly because the cache refers to both and this class
    // moves. Empty unless a contract declared groups and streaming was asked
    // for; then the destruction order is cache, source, plan.
    std::unique_ptr<pie_loader::LoadPlan> stream_plan_;
    std::unique_ptr<pie_loader::CheckpointSource> stream_source_;
    std::unique_ptr<GroupStreamCache> group_cache_;
};

namespace ops { struct RuntimeQuantScratchSpec; }

// Derive the runtime-quant scratch spec by scanning the loaded model's
// quantized weights and recording the widest FP8/INT8 weight shape we'd
// need to dequantize on the fly. `max_tokens` is the row dimension for
// the on-the-fly dequant scratch.
ops::RuntimeQuantScratchSpec runtime_quant_scratch_spec(const LoadedModel& engine,
                                                       std::size_t max_tokens);

}  // namespace pie_cuda_driver
