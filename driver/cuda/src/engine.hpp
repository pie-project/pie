#pragma once

// Engine — owns the loaded model. Built once at startup; queried from main
// to populate the READY capability JSON and (later milestones) handed to the
// shmem request handler for forward-pass execution.

#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

#include "config.hpp"
#include "loader/hf_config.hpp"
#include "loader/safetensors.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

// Per-weight metadata for quantized tensors. Lives on a side-map keyed by
// the weight's name; absent entries mean "use the raw bf16/fp16/fp32 path".
//
// The pointers reference DeviceTensors registered separately under their
// own names in Engine::weights_ — Engine::set_quant_meta does not own them
// and validates that they were inserted first.
//
// `kind` describes the scale layout:
//   * PerTensor  — `scale` shape: scalar [].
//   * PerChannel — `scale` shape: [N], one per output channel.
//                  `channel_axis` selects which weight axis N corresponds
//                  to (axis=0 for HF row-major `[N, K]` projections).
//   * PerGroup   — `scale` shape: [groups, N] for GPTQ-style row groups
//                  along the K axis. `group_size` is the K-axis stride
//                  per group (e.g. 128).
//
// `zero_point` is null for symmetric quantization; populated for asymmetric
// (e.g. AWQ int4 with zero_point=true).
struct QuantMeta {
    enum class Kind { PerTensor, PerChannel, PerGroup };
    Kind kind = Kind::PerTensor;
    const DeviceTensor* scale = nullptr;
    const DeviceTensor* zero_point = nullptr;
    int group_size = 0;
    int channel_axis = 0;
};

struct EngineCapabilities {
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

class Engine {
public:
    /// Load weights + config from disk. Throws on missing files / wrong dtypes.
    /// Pass `tp_comm` when `boot_cfg.distributed.tp_size > 1` to enable
    /// TP-aware runtime quantization (cross-rank absmax all-reduce for
    /// row-parallel weights). For single-GPU (tp_size=1) this can be null.
    static Engine load(const Config& boot_cfg, NcclComm* tp_comm = nullptr);

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept = default;
    Engine& operator=(Engine&&) noexcept = default;

    const HfConfig& hf_config() const noexcept { return hf_; }
    const DistributedConfig& distributed() const noexcept { return boot_.distributed; }
    EngineCapabilities capabilities() const;

    /// Number of weights physically resident on device.
    std::size_t num_loaded_tensors() const noexcept { return weights_.size(); }
    std::uint64_t total_weight_bytes() const noexcept;

    bool has(const std::string& name) const {
        return weights_.find(name) != weights_.end();
    }
    const DeviceTensor& get(const std::string& name) const;

    // Register a tensor (typically a non-owning view from
    // `DeviceTensor::view(...)`) under `name`. Used by per-arch bind
    // functions that synthesise virtual q/k/v slots from a fused
    // `qkv_proj.weight` (Phi-3) without copying the underlying data.
    // Throws if `name` is already registered.
    void insert(std::string name, DeviceTensor tensor);

    // Attach quantization metadata for an already-inserted weight. The
    // weight tensor must be in `weights_`; the scale / zero_point tensors
    // referenced by `meta` must also already be inserted (so that name
    // typos in bind code surface here rather than at GEMM dispatch time).
    void set_quant_meta(const std::string& name, QuantMeta meta);

    // Lookup quantization metadata for a weight. Returns std::nullopt if
    // the weight is plain bf16/fp16/fp32 (the common case).
    std::optional<QuantMeta> quant_meta(const std::string& name) const;

private:
    // M1.2.1 stores all weights by their HF name. M1.2.2 will introduce a
    // "model schema" layer that fuses Q/K/V into a single device buffer and
    // drops these raw entries.
    Config boot_;
    HfConfig hf_;
    std::unordered_map<std::string, DeviceTensor> weights_;
    std::unordered_map<std::string, QuantMeta> quant_meta_;
};

}  // namespace pie_cuda_driver
