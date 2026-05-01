#pragma once

// Engine — owns the loaded model. Built once at startup; queried from main
// to populate the READY capability JSON and (later milestones) handed to the
// shmem request handler for forward-pass execution.

#include <memory>
#include <unordered_map>
#include <utility>

#include "config.hpp"
#include "loader/hf_config.hpp"
#include "loader/safetensors.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

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

class Engine {
public:
    /// Load weights + config from disk. Throws on missing files / wrong dtypes.
    static Engine load(const Config& boot_cfg);

    Engine() = default;
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept = default;
    Engine& operator=(Engine&&) noexcept = default;

    const HfConfig& hf_config() const noexcept { return hf_; }
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

private:
    // M1.2.1 stores all weights by their HF name. M1.2.2 will introduce a
    // "model schema" layer that fuses Q/K/V into a single device buffer and
    // drops these raw entries.
    Config boot_;
    HfConfig hf_;
    std::unordered_map<std::string, DeviceTensor> weights_;
};

}  // namespace pie_cuda_driver
