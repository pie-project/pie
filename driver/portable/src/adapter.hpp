#pragma once

// LoRA adapter pool (M9). Each `Adapter` owns its own ggml context and
// backend buffer holding per-layer rank-`r` A and B matrices for the
// linear projections it targets. The pool is keyed by the runtime's
// `adapter_id` (Pie's `adapter_ptr`).
//
// v1 restriction: at most one adapter active per `fire_batch`. Mixed
// adapters in a single batch throw at plan time. The runtime can still
// drive per-context adapter selection by sending one batch per
// adapter group.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "hf_config.hpp"

namespace pie_portable_driver {

// Per-layer LoRA weights. A and B may be nullptr for projections this
// adapter doesn't target. Standard PEFT layout:
//   A[i] : [in_dim, rank]   (HF stores as [rank, in_dim] — we reverse)
//   B[i] : [rank, out_dim]  (HF stores as [out_dim, rank] — we reverse)
// Effective delta:
//   y_delta = scale * (B @ (A @ x))
struct AdapterLayerWeights {
    ggml_tensor* q_a = nullptr;
    ggml_tensor* q_b = nullptr;
    ggml_tensor* k_a = nullptr;
    ggml_tensor* k_b = nullptr;
    ggml_tensor* v_a = nullptr;
    ggml_tensor* v_b = nullptr;
    ggml_tensor* o_a = nullptr;
    ggml_tensor* o_b = nullptr;
};

class Adapter {
public:
    // Loads a LoRA from a HF-style safetensors file. Tensor naming
    // convention (PEFT default):
    //   base_model.model.model.layers.{N}.self_attn.{q,k,v,o}_proj.lora_A.weight
    //   base_model.model.model.layers.{N}.self_attn.{q,k,v,o}_proj.lora_B.weight
    // `scale` is the standard LoRA scaling (alpha / rank); the wrapper
    // computes it from the adapter_config.json and passes it in.
    Adapter(ggml_backend_t backend,
            std::int32_t   n_layers,
            std::int32_t   rank,
            float          scale,
            const std::filesystem::path& safetensors_path,
            const Hparams& hparams);
    ~Adapter();

    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;

    float        scale() const noexcept { return scale_; }
    const std::vector<AdapterLayerWeights>& layers() const noexcept { return layers_; }

private:
    float                 scale_;
    ggml_context*         ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    std::vector<AdapterLayerWeights> layers_;
};

class AdapterPool {
public:
    // Insert a freshly-loaded adapter. Replaces any existing one with the
    // same id.
    void insert(std::uint64_t id, std::unique_ptr<Adapter> adapter);

    // Look up an adapter by id. Returns nullptr if not loaded.
    const Adapter* get(std::uint64_t id) const;

private:
    mutable std::mutex mu_;
    std::unordered_map<std::uint64_t, std::unique_ptr<Adapter>> map_;
};

}  // namespace pie_portable_driver
