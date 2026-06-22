#pragma once

// Llama-like decoder graph — the shared forward pass for every "pre-norm
// transformer block with QKV/o + gated MLP" architecture: Llama 3, Qwen 2,
// Qwen 3, Mistral, plus the MoE variants (Qwen3-MoE, Mixtral). Per-arch
// quirks (Qwen2 QKV bias, Qwen3 per-head QK-norm, Mistral SWA, YaRN, MoE)
// are driven entirely by `ArchSpec` flags — no compile-time arch coupling.
//
// Mirrors driver/portable's `build_qwen3_graph` and driver/cuda's
// `llama_like_forward`, expressed against the metal driver's MLX `ops::`
// seam in the locked token-major layout.
//
// Out of scope (own builders): Gemma (four-norm sandwich, softcaps, GeGLU,
// sqrt-d embed — see gemma.hpp).

#include "ops/tensor.hpp"
#include "model/arch_spec.hpp"
#include "model/config.hpp"
#include "model/model_graph.hpp"
#include "model/weights.hpp"

namespace pie_metal_driver::model {

class LlamaLikeGraph final : public ModelGraph {
public:
    LlamaLikeGraph(ModelConfig cfg, ModelWeights weights);

    Tensor forward(const ForwardBatch& batch, KvCacheView& kv) override;

    const ModelConfig& config() const override { return cfg_; }

private:
    // One decoder layer; returns the updated residual stream.
    Tensor decoder_layer(std::int32_t il,
                          Tensor hidden,
                          const ForwardBatch& batch,
                          KvCacheView& kv);

    // FFN sub-block: dense SwiGLU or, when `spec_.n_experts > 0`, MoE.
    Tensor ffn(const LayerWeights& L, Tensor x);

    ModelConfig  cfg_;
    ModelWeights w_;
    ArchSpec     spec_;
};

}  // namespace pie_metal_driver::model
