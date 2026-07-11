#pragma once

// Gemma 2 / 3 decoder graph.
//
// Gemma needs its own builder (it doesn't fit the Llama-like skeleton):
//   * four RMSNorms per layer — the "norm sandwich": input_layernorm +
//     post_attention_layernorm around attention, pre_feedforward_layernorm +
//     post_feedforward_layernorm around the MLP;
//   * RMSNorm uses the (1 + weight) convention;
//   * embeddings scaled by sqrt(hidden_size);
//   * GeGLU (tanh-approx GELU) MLP;
//   * custom attention scale 1/sqrt(query_pre_attn_scalar);
//   * Gemma 2 attention + final logit soft-capping;
//   * per-layer sliding/full attention (Gemma 2 alternating, Gemma 3 every-6),
//     with a separate RoPE base on sliding layers (Gemma 3 rope_local_base_freq);
//   * Gemma 3 per-head Q/K RMSNorm.
//
// Mirrors the cuda driver's Gemma path, expressed against the metal driver's
// MLX `ops::` seam in the locked token-major layout.

#include "ops/tensor.hpp"
#include "model/arch_spec.hpp"
#include "model/config.hpp"
#include "model/model_graph.hpp"
#include "model/weights.hpp"

namespace pie_metal_driver::model {

class GemmaGraph final : public ModelGraph {
public:
    GemmaGraph(ModelConfig cfg, ModelWeights weights);

    Tensor forward(const ForwardBatch& batch, KvCacheView& kv) override;

    const ModelConfig& config() const override { return cfg_; }

private:
    Tensor decoder_layer(std::int32_t il,
                          Tensor hidden,
                          const ForwardBatch& batch,
                          KvCacheView& kv);

    ModelConfig  cfg_;
    ModelWeights w_;
    ArchSpec     spec_;
};

}  // namespace pie_metal_driver::model
