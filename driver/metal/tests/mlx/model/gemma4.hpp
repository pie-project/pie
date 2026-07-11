#pragma once

// Gemma-4 (E2B / E4B dense) decoder graph.
//
// Gemma-4 builds on the Gemma-3 norm-sandwich + per-head Q/K-norm skeleton but
// adds several arch-defining pieces that don't fit GemmaGraph, ported here from
// driver/cuda/src/model/gemma4.cpp (the accuracy anchor):
//
//   * PLAIN RMSNorm — Gemma-4 dropped the Gemma-2/3 `(1 + w)` convention and
//     applies the weight directly (`w * x_hat`).
//   * Per-Layer-Embeddings (PLE): a per-layer token table + main-embedding
//     projection are combined up-front into a `[N, L, ple_dim]` signal, and a
//     GeGLU-gated slice is injected back into the residual after each layer's
//     MLP.
//   * Cross-layer KV-share: the last `num_kv_shared_layers` layers don't
//     compute their own K/V — they re-attend through the most-recent earlier
//     layer of the same attention type (sliding vs full).
//   * Per-layer head_dim: sliding layers run at `head_dim`, full-attention
//     layers at the wider `global_head_dim`. The KV cache is allocated
//     per-layer (delta's PagedKvCache per-layer ctor).
//   * Weightless V-norm (unit-gain RMSNorm on V before the KV write).
//   * sm_scale = 1.0 (Q/K-norm absorbs the usual 1/sqrt(d)).
//   * Per-attention-type RoPE base (sliding = rope_local_base_freq, full =
//     rope_theta) + optional partial rotary.
//   * Optional per-layer learnable output scalar, and a final logit soft-cap.
//
// Expressed against the metal driver's MLX `ops::` seam in the locked
// token-major layout (head_dim last).

#include <optional>

#include "ops/tensor.hpp"
#include "model/arch_spec.hpp"
#include "model/config.hpp"
#include "model/model_graph.hpp"
#include "model/weights.hpp"

namespace pie_metal_driver::model {

class Gemma4Graph final : public ModelGraph {
public:
    Gemma4Graph(ModelConfig cfg, ModelWeights weights);

    Tensor forward(const ForwardBatch& batch, KvCacheView& kv) override;

    const ModelConfig& config() const override { return cfg_; }

private:
    // One decoder layer. `ple_signal`, when present, is this layer's
    // `[n_tokens, ple_dim]` per-layer-embedding slice injected after the MLP.
    Tensor decoder_layer(std::int32_t il,
                         Tensor hidden,
                         const std::optional<Tensor>& ple_signal,
                         const ForwardBatch& batch,
                         KvCacheView& kv);

    ModelConfig  cfg_;
    ModelWeights w_;
    ArchSpec     spec_;
};

}  // namespace pie_metal_driver::model
