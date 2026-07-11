#pragma once

// Qwen3.6 (Qwen3-Next "qwen3_5" family) — hybrid Gated-DeltaNet model graph.
//
// Ported from driver/cuda/src/model/qwen3_5_forward.cpp (accuracy anchor).
// The arch interleaves two decoder-layer kinds on a fixed schedule
// (`layer_types`: `full_attention` every `full_attention_interval`, the rest
// `linear_attention`); both carry a dense SwiGLU MLP. Variant-defining pieces:
//
//   • Full-attention layers: standard Qwen3 paged attention with two twists —
//       - `attn_output_gate`: q_proj is 2x wide; per head the layout is
//         [query(d) | gate(d)], and the attention output is multiplied by
//         `sigmoid(gate)` before o_proj.
//       - partial RoPE: only the first `2*floor(0.5*partial_rotary_factor*d)`
//         dims of each head are rotated.
//     Gemma-style (1+w) per-head Q/K RMSNorm; default 1/sqrt(head_dim) scale.
//
//   • Linear-attention layers: Gated DeltaNet. The graph owns the in_proj
//     (qkv / z / a / b) and out_proj linears; beta's `ops::gated_delta_net`
//     owns the conv1d + recurrent gated-delta + RMSNormGated core, gathering
//     and scattering per-request conv/recurrent state through delta's
//     `LinearStateCache`. `lin_layer` is the ordinal among linear layers.
//
// All RMSNorm gains use the Gemma (1+w) convention; embeddings are NOT scaled
// (unlike Gemma). Dense only on the 0.8B target (MoE deferred — guarded off).

#include <cstdint>
#include <optional>

#include "ops/tensor.hpp"
#include "config.hpp"
#include "weights.hpp"
#include "model/arch_spec.hpp"
#include "model/model_graph.hpp"

namespace pie_metal_driver::model {

class Qwen3_5Graph : public ModelGraph {
public:
    Qwen3_5Graph(ModelConfig cfg, ModelWeights weights);

    Tensor forward(const ForwardBatch& batch, KvCacheView& kv) override;
    const ModelConfig& config() const override { return cfg_; }

private:
    // Standard Qwen3 paged-attention block with the output gate + partial RoPE.
    Tensor full_attn_layer(std::int32_t il, Tensor hidden,
                           const ForwardBatch& batch, KvCacheView& kv);
    // Gated-DeltaNet linear-attention block. `lin_ordinal` indexes the layer's
    // conv/recurrent state in delta's LinearStateCache.
    Tensor linear_attn_layer(std::int32_t il, std::int32_t lin_ordinal,
                             Tensor hidden, const ForwardBatch& batch);
    // Shared dense SwiGLU MLP block (pre-norm + residual).
    Tensor mlp_block(std::int32_t il, Tensor hidden);

    ModelConfig  cfg_;
    ModelWeights w_;
    ArchSpec     spec_;
};

}  // namespace pie_metal_driver::model
