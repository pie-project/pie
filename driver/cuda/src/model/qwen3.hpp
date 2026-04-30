#pragma once

// Qwen3 weight schema. Holds non-owning pointers into the Engine's weight
// pool, grouped by transformer block. Unfused — we keep Q/K/V and gate/up
// as separate matrices for now; QKV fusion is an optimization for later.

#include <cstdint>
#include <vector>

#include "engine.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct Qwen3LayerWeights {
    // RMSNorm weights (1D, [hidden] each).
    const DeviceTensor* attn_norm = nullptr;   // input_layernorm
    const DeviceTensor* mlp_norm  = nullptr;   // post_attention_layernorm

    // Self-attention projections (all bf16).
    const DeviceTensor* q_proj = nullptr;      // [num_q_heads*head_dim, hidden]
    const DeviceTensor* k_proj = nullptr;      // [num_kv_heads*head_dim, hidden]
    const DeviceTensor* v_proj = nullptr;      // [num_kv_heads*head_dim, hidden]
    const DeviceTensor* o_proj = nullptr;      // [hidden, num_q_heads*head_dim]

    // Per-head QK normalization (Qwen3 quirk; weight length = head_dim).
    const DeviceTensor* q_norm = nullptr;
    const DeviceTensor* k_norm = nullptr;

    // MLP.
    const DeviceTensor* gate_proj = nullptr;   // [intermediate, hidden]
    const DeviceTensor* up_proj   = nullptr;   // [intermediate, hidden]
    const DeviceTensor* down_proj = nullptr;   // [hidden, intermediate]
};

struct Qwen3Weights {
    const DeviceTensor* embed       = nullptr;  // [vocab, hidden]
    const DeviceTensor* final_norm  = nullptr;  // [hidden]
    const DeviceTensor* lm_head     = nullptr;  // [vocab, hidden] (may alias embed)
    std::vector<Qwen3LayerWeights> layers;
};

/// Build the schema by name-binding tensors out of the engine. Throws if a
/// required weight is missing; tolerates a missing `lm_head` (falls back to
/// `embed` when `tie_word_embeddings` is set).
Qwen3Weights bind_qwen3(const Engine& engine);

}  // namespace pie_cuda_driver::model
