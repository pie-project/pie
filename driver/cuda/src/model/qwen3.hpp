#pragma once

// Llama-style transformer weight schema. Holds non-owning pointers into the
// Engine's weight pool, grouped by transformer block. Unfused — Q/K/V and
// gate/up are kept separate; QKV fusion is an optimization for later.
//
// Same struct shape covers Qwen3, Llama 3, Qwen 2, and Mistral. The Qwen3
// quirk (per-head q_norm / k_norm) is captured by leaving those pointers
// null on architectures that don't have them; the forward pass skips the
// extra RMSNorm in that case.

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
    // Null on Llama 3 / Qwen 2 / Mistral, which don't have these.
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
/// `embed` when `tie_word_embeddings` is set). Reads `cfg.use_qk_norm` to
/// decide whether to require q/k_norm weights or leave them null.
Qwen3Weights bind_llama_like(const Engine& engine);

// Backward-compatible alias for callers still using `bind_qwen3`.
inline Qwen3Weights bind_qwen3(const Engine& engine) { return bind_llama_like(engine); }

}  // namespace pie_cuda_driver::model
