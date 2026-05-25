#pragma once

// Kimi K2.x / DeepSeek-V3 style decoder weights.
//
// This architecture differs from the llama-like path in two important ways:
//   * MLA attention stores compressed latent KV (`kv_lora_rank`) plus RoPE
//     key position embeddings, not per-head K/V pages.
//   * Layers after `first_k_dense_replace` use DeepSeek-style routed MoE
//     with compressed-tensors W4A16 expert weights.

#include <cstdint>
#include <memory>
#include <vector>

#include "device_buffer.hpp"
#include "model/loaded_model.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver::model {

struct KimiExpertWeights {
    const DeviceTensor* gate_packed = nullptr;
    const DeviceTensor* gate_scale  = nullptr;
    const DeviceTensor* gate_shape  = nullptr;
    const DeviceTensor* up_packed   = nullptr;
    const DeviceTensor* up_scale    = nullptr;
    const DeviceTensor* up_shape    = nullptr;
    const DeviceTensor* down_packed = nullptr;
    const DeviceTensor* down_scale  = nullptr;
    const DeviceTensor* down_shape  = nullptr;
};

struct KimiLayerWeights {
    const DeviceTensor* attn_norm = nullptr;
    const DeviceTensor* mlp_norm  = nullptr;

    const DeviceTensor* q_a_proj = nullptr;             // [q_lora_rank, H]
    const DeviceTensor* q_a_norm = nullptr;             // [q_lora_rank]
    const DeviceTensor* q_b_proj = nullptr;             // [local_heads*(nope+rope), q_lora_rank]
    const DeviceTensor* kv_a_proj_with_mqa = nullptr;   // [kv_lora_rank+rope, H]
    const DeviceTensor* kv_a_norm = nullptr;            // [kv_lora_rank]
    const DeviceTensor* kv_b_proj = nullptr;            // [local_heads*(nope+v), kv_lora_rank]
    const DeviceTensor* o_proj = nullptr;               // [H, local_heads*v_dim]

    bool is_moe = false;

    // Dense MLP (layer 0 on Kimi K2.6).
    const DeviceTensor* dense_gate_proj = nullptr;
    const DeviceTensor* dense_up_proj   = nullptr;
    const DeviceTensor* dense_down_proj = nullptr;

    // Routed + shared MoE (layers >= first_k_dense_replace).
    const DeviceTensor* router = nullptr;               // [E, H]
    const DeviceTensor* e_score_correction_bias = nullptr; // [E] or null
    std::vector<KimiExpertWeights> experts;             // size E
    DeviceBuffer<const std::int32_t*> expert_gate_packed_ptrs;
    DeviceBuffer<const void*>         expert_gate_scale_ptrs;
    DeviceBuffer<const std::int32_t*> expert_up_packed_ptrs;
    DeviceBuffer<const void*>         expert_up_scale_ptrs;
    DeviceBuffer<const std::int32_t*> expert_down_packed_ptrs;
    DeviceBuffer<const void*>         expert_down_scale_ptrs;
    const DeviceTensor* shared_gate_proj = nullptr;     // [I_shared, H]
    const DeviceTensor* shared_up_proj   = nullptr;     // [I_shared, H]
    const DeviceTensor* shared_down_proj = nullptr;     // [H, I_shared]
};

struct KimiWeights {
    const DeviceTensor* embed      = nullptr;
    const DeviceTensor* lm_head    = nullptr;
    const DeviceTensor* final_norm = nullptr;
    int embed_tp_vocab_offset = 0;
    bool embed_tp_sharded = false;
    int lm_head_tp_vocab_offset = 0;
    bool lm_head_tp_sharded = false;

    std::vector<KimiLayerWeights> layers;
};

KimiWeights bind_kimi(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
