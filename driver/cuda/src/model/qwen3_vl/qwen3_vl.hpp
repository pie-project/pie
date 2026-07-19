#pragma once

// Qwen3-VL (`model_type` "qwen3_vl") — multimodal: a Qwen3 text decoder
// (28L, GQA 16q/8kv, head_dim 128, per-head q/k norm, M-RoPE) plus a ViT
// vision tower (`model.visual.*`) with a 2×2 spatial-merge merger into the
// text hidden space and 3 DeepStack mergers injected into early decoder
// layers. The text tower binds under the `model.language_model.` prefix
// (HF stores the LLM there alongside the vision tower); the vision tower
// binds under `model.visual.`.
//
// Mirrors gemma4.hpp's vision-weight structs + bind contract. The vision
// weight structs match the documented contract in
// qwen3_vl_vision_adapter.hpp (which converts them to the cuda-only
// QwenVisRawWeights the encoder kernels consume).

#include <cstdint>
#include <vector>

#include "model/config.hpp"
#include "model/loaded_model.hpp"
#include "model/llama_like/qwen3.hpp"  // Qwen3Weights (text tower schema)

namespace pie_cuda_driver::model {

// One ViT block (pre-norm LayerNorm gamma+beta, fused QKV+bias, plain MLP).
struct Qwen3VLVisionLayerWeights {
    const DeviceTensor* norm1_weight = nullptr;  // pre-attn LayerNorm gamma
    const DeviceTensor* norm1_bias   = nullptr;  //              beta
    const DeviceTensor* norm2_weight = nullptr;  // pre-mlp  LayerNorm gamma
    const DeviceTensor* norm2_bias   = nullptr;  //              beta
    const DeviceTensor* qkv_weight   = nullptr;  // [3*hidden, hidden]
    const DeviceTensor* qkv_bias     = nullptr;  // [3*hidden]
    const DeviceTensor* proj_weight  = nullptr;  // attn out [hidden, hidden]
    const DeviceTensor* proj_bias    = nullptr;  // [hidden]
    const DeviceTensor* fc1_weight   = nullptr;  // [intermediate, hidden]
    const DeviceTensor* fc1_bias     = nullptr;  // [intermediate]
    const DeviceTensor* fc2_weight   = nullptr;  // [hidden, intermediate]
    const DeviceTensor* fc2_bias     = nullptr;  // [hidden]
};

// A patch merger (main or deepstack): LayerNorm → 2×2 group → fc1 → GELU → fc2.
//   main     : norm over `hidden` BEFORE the 2×2 group (use_postshuffle_norm=false)
//   deepstack: norm over `4*hidden` AFTER the 2×2 group (use_postshuffle_norm=true)
struct Qwen3VLVisionMergerWeights {
    const DeviceTensor* norm_weight = nullptr;   // LayerNorm gamma
    const DeviceTensor* norm_bias   = nullptr;   // LayerNorm beta
    const DeviceTensor* fc1_weight  = nullptr;   // [4*hidden, 4*hidden]
    const DeviceTensor* fc1_bias    = nullptr;   // [4*hidden]
    const DeviceTensor* fc2_weight  = nullptr;   // [out_hidden, 4*hidden]
    const DeviceTensor* fc2_bias    = nullptr;   // [out_hidden]
    bool use_postshuffle_norm = false;           // true for deepstack
};

struct Qwen3VLVisionWeights {
    const DeviceTensor* patch_weight = nullptr;  // patch_embed.proj.weight
    const DeviceTensor* patch_bias   = nullptr;  // patch_embed.proj.bias
    const DeviceTensor* pos_embed    = nullptr;  // [num_pos_embed, hidden]
    std::vector<Qwen3VLVisionLayerWeights> layers;        // depth (24)
    Qwen3VLVisionMergerWeights merger;                    // main merger
    std::vector<Qwen3VLVisionMergerWeights> deepstack;    // 3 deepstack mergers
    std::vector<int> deepstack_layer_idx;                 // {5, 11, 17}
    Qwen3VLVisionConfig config;
};

// Bind the Qwen3-VL text decoder weights (standard Qwen3 schema) under the
// `model.language_model.` prefix. Throws on a missing required tensor.
Qwen3Weights bind_qwen3_vl_text(const LoadedModel& engine);

// Bind the Qwen3-VL vision tower from `model.visual.`. Requires
// `HfConfig.qwen3_vl_vision` populated. Throws otherwise.
Qwen3VLVisionWeights bind_qwen3_vl_vision(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
