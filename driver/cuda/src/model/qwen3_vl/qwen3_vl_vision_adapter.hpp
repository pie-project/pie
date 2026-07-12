#pragma once

// Host-side bridge from the bound `Qwen3VLVisionWeights` (DeviceTensor handles)
// to the cuda-only `QwenVisRawWeights` (raw bf16 pointers) the encoder kernels
// consume. This header includes the model headers (toml++ etc.), so it is only
// ever included by host `.cpp` files — never by an nvcc-compiled `.cu`.
//
// Mirrors gemma4_vision_adapter.hpp.
//
// EXPECTED upstream type (lives in the SHARED `model/llama_like/qwen3.hpp`,
// which this task must NOT edit — a teammate wires the bind there). The
// adapter is written against this minimal contract; adjust field names to
// match the final struct:
//
//   struct Qwen3VLVisionLayerWeights {           // one ViT block
//       const DeviceTensor* norm1_weight, *norm1_bias;   // LayerNorm (pre-attn)
//       const DeviceTensor* norm2_weight, *norm2_bias;   // LayerNorm (pre-mlp)
//       const DeviceTensor* qkv_weight,  *qkv_bias;      // [3*hidden, hidden]
//       const DeviceTensor* proj_weight, *proj_bias;     // attn out [hidden,hidden]
//       const DeviceTensor* fc1_weight,  *fc1_bias;      // [intermediate, hidden]
//       const DeviceTensor* fc2_weight,  *fc2_bias;      // [hidden, intermediate]
//   };
//   struct Qwen3VLVisionMergerWeights {          // main or deepstack merger
//       const DeviceTensor* norm_weight, *norm_bias;     // LayerNorm
//       const DeviceTensor* fc1_weight,  *fc1_bias;      // [4*hidden, 4*hidden]
//       const DeviceTensor* fc2_weight,  *fc2_bias;      // [out_hidden, 4*hidden]
//       bool use_postshuffle_norm = false;               // true for deepstack
//   };
//   struct Qwen3VLVisionWeights {
//       const DeviceTensor* patch_weight, *patch_bias;   // patch_embed.proj
//       const DeviceTensor* pos_embed;                   // [num_pos_embed, hidden]
//       std::vector<Qwen3VLVisionLayerWeights> layers;   // depth (24)
//       Qwen3VLVisionMergerWeights merger;               // main
//       std::vector<Qwen3VLVisionMergerWeights> deepstack;  // 3
//       std::vector<int> deepstack_layer_idx;            // {5, 11, 17}
//       Qwen3VLVisionConfig config;                      // dims (hidden, heads, ...)
//   };
//
// Bound from the `model.visual.` checkpoint prefix (see the integration
// checklist): patch_embed.proj.{weight,bias}, pos_embed.weight,
// blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2},
// merger.{norm,linear_fc1,linear_fc2},
// deepstack_merger_list.{d}.{norm,linear_fc1,linear_fc2}.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "model/qwen3_vl/qwen3_vl.hpp"                 // Qwen3VLVisionWeights
#include "model/qwen3_vl/qwen3_vl_vision_forward.hpp"  // QwenVisRawWeights, run_qwen3vl_vision

namespace pie_cuda_driver::model {

// Extract raw device pointers + dims from the bound weights.
QwenVisRawWeights to_vis_raw_qwen(const Qwen3VLVisionWeights& w);

}  // namespace pie_cuda_driver::model
