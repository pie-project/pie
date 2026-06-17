#pragma once
// Qwen3-VL vision tower (ViT) for the portable ggml driver. Mirrors the CUDA
// `qwen3_vl_vision_forward` encoder. Architecture (Qwen/Qwen3-VL-2B-Instruct):
//   depth 24, hidden 1024, heads 16 (head_dim 64), intermediate 4096,
//   patch 16, temporal_patch 2, spatial_merge 2 (2x2 -> 4 patches/token),
//   out_hidden 2048 (= text hidden), learned abs pos-embed [2304,1024]
//   bilinear-interpolated to the grid and ADDED, 2D-RoPE theta=10000,
//   hidden_act gelu_pytorch_tanh, pre-norm blocks with fused QKV(+bias),
//   full bidirectional attention, plain (non-gated) MLP fc2(gelu(fc1)).
//   Main merger: LayerNorm(1024) -> 2x2 group(4096) -> fc1 -> GELU -> fc2(2048).
//   Deepstack mergers (layers {5,11,17}): LayerNorm(4096) AFTER the 2x2 shuffle
//   (use_postshuffle_norm) -> 2048; injected into the text decoder at LLM
//   layers 0/1/2 on image rows.
//
// Preprocessing (patchify, smart-resize, pos-embed bilinear interpolation, the
// (row,col) RoPE coordinates) is done host-side in the Rust runtime; the driver
// receives f32 pixel_values [n_patch, patch_dim] and per-patch (x,y) positions.

#include <array>
#include <vector>

#include <ggml.h>

#include "hf_config.hpp"

namespace pie_portable_driver {

// One linear projection: weight [out, in] (ggml [in, out]) + optional bias.
struct VisLinear {
    ggml_tensor* w = nullptr;
    ggml_tensor* b = nullptr;  // null = no bias
};

// LayerNorm (with bias, unlike the LLM's RMSNorm).
struct VisLayerNorm {
    ggml_tensor* g = nullptr;  // weight [dim]
    ggml_tensor* b = nullptr;  // bias   [dim]
};

// One pre-norm ViT block. qkv is the fused [3*hidden, hidden] projection.
struct VisBlock {
    VisLayerNorm norm1;  // pre-attention
    VisLayerNorm norm2;  // pre-mlp
    VisLinear    qkv;    // [3*hidden, hidden] (+bias)
    VisLinear    o;      // [hidden, hidden]   (+bias)
    VisLinear    fc1;    // [intermediate, hidden]  (+bias)
    VisLinear    fc2;    // [hidden, intermediate]  (+bias)
};

// A patch merger (main or deepstack). The 2x2 spatial-merge groups 4 patch rows
// (already in spatial-merge order) into a 4*hidden vector, then fc1->GELU->fc2.
//   main     : norm over hidden (1024) BEFORE the 2x2 group (is_postshuffle=false)
//   deepstack: norm over 4*hidden (4096) AFTER the 2x2 group (is_postshuffle=true)
struct VisMerger {
    VisLayerNorm norm;
    VisLinear    fc1;             // [4*hidden, 4*hidden] (+bias)
    VisLinear    fc2;             // [out_hidden, 4*hidden] (+bias)
    bool         is_postshuffle = false;
};

// The full vision tower (weights live in the model's ggml context, loaded by
// the storage program like any other tensor).
struct Qwen3VLVisionWeights {
    VisLinear              patch;       // patch_embed.proj  w[hidden, patch_dim] + b[hidden]
    ggml_tensor*           pos_embed = nullptr;  // [hidden, num_pos_embed]
    std::vector<VisBlock>  blocks;      // depth (24)
    VisMerger              merger;      // main 2x2 merger -> out_hidden
    std::vector<VisMerger> deepstack;   // deepstack mergers (3)
    bool                   present = false;
};

// Result of encoding one batch of image patches. `embeddings` are the merged
// soft-token rows to splice into the hidden state at the anchor; `deepstack[k]`
// are injected at LLM layer k on the same rows.
struct VisionEncodeResult {
    ggml_tensor*              embeddings = nullptr;  // [out_hidden, n_tokens]
    std::vector<ggml_tensor*> deepstack;             // each [out_hidden, n_tokens]
};

// Build the Qwen3-VL vision encoder graph over `pixels` and return the merged
// embeddings + deepstack outputs. Inputs are graph tensors:
//   pixels       : [patch_dim, n_patch]  (f32 or bf16; patch_dim = 3*t*p*p)
//   pos_embed_in : [hidden, n_patch]     host-interpolated abs pos-embed rows
//   rope_cos/sin : [head_dim, n_patch]   precomputed 2D-RoPE cos/sin (host-built
//                  from (row,col) to match HF apply_rotary_pos_emb_vision)
// `n_patch` is a multiple of spatial_merge_size^2; n_tokens = n_patch / 4.
VisionEncodeResult build_qwen3vl_vision_graph(ggml_context* ctx,
                                              const Qwen3VLVisionWeights& w,
                                              const Hparams& h,
                                              ggml_tensor* pixels,
                                              ggml_tensor* pos_embed_in,
                                              ggml_tensor* rope_cos,
                                              ggml_tensor* rope_sin,
                                              ggml_tensor* attn_mask,
                                              std::int32_t n_patch);

// ── Host-side side-input helpers (computed on CPU before the graph runs) ──
// All produce data in spatial-merge patch order, matching the runtime's
// patchified pixel rows. Ports of transformers.vision_utils.

// (row, col) position per patch, in spatial-merge order. grid in patch units.
// Returns a flat [n_patch*2] array: {row,col, row,col, ...}.
std::vector<std::int32_t> qwen3vl_vision_positions(std::int32_t grid_t,
                                                   std::int32_t grid_h,
                                                   std::int32_t grid_w,
                                                   std::int32_t merge);

// Bilinear-interpolated abs pos-embed. `table` is the [num_pos, hidden] learned
// table (row-major f32; num_pos = side*side). Returns [n_patch*hidden] laid out
// patch-major (ggml tensor [hidden, n_patch]).
std::vector<float> qwen3vl_pos_embed_interp(const float* table, std::int32_t side,
                                            std::int32_t hidden,
                                            const std::int32_t* pos /*[n_patch*2]*/,
                                            std::int32_t n_patch,
                                            std::int32_t grid_h, std::int32_t grid_w);

// 2D-RoPE cos/sin from per-patch (row,col). Each output is [n_patch*head_dim]
// patch-major (ggml tensor [head_dim, n_patch]).
void qwen3vl_rope_cos_sin(const std::int32_t* pos /*[n_patch*2]*/,
                          std::int32_t n_patch, std::int32_t head_dim, float theta,
                          std::vector<float>& cos_out, std::vector<float>& sin_out);

}  // namespace pie_portable_driver
