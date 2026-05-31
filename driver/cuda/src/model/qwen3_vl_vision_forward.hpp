#pragma once

// Qwen3-VL vision encoder forward (bf16). Reproduces transformers
// `Qwen3VLVisionModel` (model_type "qwen3_vl") + its main `merger` and the 3
// DeepStack mergers. Architecture (from `Qwen/Qwen3-VL-2B-Instruct` config and
// transformers 5.9 `modeling_qwen3_vl.py`):
//   depth 24, hidden 1024, heads 16 (head_dim 64), intermediate 4096,
//   patch 16, temporal_patch 2, spatial_merge 2 (2×2 → 4 patches/token),
//   out_hidden 2048 (= text hidden), num_position_embeddings 2304 (48×48 learned
//   abs pos-embed, bilinear-interpolated to the grid and ADDED), 2D-RoPE θ=10000
//   in attention, hidden_act gelu_pytorch_tanh.
//   Pre-norm ViT: LayerNorm (eps 1e-6, gamma+beta) norm1 (pre-attn) / norm2
//   (pre-mlp); QKV+bias, full bidirectional attn, plain (non-gated) MLP fc2(gelu(fc1)).
//   Main merger: LayerNorm over 1024 → 2×2 group (→4096) → fc1 → GELU → fc2 → 2048.
//   DeepStack mergers (layers {5,11,17}): LayerNorm over 4096 AFTER the 2×2
//   shuffle (use_postshuffle_norm=True) → 2048; added into the text decoder at
//   LLM layers 0/1/2 on image rows.
//
// Mirrors gemma4_vision_forward.hpp: a CUDA-only header (no model/loader
// headers) so the `.cu` never pulls in the toml++-heavy config headers nvcc
// cannot parse. The host call site builds `QwenVisRawWeights` from
// `Qwen3VLVisionWeights` via `DeviceTensor::data()` (see qwen3_vl_vision_adapter).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace pie_cuda_driver::model {

// One linear projection: weight `[out, in]` (row-major) + optional bias `[out]`.
struct QVisLinear {
    const __nv_bfloat16* w = nullptr;
    const __nv_bfloat16* b = nullptr;  // null = no bias
};

// LayerNorm parameters (eps lives on QwenVisRawWeights::ln_eps): gamma + beta.
struct QVisLayerNorm {
    const __nv_bfloat16* g = nullptr;  // weight  [dim]
    const __nv_bfloat16* b = nullptr;  // bias    [dim]
};

// One ViT block (pre-norm). qkv is the fused `[3*hidden, hidden]` projection
// (split q|k|v after the matmul); o is the output projection.
struct QVisBlock {
    QVisLayerNorm norm1;  // pre-attention
    QVisLayerNorm norm2;  // pre-mlp
    QVisLinear qkv;       // [3*hidden, hidden] (+bias [3*hidden])
    QVisLinear o;         // [hidden, hidden]   (+bias [hidden])
    QVisLinear fc1;       // [intermediate, hidden]      (+bias)
    QVisLinear fc2;       // [hidden, intermediate]      (+bias)
};

// A patch merger (main or deepstack). The 2×2 spatial-merge groups 4 consecutive
// patch rows (input is already in spatial-merge order — see note below) into a
// `4*hidden` vector, then fc1 → GELU → fc2 → out_hidden.
//   main     : norm over `hidden` (1024) applied BEFORE the 2×2 group.
//              postshuffle_norm.g == nullptr.
//   deepstack: norm over `4*hidden` (4096) applied AFTER the 2×2 group
//              (use_postshuffle_norm=True). `norm` then holds the 4096-dim LN
//              and `postshuffle_norm` is the same handle (set for clarity);
//              implementations key off `is_postshuffle`.
struct QVisMerger {
    QVisLayerNorm norm;             // [hidden] (main) or [4*hidden] (deepstack)
    QVisLinear fc1;                 // [4*hidden, 4*hidden] (+bias)
    QVisLinear fc2;                 // [out_hidden, 4*hidden] (+bias)
    bool is_postshuffle = false;    // false=main (norm before shuffle), true=deepstack
};

struct QwenVisRawWeights {
    // Patch front-end: Conv3d [hidden, 3, t_patch, patch, patch] flattened to a
    // matmul `[hidden, 3*t_patch*patch*patch]` (= [1024, 1536]) + bias [hidden].
    QVisLinear patch;                       // patch_embed.proj (w + b)
    // Learned abs pos-embed table `[num_pos_embed, hidden]` (= [2304, 1024]).
    const __nv_bfloat16* pos_embed = nullptr;
    std::vector<QVisBlock> blocks;          // depth (24)
    QVisMerger merger;                      // main 2×2 merger → out_hidden
    std::vector<QVisMerger> deepstack;      // 3 deepstack mergers
    std::vector<int> deepstack_layer_idx;   // {5, 11, 17}

    int hidden = 1024;
    int heads = 16;
    int head_dim = 64;                      // hidden / heads
    int intermediate = 4096;
    int patch_size = 16;
    int temporal_patch_size = 2;
    int spatial_merge_size = 2;             // 2×2 → spatial_merge_unit = 4 patches/token
    int in_channels = 3;
    int out_hidden = 2048;                  // = text hidden
    int num_pos_embed = 2304;               // 48×48 table
    int num_grid_per_side = 48;             // int(sqrt(num_pos_embed))
    float ln_eps = 1e-6f;
    float rope_theta = 10000.0f;
};

// Per-forward image inputs for the scatter (host pointers into the request view,
// option-B pixel path). `pixels_h` is the f32 pixel_values of all images
// concatenated (each image is `[n_patch_i, 3*t_patch*patch²]` row-major);
// `pixel_byte_indptr_h[i..i+1]` is image i's byte range; `grids_h[3i..3i+3]` is
// image i's (t,h,w) in patch units; `anchor_rows_h[i]` is the batch row where
// image i's merged tokens start.
struct Qwen3VLVisionInputs {
    const QwenVisRawWeights* weights = nullptr;
    const float* pixels_h = nullptr;
    const std::uint32_t* pixel_byte_indptr_h = nullptr;
    const std::uint32_t* grids_h = nullptr;        // 3 per image (t,h,w)
    const std::uint32_t* anchor_rows_h = nullptr;
    int num_images = 0;
};

// Encode one image's patches → main merged tokens `[n_token, out_hidden]` and the
// `num_deep` DeepStack merged tokens (each `[n_token, out_hidden]`).
//   pixel    : bf16 [n_patch, 3*temporal_patch*patch²]  (n_patch = t*h*w),
//              in spatial-merge patch order (see PARITY note below)
//   grid_t/h/w : patch grid dims (t*h*w == n_patch)
//   out_main : bf16 [n_token, out_hidden]   (written; n_token = n_patch / merge²)
//   out_deep : array of `num_deep` device pointers, each bf16 [n_token, out_hidden]
//
// IMPORTANT — pos-embed side input: the learned abs pos-embed is bilinearly
// interpolated from the [2304,1024] table to (grid_h,grid_w) on the HOST (mirror
// transformers `get_vision_bilinear_indices_and_weights`) and uploaded as
// `w.pos_embed` already shaped `[n_patch, hidden]` in spatial-merge patch order;
// the kernel just ADDs it. (If a caller instead passes the raw [2304,1024]
// table, the interpolation must be done device-side — not implemented in the
// first cut.) See qwen3_vl_vision_parity_ref.py `pos_embed_interp` for the dump.
//
// Likewise the 2D-RoPE (row,col) `position_ids` `[n_patch, 2]` are computed on
// the host (mirror `get_vision_position_ids`) and passed via `rope_pos` (f32,
// device). The patch ordering produced by these two host helpers (the
// spatial-merge `reorder` permutation) is the SAME order `pixel` must be in, so
// the merger's 2×2 group is just 4 consecutive rows.
//
// Per-layer checkpoint hook (parity debugging; mirrors set_mimi_decoder_ckpt).
// When set, `run_qwen3vl_vision` emits the patch embedding ("patch_embed", before
// the pos-embed add — matching the HF module hook), the post-block hidden after
// every layer (name "layer<idx>"), and the final hidden ("last_hidden"); the
// callback receives the device bf16 pointer + numel (the stream is synced first).
typedef void (*Qwen3VLVisionCkptFn)(const char* name, const __nv_bfloat16* dev,
                                     long numel, void* user);
void set_qwen3vl_vision_ckpt(Qwen3VLVisionCkptFn fn, void* user);

// Allocates internal scratch (first-cut; a workspace arena is a follow-up).
void run_qwen3vl_vision(const QwenVisRawWeights& w,
                        const __nv_bfloat16* pixel,
                        const float* rope_pos,        // [n_patch, 2] (row,col), device
                        const __nv_bfloat16* pos_embed_interp,  // [n_patch, hidden], device
                        int grid_t, int grid_h, int grid_w,
                        __nv_bfloat16* out_main,
                        __nv_bfloat16* const* out_deep,
                        int num_deep,
                        cudaStream_t stream = 0);

// Encode each image (vision tower → main merger + deepstack mergers) and:
//   - overwrite the merged-token rows of `hidden` (bf16 `[n_rows, out_hidden]`)
//     at each image's anchor row with the main merged output;
//   - write each image's deepstack outputs into `deepstack_scratch` so the
//     decoder loop can add them at LLM layers 0/1/2 on the image rows.
//     `deepstack_scratch` layout: `num_deep` contiguous blocks, each
//     `[n_rows, out_hidden]` (block d at offset `d * n_rows * out_hidden`);
//     image i's tokens live at rows [anchor_i .. anchor_i + n_token_i) in every
//     block, mirroring the main hidden layout so the decoder's `visual_pos_mask`
//     selects the same rows.
// No-op if `in.weights == nullptr` or `in.num_images == 0`.
void scatter_qwen3vl_vision(const Qwen3VLVisionInputs& in,
                            __nv_bfloat16* hidden,
                            int n_rows,
                            int out_hidden,
                            __nv_bfloat16* deepstack_scratch,
                            int num_deep,
                            cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
