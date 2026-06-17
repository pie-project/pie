#pragma once
// Gemma-4 vision tower (SigLIP-style ViT) for the portable ggml driver. Mirrors
// the CUDA gemma4_vision_forward. Architecture (google/gemma-4-E2B-it,
// vision_config model_type "gemma4_vision"):
//   hidden 768, 16 layers, 12 heads (head_dim 64), intermediate 3072, patch 16,
//   pooling_kernel_size 3, position_embedding_size 10240 (factored x/y table),
//   rms_norm_eps 1e-6, 2D-RoPE theta 100, gelu_pytorch_tanh, use_clipped_linears.
//   Soft tokens per image = ceil(n_patch / pool^2); projected to text hidden 1536.
//
// Distinct from Qwen3-VL:
//   * Clipped linears: clamp(x, imin, imax) -> matmul -> clamp(y, omin, omax),
//     bounds are per-projection SCALARS (loaded as floats).
//   * 4 RMSNorms / layer (input, post_attn, pre_ff, post_ff); per-head q/k/v
//     RMSNorm (v with no weight).
//   * Pixel scale 2*(x-0.5); factored pos table add tb[0,x]+tb[1,y].
//   * Gated MLP: gelu_tanh(gate) * up -> down  (geglu).
//   * 2D avg-POOL (pool_k x pool_k) merger, not a 2x2 token merge.
//   * 2D-RoPE blocked pairing: dims (c, c+16) for c in [0,16) use px (theta^-c/16);
//     dims (32+c, 48+c) use py. (Differs from Qwen3-VL's row/col-halves.)
//
// Host-side (driver) precomputes, like Qwen3-VL:
//   * pos_embed_in [hidden, n_patch] = tb[0, x_n] + tb[1, y_n]  (table lookup add).
//   * rope cos/sin [head_dim, n_patch] for the blocked pairing above.
//   * pooling matrix [n_patch, n_token] (group membership * sqrt(hidden)/pool^2),
//     so pooling is a single matmul (group-mean + the CUDA sqrt(hidden) scale).

#include <vector>

#include <ggml.h>

#include "hf_config.hpp"

namespace pie_portable_driver {

// Clipped ("QK-clip") linear: clamp(x,[imin,imax]) -> W x -> clamp([omin,omax]).
// Bounds are scalars read from the checkpoint at load time. has_* false => no clamp.
struct G4VisClipLinear {
    ggml_tensor* w = nullptr;   // [in, out] ggml
    float imin = 0, imax = 0;   bool has_in  = false;
    float omin = 0, omax = 0;   bool has_out = false;
};

struct G4VisLayer {
    ggml_tensor* in_ln       = nullptr;  // input_layernorm.weight [hidden]
    ggml_tensor* post_attn_ln = nullptr; // post_attention_layernorm.weight
    ggml_tensor* pre_ff_ln   = nullptr;  // pre_feedforward_layernorm.weight
    ggml_tensor* post_ff_ln  = nullptr;  // post_feedforward_layernorm.weight
    G4VisClipLinear q, k, v, o;
    ggml_tensor* q_norm = nullptr;       // [head_dim] per-head RMSNorm
    ggml_tensor* k_norm = nullptr;       // [head_dim]
    G4VisClipLinear gate, up, down;
};

struct Gemma4VisionWeights {
    ggml_tensor*             patch_w   = nullptr;  // patch_embedder.input_proj.weight [patch_dim, hidden]
    ggml_tensor*             pos_table = nullptr;  // position_embedding_table [2, P, hidden]
    std::vector<G4VisLayer>  layers;               // 16
    ggml_tensor*             embed_proj = nullptr; // embed_vision.embedding_projection.weight [hidden, text_hidden]
    bool                     present = false;
};

// Build the Gemma-4 vision encoder graph. Returns the projected soft-token
// embeddings [text_hidden, n_token]. Inputs are graph tensors:
//   pixels        [patch_dim, n_patch]  f32 pixel_values in [0,1]
//   pos_embed_in  [hidden, n_patch]     host table-lookup add (tb[0,x]+tb[1,y])
//   rope_cos/sin  [head_dim, n_patch]   blocked 2D-RoPE
//   pool_matrix   [n_patch, n_token]    group-mean (* sqrt(hidden)/pool^2)
ggml_tensor* build_gemma4_vision_graph(ggml_context* ctx,
                                       const Gemma4VisionWeights& w,
                                       const Hparams& h,
                                       ggml_tensor* pixels,
                                       ggml_tensor* pos_embed_in,
                                       ggml_tensor* rope_cos,
                                       ggml_tensor* rope_sin,
                                       ggml_tensor* pool_matrix,
                                       ggml_tensor* attn_mask,
                                       std::int32_t n_patch,
                                       std::int32_t n_token);

// Host helpers (CPU, before the graph). pos/grid in patch units.
// Factored pos-table lookup add: out[n*hidden + d] = tb[0,x_n,d] + tb[1,y_n,d].
std::vector<float> gemma4_vision_pos_embed(const float* table /*[2,P,hidden]*/,
                                           std::int32_t P, std::int32_t hidden,
                                           const std::int32_t* pos /*[n_patch*2]*/,
                                           std::int32_t n_patch);
// Blocked 2D-RoPE cos/sin [n_patch*head_dim] (px for [0,32), py for [32,64)).
void gemma4_vision_rope_cos_sin(const std::int32_t* pos, std::int32_t n_patch,
                                std::int32_t head_dim, float theta,
                                std::vector<float>& cos_out,
                                std::vector<float>& sin_out);
// Pooling matrix [n_patch*n_token] row-major (matrix[token*n_patch + patch]),
// value sqrt(hidden)/pool^2 when patch is in token's pool_k x pool_k block.
std::vector<float> gemma4_vision_pool_matrix(const std::int32_t* pos,
                                             std::int32_t n_patch,
                                             std::int32_t grid_w, std::int32_t grid_h,
                                             std::int32_t pool_k, std::int32_t hidden,
                                             std::int32_t& n_token_out);

}  // namespace pie_portable_driver
