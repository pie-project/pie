#pragma once

// Gemma-4 vision encoder forward (bf16). Reproduces transformers
// `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`; verified against the HF
// reference (cosine 0.99994 vs HF-bf16, see MULTIMODAL.md §2.2 and
// driver/cuda/tests/gemma4_vision_full_parity_bf16.cu).
//
// CUDA-only header (no model/loader headers) so the `.cu` doesn't pull in the
// toml++-heavy config headers that nvcc cannot parse. The host call site
// builds `VisRawWeights` from `Gemma4VisionWeights` via `DeviceTensor::data()`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace pie_cuda_driver::model {

// Raw device pointers for one clipped-linear: weight + optional bf16 scalar
// clip ranges (null = no clamp on that side).
struct VisClipRaw {
    const __nv_bfloat16* w = nullptr;
    const __nv_bfloat16* imin = nullptr;
    const __nv_bfloat16* imax = nullptr;
    const __nv_bfloat16* omin = nullptr;
    const __nv_bfloat16* omax = nullptr;
};

struct VisLayerRaw {
    const __nv_bfloat16* in_ln = nullptr;
    const __nv_bfloat16* post_attn_ln = nullptr;
    const __nv_bfloat16* pre_ff_ln = nullptr;
    const __nv_bfloat16* post_ff_ln = nullptr;
    const __nv_bfloat16* q_norm = nullptr;
    const __nv_bfloat16* k_norm = nullptr;
    VisClipRaw q, k, v, o, gate, up, down;
};

struct VisRawWeights {
    const __nv_bfloat16* patch_w = nullptr;       // [hidden, 3*patch^2]
    const __nv_bfloat16* pos_table = nullptr;     // [2, pos_table_size, hidden]
    const __nv_bfloat16* embed_proj = nullptr;    // [text_hidden, hidden]
    std::vector<VisLayerRaw> layers;
    int hidden = 768;
    int heads = 12;
    int intermediate = 3072;
    int pos_table_size = 10240;
    int text_hidden = 2560;
    int pool_kernel = 3;   // 2D avg-pool kernel; soft tokens = n_patch / pool_kernel²
    float eps = 1e-6f;
    float theta = 100.0f;
};

// Per-forward image inputs for the scatter. Host pointers into the request
// view (option-B pixel path). `pixels_h` is the f32 pixel_values of all images
// concatenated; `pixel_byte_indptr_h[i..i+1]` is image i's byte range;
// `patch_positions_h` is 2 per patch; `anchor_rows_h[i]` is the batch row where
// image i's soft tokens live.
struct Gemma4VisionInputs {
    const VisRawWeights* weights = nullptr;
    const float* pixels_h = nullptr;
    const std::uint32_t* pixel_byte_indptr_h = nullptr;
    const std::uint32_t* patch_positions_h = nullptr;
    const std::uint32_t* anchor_rows_h = nullptr;
    int num_images = 0;
};

// Encode each image (vision tower → projector) and overwrite the soft-token
// rows of `hidden` (bf16 `[n_rows, text_hidden]`) at each image's anchor row.
// No-op if `in.weights == nullptr` or `in.num_images == 0`.
void scatter_gemma4_vision(const Gemma4VisionInputs& in,
                           __nv_bfloat16* hidden,
                           int n_rows,
                           int text_hidden,
                           cudaStream_t stream = 0);

// Encode one image's patches → soft-token embeddings in the text hidden space.
//   pixel    : bf16 [n_patch, 3*patch^2]   (raw patch pixels in [0,1])
//   pos      : f32  [n_patch, 2]           (patch (x,y) coords; -1 = padding)
//   grp      : i32  [n_patch]              (precomputed 2D-pool group id per patch)
//   out_proj : bf16 [out_len, text_hidden] (written; out_len = n_patch/pool_k^2)
// Allocates internal scratch (first-cut; a workspace arena is a follow-up).
void run_gemma4_vision(const VisRawWeights& w,
                       const __nv_bfloat16* pixel,
                       const float* pos,
                       const int* grp,
                       int n_patch,
                       int out_len,
                       __nv_bfloat16* out_proj,
                       cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
