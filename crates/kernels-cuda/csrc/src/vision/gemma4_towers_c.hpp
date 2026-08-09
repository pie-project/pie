#pragma once

// The gemma-4 STANDALONE towers (vision + audio) as flat launchers — the
// encode-ABI pair behind `pie_cuda_encode` (host pixels/log-mel in, host
// bf16 embedding rows out, anchor-segmented). Same tower-granularity
// bridge as `qwen3_vl_tower_c.hpp`; only the weight-struct flattening is
// new here.
//
// Layer tables (HOST arrays of DEVICE pointers, `Ty::Bufs`):
//
// VISION, stride 41 per layer (`VisLayerRaw` in declaration order; each
// clipped linear is 5 slots [w, imin, imax, omin, omax]):
//   [0]  in_ln  [1] post_attn_ln  [2] pre_ff_ln  [3] post_ff_ln
//   [4]  q_norm [5] k_norm
//   [6..11)  q clip     [11..16) k clip    [16..21) v clip
//   [21..26) o clip     [26..31) gate clip [31..36) up clip
//   [36..41) down clip
//
// AUDIO, stride 62 per layer (`AudioLayerRaw` order; FFN = [pre_ln,
// post_ln, fc1 clip×5, fc2 clip×5] = 12):
//   [0..12)  ff1        [12..24) ff2
//   [24] norm_pre_attn  [25] norm_post_attn
//   [26..31) q clip     [31..36) k clip    [36..41) v clip
//   [41..46) post clip
//   [46] relative_k     [47] per_dim_scale
//   [48] lconv_pre_ln   [49] lconv_conv_norm
//   [50..55) lconv_start clip   [55..60) lconv_end clip
//   [60] depthwise_conv [61] norm_out
//
// `output_rows_h` is a HOST bf16 buffer (`std::uint16_t*`, the C++
// spelling) and `output_row_indptr_h` a HOST u32 CSR the encode fills —
// the `PieEncodeDesc` shape, verbatim.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace pie_cuda_driver::kernels::vision {

void gemma4_vision_encode(
    const void* patch_w, const void* pos_table, const void* embed_proj,
    const void* const* layer_w, int depth,
    int hidden, int heads, int intermediate, int pos_table_size,
    int text_hidden, int pool_kernel, float eps, float theta,
    const float* pixels_h, const std::uint32_t* pixel_byte_indptr_h,
    const std::uint32_t* patch_positions_h, const std::uint32_t* anchor_rows_h,
    int num_images,
    std::uint16_t* output_rows_h, std::size_t output_bytes,
    std::uint32_t* output_row_indptr_h,
    cudaStream_t stream);

void gemma4_audio_encode(
    const void* sscp0_conv, const void* sscp0_norm, const void* sscp1_conv,
    const void* sscp1_norm, const void* sscp_input_proj,
    const void* output_proj_w, const void* output_proj_b,
    const void* embed_proj,
    const void* const* layer_w, int depth,
    int hidden, int heads, int conv_kernel, int n_mel, int sscp_ch0,
    int sscp_ch1, int out_proj_dims, int text_hidden, int chunk_size,
    int context_left, int context_right, float logit_cap,
    float residual_weight, float eps,
    const float* features_h, const std::uint32_t* feature_byte_indptr_h,
    const std::uint32_t* anchor_rows_h, int num_clips,
    std::uint16_t* output_rows_h, std::size_t output_bytes,
    std::uint32_t* output_row_indptr_h,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::vision
