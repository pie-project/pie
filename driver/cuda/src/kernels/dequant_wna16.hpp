#pragma once

// compressed-tensors WNA16 int4 -> bf16 dequantization.
//
// Kimi K2.6 expert weights are stored as symmetric group-wise int4 in
// `weight_packed` int32 rows. The scalar type is vLLM/compressed-tensors
// `uint4b8`: each 4-bit lane represents (lane - 8). One scale is stored for
// each 32 input columns per output row.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_dequant_wna16_int4b8_to_bf16(
    const std::int32_t* packed,     // [out_dim, in_dim / 8]
    const void*         scale_bf16, // [out_dim, in_dim / group_size]
    void*               out_bf16,   // [out_dim, in_dim]
    int                 out_dim,
    int                 in_dim,
    int                 group_size,
    cudaStream_t        stream);

void launch_wna16_gate_up_decode_bf16(
    const void*          act_bf16,
    const std::int32_t*  topk_idx,
    const std::int32_t* const* gate_packed,
    const void* const*   gate_scale,
    const std::int32_t* const* up_packed,
    const void* const*   up_scale,
    void*                gate_out_bf16,
    void*                up_out_bf16,
    int                  num_tokens,
    int                  top_k,
    int                  hidden,
    int                  intermediate,
    int                  group_size,
    cudaStream_t         stream);

void launch_wna16_down_decode_bf16(
    const void*          act_bf16,
    const std::int32_t*  topk_idx,
    const std::int32_t* const* down_packed,
    const void* const*   down_scale,
    void*                out_bf16,
    int                  num_tokens,
    int                  top_k,
    int                  hidden,
    int                  intermediate,
    int                  group_size,
    cudaStream_t         stream);

}  // namespace pie_cuda_driver::kernels
