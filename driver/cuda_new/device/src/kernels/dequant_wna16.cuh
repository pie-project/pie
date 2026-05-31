#pragma once

// compressed-tensors WNA16 group-wise symmetric int4 -> bf16 dequantization.
// Launcher declaration; the kernel body lives in dequant_wna16.cu, lifted
// verbatim from driver/cuda/src/kernels/dequant_wna16.cu (the base
// load-time dequant variant). The `_gate_up_decode` and `_down_decode` fused
// MoE-decode variants are lifted later as the forward bodies that need them
// land.
//
// Enables GPTQ / AWQ / compressed-tensors int4 checkpoints (e.g. Kimi K2.6
// expert weights): symmetric group-wise int4 stored as `weight_packed` int32
// rows. The scalar type is vLLM/compressed-tensors `uint4b8`: each 4-bit lane
// stores (value + 8), so the decoded value is (lane - 8). One bf16 scale is
// stored for each `group_size` input columns per output row.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void dequant_wna16_int4b8_to_bf16(
    const std::int32_t* packed,     // [out_dim, in_dim / 8]
    const void*         scale_bf16, // [out_dim, in_dim / group_size]
    void*               out_bf16,   // [out_dim, in_dim]
    int                 out_dim,
    int                 in_dim,
    int                 group_size,
    cudaStream_t        stream);

}  // namespace pie_cuda_device::kernels
