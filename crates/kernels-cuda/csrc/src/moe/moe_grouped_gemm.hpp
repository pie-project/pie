#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::moe {

// Grouped GEMM over the MoE decode's aligned expert blocks:
//
//   C[b*M .. b*M+M, :] = A[b*M .. b*M+M, :] @ W[expert_ids[b]]^T
//
// for every block b, with `W[e]` of shape [N, K] row-major and A of shape
// [max_blocks * M, K] row-major.
//
// The point of this kernel, and the only reason it exists next to cuBLAS's
// batched GEMM, is the early exit: `expert_ids[b] < 0` marks a padding block
// and its CTAs return immediately. The batch count must be a static
// worst-case for a cuBLAS call under CUDA-graph capture, and that bound
// cannot go below the expert count (each active expert can own a
// partly-filled block), while the routing actually needs about a third of
// it -- measured on Qwen3.6-35B-A3B tp2 decode: 106 blocks used against 318
// launched. Paying for the other two thirds cost 9.2 ms of a 41 ms step.
//
// Requires M == 16 (one WMMA tile) and N, K multiples of 16.
void moe_grouped_gemm_bf16(
    const void* a,                     // [max_blocks * M, K] bf16
    const void* weight_base,           // [num_experts, N, K] bf16
    void* c,                           // [max_blocks * M, N] bf16
    const std::int32_t* expert_ids,    // [max_blocks], <0 marks padding
    int max_blocks,
    int M,
    int N,
    int K,
    cudaStream_t stream);

// True when `moe_grouped_gemm_bf16` supports this shape.
bool moe_grouped_gemm_bf16_supported(int M, int N, int K);

}  // namespace pie_cuda_driver::kernels::moe
