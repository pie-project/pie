#pragma once

// Fused W4A16 GEMM — int4 group-quantized weights kept packed in memory and
// dequantized in-register on a Hopper warp-specialized tensor-core GEMM. This
// replaces the slow "dequant-to-bf16-then-bf16-GEMM" path (the bf16 GEMM lives
// in gemm.cuh) with a single CUTLASS mixed-input kernel.
//
// Computes the transformer linear-layer shape `out = act @ W^T`:
//   out[m,n] = sum_c act[m,c] * W[n,c]
// with the SAME int4 format the load-time dequant kernel uses
// (kernels/dequant_wna16.cu — the correctness oracle):
//
//   act          : [M, K] bf16, row-major.
//   packed_w     : int4 group-quantized weight, logical [N, K] (N = out feats).
//                  Packed `uint4b8`: `packed [N, K/8]` int32. Nibble at column c
//                  = value + 8 (decoded value = nibble - 8); 8 lanes per int32,
//                  lane = c%8, word = c/8.
//   scales_bf16  : group scales [N, K/group_size] bf16.
//   Effective weight  W[n,c] = (nibble(n,c) - 8) * scale[n, c/group_size].
//   out          : [M, N] bf16 = act @ W^T.
//
// Target sm_90. The public entry takes the format above; the launcher repacks
// the weight and scales into the layout CUTLASS's mixed-input collective wants
// (documented in gemm_w4a16.cu).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::ops {

// Returns cudaSuccess on success, or the first failing cudaError_t (repack /
// launch / sync). All scratch is allocated and freed internally.
cudaError_t gemm_w4a16_int4_bf16(const void*    act,        // [M,K] bf16
                                 const int32_t* packed_w,   // [N,K/8] uint4b8 int32
                                 const void*    scales_bf16,// [N,K/group_size] bf16
                                 void*          out,        // [M,N] bf16
                                 int            M,
                                 int            N,
                                 int            K,
                                 int            group_size,
                                 cudaStream_t   stream);

}  // namespace pie_cuda_device::ops
