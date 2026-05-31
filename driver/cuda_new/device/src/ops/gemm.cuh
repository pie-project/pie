#pragma once

// Minimal bf16 GEMM — the transformer linear-layer shape `y = act @ W^T`.
//   act: [M, K] row-major   W: [N, K] row-major (HF storage)   y: [M, N]
// bf16 in/out, fp32 accumulation. beta=0 overwrites; beta=1 fuses a
// residual add. Mirrors the bf16 cublasGemmEx path in
// driver/cuda/src/ops/gemm.cpp (the cuBLASLt dispatcher, FP8/INT4/quant,
// batched/grouped variants are lifted later as the bodies need them).

#include <cublas_v2.h>

namespace pie_cuda_device::ops {

// Returns the cublas status so the ABI layer can map a failure to PieStatus.
cublasStatus_t gemm_act_x_wt_bf16(cublasHandle_t handle,
                                  const void* act, const void* w, void* y,
                                  int M, int N, int K, float beta);

}  // namespace pie_cuda_device::ops
