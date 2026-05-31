#pragma once

// Grouped per-expert bf16 GEMM — the sparse-MoE GEMM that runs AFTER the
// dispatch scatter (kernels/moe_dispatch.cuh) and BEFORE the combine. Tokens
// arrive already permuted so each expert owns a contiguous block of rows; this
// op applies that expert's weight to its block:
//
//   y[r, :] = x[r, :] @ W_{e(r)}^T     for every row r,
//
// where e(r) is the expert that owns row r. It is the multi-weight
// generalization of the single linear-layer GEMM in gemm.cuh
// (`gemm_act_x_wt_bf16`): same row-major bf16 storage, same fp32 accumulation,
// same `act @ W^T` orientation — just one weight matrix per expert.
//
// FIRST PASS (correctness over peak perf): loop the experts and issue one
// `cublasGemmEx` per non-empty group on that group's row sub-block
// (M_e = expert_offsets[e+1] - expert_offsets[e]). Empty groups (M_e == 0) are
// skipped. This is simple and exactly matches the dense GEMM's numerics; the
// follow-up is documented in grouped_gemm.cu.

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::ops {

// Per-expert grouped GEMM, all weights share dims N (out feats) and K (in feats).
//
//   x              : [total_rows, K] bf16, row-major, GROUPED BY EXPERT —
//                    expert e owns rows [expert_offsets[e], expert_offsets[e+1]).
//   w              : [E, N, K] bf16, row-major, contiguous per expert
//                    (each expert's slab is the [N, K] HF weight gemm.cuh wants).
//   expert_offsets : [E+1] int32 prefix sum. **HOST pointer** — the per-group
//                    row counts must be on the host to size/launch each cuBLAS
//                    call, and reading device memory on the host is illegal, so
//                    the caller passes a host array (or a host copy of the
//                    device prefix sum). expert_offsets[0] must be 0 and
//                    expert_offsets[E] must equal total_rows.
//   y              : [total_rows, N] bf16, row-major, same row grouping as x.
//
// Returns cudaSuccess on success; on a cuBLAS failure returns
// cudaErrorUnknown (the cuBLAS status is not a cudaError_t, so the ABI layer
// should prefer the device-side gemm_act_x_wt_bf16 status when it needs the
// precise code). x, w, y are device pointers; the op runs on `stream`.
cudaError_t grouped_gemm_bf16(cublasHandle_t cublas, cudaStream_t stream,
                              const void* x, const void* w,
                              const int32_t* expert_offsets, void* y,
                              int total_rows, int E, int N, int K);

}  // namespace pie_cuda_device::ops
