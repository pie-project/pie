#include "grouped_gemm.cuh"

#include "gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// First-pass grouped GEMM: loop the experts, reuse the dense bf16 GEMM
// (gemm_act_x_wt_bf16) on each expert's contiguous row sub-block. Pointer math
// below is the only new part vs. the dense path:
//
//   x_e = x + offset_e * K  (bf16 elems)  — this expert's M_e rows of [.,K]
//   w_e = w + e * N * K      (bf16 elems)  — this expert's [N,K] weight slab
//   y_e = y + offset_e * N  (bf16 elems)  — this expert's M_e rows of [.,N]
//
// Each sub-block GEMM is byte-for-byte the single-GEMM call: the rows are
// contiguous, so x_e/y_e have the natural leading dims (K and N) and need no
// strided sub-matrix handling. beta=0 (overwrite). expert_offsets is a HOST
// array (see grouped_gemm.cuh) so we can read counts without a D2H copy.
//
// FOLLOW-UP (fused / batched, perf path): replace this serial expert loop with
// a single grouped launch so small/imbalanced groups don't serialize and pay
// per-call launch overhead. Two options:
//   (1) cublasGemmGroupedBatchedEx (cuBLAS 12.5+): one call with per-group
//       m/n/k arrays and A/B/C device pointer arrays (the dropped
//       build_moe_ptrs_* builders in moe_dispatch.cuh produced exactly these
//       pointer arrays). N,K are shared, only M_e varies, so group_count=E with
//       per-group M_e. Lets cuBLAS schedule all groups concurrently.
//   (2) A custom tiled CUTLASS grouped-GEMM kernel keyed on a per-tile
//       expert-id table (the SGLang/vLLM "grouped_gemm" / "moe_align" tile map),
//       which also fuses the dequant + activation and avoids the empty-group
//       launch entirely. This is the eventual home for the WMMA per-expert GEMM
//       that moe_dispatch.cuh notes was dropped.

namespace pie_cuda_device::ops {

cudaError_t grouped_gemm_bf16(cublasHandle_t cublas, cudaStream_t stream,
                              const void* x, const void* w,
                              const int32_t* expert_offsets, void* y,
                              int total_rows, int E, int N, int K) {
    (void)total_rows;  // implied by expert_offsets[E]; kept for the ABI shape.

    // The dense GEMM uses whatever stream is bound to the handle; bind ours so
    // all per-expert GEMMs (and any caller-side dependencies) order correctly.
    cublasStatus_t cbst = cublasSetStream(cublas, stream);
    if (cbst != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;

    const __nv_bfloat16* x_base = static_cast<const __nv_bfloat16*>(x);
    const __nv_bfloat16* w_base = static_cast<const __nv_bfloat16*>(w);
    __nv_bfloat16*       y_base = static_cast<__nv_bfloat16*>(y);

    for (int e = 0; e < E; ++e) {
        const int row_begin = expert_offsets[e];
        const int row_end   = expert_offsets[e + 1];
        const int M_e       = row_end - row_begin;
        if (M_e <= 0) continue;  // skip empty groups

        const __nv_bfloat16* x_e =
            x_base + static_cast<std::size_t>(row_begin) * K;
        const __nv_bfloat16* w_e =
            w_base + static_cast<std::size_t>(e) * N * K;
        __nv_bfloat16* y_e = y_base + static_cast<std::size_t>(row_begin) * N;

        cbst = gemm_act_x_wt_bf16(cublas, x_e, w_e, y_e, M_e, N, K, /*beta=*/0.f);
        if (cbst != CUBLAS_STATUS_SUCCESS) return cudaErrorUnknown;
    }

    return cudaSuccess;
}

}  // namespace pie_cuda_device::ops
