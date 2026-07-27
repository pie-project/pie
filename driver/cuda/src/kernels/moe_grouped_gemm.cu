#include "kernels/moe_grouped_gemm.hpp"

#include <cuda_bf16.h>
#include <mma.h>

namespace pie_cuda_driver::kernels {

namespace {

using namespace nvcuda;

constexpr int kFrag = 16;      // WMMA tile, and the aligned block's row count
constexpr int kWarps = 4;      // one n-fragment each
constexpr int kNTile = kFrag * kWarps;

// C = A @ W^T. A is [M, K] row-major, so it loads as a row_major matrix_a.
// W is [N, K] row-major, and W^T is [K, N]; a [K, N] column-major view of
// W^T is exactly W's own memory with leading dimension K, so the b-fragment
// is col_major with ld = K and needs no staging pass.
//
// That direct load costs coalescing -- a fragment is 16 rows of 32 bytes at
// a stride of 2*K -- which is why this kernel is only used where K is short.
// Staging both operands through shared memory was tried and measured: it
// helps the long-K projection (gate_up 14.6 -> 13.7 ms) but never enough to
// beat cuBLAS there (11.1), and it makes the short-K one worse (down
// 6.2 -> 6.9). See `moe_grouped_gemm_bf16_supported`.
__global__ __launch_bounds__(kWarps * 32) void moe_grouped_gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ weight_base,
    __nv_bfloat16* __restrict__ c,
    const std::int32_t* __restrict__ expert_ids,
    int N,
    int K)
{
    const int b = blockIdx.y;
    const int e = expert_ids[b];
    if (e < 0) return;  // padding block: the whole point of this kernel

    const int n0 = blockIdx.x * kNTile;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n_warp = n0 + warp * kFrag;

    const __nv_bfloat16* a_row = a + static_cast<long long>(b) * kFrag * K;
    const __nv_bfloat16* w = weight_base +
        static_cast<long long>(e) * N * K + static_cast<long long>(n_warp) * K;

    wmma::fragment<wmma::accumulator, kFrag, kFrag, kFrag, float> acc;
    wmma::fill_fragment(acc, 0.f);
    wmma::fragment<wmma::matrix_a, kFrag, kFrag, kFrag,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, kFrag, kFrag, kFrag,
                   __nv_bfloat16, wmma::col_major> b_frag;

    for (int k = 0; k < K; k += kFrag) {
        wmma::load_matrix_sync(a_frag, a_row + k, K);
        wmma::load_matrix_sync(b_frag, w + k, K);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Round to bf16 on the way out; the accumulator stayed fp32, matching
    // cuBLAS's compute type for this call.
    __shared__ float staged[kWarps][kFrag * kFrag];
    wmma::store_matrix_sync(staged[warp], acc, kFrag, wmma::mem_row_major);
    __syncwarp();
    __nv_bfloat16* c_row = c + static_cast<long long>(b) * kFrag * N + n_warp;
    for (int idx = static_cast<int>(threadIdx.x) & 31; idx < kFrag * kFrag;
         idx += 32) {
        const int r = idx / kFrag;
        const int col = idx % kFrag;
        c_row[static_cast<long long>(r) * N + col] =
            __float2bfloat16(staged[warp][idx]);
    }
}

}  // namespace

bool moe_grouped_gemm_bf16_supported(int M, int N, int K) {
    // Only where the early exit outweighs the uncoalesced b-fragment load,
    // which measurement puts at short K. On Qwen3.6-35B-A3B tp2 decode:
    //   down    K=256   7.94 -> 6.24 ms   (taken)
    //   gate_up K=2048  11.08 -> 14.60    (left on cuBLAS)
    return M == kFrag && N > 0 && K > 0 && K <= 512 &&
           (N % kNTile) == 0 && (K % kFrag) == 0;
}

void launch_moe_grouped_gemm_bf16(
    const void* a,
    const void* weight_base,
    void* c,
    const std::int32_t* expert_ids,
    int max_blocks,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
    if (max_blocks <= 0 || !moe_grouped_gemm_bf16_supported(M, N, K)) return;
    const dim3 grid(N / kNTile, max_blocks);
    moe_grouped_gemm_bf16_kernel<<<grid, kWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a),
        static_cast<const __nv_bfloat16*>(weight_base),
        static_cast<__nv_bfloat16*>(c),
        expert_ids, N, K);
}

}  // namespace pie_cuda_driver::kernels
