#include "kernels/moe_grouped_gemm.hpp"

#include <cuda_bf16.h>
#include <mma.h>

namespace pie_cuda_driver::kernels {

namespace {

using namespace nvcuda;

constexpr int kFrag = 16;      // WMMA tile, and the aligned block's row count
constexpr int kWarps = 4;      // one n-fragment each
constexpr int kNTile = kFrag * kWarps;
// Above this K cuBLAS's tuned mainloop beats the early exit; see below.
constexpr int kShortK = 512;

// C = A @ W^T. A is [M, K] row-major, so it loads as a row_major matrix_a.
// W is [N, K] row-major, and W^T is [K, N]; a [K, N] column-major view of
// W^T is exactly W's own memory with leading dimension K, so the b-fragment
// needs no staging pass -- it is col_major with ld = K.
//
// That costs coalescing: a fragment is 16 rows of 32 bytes at a stride of
// 2*K, so a quarter of each cache line is used. It is affordable only while
// K is short, which is why `moe_grouped_gemm_bf16_supported` bounds K.
//
// The long-K case was pursued and abandoned. Staging both operands through
// shared memory, then adding a `cp.async` double buffer, then deepening it
// to four stages, walked Qwen3.6's gate_up (K=2048) from 14.60 -> 13.71 ->
// 12.75 -> 11.98 ms and never reached cuBLAS's 10.57. What remains is the
// MMA-to-load ratio: each warp owns one 16-wide n-fragment, so it issues
// one `mma` per two fragment loads. Fixing that means each warp computing
// several n-fragments, which needs a wider tile -- and with only ~70 active
// blocks a wider tile drops the CTA count below what fills the device. The
// early exit alone does not pay for a GEMM this much less tuned than
// cuBLAS's.
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

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n_warp = blockIdx.x * kNTile + warp * kFrag;
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
        c_row[static_cast<long long>(idx / kFrag) * N + (idx % kFrag)] =
            __float2bfloat16(staged[warp][idx]);
    }
}

}  // namespace

bool moe_grouped_gemm_bf16_supported(int M, int N, int K) {
    // Measured on Qwen3.6-35B-A3B tp2 decode against cuBLAS:
    //   down     K=256   7.94 -> 5.91 ms   taken
    //   gate_up  K=2048  11.08 -> 11.98    left on cuBLAS (see above)
    return M == kFrag && N > 0 && K > 0 && K <= kShortK &&
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
        static_cast<__nv_bfloat16*>(c), expert_ids, N, K);
}

}  // namespace pie_cuda_driver::kernels
