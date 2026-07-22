#include "kernels/dtype_cast.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

__global__ void cast_fp16_to_bf16_kernel(
    const __half*    __restrict__ src,
    __nv_bfloat16*   __restrict__ dst,
    std::size_t                   n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__half2float(src[i]));
}

__global__ void cast_fp32_to_bf16_kernel(
    const float*     __restrict__ src,
    __nv_bfloat16*   __restrict__ dst,
    std::size_t                   n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(src[i]);
}

__global__ void cast_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ src,
    float*               __restrict__ dst,
    std::size_t                       n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __bfloat162float(src[i]);
}

}  // namespace

void launch_cast_fp16_to_bf16(
    const void* src_fp16, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_fp16_to_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __half*>(src_fp16),
        static_cast<__nv_bfloat16*>(dst_bf16), n);
}

void launch_cast_fp32_to_bf16(
    const void* src_fp32, void* dst_bf16,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_fp32_to_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const float*>(src_fp32),
        static_cast<__nv_bfloat16*>(dst_bf16), n);
}

void launch_cast_bf16_to_fp32(
    const void* src_bf16, void* dst_fp32,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_bf16_to_fp32_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_bf16),
        static_cast<float*>(dst_fp32), n);
}

namespace {

// Marlin scale permutation (per-group case). Each block of 64 scalars
// is reshuffled by the perm `i + 8*j` for (i, j) in [0..8) × [0..8).
// Equivalent to a 8×8 transpose of an 8x8 sub-block. Applied in-place
// via a temp register-shuffle: each warp reads its 64 scalars, threads
// 0..63 write back at the permuted index.
constexpr int MARLIN_GROUP_PERM_LEN = 64;

// One block per row of 64 scalars; 64 threads per block do the perm.
__global__ void marlin_permute_scales_per_group_kernel(
    __nv_bfloat16* __restrict__ s,
    int                         total64_rows)  // total elements / 64
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total64_rows || tid >= 64) return;
    __nv_bfloat16* base = s + static_cast<std::size_t>(row) * 64;
    // Read original, write to permuted slot. Using shared mem to swap.
    __shared__ __nv_bfloat16 buf[64];
    buf[tid] = base[tid];
    __syncthreads();
    // perm[idx] for idx in [0..64): packed as i*8+j for i,j in [0..8)
    //   -> reads from i + 8*j  (the inverse permutation).
    // Equivalent: write tid -> read from `(tid % 8) * 8 + (tid / 8)`
    // i.e. transpose the 8x8 layout.
    const int i = tid / 8;
    const int j = tid % 8;
    const int src_idx = j * 8 + i;
    base[tid] = buf[src_idx];
}

}  // namespace

void launch_marlin_permute_scales_bf16(
    void* bf16_scales,
    int groups, int size_n, int group_size, int size_k,
    cudaStream_t stream)
{
    if (groups == 0 || size_n == 0) return;
    if (size_n % MARLIN_GROUP_PERM_LEN != 0) {
        // Marlin requires N multiple of 64 (tile_n_size). Caller
        // should have validated.
        return;
    }
    const std::size_t total = static_cast<std::size_t>(groups) * size_n;
    if (total % MARLIN_GROUP_PERM_LEN != 0) return;
    const int total64 = static_cast<int>(total / MARLIN_GROUP_PERM_LEN);

    if (group_size > 0 && group_size < size_k) {
        // Per-group case (group_size=128 etc).
        marlin_permute_scales_per_group_kernel<<<total64, 64, 0, stream>>>(
            static_cast<__nv_bfloat16*>(bf16_scales), total64);
    }
    // Per-channel uses a different perm — skip until needed.
}

namespace {

// AWQ qzeros → marlin qzeros. Replicates vLLM's `awq_to_marlin_zero_
// points` (see vllm/.../marlin_utils.py):
//   1. unpack_cols on transposed AWQ qzeros → [N, groups] uint8 (one
//      nibble per element).
//   2. Apply AWQ's undo-interleave perm `[0,4,1,5,2,6,3,7]` over each
//      8-element stride. Equivalent to "AWQ stored values with
//      interleave [0,2,4,6,1,3,5,7]; argsort gives the inverse".
//   3. marlin_zero_points: reshape [N, groups] flat → [-1, 64] and
//      apply scale_perm (`perm[i*8+j] = i + 8*j`), THEN reshape [-1, 8]
//      and apply the 8-wide interleave [0,2,4,6,1,3,5,7]; reshape to
//      [groups, N] and pack 8 nibbles per int32.
//
// The whole pipeline is pure index arithmetic — we read the source
// nibble directly per output position rather than materialising the
// intermediate buffers.
__global__ void awq_qzero_to_marlin_w4_kernel(
    const std::uint32_t* __restrict__ in,    // AWQ [groups, N/8]
    std::uint32_t*       __restrict__ out,   // marlin [groups, N/8]
    int                                groups,
    int                                size_n)
{
    const int g_out = blockIdx.x;
    const int n8_out = blockIdx.y * blockDim.x + threadIdx.x;
    if (g_out >= groups) return;
    const int n8 = size_n / 8;
    if (n8_out >= n8) return;

    // Replicates vLLM's exact AWQ-marlin linear flow:
    // _convert_awq_tensor_layout (qzeros) + marlin_zero_points.
    //
    // For output[g_out, n_out] (flat p = g_out*N + n_out in the post-
    // marlin-perm (G, N) view):
    //   pre = (p/64)*64 + scale_perm[p%64]
    //   pre_g = pre / N, pre_n = pre % N        (LINEAR-decoded coords)
    //   nibble = (AWQ[pre_g, pre_n/8] >> (4 * reverse_order[pre_n%8])) & 0xF
    //
    // where `reverse_order = _REVERSE_AWQ_PACK_ORDER = [0,4,1,5,2,6,3,7]`
    // = bit position in the AWQ-packed int32 holding slot j of the
    // unpacked array.
    constexpr int reverse_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    // marlin_zero_points applies an 8-wide column interleave
    // (`zp.reshape(-1, 8)[:, [0,2,4,6,1,3,5,7]]`) AFTER the 64-wide
    // scale_perm and before packing. Output nibble slot j therefore carries
    // the logical column n8_out*8 + interleave[j], not the linear j.
    constexpr int interleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};

    std::uint32_t v = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int n_out = n8_out * 8 + interleave[j];
        const int p = g_out * size_n + n_out;
        const int p_in_64 = p % 64;
        const int p_in_64_perm = (p_in_64 % 8) * 8 + (p_in_64 / 8);
        const int pre = (p / 64) * 64 + p_in_64_perm;
        const int pre_g = pre / size_n;
        const int pre_n = pre % size_n;
        const std::uint32_t src = in[pre_g * n8 + (pre_n / 8)];
        const int src_bit = 4 * reverse_order[pre_n % 8];
        const std::uint32_t nibble = (src >> src_bit) & 0xFu;
        v |= nibble << (4 * j);
    }
    out[g_out * n8 + n8_out] = v;
}

}  // namespace

void launch_awq_qzero_to_marlin_w4(
    const void* awq_qzeros_in, void* qzeros_marlin_out,
    int groups, int size_n, cudaStream_t stream)
{
    if (groups == 0 || size_n == 0) return;
    if (size_n % 64 != 0) return;  // marlin requires multiple of 64
    const int n8 = size_n / 8;
    const int threads = 32;
    const dim3 grid(groups, (n8 + threads - 1) / threads);
    awq_qzero_to_marlin_w4_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const std::uint32_t*>(awq_qzeros_in),
        static_cast<std::uint32_t*>(qzeros_marlin_out),
        groups, size_n);
}

namespace {

// AWQ qweight `[K, N/8]` packed-along-N with bit interleave
// [0,2,4,6,1,3,5,7] → GPTQ qweight `[K/8, N]` packed-along-K linear bit
// order. One output int32 per (k8_out, n_out) covers 8 nibbles for
// k = k8_out*8 + i (i in 0..7), reading from AWQ_qweight[k, n_out/8] at
// bit `4 * reverse_order[n_out%8]` where reverse_order = [0,4,1,5,2,6,
// 3,7] (AWQ's stored bit position for unpack-slot j).
__global__ void awq_qweight_to_gptq_w4_kernel(
    const std::uint32_t* __restrict__ in,    // AWQ [K, N/8]
    std::uint32_t*       __restrict__ out,   // GPTQ [K/8, N]
    int                                size_k,
    int                                size_n)
{
    const int k8_out = blockIdx.x;
    const int n_out = blockIdx.y * blockDim.x + threadIdx.x;
    if (k8_out >= size_k / 8 || n_out >= size_n) return;
    constexpr int reverse_order[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const int n8 = size_n / 8;
    const int n_packed = n_out / 8;
    const int n_in_8 = n_out % 8;
    const int src_bit = 4 * reverse_order[n_in_8];

    std::uint32_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int k = k8_out * 8 + i;
        const std::uint32_t src = in[k * n8 + n_packed];
        const std::uint32_t nibble = (src >> src_bit) & 0xFu;
        v |= nibble << (4 * i);
    }
    out[k8_out * size_n + n_out] = v;
}

}  // namespace

void launch_awq_qweight_to_gptq_w4(
    const void* awq_qweight_in, void* gptq_qweight_out,
    int size_k, int size_n, cudaStream_t stream)
{
    if (size_k == 0 || size_n == 0) return;
    if (size_k % 8 != 0) return;
    const int k8 = size_k / 8;
    const int threads = 64;
    const dim3 grid(k8, (size_n + threads - 1) / threads);
    awq_qweight_to_gptq_w4_kernel<<<grid, threads, 0, stream>>>(
        static_cast<const std::uint32_t*>(awq_qweight_in),
        static_cast<std::uint32_t*>(gptq_qweight_out),
        size_k, size_n);
}

namespace {

// Direct AWQ dequant to bf16, bypassing marlin. One thread per (n, k)
// output element; computes the dequanted value and writes the [N, K]
// transposed layout that HF Linear weights use.
//
//   bf16[n, k] = (w[k, n] - zp[g(k), n]) * scales[g(k), n]
//
// where:
//   w[k, n]  = (qweight[k, n/8] >> (4 * REV[n%8])) & 0xF
//   zp[g, n] = (qzeros[g, n/8]  >> (4 * REV[n%8])) & 0xF
//   REV      = [0, 4, 1, 5, 2, 6, 3, 7]   (AWQ "gemm" reverse-pack)
__global__ void awq_dequant_to_bf16_kernel(
    const std::uint32_t* __restrict__ qweight,   // [K, N/8]
    const std::uint32_t* __restrict__ qzeros,    // [groups, N/8]
    const __nv_bfloat16* __restrict__ scales,    // [groups, N]
    __nv_bfloat16*       __restrict__ out,       // [N, K]
    int                                size_k,
    int                                size_n,
    int                                group_size)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= size_n || k >= size_k) return;

    constexpr int REV[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const int n8 = size_n / 8;
    const int n_packed = n / 8;
    const int n_in_8 = n % 8;
    const int shift = 4 * REV[n_in_8];

    const int g = k / group_size;
    const std::uint32_t w_word = qweight[k * n8 + n_packed];
    const std::uint32_t zp_word = qzeros[g * n8 + n_packed];
    const int w_int4 = static_cast<int>((w_word >> shift) & 0xFu);
    const int zp_int4 = static_cast<int>((zp_word >> shift) & 0xFu);

    const float sc = __bfloat162float(scales[g * size_n + n]);
    const float val = static_cast<float>(w_int4 - zp_int4) * sc;
    out[n * size_k + k] = __float2bfloat16(val);
}

}  // namespace

void launch_awq_dequant_to_bf16(
    const void* qweight_in,
    const void* qzeros_in,
    const void* scales_in,
    void*       bf16_out,
    int         size_k,
    int         size_n,
    int         group_size,
    cudaStream_t stream)
{
    if (size_k == 0 || size_n == 0 || group_size == 0) return;
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((size_n + BX - 1) / BX, (size_k + BY - 1) / BY);
    awq_dequant_to_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const std::uint32_t*>(qweight_in),
        static_cast<const std::uint32_t*>(qzeros_in),
        static_cast<const __nv_bfloat16*>(scales_in),
        static_cast<__nv_bfloat16*>(bf16_out),
        size_k, size_n, group_size);
}

namespace {

// GPTQ dequant: qweight packed along K (no interleave); qzeros packed
// along N (no interleave); optional g_idx for desc_act=true.
//
//   nibble_w[k, n]  = (qweight[k/8, n] >> ((k%8)*4)) & 0xF
//   nibble_zp[g, n] = (qzeros[g, n/8] >> ((n%8)*4)) & 0xF
//   g(k)            = g_idx[k]               (desc_act=true)
//                   = k / group_size         (desc_act=false / g_idx=null)
//   bf16[n, k]      = (nibble_w[k, n] - (nibble_zp[g(k), n] + 1)) * scales[g(k), n]
//
// The `+1` on the zero-point matches autogptq's storage convention:
// `qzeros = zp - 1` (canonical), so the dequanter must add it back.
// For symmetric GPTQ (kU4B8 in marlin), qzeros is filled with 7 → +1
// gives 8 (the standard bias for kU4B8), and (nibble - 8) yields the
// signed [-8, 7] range that scales applies on top of.
__global__ void gptq_dequant_to_bf16_kernel(
    const std::uint32_t* __restrict__ qweight,    // [K/8, N]
    const std::uint32_t* __restrict__ qzeros,     // [groups, N/8]
    const __nv_bfloat16* __restrict__ scales,     // [groups, N]
    const std::int32_t*  __restrict__ g_idx,      // [K] or nullptr
    __nv_bfloat16*       __restrict__ out,        // [N, K]
    int                                size_k,
    int                                size_n,
    int                                group_size)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= size_n || k >= size_k) return;

    const int n8 = size_n / 8;
    const int g = (g_idx != nullptr)
                      ? g_idx[k]
                      : (k / group_size);

    const std::uint32_t w_word = qweight[(k / 8) * size_n + n];
    const std::uint32_t z_word = qzeros[g * n8 + (n / 8)];
    const int w_int4  = static_cast<int>((w_word >> ((k % 8) * 4)) & 0xFu);
    const int zp_int4 = static_cast<int>((z_word >> ((n % 8) * 4)) & 0xFu) + 1;

    const float sc = __bfloat162float(scales[g * size_n + n]);
    const float val = static_cast<float>(w_int4 - zp_int4) * sc;
    out[n * size_k + k] = __float2bfloat16(val);
}

}  // namespace

void launch_gptq_dequant_to_bf16(
    const void* qweight_in,
    const void* qzeros_in,
    const void* scales_in,
    const void* g_idx_in,
    void*       bf16_out,
    int         size_k,
    int         size_n,
    int         group_size,
    cudaStream_t stream)
{
    if (size_k == 0 || size_n == 0 || group_size == 0) return;
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((size_n + BX - 1) / BX, (size_k + BY - 1) / BY);
    gptq_dequant_to_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const std::uint32_t*>(qweight_in),
        static_cast<const std::uint32_t*>(qzeros_in),
        static_cast<const __nv_bfloat16*>(scales_in),
        static_cast<const std::int32_t*>(g_idx_in),
        static_cast<__nv_bfloat16*>(bf16_out),
        size_k, size_n, group_size);
}

}  // namespace pie_cuda_driver::kernels
