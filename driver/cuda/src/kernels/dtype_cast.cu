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

// E8M0 stores an exponent and nothing else: byte `b` denotes `2^(b - 127)`,
// with `0xFF` reserved for NaN. That is the fp32 exponent field verbatim, so
// the decode is a shift rather than any arithmetic -- `b << 23` *is* the
// answer, and `exp2f` would be a slower way to write it.
__global__ void cast_e8m0_to_fp32_kernel(
    const std::uint8_t* __restrict__ src,
    float*              __restrict__ dst,
    std::size_t                      n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const std::uint32_t bits = static_cast<std::uint32_t>(src[i]);
    dst[i] = bits == 0xFFu ? __int_as_float(0x7FFFFFFF)
                           : __int_as_float(bits << 23);
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

void launch_cast_e8m0_to_fp32(
    const void* src_e8m0, void* dst_fp32,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    cast_e8m0_to_fp32_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const std::uint8_t*>(src_e8m0),
        static_cast<float*>(dst_fp32), n);
}

namespace {

// One multiply per element, in fp32 whatever the storage dtype. The narrow
// dtypes round once, on the store -- accumulating in bf16 would round the
// operand as well, and the loader's host executor (which multiplies in fp32
// and is compared against this) would disagree.
__global__ void scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*       __restrict__ dst,
    std::size_t                       n,
    float                             factor)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2bfloat16(__bfloat162float(src[i]) * factor);
}

__global__ void scale_fp32_kernel(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    std::size_t               n,
    float                     factor)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i] * factor;
}

__global__ void scale_fp16_kernel(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    std::size_t                n,
    float                      factor)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(__half2float(src[i]) * factor);
}

}  // namespace

void launch_scale_bf16(
    const void* src_bf16, void* dst_bf16,
    std::size_t n, float factor, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    scale_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_bf16),
        static_cast<__nv_bfloat16*>(dst_bf16), n, factor);
}

void launch_scale_fp32(
    const void* src_fp32, void* dst_fp32,
    std::size_t n, float factor, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    scale_fp32_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const float*>(src_fp32),
        static_cast<float*>(dst_fp32), n, factor);
}

void launch_scale_fp16(
    const void* src_fp16, void* dst_fp16,
    std::size_t n, float factor, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    scale_fp16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __half*>(src_fp16),
        static_cast<__half*>(dst_fp16), n, factor);
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

namespace {

__global__ void scale_rows_bf16_kernel(
    __nv_bfloat16* buf,
    const __nv_bfloat16* l,
    int rows,
    int width)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= width) return;
    const std::size_t i =
        static_cast<std::size_t>(r) * width + c;
    buf[i] = __float2bfloat16(
        __bfloat162float(buf[i]) * __bfloat162float(l[c]));
}

}  // namespace

void launch_scale_rows_bf16(
    void*         buf_bf16,
    const void*   l_bf16,
    int           rows,
    int           width,
    cudaStream_t  stream)
{
    if (rows == 0 || width == 0) return;
    constexpr int BX = 128, BY = 2;
    const dim3 block(BX, BY);
    const dim3 grid((width + BX - 1) / BX, (rows + BY - 1) / BY);
    scale_rows_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(buf_bf16),
        static_cast<const __nv_bfloat16*>(l_bf16),
        rows, width);
}

}  // namespace pie_cuda_driver::kernels
