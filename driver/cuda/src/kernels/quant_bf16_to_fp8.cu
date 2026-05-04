#include "kernels/quant_bf16_to_fp8.hpp"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;
constexpr float FP8_E4M3_MAX = 448.f;     // OCP MX spec: max representable
constexpr float INT8_MAX_F   = 127.f;     // signed int8 symmetric range

// Block-wide absmax via warp shuffles + shared-mem reduction. We do the
// final atomic into the device scalar (one atomic per block, not per
// thread) so the kernel stays scale-free on `n`.
__global__ void absmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ W,
    float*               __restrict__ out,
    std::size_t                       n)
{
    __shared__ float warp_max[BLOCK / 32];
    const unsigned tid    = threadIdx.x;
    const unsigned warp   = tid / 32;
    const unsigned lane   = tid & 31;
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + tid;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * BLOCK;

    float local = 0.f;
    for (; i < n; i += stride) {
        const float v = fabsf(__bfloat162float(W[i]));
        if (v > local) local = v;
    }
    // Warp reduction.
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    // First warp reduces the per-warp results.
    if (warp == 0) {
        local = (tid < BLOCK / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) atomicMax(reinterpret_cast<int*>(out),
                                 __float_as_int(local));
    }
}

__global__ void quant_bf16_to_fp8_kernel(
    const __nv_bfloat16*    __restrict__ W,
    __nv_fp8_storage_t*     __restrict__ out,
    float                                 scale_inv,
    std::size_t                           n)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float f = __bfloat162float(W[i]) * scale_inv;
    // __nv_cvt_float_to_fp8 already saturates to ±448 for E4M3.
    out[i] = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
}

// Stage 1: one block per row, output absmax per row to `absmax_out`.
// Used by the TP-aware quant path so the host can all-reduce the
// per-row absmax across ranks before computing the final scales.
__global__ void absmax_per_row_kernel(
    const __nv_bfloat16*    __restrict__ W,
    float*                  __restrict__ absmax_out,  // [rows]
    int                                   cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const std::size_t row_off = static_cast<std::size_t>(row) * cols;
    float local = 0.f;
    for (int j = tid; j < cols; j += BLOCK) {
        const float v = fabsf(__bfloat162float(W[row_off + j]));
        if (v > local) local = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = (tid < BLOCK / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) absmax_out[row] = local;
    }
}

// Convert per-row absmax → weight_scale_inv = absmax / FP8_MAX in place.
__global__ void absmax_to_scale_inv_kernel(float* x, int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = (v > 0.f) ? (v / FP8_E4M3_MAX) : 1.f;
}

// INT8 variant: convert per-row absmax → scale_inv = absmax / 127.
__global__ void absmax_to_scale_inv_int8_kernel(float* x, int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = (v > 0.f) ? (v / INT8_MAX_F) : 1.f;
}

// Cast bf16 → int8 using a precomputed per-row scale_inv. Mirrors the
// FP8 cast but stores int8 with INT8_MAX_F saturation.
__global__ void cast_per_channel_int8_kernel(
    const __nv_bfloat16* __restrict__ W,
    std::int8_t*         __restrict__ out,
    const float*         __restrict__ scale_inv_dev,
    int                                cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float scale_inv = scale_inv_dev[row];
    const float quant = (scale_inv > 0.f) ? (1.f / scale_inv) : 0.f;
    const std::size_t row_off = static_cast<std::size_t>(row) * cols;
    for (int j = tid; j < cols; j += BLOCK) {
        const float f = __bfloat162float(W[row_off + j]) * quant;
        int q = static_cast<int>(rintf(f));
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        out[row_off + j] = static_cast<std::int8_t>(q);
    }
}

// Stage 2: cast `[rows, cols]` bf16 → fp8 using per-row weight_scale_inv.
__global__ void cast_per_channel_kernel(
    const __nv_bfloat16*    __restrict__ W,
    __nv_fp8_storage_t*     __restrict__ out,
    const float*            __restrict__ scale_inv,  // [rows]
    int                                   cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float s = scale_inv[row];
    const float s_recip = (s > 0.f) ? (1.f / s) : 0.f;
    const std::size_t row_off = static_cast<std::size_t>(row) * cols;
    for (int j = tid; j < cols; j += BLOCK) {
        const float f = __bfloat162float(W[row_off + j]) * s_recip;
        out[row_off + j] = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
    }
}

// One block per row. Threads in the block cooperatively absmax the row
// (warp shuffle + shared mem reduction), pick `scale_inv_row = fp8_max /
// max(absmax, eps)`, then cast each element. We emit BOTH the fp8 row
// and `weight_scale_inv = absmax / fp8_max` (the multiplicative factor
// the GEMM dispatcher will hand cuBLASLt) — splitting the work means
// the dispatcher never has to compute reciprocals at runtime.
__global__ void quant_per_channel_kernel(
    const __nv_bfloat16*    __restrict__ W,
    __nv_fp8_storage_t*     __restrict__ out,
    float*                  __restrict__ scale_inv,   // [rows]
    int                                  cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const std::size_t row_off = static_cast<std::size_t>(row) * cols;
    float local = 0.f;
    for (int j = tid; j < cols; j += BLOCK) {
        const float v = fabsf(__bfloat162float(W[row_off + j]));
        if (v > local) local = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    float row_max;
    if (warp == 0) {
        local = (tid < BLOCK / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) warp_max[0] = local;
    }
    __syncthreads();
    row_max = warp_max[0];

    // Degenerate row → scale_inv = 1, all-zero fp8.
    const float fp8_inv = (row_max > 0.f) ? (FP8_E4M3_MAX / row_max) : 1.f;
    const float weight_scale_inv = (row_max > 0.f) ? (row_max / FP8_E4M3_MAX) : 1.f;
    if (tid == 0) scale_inv[row] = weight_scale_inv;

    for (int j = tid; j < cols; j += BLOCK) {
        const float f = __bfloat162float(W[row_off + j]) * fp8_inv;
        out[row_off + j] = __nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3);
    }
}

}  // namespace

void launch_absmax_bf16(
    const void* W_bf16, float* absmax_dev,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    CUDA_CHECK(cudaMemsetAsync(absmax_dev, 0, sizeof(float), stream));
    // Cap grid at 1024 blocks — enough parallelism for >256k elements
    // and keeps the atomic contention bounded.
    const unsigned blocks_full = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    const unsigned blocks = blocks_full < 1024u ? blocks_full : 1024u;
    absmax_bf16_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16), absmax_dev, n);
}

void launch_quant_bf16_to_fp8_e4m3(
    const void* W_bf16, std::uint8_t* W_fp8,
    float scale_inv, std::size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    const auto blocks = static_cast<unsigned>((n + BLOCK - 1) / BLOCK);
    quant_bf16_to_fp8_kernel<<<blocks, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16),
        reinterpret_cast<__nv_fp8_storage_t*>(W_fp8),
        scale_inv, n);
}

void quantize_bf16_to_fp8_e4m3_per_channel(
    const void* W_bf16, std::uint8_t* W_fp8,
    float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    const std::size_t shmem = (BLOCK / 32) * sizeof(float);
    quant_per_channel_kernel<<<rows, BLOCK, shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16),
        reinterpret_cast<__nv_fp8_storage_t*>(W_fp8),
        scale_inv_dev, cols);
}

namespace {

// One block per row: warp-shuffle absmax → cast bf16 → int8 with
// per-row scale. Mirrors `quant_per_channel_kernel` but stores int8
// instead of fp8 and uses INT8_MAX_F for the saturation point.
__global__ void quant_int8_per_channel_kernel(
    const __nv_bfloat16* __restrict__ W,
    std::int8_t*         __restrict__ out,
    float*               __restrict__ scale_inv,
    int                                cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float warp_max[];

    const std::size_t row_off = static_cast<std::size_t>(row) * cols;
    float local = 0.f;
    for (int j = tid; j < cols; j += BLOCK) {
        const float v = fabsf(__bfloat162float(W[row_off + j]));
        if (v > local) local = v;
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, local, off);
        if (other > local) local = other;
    }
    const int lane = tid & 31;
    const int warp = tid / 32;
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    float row_max;
    if (warp == 0) {
        local = (tid < BLOCK / 32) ? warp_max[lane] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_down_sync(0xffffffff, local, off);
            if (other > local) local = other;
        }
        if (lane == 0) warp_max[0] = local;
    }
    __syncthreads();
    row_max = warp_max[0];
    const float int_inv = (row_max > 0.f) ? (INT8_MAX_F / row_max) : 1.f;
    const float weight_scale_inv = (row_max > 0.f)
        ? (row_max / INT8_MAX_F)
        : 1.f;
    if (tid == 0) scale_inv[row] = weight_scale_inv;

    for (int j = tid; j < cols; j += BLOCK) {
        const float f = __bfloat162float(W[row_off + j]) * int_inv;
        // Symmetric round-half-to-even is the sane default; rintf is
        // round-to-nearest-even.
        int q = static_cast<int>(rintf(f));
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[row_off + j] = static_cast<std::int8_t>(q);
    }
}

}  // namespace

void quantize_bf16_to_int8_per_channel(
    const void* W_bf16, std::int8_t* W_int8,
    float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    const std::size_t shmem = (BLOCK / 32) * sizeof(float);
    quant_int8_per_channel_kernel<<<rows, BLOCK, shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16),
        W_int8, scale_inv_dev, cols);
}

// Per-token activation INT8 quant is mathematically the same op as
// per-channel weight INT8 quant: per-row symmetric absmax over a 2-D
// row-major buffer, producing one scale_inv per row. Reuse the same
// kernel — only the semantic naming differs.
void quantize_bf16_to_int8_per_token(
    const void* act_bf16, std::int8_t* act_int8,
    float* act_scale_inv, int n_tokens, int k, cudaStream_t stream)
{
    quantize_bf16_to_int8_per_channel(
        act_bf16, act_int8, act_scale_inv, n_tokens, k, stream);
}

void launch_absmax_to_scale_inv_int8(
    float* absmax_inout, int rows, cudaStream_t stream)
{
    if (rows == 0) return;
    const int blocks = (rows + BLOCK - 1) / BLOCK;
    absmax_to_scale_inv_int8_kernel<<<blocks, BLOCK, 0, stream>>>(
        absmax_inout, rows);
}

void launch_cast_bf16_to_int8_per_channel(
    const void* W_bf16, std::int8_t* W_int8,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    cast_per_channel_int8_kernel<<<rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16),
        W_int8, scale_inv_dev, cols);
}

namespace {

// W8A8 post-GEMM dequant: bf16[m,n] = int32[m,n] * act_inv[m] * w_inv[n].
// One thread per output element. Bandwidth-bound; no need to fuse with
// the GEMM since cuBLAS writes the int32 accumulator and we just scale
// it row × column afterwards.
__global__ void w8a8_dequant_kernel(
    const std::int32_t* __restrict__ acc,
    const float*        __restrict__ act_inv,
    const float*        __restrict__ w_inv,
    __nv_bfloat16*      __restrict__ out,
    int                                M,
    int                                N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;
    const float v = static_cast<float>(acc[m * N + n]) * act_inv[m] * w_inv[n];
    out[m * N + n] = __float2bfloat16(v);
}

}  // namespace

void dequant_int32_w8a8_to_bf16(
    const std::int32_t* acc_int32, const float* act_scale_inv,
    const float* w_scale_inv, void* out_bf16,
    int M, int N, cudaStream_t stream)
{
    if (M == 0 || N == 0) return;
    constexpr int BX = 32, BY = 8;
    const dim3 block(BX, BY);
    const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
    w8a8_dequant_kernel<<<grid, block, 0, stream>>>(
        acc_int32, act_scale_inv, w_scale_inv,
        static_cast<__nv_bfloat16*>(out_bf16), M, N);
}

void launch_absmax_per_row_bf16(
    const void* W_bf16, float* absmax_dev,
    int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    const std::size_t shmem = (BLOCK / 32) * sizeof(float);
    absmax_per_row_kernel<<<rows, BLOCK, shmem, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16), absmax_dev, cols);
}

void launch_absmax_to_scale_inv(
    float* absmax_inout, int rows, cudaStream_t stream)
{
    if (rows == 0) return;
    const auto blocks = static_cast<unsigned>((rows + BLOCK - 1) / BLOCK);
    absmax_to_scale_inv_kernel<<<blocks, BLOCK, 0, stream>>>(
        absmax_inout, rows);
}

void launch_cast_bf16_to_fp8_e4m3_per_channel(
    const void* W_bf16, std::uint8_t* W_fp8,
    const float* scale_inv_dev, int rows, int cols, cudaStream_t stream)
{
    if (rows == 0 || cols == 0) return;
    cast_per_channel_kernel<<<rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(W_bf16),
        reinterpret_cast<__nv_fp8_storage_t*>(W_fp8),
        scale_inv_dev, cols);
}

float quantize_bf16_to_fp8_e4m3_per_tensor(
    const void* W_bf16, std::uint8_t* W_fp8,
    std::size_t n, cudaStream_t stream)
{
    if (n == 0) return 1.f;
    // 1) absmax → tmp scalar
    float* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, sizeof(float)));
    launch_absmax_bf16(W_bf16, tmp, n, stream);

    // 2) Pull absmax to host (one sync per quant call — load-time
    // operation, not the hot path).
    float absmax = 0.f;
    CUDA_CHECK(cudaMemcpyAsync(&absmax, tmp, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));

    // Degenerate weights (all-zero) — pick scale=1.0 and let the cast
    // produce zeros. Returns weight_scale_inv = 1.0 (caller stores this
    // in the QuantMeta scale tensor; cuBLASLt treats it as a no-op).
    if (absmax == 0.f) {
        launch_quant_bf16_to_fp8_e4m3(W_bf16, W_fp8, 1.f, n, stream);
        return 1.f;
    }

    // We pick `weight_scale_inv` such that bf16 ≈ fp8 * weight_scale_inv,
    // i.e. weight_scale_inv = absmax / fp8_max. The cast multiplies by
    // the reciprocal:  fp8 = round(bf16 * (fp8_max / absmax)).
    const float weight_scale_inv = absmax / FP8_E4M3_MAX;
    const float scale_inv        = FP8_E4M3_MAX / absmax;
    launch_quant_bf16_to_fp8_e4m3(W_bf16, W_fp8, scale_inv, n, stream);
    return weight_scale_inv;
}

}  // namespace pie_cuda_driver::kernels
