#include "kernels/rmsnorm.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// Block reduction that reproduces the shared-memory tree
// `for (off = BLOCK/2; off; off >>= 1) buf[t] += buf[t+off]` **exactly**, but
// runs the last five levels on warp shuffles. Those levels only ever touch
// threads 0..31, so the pairing is identical and the fp32 rounding is
// unchanged -- what goes away is five block-wide barriers, which at decode
// (one block per token, eight blocks on 148 SMs) were a third of the kernel.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum_exact(float local, float* buf)
{
    static_assert(BLOCK >= 32 && (BLOCK & (BLOCK - 1)) == 0,
                  "block_reduce_sum_exact needs a power-of-two BLOCK >= 32");
    const int tid = threadIdx.x;
    buf[tid] = local;
    __syncthreads();
#pragma unroll
    for (int off = BLOCK / 2; off >= 32; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    if (tid < 32) {
        float v = buf[tid];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (tid == 0) buf[0] = v;
    }
    __syncthreads();
    return buf[0];
}

// One block per row. `BLOCK` threads cooperate on the L2-norm reduction;
// each thread handles `hidden / BLOCK` elements. We always launch with
// BLOCK == 256, so the per-thread chunk is small even for hidden=8192.
//
// `WEIGHT_PLUS_ONE = true` selects Gemma's `(1 + w) * x_hat` convention.
template <int BLOCK, bool WEIGHT_PLUS_ONE>
__global__ void rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* xr = x + static_cast<long long>(row) * x_row_stride;
    __nv_bfloat16* yr = y + static_cast<long long>(row) * y_row_stride;

    // L2 norm across the row, accumulated in fp32.
    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    // Block-wide reduction via shared memory. CUB would be cleaner, but this
    // avoids a heavy header for the M1.2.2 build.
    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = __bfloat162float(xr[i]);
        float wv = __bfloat162float(weight[i]);
        if constexpr (WEIGHT_PLUS_ONE) wv += 1.f;
        yr[i] = __float2bfloat16(xv * inv_rms * wv);
    }
}

template <int BLOCK>
__global__ void residual_add_rmsnorm_bf16_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __nv_bfloat16* hr = hidden + row * hidden_size;
    const __nv_bfloat16* rr = residual + row * hidden_size;
    __nv_bfloat16* nr = norm_out + row * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float sum = __bfloat162float(hr[i]) + __bfloat162float(rr[i]);
        const __nv_bfloat16 rounded = __float2bfloat16(sum);
        hr[i] = rounded;
        const float v = __bfloat162float(rounded);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float xv = __bfloat162float(hr[i]);
        const float wv = __bfloat162float(weight[i]);
        nr[i] = __float2bfloat16(xv * inv_rms * wv);
    }
}

template <int BLOCK>
__global__ void residual_add_scale_rmsnorm_bf16_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ residual,
    float scale,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __nv_bfloat16* hr = hidden + row * hidden_size;
    const __nv_bfloat16* rr = residual + row * hidden_size;
    __nv_bfloat16* nr = norm_out + row * hidden_size;
    const float scale_rounded = __bfloat162float(__float2bfloat16(scale));

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float sum = __bfloat162float(hr[i]) + __bfloat162float(rr[i]);
        const __nv_bfloat16 rounded_sum = __float2bfloat16(sum);
        const __nv_bfloat16 scaled =
            __float2bfloat16(__bfloat162float(rounded_sum) * scale_rounded);
        hr[i] = scaled;
        const float v = __bfloat162float(scaled);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float xv = __bfloat162float(hr[i]);
        const float wv = __bfloat162float(weight[i]);
        nr[i] = __float2bfloat16(xv * inv_rms * wv);
    }
}

template <int BLOCK>
__global__ void rmsnorm_residual_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ hidden,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* xr = x + row * hidden_size;
    __nv_bfloat16* hr = hidden + row * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const __nv_bfloat16 norm = __float2bfloat16(
            __bfloat162float(xr[i]) * inv_rms *
            __bfloat162float(weight[i]));
        hr[i] = __float2bfloat16(
            __bfloat162float(hr[i]) + __bfloat162float(norm));
    }
}

template <int BLOCK>
__global__ void rmsnorm_residual_add_scale_rmsnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ hidden,
    float scale,
    const __nv_bfloat16* __restrict__ next_weight,
    __nv_bfloat16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* xr = x + row * hidden_size;
    __nv_bfloat16* hr = hidden + row * hidden_size;
    __nv_bfloat16* nr = norm_out + row * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);
    const float scale_rounded = __bfloat162float(__float2bfloat16(scale));
    float local_next = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const __nv_bfloat16 norm = __float2bfloat16(
            __bfloat162float(xr[i]) * inv_rms *
            __bfloat162float(weight[i]));
        const float sum = __bfloat162float(hr[i]) + __bfloat162float(norm);
        const __nv_bfloat16 rounded_sum = __float2bfloat16(sum);
        const __nv_bfloat16 scaled =
            __float2bfloat16(__bfloat162float(rounded_sum) * scale_rounded);
        hr[i] = scaled;
        const float v = __bfloat162float(scaled);
        local_next += v * v;
    }

    __shared__ float buf_next[BLOCK];
    const float buf_next_sum = block_reduce_sum_exact<BLOCK>(local_next, buf_next);

    const float inv_next =
        rsqrtf(buf_next_sum / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += BLOCK) {
        nr[i] = __float2bfloat16(
            __bfloat162float(hr[i]) * inv_next *
            __bfloat162float(next_weight[i]));
    }
}

}  // namespace

void launch_rmsnorm_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    launch_rmsnorm_strided_bf16(
        x, weight, y, num_rows, hidden, hidden, hidden, eps, stream);
}

void launch_rmsnorm_strided_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, int x_row_stride, int y_row_stride,
    float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_bf16_kernel<BLOCK, /*WEIGHT_PLUS_ONE=*/false><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, x_row_stride, y_row_stride, eps);
}

void launch_residual_add_rmsnorm_bf16(
    void* hidden,
    const void* residual,
    const void* weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    residual_add_rmsnorm_bf16_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(norm_out),
        hidden_size, eps);
}

void launch_residual_add_scale_rmsnorm_bf16(
    void* hidden,
    const void* residual,
    float scale,
    const void* next_weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    residual_add_scale_rmsnorm_bf16_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(hidden),
        static_cast<const __nv_bfloat16*>(residual),
        scale,
        static_cast<const __nv_bfloat16*>(next_weight),
        static_cast<__nv_bfloat16*>(norm_out),
        hidden_size, eps);
}

void launch_rmsnorm_residual_add_bf16(
    const void* x,
    const void* weight,
    void* hidden,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_residual_add_bf16_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(hidden),
        hidden_size, eps);
}

void launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
    const void* x,
    const void* weight,
    void* hidden,
    float scale,
    const void* next_weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_residual_add_scale_rmsnorm_bf16_kernel<BLOCK>
        <<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x),
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<__nv_bfloat16*>(hidden),
            scale,
            static_cast<const __nv_bfloat16*>(next_weight),
            static_cast<__nv_bfloat16*>(norm_out),
            hidden_size, eps);
}

void launch_rmsnorm_gemma_bf16(
    const void* x, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_bf16_kernel<BLOCK, /*WEIGHT_PLUS_ONE=*/true><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, hidden, hidden, eps);
}

namespace {

// No-weight variant. Mirrors the templated kernel above but skips the
// gamma multiplication entirely — `y = x * rsqrt(var + eps)`.
template <int BLOCK>
__global__ void rmsnorm_no_scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16*       __restrict__ y,
    int hidden, float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* xr = x + row * hidden;
    __nv_bfloat16*       yr = y + row * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = __float2bfloat16(__bfloat162float(xr[i]) * inv_rms);
    }
}

}  // namespace

void launch_rmsnorm_no_scale_bf16(
    const void* x, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_no_scale_bf16_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(y),
        hidden, eps);
}

namespace {

// y = weight * (x * rsqrt(mean(x^2) + eps)) * silu(gate). One block per
// row; both the variance reduction and the writeback are vectorized
// the same way as the plain RMSNorm kernel.
//
// `weight` is fp32 — Qwen3.5 ships RMSNormGated weights in fp32 alongside
// bf16 activations.
template <int BLOCK>
__global__ void rmsnorm_gated_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const float*         __restrict__ weight,
    __nv_bfloat16*       __restrict__ y,
    int hidden, float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const __nv_bfloat16* xr = x    + row * hidden;
    const __nv_bfloat16* gr = gate + row * hidden;
    __nv_bfloat16*       yr = y    + row * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = __bfloat162float(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = __bfloat162float(xr[i]) * inv_rms;
        const float wv = weight[i];
        const float gv = __bfloat162float(gr[i]);
        // silu(z) = z / (1 + exp(-z)) = z * sigmoid(z).
        const float sg = gv / (1.f + __expf(-gv));
        yr[i] = __float2bfloat16(wv * xv * sg);
    }
}

// Same as rmsnorm_gated_bf16_kernel but x is fp32 (e.g. came straight
// from the GDN recurrent step which outputs fp32). Fuses the separate
// fp32→bf16 conversion that pie's qwen3.5 forward used to emit as its
// own kernel launch before calling rmsnorm_gated_bf16. Per-row HBM
// traffic drops from 12*hidden bytes (4-byte x read + 2-byte x write +
// 4-byte x read + 2-byte gate read + 2-byte y write) to 8*hidden bytes
// (4-byte x read + 2-byte gate read + 2-byte y write) — eliminates one
// full pass over the fp32→bf16 intermediate buffer.
template <int BLOCK>
__global__ void rmsnorm_gated_fp32_in_bf16_kernel(
    const float*         __restrict__ x,        // fp32 input
    const __nv_bfloat16* __restrict__ gate,
    const float*         __restrict__ weight,
    __nv_bfloat16*       __restrict__ y,
    int hidden, float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float*         xr = x    + row * hidden;
    const __nv_bfloat16* gr = gate + row * hidden;
    __nv_bfloat16*       yr = y    + row * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = xr[i];
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = xr[i] * inv_rms;
        const float wv = weight[i];
        const float gv = __bfloat162float(gr[i]);
        const float sg = gv / (1.f + __expf(-gv));
        yr[i] = __float2bfloat16(wv * xv * sg);
    }
}

}  // namespace

void launch_rmsnorm_gated_fp32_in_bf16(
    const void* x, const void* gate, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_gated_fp32_in_bf16_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const float*>(x),
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const float*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, eps);
}

void launch_rmsnorm_gated_bf16(
    const void* x, const void* gate, const void* weight, void* y,
    int num_rows, int hidden, float eps, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_rows);
    dim3 block(BLOCK);
    rmsnorm_gated_bf16_kernel<BLOCK><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const float*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, eps);
}

}  // namespace pie_cuda_driver::kernels
