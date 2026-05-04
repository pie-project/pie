#include "kernels/altup_aux.hpp"

#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace pie_cuda_driver::kernels {

namespace cg = cooperative_groups;

namespace {

// Block-wide reduction of `local` (one float per thread) to thread 0.
// Uses warp shfl + shared-mem combine. Returns the reduced value;
// only thread 0 has it. `smem` must be sized for `blockDim.x / 32` floats.
__device__ __forceinline__ float block_sum(float local, float* smem) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    for (int off = 16; off > 0; off >>= 1) {
        local += tile.shfl_down(local, off);
    }
    if (tile.thread_rank() == 0) smem[tile.meta_group_rank()] = local;
    __syncthreads();
    if (tile.meta_group_rank() == 0) {
        float v = (threadIdx.x < tile.meta_group_size()) ? smem[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            v += tile.shfl_down(v, off);
        }
        if (tile.thread_rank() == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

__global__ void compute_rms_kernel(
    const __nv_bfloat16* __restrict__ ref,
    float* __restrict__               out,
    int H, float eps)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row = ref + (long long)t * H;

    extern __shared__ float smem[];
    float local = 0.f;
    for (int h = tid; h < H; h += blockDim.x) {
        const float v = __bfloat162float(row[h]);
        local += v * v;
    }
    const float total = block_sum(local, smem);
    if (tid == 0) {
        const float mean_sq = total / static_cast<float>(H);
        out[t] = sqrtf(fmaxf(mean_sq, eps));
    }
}

__global__ void magnitude_rescale_kernel(
    __nv_bfloat16* __restrict__       x,
    const float* __restrict__         target_rms,
    int H, float eps)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    __nv_bfloat16* row = x + (long long)t * H;

    extern __shared__ float smem[];
    float local = 0.f;
    for (int h = tid; h < H; h += blockDim.x) {
        const float v = __bfloat162float(row[h]);
        local += v * v;
    }
    const float total = block_sum(local, smem);
    __shared__ float scale;
    if (tid == 0) {
        const float new_rms = sqrtf(fmaxf(total / static_cast<float>(H), eps));
        scale = target_rms[t] / new_rms;
    }
    __syncthreads();

    for (int h = tid; h < H; h += blockDim.x) {
        const float v = __bfloat162float(row[h]) * scale;
        row[h] = __float2bfloat16(v);
    }
}

__global__ void mean_streams_kernel(
    const __nv_bfloat16* __restrict__ streams,
    __nv_bfloat16* __restrict__       out,
    int K, int T, int H)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || h >= H) return;

    const long long stride = (long long)T * H;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        sum += __bfloat162float(streams[(long long)k * stride + (long long)t * H + h]);
    }
    out[(long long)t * H + h] = __float2bfloat16(sum / static_cast<float>(K));
}

__global__ void unpack_predict_coefs_kernel(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__               out,
    int T, int K)
{
    const int t  = blockIdx.x;
    const int kk = threadIdx.x;
    if (t >= T || kk >= K * K) return;
    const int k = kk / K;
    const int j = kk % K;
    // out[t, j, k] = in[t, k*K + j]  (the permute(last two))
    const float v = __bfloat162float(in[(long long)t * K * K + (long long)k * K + j]);
    out[(long long)t * K * K + (long long)j * K + k] = v;
}

__global__ void unpack_correct_coefs_kernel(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__               out,
    int T, int K)
{
    const int t = blockIdx.x;
    const int k = threadIdx.x;
    if (t >= T || k >= K) return;
    const float v = __bfloat162float(in[(long long)t * K + k]);
    out[(long long)t * K + k] = v + 1.0f;
}

__global__ void tanh_kernel(__nv_bfloat16* x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = __bfloat162float(x[idx]);
    x[idx] = __float2bfloat16(tanhf(v));
}

}  // namespace

void launch_compute_rms_bf16(
    const void* ref, float* out, int T, int H, float eps, cudaStream_t stream)
{
    if (T <= 0 || H <= 0) return;
    constexpr int BLOCK = 256;
    const int smem = (BLOCK / 32) * sizeof(float);
    compute_rms_kernel<<<T, BLOCK, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(ref), out, H, eps);
}

void launch_magnitude_rescale_bf16(
    void* x, const float* target_rms, int T, int H, float eps, cudaStream_t stream)
{
    if (T <= 0 || H <= 0) return;
    constexpr int BLOCK = 256;
    const int smem = (BLOCK / 32) * sizeof(float);
    magnitude_rescale_kernel<<<T, BLOCK, smem, stream>>>(
        static_cast<__nv_bfloat16*>(x), target_rms, H, eps);
}

void launch_mean_streams_bf16(
    const void* streams, void* out, int K, int T, int H, cudaStream_t stream)
{
    if (T <= 0 || K <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(T, (H + BLOCK - 1) / BLOCK);
    mean_streams_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(streams),
        static_cast<__nv_bfloat16*>(out),
        K, T, H);
}

void launch_altup_unpack_predict_coefs(
    const void* in_bf16, float* out_fp32, int T, int K, cudaStream_t stream)
{
    if (T <= 0 || K <= 0) return;
    unpack_predict_coefs_kernel<<<T, K * K, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16), out_fp32, T, K);
}

void launch_altup_unpack_correct_coefs(
    const void* in_bf16, float* out_fp32, int T, int K, cudaStream_t stream)
{
    if (T <= 0 || K <= 0) return;
    unpack_correct_coefs_kernel<<<T, K, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in_bf16), out_fp32, T, K);
}

void launch_tanh_bf16(void* x, int numel, cudaStream_t stream) {
    if (numel <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (numel + BLOCK - 1) / BLOCK;
    tanh_kernel<<<grid, BLOCK, 0, stream>>>(static_cast<__nv_bfloat16*>(x), numel);
}

}  // namespace pie_cuda_driver::kernels
