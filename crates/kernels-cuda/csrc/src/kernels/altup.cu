#include "kernels/altup.hpp"

#include <cuda_bf16.h>
#include <cstdio>

namespace pie_cuda_driver::kernels {

namespace {

__global__ void altup_predict_kernel(
    const __nv_bfloat16* __restrict__ streams,
    const float* __restrict__         coefs,
    __nv_bfloat16* __restrict__       predictions,
    int K, int T, int H)
{
    const int t = blockIdx.x;
    const int k = blockIdx.y;
    const int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T || k >= K || h >= H) return;

    const long long stream_stride = (long long)T * H;

    // Σ_j coefs[t, j, k] · streams[j, t, h], plus the residual streams[k, t, h].
    float sum = 0.f;
    for (int j = 0; j < K; ++j) {
        const float c = coefs[(long long)t * K * K + (long long)j * K + k];
        const float s = __bfloat162float(
            streams[(long long)j * stream_stride + (long long)t * H + h]);
        sum += c * s;
    }
    sum += __bfloat162float(streams[(long long)k * stream_stride + (long long)t * H + h]);
    predictions[(long long)k * stream_stride + (long long)t * H + h] = __float2bfloat16(sum);
}

__global__ void altup_correct_kernel(
    const __nv_bfloat16* __restrict__ predictions,
    const __nv_bfloat16* __restrict__ activated,
    const float* __restrict__         correction_coefs_plus_one,
    __nv_bfloat16* __restrict__       corrected,
    int K, int T, int H, int active_idx)
{
    const int t = blockIdx.x;
    const int k = blockIdx.y;
    const int h = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T || k >= K || h >= H) return;

    const long long stream_stride = (long long)T * H;
    const float a       = __bfloat162float(activated[(long long)t * H + h]);
    const float p_act   = __bfloat162float(predictions[(long long)active_idx * stream_stride + (long long)t * H + h]);
    const float p_k     = __bfloat162float(predictions[(long long)k * stream_stride + (long long)t * H + h]);
    const float coef    = correction_coefs_plus_one[(long long)t * K + k];
    const float result  = (a - p_act) * coef + p_k;
    corrected[(long long)k * stream_stride + (long long)t * H + h] = __float2bfloat16(result);
}

}  // namespace

void launch_altup_predict_bf16(
    const void* streams, const float* coefs, void* predictions,
    int K, int T, int H, cudaStream_t stream)
{
    if (T <= 0 || K <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);
    altup_predict_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(streams), coefs,
        static_cast<__nv_bfloat16*>(predictions),
        K, T, H);
}

void launch_altup_correct_bf16(
    const void* predictions, const void* activated,
    const float* correction_coefs_plus_one, void* corrected,
    int K, int T, int H, int active_idx, cudaStream_t stream)
{
    if (T <= 0 || K <= 0 || H <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(T, K, (H + BLOCK - 1) / BLOCK);
    altup_correct_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(predictions),
        static_cast<const __nv_bfloat16*>(activated),
        correction_coefs_plus_one,
        static_cast<__nv_bfloat16*>(corrected),
        K, T, H, active_idx);
}

}  // namespace pie_cuda_driver::kernels
