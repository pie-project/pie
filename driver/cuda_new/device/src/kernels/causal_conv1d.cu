#include "causal_conv1d.cuh"

#include <cuda_bf16.h>

// Lifted verbatim from driver/cuda/src/kernels/causal_conv1d.cu (the
// single-request prefill/forward variant). The only changes are the namespace
// (pie_cuda_driver::kernels -> pie_cuda_device::kernels) and dropping the
// launch_ prefix on the host entry point.

namespace pie_cuda_device::kernels {

namespace {

__device__ __forceinline__ float silu_f(float z) {
    return z / (1.f + __expf(-z));
}

// One block per (channel, output token range). Each thread handles a
// few output tokens in its block. The kernel size K is small (4 on
// Qwen3.5), so the K accumulator unrolls trivially.
//
//     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - K + 1 + k, c]  + bias[c] )
//
// where `x[t<0, c]` is read from the prior state window. Fresh prompts
// arrive with a zeroed state window, so this also implements causal
// padding for first-chunk prefill. The trailing K input rows are written
// back into `state_out[K, C]` (oldest first) so a follow-up decode or
// mixed prefill chunk can resume from there.
__global__ void causal_conv1d_prefill_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ state_out,
    int N, int C, int K)
{
    const int c = blockIdx.x;       // one channel per block
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    if (c >= C) return;

    const float bias_v = bias ? __bfloat162float(bias[c]) : 0.f;

    // Each thread strides through tokens.
    for (int t = tid; t < N; t += block_size) {
        float acc = bias_v;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {  // unroll up to 8 (Qwen3.5 uses K=4)
            if (k >= K) break;
            const int src_t = t - (K - 1) + k;
            float xv = 0.f;
            if (src_t < 0) {
                if (state_out) {
                    xv = __bfloat162float(state_out[(K + src_t) * C + c]);
                }
            } else {
                xv = __bfloat162float(x[src_t * C + c]);
            }
            const float wv = __bfloat162float(weight[c * K + k]);
            acc += wv * xv;
        }
        y[t * C + c] = __float2bfloat16(silu_f(acc));
    }

    __syncthreads();

    // Persist the trailing K input rows into state_out (one thread does
    // this per channel; it's a tiny copy with strided indexing).
    if (state_out && tid == 0) {
        for (int s = 0; s < K; ++s) {
            const int src_t = N - K + s;  // token index for state slot s
            const float v = (src_t < 0)
                ? __bfloat162float(state_out[(K + src_t) * C + c])
                : __bfloat162float(x[src_t * C + c]);
            state_out[s * C + c] = __float2bfloat16(v);
        }
    }
}

}  // namespace

void causal_conv1d_prefill_bf16(
    const void* x, const void* weight, const void* bias,
    void* y, void* state_out,
    int N, int C, int K, cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || K <= 0) return;
    constexpr int BLOCK = 64;
    dim3 grid(C);
    dim3 block(BLOCK);
    causal_conv1d_prefill_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(y),
        static_cast<__nv_bfloat16*>(state_out),
        N, C, K);
}

}  // namespace pie_cuda_device::kernels
