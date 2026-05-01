#include "kernels/attn_sink.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

__global__ void attn_sink_rescale_bf16_kernel(
    __nv_bfloat16* __restrict__       o,
    const float* __restrict__         lse,
    const __nv_bfloat16* __restrict__ sinks,
    int N,
    int num_q_heads,
    int head_dim)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= N || h >= num_q_heads) return;

    // r = sigmoid(lse[t,h] - sink[h]). lse is +∞ on causal masked-out
    // rows that flashinfer fills with -inf max — it would normally
    // produce 0 output through the `m == -inf ? 0 : 1/d` guard in the
    // OutputTransform; we propagate that by treating non-finite lse as
    // "no contribution" (r=1, leaves o untouched, since o is already 0).
    const float lse_val = lse[t * num_q_heads + h];
    const float sink    = __bfloat162float(sinks[h]);
    float r;
    if (!isfinite(lse_val)) {
        r = 1.0f;
    } else {
        const float diff = lse_val - sink;
        // sigmoid(x) = 1 / (1 + exp(-x)). Stable for both signs.
        r = 1.0f / (1.0f + __expf(-diff));
    }

    const int row_stride = num_q_heads * head_dim;
    __nv_bfloat16* row = o + t * row_stride + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = __bfloat162float(row[d]);
        row[d] = __float2bfloat16(v * r);
    }
}

}  // namespace

void launch_attention_sink_rescale_bf16(
    void*        o,
    const float* lse,
    const void*  sinks,
    int N,
    int num_q_heads,
    int head_dim,
    cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_q_heads));
    const int block = (head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim);
    attn_sink_rescale_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(o),
        lse,
        static_cast<const __nv_bfloat16*>(sinks),
        N, num_q_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels
