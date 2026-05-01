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

    // r = sigmoid(lse_natural - sink[h]).
    //
    // flashinfer's state_t::get_lse() returns `m + log2(d)` — that's
    // log_2(Σ_kv exp(z · sm_scale)), not the natural log. (See
    // `flashinfer/attention/state.cuh`; both prefill and decode kernels
    // write this base-2 form into `params.lse`.) The HF gpt-oss sink
    // formulation extends the softmax denominator with `exp(sink)` in
    // natural log space — so we have to convert by `ln(2)` before
    // taking the difference. Without this conversion the rescale was
    // off by a factor of 0.693 in the sigmoid argument, which matches
    // HF top-1 by accident on most prompts but produces accumulating
    // drift that degenerates greedy decoding past a few steps on some
    // inputs.
    constexpr float kLn2 = 0.69314718055994530942f;
    const float lse_val = lse[t * num_q_heads + h];
    const float sink    = __bfloat162float(sinks[h]);
    float r;
    if (!isfinite(lse_val)) {
        // lse=-inf on causal-masked-out rows; o is already 0 there, so
        // the rescale factor is don't-care. Leave r=1.
        r = 1.0f;
    } else {
        const float diff = lse_val * kLn2 - sink;
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
