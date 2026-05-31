#include "rope_partial.cuh"

#include <cuda_bf16.h>

// Lifted verbatim from driver/cuda/src/kernels/rope.cu
// (rope_partial_bf16_kernel + launch_rope_partial_bf16). The kernel inlines
// its rotation (it does not call rope.cu's rotate_pair helper), so this file
// is self-contained and does not depend on rope.cu. The launcher's
// `position_delta` is hardcoded to 0 here, matching the base entry point.

namespace pie_cuda_device::kernels {

namespace {

// Proportional RoPE (Gemma-4 full-attention layers, HF reference).
//
// HF builds the frequency table as `freq[k] = 1 / theta^(2k/head_dim)`
// for k ∈ [0, rotary_dim/2), then pads the rest of the head's lower-
// half dim entries with `cos=1 / sin=0` (identity). The pair offset
// is the *full* `head_dim/2`, NOT `rotary_dim/2` — every dim in the
// lower half rotates with its mate in the upper half, but the
// rotation angle is zero for k ≥ rotary_dim/2 (so those pairs pass
// through unchanged).
//
// Two ways the previous draft of this kernel got it wrong:
//   1. used `rotary_dim` as the frequency denominator instead of
//      `head_dim` — wrong angle progression.
//   2. used `rotary_dim/2` as the pair offset instead of
//      `head_dim/2` — paired the wrong dims with each other.
__global__ void rope_partial_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int position_delta,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int pos = positions[n] + position_delta;

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        float cos_v = 1.f, sin_v = 0.f;
        if (dim_pair < rope_angles) {
            const float freq = powf(theta,
                -2.f * static_cast<float>(dim_pair) /
                       static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        // Skip identity rotations entirely — `dim_pair ≥ rope_angles`
        // multiplies the pair by [[1,0],[0,1]] which is a no-op.
        if (dim_pair >= rope_angles) continue;

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = __bfloat162float(qp[dim_pair]);
            const float b = __bfloat162float(qp[dim_pair + half]);
            qp[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
            qp[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = __bfloat162float(kp[dim_pair]);
            const float b = __bfloat162float(kp[dim_pair + half]);
            kp[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
            kp[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
        }
    }
}

}  // namespace

void rope_partial_bf16(void* q, void* k, const std::int32_t* positions,
                       int num_tokens, int num_q_heads, int num_kv_heads,
                       int head_dim, int rotary_dim, float theta,
                       cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    rope_partial_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        0,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
}

}  // namespace pie_cuda_device::kernels
