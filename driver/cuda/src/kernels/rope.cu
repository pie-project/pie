#include "kernels/rope.hpp"

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

// One block per token; threads cover the full QK head_dim grid:
// (head, dim_pair_idx). For Qwen the convention pairs index `i` with
// `i + head_dim/2`, with frequency theta^(-2*i / head_dim).
__device__ __forceinline__ void rotate_pair(
    __nv_bfloat16* h_ptr, int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(h_ptr[dim_pair]);
    const float b = __bfloat162float(h_ptr[dim_pair + half]);
    h_ptr[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
    h_ptr[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
}

__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[n];

    // Each thread handles one (head, dim_pair_idx).
    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                       static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            rotate_pair(qp, half, dim_pair, cos_v, sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

}  // namespace

void launch_rope_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    rope_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta);
}

// ── YaRN variant ────────────────────────────────────────────────────────────

namespace {

// Piecewise-linear interp between full-scale (high-freq pairs, kept
// untouched) and `factor`-scaled (low-freq pairs); smooth band uses
// `(orig_max_pos / wavelen - low_freq_factor) / (high - low)` blended.
__device__ __forceinline__ float yarn_freq(
    float base_freq, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos)
{
    constexpr float TWO_PI = 6.2831853071795864769f;
    const float wavelen   = TWO_PI / base_freq;
    const float low_wave  = orig_max_pos / low_freq_factor;
    const float high_wave = orig_max_pos / high_freq_factor;
    if (wavelen < high_wave) return base_freq;            // high-freq: no scale
    if (wavelen > low_wave)  return base_freq / factor;   // low-freq: full scale
    const float smooth = (orig_max_pos / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor);
    return (1.f - smooth) * (base_freq / factor) + smooth * base_freq;
}

__global__ void rope_yarn_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        const float base_freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float freq = yarn_freq(base_freq, factor,
                                     low_freq_factor, high_freq_factor,
                                     orig_max_pos);
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            __nv_bfloat16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            rotate_pair(qp, half, dim_pair, cos_v, sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

}  // namespace

void launch_rope_yarn_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    int original_max_position,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    rope_yarn_bf16_kernel<<<num_tokens, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim,
        theta, factor, low_freq_factor, high_freq_factor,
        static_cast<float>(original_max_position));
}

// ── Partial rotary (Gemma-4 full-attention layers) ─────────────────────────

namespace {

// Same as `rope_bf16_kernel` but rotates only `rotary_dim` of each head's
// `head_dim` channels. Pairs are (i, i + rotary_dim/2). Channels in
// `[rotary_dim, head_dim)` are left untouched.
__global__ void rope_partial_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = rotary_dim / 2;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

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

void launch_rope_partial_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    dim3 grid(num_tokens);
    dim3 block(BLOCK);
    rope_partial_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, rotary_dim, theta);
}

}  // namespace pie_cuda_driver::kernels
