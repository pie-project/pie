#include "rope.cuh"

#include <cuda_bf16.h>

// Lifted verbatim from driver/cuda/src/kernels/rope.cu (base variant:
// rotate_pair / rotate_pair_interleaved + rope_bf16_kernel + launcher).

namespace pie_cuda_device::kernels {

namespace {

// NeoX (Llama/Qwen) pairing: index `i` with `i + head_dim/2`.
__device__ __forceinline__ void rotate_pair(
    __nv_bfloat16* h_ptr, int half, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(h_ptr[dim_pair]);
    const float b = __bfloat162float(h_ptr[dim_pair + half]);
    h_ptr[dim_pair]        = __float2bfloat16(a * cos_v - b * sin_v);
    h_ptr[dim_pair + half] = __float2bfloat16(b * cos_v + a * sin_v);
}

// GPT-J / interleaved pairing: adjacent dims (2i, 2i+1) — GLM.
__device__ __forceinline__ void rotate_pair_interleaved(
    __nv_bfloat16* h_ptr, int dim_pair, float cos_v, float sin_v)
{
    const float a = __bfloat162float(h_ptr[2 * dim_pair]);
    const float b = __bfloat162float(h_ptr[2 * dim_pair + 1]);
    h_ptr[2 * dim_pair]     = __float2bfloat16(a * cos_v - b * sin_v);
    h_ptr[2 * dim_pair + 1] = __float2bfloat16(b * cos_v + a * sin_v);
}

__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    bool interleaved)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[n];

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
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cos_v, sin_v);
            else rotate_pair(qp, half, dim_pair, cos_v, sin_v);
            continue;
        }
        {
            const int kv_h = head_idx - num_q_heads;
            __nv_bfloat16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cos_v, sin_v);
            else rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

}  // namespace

void rope_bf16(void* q, void* k, const std::int32_t* positions,
               int num_tokens, int num_q_heads, int num_kv_heads,
               int head_dim, float theta, bool interleaved, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    rope_bf16_kernel<<<dim3(num_tokens), dim3(BLOCK), 0, stream>>>(
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        positions,
        num_q_heads, num_kv_heads, head_dim, theta, interleaved);
}

}  // namespace pie_cuda_device::kernels
