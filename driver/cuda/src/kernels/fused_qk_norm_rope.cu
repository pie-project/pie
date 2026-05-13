#include "kernels/fused_qk_norm_rope.hpp"

#include <cuda_bf16.h>
#include <cub/block/block_reduce.cuh>

namespace pie_cuda_driver::kernels {

namespace {

// One block per (token, head). Block dim = head_dim threads. Shared
// memory carries one float per thread for the normalized vector,
// which RoPE reads pairwise.
template <int BLOCK>
__global__ void fused_qk_norm_rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const std::int32_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    int N,
    int num_q_heads, int num_kv_heads,
    int head_dim,
    float eps, float rope_theta)
{
    extern __shared__ float s_normed[];  // [head_dim] working buffer

    const int total_heads = num_q_heads + num_kv_heads;
    const int n         = blockIdx.x;
    const int head_idx  = blockIdx.y;
    if (n >= N || head_idx >= total_heads) return;

    const bool is_q = (head_idx < num_q_heads);
    __nv_bfloat16* dst;
    const __nv_bfloat16* weight;
    if (is_q) {
        dst = q + (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
        weight = q_norm_w;
    } else {
        const int kv_h = head_idx - num_q_heads;
        dst = k + (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
        weight = k_norm_w;
    }

    const int tid = threadIdx.x;
    const int pos = positions[n];

    // 1. Load element + accumulate variance partial.
    float x = 0.f;
    if (tid < head_dim) {
        x = __bfloat162float(dst[tid]);
    }
    float var = x * x;

    using BlockReduce = cub::BlockReduce<float, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    var = BlockReduce(temp_storage).Sum(var);

    __shared__ float s_inv_std;
    if (tid == 0) {
        s_inv_std = rsqrtf(var / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    // 2. Apply RMSNorm + weight, stash in shared.
    if (tid < head_dim) {
        const float w = __bfloat162float(weight[tid]);
        s_normed[tid] = x * s_inv_std * w;
    }
    __syncthreads();

    // 3. Pair-wise RoPE. Threads with tid < head_dim/2 handle one pair.
    const int half = head_dim / 2;
    if (tid < half) {
        const float freq = powf(rope_theta,
            -2.f * static_cast<float>(tid) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        const float a = s_normed[tid];
        const float b = s_normed[tid + half];
        dst[tid]        = __float2bfloat16(a * cos_v - b * sin_v);
        dst[tid + half] = __float2bfloat16(b * cos_v + a * sin_v);
    }
}

}  // namespace

void launch_fused_qk_norm_rope_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    const void* q_norm_weight, const void* k_norm_weight,
    int N, int num_q_heads, int num_kv_heads, int head_dim,
    float eps, float rope_theta,
    cudaStream_t stream)
{
    if (N <= 0) return;
    // head_dim is one of {64, 128, 256} across all archs that hit this
    // kernel — pick a block size that fits and dispatch the matching
    // template specialisation. Each block has `head_dim` threads.
    dim3 grid(N, num_q_heads + num_kv_heads);
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(float);

    auto* qp = static_cast<__nv_bfloat16*>(q);
    auto* kp = static_cast<__nv_bfloat16*>(k);
    const auto* qw = static_cast<const __nv_bfloat16*>(q_norm_weight);
    const auto* kw = static_cast<const __nv_bfloat16*>(k_norm_weight);

    if (head_dim == 128) {
        fused_qk_norm_rope_bf16_kernel<128><<<grid, 128, smem, stream>>>(
            qp, kp, positions, qw, kw,
            N, num_q_heads, num_kv_heads, head_dim, eps, rope_theta);
    } else if (head_dim == 64) {
        fused_qk_norm_rope_bf16_kernel<64><<<grid, 64, smem, stream>>>(
            qp, kp, positions, qw, kw,
            N, num_q_heads, num_kv_heads, head_dim, eps, rope_theta);
    } else if (head_dim == 256) {
        fused_qk_norm_rope_bf16_kernel<256><<<grid, 256, smem, stream>>>(
            qp, kp, positions, qw, kw,
            N, num_q_heads, num_kv_heads, head_dim, eps, rope_theta);
    } else {
        // Unsupported head_dim — caller should fall back to the
        // separate launches. We don't throw here (CUDA can't propagate
        // exceptions from a __global__); the caller's predicate filters
        // these cases out.
    }
}

}  // namespace pie_cuda_driver::kernels
