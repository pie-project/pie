#pragma once


#include "prelude/device.cuh"
#include "spatial/grid.cuh"

namespace pie::spatial {

__device__ __forceinline__ void welford_merge(
    float& cnt,
    float& mean,
    float& m2,
    float cnt_b,
    float mean_b,
    float m2_b)
{
    const float n = cnt + cnt_b;
    if (n == 0.f) return;
    const float delta = mean_b - mean;
    const float f = cnt_b / n;
    mean += delta * f;
    m2 += m2_b + delta * delta * cnt * f;
    cnt = n;
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void group_norm_stats(
    const bf16* __restrict__ x,
    const int* __restrict__ grid,
    float4* __restrict__ partials,
    int c,
    int groups,
    int splits)
{
    __shared__ float s_cnt[BLOCK];
    __shared__ float s_mean[BLOCK];
    __shared__ float s_m2[BLOCK];

    const int split = blockIdx.x;
    const int l = blockIdx.y;
    const Lane g = lane_at(grid, l);
    const int n = g.voxels();
    const int chunk = (n + splits - 1) / splits;
    const int begin = split * chunk;
    const int end = min(n, begin + chunk);
    const int tid = threadIdx.x;

    float cnt = 0.f;
    float mean = 0.f;
    float m2 = 0.f;
    if (tid < c) {
        const bf16* at = x + static_cast<long long>(g.off + begin) * c + tid;
        for (int v = begin; v < end; ++v, at += c) {
            const float val = bf16_to_f32(ldg(at));
            cnt += 1.f;
            const float d = val - mean;
            mean += d / cnt;
            m2 += d * (val - mean);
        }
    }
    s_cnt[tid] = cnt;
    s_mean[tid] = mean;
    s_m2[tid] = m2;
    __syncthreads();

    if (tid < groups) {
        const int cg = c / groups;
        float gcnt = 0.f;
        float gmean = 0.f;
        float gm2 = 0.f;
        for (int i = tid * cg; i < (tid + 1) * cg; ++i) {
            welford_merge(gcnt, gmean, gm2, s_cnt[i], s_mean[i], s_m2[i]);
        }
        partials[(static_cast<long long>(l) * splits + split) * groups + tid] =
            make_float4(gcnt, gmean, gm2, 0.f);
    }
}

__global__ __launch_bounds__(32) void group_norm_finalize(
    const float4* __restrict__ partials,
    float2* __restrict__ stats,
    int groups,
    int splits,
    float eps)
{
    const int l = blockIdx.y;
    const int group = blockIdx.x;
    const int lane = threadIdx.x;
    float cnt = 0.f;
    float mean = 0.f;
    float m2 = 0.f;
    for (int s = lane; s < splits; s += 32) {
        const float4 p = partials[(static_cast<long long>(l) * splits + s) * groups + group];
        welford_merge(cnt, mean, m2, p.x, p.y, p.z);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float o_cnt = __shfl_xor_sync(0xffffffffu, cnt, off);
        const float o_mean = __shfl_xor_sync(0xffffffffu, mean, off);
        const float o_m2 = __shfl_xor_sync(0xffffffffu, m2, off);
        welford_merge(cnt, mean, m2, o_cnt, o_mean, o_m2);
    }
    if (lane == 0) {
        const float var = cnt > 0.f ? m2 / cnt : 0.f;
        stats[l * groups + group] = make_float2(mean, rsqrtf(var + eps));
    }
}

template <bool SILU>
__global__ __launch_bounds__(256) void group_norm_apply(
    const bf16* __restrict__ x,
    const int* __restrict__ grid,
    const float2* __restrict__ stats,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    bf16* __restrict__ y,
    int c,
    int groups,
    int lanes,
    long long total)
{
    const long long e = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= total) return;
    const int row = static_cast<int>(e / c);
    const int col = static_cast<int>(e - static_cast<long long>(row) * c);
    Lane box;
    const int l = lane_of(grid, lanes, row, box);
    float v = 0.f;
    if (l >= 0) {
        const int group = col / (c / groups);
        const float2 st = stats[l * groups + group];
        v = fmaf((bf16_to_f32(ldg(x + e)) - st.x) * st.y, __ldg(weight + col), __ldg(bias + col));
        if constexpr (SILU) {
            v = v / (1.f + __expf(-v));
        }
    }
    y[e] = f32_to_bf16(v);
}

}
