#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

template <int HEAD_DIM_MAX, int WARPS>
__global__ void dense_bidirectional(
    const bf16* __restrict__ q,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    bf16* __restrict__ o,
    const i32* __restrict__ segments,
    int num_segments,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale)
{
    constexpr int VPT = HEAD_DIM_MAX / 32;
    const int threads = WARPS * 32;
    const float neg_inf = __int_as_float(0xff800000u);

    const int row = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    extern __shared__ float smem[];
    float* q_s = smem;
    float* wacc = q_s + head_dim;
    float* wm = wacc + WARPS * head_dim;
    float* wl = wm + WARPS;

    __shared__ int span[2];

    if (threadIdx.x == 0) {
        int begin = -1;
        int end = -1;
        const int first = segments[0];
        const int total = segments[num_segments];
        if (row >= first && row < total) {
            int lo = 0;
            int hi = num_segments - 1;
            while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (segments[mid] <= row) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            begin = segments[lo];
            end = segments[lo + 1];
        }
        span[0] = begin;
        span[1] = end;
    }
    __syncthreads();

    const int begin = span[0];
    const int end = span[1];

    bf16* out = o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    if (end <= begin) {

        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
            out[d] = f32_to_bf16(0.f);
        }
        return;
    }

    const bf16* q_row = q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
        q_s[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = head / group;

    float acc[VPT];
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        acc[u] = 0.f;
    }
    float running_max = neg_inf;
    float running_sum = 0.f;

    for (int j = begin + warp; j < end; j += WARPS) {
        const bf16* k_row =
            k + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.f;
#pragma unroll
        for (int u = 0; u < VPT; ++u) {
            const int d = lane + u * 32;
            if (d < head_dim) {
                dot += q_s[d] * bf16_to_f32(k_row[d]);
            }
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        }

        const float score = dot * sm_scale;
        const float widened = fmaxf(running_max, score);
        const float rescale = __expf(running_max - widened);
        const float weight = __expf(score - widened);

        const bf16* v_row =
            v + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;
#pragma unroll
        for (int u = 0; u < VPT; ++u) {
            const int d = lane + u * 32;
            if (d < head_dim) {
                acc[u] = acc[u] * rescale + weight * bf16_to_f32(v_row[d]);
            }
        }
        running_sum = running_sum * rescale + weight;
        running_max = widened;
    }

    if (lane == 0) {
        wm[warp] = running_max;
        wl[warp] = running_sum;
    }
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        const int d = lane + u * 32;
        if (d < head_dim) {
            wacc[warp * head_dim + d] = acc[u];
        }
    }
    __syncthreads();

    float folded_max = neg_inf;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        folded_max = fmaxf(folded_max, wm[w]);
    }
    float denominator = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        denominator += wl[w] * __expf(wm[w] - folded_max);
    }
    const float inv = denominator > 0.f ? 1.f / denominator : 0.f;

    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            sum += wacc[w * head_dim + d] * __expf(wm[w] - folded_max);
        }
        out[d] = f32_to_bf16(sum * inv);
    }
}

}
