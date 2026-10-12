#pragma once

#include "prelude/device.cuh"
#include "prelude/kv_paged_addr.cuh"

namespace pie::attn {

template <int HEAD_DIM_MAX, int WARPS, bool HND_LAYOUT>
__global__ void score_capture(
    const bf16* __restrict__ q,
    const i32* __restrict__ qo_indptr,
    const bf16* __restrict__ k_pages,
    const i32* __restrict__ kv_page_indices,
    const i32* __restrict__ kv_page_indptr,
    const i32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    int observe,
    int lane_offset,
    int plane_stride,
    int plane,
    int kv_max)
{
    constexpr int VPT = HEAD_DIM_MAX / 32;

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int threads = static_cast<int>(blockDim.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    extern __shared__ float smem[];
    float* q_s = smem;
    float* wm = q_s + head_dim;
    float* wl = wm + WARPS;

    const long long out_row =
        (static_cast<long long>(lane_offset + request) * plane_stride + plane + head) *
        static_cast<long long>(kv_max);
    float* out = scores + out_row;

    for (int i = static_cast<int>(threadIdx.x); i < kv_max; i += threads) {
        out[i] = 0.f;
    }

    const int page_first = static_cast<int>(kv_page_indptr[request]);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) - page_first;
    const int kv_len =
        pages > 0 ? (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request])
                  : 0;
    const int qo_lo = static_cast<int>(qo_indptr[request]);
    const int qo_hi = static_cast<int>(qo_indptr[request + 1]);
    const int qo_len = qo_hi - qo_lo;
    const int rows = observe < qo_len ? observe : qo_len;

    if (pages <= 0 || kv_len <= 0 || rows <= 0) {
        return;
    }

    const int group = num_q_heads / num_kv_heads;
    const int kv_head = head / group;
    const float inv_rows = 1.f / static_cast<float>(rows);

    for (int w = 0; w < rows; ++w) {

        const int q_index = qo_hi - rows + w;
        const int causal = kv_len - rows + w + 1;
        const int limit = causal < kv_len ? causal : kv_len;

        if (limit <= 0) {
            continue;
        }

        const bf16* q_row =
            q + (static_cast<long long>(q_index) * num_q_heads + head) * head_dim;
        for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += threads) {
            q_s[d] = bf16_to_f32(q_row[d]);
        }
        __syncthreads();

        float running_max = neg_inf();
        float running_sum = 0.f;
        for (int j = warp; j < limit; j += WARPS) {
            const int page_in_req = j / page_size;
            KvSlot slot;
            slot.page = static_cast<int>(kv_page_indices[page_first + page_in_req]);
            slot.offset_in_page = j - page_in_req * page_size;
            const bf16* k_row =
                k_pages + kv_dst_index<HND_LAYOUT>(slot, kv_head * head_dim, page_size,
                                                   num_kv_heads, head_dim);
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
            running_sum = running_sum * __expf(running_max - widened) +
                          __expf(score - widened);
            running_max = widened;
        }
        if (lane == 0) {
            wm[warp] = running_max;
            wl[warp] = running_sum;
        }
        __syncthreads();

        float folded_max = neg_inf();
#pragma unroll
        for (int u = 0; u < WARPS; ++u) {
            folded_max = fmaxf(folded_max, wm[u]);
        }
        float denominator = 0.f;
#pragma unroll
        for (int u = 0; u < WARPS; ++u) {

            denominator += wl[u] * __expf(wm[u] - folded_max);
        }
        const float inv = denominator > 0.f ? 1.f / denominator : 0.f;

        for (int j = warp; j < limit; j += WARPS) {
            const int page_in_req = j / page_size;
            KvSlot slot;
            slot.page = static_cast<int>(kv_page_indices[page_first + page_in_req]);
            slot.offset_in_page = j - page_in_req * page_size;
            const bf16* k_row =
                k_pages + kv_dst_index<HND_LAYOUT>(slot, kv_head * head_dim, page_size,
                                                   num_kv_heads, head_dim);
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

            if (lane == 0 && j < kv_max) {
                out[j] += __expf(dot * sm_scale - folded_max) * inv * inv_rows;
            }
        }

        __syncthreads();
    }
}

}
