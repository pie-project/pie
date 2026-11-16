#pragma once

// The attention mass a capturing lane's last `observe` rows put on each key
// (eviction scores): share / rows Σ_w softmax_j(q_w·k_j·scale), causal, summed
// over the query heads into the lane's own row (a row a capturing lane, in
// lane order; `lanes` names each row's lane in the fire). A block a
// (capturing row, head); wavefronts take keys round-robin, two passes a row
// (the softmax's max and sum, then the mass); the heads meet in the row by
// f32 atomics.

#include "prim/device.hip"
#include "attn/kv.hip"

namespace pie::attn {

// The request of this window capturing row `row` holds, or -1.
__device__ __forceinline__ int captured_request(const i32* __restrict__ lanes, int row,
                                                int lane_offset, int requests) {
    const int request = static_cast<int>(lanes[row]) - lane_offset;
    return request >= 0 && request < requests ? request : -1;
}

__global__ void score_zero(float* __restrict__ scores, const i32* __restrict__ lanes,
                           int lane_offset, int requests, int context) {
    const int row = static_cast<int>(blockIdx.x);
    if (captured_request(lanes, row, lane_offset, requests) < 0) return;
    float* out = scores + static_cast<long long>(row) * context;
    for (int i = threadIdx.x; i < context; i += blockDim.x) out[i] = 0.f;
}

template <int HEAD_DIM_MAX, int WAVES, bool HND>
__global__ __launch_bounds__(WAVES * 64) void score_capture(
    const bf16* __restrict__ q, const i32* __restrict__ qo_indptr, const bf16* __restrict__ k_pages,
    const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, float* __restrict__ scores,
    const i32* __restrict__ lanes, int lane_offset, int requests, int page_size, long long stride,
    int num_q_heads, int num_kv_heads, int head_dim, float sm_scale, float share, int observe,
    int context) {
    static_assert(HEAD_DIM_MAX % kWave == 0, "a lane takes whole elements of the head");
    constexpr int VPT = HEAD_DIM_MAX / kWave;
    extern __shared__ float smem[];
    float* q_s = smem;
    float* wm = q_s + head_dim;
    float* wl = wm + WAVES;

    const int row = static_cast<int>(blockIdx.x);
    const int request = captured_request(lanes, row, lane_offset, requests);
    if (request < 0) return;
    const int head = static_cast<int>(blockIdx.y);
    const int ln = lane();
    const int wave = static_cast<int>(threadIdx.x) / kWave;

    float* out = scores + static_cast<long long>(row) * context;

    const int page_first = page_indptr[request];
    const int pages = page_indptr[request + 1] - page_first;
    const int kv_len = pages > 0 ? (pages - 1) * page_size + last_page_lens[request] : 0;
    const int qo_hi = qo_indptr[request + 1];
    const int rows = min(observe, qo_hi - qo_indptr[request]);
    if (pages <= 0 || kv_len <= 0 || rows <= 0) return;

    const int kv_head = head / (num_q_heads / num_kv_heads);
    const float weight = share / static_cast<float>(rows);

    auto dot = [&](int j) {
        const long long page = page_indices[page_first + j / page_size];
        const bf16* k_row = k_pages + kv_at<HND>(page, j % page_size, kv_head, 0, page_size, num_kv_heads, head_dim, stride);
        float acc = 0.f;
#pragma unroll
        for (int u = 0; u < VPT; ++u) {
            const int d = ln + u * kWave;
            if (d < head_dim) acc += q_s[d] * to_f32(k_row[d]);
        }
        return wave_sum(acc) * sm_scale;
    };

    for (int w = 0; w < rows; ++w) {
        const int limit = min(kv_len - rows + w + 1, kv_len);
        if (limit <= 0) continue;

        const bf16* q_row = q + (static_cast<long long>(qo_hi - rows + w) * num_q_heads + head) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) q_s[d] = to_f32(q_row[d]);
        __syncthreads();

        float m = neg_inf(), l = 0.f;
        for (int j = wave; j < limit; j += WAVES) {
            const float s = dot(j);
            const float widened = fmaxf(m, s);
            l = l * __expf(m - widened) + __expf(s - widened);
            m = widened;
        }
        if (ln == 0) {
            wm[wave] = m;
            wl[wave] = l;
        }
        __syncthreads();

        float big = neg_inf();
#pragma unroll
        for (int u = 0; u < WAVES; ++u) big = fmaxf(big, wm[u]);
        float denominator = 0.f;
#pragma unroll
        for (int u = 0; u < WAVES; ++u) denominator += wl[u] * __expf(wm[u] - big);
        const float inv = denominator > 0.f ? weight / denominator : 0.f;

        for (int j = wave; j < limit && j < context; j += WAVES) {
            const float s = dot(j);
            if (ln == 0) atomicAdd(&out[j], __expf(s - big) * inv);
        }
        __syncthreads();
    }
}

// Each capturing lane's key row summed a kv page: a block a row, a thread a
// page.
__global__ void score_pages(const float* __restrict__ keys, float* __restrict__ pages,
                            const i32* __restrict__ lanes, int lane_offset, int requests,
                            int context, int page_size, int num_pages) {
    const int row = static_cast<int>(blockIdx.x);
    if (captured_request(lanes, row, lane_offset, requests) < 0) return;
    const float* in = keys + static_cast<long long>(row) * context;
    float* out = pages + static_cast<long long>(row) * num_pages;
    for (int p = threadIdx.x; p < num_pages; p += blockDim.x) {
        const int lo = p * page_size;
        const int hi = min(lo + page_size, context);
        float sum = 0.f;
        for (int j = lo; j < hi; ++j) sum += in[j];
        out[p] = sum;
    }
}

}  // namespace pie::attn
