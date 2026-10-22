#pragma once

// The attention mass a request's last `observe` rows put on each key
// (eviction scores): (1 / rows) Σ_w softmax_j(q_w·k_j·scale), causal. A block
// a (request, head); wavefronts take keys round-robin, two passes a row (the
// softmax's max and sum, then the mass).

#include "prelude/device.hip"
#include "attn/kv.hip"

namespace pie::attn {

template <int HEAD_DIM_MAX, int WAVES, bool HND>
__global__ __launch_bounds__(WAVES * 64) void score_capture(
    const bf16* __restrict__ q, const i32* __restrict__ qo_indptr, const bf16* __restrict__ k_pages,
    const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, float* __restrict__ scores,
    const i32* __restrict__ seat_of, int page_size,
    int num_q_heads, int num_kv_heads, int head_dim, float sm_scale, int observe, int lane_offset,
    int plane_stride, int plane, int kv_max) {
    static_assert(HEAD_DIM_MAX % kWave == 0, "a lane takes whole elements of the head");
    constexpr int VPT = HEAD_DIM_MAX / kWave;
    extern __shared__ float smem[];
    float* q_s = smem;
    float* wm = q_s + head_dim;
    float* wl = wm + WAVES;

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int ln = lane();
    const int wave = static_cast<int>(threadIdx.x) / kWave;

    const int seat = seat_of[lane_offset + request];
    if (seat < 0) return;
    float* out = scores + (static_cast<long long>(seat) * plane_stride + plane + head) *
                              static_cast<long long>(kv_max);
    for (int i = threadIdx.x; i < kv_max; i += blockDim.x) out[i] = 0.f;

    const int page_first = page_indptr[request];
    const int pages = page_indptr[request + 1] - page_first;
    const int kv_len = pages > 0 ? (pages - 1) * page_size + last_page_lens[request] : 0;
    const int qo_hi = qo_indptr[request + 1];
    const int rows = min(observe, qo_hi - qo_indptr[request]);
    if (pages <= 0 || kv_len <= 0 || rows <= 0) return;

    const int kv_head = head / (num_q_heads / num_kv_heads);
    const float inv_rows = 1.f / static_cast<float>(rows);

    auto dot = [&](int j) {
        const long long page = page_indices[page_first + j / page_size];
        const bf16* k_row = k_pages + kv_at<HND>(page, j % page_size, kv_head, 0, page_size, num_kv_heads, head_dim);
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
        const float inv = denominator > 0.f ? 1.f / denominator : 0.f;

        // A key's lane-0 thread is the same thread every row, so its sum
        // needs no ordering across rows.
        for (int j = wave; j < limit; j += WAVES) {
            const float s = dot(j);
            if (ln == 0 && j < kv_max) out[j] += __expf(s - big) * inv * inv_rows;
        }
        __syncthreads();
    }
}

}  // namespace pie::attn
