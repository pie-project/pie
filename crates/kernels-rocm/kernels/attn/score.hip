#pragma once

// The attention mass a capturing lane's last `observe` rows put on each key
// of the load's context (eviction scores), averaged over the rows and the
// model's query heads and added into the lane's key row: (1 / (rows * heads))
// Σ_h Σ_w softmax_j(q_hw·k_j·scale), causal. A block a lane walks its heads
// in order; wavefronts take keys round-robin, two passes a row (the
// softmax's max and sum, then the mass).

#include "prim/device.hip"
#include "attn/kv.hip"

namespace pie::attn {

template <int HEAD_DIM_MAX, int WAVES, bool HND>
__global__ __launch_bounds__(WAVES * 64) void score_capture(
    const bf16* __restrict__ q, const i32* __restrict__ qo_indptr, const bf16* __restrict__ k_pages,
    const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ last_page_lens, const unsigned long long* __restrict__ keys_at,
    int page_size, int num_q_heads, int num_kv_heads, int head_dim, float sm_scale, int observe,
    int lane_offset, int context, int model_heads, int zero, long long stride) {
    static_assert(HEAD_DIM_MAX % kWave == 0, "a lane takes whole elements of the head");
    constexpr int VPT = HEAD_DIM_MAX / kWave;
    extern __shared__ float smem[];
    float* q_s = smem;
    float* wm = q_s + head_dim;
    float* wl = wm + WAVES;

    const int request = static_cast<int>(blockIdx.x);
    const int ln = lane();
    const int wave = static_cast<int>(threadIdx.x) / kWave;

    // Each lane's key row lies where its reader keeps it; 0: no reader, no row.
    const unsigned long long row = keys_at[lane_offset + request];
    if (row == 0ull) return;
    float* out = reinterpret_cast<float*>(row);
    if (zero) {
        for (int i = threadIdx.x; i < context; i += blockDim.x) out[i] = 0.f;
        __syncthreads();
    }

    const int page_first = page_indptr[request];
    const int pages = page_indptr[request + 1] - page_first;
    const int kv_len = pages > 0 ? (pages - 1) * page_size + last_page_lens[request] : 0;
    const int qo_hi = qo_indptr[request + 1];
    const int rows = min(observe, qo_hi - qo_indptr[request]);
    if (pages <= 0 || kv_len <= 0 || rows <= 0) return;

    const int group = num_q_heads / num_kv_heads;
    const float share = 1.f / (static_cast<float>(rows) * static_cast<float>(model_heads));

    for (int head = 0; head < num_q_heads; ++head) {
        const int kv_head = head / group;
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
            const float inv = denominator > 0.f ? 1.f / denominator : 0.f;

            // A key's lane-0 thread is the same thread every row and head,
            // so its sum needs no ordering across them.
            for (int j = wave; j < limit; j += WAVES) {
                const float s = dot(j);
                if (ln == 0 && j < context) out[j] += __expf(s - big) * inv * share;
            }
            __syncthreads();
        }
    }
}

// A capturing lane's key row summed a page of `page_size` keys into its page
// row; a block a lane, a thread a page. 0 in `keys_at`: no reader, no row.
__global__ void score_pages(const unsigned long long* __restrict__ keys_at,
                            const unsigned long long* __restrict__ pages_at, int lane_offset,
                            int context, int page_size) {
    const int at = lane_offset + static_cast<int>(blockIdx.x);
    const unsigned long long keys = keys_at[at];
    const unsigned long long pages = pages_at[at];
    if (keys == 0ull || pages == 0ull) return;
    const float* key_row = reinterpret_cast<const float*>(keys);
    float* page_row = reinterpret_cast<float*>(pages);
    const int count = (context + page_size - 1) / page_size;
    for (int p = threadIdx.x; p < count; p += blockDim.x) {
        const int first = p * page_size;
        const int last = min(first + page_size, context);
        float sum = 0.f;
        for (int k = first; k < last; ++k) sum += key_row[k];
        page_row[p] = sum;
    }
}

}  // namespace pie::attn
