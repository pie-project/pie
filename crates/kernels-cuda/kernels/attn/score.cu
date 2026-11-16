#include "prim/device.cuh"
#include "prim/paged_kv.cuh"

namespace pie::attn {

// A capturing lane's score row: the attention mass each key of the load's
// context takes from the lane's last `observe` query rows, averaged over the
// rows and over the model's query heads (`share` = 1 / q_heads, every
// rank's), added to the row (a layer before's) or to a zeroed one. A block a
// lane, its heads one after another, so each key's sum runs in one order.
// The row lies where the lane's entry of `rows` says (0: the lane captures
// none): a recorded launch keeps the table, and the rows move with the
// instances that hold them.
template <int HEAD_DIM_MAX, int WARPS, bool HND_LAYOUT>
__global__ void score_capture(
    const bf16* __restrict__ q,
    const i32* __restrict__ qo_indptr,
    const bf16* __restrict__ k_pages,
    const i32* __restrict__ kv_page_indices,
    const i32* __restrict__ kv_page_indptr,
    const i32* __restrict__ kv_last_page_lens,
    const unsigned long long* __restrict__ rows_at,
    int page_size,
    long long stride_page,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    int observe,
    int lane_offset,
    int context,
    float share,
    int accumulate)
{
    constexpr int VPT = HEAD_DIM_MAX / 32;

    const int request = static_cast<int>(blockIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    const unsigned long long row_at = rows_at[lane_offset + request];
    if (row_at == 0ull) {
        return;
    }
    float* const out = reinterpret_cast<float*>(row_at);

    extern __shared__ float smem[];
    float* q_s = smem;
    float* wm = q_s + head_dim;
    float* wl = wm + WARPS;

    if (accumulate == 0) {
        for (int i = static_cast<int>(threadIdx.x); i < context; i += threads) {
            out[i] = 0.f;
        }
    }
    __syncthreads();

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
    const float weight = share / static_cast<float>(rows);

    for (int head = 0; head < num_q_heads; ++head) {
        const int kv_head = head / group;
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
                                                       num_kv_heads, head_dim, stride_page);
                float dot = 0.f;
#pragma unroll
                for (int u = 0; u < VPT; ++u) {
                    const int d = lane + u * 32;
                    if (d < head_dim) {
                        dot += q_s[d] * bf16_to_f32(k_row[d]);
                    }
                }

                dot = ::pie::warp_sum(dot);
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

            // A key's sum is its warp's lane 0's, the same thread every row
            // and head: one order, no atomics.
            for (int j = warp; j < limit; j += WARPS) {
                const int page_in_req = j / page_size;
                KvSlot slot;
                slot.page = static_cast<int>(kv_page_indices[page_first + page_in_req]);
                slot.offset_in_page = j - page_in_req * page_size;
                const bf16* k_row =
                    k_pages + kv_dst_index<HND_LAYOUT>(slot, kv_head * head_dim, page_size,
                                                       num_kv_heads, head_dim, stride_page);
                float dot = 0.f;
#pragma unroll
                for (int u = 0; u < VPT; ++u) {
                    const int d = lane + u * 32;
                    if (d < head_dim) {
                        dot += q_s[d] * bf16_to_f32(k_row[d]);
                    }
                }
                dot = ::pie::warp_sum(dot);

                if (lane == 0 && j < context) {
                    out[j] += __expf(dot * sm_scale - folded_max) * inv * weight;
                }
            }

            __syncthreads();
        }
    }
}

// Each capturing lane's key row summed a kv page: the page row lies right
// after the key row (`context` floats on). A thread a page, its keys in
// order.
__global__ void score_pages(
    const unsigned long long* __restrict__ rows_at,
    int lane_offset,
    int context,
    int page_size)
{
    const unsigned long long row_at = rows_at[lane_offset + static_cast<int>(blockIdx.x)];
    if (row_at == 0ull) {
        return;
    }
    const float* keys = reinterpret_cast<const float*>(row_at);
    float* pages = reinterpret_cast<float*>(row_at) + context;
    const int count = (context + page_size - 1) / page_size;
    for (int p = static_cast<int>(threadIdx.x); p < count; p += static_cast<int>(blockDim.x)) {
        const int lo = p * page_size;
        const int hi = lo + page_size < context ? lo + page_size : context;
        float sum = 0.f;
        for (int k = lo; k < hi; ++k) {
            sum += keys[k];
        }
        pages[p] = sum;
    }
}

}
