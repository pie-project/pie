#include "prim/device.cuh"
#include "prim/paged_kv.cuh"

namespace pie::attn {

// The request of this window capturing row `row` holds, or -1: `lanes`
// names each capturing row's lane in the fire, and the window's requests are
// the fire's lanes from `lane_offset`.
__device__ __forceinline__ int captured_request(const i32* __restrict__ lanes, int row,
                                                int lane_offset, int requests)
{
    const int request = static_cast<int>(lanes[row]) - lane_offset;
    return request >= 0 && request < requests ? request : -1;
}

// Zeroes the row of each capturing lane of this window: a block a row.
__global__ void score_zero(
    float* __restrict__ scores,
    const i32* __restrict__ lanes,
    int lane_offset,
    int requests,
    int context)
{
    const int row = static_cast<int>(blockIdx.x);
    if (captured_request(lanes, row, lane_offset, requests) < 0) {
        return;
    }
    float* out = scores + static_cast<long long>(row) * context;
    for (int i = static_cast<int>(threadIdx.x); i < context; i += static_cast<int>(blockDim.x)) {
        out[i] = 0.f;
    }
}

// Adds to each capturing lane's row the attention mass its query head
// `blockIdx.y` puts on each key over the lane's last `observe` query rows,
// times `share` (one over the model's query heads) over the rows observed:
// a block a (capturing row, query head), its heads meeting in the row by
// f32 atomics, so the row is summed over the heads in the pass.
template <int HEAD_DIM_MAX, int WARPS, bool HND_LAYOUT>
__global__ void score_capture(
    const bf16* __restrict__ q,
    const i32* __restrict__ qo_indptr,
    const bf16* __restrict__ k_pages,
    const i32* __restrict__ kv_page_indices,
    const i32* __restrict__ kv_page_indptr,
    const i32* __restrict__ kv_last_page_lens,
    float* __restrict__ scores,
    const i32* __restrict__ lanes,
    int lane_offset,
    int requests,
    int page_size,
    long long stride_page,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float sm_scale,
    float share,
    int observe,
    int context)
{
    constexpr int VPT = HEAD_DIM_MAX / 32;

    const int row = static_cast<int>(blockIdx.x);
    const int request = captured_request(lanes, row, lane_offset, requests);
    if (request < 0) {
        return;
    }
    const int head = static_cast<int>(blockIdx.y);
    const int threads = static_cast<int>(blockDim.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    extern __shared__ float smem[];
    float* q_s = smem;
    float* wm = q_s + head_dim;
    float* wl = wm + WARPS;

    float* out = scores + static_cast<long long>(row) * context;

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
    const float weight = share / static_cast<float>(rows);

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
        const float inv = denominator > 0.f ? weight / denominator : 0.f;

        for (int j = warp; j < limit && j < context; j += WARPS) {
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

            if (lane == 0) {
                atomicAdd(&out[j], __expf(dot * sm_scale - folded_max) * inv);
            }
        }

        __syncthreads();
    }
}

// Each capturing lane's key row summed a kv page: a block a row, a thread
// a page.
__global__ void score_pages(
    const float* __restrict__ keys,
    float* __restrict__ pages,
    const i32* __restrict__ lanes,
    int lane_offset,
    int requests,
    int context,
    int page_size,
    int num_pages)
{
    const int row = static_cast<int>(blockIdx.x);
    if (captured_request(lanes, row, lane_offset, requests) < 0) {
        return;
    }
    const float* in = keys + static_cast<long long>(row) * context;
    float* out = pages + static_cast<long long>(row) * num_pages;
    for (int p = static_cast<int>(threadIdx.x); p < num_pages; p += static_cast<int>(blockDim.x)) {
        const int lo = p * page_size;
        const int hi = lo + page_size < context ? lo + page_size : context;
        float sum = 0.f;
        for (int j = lo; j < hi; ++j) {
            sum += in[j];
        }
        out[p] = sum;
    }
}

}
