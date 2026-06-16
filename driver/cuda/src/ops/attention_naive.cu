#include "ops/attention_naive.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>

namespace pie_cuda_driver::ops {

namespace {

// One block per (query_head, query_pos). Threads cover the head_dim and
// the key range cooperatively.
//
// Algorithm per block:
//   1. Compute scores[j] = (q · k_j) / sqrt(d)  for j in [0..query_pos]
//   2. Reduce-max for numerical stability.
//   3. Compute exp(scores[j] - max), and sum.
//   4. Output[d] = sum_j (exp_score[j] / total) * v_j.
//
// Shared memory layout:
//   - scratch[num_tokens] floats: scores, then exp(scores - max)
//   - reduce_buf[BLOCK]: reduce buffer for max + sum
//
// The kernel is sized for parity tests (≤ 1024 tokens). Production paths
// will use flashinfer.
constexpr int BLOCK = 256;

__global__ void attn_naive_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ o,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale)
{
    extern __shared__ float smem[];
    float* scores = smem;                         // size: num_tokens
    float* reduce_buf = smem + num_tokens;        // size: BLOCK

    const int head      = blockIdx.x;
    const int query_pos = blockIdx.y;
    const int tid       = threadIdx.x;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head   = head / gqa_ratio;

    const __nv_bfloat16* q_vec =
        q + (static_cast<long long>(query_pos) * num_q_heads + head) * head_dim;

    // Pass 1: scores
    for (int j = tid; j <= query_pos; j += BLOCK) {
        const __nv_bfloat16* k_vec =
            k + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Pass 2: max
    float m = -INFINITY;
    for (int j = tid; j <= query_pos; j += BLOCK) {
        m = fmaxf(m, scores[j]);
    }
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    // Pass 3: exp & sum
    float local_sum = 0.f;
    for (int j = tid; j <= query_pos; j += BLOCK) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] += reduce_buf[tid + off];
        __syncthreads();
    }
    const float inv_total = 1.0f / reduce_buf[0];

    // Pass 4: weighted sum of V
    __nv_bfloat16* o_vec =
        o + (static_cast<long long>(query_pos) * num_q_heads + head) * head_dim;

    for (int i = tid; i < head_dim; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j <= query_pos; ++j) {
            const __nv_bfloat16* v_vec =
                v + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;
            acc += scores[j] * inv_total * __bfloat162float(v_vec[i]);
        }
        o_vec[i] = __float2bfloat16(acc);
    }
}

__global__ void attn_mtp_history_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ o,
    int num_tokens,
    int history_steps,
    int history_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale)
{
    extern __shared__ float smem[];
    float* scores = smem;                         // size: history_steps
    float* reduce_buf = smem + history_steps;     // size: BLOCK

    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = head / gqa_ratio;

    if (row >= num_tokens) return;

    const __nv_bfloat16* q_vec =
        q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;

    for (int j = tid; j < history_steps; j += BLOCK) {
        const int hist_row = j * history_stride + row;
        const __nv_bfloat16* k_vec =
            k + (static_cast<long long>(hist_row) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int j = tid; j < history_steps; j += BLOCK) {
        m = fmaxf(m, scores[j]);
    }
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    float local_sum = 0.f;
    for (int j = tid; j < history_steps; j += BLOCK) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] += reduce_buf[tid + off];
        __syncthreads();
    }
    const float inv_total = 1.0f / reduce_buf[0];

    __nv_bfloat16* o_vec =
        o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j < history_steps; ++j) {
            const int hist_row = j * history_stride + row;
            const __nv_bfloat16* v_vec =
                v + (static_cast<long long>(hist_row) * num_kv_heads + kv_head) * head_dim;
            acc += scores[j] * inv_total * __bfloat162float(v_vec[i]);
        }
        o_vec[i] = __float2bfloat16(acc);
    }
}

__device__ __forceinline__ int find_request_u32(
    const std::uint32_t* __restrict__ qo_indptr,
    int R,
    int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

__global__ void mtp_shift_hidden_kernel(
    const __nv_bfloat16* __restrict__ target_hidden,
    const __nv_bfloat16* __restrict__ pending_hidden,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::int32_t* __restrict__ slot_ids,
    __nv_bfloat16* __restrict__ out,
    int total_tokens,
    int num_requests,
    int hidden_size)
{
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= total_tokens) return;
    const int r = find_request_u32(qo_indptr, num_requests, t);
    const bool first_in_request = t == static_cast<int>(qo_indptr[r]);
    const int slot = slot_ids != nullptr ? slot_ids[r] : 0;
    const __nv_bfloat16* src = first_in_request
        ? pending_hidden + static_cast<long long>(slot) * hidden_size
        : target_hidden + static_cast<long long>(t - 1) * hidden_size;
    __nv_bfloat16* dst = out + static_cast<long long>(t) * hidden_size;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__global__ void mtp_update_pending_hidden_kernel(
    const __nv_bfloat16* __restrict__ target_hidden,
    __nv_bfloat16* __restrict__ pending_hidden,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::int32_t* __restrict__ slot_ids,
    int num_requests,
    int hidden_size)
{
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= num_requests) return;
    const int lo = static_cast<int>(qo_indptr[r]);
    const int hi = static_cast<int>(qo_indptr[r + 1]);
    if (hi <= lo) return;
    const int slot = slot_ids != nullptr ? slot_ids[r] : 0;
    const __nv_bfloat16* src =
        target_hidden + static_cast<long long>(hi - 1) * hidden_size;
    __nv_bfloat16* dst =
        pending_hidden + static_cast<long long>(slot) * hidden_size;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

__device__ __forceinline__ const __nv_bfloat16* mtp_paged_vec(
    const __nv_bfloat16* __restrict__ pages,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    int request,
    int pos,
    int page_size,
    int num_kv_heads,
    int head_dim,
    int kv_head,
    bool hnd_layout)
{
    const int page_in_req = pos / page_size;
    const int off = pos - page_in_req * page_size;
    const int actual_page = static_cast<int>(
        kv_page_indices[kv_page_indptr[request] + page_in_req]);
    if (hnd_layout) {
        return pages +
            (((static_cast<long long>(actual_page) * num_kv_heads + kv_head) *
              page_size + off) * head_dim);
    }
    return pages +
        (((static_cast<long long>(actual_page) * page_size + off) *
          num_kv_heads + kv_head) * head_dim);
}

__global__ void attn_mtp_paged_history_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pages,
    const __nv_bfloat16* __restrict__ v_pages,
    const __nv_bfloat16* __restrict__ k_history,
    const __nv_bfloat16* __restrict__ v_history,
    __nv_bfloat16* __restrict__ o,
    const std::int32_t* __restrict__ position_ids,
    const std::int32_t* __restrict__ request_ids,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int num_tokens,
    int history_steps,
    int history_stride,
    int max_global_tokens,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    float scale,
    bool prefix_global)
{
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce_buf = smem + max_global_tokens + history_steps;

    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= num_tokens) return;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = head / gqa_ratio;
    const int request = request_ids[row];
    const int page_lo = static_cast<int>(kv_page_indptr[request]);
    const int page_hi = static_cast<int>(kv_page_indptr[request + 1]);
    const int pages = page_hi - page_lo;
    const int max_kv_len = pages <= 0
        ? 0
        : (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    int global_len = position_ids[row] -
                     (prefix_global ? (history_steps - 1) : 0);
    if (global_len < 0) global_len = 0;
    if (global_len > max_kv_len) global_len = max_kv_len;
    if (global_len > max_global_tokens) global_len = max_global_tokens;
    const int total_steps = global_len + history_steps;

    const __nv_bfloat16* q_vec =
        q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;

    for (int j = tid; j < total_steps; j += BLOCK) {
        const __nv_bfloat16* k_vec = nullptr;
        if (j < global_len) {
            k_vec = mtp_paged_vec(k_pages, kv_page_indices, kv_page_indptr,
                request, j, page_size, num_kv_heads, head_dim, kv_head,
                hnd_layout);
        } else {
            const int hist_step = j - global_len;
            const int hist_row = hist_step * history_stride + row;
            k_vec = k_history +
                (static_cast<long long>(hist_row) * num_kv_heads + kv_head) *
                    head_dim;
        }
        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float m = -INFINITY;
    for (int j = tid; j < total_steps; j += BLOCK) {
        m = fmaxf(m, scores[j]);
    }
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    float local_sum = 0.f;
    for (int j = tid; j < total_steps; j += BLOCK) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] += reduce_buf[tid + off];
        __syncthreads();
    }
    const float inv_total = 1.0f / reduce_buf[0];

    __nv_bfloat16* o_vec =
        o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j < total_steps; ++j) {
            const __nv_bfloat16* v_vec = nullptr;
            if (j < global_len) {
                v_vec = mtp_paged_vec(v_pages, kv_page_indices, kv_page_indptr,
                    request, j, page_size, num_kv_heads, head_dim, kv_head,
                    hnd_layout);
            } else {
                const int hist_step = j - global_len;
                const int hist_row = hist_step * history_stride + row;
                v_vec = v_history +
                    (static_cast<long long>(hist_row) * num_kv_heads + kv_head) *
                        head_dim;
            }
            acc += scores[j] * inv_total * __bfloat162float(v_vec[i]);
        }
        o_vec[i] = __float2bfloat16(acc);
    }
}

}  // namespace

void launch_attention_naive_bf16(
    const void* q, const void* k, const void* v,
    void* o,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(num_q_heads, num_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes =
        sizeof(float) * (static_cast<std::size_t>(num_tokens) + BLOCK);

    attn_naive_kernel<<<grid, block, shmem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v),
        static_cast<__nv_bfloat16*>(o),
        num_tokens, num_q_heads, num_kv_heads, head_dim, scale);
}

void launch_attention_mtp_history_bf16(
    const void* q,
    const void* k_history,
    const void* v_history,
    void* o,
    int num_tokens,
    int history_steps,
    int history_stride,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || history_steps <= 0) return;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(num_q_heads, num_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes =
        sizeof(float) * (static_cast<std::size_t>(history_steps) + BLOCK);
    attn_mtp_history_kernel<<<grid, block, shmem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k_history),
        static_cast<const __nv_bfloat16*>(v_history),
        static_cast<__nv_bfloat16*>(o),
        num_tokens, history_steps, history_stride,
        num_q_heads, num_kv_heads, head_dim, scale);
}

void launch_attention_mtp_paged_history_bf16(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const void* k_history,
    const void* v_history,
    void* o,
    const std::int32_t* position_ids,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_tokens,
    int history_steps,
    int history_stride,
    int max_global_tokens,
    int page_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    bool hnd_layout,
    bool global_cache_uses_prefix_position,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || history_steps <= 0) return;
    if (max_global_tokens <= 0) {
        launch_attention_mtp_history_bf16(
            q, k_history, v_history, o, num_tokens, history_steps,
            history_stride, num_q_heads, num_kv_heads, head_dim, stream);
        return;
    }
    // Keep the reference kernel inside portable shared-memory limits. Long
    // contexts should use a FlashInfer-backed MTP decode path; until then,
    // fall back to local draft history instead of failing the launch.
    if (max_global_tokens + history_steps > 8192) {
        launch_attention_mtp_history_bf16(
            q, k_history, v_history, o, num_tokens, history_steps,
            history_stride, num_q_heads, num_kv_heads, head_dim, stream);
        return;
    }
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(num_q_heads, num_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes = sizeof(float) *
        (static_cast<std::size_t>(max_global_tokens + history_steps) + BLOCK);
    attn_mtp_paged_history_kernel<<<grid, block, shmem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k_pages),
        static_cast<const __nv_bfloat16*>(v_pages),
        static_cast<const __nv_bfloat16*>(k_history),
        static_cast<const __nv_bfloat16*>(v_history),
        static_cast<__nv_bfloat16*>(o),
        position_ids, request_ids,
        kv_page_indices, kv_page_indptr, kv_last_page_lens,
        num_tokens, history_steps, history_stride, max_global_tokens,
        page_size, num_q_heads, num_kv_heads, head_dim, hnd_layout, scale,
        global_cache_uses_prefix_position);
}

void launch_mtp_shift_hidden_bf16(
    const void* target_hidden,
    const void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    void* out,
    int total_tokens,
    int num_requests,
    int hidden_size,
    cudaStream_t stream)
{
    if (total_tokens <= 0 || num_requests <= 0 || hidden_size <= 0 ||
        pending_hidden == nullptr) {
        return;
    }
    mtp_shift_hidden_kernel<<<total_tokens, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(target_hidden),
        static_cast<const __nv_bfloat16*>(pending_hidden),
        qo_indptr, slot_ids,
        static_cast<__nv_bfloat16*>(out),
        total_tokens, num_requests, hidden_size);
}

void launch_mtp_update_pending_hidden_bf16(
    const void* target_hidden,
    void* pending_hidden,
    const std::uint32_t* qo_indptr,
    const std::int32_t* slot_ids,
    int num_requests,
    int hidden_size,
    cudaStream_t stream)
{
    if (num_requests <= 0 || hidden_size <= 0 || pending_hidden == nullptr) {
        return;
    }
    mtp_update_pending_hidden_kernel<<<num_requests, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(target_hidden),
        static_cast<__nv_bfloat16*>(pending_hidden),
        qo_indptr, slot_ids, num_requests, hidden_size);
}

}  // namespace pie_cuda_driver::ops
