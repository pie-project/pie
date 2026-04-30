#include "ops/attention_paged.hpp"

#include <cmath>
#include <cuda_bf16.h>

namespace pie_cuda_driver::ops {

namespace {

constexpr int BLOCK = 256;

__device__ __forceinline__ int find_request(const std::uint32_t* qo_indptr,
                                            int R, int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

// One block per (q_head, token). Threads cooperate over the head_dim and
// the per-token KV range.
//
// Shared memory layout:
//   - scores[max_kv_len]   floats
//   - reduce_buf[BLOCK]    floats
__global__ void paged_attn_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_pages,
    const __nv_bfloat16* __restrict__ v_pages,
    __nv_bfloat16* __restrict__ o,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int R,
    int h_q, int h_kv, int d, int page_size,
    int max_kv_len,
    float scale)
{
    extern __shared__ float smem[];
    float* scores     = smem;
    float* reduce_buf = smem + max_kv_len;

    const int head      = blockIdx.x;
    const int token_idx = blockIdx.y;
    const int tid       = threadIdx.x;
    const int gqa_ratio = h_q / h_kv;
    const int kv_head   = head / gqa_ratio;

    // Resolve request, position, KV length for this token.
    const int r = find_request(qo_indptr, R, token_idx);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = token_idx - qo_lo;

    const int pages_first = kv_page_indptr[r];
    const int pages_last  = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after = (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_qpos = pre_kv_len + offset_in_new;
    const int kv_len = abs_qpos + 1;  // causal mask

    const __nv_bfloat16* q_vec =
        q + (static_cast<long long>(token_idx) * h_q + head) * d;

    // Pass 1: scores
    for (int j = tid; j < kv_len; j += BLOCK) {
        const int page_in_req    = j / page_size;
        const int offset_in_page = j % page_size;
        const int actual_page    = static_cast<int>(
            kv_page_indices[pages_first + page_in_req]);
        const __nv_bfloat16* k_vec =
            k_pages
            + ((static_cast<long long>(actual_page) * page_size) + offset_in_page)
              * h_kv * d
            + kv_head * d;

        float dot = 0.f;
        for (int i = 0; i < d; ++i) {
            dot += __bfloat162float(q_vec[i]) * __bfloat162float(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Pass 2: reduce-max
    float m = -INFINITY;
    for (int j = tid; j < kv_len; j += BLOCK) m = fmaxf(m, scores[j]);
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    // Pass 3: exp & sum
    float local_sum = 0.f;
    for (int j = tid; j < kv_len; j += BLOCK) {
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

    // Pass 4: weighted V
    __nv_bfloat16* o_vec =
        o + (static_cast<long long>(token_idx) * h_q + head) * d;
    for (int i = tid; i < d; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j < kv_len; ++j) {
            const int page_in_req    = j / page_size;
            const int offset_in_page = j % page_size;
            const int actual_page    = static_cast<int>(
                kv_page_indices[pages_first + page_in_req]);
            const __nv_bfloat16* v_vec =
                v_pages
                + ((static_cast<long long>(actual_page) * page_size) + offset_in_page)
                  * h_kv * d
                + kv_head * d;
            acc += scores[j] * inv_total * __bfloat162float(v_vec[i]);
        }
        o_vec[i] = __float2bfloat16(acc);
    }
}

}  // namespace

void launch_attention_paged_bf16(
    const void* q,
    const void* k_pages, const void* v_pages,
    void* o,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_kv_len,
    cudaStream_t stream)
{
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(num_q_heads, total_tokens);
    dim3 block(BLOCK);
    const std::size_t shmem_bytes =
        sizeof(float) * (static_cast<std::size_t>(max_kv_len) + BLOCK);

    paged_attn_kernel<<<grid, block, shmem_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k_pages),
        static_cast<const __nv_bfloat16*>(v_pages),
        static_cast<__nv_bfloat16*>(o),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        num_requests,
        num_q_heads, num_kv_heads, head_dim, page_size,
        max_kv_len, scale);
}

}  // namespace pie_cuda_driver::ops
