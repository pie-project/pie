#include "ops/attention_naive.hpp"

#include <cmath>
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

}  // namespace pie_cuda_driver::ops
