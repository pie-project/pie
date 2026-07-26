#include "kernels/topk_softmax.hpp"

#include <cuda_bf16.h>
#include <cfloat>
#include <stdexcept>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 64;
// Qwen3.6-35B-A3B uses 256 experts; Kimi K2.6 uses 384. Keep a single
// static shared-memory slab large enough for both. 512 floats == 2 KB.
constexpr int MAX_EXPERTS = 512;

// Block-wide argmax over `scores[0..num_experts)`, ties resolved to the
// LOWEST index — the same winner a serial `for (j) if (s[j] > best)` scan
// picks, so routing decisions stay bit-identical.
//
// The serial form cost K * num_experts iterations on thread 0 while the
// other 63 threads idled: at 256 experts and K = 8 that is 2048 dependent
// shared-memory reads, which measured 21 us per layer (7% of a Qwen3.6
// decode step). Strided scan plus a log-depth reduction is ~10 steps.
__device__ inline void block_argmax(
    const float* __restrict__ scores,
    int num_experts,
    float floor_value,
    float* __restrict__ value_buf,
    int* __restrict__ index_buf,
    float& best_value,
    int& best_index)
{
    const int tid = threadIdx.x;
    // Strictly above `floor_value`, matching the serial scan's seed: an
    // already-picked expert is excluded by writing the floor back into
    // `scores`, and the floor itself must never win.
    float local_v = floor_value;
    int local_i = -1;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float v = scores[j];
        if (v > local_v) {
            local_v = v;
            local_i = j;
        }
    }
    value_buf[tid] = local_v;
    index_buf[tid] = local_i;
    __syncthreads();
    // A shared-memory tree over all BLOCK lanes costs log2(BLOCK)
    // __syncthreads PER ROUND, and there are K rounds. Fold the upper
    // warp once, then finish inside warp 0 with shuffles, which need no
    // barrier at all: 2 barriers per round instead of 8.
    static_assert(BLOCK == 64, "block_argmax folds exactly one upper warp");
    if (tid < 32) {
        float v = value_buf[tid];
        int i = index_buf[tid];
        // A strided scan gives thread t the indices t, t+BLOCK, ..., so the
        // lower index of a tie is not always in the lower lane: compare
        // indices explicitly rather than relying on lane order. This keeps
        // the winner identical to a serial `if (s[j] > best)` scan.
        auto take = [](float& v, int& i, float ov, int oi) {
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
                v = ov;
                i = oi;
            }
        };
        take(v, i, value_buf[tid + 32], index_buf[tid + 32]);
        for (int off = 16; off > 0; off >>= 1) {
            take(v, i,
                 __shfl_down_sync(0xffffffffu, v, off),
                 __shfl_down_sync(0xffffffffu, i, off));
        }
        if (tid == 0) {
            value_buf[0] = v;
            index_buf[0] = i;
        }
    }
    __syncthreads();
    best_value = value_buf[0];
    best_index = index_buf[0];
    // No trailing barrier: every caller syncs after acting on the winner,
    // which is what orders the next round's `value_buf` writes.
}

// One block per token. Phase 1: thread-local max-reduce + exp+sum-reduce
// for softmax. Phase 2: K iterations of argmax-with-exclusion to pick the
// top-K probs. Phase 3: thread 0 renormalizes and writes back.
__global__ void topk_softmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts, int K)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float buf[BLOCK];
    __shared__ int ibuf[BLOCK];

    // 1. Stage row into shared memory + find max.
    float local_max = -FLT_MAX;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float v = __bfloat162float(row[j]);
        probs[j] = v;
        if (v > local_max) local_max = v;
    }
    buf[tid] = local_max;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    const float row_max = buf[0];
    __syncthreads();

    // 2. exp + sum.
    float local_sum = 0.f;
    for (int j = tid; j < num_experts; j += BLOCK) {
        const float e = expf(probs[j] - row_max);
        probs[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_Z = 1.f / buf[0];
    __syncthreads();

    // 3. Normalize in shared mem, then K block-wide argmaxes with exclusion.
    for (int j = tid; j < num_experts; j += BLOCK) probs[j] *= inv_Z;
    __syncthreads();

    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float*        out_w   = topk_w   + static_cast<long long>(n) * K;
    float w_sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -1.f;
        int best_i = -1;
        block_argmax(probs, num_experts, -1.f, buf, ibuf, best_v, best_i);
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = best_v;
            if (best_i >= 0) probs[best_i] = -1.f;  // exclude on next pass
        }
        w_sum += best_v;
        __syncthreads();
    }
    if (tid == 0) {
        const float inv_w = 1.f / w_sum;
        for (int k = 0; k < K; ++k) out_w[k] *= inv_w;
    }
}

}  // namespace

void launch_topk_softmax_bf16(
    const void* logits,
    std::int32_t* topk_idx, float* topk_w,
    int N, int num_experts, int K,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    if (num_experts > MAX_EXPERTS) {
        throw std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS");
    }
    topk_softmax_bf16_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        topk_idx, topk_w,
        num_experts, K);
}

namespace {

__global__ void apply_per_expert_scale_kernel(
    const std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    const __nv_bfloat16* __restrict__ per_expert_scale,
    int total)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    const int e = topk_idx[t];
    const float s = __bfloat162float(per_expert_scale[e]);
    topk_w[t] *= s;
}

}  // namespace

void launch_apply_per_expert_scale_bf16(
    const std::int32_t* topk_idx,
    float* topk_w,
    const void* per_expert_scale_bf16,
    int N, int K,
    cudaStream_t stream)
{
    const int total = N * K;
    if (total <= 0) return;
    constexpr int BLOCK_T = 256;
    const int grid = (total + BLOCK_T - 1) / BLOCK_T;
    apply_per_expert_scale_kernel<<<grid, BLOCK_T, 0, stream>>>(
        topk_idx, topk_w,
        static_cast<const __nv_bfloat16*>(per_expert_scale_bf16),
        total);
}

namespace {

__global__ void topk_sigmoid_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ correction_bias,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row =
        logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float choice[MAX_EXPERTS];
    __shared__ float buf[BLOCK];
    __shared__ int ibuf[BLOCK];

    for (int j = tid; j < num_experts; j += BLOCK) {
        const float z = __bfloat162float(row[j]);
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -FLT_MAX;
        int best_i = -1;
        block_argmax(choice, num_experts, -FLT_MAX, buf, ibuf, best_v, best_i);
        const float weight = best_i >= 0 ? probs[best_i] : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = weight;
            if (best_i >= 0) choice[best_i] = -FLT_MAX;
        }
        sum += weight;
        __syncthreads();
    }
    if (tid == 0) {
        const float scale =
            normalize ? (routed_scaling_factor / (sum + 1e-20f))
                      : routed_scaling_factor;
        for (int k = 0; k < K; ++k) {
            out_w[k] *= scale;
        }
    }
}

__global__ void topk_sigmoid_bias_fp32_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ correction_bias,
    std::int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_w,
    int num_experts,
    int K,
    int normalize,
    float routed_scaling_factor)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = logits + static_cast<long long>(n) * num_experts;

    __shared__ float probs[MAX_EXPERTS];
    __shared__ float choice[MAX_EXPERTS];
    __shared__ float buf[BLOCK];
    __shared__ int ibuf[BLOCK];

    for (int j = tid; j < num_experts; j += BLOCK) {
        const float z = row[j];
        const float p = 1.f / (1.f + __expf(-z));
        probs[j] = p;
        choice[j] = p + correction_bias[j];
    }
    __syncthreads();

    std::int32_t* out_idx = topk_idx + static_cast<long long>(n) * K;
    float* out_w = topk_w + static_cast<long long>(n) * K;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        float best_v = -FLT_MAX;
        int best_i = -1;
        block_argmax(choice, num_experts, -FLT_MAX, buf, ibuf, best_v, best_i);
        const float weight = best_i >= 0 ? probs[best_i] : 0.f;
        if (tid == 0) {
            out_idx[k] = best_i;
            out_w[k] = weight;
            if (best_i >= 0) choice[best_i] = -FLT_MAX;
        }
        sum += weight;
        __syncthreads();
    }
    if (tid == 0) {
        const float scale =
            normalize ? (routed_scaling_factor / (sum + 1e-20f))
                      : routed_scaling_factor;
        for (int k = 0; k < K; ++k) {
            out_w[k] *= scale;
        }
    }
}

}  // namespace

void launch_topk_sigmoid_bias_bf16(
    const void* logits,
    const float* correction_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    topk_sigmoid_bias_bf16_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

void launch_topk_sigmoid_bias_fp32(
    const float* logits,
    const float* correction_bias,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream)
{
    if (N <= 0 || num_experts <= 0 || K <= 0) return;
    topk_sigmoid_bias_fp32_kernel<<<N, BLOCK, 0, stream>>>(
        logits,
        correction_bias,
        topk_idx,
        topk_w,
        num_experts,
        K,
        normalize ? 1 : 0,
        routed_scaling_factor);
}

}  // namespace pie_cuda_driver::kernels
