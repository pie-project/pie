#pragma once

// Block-level softmax reductions shared by `entropy`, `logprobs`, and
// `dist` kernels. All three iterate over a row of bf16 logits, find the
// max for numerical stability, then sum exp(logit - max). Only the
// post-reduction step differs (scatter probs vs. weighted-sum vs.
// per-label logprob), so the two reduction passes are factored here.
//
// Conventions:
//   * One CUDA block per row; `BLOCK` threads per block.
//   * Caller owns the `__shared__ float buf[BLOCK]` scratch and passes
//     it in — keeps shared-memory footprint visible at the call site.
//   * `scale` is applied before the max / exp (used to fold a temperature
//     `1/T` into the dist softmax; entropy/logprobs pass `1.f`).
//   * Both helpers leave `__syncthreads()` outstanding on their tail —
//     callers can `__syncthreads()` before reusing `buf` for the next
//     pass.

#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels::softmax {

// Reduce `max_j (row[j] * scale)` across the block. Every thread
// receives the result.
template <int BLOCK>
__device__ inline float block_row_max(
    const __nv_bfloat16* __restrict__ row,
    int vocab,
    float scale,
    float* __restrict__ buf)
{
    const int tid = threadIdx.x;
    float local = -INFINITY;
    for (int j = tid; j < vocab; j += BLOCK) {
        local = fmaxf(local, __bfloat162float(row[j]) * scale);
    }
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    return buf[0];
}

// Reduce `Σ_j exp(row[j] * scale - row_max)` across the block. Every
// thread receives the result. `row_max` should match the value
// previously returned by `block_row_max(row, vocab, scale, ...)` to
// keep the exp argument numerically bounded above by 0.
template <int BLOCK>
__device__ inline float block_row_sum_exp(
    const __nv_bfloat16* __restrict__ row,
    int vocab,
    float scale,
    float row_max,
    float* __restrict__ buf)
{
    const int tid = threadIdx.x;
    float local = 0.f;
    for (int j = tid; j < vocab; j += BLOCK) {
        local += expf(__bfloat162float(row[j]) * scale - row_max);
    }
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    return buf[0];
}

}  // namespace pie_cuda_driver::kernels::softmax
