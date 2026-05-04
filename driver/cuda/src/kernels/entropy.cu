#include "kernels/entropy.hpp"

#include <cuda_bf16.h>

#include "kernels/softmax_common.cuh"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

// Shannon entropy of softmax(logits) over each requested row. Three
// reductions:
//   max                = max_j x_j
//   sum_exp            = Σ_j exp(x_j - max)
//   weighted_sum_shift = Σ_j exp(x_j - max) * (x_j - max)
//
// Then entropy = log(sum_exp) - weighted_sum_shift / sum_exp, derived from
//   p_j = exp(x_j - max) / Z   with Z = sum_exp,
//   log p_j = (x_j - max) - log Z,
//   -Σ p_j log p_j = log Z - (1/Z) Σ exp(x_j - max) * (x_j - max).
__global__ void entropy_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ sample_rows,
    float* __restrict__ out,
    int vocab)
{
    const int tid = threadIdx.x;
    const int row = sample_rows[blockIdx.x];
    const __nv_bfloat16* row_in =
        logits + static_cast<long long>(row) * vocab;

    __shared__ float buf[BLOCK];

    const float row_max = softmax::block_row_max<BLOCK>(row_in, vocab, 1.f, buf);
    __syncthreads();
    const float sum_exp = softmax::block_row_sum_exp<BLOCK>(row_in, vocab, 1.f, row_max, buf);
    __syncthreads();

    // Pass 3: Σ exp(x-max) * (x-max). Reuses `buf` for the weighted-sum
    // reduction.
    float local_wsum = 0.f;
    for (int j = tid; j < vocab; j += BLOCK) {
        const float xs = __bfloat162float(row_in[j]) - row_max;
        local_wsum += expf(xs) * xs;
    }
    buf[tid] = local_wsum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = logf(sum_exp) - buf[0] / sum_exp;
    }
}

}  // namespace

void launch_entropy_bf16(
    const void* logits,
    const std::int32_t* sample_rows,
    float* out,
    int num_samples, int vocab,
    cudaStream_t stream)
{
    if (num_samples <= 0) return;
    entropy_kernel<<<num_samples, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        sample_rows, out, vocab);
}

}  // namespace pie_cuda_driver::kernels
