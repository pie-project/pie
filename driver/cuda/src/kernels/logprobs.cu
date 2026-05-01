#include "kernels/logprobs.hpp"

#include <cuda_bf16.h>

#include "kernels/softmax_common.cuh"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;

// One block per sample row. Compute (max, log(sum_exp)) once via the
// shared softmax reductions, then scatter
//   log p(label) = (logit[label] - max) - log(sum_exp)
// for every requested label of that row.
__global__ void logprobs_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ sample_rows,
    const std::int32_t* __restrict__ label_indptr,
    const std::int32_t* __restrict__ label_ids,
    float* __restrict__ out,
    int vocab)
{
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = sample_rows[i];
    const int lo = label_indptr[i];
    const int hi = label_indptr[i + 1];
    const __nv_bfloat16* row_in =
        logits + static_cast<long long>(row) * vocab;

    __shared__ float buf[BLOCK];

    const float row_max = softmax::block_row_max<BLOCK>(row_in, vocab, 1.f, buf);
    __syncthreads();
    const float sum_exp = softmax::block_row_sum_exp<BLOCK>(row_in, vocab, 1.f, row_max, buf);
    const float log_z = logf(sum_exp);

    // Scatter logprobs at requested labels.
    for (int k = lo + tid; k < hi; k += BLOCK) {
        const int label = label_ids[k];
        if (label < 0 || label >= vocab) {
            out[k] = -INFINITY;
            continue;
        }
        const float logit = __bfloat162float(row_in[label]);
        out[k] = (logit - row_max) - log_z;
    }
}

}  // namespace

void launch_logprobs_bf16(
    const void* logits,
    const std::int32_t* sample_rows,
    const std::int32_t* label_indptr,
    const std::int32_t* label_ids,
    float* out,
    int num_samples, int vocab,
    cudaStream_t stream)
{
    if (num_samples <= 0) return;
    logprobs_kernel<<<num_samples, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        sample_rows, label_indptr, label_ids, out, vocab);
}

}  // namespace pie_cuda_driver::kernels
