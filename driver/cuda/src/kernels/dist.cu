#include "kernels/dist.hpp"

#include <cuda_bf16.h>

#include "kernels/softmax_common.cuh"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;
constexpr float GREEDY_T = 1e-5f;

// Per-sample softmax(logits / T): one block per requested sample row,
// writing fp32 probs to `out_probs[i, j]`. T ≤ GREEDY_T is clamped to
// avoid div-by-zero (matches pie_driver's `scaled_softmax` greedy
// fallback).
__global__ void softmax_temp_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const std::int32_t* __restrict__ sample_rows,
    const float* __restrict__ temperatures,
    float* __restrict__ out_probs,
    int vocab)
{
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int row = sample_rows[i];
    const __nv_bfloat16* row_in =
        logits + static_cast<long long>(row) * vocab;
    float* row_out = out_probs + static_cast<long long>(i) * vocab;

    float T = temperatures[i];
    if (!(T > GREEDY_T)) T = GREEDY_T;
    const float inv_T = 1.f / T;

    __shared__ float buf[BLOCK];

    const float row_max = softmax::block_row_max<BLOCK>(row_in, vocab, inv_T, buf);
    __syncthreads();
    const float sum_exp = softmax::block_row_sum_exp<BLOCK>(row_in, vocab, inv_T, row_max, buf);
    const float inv_Z = 1.f / sum_exp;

    for (int j = tid; j < vocab; j += BLOCK) {
        row_out[j] = expf(__bfloat162float(row_in[j]) * inv_T - row_max) * inv_Z;
    }
}

}  // namespace

void launch_softmax_temp_bf16(
    const void* logits,
    const std::int32_t* sample_rows,
    const float* temperatures,
    float* out_probs,
    int num_samples, int vocab,
    cudaStream_t stream)
{
    if (num_samples <= 0) return;
    softmax_temp_kernel<<<num_samples, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        sample_rows, temperatures, out_probs, vocab);
}

}  // namespace pie_cuda_driver::kernels
