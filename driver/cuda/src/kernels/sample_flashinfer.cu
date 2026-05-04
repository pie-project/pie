#include "kernels/sample_flashinfer.hpp"

#include <cuda_bf16.h>

#include <flashinfer/sampling.cuh>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels {

namespace {

constexpr int SOFTMAX_BLOCK = 256;

// Per-row softmax with temperature. Reads bf16 logits, writes fp32 probs
// (matching flashinfer's expected dtype). Three passes: max-reduce, exp+sum,
// normalize. T <= 0 falls back to plain softmax (we still want a valid
// distribution even for unused rows).
__global__ void softmax_temp_bf16_to_fp32_kernel(
    const __nv_bfloat16* __restrict__ logits,
    float* __restrict__ probs,
    const float* __restrict__ temperatures,
    int vocab)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row_in = logits + static_cast<long long>(row) * vocab;
    float* row_out = probs + static_cast<long long>(row) * vocab;

    const float T = temperatures[row];
    const float inv_T = (T > 0.f) ? (1.f / T) : 1.f;

    __shared__ float buf[SOFTMAX_BLOCK];

    float local_max = -INFINITY;
    for (int j = tid; j < vocab; j += SOFTMAX_BLOCK) {
        const float v = __bfloat162float(row_in[j]) * inv_T;
        if (v > local_max) local_max = v;
    }
    buf[tid] = local_max;
    __syncthreads();
    for (int off = SOFTMAX_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] = fmaxf(buf[tid], buf[tid + off]);
        __syncthreads();
    }
    const float row_max = buf[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int j = tid; j < vocab; j += SOFTMAX_BLOCK) {
        const float v = __bfloat162float(row_in[j]) * inv_T;
        const float e = expf(v - row_max);
        row_out[j] = e;
        local_sum += e;
    }
    buf[tid] = local_sum;
    __syncthreads();
    for (int off = SOFTMAX_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_sum = 1.f / buf[0];

    for (int j = tid; j < vocab; j += SOFTMAX_BLOCK) {
        row_out[j] *= inv_sum;
    }
}

}  // namespace

void launch_sample_topk_topp_bf16(
    const void* logits, void* probs_scratch,
    const float* temperatures_per_row,
    const std::int32_t* sample_row_indices,
    const std::int32_t* top_k_arr,
    const float* top_p_arr,
    const std::uint64_t* seed_arr,
    bool* valid_scratch,
    std::int32_t* out,
    int num_rows, int num_samples, int vocab,
    std::uint64_t prng_offset,
    cudaStream_t stream)
{
    // 1. Softmax (bf16 → fp32) over every row of logits we have. Wasteful
    //    for non-sampled rows but simpler than gathering first.
    softmax_temp_bf16_to_fp32_kernel<<<num_rows, SOFTMAX_BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits),
        static_cast<float*>(probs_scratch),
        temperatures_per_row,
        vocab);

    // 2. flashinfer top-k + top-p sampling, restricted to sampled rows via
    //    `indices`. Per-sample top_k / top_p / seed are aligned with
    //    `sample_row_indices`.
    auto status = ::flashinfer::sampling::TopKTopPSamplingFromProb<float, std::int32_t>(
        static_cast<float*>(probs_scratch),
        const_cast<std::int32_t*>(top_k_arr),
        const_cast<float*>(top_p_arr),
        out,
        /*valid=*/valid_scratch,
        /*indices=*/const_cast<std::int32_t*>(sample_row_indices),
        static_cast<uint32_t>(num_samples),
        /*top_k_val=*/0, /*top_p_val=*/1.0f,
        static_cast<uint32_t>(vocab),
        /*deterministic=*/false,
        const_cast<std::uint64_t*>(seed_arr),
        /*seed_val=*/0,
        /*offset_arr=*/nullptr,
        /*offset_val=*/prng_offset,
        stream);
    CUDA_CHECK(status);
}

}  // namespace pie_cuda_driver::kernels
