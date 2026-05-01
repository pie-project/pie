#pragma once

// Per-sample log-probabilities at specified token IDs (no temperature).
// Implements `Sampler::Logprob` (1 token id) and `Sampler::Logprobs`
// (K token ids) by computing log_softmax(logits) over the un-temperatured
// row and scattering values to the requested label positions.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_logprobs_bf16(
    const void* logits,                      // [num_rows, vocab] bf16
    const std::int32_t* sample_rows,         // [num_samples] device — row idx
    const std::int32_t* label_indptr,        // [num_samples + 1] device
    const std::int32_t* label_ids,           // [label_indptr[num_samples]] device
    float* out,                              // [label_indptr[num_samples]] device
    int num_samples,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
