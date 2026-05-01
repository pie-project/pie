#pragma once

// Per-row Shannon entropy of softmax(logits) over un-temperatured logits.
// Output one fp32 value per requested sample row. Matches `pie_driver`'s
// Sampler::Entropy semantics (log_softmax on raw logits, no temperature).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_entropy_bf16(
    const void* logits,                // [num_rows, vocab] bf16
    const std::int32_t* sample_rows,   // [num_samples] device — row indices
    float* out,                        // [num_samples] device
    int num_samples,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
