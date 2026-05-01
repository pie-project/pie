#pragma once

// Per-sample temperature-scaled softmax. Implements `Sampler::Dist`'s
// "return the distribution itself" output: probs = softmax(logits / T)
// computed once per requested sample row, written to fp32. The host then
// sorts each row descending and emits a (token_ids, probs) tuple per slot.
//
// Temperature ≤ 0 is clamped to 1e-5 to mirror pie_driver's
// `scaled_softmax` greedy-fallback behavior (T=0 → near-one-hot at argmax).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_softmax_temp_bf16(
    const void* logits,                      // [num_rows, vocab] bf16
    const std::int32_t* sample_rows,         // [num_samples] device — row idx
    const float* temperatures,               // [num_samples] device — T per sample
    float* out_probs,                        // [num_samples, vocab] device — fp32 probs
    int num_samples,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
