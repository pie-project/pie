#pragma once

// flashinfer-backed temperature + top-k + top-p sampling.
//
// Pipeline:
//   probs[N, V] = softmax(logits[N, V] / temperature[N])      (OnlineSoftmax)
//   tokens[N]   = TopKTopPSamplingFromProb(probs, top_k, top_p, seeds)
//
// All per-row arrays are device pointers, expected dtypes:
//   logits, probs:        bf16
//   temperature, top_p:   bf16
//   top_k:                int32
//   seeds:                uint64
//
// The workspace is the same `AttentionWorkspace.float_buffer()` used by
// flashinfer's attention plan — `OnlineSoftmax` only needs ~hundreds of
// bytes per row for vocab≈150k, well under the 64 MiB we reserve.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Per-row softmax over `num_rows` of logits (using `temperatures_per_row`),
// followed by top-k / top-p sampling over `num_samples` of those rows
// chosen by `sample_row_indices`. Per-sample arrays (`top_k`, `top_p`,
// `seed`) have length `num_samples` and are aligned with
// `sample_row_indices`.
// `valid_scratch` is a device buffer of length >= num_samples, written by
// the kernel (caller does not need to read it back).
void launch_sample_topk_topp_bf16(
    const void* logits,                            // [num_rows, V] bf16
    void* probs_scratch,                           // [num_rows, V] fp32
    const float* temperatures_per_row,             // [num_rows]
    const std::int32_t* sample_row_indices,        // [num_samples] device
    const std::int32_t* top_k_arr,                 // [num_rows]
    const float* top_p_arr,                        // [num_rows]
    const std::uint64_t* seed_arr,                 // [num_rows]
    bool* valid_scratch,                           // [num_samples] device
    std::int32_t* out,                             // [num_samples]
    int num_rows,
    int num_samples,
    int vocab,
    std::uint64_t prng_offset,                     // monotonic per fire_batch
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
