#pragma once

// Per-row argmax over [num_rows, vocab] bf16 logits → [num_rows] i32 token ids.
// Used as the greedy sampler (temperature=0).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_argmax_bf16(
    const void* logits,        // [num_rows, vocab] bf16
    std::int32_t* token_ids,   // [num_rows]
    int num_rows,
    int vocab,
    cudaStream_t stream);

void launch_argmax_bf16_partitioned_pairs(
    const void* logits,              // [num_rows, vocab] bf16
    std::uint64_t* partial_pairs,    // [parts, num_rows]
    int num_rows,
    int vocab,
    int parts,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
