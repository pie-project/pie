#pragma once

// Gaussian-quantile sparsity gate, used by Gemma-3n's MLP for layers
// with `activation_sparsity_pattern[layer] > 0` (the first ~10 layers
// of E2B/E4B). Per row of `[N, dim]`:
//
//     mean   = mean(x[i, :])
//     std    = std(x[i, :])              # population std, unbiased=False
//     cutoff = mean + std · std_multiplier
//     x[i,j] = relu(x[i,j] - cutoff)
//
// `std_multiplier` is `Φ⁻¹(target_sparsity)` computed once on the host
// per layer; the kernel doesn't need to know `target_sparsity`. For
// `target_sparsity = 0.95`, `std_multiplier ≈ 1.6448536269514722`.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_gaussian_topk_bf16(
    void* x,                  // bf16 [N, dim], in-place
    int   N,
    int   dim,
    float std_multiplier,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
