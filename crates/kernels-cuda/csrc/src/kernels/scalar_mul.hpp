#pragma once

// In-place scalar multiply: `x[i] *= s` over `n` bf16 elements. Used by
// Gemma family for two distinct purposes:
//   * Embedding scale — y *= sqrt(hidden_size) right after the lookup.
//   * Query pre-attention scale — q *= 1/sqrt(query_pre_attn_scalar).
//
// Plain bf16 round-trip — no fma, no fused load/store optimization. The
// scale-only step is bandwidth-bound and tiny relative to the GEMMs
// around it; readability beats micro-tuning here.

#include <cstddef>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_scalar_mul_bf16(
    void* x,
    float s,
    std::size_t n,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
