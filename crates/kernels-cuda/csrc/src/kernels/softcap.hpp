#pragma once

// Logit soft-cap: `out[i] = cap * tanh(x[i] / cap)`. Used by Gemma-2/3
// on the final lm_head output (`final_logit_softcapping`) and -- in
// principle -- on per-step attention logits, though flashinfer's TC
// kernel doesn't expose that hook so we skip the attention soft-cap on
// HF parity grounds (matches pie_driver's Gemma-2 adapter).
//
// `n` is the total element count; the kernel is dimension-agnostic.

#include <cstddef>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_logit_softcap_bf16(
    void* x,            // [n] bf16 in-place
    float cap,
    std::size_t n,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
