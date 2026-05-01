#pragma once

// Gated MLP activations:
//   * SwiGLU       — y = silu(gate) * up                  (Llama / Qwen / Mistral / OLMo / Phi)
//   * GeGLU(tanh)  — y = gelu_tanh(gate) * up             (Gemma family — `gelu_pytorch_tanh`)
//
// `silu(x)        = x * sigmoid(x) = x / (1 + exp(-x))`
// `gelu_tanh(x)   = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`
//
// Element-wise. `gate`, `up`, `y` are bf16 row-major, all the same size.

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// `clip_limit > 0` enables the GPT-OSS variant:
//     gate'  = clamp(gate, -limit, +limit)
//     up'    = clamp(up,   -limit, +limit)
//     y      = silu(gate') · (up' + 1)
// (The `+1` shifts the residual so a zero-init expert outputs the
// identity contribution.) `clip_limit == 0` is the standard
// `silu(gate) * up` used by Llama / Qwen / Mistral / Mixtral.
void launch_swiglu_bf16(
    const void* gate,
    const void* up,
    void* y,
    int num_elements,
    cudaStream_t stream,
    float clip_limit = 0.f);

// GeLU-tanh-glu (Gemma).
void launch_geglu_tanh_bf16(
    const void* gate,
    const void* up,
    void* y,
    int num_elements,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
