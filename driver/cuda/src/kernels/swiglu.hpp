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

// Standard SwiGLU. Used by Llama / Qwen / Mistral / Mixtral.
//     y = silu(gate) * up = gate * sigmoid(gate) * up
void launch_swiglu_bf16(
    const void* gate,
    const void* up,
    void* y,
    int num_elements,
    cudaStream_t stream);

// GeLU-tanh-glu (Gemma).
void launch_geglu_tanh_bf16(
    const void* gate,
    const void* up,
    void* y,
    int num_elements,
    cudaStream_t stream);

// GPT-OSS expert activation. Distinct from SwiGLU on three counts:
//
//   * Asymmetric clamp on gate (upper-only): `gate' = min(gate, +limit)`.
//   * QuickGELU-style activation: `glu = gate' * sigmoid(alpha * gate')`
//     with `alpha = 1.702`. (Standard SwiGLU uses `alpha = 1`.)
//   * Symmetric clamp on up plus a `+1` residual shift:
//         up' = clamp(up, -limit, +limit)
//         y   = (up' + 1) * glu
//
// Matches `transformers/models/gpt_oss/modeling_gpt_oss.py::_apply_gate`.
void launch_gpt_oss_glu_bf16(
    const void* gate,
    const void* up,
    void* y,
    int num_elements,
    cudaStream_t stream,
    float limit,
    float alpha = 1.702f);

// Elementwise `x[i] *= sigmoid(gate[i])`. Used by Qwen3.5 full-
// attention's per-token output gate (a' = a * σ(g)).
void launch_sigmoid_gate_inplace_bf16(
    void*       x,      // bf16, in-place
    const void* gate,   // bf16, same shape as x
    int num_elements,
    cudaStream_t stream);

// Qwen3.6-MoE expert MLP fuses SwiGLU with the gate/up split — the
// expert's `gate_up_proj` GEMM produces a `[N, 2*I]` tensor where
// columns `[0, I)` are the gate features and `[I, 2*I)` are the up
// features. Read both halves of each row and emit `silu(gate) * up`
// directly into a `[N, I]` output, skipping the intermediate
// deinterleave that an unfused path would need.
//
//     y[n, i] = silu(packed[n, i]) * packed[n, I + i]
void launch_chunked_swiglu_bf16(
    const void* packed,  // [N, 2*I] bf16 (gate first, up second)
    void*       y,       // [N, I]   bf16
    int N, int I,
    cudaStream_t stream);

// In-place per-token sigmoid gate on a `[N, H]` tensor: `x[n, h] *=
// sigmoid(scalar_gate[n])`. Used by the Qwen3.6-MoE shared-expert path,
// where the gate is a single scalar per token (output of the `[N, 1]`
// shared_expert_gate projection).
void launch_sigmoid_scalar_gate_inplace_bf16(
    void*       x,             // bf16 [N, H], in-place
    const void* scalar_gate,   // bf16 [N]
    int N, int H,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
