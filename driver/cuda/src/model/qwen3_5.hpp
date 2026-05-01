#pragma once

// Qwen-3.5 (hybrid Gated DeltaNet + full-attention). Most layers use a
// recurrent linear-attention SSM (GatedDeltaNet); a few full-attention
// layers preserve global retrieval. Architectural pieces:
//
//   * Linear-attention layer: sequence-recurrent state-space update
//     with per-step
//         h_t = (1 - α_t)·h_{t-1} + β_t·(K_t ⊗ V_t)
//         o_t = Q_t·h_t
//     followed by a 1D causal conv on the in-projection. Implemented in
//     pie_kernels (Triton) on the Python driver — the C++ side needs a
//     matching CUDA kernel: `gated_delta_net_paged_bf16` that walks
//     each request's KV state across positions.
//   * Q/K/V projections in full-attention layers go through a "gated"
//     projection where Q is concatenated with a per-head gate; the
//     output side multiplies the gate elementwise into the post-attn
//     output. flashinfer's `DefaultAttention` doesn't have a built-in
//     gate; can be applied as a pre/post step around the existing
//     attention call.
//   * Per-layer state: the SSM holds a `[batch, num_heads, key_dim,
//     value_dim]` running state plus a `[batch, num_heads, conv_dim]`
//     causal-conv buffer. Both reset at every prefill (no cross-prefill
//     state continuity in this driver's batching model).
//
// Status: NOT YET IMPLEMENTED. The blocker is the GatedDeltaNet kernel
// itself — there's no flashinfer counterpart, and writing one from
// scratch in CUDA is multi-day work. The full-attention portion can
// reuse `llama_like_forward_paged` once the gate-around-Q step lands
// as a thin wrapper.

#include "engine.hpp"
#include "model/qwen3.hpp"

namespace pie_cuda_driver::model {

Qwen3Weights bind_qwen3_5(Engine& engine);

}  // namespace pie_cuda_driver::model
