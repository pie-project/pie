#pragma once

// MoE building blocks. Launcher declarations; the kernel bodies live in
// moe.cu, lifted verbatim from driver/cuda/src/kernels/topk_softmax.cu and
// driver/cuda/src/kernels/swiglu.cu (the base variants). The per-expert-scale
// / sigmoid-bias router variants and the strided / geglu / sigmoid-gate
// activation variants are lifted later as the forward bodies that need them
// land.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// Top-K from softmaxed router logits, with renormalization. Implements
// Mixtral / GPT-OSS style sparse-MoE routing:
//
//   probs    = softmax(logits, dim=-1)            # [N, num_experts]
//   topk_w, topk_idx = topk(probs, K, dim=-1)     # [N, K]
//   topk_w  /= topk_w.sum(dim=-1, keepdim=True)   # renormalize
//
// `logits` is bf16 (matches the rest of the activations); `topk_idx` is
// i32, `topk_w` is fp32 (downstream multiplies expert outputs in fp32
// to avoid bf16 round-trip noise). One block per token; each block
// runs a sequential top-K which is fine for E <= 64.
void topk_softmax_bf16(
    const void* logits,        // [N, num_experts] bf16
    std::int32_t* topk_idx,    // [N, K] i32 — expert indices
    float* topk_w,             // [N, K] fp32 — renormalized routing weights
    int N,
    int num_experts,
    int K,
    cudaStream_t stream);

// Qwen3.6-MoE expert MLP fuses SwiGLU with the gate/up split — the
// expert's `gate_up_proj` GEMM produces a `[N, 2*I]` tensor where
// columns `[0, I)` are the gate features and `[I, 2*I)` are the up
// features. Read both halves of each row and emit `silu(gate) * up`
// directly into a `[N, I]` output, skipping the intermediate
// deinterleave that an unfused path would need.
//
//     y[n, i] = silu(packed[n, i]) * packed[n, I + i]
void chunked_swiglu_bf16(
    const void* packed,  // [N, 2*I] bf16 (gate first, up second)
    void*       y,       // [N, I]   bf16
    int N, int I,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
