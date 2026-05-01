#pragma once

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
// runs a sequential top-K which is fine for E ≤ 64.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_topk_softmax_bf16(
    const void* logits,        // [N, num_experts] bf16
    std::int32_t* topk_idx,    // [N, K] i32 — expert indices
    float* topk_w,             // [N, K] fp32 — renormalized routing weights
    int N,
    int num_experts,
    int K,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
