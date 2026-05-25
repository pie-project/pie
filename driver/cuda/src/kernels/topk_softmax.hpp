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

// Gemma-4 26B-A4B's router applies a per-expert scalar gain *after*
// the renormalised top-K weights. Multiplies `topk_w[n, k] *=
// per_expert_scale[topk_idx[n, k]]` in place. `per_expert_scale` is
// stored bf16 in the ckpt; we read it bf16 → fp32.
void launch_apply_per_expert_scale_bf16(
    const std::int32_t* topk_idx,        // [N, K]
    float* topk_w,                       // [N, K] in/out
    const void* per_expert_scale_bf16,   // [num_experts] bf16
    int N, int K,
    cudaStream_t stream);

// Nemotron-H router:
//   p = sigmoid(logits)
//   choice = p + correction_bias
//   topk_idx = topk(choice, K)
//   topk_w = p[topk_idx], optionally renormalized, then multiplied by
//            routed_scaling_factor.
//
// This covers the published Nano-Omni config where n_group=topk_group=1.
void launch_topk_sigmoid_bias_bf16(
    const void* logits,                  // [N, num_experts] bf16
    const float* correction_bias,        // [num_experts] fp32
    std::int32_t* topk_idx,              // [N, K]
    float* topk_w,                       // [N, K]
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream);

void launch_topk_sigmoid_bias_fp32(
    const float* logits,                 // [N, num_experts] fp32
    const float* correction_bias,        // [num_experts] fp32
    std::int32_t* topk_idx,              // [N, K]
    float* topk_w,                       // [N, K]
    int N,
    int num_experts,
    int K,
    bool normalize,
    float routed_scaling_factor,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
