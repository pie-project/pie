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

namespace pie_cuda_driver::kernels::moe {

void topk_softmax_bf16(
    const void* logits,        // [N, num_experts] bf16
    std::int32_t* topk_idx,    // [N, K] i32 — expert indices
    float* topk_w,             // [N, K] fp32 — renormalized routing weights
    int N,
    int num_experts,
    int K,
    cudaStream_t stream);

// As above, but selects the implementation explicitly instead of from the
// environment. Exists for the microbenchmark: the production entry point
// caches its env read in a function-local static, so a harness that flips
// PIE_TOPK_WARP between calls silently gets the SAME form twice and its
// comparison passes without having compared anything. `use_warp` is honoured
// only where the warp form applies (K <= 8 and num_experts <= 128); outside
// that range both values give the block form.
void topk_softmax_bf16_form(
    const void* logits,
    std::int32_t* topk_idx,
    float* topk_w,
    int N,
    int num_experts,
    int K,
    bool use_warp,
    cudaStream_t stream);

// MEASURED DEAD END, kept because the reasoning is not obvious and someone
// will try it again: fusing the router projection into the top-K that consumes
// it costs 2x. The top-K has to see every expert to pick from them, so it is
// one block per token -- and folding the projection in drags it down to that
// same one block, from the 32 the standalone GEMV gets. Trading 32 SMs for a
// saved launch is not a trade. gpt-oss measured 291 -> 134 tok/s.
void router_topk_softmax_bf16(
    const void* act,            // [N, hidden] bf16
    const void* router_weight,  // [num_experts, hidden] bf16
    const void* router_bias,    // [num_experts] bf16, or null
    std::int32_t* topk_idx,     // [N, K]
    float* topk_w,              // [N, K]
    int N,
    int num_experts,
    int K,
    int hidden,
    cudaStream_t stream);

// Gemma-4 26B-A4B's router applies a per-expert scalar gain *after*
// the renormalised top-K weights. Multiplies `topk_w[n, k] *=
// per_expert_scale[topk_idx[n, k]]` in place. `per_expert_scale` is
// stored bf16 in the ckpt; we read it bf16 → fp32.
void apply_per_expert_scale_bf16(
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
void topk_sigmoid_bias_bf16(
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

void topk_sigmoid_bias_fp32(
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

}  // namespace pie_cuda_driver::kernels::moe
