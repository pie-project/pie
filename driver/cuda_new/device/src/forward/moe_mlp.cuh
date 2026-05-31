#pragma once

// MoE MLP sublayer (Qwen3.5-MoE / GPT-OSS style), dense reference form.
//
//   logits = hidden @ router^T
//   (idx, w) = topk_softmax(logits)              # top-K experts + renorm weights
//   for each expert e:  ffn_e = (silu(hidden @ Wgu_e^T[:I]) * (·)[I:]) @ Wdown_e^T
//   out[t] = Σ_k w[t,k] · ffn_{idx[t,k]}[t]
//
// This computes ALL E experts densely and selects the top-K per token — correct
// MoE semantics, O(E) work. The sparse token-dispatch perf path (the banked
// `moe_dispatch` routing kernels: permute tokens to expert-contiguous order,
// run only the routed tokens, scatter back) is the follow-up; this dense form
// is the correctness baseline (naive-attention :: flashinfer).
//
// All bf16. hidden [T,H]; router [E,H]; wgu [E, 2I, H] (gate||up); wdown [E,H,I];
// out [T,H].

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace pie_cuda_device::forward {

cudaError_t moe_mlp_block_bf16(
    cublasHandle_t cublas, cudaStream_t stream,
    const void* hidden, const void* router_w, const void* wgu, const void* wdown, void* out,
    int num_tokens, int hidden_size, int intermediate, int num_experts, int top_k);

}  // namespace pie_cuda_device::forward
