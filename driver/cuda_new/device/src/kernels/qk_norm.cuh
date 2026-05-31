#pragma once

// Qwen3-style per-head query/key RMSNorm, applied IN PLACE to q and k BEFORE
// rope. Each head_dim-vector (one per (token, head)) is RMSNorm'd over head_dim
// with a learned per-dim gain (q_norm.weight / k_norm.weight, each [head_dim]).
//
//   y[d] = x[d] * rsqrt(mean_d(x[d]^2) + eps) * weight[d]
//
// i.e. plain RMSNorm over the head_dim axis (same math as rmsnorm_bf16), where
// the "row" is each (token,head) head-vector and the gain is shared across all
// heads. Because the q / k layouts are row-major contiguous in head_dim, this
// is exactly rmsnorm_bf16 invoked with num_rows = num_tokens*num_heads,
// hidden = head_dim, and the [head_dim] gain reused as the per-row weight.
//
// Layout: q [num_tokens, num_q_heads, head_dim], k [num_tokens, num_kv_heads,
// head_dim], all bf16 row-major. q_weight / k_weight: [head_dim] bf16.

#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// q [num_tokens, num_q_heads, head_dim], k [num_tokens, num_kv_heads, head_dim]
// bf16, in place. q_weight/k_weight [head_dim] bf16. Per (token,head) RMSNorm
// over head_dim with the gain.
void qk_norm_bf16(void* q, void* k, const void* q_weight, const void* k_weight,
                  int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
                  float eps, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
