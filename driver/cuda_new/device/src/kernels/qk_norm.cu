#include "qk_norm.cuh"

#include "rmsnorm.cuh"

// Per-head q/k RMSNorm (Qwen3). The q / k tensors are row-major contiguous in
// head_dim, so each (token,head) head-vector is exactly one RMSNorm "row" of
// width head_dim. rmsnorm_bf16 RMSNorms each [hidden]-wide row independently and
// applies a [hidden]-wide gain; reusing the [head_dim] gain across every
// (token,head) row gives the shared per-head gain Qwen3 wants. So this is just
// rmsnorm_bf16 invoked twice in place: once for q (num_rows = T*nq), once for k
// (num_rows = T*nkv), both with hidden = head_dim. That is the simplest correct
// implementation and is exercised by qk_norm_selftest.cu.

namespace pie_cuda_device::kernels {

void qk_norm_bf16(void* q, void* k, const void* q_weight, const void* k_weight,
                  int num_tokens, int num_q_heads, int num_kv_heads, int head_dim,
                  float eps, cudaStream_t stream) {
    // q: one RMSNorm row per (token, q_head), width head_dim, shared gain.
    rmsnorm_bf16(q, q_weight, q, num_tokens * num_q_heads, head_dim, eps, stream);
    // k: one RMSNorm row per (token, kv_head), width head_dim, shared gain.
    rmsnorm_bf16(k, k_weight, k, num_tokens * num_kv_heads, head_dim, eps, stream);
}

}  // namespace pie_cuda_device::kernels
