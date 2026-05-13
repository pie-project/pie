#pragma once

// Fuses the per-head Q/K RMSNorm pair (Qwen3 / Gemma-3 / OLMo-3
// pattern) with the immediately-following RoPE rotation into a single
// kernel launch per layer. Replaces three separate kernels in the
// transformer forward:
//
//     launch_rmsnorm_bf16(q, q_norm_weight, ...)
//     launch_rmsnorm_bf16(k, k_norm_weight, ...)
//     launch_rope_bf16(q, k, positions, ...)
//
// The fused kernel:
//   * has one block per (token, head) — total = N × (num_q + num_kv)
//   * each block has `head_dim` threads
//   * loads its head's d elements from gmem, reduces variance in a
//     block-wide sum, scales by rsqrt(var/d + eps) * weight, stages
//     the normalized vector in shared memory, then applies the
//     standard split-half RoPE pair-wise.
//
// vLLM's equivalent is `fused_qk_norm_rope` (csrc/layernorm_kernels);
// the pie implementation matches the existing pie RoPE convention
// (pair offset = head_dim/2, frequency = theta^(-2i/head_dim),
// rotated as `(a*cos - b*sin, b*cos + a*sin)`), so no parity drift.
//
// Falls back to separate kernels for any arch that uses YaRN /
// partial-rotary / non-RMS-norm — those paths still call the
// individual `launch_rmsnorm_bf16` / `launch_rope_bf16` entries.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_fused_qk_norm_rope_bf16(
    void*       q,             // bf16 [N, num_q_heads * head_dim], in-place
    void*       k,             // bf16 [N, num_kv_heads * head_dim], in-place
    const std::int32_t* positions,  // [N]
    const void* q_norm_weight, // bf16 [head_dim]
    const void* k_norm_weight, // bf16 [head_dim]
    int N,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    float rope_theta,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
