#pragma once

// RoPE applied to Q and K in place. `interleaved=false` → NeoX half/half
// pairing (dim i with i+d/2; Llama/Qwen/DeepSeek); `true` → GPT-J adjacent
// pairing (2i, 2i+1; GLM). Per-pair frequency theta^(-2i/head_dim).
// Launcher decl; body lifted verbatim from driver/cuda/src/kernels/rope.cu
// (base launch_rope_bf16). YaRN / partial / fused-qknorm variants are
// lifted when the archs that need them land.
//
// Layout: q [num_tokens, num_q_heads, head_dim], k [num_tokens, num_kv_heads,
// head_dim], all bf16 row-major. positions: [num_tokens] int32 (device).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void rope_bf16(void* q, void* k, const std::int32_t* positions,
               int num_tokens, int num_q_heads, int num_kv_heads,
               int head_dim, float theta, bool interleaved, cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
