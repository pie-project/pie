#pragma once

// Partial rotary position embedding (Gemma-4 full-attention layers).
// Rotates only the first `rotary_dim` of each head's `head_dim` channels;
// the trailing `head_dim - rotary_dim` channels pass through unchanged.
// Launcher decl; body lifted verbatim from
// driver/cuda/src/kernels/rope.cu (launch_rope_partial_bf16 +
// rope_partial_bf16_kernel).
//
// HF reference pairing: the pair offset is the *full* `head_dim/2`
// (NeoX-style: dim `i` with `i + head_dim/2`), but the rotation angle is
// only non-zero for the first `rotary_dim/2` pairs — frequency
// theta^(-2i/head_dim). Pairs with `i >= rotary_dim/2` get the identity
// rotation (cos=1, sin=0) and are skipped, so those channels (and their
// upper-half mates) pass through unchanged.
//
// Layout: q [num_tokens, num_q_heads, head_dim], k [num_tokens, num_kv_heads,
// head_dim], all bf16 row-major. positions: [num_tokens] int32 (device).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void rope_partial_bf16(void* q, void* k, const std::int32_t* positions,
                       int num_tokens, int num_q_heads, int num_kv_heads,
                       int head_dim, int rotary_dim, float theta,
                       cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
