#pragma once

// Llama-3 YaRN RoPE scaling — applied to Q and K in place. The per-pair
// base frequency theta^(-2i/head_dim) is rescaled by a piecewise-linear
// interpolation between full scale (high-frequency pairs, untouched) and
// `factor`-scaled (low-frequency pairs), with a smooth band in between.
// Pass `factor = 1.0f` for the un-scaled base RoPE. NeoX half/half pairing
// (dim i with i+d/2; Llama-3 / Mistral / OLMo). Launcher decl; body lifted
// verbatim from driver/cuda/src/kernels/rope.cu (launch_rope_yarn_bf16,
// renamed rope_yarn_bf16; the rotate_pair + yarn_freq helpers are pulled in
// as a self-contained private copy).
//
// Layout: q [num_tokens, num_q_heads, head_dim], k [num_tokens, num_kv_heads,
// head_dim], all bf16 row-major. positions: [num_tokens] int32 (device).
//
// Per-frequency formula:
//   wavelen = 2π / base_freq
//   low_w   = original_max_position / low_freq_factor
//   high_w  = original_max_position / high_freq_factor
//   if wavelen < high_w:  freq = base_freq                 (no scale)
//   else if wavelen > low_w: freq = base_freq / factor     (full scale)
//   else smooth = (original_max_position / wavelen - low_freq_factor) /
//                 (high_freq_factor - low_freq_factor)
//        freq = (1 - smooth) * base_freq / factor + smooth * base_freq

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

void rope_yarn_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    int original_max_position,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
