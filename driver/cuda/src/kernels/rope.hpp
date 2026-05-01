#pragma once

// RoPE (Rotary Position Embedding) — applied to Q and K in place.
//
// For each token n at position p, and each head h, the head_dim is split
// into pairs (i, i + d/2). The pair is rotated by angle p / theta^(2i/d).
// (Llama / Qwen convention — pair the first half with the second half,
// not adjacent indices.)
//
// Layout: q  [num_tokens, num_q_heads,  head_dim]
//         k  [num_tokens, num_kv_heads, head_dim]
// All bf16, contiguous, row-major.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_rope_bf16(
    void* q, void* k,
    const std::int32_t* positions,  // [num_tokens]
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream);

// YaRN (Llama-3 / OLMo / Mistral-3 / GPT-OSS) RoPE scaling. Frequency
// per pair is modified by a piecewise-linear interpolation between
// `low_freq_factor` and `high_freq_factor` (in units of cycles / window),
// with overall scaling `factor` for the low-frequency tail. Pass
// `factor = 1.0f` to fall back to the un-scaled RoPE used by Qwen.
//
// Reference: Llama-3 paper, "RoPE scaling"; per-frequency formula:
//   wavelen = 2π / freq
//   low_w  = original_max_pos / low_freq_factor
//   high_w = original_max_pos / high_freq_factor
//   if wavelen < high_w:                no scaling
//   else if wavelen > low_w:            freq /= factor
//   else: smooth interp between the two regimes
void launch_rope_yarn_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    int   original_max_position,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
