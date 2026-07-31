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

// Build a per-token standard RoPE table. Layout is [num_tokens, head_dim]:
// row[0:head_dim/2] contains cos, row[head_dim/2:head_dim] contains sin.
void launch_rope_standard_table(
    const std::int32_t* positions,
    float* table,
    int num_tokens,
    int head_dim,
    float theta,
    cudaStream_t stream);

// `interleaved=false` uses the half/half (NeoX) pairing (dim i with i+d/2),
// used by Llama/Qwen/DeepSeek/Kimi. `interleaved=true` uses the GPT-J pairing
// (adjacent dims 2i, 2i+1), required by GLM (config `rope_interleave=true`).
void launch_rope_bf16(
    void* q, void* k,
    const std::int32_t* positions,  // [num_tokens]
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    cudaStream_t stream,
    bool interleaved = false);

// Fused per-head Q/K RMSNorm + standard RoPE. This matches models such as
// Qwen3 where q_norm/k_norm have shape [head_dim] and RoPE is the standard
// first-half/second-half pairing.
void launch_qk_rmsnorm_rope_bf16(
    void* q,
    void* k,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream);

// Same fused Q/K RMSNorm + standard RoPE, but preserves the bf16
// materialization point of the unfused sequence:
//   q = bf16(rmsnorm(q)); k = bf16(rmsnorm(k)); rope(q, k)
// Gemma-4 parity is sensitive to this rounding boundary.
void launch_qk_rmsnorm_rope_bf16_rounded(
    void* q,
    void* k,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    cudaStream_t stream);

// Fused per-head Q/K RMSNorm + interleaved M-RoPE (Qwen3-VL text tower).
// `positions` is `[num_tokens, 3]` row-major: the (t, h, w) M-RoPE component
// for each token. The frequency axis per rotary index follows HF's
// `apply_interleaved_mrope` with section split (t, h, w). Text-only rows pass
// t == h == w and collapse to ordinary RoPE. Matches Qwen3's per-head q/k norm
// (weight shape [head_dim]) and the standard half/half rotate_half pairing.
void launch_qk_rmsnorm_mrope_bf16(
    void* q,
    void* k,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,  // [num_tokens, 3] (t,h,w)
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    int mrope_section_t,
    int mrope_section_h,
    int mrope_section_w,
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

// Original YaRN (Peng et al. 2023) RoPE scaling. Used by OLMo-3 and
// gpt-oss. Differs from the Llama-3 variant in two ways:
//   1. The interpolation ramp is over **dim index**, not over wavelen.
//      `low_dim`, `high_dim` are derived from `beta_slow`, `beta_fast`
//      (target rotations within `original_max_pos`):
//          correction_dim(rot) = d * ln(max_pos / (rot * 2π)) / (2 * ln(theta))
//          low_dim  = floor(correction_dim(beta_slow))   // "low rot"
//          high_dim = ceil(correction_dim(beta_fast))    // "high rot"
//      Below low_dim: keep base inv_freq (extrapolation). Above
//      high_dim: divide by `factor` (interpolation). Linear blend in
//      the middle.
//   2. An attention-magnitude scale `attention_factor` (a.k.a. "mscale")
//      is multiplied into both cos and sin so that the post-rotation
//      magnitude scales attention scores. HF passes 1.21 for
//      `factor=8` on OLMo-3.
void launch_rope_yarn_original_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float factor,
    float beta_fast,
    float beta_slow,
    float attention_factor,
    int   original_max_position,
    cudaStream_t stream);

// Partial rotary embedding (Gemma-4 full-attention layers). Rotates
// only the first `rotary_dim` of each head's `head_dim` channels;
// the trailing `head_dim - rotary_dim` channels pass through
// unchanged. The pair convention is HF's (i, i + rotary_dim/2).
void launch_rope_partial_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream);

void launch_rope_partial_bf16_position_delta(
    void* q, void* k,
    const std::int32_t* positions,
    int position_delta,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream);

// Partial rotary embedding on the LAST `rotary_dim` dimensions of each head.
// Used by DeepSeek V4 where RoPE is applied to the trailing 64 dims of
// head_dim=512. Pair convention: ADJACENT dims (offset+2p, offset+2p+1)
// — the GPT-J / interleaved layout that DeepSeek checkpoints are trained
// with (reference `rope_tail_ext_inplace`), NOT the half/half NeoX split.
//
// Optional YaRN long-context interpolation (DSv4 compressed layers):
// pass `freq_scale = 1/factor` and `ext_factor = 1` to blend each pair's
// angle between interpolation and extrapolation with the beta_fast /
// beta_slow correction ramp. The DSv4 reference cancels YaRN's magnitude
// scale, so no attention factor is applied. Defaults (`freq_scale = 1`,
// `ext_factor = 0`) give plain unscaled RoPE.
void launch_rope_partial_last_bf16(
    void* q, void* k,
    const std::int32_t* positions,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    cudaStream_t stream,
    bool inverse = false,
    float freq_scale = 1.0f,
    float ext_factor = 0.0f,
    float beta_fast = 32.0f,
    float beta_slow = 1.0f,
    int original_max_position = 0);

}  // namespace pie_cuda_driver::kernels
