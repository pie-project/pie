#pragma once

// Row-wise RMSNorm: y[n, h] = x[n, h] * rsqrt(mean(x[n,:]^2) + eps) * weight[h].
//
// Designed for the Qwen-style transformer block. Input/output are bf16 row-
// major contiguous; weight is bf16, length = hidden.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::norm {

void rmsnorm_bf16(
    const void* x,        // [num_rows, hidden]
    const void* weight,   // [hidden]
    void* y,              // [num_rows, hidden]
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

// RMSNorm over the leading `hidden` columns of rows that may be wider than
// they are normalized. Kimi's fused `q_a + kv_a` projection lands both halves
// in one row-major buffer, and the q half is normalized in place from there
// rather than compacted first.
// As above, plus an fp16 copy of the result for a consumer that wants fp16.
// `y_fp16` may be null, in which case this is exactly `rmsnorm_bf16`.
// Sweep entry point for the microbenchmark: block width chosen explicitly.

void rmsnorm_bf16_with_fp16(
    const void* x, const void* weight, void* y, void* y_fp16,
    int num_rows, int hidden, float eps, cudaStream_t stream);

void rmsnorm_strided_bf16(
    const void* x,        // [num_rows, x_row_stride], reads [:, :hidden]
    const void* weight,   // [hidden]
    void* y,              // [num_rows, y_row_stride], writes [:, :hidden]
    int num_rows,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps,
    cudaStream_t stream);

// Fused pre-norm TP helper:
//   hidden = round_bf16(hidden + residual)
//   norm_out = rmsnorm(hidden, weight)
// The hidden update matches residual_add_bf16's bf16 rounding before
// the norm pass, so it is numerically equivalent to the two-kernel sequence.

// Fused Gemma4 end-of-layer helper:
//   hidden = round_bf16(round_bf16(hidden + residual) * round_bf16(scale))
//   norm_out = rmsnorm(hidden, next_weight)
// This matches the separate PLE residual add, layer scalar, and next-layer
// attention pre-norm sequence while avoiding two extra full-row passes.

// Fuses:
//   tmp = rmsnorm(x, weight)
//   hidden = round_bf16(hidden + tmp)
// preserving the bf16 tmp materialization of the unfused sequence.

// Fuses:
//   tmp = rmsnorm(x, weight)
//   hidden = round_bf16(round_bf16(hidden + tmp) * round_bf16(scale))
//   norm_out = rmsnorm(hidden, next_weight)
// This is the exact fused form of Gemma4's PLE post-norm, residual add,
// layer scalar, and next-layer attention pre-norm sequence.
void rmsnorm_residual_add_scale_rmsnorm_bf16(
    const void* x,
    const void* weight,
    void* hidden,
    float scale,
    const void* next_weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream);

// Gemma family RMSNorm — applies `(1 + w) * x_hat` instead of `w * x_hat`.
// HF stores Gemma's RMSNorm gamma centered at zero; this lets the loaded
// tensor be inspected/initialized like a residual gate, but downstream
// math expects the +1 shift.
// Sweep entry point for the microbenchmark: block width chosen explicitly.


// RMSNorm with no learnable scale (gamma == 1). Used by Gemma-4's
// V-Norm — `v / rms(v)` per-head, no weight. Equivalent to running
// `rmsnorm_bf16` against an all-ones weight tensor, but
// allocation-free.

// RMSNorm fused with sigmoid-gating (Qwen3.5 GatedDeltaNet's `norm`
// step on `core_attn_out`). Per-row:
//
//   x_hat = x * rsqrt(mean(x^2) + eps)
//   y     = weight * x_hat * silu(gate)
//
// Plain weight (no `1+w` convention). `gate` matches `x` in shape.

// Same as `rmsnorm_gated_bf16` but `x` is fp32 — used by the
// Qwen3.5 linear-attention path where the GDN recurrent step outputs
// fp32 and we want to drop the separate fp32→bf16 conversion launch.
void rmsnorm_gated_fp32_in_bf16(
    const void* x,            // fp32, shape [num_rows, hidden]
    const void* gate,         // bf16, same shape as `x`
    const void* weight,       // fp32, [hidden]
    void* y,                  // bf16, [num_rows, hidden]
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::norm
