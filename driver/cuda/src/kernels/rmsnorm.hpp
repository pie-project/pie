#pragma once

// Row-wise RMSNorm: y[n, h] = x[n, h] * rsqrt(mean(x[n,:]^2) + eps) * weight[h].
//
// Designed for the Qwen-style transformer block. Input/output are bf16 row-
// major contiguous; weight is bf16, length = hidden.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_rmsnorm_bf16(
    const void* x,        // [num_rows, hidden]
    const void* weight,   // [hidden]
    void* y,              // [num_rows, hidden]
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

// Fused pre-norm TP helper:
//   hidden = round_bf16(hidden + residual)
//   norm_out = rmsnorm(hidden, weight)
// The hidden update matches launch_residual_add_bf16's bf16 rounding before
// the norm pass, so it is numerically equivalent to the two-kernel sequence.
void launch_residual_add_rmsnorm_bf16(
    void* hidden,          // [num_rows, hidden_size] bf16, in-place
    const void* residual,  // [num_rows, hidden_size] bf16
    const void* weight,    // [hidden_size]
    void* norm_out,        // [num_rows, hidden_size] bf16
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream);

// Fused Gemma4 end-of-layer helper:
//   hidden = round_bf16(round_bf16(hidden + residual) * round_bf16(scale))
//   norm_out = rmsnorm(hidden, next_weight)
// This matches the separate PLE residual add, layer scalar, and next-layer
// attention pre-norm sequence while avoiding two extra full-row passes.
void launch_residual_add_scale_rmsnorm_bf16(
    void* hidden,
    const void* residual,
    float scale,
    const void* next_weight,
    void* norm_out,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream);

// Fuses:
//   tmp = rmsnorm(x, weight)
//   hidden = round_bf16(hidden + tmp)
// preserving the bf16 tmp materialization of the unfused sequence.
void launch_rmsnorm_residual_add_bf16(
    const void* x,
    const void* weight,
    void* hidden,
    int num_rows,
    int hidden_size,
    float eps,
    cudaStream_t stream);

// Fuses:
//   tmp = rmsnorm(x, weight)
//   hidden = round_bf16(round_bf16(hidden + tmp) * round_bf16(scale))
//   norm_out = rmsnorm(hidden, next_weight)
// This is the exact fused form of Gemma4's PLE post-norm, residual add,
// layer scalar, and next-layer attention pre-norm sequence.
void launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
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
void launch_rmsnorm_gemma_bf16(
    const void* x,
    const void* weight,
    void* y,
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

// RMSNorm with no learnable scale (gamma == 1). Used by Gemma-4's
// V-Norm — `v / rms(v)` per-head, no weight. Equivalent to running
// `launch_rmsnorm_bf16` against an all-ones weight tensor, but
// allocation-free.
void launch_rmsnorm_no_scale_bf16(
    const void* x,
    void* y,
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

// RMSNorm fused with sigmoid-gating (Qwen3.5 GatedDeltaNet's `norm`
// step on `core_attn_out`). Per-row:
//
//   x_hat = x * rsqrt(mean(x^2) + eps)
//   y     = weight * x_hat * silu(gate)
//
// Plain weight (no `1+w` convention). `gate` matches `x` in shape.
void launch_rmsnorm_gated_bf16(
    const void* x,
    const void* gate,
    const void* weight,
    void* y,
    int num_rows,
    int hidden,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
