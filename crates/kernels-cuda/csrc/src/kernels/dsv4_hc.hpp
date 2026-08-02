#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// HC pre: extract one stream for attention/FFN from the multi-stream residual.
//
// 1. GEMM: mixes = rmsnorm(residual_flat) @ fn^T  (done externally via cuBLAS)
// 2. This kernel processes the GEMM output:
//    - pre_mix = sigmoid(mixes[:, :M] * scale[0] + base[:M]) + eps
//    - post_mix = sigmoid(mixes[:, M:2M] * scale[1] + base[M:2M]) * alpha
//    - comb_mix = sinkhorn(softmax(mixes[:, 2M:].reshape(M,M) * scale[2] + base[2M:]))
//    - layer_input = sum_i(pre_mix_i * residual_i) → [N, H]
//
// mixes_and_residual layout:
//   mixes: [N, mix_hc] F32 (GEMM output, already RMSNorm'd)
//   residual: [N, hc_mult, H] BF16 (multi-stream residual)
void launch_hc_pre_postprocess_bf16(
    const float* mixes,           // [N, mix_hc] GEMM output
    const float* scale,           // [3]
    const float* base,            // [mix_hc]
    const void* residual,         // [N, hc_mult, H] BF16
    float* post_mix,              // [N, hc_mult] output
    float* comb_mix,              // [N, hc_mult, hc_mult] output
    void* layer_input,            // [N, H] BF16 output
    int N,
    int hc_mult,
    int hidden_size,
    float hc_eps,
    float hc_post_alpha,          // typically 2.0
    int sinkhorn_iters,
    cudaStream_t stream);

// HC post: combine layer output with multi-stream residual.
//   new_residual_j = comb_mix_{ij} * residual_i + post_mix_j * x
void launch_hc_post_bf16(
    const void* x,                // [N, H] BF16 (layer output)
    const void* residual,         // [N, hc_mult, H] BF16 (current residual)
    const float* post_mix,        // [N, hc_mult]
    const float* comb_mix,        // [N, hc_mult, hc_mult]
    void* out_residual,           // [N, hc_mult, H] BF16 output
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream);

// HC head: collapse multi-stream residual to single stream.
//   out = sum_i(gate_i * residual_i)
// where gate = sigmoid(rmsnorm(residual_flat) @ fn^T * scale + base)
void launch_hc_head_postprocess_bf16(
    const float* mixes,           // [N, hc_mult] GEMM output (RMSNorm'd)
    const float* scale,           // [1]
    const float* base,            // [hc_mult]
    const void* residual,         // [N, hc_mult, H] BF16
    void* out,                    // [N, H] BF16 output
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream,
    float hc_eps = 1e-6f);

// Expand embedding [N, H] → [N, hc_mult, H] by replicating.
void launch_hc_expand_bf16(
    const void* input,            // [N, H] BF16
    void* output,                 // [N, hc_mult, H] BF16
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream);

// RMSNorm for the flattened HC residual → F32.
// input [N, hc_mult * H] BF16 → output [N, hc_mult * H] F32
void launch_hc_rmsnorm_to_f32(
    const void* input,            // [N, hc_mult * H] BF16
    float* output,                // [N, hc_mult * H] F32
    int N,
    int dim,                      // hc_mult * H
    float eps,
    cudaStream_t stream);

// Post-hoc attention sink correction: o *= sigmoid(lse - sink_h)
// Applied per-head after standard attention to account for V4's learnable
// denominator extension.
void launch_attn_sink_correction_bf16(
    void* attn_out,               // [N, num_heads, head_dim] BF16 (in-place)
    const float* lse,             // [N, num_heads] FP32
    const float* sink,            // [num_heads] FP32
    int N,
    int num_heads,
    int head_dim,
    cudaStream_t stream);

// Per-head RMSNorm without gamma weight: q *= rsqrt(mean(q^2) + eps).
// Used by V4 attention on Q after wq_b (applied per head independently).
void launch_per_head_rmsnorm_bf16(
    void* q,                      // [N, num_heads, head_dim] BF16 (in-place)
    int N,
    int num_heads,
    int head_dim,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
