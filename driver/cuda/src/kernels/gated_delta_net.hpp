#pragma once

// Gated DeltaNet — linear-attention recurrence used by Qwen3.5's
// `linear_attention` layers. Mirrors HF's `torch_recurrent_gated_delta_rule`
// (decode, T=1) and `torch_chunk_gated_delta_rule` (prefill, T>1).
//
// State per (request, layer): `state[V_h, K_d, V_d]` fp32 — running
// linear-attention memory. Persisted across decode steps.
//
// Per-step recurrence (decode):
//
//     state ← state * exp(g_h)
//     kv_mem[v]  = Σ_k state[k, v] * k_t[k]
//     delta[v]   = (v_t[v] − kv_mem[v]) * β_h
//     state[k,v] ← state[k,v] + k_t[k] * delta[v]
//     out[v]     = Σ_k state[k, v] * q_t[k]
//
// q_t / k_t are L2-normalised along K_d *before* this kernel; q_t is
// also pre-scaled by 1/√K_d. β is sigmoid'd; g is the raw per-head log
// already (−A_h · softplus(a_h + dt_bias_h) form), and gets exp() inside.
//
// All numerics in fp32 — bf16 inputs are widened on load; the state is
// always fp32 to avoid drift across many recurrent steps.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Single-token decode step. One block per (request, head).
//
//     q_norm, k_norm : [B, V_h, K_d] fp32 (L2-normalised; q pre-scaled)
//     v              : [B, V_h, V_d] fp32
//     g_log          : [B, V_h]      fp32 (raw — kernel applies exp)
//     beta           : [B, V_h]      fp32 (already sigmoid'd)
//     state          : [B, V_h, K_d, V_d] fp32 — updated in-place
//     out            : [B, V_h, V_d] fp32
//
// The kernel allocates K_d + V_d fp32 entries of shared memory.
void launch_recurrent_gated_delta_step(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state,
    float*       out,
    int B, int V_h, int K_d, int V_d,
    cudaStream_t stream);

// Chunked prefill (T tokens at a time, per request). Mirrors
// `torch_chunk_gated_delta_rule` — see the HF reference for the exact
// recurrence. Reuses the same per-request `state` buffer; final state
// after T steps is left in `state`.
//
//     q_norm, k_norm : [T, V_h, K_d] fp32 (already L2-normalised + q-scaled)
//     v              : [T, V_h, V_d] fp32
//     g_log          : [T, V_h]      fp32
//     beta           : [T, V_h]      fp32
//     state          : [V_h, K_d, V_d] fp32 — updated in-place
//     out            : [T, V_h, V_d] fp32
//     chunk_size     : 64 by default (must divide T after padding)
//
// Single-request only for now (no batched-prefill); when serving
// multiple requests the caller invokes this once per request.
void launch_chunk_gated_delta_prefill(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state,
    float*       out,
    int T, int V_h, int K_d, int V_d,
    int chunk_size,
    cudaStream_t stream);

// L2-normalise rows of `[N, hidden]` bf16, optionally scale each row
// element by `scale` after normalisation, and emit fp32 output. Used
// to prep Q/K for the gated-delta step.
//
//     y[r, c] = (x[r, c] / sqrt(Σ x[r, .]^2 + eps)) * scale
void launch_l2norm_scale_bf16_to_fp32(
    const void* x,        // [N, hidden] bf16
    float*      y,        // [N, hidden] fp32
    int N, int hidden,
    float scale,
    float eps,
    cudaStream_t stream);

// Helper: bf16 → fp32 widen (vec cast). For passing v_t to the kernel
// when v lives in bf16 in workspace.
void launch_bf16_to_fp32(
    const void* x, float* y, std::size_t n, cudaStream_t stream);

// repeat_interleave on the head dimension: duplicate each of K_h heads
// `repeat` times so the result has V_h = K_h * repeat heads. Mirrors
// HF's `query.repeat_interleave(V_h // K_h, dim=2)`.
//
//     in  : [N, K_h, D] fp32
//     out : [N, V_h, D] fp32   where  out[n, h_v, d] = in[n, h_v / repeat, d]
void launch_repeat_interleave_heads_fp32(
    const float* in, float* out,
    int N, int K_h, int V_h, int D,
    cudaStream_t stream);

// fp32 → bf16 narrow. For shipping the recurrent kernel's fp32 output
// into the bf16 workspace consumed by the post-norm + out_proj stage.
void launch_fp32_to_bf16(
    const float* x, void* y, std::size_t n, cudaStream_t stream);

// Compute per-step (g_log, beta) from the four small per-head inputs:
//
//     g_log = -exp(A_log) * softplus(a + dt_bias)
//     beta  = sigmoid(b)
//
//     A_log   : [V_h]    bf16   (learned, layer-shared across steps)
//     dt_bias : [V_h]    bf16   (learned, layer-shared)
//     a       : [N, V_h] bf16   (per-token from in_proj_a)
//     b       : [N, V_h] bf16   (per-token from in_proj_b)
//     g_log_out, beta_out : [N, V_h] fp32
void launch_gated_delta_g_beta(
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    float*      g_log_out,
    float*      beta_out,
    int N, int V_h,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
