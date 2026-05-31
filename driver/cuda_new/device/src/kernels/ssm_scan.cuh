#pragma once

// Mamba-2 / SSD selective-scan recurrence — the core state-space step that
// powers Nemotron-H (and the Mamba-2 family generally). This is the per-channel
// linear recurrence applied after the depthwise causal conv:
//
//     dt   = max( softplus(dt_raw + dt_bias_h),  time_step_min )   [per (token, head)]
//     dA   = exp(dt * A_h)                                          [per (token, head)]
//     for each head h, head-dim channel `dim`, state index s:
//         h_state[dim, s] = h_state[dim, s] * dA + (dt * B[s]) * x[dim]
//         y[dim]         += h_state[dim, s] * C[s]
//     y[dim]             += D_h * x[dim]
//
// where the recurrence runs sequentially over tokens within a request (token
// order preserved), and B/C are shared across the heads in a group (grouped /
// "n_groups" Mamba-2 B,C broadcast). State accumulates in fp32; inputs/outputs
// and the persisted state cache are bf16.
//
// Lifted from driver/cuda/src/kernels/nemotron_h.cu — specifically the
// `mamba_ssm_batched_warp_kernel` (the decode/general warp-reduction variant of
// `launch_nemotron_mamba_ssm_batched_bf16`). One CUDA block owns one
// (request, mamba-head) stream; one warp maps to one head-dim channel and
// reduces over the state axis. This is the canonical recurrence kernel.
//
// INTENTIONALLY LEFT BEHIND (lives with the rest of the Nemotron-H state
// machinery in a later focused pass, not part of this scan lift):
//   * mamba_ssm_batched_prefill_reg_kernel  — register-resident prefill
//     specialization (same math, keeps state in registers across the span;
//     bounded by kMaxStatePerLane=8 i.e. state_size<=256). Perf variant only.
//   * mamba_ssm_batched_kernel              — shared-memory atomicAdd fallback
//     (same math). Perf variant only.
//   * mamba_split_kernel / mamba_split_conv_dt_kernel — in-projection splits.
//   * prepare_mamba_params_kernel (A = -exp(A_log), D, dt_bias to fp32) and
//     prepare_mamba_dt_da_kernel (dt/dA precompute) — host-side param prep.
//   * zamba_rmsnorm_gated / MoE pointer builders — unrelated ops.
// The verbatim recurrence math of all three scan variants is identical; we lift
// the warp variant as the single clean entry point. dt_precomputed/dA_precomputed
// fast paths are preserved (pass nullptr to compute inline from dt/A/dt_bias).
//
// Tensor layouts (token-major / "channels-last", matching upstream):
//
//   conv_out : [N, conv_dim] bf16  — packed post-conv activations, where for
//              row (token) `row`:
//                  x[h, dim] = conv_out[row, h*head_dim + dim]
//                  B[g, s]   = conv_out[row, intermediate + g*state_size + s]
//                  C[g, s]   = conv_out[row, intermediate + n_groups*state_size
//                                            + g*state_size + s]
//              with conv_dim = intermediate + 2*n_groups*state_size,
//              intermediate = num_heads*head_dim, group g = h / (num_heads/n_groups).
//   dt       : [N, num_heads] bf16 — raw dt (used only if dt_precomputed==nullptr).
//   A        : [num_heads] fp32    — A_h = -exp(A_log_h) (precomputed by host).
//   D        : [num_heads] fp32    — skip term.
//   dt_bias  : [num_heads] fp32.
//   dt_precomputed : [N, num_heads] fp32, nullable — softplus'd dt.
//   dA_precomputed : [N, num_heads] fp32, nullable — exp(dt*A).
//   ssm_state_base : [num_slots, num_heads, head_dim, state_size] bf16 — the
//              persistent recurrent state cache; read in / written out in place.
//   slot_ids : [R] int32, nullable — slot for request r (nullptr => slot 0).
//   qo_indptr: [R+1] uint32 — token range [qo_indptr[r], qo_indptr[r+1]) for
//              request r (CSR-style), so a request's tokens are scanned in order.
//   y        : [N, intermediate] bf16 — scan output (y[row, h*head_dim + dim]).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// Batched Mamba-2 / SSD selective scan. Runs R requests; each request's tokens
// (delimited by qo_indptr) are scanned sequentially, carrying / updating the
// per-slot recurrent state in ssm_state_base. Pass dt_precomputed/dA_precomputed
// = nullptr to compute dt = max(softplus(dt + dt_bias), time_step_min) and
// dA = exp(dt*A) inline.
void ssm_selective_scan_bf16(
    const void*          conv_out,        // [N, conv_dim] bf16
    const void*          dt,              // [N, num_heads] bf16
    const float*         A,               // [num_heads] fp32, -exp(A_log)
    const float*         D,               // [num_heads] fp32
    const float*         dt_bias,         // [num_heads] fp32
    const float*         dt_precomputed,  // [N, num_heads] fp32, nullable
    const float*         dA_precomputed,  // [N, num_heads] fp32, nullable
    void*                ssm_state_base,  // [slots, heads, head_dim, state] bf16
    const std::int32_t*  slot_ids,        // [R] int32, nullable
    const std::uint32_t* qo_indptr,       // [R+1] uint32
    void*                y,               // [N, intermediate] bf16
    int                  R,
    int                  num_heads,
    int                  head_dim,
    int                  state_size,
    int                  n_groups,
    int                  conv_dim,
    int                  intermediate,
    float                time_step_min,
    cudaStream_t         stream);

}  // namespace pie_cuda_device::kernels
