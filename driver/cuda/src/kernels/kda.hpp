#pragma once

// KDA — Kimi Delta Attention, the linear-attention half of Kimi-K3.
//
// The recurrence is the same gated delta rule Qwen3.5 runs
// (`gated_delta_net.hpp`), with one difference that changes every kernel:
// **the decay is per key channel, not per head**. Qwen3.5 multiplies the whole
// [K_d, V_d] state slab by one scalar `exp(g_h)`; KDA multiplies column `k` by
// `exp(gate[h, k])`. That is the "delta" in the name -- a fine-grained forget
// gate -- and it is why these kernels exist beside the GDN ones instead of
// reusing them with a broadcast.
//
// Per-token recurrence, per (request, head):
//
//     q ← l2norm(q) * K_d^(-1/2)      (done by the caller)
//     k ← l2norm(k)
//     state[k, v] ← state[k, v] * exp(gate[k])
//     delta[v]    = (v[v] − Σ_k state[k, v] * k[k]) * beta
//     state[k, v] ← state[k, v] + k[k] * delta[v]
//     out[v]      = Σ_k state[k, v] * q[k]
//
// and the gate itself is, per (token, head, channel):
//
//     a         = exp(A_log[h])                    (A_log is per head)
//     gate      = lower_bound * sigmoid(a * (raw_g + dt_bias[h, d]))
//                 when a lower bound is configured (K3 ships −5.0), else
//               = −a * softplus(raw_g + dt_bias[h, d])
//     beta      = sigmoid(raw_beta[h])
//
// All arithmetic is fp32; bf16 inputs are widened first.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Gate + beta preprocessing.
//
//     raw_g     : [T, H*D] bf16 — f_b_proj(f_a_proj(x))
//     raw_beta  : [T, H]   bf16 — b_proj(x)
//     A_log     : [H]      fp32 — per-head, already sliced to this rank
//     dt_bias   : [H*D]    fp32 — per-head-channel, sliced to this rank
//     gate_out  : [T, H*D] fp32
//     beta_out  : [T, H]   fp32
//
// `lower_bound` of 0 selects the softplus form; any negative value selects the
// bounded sigmoid form and is used as the bound.
void launch_kda_gate_beta_bf16(
    const void*  raw_g,
    const void*  raw_beta,
    const float* A_log,
    const float* dt_bias,
    float*       gate_out,
    float*       beta_out,
    int T, int H, int D,
    float lower_bound,
    cudaStream_t stream);

// Single-token decode over R requests, one state slot each.
//
//     q_norm, k_norm : [R, H, D]                fp32 (l2-normed; q pre-scaled)
//     v              : [R, H, D]                fp32
//     gate           : [R, H, D]                fp32 (log-space, per channel)
//     beta           : [R, H]                   fp32
//     state_base     : [num_slots, H, D, D]     fp32 — slot 0
//     slot_ids       : [R] int32 device
//     slot_stride_elems : H * D * D
//     out            : [R, H, D]                fp32
//
// The state is laid out `[H][v][k]` — v-major — because that is the layout the
// reference kernels write and the one a per-`v` thread walks contiguously.
void launch_kda_recurrent_step_batched(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* gate,
    const float* beta,
    float*       state_base,
    const std::int32_t* slot_ids,
    long long    slot_stride_elems,
    float*       out,
    int R, int H, int D,
    cudaStream_t stream);

// Multi-request prefill. One block per (request, head); the block walks its
// window from `qo_indptr` one token at a time, because the recurrence has a
// strict per-token state dependency.
//
//     q_norm, k_norm, v, gate : [N_total, H, D] fp32
//     beta                    : [N_total, H]    fp32
//     out                     : [N_total, H, D] fp32
//
// TODO(perf): the chunked form (`chunk_kda`) blocks the recurrence into
// 64-token tiles and turns most of the work into GEMMs, the way
// `launch_chunk_gated_delta_prefill_batched` does for GDN. Sequential is what
// the bring-up needs -- it is the definition of the recurrence, so a parity
// mismatch here is a real mismatch and not a chunking artifact.
void launch_kda_prefill_batched(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* gate,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int H, int D,
    cudaStream_t stream);

// Gated RMSNorm over the head dim with a sigmoid gate, then narrow to bf16:
//
//     y[t, h, d] = rmsnorm(o[t, h, :])[d] * weight[d] * sigmoid(g[t, h, d])
//
//     o      : [T, H, D]   fp32 (the recurrence output)
//     g      : [T, H*D]    bf16 (g_proj(x))
//     weight : [D]         fp32
//     out    : [T, H*D]    bf16
void launch_kda_o_norm_gated_bf16(
    const float* o,
    const void*  g,
    const float* weight,
    void*        out,
    int T, int H, int D,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
