#pragma once

// Gated DeltaNet — linear-attention recurrence used by Qwen3.5's
// `linear_attention` layers. Mirrors HF's `torch_recurrent_gated_delta_rule`
// (decode, T=1) and `torch_chunk_gated_delta_rule` (prefill, T>1).
//
// State per (request, layer): `state[V_h, K_d, V_d]` fp32 —
// running linear-attention memory. Persisted across decode steps.
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
// All arithmetic is fp32. bf16 inputs are widened before the recurrent kernel.

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
void launch_recurrent_gated_delta_step_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state,
    float*       out,
    int B, int V_h, int K_d, int V_d,
    cudaStream_t stream);

// Multi-request batched variant. Same per-(request, head) compute as
// `_step` above; outer R dimension picks per-request inputs/outputs and
// per-request slot in the state pool.
//
//     q_norm, k_norm : [R, V_h, K_d]               fp32
//     v              : [R, V_h, V_d]               fp32
//     g_log, beta    : [R, V_h]                    fp32
//     state_base     : [num_slots, V_h, K_d, V_d]  fp32 — slot 0 ptr
//     slot_ids       : [R] int32, device-resident
//     slot_stride_elems : V_h * K_d * V_d (per-slot state-element stride)
//     out            : [R, V_h, V_d]               fp32
//
// One launch covers all R requests on the decode path; prefer over
// host-looping `_step` per request.
void launch_recurrent_gated_delta_step_batched(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t* slot_ids,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);
void launch_recurrent_gated_delta_step_batched_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t* slot_ids,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);

// Batched decode variant for grouped-query GDN layouts where Q/K have K_h
// heads and V has V_h heads. Avoids materializing repeated Q/K heads.
void launch_recurrent_gated_delta_step_batched_gqa(
    const float* q_norm_kh,
    const float* k_norm_kh,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t* slot_ids,
    long long    slot_stride_elems,
    float*       out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream);
void launch_recurrent_gated_delta_step_batched_gqa_state_bf16(
    const float* q_norm_kh,
    const float* k_norm_kh,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t* slot_ids,
    long long    slot_stride_elems,
    float*       out,
    int R, int K_h, int V_h, int K_d, int V_d,
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
// Single-request entry. Multi-request callers use
// `launch_chunk_gated_delta_prefill_batched` below — this one is kept
// for the legacy parity entrypoint and as a single-request fast path.
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
void launch_chunk_gated_delta_prefill_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state,
    float*       out,
    int T, int V_h, int K_d, int V_d,
    int chunk_size,
    cudaStream_t stream);

// Multi-request batched chunked prefill. One block per (request, head);
// each block walks its T_r-token window from `qo_indptr` sequentially
// (the recurrence has a strict per-token state dependency) and updates
// the request's state slab indexed by `slot_ids[r]`.
//
//     q_norm, k_norm : [N_total, V_h, K_d]          fp32
//     v              : [N_total, V_h, V_d]          fp32
//     g_log, beta    : [N_total, V_h]               fp32
//     state_base     : [num_slots, V_h, K_d, V_d]   fp32 — slot 0 ptr
//     slot_ids       : [R]   int32 device
//     qo_indptr      : [R+1] u32   device
//     slot_stride_elems : V_h * K_d * V_d
//     out            : [N_total, V_h, V_d]          fp32
//
// TODO(perf): on Hopper, flashinfer ships a warp-specialized DeltaRule
// prefill (`flashinfer::flat::launch_delta_rule_prefill_kernel_gbai`,
// SM90-only — its non-Hopper branch throws) with TMA + chunk-level
// parallelism, expected to be substantially faster on prefills > 256
// tokens. Integration is non-trivial: flashinfer wants contiguous
// [num_seqs, V_h, K_d, V_d] state but ours is slot-sparse, and it takes
// bf16 inputs while we widen to fp32 for the l2norm. Either a per-fire
// scatter/gather adapter or a templated flashinfer collective_load that
// takes a slot_ids indirection would do; pre-SM90 hardware always falls
// through to this kernel.
void launch_chunk_gated_delta_prefill_batched(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);
void launch_chunk_gated_delta_prefill_batched_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);

// Small-T variant for target verification. One block per (request, head)
// caches the [K_d, V_d] recurrent state tile in shared memory, walks the
// request's short token window, and writes final state back once. This avoids
// rereading/rewriting the full state for every drafted token.
void launch_chunk_gated_delta_prefill_batched_cached(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);
void launch_chunk_gated_delta_prefill_batched_cached_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);

// Warp-tiled small-T variant. Four warps per block process four V rows for a
// single (request, head), keeping each lane's K-fragment of recurrent state in
// registers across the short verification window.
void launch_chunk_gated_delta_prefill_batched_warp_tiled(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);
void launch_chunk_gated_delta_prefill_batched_warp_tiled_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    cudaStream_t stream);

void launch_chunk_gated_delta_prefill_batched_warp_tiled_snapshot(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    int snapshot_base_slot,
    int snapshot_count,
    cudaStream_t stream);
void launch_chunk_gated_delta_prefill_batched_warp_tiled_snapshot_state_bf16(
    const float* q_norm,
    const float* k_norm,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int V_h, int K_d, int V_d,
    int snapshot_base_slot,
    int snapshot_count,
    cudaStream_t stream);

// Same warp-tiled small-T recurrence, but Q/K are stored with fewer heads
// than V and are repeated logically (`V_h % K_h == 0`). This avoids
// materialising repeat_interleave(Q/K) for GQA-style GDN layers.
void launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    const float* q_norm_kh,
    const float* k_norm_kh,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream);
void launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    const float* q_norm_kh,
    const float* k_norm_kh,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int K_h, int V_h, int K_d, int V_d,
    cudaStream_t stream);

void launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_snapshot(
    const float* q_norm_kh,
    const float* k_norm_kh,
    const float* v,
    const float* g_log,
    const float* beta,
    float*       state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int K_h, int V_h, int K_d, int V_d,
    int snapshot_base_slot,
    int snapshot_count,
    cudaStream_t stream);
void launch_chunk_gated_delta_prefill_batched_warp_tiled_gqa_snapshot_state_bf16(
    const float* q_norm_kh,
    const float* k_norm_kh,
    const float* v,
    const float* g_log,
    const float* beta,
    void*        state_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long    slot_stride_elems,
    float*       out,
    int R, int K_h, int V_h, int K_d, int V_d,
    int snapshot_base_slot,
    int snapshot_count,
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

// Fused Qwen GDN post-conv prep:
//   q/k split + L2 normalization, v bf16-to-fp32, and g/beta gating.
// `qkv_post` is [N, 2*K_h*K_d + V_h*V_d] bf16 in [q | k | v] channel order.
// `a` and `b` are [N, V_h] bf16.
void launch_qwen_gdn_post_conv_prep_bf16(
    const void* qkv_post,
    const void* a,
    const void* b,
    const void* A_log,
    const void* dt_bias,
    float* q_norm_kh,
    float* k_norm_kh,
    float* v_fp32,
    float* g_log_out,
    float* beta_out,
    int N, int K_h, int V_h, int K_d, int V_d, int conv_dim,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
