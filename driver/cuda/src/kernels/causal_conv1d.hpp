#pragma once

// Depthwise causal 1D convolution used by the Qwen3.5 GatedDeltaNet
// in-projection step. Two entry points:
//
//   * `launch_causal_conv1d_prefill_bf16`: full-prefill of N tokens
//     for a single request. Each output position t convolves the K
//     prior input positions (0-padded for t < K-1). Fused silu.
//     Optionally writes the trailing K elements of input into the
//     conv_state buffer for downstream decode steps.
//
//   * `launch_causal_conv1d_update_bf16`: single-token (T=1) update
//     for decode. Shifts conv_state left by 1, appends the new x,
//     emits one silu(conv) output token, and persists the updated
//     state. Used per request per linear-attn layer.
//
// Memory layout (channels-last, matches the GatedDeltaNet in-proj
// output shape after the linear: x[N, C] flat, then split by the
// caller into Q/K/V chunks along C):
//
//     x       : [N, C]    bf16 — N tokens, C channels (= 2*K_dim + V_dim)
//     weight  : [C, K]    bf16 — per-channel kernel of length K
//     bias    : [C]       bf16 — optional per-channel bias (nullptr = none)
//     state   : [K, C]    bf16 — last K input rows, oldest first
//     y       : [N, C]    bf16 — conv output

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Single-request prefill. `state_out` may be nullptr to skip persisting
// the trailing K-window into a state buffer.
void launch_causal_conv1d_prefill_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void*       y,
    void*       state_out,
    int N,
    int C,
    int K,
    cudaStream_t stream);

// Single-token decode update. Reads state, appends `x` (one row),
// writes the conv output to `y` (one row), updates `state` in place.
void launch_causal_conv1d_update_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void*       state,
    void*       y,
    int C,
    int K,
    cudaStream_t stream);

// Multi-request batched decode update. Replaces the host-loop of R
// single-request `_update_bf16` calls with one kernel launch:
//
//     x          : [R, C]            bf16 — one new token per request
//     y          : [R, C]            bf16 — outputs
//     state_base : [num_slots, K, C] bf16 — slot 0's address; the kernel
//                  picks slot `slot_ids[r]` for each request r and reads
//                  / writes its [K, C] slab in-place
//     slot_ids   : [R]               int32 (device-resident)
//     slot_stride_elems : K * C — stride between consecutive slots in
//                  the state buffer, in bf16 elements
//
// Eliminates `R × num_linear_layers` per-token launch overhead on the
// decode hot path.
void launch_causal_conv1d_update_batched_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void*       state_base,
    const std::int32_t* slot_ids,
    long long   slot_stride_elems,
    void*       y,
    int R, int C, int K,
    cudaStream_t stream);

// Multi-request batched prefill. Replaces the host-loop of R
// single-request `_prefill_bf16` calls with one kernel launch:
//
//     x              : [N_total, C]      bf16 — concatenated request tokens
//     y              : [N_total, C]      bf16 — outputs
//     state_out_base : [num_slots, K, C] bf16 — slot 0's address
//     slot_ids       : [R]               int32 device — slot per request
//     qo_indptr      : [R+1]             u32 device — token offsets per req
//     slot_stride_elems : K * C bf16 elements
//
// Each (channel, request) block reads its own (t0_r, Nr_r) window from
// qo_indptr and walks tokens internally. The trailing K-window is
// persisted into the request's state slab for the follow-up decode.
void launch_causal_conv1d_prefill_batched_bf16(
    const void* x,
    const void* weight,
    const void* bias,
    void*       y,
    void*       state_out_base,
    const std::int32_t*  slot_ids,
    const std::uint32_t* qo_indptr,
    long long   slot_stride_elems,
    int R, int C, int K,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
