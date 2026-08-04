#pragma once

// Split a fused matmul output into separately-packed buffers.
//
// The fused QKV / gate-up matmuls write a row-major `[N, A + B (+ C)]`
// tensor where columns [0,A) are the first output, [A,A+B) the second,
// etc. Downstream kernels (rope, kv_paged, swiglu, …) want each output
// in its own packed `[N, A]` / `[N, B]` buffer so they can use the
// existing addressing.
//
// One pass over packed memory; pure copy, no compute.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// `packed` is row-major [N, q_dim + 2*kv_dim]; outputs are row-major
// [N, q_dim] / [N, kv_dim] / [N, kv_dim]. Buffers must not overlap with
// `packed`.
// Peel device-window variant: {start, len} in device memory, full-grid
// launch with early-out, base pointers.
void launch_split_qkv_bf16_devwin(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    const std::uint32_t* win_d,
    int n_max, int q_dim, int kv_dim,
    cudaStream_t stream);

void launch_split_qkv_bf16(
    const void* packed,
    void* q_out, void* k_out, void* v_out,
    int n_tokens, int q_dim, int kv_dim,
    cudaStream_t stream);

// `packed` is row-major [N, 2*inter]; outputs are row-major [N, inter].
void launch_split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream);

// Fused postprocess for fused-QKV projections with per-head Q/K RMSNorm and
// standard RoPE: normalize, rotate, write Q to [num_rows, num_q_heads,
// head_dim], and append K/V straight into the paged cache -- replacing the
// split-qkv → qk-norm+rope → write-kv chain with a single pass.
//
// `qo_indptr` selects the shape. Pass `nullptr` for a pure-decode fire, where
// each row is its own request and the append target is that request's last
// occupied slot. Pass the fire's CSR query offsets (R+1 entries) for a prefill
// or mixed fire, where rows are query TOKENS: request `r` then owns rows
// [qo_indptr[r], qo_indptr[r+1]) and each row appends at its own offset within
// the span. `num_rows` is the query-token count either way; `num_requests`
// bounds `qo_indptr` and is unused when it is null.
//
// Every other per-row input -- `positions`, `rope_table`, `row_valid`, and the
// WSlot/WOff descriptor -- is indexed by row in both shapes.
void launch_qkv_qk_norm_rope_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    const float* rope_table,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* w_page,
    const std::uint32_t* w_off,
    const std::uint8_t* row_valid,
    int num_rows,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream);

// Peel device-window variant (PREFIX form): the fused decode epilogue
// owns the hook-free prefix, rows [0, win_d[0]) — the window word's
// START is this kernel's row count (the tail region starts where the
// prefix ends). Grid spans the full `n_max` lanes; out-of-window rows
// early-out, so a captured launch replays across row splits.
void launch_qkv_decode_qk_norm_rope_write_kv_bf16_devwin(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    const float* rope_table,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* w_page,
    const std::uint32_t* w_off,
    const std::uint8_t* row_valid,
    const std::uint32_t* win_d,
    int n_max,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream);

// Gemma4 row-decode verifier fast path for packed [Q;K;V] projection output.
// Each input row has a corresponding decode-style KV page table row. The
// kernel writes only Q scratch plus normalized/rotated K and normalized V
// directly into the paged cache, preserving the unfused bf16 rounding points.
void launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
    const void* packed,
    void* q_out,
    void* k_pages,
    void* v_pages,
    const void* q_weight,
    const void* k_weight,
    const std::int32_t* positions,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint8_t* row_valid,
    int num_rows,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
