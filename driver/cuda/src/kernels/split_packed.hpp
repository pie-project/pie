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

// Pure-decode fast path for fused QKV projections with per-head Q/K RMSNorm
// and standard RoPE. Reads packed [R, q_dim + 2 * kv_dim], writes Q to
// [R, num_q_heads, head_dim], and writes K/V directly into the paged cache at
// the current decode position for each request.
void launch_qkv_decode_qk_norm_rope_write_kv_bf16(
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
    int num_requests,
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
