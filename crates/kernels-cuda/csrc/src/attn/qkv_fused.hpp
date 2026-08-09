#pragma once

// Fused QKV-projection epilogues: one kernel that reads the packed matmul
// output and lands Q, K and V in their final places, doing the per-head
// Q/K RMSNorm and the RoPE on the way.
//
// These were carved out of `attn/split_packed.hpp`, whose contract is "one
// pass over packed memory; pure copy, no compute" — true of the two splits
// still there, false of every kernel below. Three of that file's five
// kernels normalise, rotate and write into the paged cache; the header
// promising no compute was the marker for this split.
//
// The unfused decomposition of the same work is
// `split_qkv_bf16` → `rope/rope.hpp` → `attn/kv_paged.hpp`, and it stays
// the reference: these are a decode fast path, not a replacement. RoPE has
// already fanned the heads across `blockIdx.y` and holds the rotated K in
// registers exactly where `write_kv_to_pages` would re-read it, so the
// fusion buys the KV write for free (`rope/rope.hpp` documents the same
// trade for its own fused form).
//
// Why they live in `attn/` and not in a KV family: they span the split
// (layout), the norm, the rotation and the cache write. No one family owns
// them, and the family that comes closest is the one they feed.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::attn {

// Pure-decode fast path for fused QKV projections with per-head Q/K RMSNorm
// and standard RoPE. Reads packed [R, q_dim + 2 * kv_dim], writes Q to
// [R, num_q_heads, head_dim], and writes K/V directly into the paged cache at
// the current decode position for each request.
void qkv_decode_qk_norm_rope_write_kv_bf16(
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
void qkv_decode_qk_norm_rope_write_kv_bf16_devwin(
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
void qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
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

}  // namespace pie_cuda_driver::kernels::attn
