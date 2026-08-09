#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::attn {

// Average-pool consecutive tokens: out[i] = mean(in[i*ratio : (i+1)*ratio])
// Used by the compressor to reduce token count by compress_ratio.
// `average_pool_bf16`, `add_ape_f32`, `gated_softmax_pool_bf16` and
// `dsv4_compress_gather_bf16` WERE declared here, and `attention_compressed_bf16`
// below them. All five launchers are deleted -- they are the unpaged half of
// DSv4 compression and the only thing that called any of them was another one
// of the five. The `.cu` carries the measurement.

void dsv4_compress_gather_paged_bf16(
    const void* state_kv,               // [num_pages, page_size, coff*head_dim] BF16
    const void* state_score,            // [num_pages, page_size, coff*head_dim] BF16
    const float* ape,                   // [ratio, coff*head_dim] F32 or null
    const std::int32_t* boundary_pos,   // [C]
    const std::int32_t* boundary_req,   // [C]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    void* out,                          // [C, head_dim] BF16
    int num_entries,
    int head_dim,
    int ratio,
    int coff,
    int page_size,
    cudaStream_t stream);

// Writes finished compressed entries into the paged compressed-KV cache at
// their boundary token's own slot, so entry `c` of a request is always found
// at absolute position `(c + 1) * ratio - 1`.
/// CUDA-graph-safe boundary metadata for pure decode. Emits one slot per
/// token (`n` == number of requests); tokens whose position is not a window
/// boundary get `out_pos = -1`, which `dsv4_compress_gather_paged_bf16`
/// zero-fills and `dsv4_store_comp_entries_bf16` skips. Replaces the
/// host scan + compaction, which required a D2H sync and blocked graph capture.
void dsv4_boundary_meta_decode(
    const std::int32_t* positions,
    std::int32_t*       out_pos,
    std::int32_t*       out_req,
    std::int32_t*       out_rope,
    int                 n,
    int                 ratio,
    cudaStream_t        stream,
    const std::uint8_t* row_valid = nullptr);

/// The same metadata for a fire that brings MANY rows per request.
///
/// `dsv4_boundary_meta_decode` shortcuts the request index to the token index,
/// which holds only when each request contributes one row. Here the request a
/// token belongs to is the CSR row its index falls in, read from `qo_indptr`
/// (`num_requests + 1` entries, non-decreasing). Everything else — whether a
/// position closes a window, and the rope base of the window it closes — is a
/// per-token fact and is computed identically.
void dsv4_boundary_meta_paged(
    const std::int32_t*  positions,
    const std::uint32_t* qo_indptr,
    std::int32_t*        out_pos,
    std::int32_t*        out_req,
    std::int32_t*        out_rope,
    int                  n,
    int                  num_requests,
    int                  ratio,
    cudaStream_t         stream,
    const std::uint8_t*  row_valid = nullptr);

void dsv4_store_comp_entries_bf16(
    const void* entries,                // [C, head_dim] BF16
    void* comp_kv_pages,                // [num_pages, page_size, head_dim] BF16
    const std::int32_t* boundary_pos,   // [C]
    const std::int32_t* boundary_req,   // [C]
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    int num_entries,
    int head_dim,
    int page_size,
    cudaStream_t stream);

// Dense attention of every query against its request's compressed entries,
// read through the KV page table: entry `c` of a request lives at absolute
// position `(c + 1) * ratio - 1`. A query at absolute position `p` may attend
// to entry `c` iff that boundary position is `<= p`.
void attention_compressed_paged_bf16(
    const void* q,                      // [N, num_q_heads, head_dim] BF16
    const void* comp_kv_pages,          // [num_pages, page_size, head_dim] BF16
    void* o,                            // [N, num_q_heads, head_dim] BF16
    float* lse_out,                     // [N, num_q_heads] F32 or null
    const std::int32_t* positions,      // [N] absolute position of each query
    const std::uint32_t* qo_indptr,     // [R+1] device
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::int32_t* req_of_token,   // [N] request index of each query
    int total_tokens,
    int num_q_heads,
    int head_dim,
    int ratio,
    int page_size,
    float sm_scale,
    cudaStream_t stream);

// ── Combine two attention outputs using LSE ────────────────────────────
// Given two partial attention results (o1, lse1) and (o2, lse2), produces the
// exact combined output as if all KV entries were attended jointly:
//   lse_max = max(lse1, lse2)
//   w1 = exp(lse1 - lse_max),  w2 = exp(lse2 - lse_max)
//   o = (o1 * w1 + o2 * w2) / (w1 + w2)
//   combined_lse = lse_max + log(w1 + w2)
//
// When lse2 is -inf (no compressed entries for this token), output is unchanged.
//
// Layout:
//   o1, o2     [N, num_heads * head_dim]   BF16
//   lse1, lse2 [N, num_heads]              F32
//   o_out      [N, num_heads * head_dim]   BF16 (may alias o1)
//   lse_out    [N, num_heads]              F32 (may alias lse1)
void combine_attn_outputs_bf16(
    const void* o1, const float* lse1,
    const void* o2, const float* lse2,
    void* o_out, float* lse_out,
    int N, int num_heads, int head_dim,
    cudaStream_t stream);

// `attention_compressed_bf16` -- dense attention over compressed KV with
// per-request causal masking -- WAS declared here, with its layout table.
// Deleted with the other four unpaged launchers; the `.cu` carries the
// measurement, and `attn/dsv4_compress.cuh`'s `attn_compressed` is still the
// kernel and still compiled.

}  // namespace pie_cuda_driver::kernels::attn
