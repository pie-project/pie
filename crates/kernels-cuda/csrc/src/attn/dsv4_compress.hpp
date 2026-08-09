#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::attn {

// Average-pool consecutive tokens: out[i] = mean(in[i*ratio : (i+1)*ratio])
// Used by the compressor to reduce token count by compress_ratio.
void average_pool_bf16(
    const void* input,       // [N, dim] BF16
    void* output,            // [N/ratio, dim] BF16
    int N,
    int dim,
    int ratio,
    cudaStream_t stream);

// Add Accumulated Positional Embedding (APE) to compressed KV.
// output[i] += ape[i % ratio]
void add_ape_f32(
    void* data,              // [N_compressed, dim] BF16 (modified in-place)
    const float* ape,        // [ratio, dim] F32
    int N_compressed,
    int dim,
    int ratio,
    cudaStream_t stream);

// ── Gated softmax pooling ─────────────────────────────────────────────
// For each group g of `ratio` consecutive tokens, and for each element d:
//   output[g, d] = sum_{i=0}^{ratio-1} kv[g*ratio+i, d] * softmax(score[g*ratio:g*ratio+ratio, d], dim=0)[i]
// The softmax is computed independently per element d along the ratio dimension.
//
// Layout:
//   kv     [N, dim]         BF16 — projected KV values
//   score  [N, dim]         BF16 — gating scores (with APE already added)
//   output [N/ratio, dim]   BF16 — compressed output
//
// N must be divisible by ratio.
void gated_softmax_pool_bf16(
    const void* kv,
    const void* score,
    void* output,
    int N,
    int dim,
    int ratio,
    cudaStream_t stream);

// ── DeepSeek-V4 compressor gather ──────────────────────────────────────
// Produces one compressed KV entry per *boundary* token — a token whose
// absolute position `p` satisfies `(p + 1) % ratio == 0`. Each entry
// summarises the `coff * ratio` tokens ending at that boundary via a
// per-dimension softmax over the gate scores:
//
//   s[i, d] = score[tok_i, off(i) + d] + ape[pos_i % ratio, off(i) + d]
//   w[i, d] = softmax_i(s[:, d])
//   out[c, d] = Σ_i kv[tok_i, off(i) + d] * w[i, d]
//
// `off(i) = (i >= ratio) ? head_dim : 0`. The C4 compressor overlaps
// (`coff == 2`, window 2*ratio): the *older* `ratio` tokens read the first
// `head_dim` columns of the projection and the *newer* `ratio` tokens read
// the second — matching vLLM's `head_offset = (tokens >= ratio) * HEAD_SIZE`
// in `_fused_kv_compress_norm_rope_insert_sparse_attn`. Non-overlapping
// compressors (`coff == 1`) only ever hit the first branch.
//
// Window slots that fall before the request's first token, or before
// absolute position 0, are masked out (score `-inf`, weight 0).
void dsv4_compress_gather_bf16(
    const void* kv_proj,                // [N, coff*head_dim] BF16
    const void* score_proj,             // [N, coff*head_dim] BF16
    const float* ape,                   // [ratio, coff*head_dim] F32 or null
    const std::int32_t* boundary_tok,   // [C] flat token index of the boundary
    const std::int32_t* boundary_pos,   // [C] absolute position of the boundary
    const std::int32_t* window_lo,      // [C] first flat token index of the request
    void* out,                          // [C, head_dim] BF16
    int num_entries,
    int head_dim,
    int ratio,
    int coff,
    cudaStream_t stream);

// Paged variant of the gather above: reads the compressor state out of the
// page-parallel state cache (`[num_pages, page_size, width]` BF16, addressed by
// the request's KV page table) rather than a flat per-batch array, so window
// slots written by earlier forward passes are visible during decode.
// `boundary_pos[c]` is the absolute position of entry `c`'s boundary token and
// `boundary_req[c]` the index of the owning request.
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

// ── Dense attention over compressed KV with per-request causal masking ──
// Computes multi-head attention between Q and compressed KV entries.
//
// For each request r with qo_range [qo_lo, qo_hi), the query at local offset
// `t` (absolute position qo_lo + t within the request's token sequence) can
// attend to compressed entry c only if c < (t + 1) / ratio. This implements
// the causal constraint for compressed attention during prefill.
//
// The compressed KV entries for request r are stored in comp_kv starting at
// row comp_offsets[r], with comp_lens[r] entries.
//
// Layout:
//   q            [total_tokens, num_q_heads, head_dim]  BF16
//   comp_kv      [total_comp, 1, head_dim]              BF16
//   o            [total_tokens, num_q_heads, head_dim]  BF16
//   lse_out      [total_tokens, num_q_heads]            F32 (nullptr = skip)
//   qo_indptr    [R+1]                                  host
//   comp_offsets [R]                                     host
//   comp_lens    [R]                                     host
void attention_compressed_bf16(
    const void* q,
    const void* comp_kv,
    void* o,
    float* lse_out,
    const int* qo_indptr,      // host: [R+1] query offsets
    const int* comp_offsets,   // host: [R] compressed KV offsets
    const int* comp_lens,      // host: [R] compressed KV lengths
    const int* comp_ratios,    // host: [R] compression ratios per request
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int head_dim,
    float sm_scale,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels::attn
