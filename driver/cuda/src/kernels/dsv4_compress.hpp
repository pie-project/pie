#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Average-pool consecutive tokens: out[i] = mean(in[i*ratio : (i+1)*ratio])
// Used by the compressor to reduce token count by compress_ratio.
void launch_average_pool_bf16(
    const void* input,       // [N, dim] BF16
    void* output,            // [N/ratio, dim] BF16
    int N,
    int dim,
    int ratio,
    cudaStream_t stream);

// Add Accumulated Positional Embedding (APE) to compressed KV.
// output[i] += ape[i % ratio]
void launch_add_ape_f32(
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
void launch_gated_softmax_pool_bf16(
    const void* kv,
    const void* score,
    void* output,
    int N,
    int dim,
    int ratio,
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
void launch_combine_attn_outputs_bf16(
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
void launch_attention_compressed_bf16(
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

}  // namespace pie_cuda_driver::kernels
