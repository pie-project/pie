#pragma once

// Per-row argmax over [num_rows, vocab] bf16 logits → [num_rows] i32 token ids.
// Used as the greedy sampler (temperature=0).

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

void launch_argmax_bf16(
    const void* logits,        // [num_rows, vocab] bf16
    std::int32_t* token_ids,   // [num_rows]
    int num_rows,
    int vocab,
    cudaStream_t stream);

void launch_argmax_bf16_compact_scatter(
    const void* logits,                    // [num_rows, vocab] bf16
    const std::int32_t* row_indices,       // [num_rows] original row ids
    std::int32_t* token_ids,               // [original rows]
    int num_rows,
    int vocab,
    cudaStream_t stream);

// NOTE: a `launch_lm_head_argmax_bf16` used to live here — a fused
// hidden@lm_head.T + argmax with grid=(num_rows, vocab/8). It had no call
// sites and could not serve the sampler: every row re-read the whole weight
// matrix, so at rows=512 it moved 512x the LM head weights. The chunked
// accumulator below is the shape that works at batch. Deleted rather than
// left as a trap for the next reader (§20.36).

// ── Chunked (vocab-streaming) argmax ─────────────────────────────────
// The sampler's read of a materialised [rows, vocab] logits tensor is pure
// HBM traffic that exists only because the reduction is a separate kernel
// from the LM head GEMM. When the caller can produce the logits one vocab
// slab at a time into a reused buffer, the slab stays L2-resident and both
// the write and the read stop reaching HBM (§20.36: 185 us/step at
// rows=512, vocab=151936).
//
// `launch_argmax_accumulate_bf16` folds one slab into a running per-warp
// best; `launch_argmax_finalize_bf16` collapses that to token ids. Results
// are bit-identical to `launch_argmax_bf16` over the concatenated slabs:
// the ordering is a total order on (value, -index), so slab order and scan
// order do not matter.
//
// `acc_val` / `acc_idx` are caller-owned scratch of
// `rows * kArgmaxAccumSlots` elements each.
constexpr int kArgmaxAccumSlots = 32;

void launch_argmax_accumulate_bf16(
    const void* slab,          // [rows, row_stride] bf16
    int rows,
    int width,                 // valid columns in this slab
    int row_stride,            // elements between slab rows
    int vocab_base,            // global vocab index of column 0
    float* acc_val,            // [rows, kArgmaxAccumSlots]
    std::int32_t* acc_idx,     // [rows, kArgmaxAccumSlots]
    bool init,                 // true for the first slab
    cudaStream_t stream);

void launch_argmax_finalize_bf16(
    const float* acc_val,
    const std::int32_t* acc_idx,
    std::int32_t* token_ids,   // [rows]
    int rows,
    cudaStream_t stream);

// Per-row argmax over [num_rows, vocab] fp32 logits → [num_rows] i32 token ids.
void launch_argmax_fp32(
    const void* logits,        // [num_rows, vocab] fp32
    std::int32_t* token_ids,   // [num_rows]
    int num_rows,
    int vocab,
    cudaStream_t stream);

// Gemma4 MTP ordered-embedding argmax. The assistant first selects top
// centroids, then scores only the tokens assigned to those centroids.
void launch_masked_embedding_argmax_bf16(
    const void* centroid_logits,       // [num_rows, num_centroids] bf16
    const void* hidden_states,         // [num_rows, hidden] bf16
    const void* lm_head_weight,        // [vocab, hidden] bf16
    const std::int64_t* token_ordering, // [num_centroids * vocab_per_centroid]
    std::int32_t* token_ids,           // [num_rows]
    int num_rows,
    int hidden,
    int num_centroids,
    int centroid_top_k,
    int vocab_per_centroid,
    cudaStream_t stream);

void launch_topk_centroids_bf16(
    const void* centroid_logits,       // [num_rows, num_centroids] bf16
    std::int32_t* top_centroids,       // [num_rows, centroid_top_k]
    int num_rows,
    int num_centroids,
    int centroid_top_k,
    cudaStream_t stream);

void launch_masked_embedding_tile_argmax_pairs_bf16(
    const std::int32_t* top_centroids, // [num_rows, centroid_top_k]
    const void* hidden_states,         // [num_rows, hidden] bf16
    const void* lm_head_weight,        // [vocab, hidden] bf16
    const std::int64_t* token_ordering, // [num_centroids * vocab_per_centroid]
    std::uint64_t* partial_pairs,      // [num_tiles, num_rows]
    int num_rows,
    int hidden,
    int centroid_top_k,
    int vocab_per_centroid,
    int num_tiles,
    cudaStream_t stream);

// Fused GEMV + argmax for MTP lm_head scoring. Returns greedy token IDs
// without materializing logits.
void launch_lm_head_gemv_argmax_int8(
    const void* hidden_states,        // [num_rows, hidden] bf16
    const std::int8_t* lm_head_weight, // [vocab, hidden] int8
    const float* scale_inv,           // [vocab] fp32 per-channel
    std::int32_t* token_ids,          // [num_rows]
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream);

void launch_lm_head_gemv_argmax_bf16(
    const void* hidden_states,        // [num_rows, hidden] bf16
    const void* lm_head_weight,       // [vocab, hidden] bf16
    std::int32_t* token_ids,          // [num_rows]
    int num_rows,
    int hidden,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
