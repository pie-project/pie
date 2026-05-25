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

void launch_argmax_bf16_partitioned_pairs(
    const void* logits,              // [num_rows, vocab] bf16
    std::uint64_t* partial_pairs,    // [parts, num_rows]
    int num_rows,
    int vocab,
    int parts,
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

}  // namespace pie_cuda_driver::kernels
