#pragma once

// Per-row temperature-scaled multinomial sampling via the Gumbel-max trick:
//
//     g_j = -log(-log(uniform(0,1)))
//     sampled = argmax_j (logit[j] / T + g_j)
//
// Equivalent to drawing from softmax(logit / T) but needs only one pass
// over the row and no full distribution storage.
//
// `temperatures` and `seeds` are per-row. If `temperatures[r] <= 0`, the
// row collapses to plain argmax (deterministic).
//
// `seeds[r]` and the absolute index `j` are mixed by a SplitMix-style hash
// so the per-element noise is reproducible regardless of thread mapping.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// `min_ps` is the per-row min-p threshold (0 = disabled). When > 0, tokens
// with `prob[j] / max_prob < min_p` are masked out before Gumbel-max — i.e.,
// `logit[j] < max_logit + log(min_p)` is excluded.
void launch_sample_temp_bf16(
    const void* logits,                 // [num_rows, vocab] bf16
    const float* temperatures,          // [num_rows]
    const float* min_ps,                // [num_rows]
    const std::uint32_t* seeds,         // [num_rows]
    std::int32_t* out,                  // [num_rows]
    int num_rows,
    int vocab,
    cudaStream_t stream);

// Same sampler as launch_sample_temp_bf16, but `logits` is compacted to
// [num_samples, vocab]. `row_indices[compact_row]` names the original row
// whose sampler params should be used and where the sampled token is written.
void launch_sample_temp_bf16_compact_scatter(
    const void* logits,                 // [num_samples, vocab] bf16
    const std::int32_t* row_indices,    // [num_samples] original row ids
    const float* temperatures,          // [original num_rows]
    const float* min_ps,                // [original num_rows]
    const std::uint32_t* seeds,         // [original num_rows]
    std::int32_t* out,                  // [original num_rows]
    int num_samples,
    int vocab,
    cudaStream_t stream);

// Greedy TP helper: scan a row-major BF16 logits shard and return each
// row's best score plus global token id (`vocab_offset + local_col`).
void launch_argmax_bf16_with_offset(
    const void* logits,                 // [num_rows, vocab_shard] bf16
    std::int32_t* out_tokens,           // [num_rows]
    float* out_values,                  // [num_rows]
    int num_rows,
    int vocab_shard,
    int vocab_offset,
    cudaStream_t stream);

// Same scan as launch_argmax_bf16_with_offset, but packs each row's
// local max as {float value bits, int32 token id} so TP can use one
// all-gather instead of separate value/token collectives.
void launch_argmax_bf16_pair_with_offset(
    const void* logits,                 // [num_rows, vocab_shard] bf16
    std::uint64_t* out_pairs,           // [num_rows]
    int num_rows,
    int vocab_shard,
    int vocab_offset,
    cudaStream_t stream);

// Rank-0 helper after gathering local maxima from every TP rank.
// `values_by_rank` and `tokens_by_rank` are rank-major [world, num_rows].
void launch_select_global_argmax(
    const float* values_by_rank,
    const std::int32_t* tokens_by_rank,
    std::int32_t* out_tokens,
    int num_rows,
    int world_size,
    cudaStream_t stream);

// Rank-0 helper for packed pair gather.
void launch_select_global_argmax_pairs(
    const std::uint64_t* pairs_by_rank,
    std::int32_t* out_tokens,
    int num_rows,
    int world_size,
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
