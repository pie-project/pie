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

}  // namespace pie_cuda_driver::kernels
