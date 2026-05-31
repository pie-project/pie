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
//
// Launcher declaration; the kernel body and its SplitMix/Gumbel helpers live
// in sample.cu, lifted verbatim from driver/cuda/src/kernels/sample_temp.cu
// (the `launch_sample_temp_bf16` variant only). The compact-scatter and
// argmax/TP-select variants are lifted later as the bodies that need them
// land.

#include <cstdint>
#include <cuda_runtime.h>

namespace pie_cuda_device::kernels {

// Truncation filters, all expressed as a per-row logit cutoff applied before
// the Gumbel-max draw (the kept set is the intersection — most restrictive
// cutoff wins). Any of `top_ps`/`top_ks`/`min_ps` may be NULL (disabled for
// every row). All are ignored on greedy rows (temperature <= 0 → argmax).
//   * `min_ps[r]` (0 = off): keep `prob[j]/max_prob >= min_p`, i.e.
//     `logit[j] >= max_logit + log(min_p)`.
//   * `top_ps[r]` (>=1 or <=0 = off): nucleus — keep the smallest set of
//     highest-prob tokens whose cumulative softmax mass >= p. Found by binary
//     search for the largest logit cutoff whose retained mass still covers p.
//   * `top_ks[r]` (<=0 = off): keep the `k` highest-logit tokens. Binary
//     search for the largest cutoff retaining >= k tokens.
void sample_temp_bf16(
    const void* logits,                 // [num_rows, vocab] bf16
    const float* temperatures,          // [num_rows]
    const float* top_ps,                // [num_rows] or NULL
    const std::int32_t* top_ks,         // [num_rows] or NULL
    const float* min_ps,                // [num_rows] or NULL
    const std::uint32_t* seeds,         // [num_rows]
    std::int32_t* out,                  // [num_rows]
    int num_rows,
    int vocab,
    cudaStream_t stream);

}  // namespace pie_cuda_device::kernels
