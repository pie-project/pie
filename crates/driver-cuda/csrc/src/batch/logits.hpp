#pragma once

// Selected-row logit gather + bf16 -> f32 staging for pipeline dispatch.
//
// `pipeline::Dispatch` consumes a dense F32 logits matrix ordered by
// `sampling_indices`. The model forward writes BF16 logits for every row of
// `ws.logits`; this module gathers just the sampled rows (via
// `engine.inputs.sample_idx`) and widens them into `engine.ptir_logits_f32`,
// growing the BF16/F32 scratch buffers on `engine` as needed.

#include <cstdint>

namespace pie_cuda_driver {

struct BatchEngine;

// Gather the `num_sampling` selected BF16 logit rows (vocab-wide) out of
// `engine.ws.logits` via `engine.inputs.sample_idx`, widen them to F32, and
// return a pointer to the F32 scratch (owned by `engine.ptir_logits_f32`).
// Returns nullptr when there is nothing to sample (`num_sampling == 0`).
const float* gather_selected_logits_f32(
    BatchEngine& engine, int num_sampling, std::uint32_t vocab);

}  // namespace pie_cuda_driver
