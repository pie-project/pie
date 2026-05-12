#pragma once

// Token-sampler dispatch. Wraps the two production sampling kernels
// behind a single entry point:
//
//   * `launch_sample_topk_topp_bf16` (flashinfer): when any slot has a
//     non-trivial top-k or top-p filter and a positive temperature.
//   * `launch_sample_temp_bf16` (custom Gumbel-max + min-p mask):
//     temperature/min-p only, including the greedy-argmax fallback
//     (T == 0 → argmax via score = logit).
//
// Per-row params (`temp`/`top_p`/`top_k`/`min_p`/`seed`) are aligned
// with logit rows; `sample_idx` maps each global sampling slot to its
// logit row. Caller is responsible for ensuring the per-row arrays are
// populated only for rows that are actual token-sampler slots — other
// rows can have zero values; their kernel output is unused.
//
// On exit, `d_sampled` (length N) is filled with sampled token IDs at
// every sample-row position. Non-sample rows are unspecified.

#include <cstdint>
#include <span>

#include "device_buffer.hpp"

namespace pie_cuda_driver::model { struct Qwen3Workspace; }

namespace pie_cuda_driver {

struct PersistentInputs;

struct SamplingPlan {
    // Whether any slot needs the topk+top-p kernel. False routes to the
    // simpler temperature/min-p path.
    bool any_topk_topp;

    // Per-row arrays (length N). For rows with no sampler attached the
    // values are ignored, but the arrays must be sized for N.
    std::span<const float>        temp;
    std::span<const float>        top_p;
    std::span<const float>        min_p;
    std::span<const std::int32_t> top_k;   // 0 already mapped to vocab.
    std::span<const std::uint32_t> seed;   // u32; widened to u64 inside.

    // Per-slot global logit-row index (length num_sampling). Only used
    // by the topk+top-p kernel; unused for the temp/min-p path.
    std::span<const std::int32_t> sample_idx;
};

// `d_sampled_out` must point at a device buffer of capacity ≥ N
// `int32_t`s. Non-sample rows are left unspecified; only positions
// listed in `plan.sample_idx` are written by the topk+top-p path,
// while the temp/min-p path writes every row.
//
// `pi` supplies persistent device + pinned-host scratch (alloced once
// at engine init). The plan arrays are uploaded via async H2D into the
// stable buffers; the topk/topp scatter uses pinned host staging so
// the D2H + H2D copies bypass the driver's pageable-staging fallback.
void dispatch_sampling(
    model::Qwen3Workspace& ws,
    PersistentInputs& pi,
    std::int32_t* d_sampled_out,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset);

}  // namespace pie_cuda_driver
