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

#include <cuda_runtime.h>

#include "device_buffer.hpp"

namespace pie_cuda_driver { struct PersistentInputs; }
namespace pie_cuda_driver::model { struct Qwen3Workspace; }

namespace pie_cuda_driver {

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

// Upload per-row sampler params into `pi.sample_*` device buffers via
// cudaMemcpyAsync on `stream`. Caller is expected to run this BEFORE
// entering any CUDA-graph capture region — host source pointers
// (`plan.*` spans) are per-fire stack locals, so capturing the memcpy
// itself would dangling-reference them on replay. Issued on the
// default stream by convention, where legacy-default-stream semantics
// keep it ordered before any subsequent graph launch.
void upload_sampling_inputs(
    PersistentInputs& pi,
    const SamplingPlan& plan,
    int N,
    cudaStream_t stream = nullptr);

// Launch the sampling kernel(s) + on-device scatter against the
// already-uploaded `pi.sample_*` buffers. Pure device-side work — no
// host roundtrips, no allocations. Safe to issue inside a CUDA-graph
// capture region. `d_sampled_out` must point at `pi.sampled.data()`
// (or another stable device buffer of capacity ≥ N int32_t).
//
// On exit, `d_sampled_out[row]` holds the sampled token id for every
// row in `[0, N)` for the temp/min-p path, or only at positions
// listed in `plan.sample_idx` for the topk+top-p path (other rows
// are zeroed by the scatter kernel).
void launch_sampling_kernel(
    model::Qwen3Workspace& ws,
    std::int32_t* d_sampled_out,
    PersistentInputs& pi,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset,
    cudaStream_t stream);

// Legacy entry point that bundles upload + kernel. Used by callers
// that don't capture into a CUDA graph (prefill, custom-mask paths).
// Equivalent to upload_sampling_inputs(...) then launch_sampling_kernel(...).
void dispatch_sampling(
    model::Qwen3Workspace& ws,
    std::int32_t* d_sampled_out,
    PersistentInputs& pi,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset,
    cudaStream_t stream = nullptr);

}  // namespace pie_cuda_driver
