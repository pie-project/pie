#pragma once

// Per-sampler-type sub-passes that populate rich response fields
// (raw logits, entropies, logprobs, top-K distributions). Each
// function walks the batch, gathers the slots that match its sampler
// type, runs one fused kernel (or a host-side gather for RawLogits),
// and pushes the per-slot payload into the matching field of
// `pie_driver::PerRequestOutput` in slot-iteration order — matching
// the runtime's `*_iter` consumption in
// `scheduler.rs::convert_output`.
//
// Sub-passes are independent: they share read-only inputs but write
// to disjoint fields of `per_req`.

#include <cstdint>
#include <span>
#include <vector>

#include <pie_schema/response_builder.hpp>

namespace pie_cuda_driver::model { struct Qwen3Workspace; }

namespace pie_cuda_driver {

// Inputs every sub-pass needs. Aggregated to keep the per-pass
// signatures short.
struct ResponseSubpassContext {
    model::Qwen3Workspace& ws;
    int R;
    int num_sampling;
    int vocab_size;

    // Per-slot type as decoded from the active (possibly spec-expanded)
    // wire. Length == num_sampling.
    std::span<const std::uint32_t> per_slot_type;
    // Per-slot temperature / top_k as already mapped (0 → vocab_size).
    // Only used by the Dist sub-pass.
    std::span<const float>         per_slot_temp;
    std::span<const std::int32_t>  per_slot_top_k;

    // Active wire control views (see executor).
    std::span<const std::uint32_t> qo_indptr;       // length R+1
    std::span<const std::uint32_t> sampling_indptr; // length R+1
    std::span<const std::uint32_t> sampling_indices; // length num_sampling
    std::span<const std::uint32_t> request_num_samplers; // length R
};

// RawLogits (type=7): gather each Logits slot's row, convert bf16 → f32
// bytes, push as a `bytes` payload. Issues one D2H copy per slot.
void gather_raw_logits(
    const ResponseSubpassContext& ctx,
    std::vector<pie_driver::PerRequestOutput>& per_req);

// Entropy (type=10): launch a single fused entropy kernel over all
// entropy slots in the batch and push one float per slot.
void compute_entropy_slots(
    const ResponseSubpassContext& ctx,
    std::vector<pie_driver::PerRequestOutput>& per_req);

// Logprob (type=8) / Logprobs (type=9): build the CSR label layout
// across both types, run the fused logprobs kernel, and push one
// `vector<float>` per slot (length 1 for Logprob, K for Logprobs).
// Reads per-sampler label arrays from the request view.
void compute_logprob_slots(
    const ResponseSubpassContext& ctx,
    const pie_driver::PieForwardRequestView& view,
    std::vector<pie_driver::PerRequestOutput>& per_req);

// Dist (type=0): temperature-scaled softmax kernel + host-side top-K
// partial sort. Pushes (token_ids, probs) per slot.
void compute_dist_slots(
    const ResponseSubpassContext& ctx,
    std::vector<pie_driver::PerRequestOutput>& per_req);

}  // namespace pie_cuda_driver
