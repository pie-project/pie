#pragma once

// Spec-decoding batch expansion. Mirrors pie_driver's `get_spec_expanded_*`
// helpers: when `spec_token_ids` is non-empty for some request, splice
// the proposed drafts into tokens/positions/qo + bump `kv_last_page_lens`,
// then append a verification block (n_drafts+1 cloned token samplers per
// spec request) to the sampling layout.
//
// Returns owned vectors; the request handler switches its active spans
// to point at them when `has_drafts` is true.

#include <cstdint>
#include <span>
#include <vector>

namespace pie_cuda_driver {

struct SpecExpansionInputs {
    std::span<const std::uint32_t> tokens;
    std::span<const std::uint32_t> positions;
    std::span<const std::uint32_t> qo_indptr;
    std::span<const std::uint32_t> kv_last_page_lens;
    std::span<const std::uint32_t> sampling_indices;
    std::span<const std::uint32_t> sampling_indptr;
    std::span<const std::uint32_t> request_num_samplers;
    std::span<const std::uint32_t> sampler_types;
    std::span<const std::uint32_t> sampler_top_k;
    std::span<const std::uint32_t> sampler_seeds;
    std::span<const float>         sampler_temperatures;
    std::span<const float>         sampler_top_p;
    std::span<const float>         sampler_min_p;
    std::span<const std::uint32_t> spec_token_ids;
    std::span<const std::uint32_t> spec_position_ids;
    std::span<const std::uint32_t> spec_indptr;
    int page_size;
};

struct SpecExpansion {
    bool has_drafts = false;

    // Spec-expanded copies of the BPIQ wire arrays. Populated only when
    // `has_drafts` is true; otherwise default-constructed and ignored.
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> positions;
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<std::uint32_t> sampling_indices;
    std::vector<std::uint32_t> sampling_indptr;
    std::vector<std::uint32_t> request_num_samplers;
    std::vector<std::uint32_t> sampler_types;
    std::vector<std::uint32_t> sampler_top_k;
    std::vector<std::uint32_t> sampler_seeds;
    std::vector<float>         sampler_temperatures;
    std::vector<float>         sampler_top_p;
    std::vector<float>         sampler_min_p;

    // Per-request verify-block metadata. `verify_slot_start[r]` is the
    // global slot index in the expanded sampling layout where request
    // `r`'s verification block starts, or -1 if `r` had no drafts.
    // `verify_n_drafts[r]` is the corresponding draft count.
    std::vector<int> verify_slot_start;
    std::vector<int> verify_n_drafts;
};

// Build the spec-expanded batch. `R` is `qo_indptr.size() - 1` (number
// of requests). Returns a default-constructed `SpecExpansion` (with
// `has_drafts == false`) when `spec_token_ids` is empty.
SpecExpansion expand_spec_batch(const SpecExpansionInputs& in, int R);

}  // namespace pie_cuda_driver
