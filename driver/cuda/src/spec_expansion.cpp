#include "spec_expansion.hpp"

#include <iostream>
#include <utility>

#include "sampler_type.hpp"

namespace pie_cuda_driver {

SpecExpansion expand_spec_batch(const SpecExpansionInputs& in, int R) {
    SpecExpansion out;
    out.verify_slot_start.assign(R, -1);
    out.verify_n_drafts.assign(R, 0);

    if (in.spec_token_ids.empty() || R <= 0) {
        return out;  // has_drafts == false; caller uses originals.
    }
    out.has_drafts = true;

    // ---- Per-request token / position / qo expansion + kv_last_page_lens ----
    int new_total = 0;
    for (int r = 0; r < R; ++r) {
        const int n_in = static_cast<int>(in.qo_indptr[r + 1] - in.qo_indptr[r]);
        const int n_d = (r + 1 < static_cast<int>(in.spec_indptr.size()))
            ? static_cast<int>(in.spec_indptr[r + 1] - in.spec_indptr[r])
            : 0;
        new_total += n_in + n_d;
    }
    out.tokens.resize(new_total);
    out.positions.resize(new_total);
    out.qo_indptr.resize(R + 1);
    out.qo_indptr[0] = 0;
    out.kv_last_page_lens.resize(R);

    int dst = 0;
    for (int r = 0; r < R; ++r) {
        const int qo_lo = static_cast<int>(in.qo_indptr[r]);
        const int n_in = static_cast<int>(in.qo_indptr[r + 1]) - qo_lo;
        const int spec_lo = (r < static_cast<int>(in.spec_indptr.size()))
            ? static_cast<int>(in.spec_indptr[r]) : 0;
        const int n_d = (r + 1 < static_cast<int>(in.spec_indptr.size()))
            ? static_cast<int>(in.spec_indptr[r + 1]) - spec_lo : 0;
        for (int j = 0; j < n_in; ++j) {
            out.tokens[dst + j]    = in.tokens[qo_lo + j];
            out.positions[dst + j] = in.positions[qo_lo + j];
        }
        for (int j = 0; j < n_d; ++j) {
            out.tokens[dst + n_in + j]    = in.spec_token_ids[spec_lo + j];
            out.positions[dst + n_in + j] = in.spec_position_ids[spec_lo + j];
        }
        dst += n_in + n_d;
        out.qo_indptr[r + 1] = static_cast<std::uint32_t>(dst);

        // Bump `kv_last_page_len` by `n_drafts` and wrap at `page_size`:
        // ((old + n_d - 1) % page_size) + 1. Mirrors pie_driver's
        // `kv_last_page_lens_ext`. Pages were already reserved upstream.
        const int old_lpl = static_cast<int>(in.kv_last_page_lens[r]);
        const int new_total_lpl = old_lpl + n_d;
        out.kv_last_page_lens[r] = static_cast<std::uint32_t>(
            ((new_total_lpl - 1) % in.page_size) + 1);

        out.verify_n_drafts[r] = n_d;
    }

    // ---- Sampling expansion ----
    // Original samplers stay as-is (sidx is per-request-relative, so
    // it remains correct under the new qo offsets). For each spec
    // request, append n_d+1 verification samplers cloned from that
    // request's first sampler config; their relative sampling indices
    // are [n_in - 1, n_in, ..., n_in + n_d - 1].
    out.sampler_types.reserve(in.sampler_types.size() + in.spec_token_ids.size() + R);
    out.sampler_temperatures.reserve(in.sampler_temperatures.size() + in.spec_token_ids.size() + R);
    out.sampler_top_p.reserve(in.sampler_top_p.size() + in.spec_token_ids.size() + R);
    out.sampler_min_p.reserve(in.sampler_min_p.size() + in.spec_token_ids.size() + R);
    out.sampler_top_k.reserve(in.sampler_top_k.size() + in.spec_token_ids.size() + R);
    out.sampler_seeds.reserve(in.sampler_seeds.size() + in.spec_token_ids.size() + R);
    out.request_num_samplers.assign(R, 0u);

    // First-sampler index per request (used to clone its config).
    std::vector<int> first_sampler_idx(R, -1);
    {
        int s_off = 0;
        for (int r = 0; r < R; ++r) {
            const int ns = (r < static_cast<int>(in.request_num_samplers.size()))
                ? static_cast<int>(in.request_num_samplers[r]) : 0;
            if (ns > 0) first_sampler_idx[r] = s_off;
            s_off += ns;
        }
    }

    // Walk requests; for each spec request, splice the verification
    // block into sampling_indices/indptr/request_num_samplers and
    // clone sampler params for the new slots. A spec request's normal
    // next-token sampler is redundant when it is the simple token sampler
    // at the last input row: the verification block samples that same row
    // as element 0, and the executor discards the original token slot for
    // spec responses. Dropping it keeps the verify sampling rows dense
    // ([0..n_drafts]) so the CUDA executor can use the greedy argmax fast
    // path instead of the general per-slot sampler.
    out.sampling_indices.reserve(in.sampling_indices.size());
    out.sampling_indptr.reserve(R + 1);
    out.sampling_indptr.push_back(0u);
    int cumulative_slots = 0;
    int original_sampler_off = 0;
    auto push_sampler_config = [&](int idx) {
        out.sampler_types.push_back(
            idx < static_cast<int>(in.sampler_types.size())
                ? in.sampler_types[idx]
                : 1u);
        out.sampler_temperatures.push_back(
            idx < static_cast<int>(in.sampler_temperatures.size())
                ? in.sampler_temperatures[idx]
                : 1.f);
        out.sampler_top_p.push_back(
            idx < static_cast<int>(in.sampler_top_p.size())
                ? in.sampler_top_p[idx]
                : 1.f);
        out.sampler_min_p.push_back(
            idx < static_cast<int>(in.sampler_min_p.size())
                ? in.sampler_min_p[idx]
                : 0.f);
        out.sampler_top_k.push_back(
            idx < static_cast<int>(in.sampler_top_k.size())
                ? in.sampler_top_k[idx]
                : 0u);
        out.sampler_seeds.push_back(
            idx < static_cast<int>(in.sampler_seeds.size())
                ? in.sampler_seeds[idx]
                : 0u);
    };
    for (int r = 0; r < R; ++r) {
        const std::uint32_t lo = in.sampling_indptr[r];
        const std::uint32_t hi = in.sampling_indptr[r + 1];
        const int n_d = out.verify_n_drafts[r];
        const int n_in = static_cast<int>(in.qo_indptr[r + 1] - in.qo_indptr[r]);
        const int fs = first_sampler_idx[r];
        const bool omit_redundant_spec_sampler =
            n_d > 0 &&
            hi == lo + 1 &&
            fs >= 0 &&
            fs < static_cast<int>(in.sampler_types.size()) &&
            is_token_sampler(in.sampler_types[fs]) &&
            n_in > 0 &&
            in.sampling_indices[lo] ==
                static_cast<std::uint32_t>(n_in - 1);
        if (!omit_redundant_spec_sampler) {
            for (std::uint32_t k = lo; k < hi; ++k) {
                out.sampling_indices.push_back(in.sampling_indices[k]);
                push_sampler_config(
                    original_sampler_off + static_cast<int>(k - lo));
            }
            cumulative_slots += static_cast<int>(hi - lo);
            out.request_num_samplers[r] += hi - lo;
        }
        original_sampler_off += static_cast<int>(hi - lo);

        if (n_d > 0) {
            out.verify_slot_start[r] = cumulative_slots;
            const int base = n_in - 1;
            for (int j = 0; j <= n_d; ++j) {
                out.sampling_indices.push_back(static_cast<std::uint32_t>(base + j));
            }
            cumulative_slots += n_d + 1;

            if (fs < 0) {
                std::cerr << "[pie-driver-cuda] spec error: req "
                          << r << " has drafts but no sampler\n";
            } else {
                const std::uint32_t stype = in.sampler_types[fs];
                if (!is_token_sampler(stype)) {
                    std::cerr << "[pie-driver-cuda] spec error: req "
                              << r << " sampler type=" << stype
                              << " incompatible with verify_drafts\n";
                }
                for (int j = 0; j <= n_d; ++j) {
                    push_sampler_config(fs);
                }
            }
            out.request_num_samplers[r] += static_cast<std::uint32_t>(n_d + 1);
        }
        out.sampling_indptr.push_back(static_cast<std::uint32_t>(cumulative_slots));
    }

    return out;
}

}  // namespace pie_cuda_driver
