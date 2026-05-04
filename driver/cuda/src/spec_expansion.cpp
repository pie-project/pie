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
    out.sampler_types.assign       (in.sampler_types.begin(),        in.sampler_types.end());
    out.sampler_temperatures.assign(in.sampler_temperatures.begin(), in.sampler_temperatures.end());
    out.sampler_top_p.assign       (in.sampler_top_p.begin(),        in.sampler_top_p.end());
    out.sampler_min_p.assign       (in.sampler_min_p.begin(),        in.sampler_min_p.end());
    out.sampler_top_k.assign       (in.sampler_top_k.begin(),        in.sampler_top_k.end());
    out.sampler_seeds.assign       (in.sampler_seeds.begin(),        in.sampler_seeds.end());
    out.request_num_samplers.assign(in.request_num_samplers.begin(), in.request_num_samplers.end());

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
    // clone sampler params for the new slots.
    out.sampling_indices.reserve(in.sampling_indices.size());
    out.sampling_indptr.reserve(R + 1);
    out.sampling_indptr.push_back(0u);
    int cumulative_slots = 0;
    for (int r = 0; r < R; ++r) {
        const std::uint32_t lo = in.sampling_indptr[r];
        const std::uint32_t hi = in.sampling_indptr[r + 1];
        for (std::uint32_t k = lo; k < hi; ++k) {
            out.sampling_indices.push_back(in.sampling_indices[k]);
        }
        cumulative_slots += static_cast<int>(hi - lo);

        const int n_d = out.verify_n_drafts[r];
        if (n_d > 0) {
            const int n_in = static_cast<int>(in.qo_indptr[r + 1] - in.qo_indptr[r]);
            out.verify_slot_start[r] = cumulative_slots;
            const int base = n_in - 1;
            for (int j = 0; j <= n_d; ++j) {
                out.sampling_indices.push_back(static_cast<std::uint32_t>(base + j));
            }
            cumulative_slots += n_d + 1;

            const int fs = first_sampler_idx[r];
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
                    out.sampler_types.push_back(stype);
                    out.sampler_temperatures.push_back(in.sampler_temperatures[fs]);
                    out.sampler_top_p.push_back(in.sampler_top_p[fs]);
                    out.sampler_min_p.push_back(in.sampler_min_p[fs]);
                    out.sampler_top_k.push_back(in.sampler_top_k[fs]);
                    out.sampler_seeds.push_back(in.sampler_seeds[fs]);
                }
            }
            out.request_num_samplers[r] += static_cast<std::uint32_t>(n_d + 1);
        }
        out.sampling_indptr.push_back(static_cast<std::uint32_t>(cumulative_slots));
    }

    return out;
}

}  // namespace pie_cuda_driver
