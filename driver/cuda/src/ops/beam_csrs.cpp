// Beam [B,P] per-beam CSR construction — pure host logic (SEAM 1+3 glue). See
// beam_csrs.hpp for the contract.
#include "ops/beam_csrs.hpp"

#include <stdexcept>

namespace pie_cuda_driver::ops {

void beam_build_csrs(
    const std::uint32_t* np,
    const std::uint32_t* pages_phys,
    const std::uint32_t* klen,
    int B, int P, int page_size,
    BeamCsrs& out)
{
    out.qo_indptr_h.assign(static_cast<std::size_t>(B) + 1, 0);
    out.kv_page_indptr_h.assign(static_cast<std::size_t>(B) + 1, 0);
    out.kv_last_page_lens_h.assign(static_cast<std::size_t>(B), 0);
    out.mask_indptr_h.assign(static_cast<std::size_t>(B) + 1, 0);
    out.kv_page_indices_h.clear();

    for (int b = 0; b < B; ++b) {
        const int npb = static_cast<int>(np[b]);
        if (npb < 0 || npb > P)
            throw std::runtime_error("beam_build_csrs: np out of [0,P]");
        // One new-token query per beam.
        out.qo_indptr_h[b + 1] = static_cast<std::uint32_t>(b + 1);
        // Page run: the FIRST np[b] slots of row b (drop the padding tail by COUNT,
        // never by value — slot id 0 is a valid page; alpha's guardrail #1).
        out.kv_page_indptr_h[b + 1] =
            out.kv_page_indptr_h[b] + static_cast<std::uint32_t>(npb);
        for (int j = 0; j < npb; ++j)
            out.kv_page_indices_h.push_back(
                pages_phys[static_cast<std::size_t>(b) * P + j]);
        // last_page_len = klen - (np-1)*PAGE (the fill of the final page).
        const int kl = static_cast<int>(klen[b]);
        const int lpl = (npb > 0) ? (kl - (npb - 1) * page_size) : 0;
        if (npb > 0 && (lpl <= 0 || lpl > page_size))
            throw std::runtime_error("beam_build_csrs: last_page_len out of range");
        out.kv_last_page_lens_h[b] = static_cast<std::uint32_t>(lpl);
        // Custom-mask byte offset = prefix-sum of ceil(klen/8).
        out.mask_indptr_h[b + 1] =
            out.mask_indptr_h[b] + (kl + 7) / 8;
    }
}

void beam_resolve_pages(
    const std::uint32_t* pages_slot,
    const std::uint32_t* np,
    const std::uint32_t* slot_to_phys,
    int B, int P, int dict_len,
    std::vector<std::uint32_t>& pages_phys_out)
{
    pages_phys_out.assign(static_cast<std::size_t>(B) * P, 0);
    for (int b = 0; b < B; ++b) {
        const int npb = static_cast<int>(np[b]);
        if (npb < 0 || npb > P)
            throw std::runtime_error("beam_resolve_pages: np out of [0,P]");
        for (int j = 0; j < npb; ++j) {
            const std::uint32_t slot =
                pages_slot[static_cast<std::size_t>(b) * P + j];
            if (slot >= static_cast<std::uint32_t>(dict_len))
                throw std::runtime_error("beam_resolve_pages: slot id out of dict range");
            // Direct index by slot id — slot 0 is a valid entry, never special-cased.
            pages_phys_out[static_cast<std::size_t>(b) * P + j] = slot_to_phys[slot];
        }
    }
}

}  // namespace pie_cuda_driver::ops
