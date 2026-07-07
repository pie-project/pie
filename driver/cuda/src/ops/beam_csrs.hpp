// Beam [B,P] per-beam CSR construction (SEAM 1+3 host glue, charlie's G2).
// Pure host logic (no CUDA) so it is independently unit-testable against the beam
// goldens. Derives the FlashInfer-shaped CSRs (page-run indptr, flat physical
// page indices, last-page lens, custom-mask byte offsets) from the epilogue's
// per-beam np/Pages/klen. Padding is dropped by np[b] COUNT, NEVER by Pages==0
// (slot id 0 is a valid page — alpha's guardrail #1).
#pragma once

#include <cstdint>
#include <vector>

namespace pie_cuda_driver::ops {

struct BeamCsrs {
    std::vector<std::uint32_t> qo_indptr_h;         // [B+1] = [0,1,..,B]
    std::vector<std::uint32_t> kv_page_indptr_h;    // [B+1] page-run CSR
    std::vector<std::uint32_t> kv_page_indices_h;   // [sum np] flat physical pages
    std::vector<std::uint32_t> kv_last_page_lens_h; // [B]
    std::vector<std::int32_t>  mask_indptr_h;       // [B+1] custom-mask BYTE offsets
};

// Build the per-beam CSRs. `pages_phys` is [B*P] physical page ids row-major
// (padded to P per beam); `np[b]` is the live page count; `klen[b]` the physical
// span. Throws on np/last_page_len out of range.
void beam_build_csrs(
    const std::uint32_t* np,          // [B]
    const std::uint32_t* pages_phys,  // [B*P]
    const std::uint32_t* klen,        // [B]
    int B, int P, int page_size,
    BeamCsrs& out);

// Resolve the epilogue's device-produced `Pages` (working-set SLOT ids, read back
// to host) to PHYSICAL page ids via the fire's slot→physical dict (alpha's gap-2
// (a): host-resolve now, reuse the normal path's table; the device-produced
// resolver is a deferred perf pass). Indexes `slot_to_phys` DIRECTLY by slot id
// (slot 0 is valid — never special-cased); one shared dict covers all B beams
// (fork shares one slot space). Only the first `np[b]` slots per beam are resolved
// (the padding tail is left as-is and dropped later by `beam_build_csrs`). Throws
// on an out-of-range slot.
void beam_resolve_pages(
    const std::uint32_t* pages_slot,  // [B*P] slot ids (host copy of device Pages)
    const std::uint32_t* np,          // [B]
    const std::uint32_t* slot_to_phys,// [dict_len] slot id → physical page id
    int B, int P, int dict_len,
    std::vector<std::uint32_t>& pages_phys_out);  // [B*P] physical (pad preserved)

}  // namespace pie_cuda_driver::ops
