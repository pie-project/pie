// beam_build_csrs unit test — the per-beam CSR construction (SEAM 1+3 host glue)
// against the fork-freeze golden geometry. Verifies the page-run CSR, last-page
// lens, and mask byte-offsets, and the CRITICAL guardrail: padding is dropped by
// np[b] COUNT, never by Pages==0 (slot 0 is a valid page id). Self-contained; no
// framework; failures abort non-zero for CTest.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "ops/beam_csrs.hpp"

namespace {
int g_failures = 0;
#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",                \
                         __FILE__, __LINE__, msg);                   \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

void test_fork_freeze_csrs() {
    // Golden (ptir_examples.rs:232): P=3, PAGE=4, BB=2. Pages=[5,6,7|5,6,0],
    // np=[3,2], klen=[9,7]. (Row 1's 3rd slot is padding — dropped by np, not by
    // value; note slot id 0 here is a legitimate value the count-drop must NOT eat.)
    const int B = 2, P = 3, PAGE = 4;
    const std::vector<std::uint32_t> np = {3, 2};
    const std::vector<std::uint32_t> pages = {5, 6, 7, 5, 6, 0};
    const std::vector<std::uint32_t> klen = {9, 7};

    pie_cuda_driver::ops::BeamCsrs c;
    pie_cuda_driver::ops::beam_build_csrs(np.data(), pages.data(), klen.data(),
                                          B, P, PAGE, c);

    const std::vector<std::uint32_t> want_qo = {0, 1, 2};
    const std::vector<std::uint32_t> want_kvpp = {0, 3, 5};
    const std::vector<std::uint32_t> want_idx = {5, 6, 7, 5, 6};  // pad slot 0 dropped by COUNT
    const std::vector<std::uint32_t> want_lpl = {1, 3};           // 9-2*4=1 ; 7-1*4=3
    const std::vector<std::int32_t>  want_mip = {0, 2, 3};        // ceil(9/8)=2 ; +ceil(7/8)=1

    CHECK(c.qo_indptr_h == want_qo, "qo_indptr mismatch");
    CHECK(c.kv_page_indptr_h == want_kvpp, "kv_page_indptr mismatch");
    CHECK(c.kv_page_indices_h == want_idx, "kv_page_indices mismatch (pad-drop by count?)");
    CHECK(c.kv_last_page_lens_h == want_lpl, "kv_last_page_lens mismatch");
    CHECK(c.mask_indptr_h == want_mip, "mask_indptr (bytes) mismatch");
}

void test_slot_zero_not_dropped() {
    // Guardrail #1 sharp case: a beam whose FIRST (valid) page is slot 0. A
    // value-based Pages!=0 filter would corrupt this run; the count-based drop
    // keeps it. np=2 → keep [0, 7]; the trailing 0 (pad) is dropped by count.
    const int B = 1, P = 3, PAGE = 4;
    const std::vector<std::uint32_t> np = {2};
    const std::vector<std::uint32_t> pages = {0, 7, 0};  // slot 0 is a REAL page here
    const std::vector<std::uint32_t> klen = {5};         // (2-1)*4+1

    pie_cuda_driver::ops::BeamCsrs c;
    pie_cuda_driver::ops::beam_build_csrs(np.data(), pages.data(), klen.data(),
                                          B, P, PAGE, c);
    const std::vector<std::uint32_t> want_idx = {0, 7};  // slot 0 KEPT, trailing pad dropped
    CHECK(c.kv_page_indices_h == want_idx, "slot-0 page must NOT be dropped (guardrail #1)");
    CHECK(c.kv_last_page_lens_h == std::vector<std::uint32_t>{1}, "lpl for klen=5,np=2");
}

void test_resolve_pages() {
    // Epilogue Pages are SLOT ids; resolve to PHYSICAL via a slot→phys dict. Fork
    // shares one slot space (single dict all B). Slot 0 is a valid dict entry —
    // resolved by DIRECT index, never special-cased.
    const int B = 2, P = 3;
    // dict: slot -> physical block id (slot 0 → real page 42, not a sentinel).
    const std::vector<std::uint32_t> dict = {42, 11, 12, 13, 14, 15, 16, 17};
    const std::vector<std::uint32_t> np = {3, 2};
    // Beam 0 uses slots [5,6,7]; beam 1 uses [0,6] (slot 0 first — the sharp case).
    const std::vector<std::uint32_t> pages_slot = {5, 6, 7, 0, 6, 99};  // 99 = pad (np=2 drops it)

    std::vector<std::uint32_t> phys;
    pie_cuda_driver::ops::beam_resolve_pages(
        pages_slot.data(), np.data(), dict.data(), B, P, (int)dict.size(), phys);

    // Resolved: beam0 [dict[5],dict[6],dict[7]]=[15,16,17]; beam1 [dict[0],dict[6],pad]
    //         = [42,16,0]. Pad slot 99 NOT resolved (out of np) → stays 0.
    const std::vector<std::uint32_t> want = {15, 16, 17, 42, 16, 0};
    CHECK(phys == want, "resolve: slot→phys via direct index (slot 0 = dict[0], pad untouched)");

    // And the resolved physical pages feed beam_build_csrs correctly.
    const std::vector<std::uint32_t> klen = {9, 5};
    pie_cuda_driver::ops::BeamCsrs c;
    pie_cuda_driver::ops::beam_build_csrs(np.data(), phys.data(), klen.data(), B, P, 4, c);
    CHECK(c.kv_page_indices_h == (std::vector<std::uint32_t>{15, 16, 17, 42, 16}),
          "resolved physical pages → CSR indices (beam1 keeps physical 42 from slot 0)");
}

}  // namespace

int main() {
    test_fork_freeze_csrs();
    test_slot_zero_not_dropped();
    test_resolve_pages();
    if (g_failures) {
        std::fprintf(stderr, "beam_build_csrs: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("beam_build_csrs: fork-freeze CSRs + slot-0 guardrail OK\n");
    return 0;
}
