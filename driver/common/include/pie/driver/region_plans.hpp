#pragma once

// V2 rung 3c (north-star-dsl.md "RUNG 3 SPEC"): the region table is the
// SOURCE OF TRUTH for the four planned words. `apply_region_plans` runs
// at the StepLaunch -> LaunchView assembly — before any consumer — so
// every reader downstream (frame prepare, the walkers, dispatch) sees
// values DERIVED from the table. Since 3c-ii the wire words are gone:
// this derivation IS the plans' single source (the 3b live equivalence
// proof — strict word equality on every fire shape, declines included
// — is what licensed retiring them).
//
// The derivation mirrors the engine's plan/decline rules LITERALLY
// (batch.rs planned_prefix_wire_rows / planned_unmasked_prefix_wire_
// rows / planned_full_depth_request_split and the uniform-k stamp):
// the table states FACTS, these rules turn them into the PLANS.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "fire/view.hpp"

namespace pie_native {

struct RegionPlans {
    std::uint32_t hook_free_prefix_rows = 0xffffffffu;
    std::uint32_t unmasked_prefix_rows = 0xffffffffu;
    std::uint32_t max_layers = 0xffffffffu;
    std::uint32_t full_depth_rows = 0xffffffffu;
};

// Derives the plans from the view's region table. Returns false (and
// leaves `out` all-unplanned) when no table is present. Throws on a
// structurally invalid table.
inline bool derive_region_plans(const LaunchView& view, RegionPlans& out) {
    if (view.region_row_indptr.empty()) return false;
    const std::uint32_t* ind = view.region_row_indptr.data();
    const std::uint32_t* sig = view.region_sig.data();
    const std::uint32_t* rk = view.region_k.data();
    const std::size_t nregions = view.region_row_indptr.size() - 1;
    if (view.region_sig.size() != nregions ||
        view.region_k.size() != nregions) {
        throw std::runtime_error(
            "region table: parallel slice lengths disagree");
    }
    for (std::size_t r = 0; r < nregions; ++r) {
        if (ind[r + 1] <= ind[r]) {
            throw std::runtime_error(
                "region table: bounds not strictly ascending");
        }
    }
    constexpr std::uint32_t kUnplanned = 0xffffffffu;
    const std::uint32_t total = ind[nregions];
    const auto first_row_with = [&](std::uint32_t bit) -> std::uint32_t {
        for (std::size_t r = 0; r < nregions; ++r) {
            if (sig[r] & bit) return ind[r];
        }
        return kUnplanned;
    };

    // hook: UNPLANNED iff no hook member; else the first hook row.
    out.hook_free_prefix_rows = first_row_with(PIE_REGION_SIG_HOOK);

    // mask: UNPLANNED iff a lane carries hook AND mask, or no lane
    // carries a mask; else the first masked row.
    out.unmasked_prefix_rows = first_row_with(PIE_REGION_SIG_MASK);
    for (std::size_t r = 0; r < nregions; ++r) {
        if ((sig[r] & PIE_REGION_SIG_HOOK) &&
            (sig[r] & PIE_REGION_SIG_MASK)) {
            out.unmasked_prefix_rows = kUnplanned;
        }
    }

    // depth split (S-2, mirrored literally): declined by the disarm
    // env, any both-axes lane (trunc with hook/mask/lora), any
    // multi-token lane, no/all truncation, mixed k, or a
    // non-contiguous truncated block off the (mask|hook) tail.
    static const bool union_armed = [] {
        const char* v = std::getenv("PIE_DEPTH_UNION");
        return v == nullptr || v[0] != '0';
    }();
    std::uint32_t trunc_rows = 0;
    bool trunc_mixed_k = false, decline = !union_armed;
    std::uint32_t uniform_k = kUnplanned;
    for (std::size_t r = 0; r < nregions; ++r) {
        if (sig[r] & PIE_REGION_SIG_MULTI_TOKEN) decline = true;
        if (sig[r] & PIE_REGION_SIG_TRUNCATED) {
            if ((sig[r] & (PIE_REGION_SIG_HOOK | PIE_REGION_SIG_MASK |
                           PIE_REGION_SIG_LORA)) != 0) {
                decline = true;
            }
            trunc_rows += ind[r + 1] - ind[r];
            if (uniform_k == kUnplanned) {
                uniform_k = rk[r];
            } else if (uniform_k != rk[r]) {
                trunc_mixed_k = true;
            }
        }
    }
    if (!decline && !trunc_mixed_k && trunc_rows != 0 &&
        trunc_rows != total) {
        std::uint32_t tail_rows = 0;
        std::size_t r = nregions;
        while (r > 0 && (sig[r - 1] &
                         (PIE_REGION_SIG_MASK | PIE_REGION_SIG_HOOK))) {
            tail_rows += ind[r] - ind[r - 1];
            --r;
        }
        const std::uint32_t split = total - tail_rows - trunc_rows;
        // The truncated block must exactly fill [split, total - tail).
        bool block_ok = true;
        for (std::size_t q = 0; q < nregions; ++q) {
            const bool inside =
                ind[q] >= split && ind[q + 1] <= total - tail_rows;
            const bool trunc = (sig[q] & PIE_REGION_SIG_TRUNCATED) != 0;
            if (inside != trunc) block_ok = false;
        }
        if (block_ok) out.full_depth_rows = split;
    }

    // max_layers: the split's uniform suffix k, or the uniform stamp
    // (EVERY member truncated with one k); FULL otherwise.
    if (out.full_depth_rows != kUnplanned) {
        out.max_layers = uniform_k;
    } else if (trunc_rows == total && total != 0 && !trunc_mixed_k) {
        out.max_layers = uniform_k;
    }
    return true;
}

// Derivation at the assembly boundary (3c-ii: the wire words are
// gone, so this IS the plans' birth). `view.region_*` must already be
// populated; no table -> everything UNPLANNED, the legacy discipline.
inline void apply_region_plans(LaunchView& view) {
    RegionPlans d;
    const bool present = derive_region_plans(view, d);
    if (std::getenv("PIE_REGION_TRACE") != nullptr) {
        std::fprintf(stderr, "[region] table=%d regions=%zu k=%d\n",
                     present ? 1 : 0,
                     present ? view.region_row_indptr.size() - 1 : 0,
                     d.max_layers == 0xffffffffu
                         ? -1
                         : static_cast<int>(d.max_layers));
    }
    view.planned_hook_free_prefix_rows = d.hook_free_prefix_rows;
    view.planned_unmasked_prefix_rows = d.unmasked_prefix_rows;
    view.planned_max_layers = d.max_layers;
    view.planned_full_depth_rows = d.full_depth_rows;
}

}  // namespace pie_native
