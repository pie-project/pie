#pragma once

#include <cstddef>
#include <cstdint>

namespace pie_cuda_driver {

// ── Forward-graph-cache variant bitfield (#24) ──────────────────────────────
// `graph_variant` keys the captured-CUDA-graph cache together with (R, N). It
// packs a few boolean capture-shape flags plus the model's `graph_layout`
// descriptor. The flag bits MUST live entirely BELOW the `graph_layout` field —
// if a layout value's bits reach a flag bit, two different forward configs hash
// to the same `ForwardGraphKey` and the WRONG captured graph is replayed (a
// silent miscompute).
//
// #24 was a LATENT instance of exactly that: the old encoding shifted
// `graph_layout << 3` with the spec flags at bits 9/10 (`small_spec`=0x200,
// `rs_verify`=0x400). The graph cache is decode-only and real decode layouts
// stay < 64 (`xqa_decode_graph_layout` returns 48..63 → `63<<3 = 0x1F8`, one
// page-bucket bump below bit 9), so it never fired with real values — but
// `64<<3 == 0x200` aliases `small_spec` by construction. The fix shifts the
// layout above ALL flags and `static_assert`s the masks can't overlap.
inline constexpr std::uint32_t kGvTpGreedy    = 1u << 0;
inline constexpr std::uint32_t kGvSingleGpu   = 1u << 1;
inline constexpr std::uint32_t kGvFwdHandles  = 1u << 2;
inline constexpr std::uint32_t kGvSmallSpec   = 1u << 3;
inline constexpr std::uint32_t kGvRsVerify    = 1u << 4;
inline constexpr int           kGvLayoutShift = 5;

inline constexpr std::uint32_t kGvFlagMask =
    kGvTpGreedy | kGvSingleGpu | kGvFwdHandles | kGvSmallSpec | kGvRsVerify;

// By construction: every flag is below the layout field, so no `graph_layout`
// value can ever alias a flag bit.
static_assert(kGvFlagMask < (1u << kGvLayoutShift),
              "graph_variant flag bits overlap the graph_layout field");

constexpr std::uint32_t make_graph_variant(bool tp_greedy, bool single_gpu,
                                           bool fwd_handles, bool small_spec,
                                           bool rs_verify,
                                           std::uint32_t graph_layout) {
    return (tp_greedy   ? kGvTpGreedy   : 0u) |
           (single_gpu  ? kGvSingleGpu  : 0u) |
           (fwd_handles ? kGvFwdHandles : 0u) |
           (small_spec  ? kGvSmallSpec  : 0u) |
           (rs_verify   ? kGvRsVerify   : 0u) |
           (graph_layout << kGvLayoutShift);
}

inline bool graph_replay_has_no_host_resets(
    bool uses_slots,
    const std::uint8_t* is_fresh,
    std::size_t requests) noexcept {
    if (!uses_slots) return true;
    if (is_fresh == nullptr) return false;
    for (std::size_t request = 0; request < requests; ++request) {
        if (is_fresh[request] != 0) return false;
    }
    return true;
}

// The OLD (pre-#24) encoding, kept only to compile-time-prove the latent
// collision was real and the fix closes it at the one-bump-away boundary.
constexpr std::uint32_t gv_old_encode_for_proof(std::uint32_t graph_layout,
                                                bool small_spec,
                                                bool rs_verify) {
    return (graph_layout << 3) | (small_spec ? 0x200u : 0u) |
           (rs_verify ? 0x400u : 0u);
}

// Precondition (the bug WAS real at the boundary): under the old encoding,
// {graph_layout=64, no flags} aliased {graph_layout=0, small_spec}.
static_assert(gv_old_encode_for_proof(64u, false, false) ==
                  gv_old_encode_for_proof(0u, true, false),
              "#24 precondition: OLD encoding aliased graph_layout=64 with "
              "small_spec — the latent collision this fix closes");
// And the fix keeps them distinct.
static_assert(make_graph_variant(false, false, false, false, false, 64u) !=
                  make_graph_variant(false, false, false, true, false, 0u),
              "#24 fix: graph_layout=64 must hash distinctly from small_spec");

}  // namespace pie_cuda_driver
