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
inline constexpr std::uint32_t kGvSmallSpec   = 1u << 0;
inline constexpr std::uint32_t kGvRsVerify    = 1u << 1;
inline constexpr std::uint32_t kGvCustomMask  = 1u << 2;
// The LM head either materialises `[rows, vocab]` logits or reduces them to
// token ids as it produces them (§20.37). Those are different kernel sequences,
// so they are different captures and must not share a key.
inline constexpr std::uint32_t kGvFusedArgmax = 1u << 3;
// A prefill-carrying wave either emits the full `[N, vocab]` logits or gathers
// the R sampled rows into a compact `[R, vocab]` block, and the captured graph
// bakes in which (`logit_row_indices_d` / `num_logit_rows` are recorded, not
// re-read at replay). `frame.cpp` decides it per fire — `compact_logits` is
// false whenever MTP draft rows are in play — so without this bit a wave that
// needs the full emit can replay a compact capture and sample from rows that
// were never written. Same rule as `kGvFusedArgmax`: different kernel
// sequence, different key.
inline constexpr std::uint32_t kGvCompactLogits = 1u << 4;
inline constexpr int           kGvLayoutShift = 5;

inline constexpr std::uint32_t kGvFlagMask =
    kGvSmallSpec | kGvRsVerify | kGvCustomMask | kGvFusedArgmax |
    kGvCompactLogits;

// By construction: every flag is below the layout field, so no `graph_layout`
// value can ever alias a flag bit.
static_assert(kGvFlagMask < (1u << kGvLayoutShift),
              "graph_variant flag bits overlap the graph_layout field");

constexpr std::uint32_t make_graph_variant(bool small_spec,
                                           bool rs_verify,
                                           bool custom_mask,
                                           bool fused_argmax,
                                           bool compact_logits,
                                           std::uint32_t graph_layout) {
    return (small_spec     ? kGvSmallSpec     : 0u) |
           (rs_verify      ? kGvRsVerify      : 0u) |
           (custom_mask    ? kGvCustomMask    : 0u) |
           (fused_argmax   ? kGvFusedArgmax   : 0u) |
           (compact_logits ? kGvCompactLogits : 0u) |
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
                  make_graph_variant(true, false, false, false, false, 0u),
              "#24 fix: graph_layout=64 must hash distinctly from small_spec");
// The same property for each flag added since: the layout field starts above
// ALL of them, so no layout value can alias one.
static_assert(make_graph_variant(false, false, false, false, false, 1u) !=
                  make_graph_variant(false, false, false, true, false, 0u),
              "graph_layout=1 must hash distinctly from fused_argmax");
static_assert(make_graph_variant(false, false, false, false, false, 1u) !=
                  make_graph_variant(false, false, false, false, true, 0u),
              "graph_layout=1 must hash distinctly from compact_logits");
// And the two newest flags are distinct from each other — the property that
// actually matters here is that a compact-logits capture and a full-emit
// capture of the same shape never share a key.
static_assert(make_graph_variant(false, false, false, false, true, 3u) !=
                  make_graph_variant(false, false, false, false, false, 3u),
              "compact_logits must change the key at a fixed layout");

}  // namespace pie_cuda_driver
