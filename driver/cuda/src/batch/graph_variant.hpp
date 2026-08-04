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
// Stage 6 increment 4: a capture whose body carried live stage hooks (the
// per-layer PTIR attention-phase launches are inside the graph). A hook fire
// and a plain fire of the same (R, N, layout) have different kernel
// sequences, so they must never share a cache entry. Layout shifts to bit 4.
inline constexpr std::uint32_t kGvHasHooks    = 1u << 3;
// The unionized supergraph (S3): a HIGH bit so it can never collide with
// the shifted layout field. A supergraph key deliberately OMITS the
// custom-mask bit — folding masked and unmasked fires into one exec is
// the union's whole point — so the bit keeps supergraph and plain
// captures from aliasing at the same (R, N, layout).
inline constexpr std::uint32_t kGvSupergraph  = 1u << 31;
// Lora-carrying captures (campaign step 3b): their execs live in the
// fingerprint-partitioned lora store, and the bit keeps their shape keys
// from aliasing plain or supergraph captures at the same (R, N, layout).
inline constexpr std::uint32_t kGvLora        = 1u << 30;

// NS-3: the spatial mask fire's exec partition. The SPLIT itself rides
// the model's graph layout hash (llama_like_decode_graph_layout mixes
// both plans' layouts with the split — the same hash-keyed posture every
// layout already takes); this bit only keeps spatial and non-spatial
// execs out of each other's slots.
inline constexpr std::uint32_t kGvSpatial     = 1u << 29;
inline constexpr int           kGvLayoutShift = 4;

// Upstream §20.37 (merged): the LM head either materialises logits or
// reduces to token ids in-GEMM — different kernel sequences, different
// captures. Rehomed to a HIGH bit: upstream gave it bit 3, which the
// tart hook bit already owns.
inline constexpr std::uint32_t kGvFusedArgmax = 1u << 28;

inline constexpr std::uint32_t kGvFlagMask =
    kGvSmallSpec | kGvRsVerify | kGvCustomMask | kGvHasHooks;

// By construction: every flag is below the layout field, so no `graph_layout`
// value can ever alias a flag bit.
static_assert(kGvFlagMask < (1u << kGvLayoutShift),
              "graph_variant flag bits overlap the graph_layout field");

constexpr std::uint32_t make_graph_variant(bool small_spec,
                                           bool rs_verify,
                                           bool custom_mask,
                                           bool fused_argmax,
                                           std::uint32_t graph_layout,
                                           bool has_hooks = false) {
    return (small_spec   ? kGvSmallSpec   : 0u) |
           (rs_verify    ? kGvRsVerify    : 0u) |
           (custom_mask  ? kGvCustomMask  : 0u) |
           (fused_argmax ? kGvFusedArgmax : 0u) |
           (has_hooks    ? kGvHasHooks    : 0u) |
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
static_assert(make_graph_variant(false, false, false, false, 64u) !=
                  make_graph_variant(true, false, false, false, 0u),
              "#24 fix: graph_layout=64 must hash distinctly from small_spec");
// Increment 4 re-proof at the new boundary (layout shift 3 -> 4): the layout
// value whose lowest bit would land on the NEW flag (kGvHasHooks, bit 3)
// must stay distinct from the hook flag, and a hook capture must never
// alias a hookless one.
static_assert(make_graph_variant(false, false, false, false, 1u) !=
                  make_graph_variant(false, false, false, false, 0u,
                                     /*has_hooks=*/true),
              "#24 discipline: graph_layout=1 must hash distinctly from "
              "kGvHasHooks after the layout shift moved to 4");
static_assert(make_graph_variant(false, false, false, false, 7u, true) !=
                  make_graph_variant(false, false, false, false, 7u, false),
              "kGvHasHooks must separate hook captures from plain ones");
// The same property for the merged upstream flag (rehomed to bit 28).
static_assert(make_graph_variant(false, false, false, false, 1u) !=
                  make_graph_variant(false, false, false, true, 0u),
              "graph_layout=1 must hash distinctly from fused_argmax");

}  // namespace pie_cuda_driver
