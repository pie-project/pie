#pragma once

/// The routed FFN's launch shapes, and the two bounds the router kernel has.
///
/// Everything here is read off `moe_route.metal` and `quantized_qmv.metal`:
/// how wide a threadgroup has to be for the router's cross-simdgroup
/// reduction, how many output rows a routed matvec's threadgroup owns, how
/// many rows a sort can produce. None of it is a property of the model that
/// dispatches it, which is why it is on this side.
///
/// **What deliberately did NOT come down with it.** `moe_should_batch` and
/// `moe_tile_rows` read `DeviceTuning` — a sweep of one machine, overridable
/// by an env var — and they stay in `driver-metal`. A tuning table pushed
/// downhill so that a launch shape can see it is not a one-way dependency; it
/// is a two-way one with the arrow hidden, which is exactly the defect
/// `.wiki/kernel-refactor.md` §1.1 records for CUDA's `runahead.hpp`. So the
/// functions here that need a tile TAKE one:
///
///     const int tile = moe_tile_rows(pairs, experts);   // driver, policy
///     const int rows = moe::sorted_rows(pairs, experts, tile);  // kernels
///
/// which is the same fix `AttentionWorkspace::allocate` took when it stopped
/// reading `frame_dispatch_depth` and started being handed it.

#include <cstddef>

#include "pie/kernels/grid.h"

namespace pie::kernels::moe {

/// The router's two hard bounds, both the KERNEL's.
///
/// Mirrored for the host so the launch shape and the geometry that refuses an
/// oversized config read the same number. The kernel clamps to them; a host
/// that also clamped would route with fewer experts than the config asked for
/// and say nothing, so the caller's geometry check refuses instead.
constexpr int kRouterMaxTopK = 16;      // moe_route.metal
constexpr int kRouterMaxExperts = 1024; // one lane per expert

/// The three row tiles the routed GEMM is compiled for, narrow first.
///
/// The same three the table declares as `TILE_M` in
/// `crates/kernels-metal/src/axes.rs`; a fourth would be a fourth
/// instantiation, not a constant.
inline constexpr int kTileWidths[3] = {16, 32, 64};

/// The narrow tile, and the one the batching threshold is written against.
constexpr int kTileRowsNarrow = 16;

/// The launch width for a router kernel that gives every expert a lane.
///
/// Rounded up to a whole simdgroup, because the kernel reduces ACROSS
/// simdgroups and a partial one would leave a reduction slot uninitialised.
/// Clamped first, which is the same answer as clamping after: the cap is a
/// multiple of 32, so `round_up_32(min(n, 1024))` and `min(round_up_32(n),
/// 1024)` agree everywhere.
inline std::uint32_t router_lane_width(int n_experts) {
    const int lanes =
        n_experts < 1 ? 1 : (n_experts > kRouterMaxExperts ? kRouterMaxExperts : n_experts);
    return (std::uint32_t(lanes) + 31u) / 32u * 32u;
}

/// `router_topk` in moe_route.metal: one threadgroup per row, one lane per
/// expert. Capped at the 1024-thread threadgroup limit, which is also the
/// widest expert count this shape can serve. 32 on gpt-oss, 128 on Qwen3-MoE.
inline void router_topk_dispatch(int n_experts, Grid& g, Threadgroup& tg, int rows = 1) {
    const std::uint32_t w = router_lane_width(n_experts);
    g = Grid{w, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{w, 1, 1};
}

/// `expert_combine`: one thread per output element, one row per `gid.y`. The k
/// slots are summed inside the kernel, so they do not appear in the grid.
inline void expert_combine_dispatch(int hidden, Grid& g, Threadgroup& tg, int rows = 1) {
    const std::uint32_t w = std::uint32_t(hidden > 0 ? hidden : 1);
    g = Grid{w, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{w < 256u ? w : 256u, 1, 1};
}

/// The routed matvec's launch shape.
///
/// The same row decomposition as the dense `qmv_dispatch`, because it is the
/// same kernel body: `qmv_gptoss_impl` computes
/// `out_row = tid.y * (num_simdgroups * results_per_simdgroup)` with
/// `num_simdgroups` fixed at 2, so a threadgroup owns EIGHT output rows and
/// needs two simdgroups to write them. One simdgroup covers only the first
/// four, and `grid.y = N` then runs `out_row` up to 8N -- half the rows stale,
/// and the write past the end of the buffer.
///
/// The two axes the dense shape does not have are the token row on `tid.x` and
/// the expert slot on `tid.z`, and they are NOT interchangeable: the kernel
/// selects its expert with `sel = row * slots_per_row + slot`, so folding the
/// rows into the slot axis routes every row through row 0's experts.
inline void routed_qmv_dispatch(int N, int experts_per_token, Grid& g, Threadgroup& tg,
                                int rows = 1) {
    const std::uint32_t r = std::uint32_t(rows > 0 ? rows : 1);
    const std::uint32_t slots = std::uint32_t(experts_per_token > 0 ? experts_per_token : 1);
    // Rounded UP: the kernel writes four outputs a simdgroup and guards each
    // against `out_vec_size`, so a width that is not a multiple of four gets
    // a partial group rather than losing its tail. See `qmv_dispatch`.
    // The 4 is `results_per_simdgroup` in `quantized_qmv.metal`, which was swept
    // and is at a peak in both directions -- see the table there before moving it.
    g = Grid{32u * r, (std::uint32_t(N > 0 ? N : 1) + 3u) / 4u, slots};
    tg = Threadgroup{32, 2, 1};
}

/// `moe_route_sort`: one threadgroup, sized to the expert count it scans.
inline void route_sort_dispatch(int n_experts, Grid& g, Threadgroup& tg) {
    const std::uint32_t w = router_lane_width(n_experts);
    g = Grid{w, 1, 1};
    tg = Threadgroup{w, 1, 1};
}

/// `moe_route_gather` / `moe_route_scatter`: one thread per element of the
/// sorted stack. `rows` is the padded count, because the padding rows are what
/// the gather has to zero.
inline void route_rows_dispatch(int width, int rows, Grid& g, Threadgroup& tg) {
    const std::uint32_t w = std::uint32_t(width > 0 ? width : 1);
    g = Grid{w, std::uint32_t(rows > 0 ? rows : 1), 1};
    tg = Threadgroup{w < 256u ? w : 256u, 1, 1};
}

/// Which row of a routed PSO table a tile selects. The tables hold the widths
/// `kTileWidths` lists, in that order.
inline int bm_slot(int tile) { return tile >= 64 ? 2 : (tile >= 32 ? 1 : 0); }

/// How many sorted rows a batch of `n_pairs` can produce, at a given tile.
///
/// The worst case, not the actual: the real count depends on how the router
/// spread the rows, which is a number the GPU has and the host would have to
/// stall to read. Every touched expert can waste `tile - 1` rows and at most
/// `min(n_pairs, n_experts)` experts are touched, so this bound is reached and
/// cannot be tightened without the routing itself.
///
/// `tile` is a PARAMETER and not a call to `moe_tile_rows`: which tile a batch
/// gets is a tuning decision (see this header's opening note), and a bound the
/// kernel guarantees must not depend on one.
inline int sorted_rows(int n_pairs, int n_experts, int tile) {
    const int n = n_pairs > 0 ? n_pairs : 0;
    if (tile <= 1) return n;
    const int touched = n < n_experts ? n : n_experts;
    const int bound = n + touched * (tile - 1);
    return ((bound + tile - 1) / tile) * tile;
}

/// Flat elementwise `dispatchThreads` grid: one thread per element, capped at a
/// 256-wide threadgroup. Not MoE's, but the mixture's SwiGLU runs on it over
/// the whole `[rows, k, width]` expert stack, and it has no better home until
/// the rest of the families come down.
inline void elementwise_dispatch(int n, Grid& g, Threadgroup& tg) {
    const int width = n < 256 ? (n > 0 ? n : 1) : 256;
    g = Grid{std::uint32_t(n > 0 ? n : 1), 1, 1};
    tg = Threadgroup{std::uint32_t(width), 1, 1};
}

/// The routed SiLU-mul, over the whole `[rows * k, moe_intermediate]` stack. It
/// is one flat elementwise dispatch precisely because gate, up and out share a
/// layout -- the slot axis needs no special handling.
inline void expert_silu_dispatch(int moe_intermediate, int experts_per_token, Grid& g,
                                 Threadgroup& tg, int rows = 1) {
    const std::size_t n = std::size_t(moe_intermediate > 0 ? moe_intermediate : 1) *
                          std::size_t(experts_per_token > 0 ? experts_per_token : 1) *
                          std::size_t(rows > 0 ? rows : 1);
    elementwise_dispatch(static_cast<int>(n), g, tg);
}

}  // namespace pie::kernels::moe
