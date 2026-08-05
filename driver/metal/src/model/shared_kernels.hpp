#pragma once

/// Host mirrors and launch geometry for kernels that are NOT family-specific.
///
/// A `.metal` buffer layout is a property of the kernel, not of the model that
/// happens to dispatch it. `rms_norm.metal` and `row_gather.metal` are shared
/// sources, so their host mirrors belong here once rather than once per family
/// — three private copies of the same struct drift silently, and a mismatch is
/// not a compile error: the GPU reads whatever bytes sit at the offset.

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "../mtl4_context.hpp"

namespace pie::metal::shared_kernels {

/// Params structs, replicated EXACTLY from the .metal sources.
#include "../kernels/rms_params.h"
static_assert(sizeof(RmsParams) == 20);
static_assert(sizeof(VNormParams) == 8);
static_assert(sizeof(GatedRmsParams) == 8);
struct RowGatherParams {    // row_gather.metal       (buffer 3)
    std::uint32_t width;
    std::uint32_t count;
};
#include "../kernels/moe_params.h"
static_assert(sizeof(RouterParams) == 8);
static_assert(sizeof(ExpertCombineParams) == 8);
static_assert(sizeof(MoeRouteParams) == 24);

/// Flat elementwise `dispatchThreads` grid: one thread per element, capped at a
/// 256-wide threadgroup.
inline void elementwise_dispatch(int n, Grid& g, Threadgroup& tg) {
    const int width = n < 256 ? (n > 0 ? n : 1) : 256;
    g = Grid{std::uint32_t(n > 0 ? n : 1), 1, 1};
    tg = Threadgroup{std::uint32_t(width), 1, 1};
}

/// The router's two hard bounds.
///
/// Both are the KERNEL's, mirrored here so the launch shape and the geometry
/// that refuses an oversized config read the same number. The kernel clamps to
/// them; a host that also clamped would route with fewer experts than the
/// config asked for and say nothing, so `geometry_from_facts` refuses instead.
constexpr int kRouterMaxTopK = 16;        // moe_route.metal
constexpr int kRouterMaxExperts = 1024;   // one lane per expert

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

/// `router_topk` in moe_route.metal: one threadgroup per row, one lane per expert.
///
/// Capped at the 1024-thread threadgroup limit, which is also the widest expert
/// count this shape can serve. 32 on gpt-oss, 128 on Qwen3-MoE.
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
    g = Grid{32u * r, std::uint32_t(N > 0 ? N : 1) / 4u, slots};
    tg = Threadgroup{32, 2, 1};
}

/// The tile a batched mixture pads each expert's run to.
///
/// `kQmmBM`'s narrow tile rather than the wide one, and deliberately: the
/// padding an expert wastes is up to `tile - 1` rows, paid `n_experts` times,
/// so the wide tile doubles the waste to buy a shape the routed case rarely
/// fills anyway.
constexpr int kMoeTileRows = 16;

/// When sorting the rows by expert pays for itself.
///
/// The sort turns `n` matvecs into `ceil(count_e / tile)` summed over the
/// experts -- fewer reads of each expert's weights, but a tile that is only
/// part full does the arithmetic of a whole one. The two meet when an expert's
/// run half fills a tile, which for `min(n, n_experts)` touched experts is
/// `n >= n_experts * tile / 2`.
///
/// Below that the matvec wins outright, and it is not close: a decode routes
/// eight pairs over a hundred and twenty-eight experts, where every tile would
/// be one live row in sixteen.
inline bool moe_should_batch(int n_pairs, int n_experts) {
    return n_experts > 0 && n_pairs >= n_experts * kMoeTileRows / 2;
}

/// Rows each expert's run is padded to, for a batch of `n_pairs`.
inline int moe_tile_rows(int n_pairs, int n_experts) {
    return moe_should_batch(n_pairs, n_experts) ? kMoeTileRows : 1;
}

/// How many sorted rows a batch of `n_pairs` can produce.
///
/// The worst case, not the actual: the real count depends on how the router
/// spread the rows, which is a number the GPU has and the host would have to
/// stall to read. Every touched expert can waste `tile - 1` rows and at most
/// `min(n_pairs, n_experts)` experts are touched, so this bound is reached and
/// cannot be tightened without the routing itself.
inline int moe_sorted_rows(int n_pairs, int n_experts) {
    const int n = n_pairs > 0 ? n_pairs : 0;
    const int tile = moe_tile_rows(n, n_experts);
    if (tile <= 1) return n;
    const int touched = n < n_experts ? n : n_experts;
    const int bound = n + touched * (tile - 1);
    return ((bound + tile - 1) / tile) * tile;
}

/// `moe_route_sort`: one threadgroup, sized to the expert count it scans.
inline void moe_route_sort_dispatch(int n_experts, Grid& g, Threadgroup& tg) {
    const std::uint32_t w = router_lane_width(n_experts);
    g = Grid{w, 1, 1};
    tg = Threadgroup{w, 1, 1};
}

/// `moe_route_gather` / `moe_route_scatter`: one thread per element of the
/// sorted stack. `rows` is the padded count, because the padding rows are what
/// the gather has to zero.
inline void moe_route_rows_dispatch(int width, int rows, Grid& g, Threadgroup& tg) {
    const std::uint32_t w = std::uint32_t(width > 0 ? width : 1);
    g = Grid{w, std::uint32_t(rows > 0 ? rows : 1), 1};
    tg = Threadgroup{w < 256u ? w : 256u, 1, 1};
}

/// Bind a POD constant value into a fresh resident slot at (ordinal, index).
///
/// `who` names the caller in the failure, which is the only thing the families
/// ever varied here.
template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, std::uint8_t idx, const V& val,
                       int* count, const char* who) {
    SlotHandle s = ctx.const_slot(ord, idx, sizeof(V));
    if (!s.valid()) {
        throw std::runtime_error(std::string(who) +
                                 " consts: heap_alloc failed (budget too small)");
    }
    std::memcpy(s.contents(), &val, sizeof(V));
    ctx.arg_bind_ordinal(ord, idx, s);
    if (count != nullptr) ++*count;
}

}  // namespace pie::metal::shared_kernels
