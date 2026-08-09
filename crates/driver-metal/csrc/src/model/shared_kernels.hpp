#pragma once

/// Host mirrors and launch geometry for kernels that are NOT family-specific.
///
/// A `.metal` buffer layout is a property of the kernel, not of the model that
/// happens to dispatch it. `rms_norm.metal` and `row_gather.metal` are shared
/// sources, so their host mirrors belong here once rather than once per family
/// — three private copies of the same struct drift silently, and a mismatch is
/// not a compile error: the GPU reads whatever bytes sit at the offset.

#include <cstdint>
#include <cstdlib>

#include "device_tuning.hpp"
#include <cstring>
#include <stdexcept>

#include "../mtl4_context.hpp"
#include "pie/kernels/moe.h"

namespace pie::metal::shared_kernels {

/// The launch shapes that went down to `kernels-metal`, named here so the
/// family headers that already spell `shared_kernels::router_topk_dispatch`
/// keep reading. What is re-exported is the KERNELS' knowledge; what is
/// defined below is the DRIVER's.
using pie::kernels::moe::bm_slot;
using pie::kernels::moe::elementwise_dispatch;
using pie::kernels::moe::expert_combine_dispatch;
using pie::kernels::moe::kRouterMaxExperts;
using pie::kernels::moe::kRouterMaxTopK;
using pie::kernels::moe::route_rows_dispatch;
using pie::kernels::moe::route_sort_dispatch;
using pie::kernels::moe::routed_qmv_dispatch;
using pie::kernels::moe::router_lane_width;
using pie::kernels::moe::router_topk_dispatch;

/// The three widths a routed GEMM is compiled for, narrow first.
inline constexpr const int* kMoeTileWidths = pie::kernels::moe::kTileWidths;

/// The narrow tile, and the one the batching threshold is written against.
constexpr int kMoeTileRows = pie::kernels::moe::kTileRowsNarrow;

/// Params structs, replicated EXACTLY from the .metal sources.
///
/// Included, not retyped: each of these headers is the one the shader itself
/// includes, so the two sides cannot drift. The `static_assert`s are what
/// catches a change to a header that the host's binding order has not
/// followed.
#include "norm/rms_params.h"
static_assert(sizeof(RmsParams) == 20);
static_assert(sizeof(VNormParams) == 8);
static_assert(sizeof(GatedRmsParams) == 8);
#include "layout/row_gather_params.h"
static_assert(sizeof(RowGatherParams) == 8);
#include "moe/params.h"
static_assert(sizeof(RouterParams) == 16);
static_assert(sizeof(ExpertCombineParams) == 12);
static_assert(sizeof(MoeRouteParams) == 28);

/// When sorting the rows by expert pays for itself.
///
/// The sort turns `n` matvecs into `ceil(count_e / tile)` summed over the
/// experts -- fewer reads of each expert's weights, but a tile that is only
/// part full does the arithmetic of a whole one. The two meet when an expert's
/// run half fills a tile, which for `min(n, n_experts)` touched experts is
/// `n >= n_experts * tile / 2`.
///
/// Written against the NARROW tile, because that is the cheapest way in: a
/// batch that cannot pay for a 16-row tile cannot pay for a wider one either,
/// and `moe_tile_rows` widens only after this has said yes.
///
/// Below that the matvec wins outright, and it is not close: a decode routes
/// eight pairs over a hundred and twenty-eight experts, where every tile would
/// be one live row in sixteen.
inline bool moe_should_batch(int n_pairs, int n_experts) {
    return n_experts > 0 && n_pairs >= n_experts * moe_batch_min_per_expert();
}

/// Rows each expert's run is padded to, for a batch of `n_pairs`.
///
/// A wider tile does the arithmetic faster and rounds each expert's run up
/// further. What it does NOT cost is the allocation's worst case:
/// `moe_sorted_rows` is deliberately pessimistic and the tiles past the routing
/// decline at `tile_expert < 0`, so a wider tile dispatches more threadgroups
/// that do nothing rather than more arithmetic. An earlier rule here priced
/// that worst case as work, which is why it refused BM=64 everywhere -- at
/// gpt-oss's 448 rows all three widths do the same 2048 rows of work.
///
/// Priced instead off ROWS PER EXPERT, because that is what decides how much of
/// a tile a run fills, and measured end to end rather than modelled -- a model
/// built on the probe's rates picked 64 at 448 rows where the machine prefers
/// 32. The thresholds themselves live in `DeviceTuning::moe_tile_mid_per` and
/// `moe_tile_wide_per`, which carries the current sweep; what belongs here is
/// why the sweep has to be redone whenever the routed GEMM changes.
///
/// It has been redone once already. The first sweep put the crossovers at 12
/// and 88 with the routed GEMM emulating a bfloat matrix unit; on the FP16
/// instruction the same machine wants 32 and never, because making a tile's
/// arithmetic 40% cheaper does not make the rows it pads any cheaper. The
/// padding is the whole trade and the trade moved.
///
/// A width past 64 was tried and is not compiled. `roofline_probe` puts the
/// MXFP4 kernel at 4504 GFLOP/s at BM=64 and 5057 at BM=128, and in a real
/// mixture BM=128 is SLOWER: 558.5 -> 545.5 tok/s at 1024 rows, where 128 rows
/// an expert makes it exactly one tile with no padding to blame. That is the
/// second time the probe has over-promised here -- it also preferred 64 at 448
/// rows where the machine wanted 32 -- and the reason is the same both times.
/// The probe reads ONE expert with a hot cache; a mixture's threadgroups read
/// thirty-two and would rather be many and small than few and large. Which is
/// why the thresholds are a table of measurements and not a curve.
///
/// The other shape of the same idea was tried too and is not here either:
/// swapping the routed grid's axes so that row tiles run in x, which makes
/// consecutive threadgroups consecutive tiles of the SAME expert reading the
/// SAME weight slice where before they were `out_vec/bn` apart and shared
/// nothing. Correct -- the answers do not move -- and 558.6 -> 557.5 tok/s. So
/// the reuse is real on paper and the machine does not pay for it either way.
inline int moe_tile_rows(int n_pairs, int n_experts) {
    if (!moe_should_batch(n_pairs, n_experts)) return 1;
    const int per = n_pairs / n_experts;
    if (per >= moe_tile_wide_per()) return 64;
    return per >= moe_tile_mid_per() ? 32 : 16;
}


/// How many sorted rows a batch can produce.
///
/// The bound is the kernel's (`pie::kernels::moe::sorted_rows`); which tile the
/// batch gets is this driver's, because `moe_tile_rows` reads `DeviceTuning`.
/// So the tile is computed here and PASSED DOWN rather than read from below --
/// see the note at the top of `pie/kernels/moe.h`.
inline int moe_sorted_rows(int n_pairs, int n_experts) {
    const int n = n_pairs > 0 ? n_pairs : 0;
    return pie::kernels::moe::sorted_rows(n, n_experts, moe_tile_rows(n, n_experts));
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
