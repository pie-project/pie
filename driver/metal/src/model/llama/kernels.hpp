#pragma once

/// The llama families' kernel PSOs and launch geometry.
///
/// This file is short, and that is the finding rather than an omission. The
/// shared decode table in `kernels/decode_psos.cpp` already carries every
/// dispatch a dense llama layer makes -- embedding, rms, the affine matvec,
/// rope, kv-append, sdpa, silu-mul, residual-add, argmax -- because the shared
/// `Kernel` enum's common prefix IS a llama decoder. Adding the family needed
/// no new dense kernel at all.
///
/// What it did need is three things the shared table has no reason to carry:
///
///   * a 128-wide attention head. `sdpa_vector` and `sdpa_paged` are templated
///     on D and were instantiated at 256 (qwen3.5), 512 (gemma4 full) and 64
///     (gpt-oss). Llama and every Qwen use 128, so that is one instantiation
///     each, not a kernel.
///   * `row_gather`, for the prefill tail.
///   * the routed FFN, which is entirely borrowed: `router_topk` and
///     `expert_combine` from `gptoss.metal` -- both generic, neither
///     gpt-oss-specific -- and the unbiased routed matvec, which is
///     `qmv_gptoss_impl` with BIASED off.
///
/// Notably NOT borrowed: `gptoss_swiglu`. That kernel bakes in gpt-oss's
/// asymmetric clamp, its `alpha`, and its `(up + 1)` term. Qwen3-MoE uses plain
/// SwiGLU, so the routed path uses the ordinary `silu_mul` over the whole
/// [rows, k, width] expert stack.

#include <cstdint>
#include <string>

#include "../../batch/decode_abi.hpp"
#include "../../mtl4_context.hpp"
#include "../shared_kernels.hpp"
#include "geometry.hpp"

namespace pie::metal::llama {

// The routing kernels and their launch shapes are shared: `gptoss.metal` is
// where they live, but nothing in `router_topk` or `expert_combine` is
// gpt-oss-specific, and this family dispatches both.
using shared_kernels::ExpertCombineParams;
using shared_kernels::MoeRouteParams;
using shared_kernels::RouterParams;
using shared_kernels::RowGatherParams;
using shared_kernels::elementwise_dispatch;
using shared_kernels::expert_combine_dispatch;
using shared_kernels::moe_route_rows_dispatch;
using shared_kernels::moe_route_sort_dispatch;
using shared_kernels::routed_qmv_dispatch;
using shared_kernels::router_topk_dispatch;

/// The PSOs this family needs beyond the shared decode set.
struct LlamaPsos {
    /// 128-wide heads, decode and paged.
    Pso sdpa_d128{};
    Pso sdpa_paged_d128{};
    /// The same paged attention with the query rows tiled -- one row per
    /// simdgroup, K/V staged per threadgroup. Chosen by row count, not by
    /// model: see `sdpa_should_tile`.
    Pso sdpa_paged_tiled_d128{};
    Pso row_gather{};

    // Routed FFN. Left invalid on a dense checkpoint -- see `valid()`.
    Pso router_topk{};
    /// The routed matvec. Unbiased: Qwen's experts carry no bias, unlike
    /// gpt-oss's.
    Pso qmv_routed{};
    /// The expert-major reordering the batched form runs on.
    Pso moe_sort{};
    Pso moe_gather{};
    Pso moe_combine{};
    /// The routed matmul, one column tile per entry: bn 16, 32, 64. The row
    /// tile does not vary -- `kMoeTileRows` is what the sort padded to, and a
    /// second row tile would be a second thing for the sort to agree with.
    Pso qmm_routed[3]{};

    bool dense_valid() const {
        return sdpa_d128.valid() && sdpa_paged_d128.valid() &&
               sdpa_paged_tiled_d128.valid() && row_gather.valid();
    }
    bool moe_valid() const {
        return router_topk.valid() && qmv_routed.valid() &&
               moe_sort.valid() && moe_gather.valid() && moe_combine.valid() &&
               qmm_routed[0].valid() && qmm_routed[1].valid() && qmm_routed[2].valid();
    }
    /// A dense checkpoint must not be held to the MoE PSOs: compiling them for
    /// a model that never dispatches them would make an unrelated shader error
    /// fail a load that would otherwise work.
    bool valid_for(const LlamaGeometry& g) const {
        return dense_valid() && (!g.is_moe() || moe_valid());
    }
};

/// Compile them. `g` decides whether the routed set is requested at all.
bool build_llama_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                      const LlamaGeometry& g, LlamaPsos& out, std::string* err);

// ── Launch geometry ─────────────────────────────────────────────────────────

/// The routed SiLU-mul, over the whole [rows * k, moe_intermediate] stack. It
/// is one flat elementwise dispatch precisely because gate, up and out share a
/// layout -- the slot axis needs no special handling.
inline void expert_silu_dispatch(int moe_intermediate, int experts_per_token, Grid& g,
                                 Threadgroup& tg, int rows = 1) {
    const std::size_t n = std::size_t(moe_intermediate > 0 ? moe_intermediate : 1) *
                          std::size_t(experts_per_token > 0 ? experts_per_token : 1) *
                          std::size_t(rows > 0 ? rows : 1);
    elementwise_dispatch(static_cast<std::uint32_t>(n), g, tg);
}

}  // namespace pie::metal::llama
