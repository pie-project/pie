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
///   * the routed FFN, built from the generic `moe_route.metal` kernels and
///     the unbiased routed matvec (`qmv_gptoss_impl` with BIASED off).
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

// Routing kernels and launch shapes are shared across every MoE family.
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
    /// Attention at the geometry's OWN head width, decode and paged.
    ///
    /// These used to be named `_d128` and compiled from a literal 128, which is
    /// the width llama, mistral, qwen2, qwen3 and the Qwen MoEs all happen to
    /// use. Llama-3.2-1B is 32 heads of 64, and a d=128 pipeline handed 64-wide
    /// heads does not fail -- it strides past the end of every head and writes
    /// zeros. The width is now spelled from `g.head_dim`, so a checkpoint whose
    /// width has no instantiation fails to COMPILE a pipeline, by name, at load.
    Pso sdpa{};
    Pso sdpa_paged{};
    Pso sdpa_paged_sg8{};
    /// The same paged attention with the query rows tiled -- one row per
    /// simdgroup, K/V staged per threadgroup. Chosen by row count, not by
    /// model: see `sdpa_should_tile`.
    Pso sdpa_paged_tiled{};
    Pso row_gather{};
    /// RoPE from a supplied frequency table, decode and batched. Compiled only
    /// when `g.rope_freq_table` -- a checkpoint whose frequencies really are a
    /// geometric series runs the base form, and compiling a kernel it never
    /// dispatches would let an unrelated shader error fail a load that works.
    Pso rope_freqs{};
    Pso rope_freqs_mb{};

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
        return sdpa.valid() && sdpa_paged.valid() &&
               sdpa_paged_tiled.valid() && row_gather.valid();
    }
    bool rope_table_valid() const {
        return rope_freqs.valid() && rope_freqs_mb.valid();
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
        return dense_valid() && (!g.is_moe() || moe_valid()) &&
               (!g.rope_freq_table || rope_table_valid());
    }
};

/// Compile them. `g` decides whether the routed set is requested at all.
bool build_llama_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                      const LlamaGeometry& g, LlamaPsos& out, std::string* err);

/// Llama 3.1's rotary frequencies, ported from
/// `mlx_lm/models/rope_utils.py::Llama3RoPE`.
///
/// A rotary dimension's WAVELENGTH decides what happens to it. Dimensions
/// whose wavelength is longer than the original context could hold (they turn
/// less than `low_freq_factor` times in it) are interpolated by the full
/// factor; those short enough to turn more than `high_freq_factor` times are
/// left alone, because extrapolating them is safe; between the two the
/// schedule ramps smoothly. That is a closed form over `rotary_dims/2` values
/// with no dependence on position, so the host computes it once at setup
/// rather than every head recomputing it every token -- which is exactly what
/// `rope_neox_freqs_*` was built to consume.
///
/// Returns `inv_freq`, the RECIPROCAL of mlx's `_freqs`: `mx.fast.rope` divides
/// the position by its table and this kernel multiplies by its own.
std::vector<float> llama3_inv_freq(const LlamaGeometry& g);

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
