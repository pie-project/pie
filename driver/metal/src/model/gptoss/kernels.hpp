#pragma once

/// GPT-OSS's kernel PSOs, params mirrors and launch geometry.
///
/// The launch shapes come from the sources' index contracts, not from another
/// family's equivalents. Two are worth stating up front:
///
///  * The matvec is `qmv_dispatch`'s shape -- grid `{32, N/4, 1}`, tg
///    `{32,2,1}` -- with a THIRD grid axis for the routed variants, one plane
///    per selected expert. gemma4's bring-up wrote its own matvec shape and got
///    a grid whose x was one thread wide, so half of every output was never
///    written; this reuses the shared helper for exactly that reason.
///  * The router's 8-bit matvec is one simdgroup per output row.

#include <cstdint>
#include <string>
#include <vector>

#include "../../batch/decode_abi.hpp"
#include "../../mtl4_context.hpp"
#include "../shared_kernels.hpp"
#include "geometry.hpp"

namespace pie::metal::gptoss {

// Shared with the llama family's MoE path: the shape is a property of
// `router_topk`, not of gpt-oss.
using shared_kernels::router_topk_dispatch;

using shared_kernels::RowGatherParams;
using shared_kernels::elementwise_dispatch;

/// Params structs, replicated EXACTLY from the .metal sources. A mismatch here
/// is silent: the GPU reads whatever bytes are at the offset.
struct RouterParams {          // gptoss.metal
    std::uint32_t n_experts;
    std::uint32_t experts_per_token;
};
struct SwiGluParams {          // gptoss.metal
    std::uint32_t n;
    float limit;
    float alpha;
};
struct ExpertCombineParams {   // gptoss.metal
    std::uint32_t width;
    std::uint32_t experts_per_token;
};

/// The PSOs this family needs beyond the shared set.
struct GptOssPsos {
    /// K=2880 is a whole number of quantization groups but not of any reduction
    /// block, so every projection here runs the tail-handling matvec.
    Pso qmv_tail{};
    Pso qmv_tail_bias{};
    Pso qmv_routed_bias{};
    /// The router's matvec, at whatever width the checkpoint quantized it to.
    /// `mlx_lm`'s predicate usually keeps it at 8 bits while everything else
    /// goes to 4, but a uniformly-quantized checkpoint ships a 4-bit one. Same
    /// kernel either way -- `build_gptoss_psos` names the b=4 or b=8
    /// instantiation of the dense biased matvec -- so it needs no launch
    /// geometry of its own.
    Pso qmv_router{};
    Pso router_topk{};
    Pso swiglu{};
    Pso expert_combine{};
    /// head_dim 64, with the per-head sink in the softmax denominator.
    Pso sdpa_sink{};
    /// The same attention against page-addressed KV, which is what lets several
    /// sequences be resident at once.
    Pso sdpa_sink_paged{};
    /// YaRN, as a frequency table the host computed once.
    Pso rope_freqs{};
    /// The M>1 counterparts: the two kernels whose ROW indexing differs, rather
    /// than only their launch width.
    Pso rope_freqs_mb{};
    /// The sampled rows, compacted before the tail. Family-agnostic: the same
    /// kernel gemma4 uses.
    Pso row_gather{};

    bool valid() const {
        return qmv_tail.valid() && qmv_tail_bias.valid() && qmv_routed_bias.valid() &&
               qmv_router.valid() && router_topk.valid() &&
               swiglu.valid() && expert_combine.valid() && sdpa_sink.valid() &&
               rope_freqs.valid();
    }
};

/// Compile them. `err` names the first one that failed, so a missing kernel is
/// reported as itself rather than as a generic setup failure.
///
/// The geometry decides two things here, and both are refused rather than
/// defaulted because either wrong answer runs.
///
///  * `router_bits` selects the router's matvec: 8 for the width `mlx_lm`'s
///    quantization predicate usually leaves it at, 4 for a uniformly-quantized
///    checkpoint. Either kernel over the other's packing produces fluent wrong
///    text instead of an error.
///  * `head_dim` names the attention instantiation. This used to be the literal
///    64 that every released gpt-oss happens to use, while the geometry read the
///    width from the config -- so a variant that shipped any other width would
///    have run a d=64 pipeline over its heads, striding past the end of each one
///    and writing zeros. Spelled from the geometry, an uninstantiated width
///    fails to build a pipeline BY NAME at load.
bool build_gptoss_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                       const GptOssGeometry& g, GptOssPsos& out, std::string* err);

/// The YaRN frequency table, `head_dim/2` entries.
///
/// A closed form over the dimension that does not depend on the position, so the
/// host computes it once and the kernel reads it. Mirrors mlx_lm's `YarnRoPE`:
/// interpolate between the extended and original frequency by a ramp between the
/// dimensions where the original context held one and `beta_fast` full rotations.
std::vector<float> yarn_inv_freq(const GptOssGeometry& g);

/// YaRN's `mscale`: the attention-temperature correction that scales q and k.
float yarn_mscale(const GptOssGeometry& g);

// ── Launch geometry ─────────────────────────────────────────────────────────

/// The quantized matvec, at either width. `slots` is 1 for an ordinary projection and
/// `experts_per_token` for a routed one, on the third grid axis.
/// `rows` folds onto the x axis, which the kernel already reads as the token
/// row: the grid is in THREADS, so one threadgroup of 32 per row.
inline void qmv_dispatch(int N, int slots, Grid& g, Threadgroup& tg, int rows = 1) {
    shared_kernels::routed_qmv_dispatch(N, slots, g, tg, rows);
}

/// `sdpa_vector_decode_sink`: one threadgroup of 1024 per query head. The grid
/// is in THREADS, so the head count multiplies the threadgroup width.
inline void sdpa_sink_dispatch(int n_q_heads, Grid& g, Threadgroup& tg, int rows = 1) {
    g = Grid{std::uint32_t(n_q_heads) * 1024, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{1024, 1, 1};
}

}  // namespace pie::metal::gptoss
