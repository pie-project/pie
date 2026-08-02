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
#include "geometry.hpp"

namespace pie::metal::gptoss {

/// Params structs, replicated EXACTLY from the .metal sources. A mismatch here
/// is silent: the GPU reads whatever bytes are at the offset.
struct RowGatherParams {   // row_gather.metal (buffer 3)
    std::uint32_t width;
    std::uint32_t count;
};
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
    /// goes to 4, but a uniformly-quantized checkpoint ships a 4-bit one --
    /// `build_gptoss_psos` picks between `qmv_u8_bias` and `qmv_tail_bias` from
    /// the width the staged tensors state.
    Pso qmv_router{};
    /// The 8-bit router matvec; mlx's predicate keeps it wider than everything
    /// else. Held separately from `qmv_router` so the choice stays inspectable.
    Pso qmv_u8_bias{};
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
               qmv_u8_bias.valid() && qmv_router.valid() && router_topk.valid() &&
               swiglu.valid() && expert_combine.valid() && sdpa_sink.valid() &&
               rope_freqs.valid();
    }
};

/// Compile them. `err` names the first one that failed, so a missing kernel is
/// reported as itself rather than as a generic setup failure.
///
/// `router_bits` selects the router's matvec: 8 for the width `mlx_lm`'s
/// quantization predicate usually leaves it at, 4 for a uniformly-quantized
/// checkpoint. It is refused rather than defaulted, because either kernel over
/// the other's packing produces fluent wrong text instead of an error.
bool build_gptoss_psos(RawMetalContext& ctx, const std::string& kernels_dir, int router_bits,
                       GptOssPsos& out, std::string* err);

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

/// Flat elementwise kernels: one thread per element.
inline void elementwise_dispatch(int n, Grid& g, Threadgroup& tg) {
    const int width = n < 256 ? (n > 0 ? n : 1) : 256;
    g = Grid{std::uint32_t(n > 0 ? n : 1), 1, 1};
    tg = Threadgroup{std::uint32_t(width), 1, 1};
}

/// `router_topk`: one threadgroup, one lane per expert. 32 on this family, so a
/// single simdgroup holds the whole distribution.
inline void router_topk_dispatch(int n_experts, Grid& g, Threadgroup& tg, int rows = 1) {
    const std::uint32_t w = std::uint32_t(n_experts < 32 ? 32 : n_experts);
    g = Grid{w, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{w, 1, 1};
}

/// `affine_qmv_u8_bias`: one simdgroup per output row.
inline void qmv_u8_dispatch(int N, Grid& g, Threadgroup& tg, int rows = 1) {
    g = Grid{32u, std::uint32_t(N > 0 ? N : 1), std::uint32_t(rows < 1 ? 1 : rows)};
    tg = Threadgroup{32, 1, 1};
}

/// The 4-bit matvec. `slots` is 1 for an ordinary projection and
/// `experts_per_token` for a routed one, on the third grid axis.
/// `rows` folds onto the x axis, which the kernel already reads as the token
/// row: the grid is in THREADS, so one threadgroup of 32 per row.
inline void qmv_dispatch(int N, int slots, Grid& g, Threadgroup& tg, int rows = 1) {
    g = Grid{32u * std::uint32_t(rows < 1 ? 1 : rows), std::uint32_t(N) / 4,
             std::uint32_t(slots < 1 ? 1 : slots)};
    tg = Threadgroup{32, 2, 1};
}

/// `sdpa_vector_decode_sink`: one threadgroup of 1024 per query head. The grid
/// is in THREADS, so the head count multiplies the threadgroup width.
inline void sdpa_sink_dispatch(int n_q_heads, Grid& g, Threadgroup& tg, int rows = 1) {
    g = Grid{std::uint32_t(n_q_heads) * 1024, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{1024, 1, 1};
}

}  // namespace pie::metal::gptoss
