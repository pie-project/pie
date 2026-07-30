#pragma once

/// Gemma 4's kernel PSOs and launch geometry.
///
/// Six `.metal` files shipped with this family's bring-up and had no PSO entry,
/// no `bind::` enum and no host-side params mirror — they compiled and nothing
/// could reach them. This is the other half.
///
/// The launch shapes come from the sources, not from qwen3.5's equivalents:
/// `vnorm_single_row` is one threadgroup per row like `rms_single_row`, but
/// `geglu_tanh` / `logit_softcap` / `layer_scalar_mul` / `ple_combine` are flat
/// elementwise `dispatchThreads` grids, which is why they take a plain `n`.

#include <cstdint>
#include <string>

#include "../../batch/decode_abi.hpp"
#include "../../mtl4_context.hpp"

namespace pie::metal::gemma4 {

/// Params structs, replicated EXACTLY from the .metal sources. A mismatch here
/// is silent: the GPU reads whatever bytes are at the offset.
struct GegluParams {        // geglu_tanh.metal:15   (buffer 3)
    std::uint32_t n;
};
struct SoftcapParams {      // logit_softcap.metal:13 (buffer 2)
    float cap;
    std::uint32_t n;
};
struct LayerScalarParams {  // layer_scalar.metal:14  (buffer 3)
    std::uint32_t n;
};
struct PleCombineParams {   // ple_combine.metal:15   (buffer 3)
    float inv_sqrt2;
    std::uint32_t n;
};
struct VNormParams {        // vnorm.metal:14         (buffer 2)
    float eps;
    std::uint32_t axis_size;
};

/// The PSOs this family needs beyond the shared set.
struct Gemma4Psos {
    // Two head widths, because head_dim is per attention type: sliding layers
    // are 256 and full layers 512. Both are instantiated in the source.
    Pso sdpa_swa_d256{};
    Pso sdpa_swa_d512{};
    Pso geglu_tanh{};
    Pso logit_softcap{};
    Pso layer_scalar{};
    Pso ple_combine{};
    Pso vnorm{};
    /// gemma4 scales the gathered embedding (by sqrt(hidden), and sqrt(ple_dim)
    /// for the per-layer table). The scale must NOT be folded into the weights:
    /// the LM head reads the very same tied table unscaled.
    Pso embed_scaled{};
    /// `per_layer_projection` is K=256, the one projection whose K is not a
    /// multiple of `affine_qmv_fast`'s 512-wide reduction block.
    Pso qmv_narrow{};
    /// Partial rotary over the whole head, which is what gemma4's full-attention
    /// layers mean by "a quarter of it". See rope.metal.
    Pso rope_prop{};
    /// The M>1 counterparts. The prefill path shares this family's semantics,
    /// not qwen3.5's, so it needs its own pipelines for exactly the two kernels
    /// whose MEANING differs rather than only their shape.
    Pso embed_scaled_mb{};
    Pso rope_prop_mb{};

    bool valid() const {
        return sdpa_swa_d256.valid() && sdpa_swa_d512.valid() && geglu_tanh.valid() &&
               logit_softcap.valid() && layer_scalar.valid() && ple_combine.valid() &&
               vnorm.valid() && embed_scaled.valid() && qmv_narrow.valid() &&
               rope_prop.valid() && embed_scaled_mb.valid() && rope_prop_mb.valid();
    }
};

/// Compile them. `err` names the first one that failed, so a missing kernel is
/// reported as itself rather than as a generic setup failure.
bool build_gemma4_psos(RawMetalContext& ctx, const std::string& kernels_dir, Gemma4Psos& out,
                       std::string* err);

// ── Launch geometry ─────────────────────────────────────────────────────────

/// Flat elementwise kernels: one thread per element, `dispatchThreads` style.
inline void elementwise_dispatch(int n, Grid& g, Threadgroup& tg) {
    const int width = n < 256 ? (n > 0 ? n : 1) : 256;
    g = Grid{std::uint32_t(n > 0 ? n : 1), 1, 1};
    tg = Threadgroup{std::uint32_t(width), 1, 1};
}

/// `vnorm_single_row`: one threadgroup per row, the row's width in threads,
/// four elements each — the same shape `rms_single_row` uses.
inline void vnorm_dispatch(int rows, int axis, Grid& g, Threadgroup& tg) {
    constexpr int kNReads = 4;
    const int threads = (axis + kNReads - 1) / kNReads;
    g = Grid{std::uint32_t(threads) * std::uint32_t(rows > 0 ? rows : 1), 1, 1};
    tg = Threadgroup{std::uint32_t(threads), 1, 1};
}

/// Sliding-window decode attention: one threadgroup per query head, as
/// `sdpa_vector` does, with BN=32/BD=32 inside.
/// One threadgroup of 1024 threads per (head, query row).
///
/// The grid is in THREADS -- `StepEncoder::dispatch` calls `dispatchThreads` --
/// so the head count multiplies the threadgroup width rather than standing alone.
/// Writing it the other way launches `n_q_heads` threads TOTAL, which is not an
/// error the hardware reports: the kernel's simd reductions just read lanes that
/// were never dispatched.
inline void sdpa_sliding_dispatch(int n_q_heads, Grid& g, Threadgroup& tg, int rows = 1) {
    g = Grid{std::uint32_t(n_q_heads) * 1024, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{1024, 1, 1};
}

}  // namespace pie::metal::gemma4
