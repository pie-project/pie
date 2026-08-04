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
#include "../shared_kernels.hpp"
#include "geometry.hpp"

namespace pie::metal::gemma4 {

using shared_kernels::RowGatherParams;
using shared_kernels::elementwise_dispatch;

/// Params structs, replicated EXACTLY from the .metal sources. A mismatch here
/// is silent: the GPU reads whatever bytes are at the offset.
struct GegluParams {        // geglu_tanh.metal:15   (buffer 3)
    std::uint32_t n;
};
struct GegluStridedParams { // geglu_tanh.metal       (buffer 3)
    std::uint32_t width;
    std::uint32_t rows;
    std::uint32_t gate_pitch;
    std::uint32_t up_pitch;
    std::uint32_t out_pitch;
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
    /// The sampled rows, compacted before the tail.
    Pso row_gather{};
    /// gemma4 scales the gathered embedding (by sqrt(hidden), and sqrt(ple_dim)
    /// for the per-layer table). The scale must NOT be folded into the weights:
    /// the LM head reads the very same tied table unscaled.
    Pso embed_scaled{};
    /// The bounds-checked matvec, for every projection whose K is not a whole
    /// number of the aligned kernel's reduction blocks. gemma 4 has three such
    /// shapes and no other family has any: `per_layer_projection` is K=256,
    /// and the 26B's hidden 2816 and intermediate 2112 are 5.5 and 8.25 blocks
    /// at the widths their tensors are stored in. The aligned kernel reduces a
    /// whole block past the end of every row, which is a NaN at the end of a
    /// tensor and a wrong number everywhere else.
    Pso qmv_tail{};
    /// The same kernel at the checkpoint's SECOND affine width, for the dense
    /// FFN and router a mixed-precision checkpoint spares at 8 bits. Invalid,
    /// and never selected, when there is only one width.
    Pso qmv_tail_alt{};
    /// Partial rotary over the whole head, which is what gemma4's full-attention
    /// layers mean by "a quarter of it". See rope.metal.
    Pso rope_prop{};
    /// The M>1 counterparts. The prefill path shares this family's semantics,
    /// not qwen3.5's, so it needs its own pipelines for exactly the two kernels
    /// whose MEANING differs rather than only their shape.
    Pso embed_scaled_mb{};
    Pso rope_prop_mb{};
    /// The norm sandwich's closing norm and the residual add it always precedes,
    /// in one dispatch. Three a layer, so 105 of the step's barriers.
    Pso rms_residual{};
    Pso rms_residual_scaled{};
    /// The PLE gate at M>1, whose `up` operand is one layer's slice of a table
    /// that is `n_layers * ple_dim` wide per row.
    Pso geglu_strided{};

    // ── the mixture (gemma-4-26B-A4B), built only for a model that has one ──
    /// gemma's router: the shared top-k, plus the learned `per_expert_scale`
    /// the softmax is multiplied by. A separate instantiation of the same
    /// kernel rather than an optional buffer -- an unbound buffer is not a null
    /// pointer here, so llama and gpt-oss would have to bind a tensor they do
    /// not have.
    Pso router_topk{};
    Pso qmv_routed{};
    /// The batched form's three column tiles.
    Pso qmm_routed[3]{};
    Pso moe_sort{};
    Pso moe_gather{};
    Pso moe_combine{};
    /// `h1 + h2`. The plain elementwise add -- gemma's dense path never needs
    /// one, because every add it does is fused into a norm.
    Pso residual_add{};

    bool moe_valid() const {
        return router_topk.valid() && qmv_routed.valid() && moe_sort.valid() &&
               moe_gather.valid() && moe_combine.valid() && residual_add.valid() &&
               qmm_routed[0].valid() && qmm_routed[1].valid() && qmm_routed[2].valid();
    }

    bool valid() const {
        return sdpa_swa_d256.valid() && sdpa_swa_d512.valid() && geglu_tanh.valid() &&
               logit_softcap.valid() && layer_scalar.valid() && ple_combine.valid() &&
               vnorm.valid() && embed_scaled.valid() && qmv_tail.valid() &&
               rope_prop.valid() && embed_scaled_mb.valid() && rope_prop_mb.valid() &&
               rms_residual.valid() && rms_residual_scaled.valid() && geglu_strided.valid();
    }
};

/// Compile them. `err` names the first one that failed, so a missing kernel is
/// reported as itself rather than as a generic setup failure.
///
/// The two attention widths come from the geometry, not from literals. This
/// family is the one where that matters most: it has TWO of them --
/// `head_dim` for the sliding layers and `global_head_dim` for the full ones --
/// so a checkpoint that moved either would have run one width's pipeline over
/// the other's heads, which strides past the end of each head and writes zeros
/// rather than failing.
bool build_gemma4_psos(RawMetalContext& ctx, const std::string& kernels_dir,
                       const Gemma4Geometry& g, Gemma4Psos& out, std::string* err);

// ── Launch geometry ─────────────────────────────────────────────────────────

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
