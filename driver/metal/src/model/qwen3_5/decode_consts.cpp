// decode_consts.cpp — bind all geometry-derived const params, by ordinal, over beta's DAG.

#include "decode_consts.hpp"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "decode_step.hpp"     // beta: Dispatch{kind,ordinal,layer,grid,tg}
#include "mtl4_context.hpp"
#include "../../kernels/gdn_params.h"
#include "../shared_kernels.hpp"

namespace pie::metal {

namespace {

// ── Kernel param structs, replicated EXACTLY from the .metal sources ──
using shared_kernels::ExpertCombineParams;
using shared_kernels::GatedRmsParams;
using shared_kernels::MoeRouteParams;
using shared_kernels::RmsParams;
using shared_kernels::RouterParams;
static_assert(sizeof(GdnCoreParams) == 44);

GdnCoreParams gdn_core_params(const DecodeGeometry& g) {
    return {
        g.gdn_k_dim,
        g.gdn_v_dim,
        g.gdn_k_heads,
        g.gdn_v_heads,
        g.gdn_conv_dim,
        g.gdn_conv_k,
        0,
        g.gdn_k_heads * g.gdn_k_dim,
        2 * g.gdn_k_heads * g.gdn_k_dim,
        g.eps,
        1.0f / std::sqrt(float(g.gdn_k_dim)),
    };
}

// Bind a POD constant value into a fresh resident slot at (ordinal, bind_index).
template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, uint8_t idx, const V& val, int* count) {
    shared_kernels::bind_const(ctx, ord, idx, val, count, "decode");
}


/// Whether a kind is a matvec, asked of THIS model's geometry.
///
/// It used to ask a default-constructed one, which worked only because every
/// dense projection's width is nonzero whatever the numbers are. A routed
/// projection's is not: `moe_intermediate` and `n_experts` are zero in a
/// default geometry, so the mixture's matvecs would answer "not a matvec",
/// skip the K/N binding entirely, and run against whatever the pool held at
/// those ordinals.
bool is_qmv(Kernel k, const DecodeGeometry& g) { return qmv_kn(k, g).N != 0; }

/// The three projections that read one expert's slice per row.
bool is_routed(Kernel k) {
    return k == Kernel::LlExpertGate || k == Kernel::LlExpertUp || k == Kernel::LlExpertDown;
}

}  // namespace

// qmv in_vec (K) / out_vec (N) per kind, from geometry (matches the staged weight shapes).
KN qmv_kn(Kernel k, const DecodeGeometry& g) {
    const int H = g.hidden;
    const int q_wide = 2 * g.n_q_heads * g.head_dim;   // 2×-wide gated q_proj (4096)
    const int kv_dim = g.n_kv_heads * g.head_dim;      // 512
    const int q_dim  = g.n_q_heads * g.head_dim;       // 2048
    switch (k) {
        case Kernel::QmvIn:     return {H, g.gdn_conv_dim};   // 1024 → 6144
        case Kernel::QmvInZ:    return {H, g.gdn_v_total};    // 1024 → 2048
        // The GDN decay and beta projections. These were bound as DENSE bf16
        // matrices, which is what a Qwen3-Next preview repack shipped and what
        // no released checkpoint ships: both Qwen3.5-0.8B and Qwen3.5-35B-A3B
        // quantize them like every other projection in the layer, so reading
        // `in_proj_a.weight` as bf16 read packed 4-bit nibbles as floats. It
        // was NaN in the first four output channels and small, plausible,
        // wrong numbers in the other twelve -- the shape was right, the
        // dispatch succeeded, and the model produced token 0 forever.
        case Kernel::GdnInA:
        case Kernel::GdnInB:    return {H, g.gdn_v_heads};     // 1024 → 16
        case Kernel::QmvOut:    return {g.gdn_v_total, H};    // 2048 → 1024
        case Kernel::QmvQ:      return {H, q_wide};           // 1024 → 4096
        case Kernel::QmvK:      return {H, kv_dim};           // 1024 → 512
        case Kernel::QmvV:      return {H, kv_dim};           // 1024 → 512
        case Kernel::QmvO:      return {q_dim, H};            // 2048 → 1024
        case Kernel::QmvGate:   return {H, g.intermediate};   // 1024 → 3584
        case Kernel::QmvUp:     return {H, g.intermediate};   // 1024 → 3584
        case Kernel::QmvDown:   return {g.intermediate, H};   // 3584 → 1024
        case Kernel::QmvLmHead:
        case Kernel::LmHeadUntied: return {H, g.vocab};       // 1024 → 248320
        // The mixture. The router is an ordinary matvec into one logit per
        // expert; the three expert projections have a K and an N like any
        // other, and what makes them routed is which slice of the weight a row
        // reads, not their shape.
        case Kernel::LlRouter:     return {H, g.n_experts};
        case Kernel::LlExpertGate: return {H, g.moe_intermediate};
        case Kernel::LlExpertUp:   return {H, g.moe_intermediate};
        case Kernel::LlExpertDown: return {g.moe_intermediate, H};
        // The shared expert: an ordinary dense FFN, so ordinary dense shapes.
        // The gate is the odd one -- `hidden -> 1`, one logit a token -- and it
        // is a matvec only in the sense that everything with a K and an N is.
        case Kernel::LlSharedGate: return {H, g.shared_intermediate};
        case Kernel::LlSharedUp:   return {H, g.shared_intermediate};
        case Kernel::LlSharedDown: return {g.shared_intermediate, H};
        case Kernel::LlSharedGateProj: return {H, 1};
        default:                return {0, 0};
    }
}


// ── the constants that are a function of the batch width ─────────────────────
//
// Every other constant this file binds is per-ROW -- a width, a stride, an
// epsilon -- so it is the same number whether the step carries one token or a
// thousand, which is why they are bound once at setup and never looked at
// again. The mixture's routing breaks that. `moe_route_sort` is told how many
// (token, slot) pairs exist and how many rows their tile-padded runs occupy,
// and both scale with the batch: bound at one width and fired at another, the
// sort either walks off the end of the routing or silently drops the pairs
// past the count it was given.
//
// So they are split out here rather than left in the walk below, and the fire
// path rebinds THIS when the token count changes. It is a separate function
// because the alternative -- re-walking all ~400 dispatches per step to rewrite
// the two dozen that could have changed -- pays the whole argument table for
// the mixture's share of it. `const_slot` caches by (ordinal, index), so this
// overwrites the same slots in place and allocates nothing after the first.
int bind_token_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                      const DecodeGeometry& g, int n_tokens, int row_pitch) {
    int count = 0;
    const int rows = n_tokens > 0 ? n_tokens : 1;
    const int pairs = rows * g.experts_per_token;
    const int sorted = moe_sorted_rows(g, rows);

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        switch (d.kind) {
            case Kernel::LlExpertSiluMul:
                break;

            case Kernel::LlMoeSort:
            case Kernel::LlMoeGather: {
                // One params struct for both, so the sort's padding and the
                // gather's bounds cannot disagree about how many rows exist.
                // `width` is read only by the gather, which moves hidden-wide
                // rows.
                const MoeRouteParams p{(uint32_t)pairs,
                                       (uint32_t)g.n_experts,
                                       (uint32_t)g.experts_per_token,
                                       (uint32_t)shared_kernels::moe_tile_rows(pairs, g.n_experts),
                                       (uint32_t)sorted,
                                       (uint32_t)g.hidden,
                                       (uint32_t)row_pitch};
                bind_const<MoeRouteParams>(ctx, ord,
                                           d.kind == Kernel::LlMoeSort
                                               ? (uint8_t)bind::MoeRouteSort::Params
                                               : (uint8_t)bind::MoeRouteRows::Params,
                                           p, &count);
                break;
            }

            default:
                break;
        }
    }
    return count;
}

int bind_decode_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const DecodeGeometry& g, int max_ctx, bool gdn_prep, int n_tokens,
                       int row_pitch) {
    // The width-dependent ones first, so a DAG is never left half-bound: this
    // function is the complete binding, and the fire path's rebind is a subset
    // of it rather than a second, separate contract.
    int count = bind_token_consts(ctx, dag, g, n_tokens, row_pitch);

    // rope: x[h*head_dim + i], rotary half from grid.x. scale=1.0 (qwen3.6 default mrope),
    // base = log2(theta).
    const float rope_scale = 1.0f;
    const float rope_base  = std::log2(g.rope_theta);

    // sdpa / kv_append cache layout [n_kv_heads, max_ctx, head_dim]:
    const size_t head_stride = size_t(max_ctx) * size_t(g.head_dim);  // *_head_stride
    const size_t seq_stride  = size_t(g.head_dim);                    // *_seq_stride
    const int    gqa_factor  = g.n_q_heads / g.n_kv_heads;            // 4
    const float  sdpa_scale  = 1.0f / std::sqrt(float(g.head_dim));   // 1/sqrt(256)

    for (const auto& d : dag) {
        const int ord = d.ordinal;
        const Kernel k = d.kind;

        if (is_qmv(k, g)) {
            const KN kn = qmv_kn(k, g);
            bind_const<int>(ctx, ord, (uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<int>(ctx, ord, (uint8_t)bind::Qmv::N, kn.N, &count);
            if (is_routed(k)) {
                using Q = bind::GoQmv;
                // The routed matvec reads the SORTED stack: one row per
                // (token, slot) pair rather than k slots hanging off a token
                // row. The sort is what made the pair axis disappear, so all
                // three of these collapse to the dense answer -- no slot
                // stride, a row pitch that is just the input width, and one
                // expert per row, named by `row_expert` rather than by `tid.z`.
                bind_const<int>(ctx, ord, (uint8_t)Q::XSlotStride, 0, &count);
                bind_const<int>(ctx, ord, (uint8_t)Q::XRowStride, kn.K, &count);
                bind_const<int>(ctx, ord, (uint8_t)Q::SlotsPerRow, 1, &count);
            }
            continue;
        }

        switch (k) {
            case Kernel::EmbedUntied:
            case Kernel::EmbedGather:
                bind_const<int>(ctx, ord, (uint8_t)bind::Embed::Hidden, g.hidden, &count);
                break;

            // RMSNorm variants — gain is the RAW weight. qwen3.5 is not Gemma:
            // its norm weights are absolute (input_layernorm averages 1.24 and
            // `model.norm` 4.31 on the 0.8B checkpoint), not the (w-1) offsets a
            // (1+w) gain expects, and mlx_lm builds every one of these as a plain
            // nn.RMSNorm. A (1+w) gain here is finite and quiet, so it survives as
            // a ~80% per-norm error that the residual stream compounds.
            case Kernel::Rms:
            case Kernel::FfnRms:
            case Kernel::FinalRms:
                bind_const<RmsParams>(ctx, ord, (uint8_t)bind::Rms::Params,
                                      RmsParams{g.eps, (uint32_t)g.hidden, 1u, 0u, 1.0f},
                                      &count);
                break;
            case Kernel::QNorm:
            case Kernel::KNorm:
                bind_const<RmsParams>(ctx, ord, (uint8_t)bind::Rms::Params,
                                      RmsParams{g.eps, (uint32_t)g.head_dim, 1u, 0u, 1.0f},
                                      &count);
                break;

            case Kernel::GdnPrep: {
                const GdnCoreParams gp = gdn_core_params(g);
                bind_const<GdnCoreParams>(ctx, ord, (uint8_t)bind::GdnPrep::Params, gp, &count);
                break;
            }

            case Kernel::GdnPrepSlotted: {
                const GdnCoreParams gp = gdn_core_params(g);
                bind_const<GdnCoreParams>(ctx, ord, (uint8_t)bind::GdnPrep::Params, gp, &count);
                break;
            }

            case Kernel::GdnCore: {
                const GdnCoreParams gp = gdn_core_params(g);
                const uint8_t pbuf = gdn_prep ? (uint8_t)bind::GdnCoreRecurrent::Params
                                              : (uint8_t)bind::GdnCore::Params;
                bind_const<GdnCoreParams>(ctx, ord, pbuf, gp, &count);
                break;
            }

            case Kernel::GdnCoreSlotted: {
                const GdnCoreParams gp = gdn_core_params(g);
                bind_const<GdnCoreParams>(ctx, ord, (uint8_t)bind::GdnCoreRecurrent::Params,
                                          gp, &count);
                break;
            }

            case Kernel::GatedRms:
                bind_const<GatedRmsParams>(ctx, ord, (uint8_t)bind::GatedRms::Params,
                                           GatedRmsParams{g.eps, (uint32_t)g.gdn_v_dim}, &count);
                break;

            case Kernel::QSplit:
                bind_const<int>(ctx, ord, (uint8_t)bind::QSplit::HeadDim, g.head_dim, &count);
                break;

            case Kernel::Rope:
            case Kernel::RopeK:
                bind_const<float>(ctx, ord, (uint8_t)bind::Rope::Scale,   rope_scale, &count);
                bind_const<float>(ctx, ord, (uint8_t)bind::Rope::Base,    rope_base,  &count);
                bind_const<int>  (ctx, ord, (uint8_t)bind::Rope::HeadDim, g.head_dim, &count);
                break;

            case Kernel::KvAppend:
                bind_const<int>   (ctx, ord, (uint8_t)bind::KvAppend::HeadDim,     g.head_dim,  &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppend::KHeadStride, head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppend::KSeqStride,  seq_stride,  &count);
                break;

            case Kernel::KvAppendPaged:
                bind_const<int>(ctx, ord, (uint8_t)bind::KvAppendPaged::HeadDim, g.head_dim, &count);
                // These two preserved M=1 ABI entries are unused by the paged
                // shader but intentionally bound so every declared table slot
                // has a concrete value.
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppendPaged::KHeadStride,
                                   head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::KvAppendPaged::KSeqStride,
                                   seq_stride, &count);
                bind_const<int>(ctx, ord, (uint8_t)bind::KvAppendPaged::PageSize,
                                g.kv_page_size, &count);
                bind_const<int>(ctx, ord, (uint8_t)bind::KvAppendPaged::NKvHeads,
                                g.n_kv_heads, &count);
                break;

            case Kernel::Sdpa:
                bind_const<int>   (ctx, ord, (uint8_t)bind::Sdpa::GqaFactor,   gqa_factor,  &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::KHeadStride, head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::KSeqStride,  seq_stride,  &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::VHeadStride, head_stride, &count);
                bind_const<size_t>(ctx, ord, (uint8_t)bind::Sdpa::VSeqStride,  seq_stride,  &count);
                bind_const<float> (ctx, ord, (uint8_t)bind::Sdpa::Scale,       sdpa_scale,  &count);
                break;

            case Kernel::SdpaPaged:
                bind_const<int>(ctx, ord, (uint8_t)bind::SdpaPaged::GqaFactor, gqa_factor, &count);
                bind_const<int>(ctx, ord, (uint8_t)bind::SdpaPaged::PageSize,
                                g.kv_page_size, &count);
                bind_const<int>(ctx, ord, (uint8_t)bind::SdpaPaged::NKvHeads,
                                g.n_kv_heads, &count);
                bind_const<float>(ctx, ord, (uint8_t)bind::SdpaPaged::Scale, sdpa_scale, &count);
                // qwen3.5's attention layers are all full, but the kernel they
                // share now takes a window. Binding 0 says so; leaving it
                // unbound would read a window out of uninitialized memory.
                bind_const<int>(ctx, ord, (uint8_t)bind::SdpaPaged::Window, 0, &count);
                break;

            case Kernel::SiluMul:
            case Kernel::AttnGate:
                break;
            case Kernel::LlExpertSiluMul:
            case Kernel::LlMoeSort:
            case Kernel::LlMoeGather:
                // Bound above, by `bind_token_consts`: these are the only
                // constants in this DAG that the batch width can change.
                break;

            // ── the mixture's routing ──
            // Width-INVARIANT, both of them: the router reads one row and
            // writes k logits whatever the batch is, and the combine sums k
            // slots into one row. Only the sort and the gather in between know
            // how many rows there are -- and, when a prefill runs the whole
            // group at once, the pitch its rows are laid out at.
            case Kernel::GoRouterTopK:
                bind_const<RouterParams>(
                    ctx, ord, (uint8_t)bind::GoRouterTopK::Params,
                    RouterParams{(uint32_t)g.n_experts, (uint32_t)g.experts_per_token,
                                 g.norm_topk_prob ? 0u : 1u, (uint32_t)row_pitch},
                    &count);
                break;
            case Kernel::LlMoeCombine:
                bind_const<ExpertCombineParams>(
                    ctx, ord, (uint8_t)bind::GoExpertCombine::Params,
                    ExpertCombineParams{(uint32_t)g.hidden, (uint32_t)g.experts_per_token,
                                        (uint32_t)row_pitch},
                    &count);
                break;
            case Kernel::LlSharedCombine:
                // The gate is one value per ROW, so this needs the row WIDTH
                // and not the element count -- the kernel indexes `gate[row]`.
                bind_const<uint32_t>(ctx, ord, (uint8_t)bind::SharedCombine::Width,
                                     (uint32_t)g.hidden, &count);
                break;

            case Kernel::Residual:
            case Kernel::LayerOut:
                break;

            // No const params: Argmax.
            default:
                break;
        }
    }
    return count;
}

size_t decode_consts_budget(const std::vector<Dispatch>& dag) {
    // Worst case 6 const slots/dispatch (sdpa), each ≤ 256-aligned. Be generous.
    return (dag.size() * 6 + 64) * 256;
}

}  // namespace pie::metal
