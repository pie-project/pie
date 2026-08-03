// The llama families' per-dispatch constants.
//
// Every slot a dispatch declares is bound, including the ones the M=1 kernel
// never reads. An unbound slot is not a crash: the argument table holds
// whatever was there last, so a stride comes back as a pointer or a row count
// out of uninitialised memory. Binding all of them costs a few writes at setup
// and removes the whole class.
//
// The routed FFN is where the ABI needs the most care, and the reason is that
// the three expert projections do NOT read the same thing. `gate` and `up` read
// one shared norm output -- every slot of every row sees the same vector, so
// their slot stride is 0. `down` reads the SwiGLU's [rows, k, moe_intermediate]
// stack, where each slot has its own vector. A kernel that guessed would read
// four copies of the first expert's activation and produce a plausible wrong
// token, which is why `XSlotStride` is stated rather than inferred from the
// kind.

#include "decode_consts.hpp"

#include <algorithm>
#include <cmath>

#include "../../batch/decode_abi.hpp"
#include "../shared_kernels.hpp"
#include "encode.hpp"
#include "scratch.hpp"

namespace pie::metal::llama {

using shared_kernels::ExpertCombineParams;
using shared_kernels::MoeRouteParams;
using shared_kernels::RmsParams;
using shared_kernels::RouterParams;
using shared_kernels::RowGatherParams;

namespace {

template <class V>
void bind_const(RawMetalContext& ctx, int ord, std::uint8_t idx, const V& v, int* count) {
    shared_kernels::bind_const<V>(ctx, ord, idx, v, count, "llama");
}

RmsParams rms_params(const LlamaGeometry& g, int axis) {
    // `plus_one` is 0: llama's RMSNorm applies the learned gain directly, where
    // gemma stores `w - 1` and applies `(1 + w)`. Getting this wrong is not a
    // crash -- it is every norm off by one, which still produces text.
    return RmsParams{g.eps, std::uint32_t(axis), 1u, 0u};
}

/// The contiguous KV cache's layout, [n_kv_heads, max_ctx, head_dim].
std::size_t k_head_stride(const LlamaGeometry& g) {
    const int ctx = g.kv_max_ctx > 0 ? g.kv_max_ctx : 1;
    return std::size_t(ctx) * std::size_t(g.head_dim);
}

}  // namespace

KN qmv_kn(Kind k, const LlamaGeometry& g) {
    const int H = g.hidden;
    switch (k) {
        case Kind::QmvQ: return {H, g.q_width()};
        case Kind::QmvK: return {H, g.kv_width()};
        case Kind::QmvV: return {H, g.kv_width()};
        case Kind::QmvO: return {g.q_width(), H};
        case Kind::QmvGate: return {H, g.intermediate};
        case Kind::QmvUp: return {H, g.intermediate};
        case Kind::QmvDown: return {g.intermediate, H};
        case Kind::LmHead: return {H, g.vocab};
        // The router is an ordinary matvec, `n_experts` wide.
        case Kind::Router: return {H, g.n_experts};
        // Routed. Same K and N as their dense counterparts -- the expert axis
        // is a stride into the weight stack, not a wider matrix.
        case Kind::ExpertGate: return {H, g.moe_intermediate};
        case Kind::ExpertUp: return {H, g.moe_intermediate};
        case Kind::ExpertDown: return {g.moe_intermediate, H};
        default: return {0, 0};
    }
}

int bind_llama_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                      const LlamaGeometry& g, int rows, bool paged) {
    const std::uint32_t R = std::uint32_t(rows < 1 ? 1 : rows);
    const std::uint32_t K = std::uint32_t(g.experts_per_token > 0 ? g.experts_per_token : 1);
    int count = 0;

    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;

        if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::N, kn.N, &count);
            if (is_routed(d.kind)) {
                using Q = bind::GoQmv;
                // The matvec form now reads the SORTED stack, one row per
                // (token, slot) pair rather than k slots hanging off a token
                // row. That collapses all three of these to the dense case:
                // no slot axis, a row pitch that is just the input width, and
                // one expert per row -- the sort is what made the pair axis
                // disappear, and `row_expert` is what replaced `tid.z`.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::XSlotStride, 0, &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::XRowStride,
                                         std::int32_t(kn.K), &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::SlotsPerRow, 1, &count);
            }
            continue;
        }

        switch (d.kind) {
            // ── norms ──
            case Kind::AttnNorm:
            case Kind::FfnNorm:
            case Kind::FinalRms:
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, g.hidden), &count);
                break;
            case Kind::QNorm:
            case Kind::KNorm:
                // Qwen3's qk-norms are per HEAD: the axis is head_dim, not the
                // whole projection. Normalising across heads instead would mix
                // them, which is wrong attention rather than a crash.
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, g.head_dim), &count);
                break;

            // ── rope ──
            case Kind::RopeQ:
            case Kind::RopeK:
                // `rope_scale` is the linear position divisor. `geometry_from_facts`
                // refuses any non-linear scaling, so there is nothing else to
                // express here.
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Rope::Scale,
                                  g.rope_scale != 0.0f ? 1.0f / g.rope_scale : 1.0f, &count);
                // `base` is log2(theta), not theta -- the kernel raises 2 to it.
                // Bound as theta, `exp2(-d * 5e5)` underflows to 0 for every
                // frequency but the first, which is a rope that does nothing.
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Rope::Base,
                                  std::log2(g.rope_theta), &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Rope::HeadDim,
                                         g.head_dim, &count);
                break;

            // ── attention: two ABIs, one kind ──
            case Kind::Sdpa: {
                const std::int32_t gqa = g.n_kv_heads > 0 ? g.n_q_heads / g.n_kv_heads : 1;
                // 1/sqrt(head_dim), unlike gemma4 -- which folds its scale into
                // the q-norm weights and so passes 1.0. Llama does not.
                const float scale =
                    g.head_dim > 0 ? 1.0f / std::sqrt(float(g.head_dim)) : 1.0f;
                if (paged) {
                    using P = bind::SdpaPaged;
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::GqaFactor, gqa, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::PageSize,
                                             g.kv_page_size, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::NKvHeads, g.n_kv_heads,
                                             &count);
                    bind_const<float>(ctx, ord, (std::uint8_t)P::Scale, scale, &count);
                    // Full attention, but the window is still BOUND. One paged
                    // kernel serves the sliding families and this one, so the
                    // slot exists either way, and <= 0 is how "no window" is
                    // spelled. Leaving it unbound reads whatever the argument
                    // table last held at that ordinal -- a window nobody asked
                    // for, which truncates attention rather than crashing.
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::Window, 0, &count);
                    break;
                }
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Sdpa::GqaFactor, gqa,
                                         &count);
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Sdpa::Scale, scale, &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::KHeadStride,
                                        k_head_stride(g), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::KSeqStride,
                                        std::size_t(g.head_dim), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::VHeadStride,
                                        k_head_stride(g), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::VSeqStride,
                                        std::size_t(g.head_dim), &count);
                break;
            }
            case Kind::KvAppend:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::KvAppend::HeadDim,
                                         g.head_dim, &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::KvAppend::KHeadStride,
                                        k_head_stride(g), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::KvAppend::KSeqStride,
                                        std::size_t(g.head_dim), &count);
                if (paged) {
                    // The paged scatter keeps the contiguous prefix verbatim and
                    // appends its own two; both are bound so no declared slot is
                    // left holding whatever the table had.
                    bind_const<std::int32_t>(ctx, ord,
                                             (std::uint8_t)bind::KvAppendPaged::PageSize,
                                             g.kv_page_size, &count);
                    bind_const<std::int32_t>(ctx, ord,
                                             (std::uint8_t)bind::KvAppendPaged::NKvHeads,
                                             g.n_kv_heads, &count);
                }
                break;

            // ── elementwise ──
            case Kind::AttnResidual:
            case Kind::FfnResidual:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Residual::Width,
                                         std::int32_t(R * std::uint32_t(g.hidden)), &count);
                break;
            case Kind::SiluMul:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::SiluMul::Width,
                                         std::int32_t(R * std::uint32_t(g.intermediate)),
                                         &count);
                break;
            case Kind::ExpertSiluMul:
                // The whole SORTED stack, flat: gate, up and out share a
                // layout, so the pair axis is just more elements. Sized off
                // the sorted count and not `rows * k`, because the sort pads
                // every expert's run to a tile and the padding rows are real
                // rows of the buffers this reads.
                bind_const<std::int32_t>(
                    ctx, ord, (std::uint8_t)bind::SiluMul::Width,
                    std::int32_t(llama_moe_sorted_rows(g, int(R)) * g.moe_intermediate),
                    &count);
                break;

            // ── routing ──
            case Kind::RouterTopK: {
                const RouterParams p{std::uint32_t(g.n_experts),
                                     std::uint32_t(g.experts_per_token)};
                bind_const<RouterParams>(ctx, ord, (std::uint8_t)bind::GoRouterTopK::Params, p,
                                         &count);
                break;
            }
            case Kind::ExpertSort:
            case Kind::ExpertGather: {
                // One params struct for both, so the sort's padding and the
                // gather's bounds cannot disagree. `width` is only read by the
                // gather, which moves hidden-wide rows.
                const int sorted = llama_moe_sorted_rows(g, int(R));
                const MoeRouteParams p{
                    std::uint32_t(int(R) * int(K)),
                    std::uint32_t(g.n_experts),
                    K,
                    std::uint32_t(llama_moe_tile_rows(g, int(R))),
                    std::uint32_t(sorted),
                    std::uint32_t(g.hidden)};
                const std::uint8_t idx = d.kind == Kind::ExpertSort
                                             ? (std::uint8_t)bind::MoeRouteSort::Params
                                             : (std::uint8_t)bind::MoeRouteRows::Params;
                bind_const<MoeRouteParams>(ctx, ord, idx, p, &count);
                break;
            }
            case Kind::ExpertCombine: {
                const ExpertCombineParams p{std::uint32_t(g.hidden),
                                            std::uint32_t(g.experts_per_token)};
                bind_const<ExpertCombineParams>(
                    ctx, ord, (std::uint8_t)bind::GoExpertCombine::Params, p, &count);
                break;
            }

            case Kind::RowGather: {
                const RowGatherParams p{std::uint32_t(g.hidden), R};
                bind_const<RowGatherParams>(ctx, ord, (std::uint8_t)bind::RowGather::Params, p,
                                            &count);
                break;
            }

            // The gather's row WIDTH. It looks like geometry the kernel could
            // infer and is not: `embed_gather_4bit` derives the packed row
            // pitch and the group count from it, so a stale value does not
            // truncate the embedding, it reads a different row of the table.
            case Kind::EmbedGather:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Embed::Hidden, g.hidden,
                                         &count);
                break;

            // The argmax takes its length from the LM head's `N`, which the
            // matvec branch above already bound.
            case Kind::Argmax:
            default:
                break;
        }
    }
    return count;
}

}  // namespace pie::metal::llama
