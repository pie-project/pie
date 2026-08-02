// GPT-OSS's per-dispatch constants.
//
// Unlike gemma4's, none of these depend on the LAYER: every layer has the same
// head width, the same rope, the same MLP shape. What varies is the attention
// type, and that is one value (the window).
//
// Two of them are the family's own, and both are easy to get silently wrong:
//
//   * The attention SCALE is `1/sqrt(head_dim)` applied to q, as in qwen3.5 --
//     not gemma4's 1.0, which is what a family that folds the scale into a
//     q-norm uses. gpt-oss has no q-norm.
//   * The routed matvecs bind the SAME K and N as an unrouted one. The expert
//     axis is not a shape the kernel is told about; it derives every stride from
//     K and N, so a routed projection's constants are indistinguishable from a
//     dense one's. That is deliberate -- one fewer thing to disagree.

#include "decode_consts.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../../batch/decode_abi.hpp"
#include "kernels.hpp"
#include "../shared_kernels.hpp"

namespace pie::metal::gptoss {
namespace {

using shared_kernels::RmsParams;

template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, std::uint8_t idx, const V& val, int* count) {
    shared_kernels::bind_const(ctx, ord, idx, val, count, "gpt-oss");
}

/// The KV cache is [n_kv_heads, kv_max_ctx, head_dim] per layer.
std::size_t k_head_stride(const GptOssGeometry& g) {
    return std::size_t(g.kv_max_ctx) * std::size_t(g.head_dim);
}

}  // namespace

KN qmv_kn(Kind k, const GptOssGeometry& g) {
    const int H = g.hidden;
    switch (k) {
        case Kind::QmvQ: return {H, g.q_dim()};
        case Kind::QmvK: return {H, g.kv_dim()};
        case Kind::QmvV: return {H, g.kv_dim()};
        case Kind::QmvO: return {g.q_dim(), H};
        case Kind::RouterGemv: return {H, g.n_experts};
        // The routed three. Same shape as a dense projection: the expert axis is
        // a base offset the kernel computes, not a dimension it is told.
        case Kind::ExpertGate: return {H, g.intermediate};
        case Kind::ExpertUp: return {H, g.intermediate};
        case Kind::ExpertDown: return {g.intermediate, H};
        case Kind::LmHead: return {H, g.vocab};
        default: return {0, 0};
    }
}

int bind_gptoss_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const GptOssGeometry& g, int rows, bool paged, int head_rows) {
    const std::uint32_t R = std::uint32_t(rows < 1 ? 1 : rows);
    const std::uint32_t K = std::uint32_t(g.experts_per_token);
    // The tail runs on the SAMPLED rows, which `RowGather` compacted. 0 means
    // "all of them", which is what a test that reads every row wants.
    const std::uint32_t S =
        head_rows < 1 ? R : std::min(R, std::uint32_t(head_rows));
    int count = 0;

    // YaRN's frequency table and its temperature correction: computed once, and
    // the same buffer serves every rope dispatch in the model.
    const std::vector<float> inv_freq = yarn_inv_freq(g);
    SlotHandle freqs = ctx.heap_alloc(inv_freq.size() * sizeof(float));
    if (!freqs.valid()) {
        throw std::runtime_error("gpt-oss consts: heap_alloc failed for the YaRN table");
    }
    std::memcpy(freqs.contents(), inv_freq.data(), inv_freq.size() * sizeof(float));
    const float mscale = yarn_mscale(g);

    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;

        if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::GoQmv::K, kn.K, &count);
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::GoQmv::N, kn.N, &count);
            const bool routed = d.kind == Kind::ExpertGate || d.kind == Kind::ExpertUp ||
                                d.kind == Kind::ExpertDown;
            if (routed) {
                // Only `down` reads a per-expert input.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::GoQmv::XSlotStride,
                                         d.kind == Kind::ExpertDown ? kn.K : 0, &count);
            }
            // The token row's pitch in `x`, and how many expert slots a row
            // selects. Bound for EVERY matvec, routed or not: at M=1 the row is
            // 0 and neither is read, and an unbound `constant int&` on the M>1
            // path is a stride out of uninitialized memory.
            bind_const<std::int32_t>(
                ctx, ord, (std::uint8_t)bind::GoQmv::XRowStride,
                d.kind == Kind::ExpertDown ? std::int32_t(K) * kn.K : kn.K, &count);
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::GoQmv::SlotsPerRow,
                                     routed ? std::int32_t(K) : 1, &count);
            continue;
        }

        switch (d.kind) {
            case Kind::AttnNorm:
            case Kind::FfnNorm:
            case Kind::FinalRms:
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      RmsParams{g.eps, std::uint32_t(g.hidden), 1u, 0u}, &count);
                break;

            case Kind::RopeQ:
            case Kind::RopeK:
                bind_const<float>(ctx, ord, (std::uint8_t)bind::RopeFreqs::Scale, 1.0f, &count);
                ctx.arg_bind_ordinal(ord, (std::uint8_t)bind::RopeFreqs::InvFreq, freqs);
                ++count;
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::RopeFreqs::HeadDim,
                                         g.head_dim, &count);
                bind_const<float>(ctx, ord, (std::uint8_t)bind::RopeFreqs::MScale, mscale, &count);
                // `rope_neox_freqs_mb`'s row pitch. q and k have different head
                // counts and share the kernel, so its grid cannot supply it.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::RopeFreqs::RowStride,
                                         d.kind == Kind::RopeQ ? g.q_dim() : g.kv_dim(),
                                         &count);
                break;

            case Kind::KvAppend:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::KvAppend::HeadDim,
                                         g.head_dim, &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::KvAppend::KHeadStride,
                                        k_head_stride(g), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::KvAppend::KSeqStride,
                                        std::size_t(g.head_dim), &count);
                if (paged) {
                    bind_const<std::int32_t>(ctx, ord,
                                             (std::uint8_t)bind::KvAppendPaged::PageSize,
                                             g.kv_page_size, &count);
                    bind_const<std::int32_t>(ctx, ord,
                                             (std::uint8_t)bind::KvAppendPaged::NKvHeads,
                                             g.n_kv_heads, &count);
                }
                break;

            case Kind::SdpaSink: {
                if (paged) {
                    // The paged ABI puts different meanings at the same slots,
                    // so it is bound as its own thing rather than as a subset:
                    // `SdpaSink::N` is `SdpaPaged::PositionIds`, a length read
                    // as a pointer.
                    using P = bind::SdpaPaged;
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::GqaFactor,
                                             g.gqa_factor(), &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::PageSize,
                                             g.kv_page_size, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::NKvHeads,
                                             g.n_kv_heads, &count);
                    bind_const<float>(ctx, ord, (std::uint8_t)P::Scale,
                                      1.0f / std::sqrt(float(g.head_dim)), &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::Window,
                                             d.sliding ? g.sliding_window : 0, &count);
                    break;
                }
                using S = bind::SdpaSink;
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)S::GqaFactor, g.gqa_factor(),
                                         &count);
                // 1/sqrt(head_dim) on q. gpt-oss has no q-norm to fold it into.
                bind_const<float>(ctx, ord, (std::uint8_t)S::Scale,
                                  1.0f / std::sqrt(float(g.head_dim)), &count);
                // 0 means "attend all", which is what a full layer wants; the
                // shared kernel reads this slot either way.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)S::Window,
                                         d.sliding ? g.sliding_window : 0, &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)S::KHeadStride, k_head_stride(g),
                                        &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)S::KSeqStride,
                                        std::size_t(g.head_dim), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)S::VHeadStride, k_head_stride(g),
                                        &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)S::VSeqStride,
                                        std::size_t(g.head_dim), &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)S::QRowStride, g.q_dim(),
                                         &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)S::ORowStride, g.q_dim(),
                                         &count);
                break;
            }

            case Kind::RouterTopK:
                bind_const<RouterParams>(ctx, ord, (std::uint8_t)bind::GoRouterTopK::Params,
                                         RouterParams{std::uint32_t(g.n_experts), K}, &count);
                break;

            case Kind::ExpertSwiGlu:
                bind_const<SwiGluParams>(
                    ctx, ord, (std::uint8_t)bind::GoSwiGlu::Params,
                    SwiGluParams{R * K * std::uint32_t(g.intermediate), g.swiglu_limit,
                                 g.swiglu_alpha},
                    &count);
                break;

            case Kind::ExpertCombine:
                bind_const<ExpertCombineParams>(
                    ctx, ord, (std::uint8_t)bind::GoExpertCombine::Params,
                    // A row PITCH, not a count: the kernel takes the row on its
                    // second grid axis and strides `y`, the weights and `out`
                    // by it.
                    ExpertCombineParams{std::uint32_t(g.hidden), K}, &count);
                break;

            case Kind::AttnResidual:
            case Kind::FfnResidual:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Residual::Width,
                                         std::int32_t(R) * g.hidden, &count);
                break;

            case Kind::EmbedGather:
                // A row pitch, not a count: it does not scale with rows.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Embed::Hidden, g.hidden,
                                         &count);
                break;

            case Kind::RowGather:
                bind_const<RowGatherParams>(ctx, ord, (std::uint8_t)bind::RowGather::Params,
                                            RowGatherParams{std::uint32_t(g.hidden), S},
                                            &count);
                break;

            default:
                break;
        }
    }
    return count;
}

}  // namespace pie::metal::gptoss
