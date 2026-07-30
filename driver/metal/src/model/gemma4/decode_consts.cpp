// Gemma 4's per-dispatch constants.
//
// qwen3.5's equivalent is a switch on the KIND: every `QmvQ` in the stack has
// the same shape, so one answer serves all of them. Gemma 4's cannot be, because
// four of its constants depend on the LAYER:
//
//   * `head_dim` — 256 on sliding layers, 512 on full ones
//   * the RoPE base — 1e4 sliding, 1e6 full
//   * the rotated fraction — all of the head sliding, a quarter of it full
//   * the MLP width — doubles over the KV-shared range
//
// So this walks the DAG and asks the geometry per dispatch. `Dispatch::layer`
// is what makes that possible; it was already carried for golden-dump naming.

#include "decode_consts.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../../batch/decode_abi.hpp"
#include "kernels.hpp"

namespace pie::metal::gemma4 {
namespace {

// Replicated EXACTLY from the .metal sources; see kernels.hpp for the others.
struct RmsParams {  // rms_norm.metal:22 (buffer 3)
    float eps;
    std::uint32_t axis_size;
    std::uint32_t w_stride;
    std::uint32_t plus_one;
};

template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, std::uint8_t idx, const V& val, int* count) {
    SlotHandle s = ctx.heap_alloc(sizeof(V));
    if (!s.valid()) {
        throw std::runtime_error("gemma4 consts: heap_alloc failed (budget too small)");
    }
    std::memcpy(s.contents(), &val, sizeof(V));
    ctx.arg_bind_ordinal(ord, idx, s);
    if (count != nullptr) ++*count;
}

/// Gemma 4 stores plain RMS weights. `plus_one` stays 0 — the `(1 + w)` gain is
/// an earlier Gemma's, and applying it here is an ~80% error per norm that the
/// residual stream compounds.
RmsParams rms_params(const Gemma4Geometry& g, int axis) {
    return RmsParams{g.eps, std::uint32_t(axis), 1u, g.norm_plus_one ? 1u : 0u};
}

}  // namespace

KN qmv_kn(Kind k, const Gemma4Geometry& g, int layer) {
    const int H = g.hidden;
    const int hd = layer >= 0 ? g.head_dim_of(layer) : g.head_dim;
    const int q_dim = g.n_q_heads * hd;
    const int kv_dim = g.n_kv_heads * hd;
    const int inter = layer >= 0 ? g.intermediate_of(layer) : g.intermediate;
    const int ple_total = g.n_layers * g.per_layer_emb_dim;
    switch (k) {
        case Kind::QmvQ: return {H, q_dim};
        case Kind::QmvK: return {H, kv_dim};
        case Kind::QmvV: return {H, kv_dim};
        case Kind::QmvO: return {q_dim, H};
        case Kind::QmvGate: return {H, inter};
        case Kind::QmvUp: return {H, inter};
        case Kind::QmvDown: return {inter, H};
        case Kind::LmHead: return {H, g.vocab};
        // PLE: the model projection fans hidden out to the whole table; the
        // per-layer gate and projection work one layer's slice at a time.
        case Kind::PleProjGemv: return {H, ple_total};
        case Kind::PleGateGemv: return {H, g.per_layer_emb_dim};
        case Kind::PleProjLayerGemv: return {g.per_layer_emb_dim, H};
        default: return {0, 0};
    }
}

int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const Gemma4Geometry& g) {
    int count = 0;
    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;

        if (const KN kn = qmv_kn(d.kind, g, L); kn.N != 0) {
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::N, kn.N, &count);
            continue;
        }

        switch (d.kind) {
            // ── norms ──
            case Kind::AttnNorm:
            case Kind::PostAttnNorm:
            case Kind::FfnNorm:
            case Kind::PostFfnNorm:
            case Kind::FinalRms:
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, g.hidden), &count);
                break;
            case Kind::QNorm:
            case Kind::KNorm:
                // Per-head, so the axis is this layer's head_dim, not hidden.
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, hd), &count);
                break;
            case Kind::PleNorm:
            case Kind::PleProjNorm:
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, g.per_layer_emb_dim), &count);
                break;
            case Kind::VNorm: {
                // Weightless: eps and the axis, nothing else.
                const VNormParams p{g.eps, std::uint32_t(hd)};
                bind_const<VNormParams>(ctx, ord, (std::uint8_t)bind::VNorm::Params, p, &count);
                break;
            }

            // ── rope: base and rotated width are both per attention type ──
            case Kind::RopeQ:
            case Kind::RopeK:
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Rope::Scale, 1.0f, &count);
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Rope::Base,
                                  L >= 0 ? g.rope_theta_of(L) : g.rope_theta_global, &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Rope::HeadDim, hd, &count);
                break;

            // ── attention ──
            case Kind::Sdpa: {
                const std::int32_t gqa = g.n_kv_heads > 0 ? g.n_q_heads / g.n_kv_heads : 1;
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Sdpa::GqaFactor, gqa,
                                         &count);
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Sdpa::Scale,
                                  1.0f / std::sqrt(float(hd)), &count);
                // The window is what makes a sliding layer sliding; a full
                // layer's PSO does not read the slot at all.
                if (d.sliding) {
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::SdpaSliding::Window,
                                             g.sliding_window, &count);
                }
                break;
            }
            case Kind::KvAppend:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::KvAppend::HeadDim, hd,
                                         &count);
                break;

            // ── elementwise ──
            case Kind::GegluTanh: {
                const GegluParams p{std::uint32_t(L >= 0 ? g.intermediate_of(L) : g.intermediate)};
                bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p, &count);
                break;
            }
            case Kind::PleGeglu: {
                const GegluParams p{std::uint32_t(g.per_layer_emb_dim)};
                bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p, &count);
                break;
            }
            case Kind::LayerScalar: {
                const LayerScalarParams p{std::uint32_t(g.hidden)};
                bind_const<LayerScalarParams>(ctx, ord, (std::uint8_t)bind::LayerScalar::Params, p,
                                              &count);
                break;
            }
            case Kind::PleCombine: {
                const PleCombineParams p{0.70710678118654752f,
                                         std::uint32_t(g.n_layers * g.per_layer_emb_dim)};
                bind_const<PleCombineParams>(ctx, ord, (std::uint8_t)bind::PleCombine::Params, p,
                                             &count);
                break;
            }
            case Kind::FinalSoftcap: {
                const SoftcapParams p{g.final_softcap, std::uint32_t(g.vocab)};
                bind_const<SoftcapParams>(ctx, ord, (std::uint8_t)bind::Softcap::Params, p, &count);
                break;
            }
            case Kind::AttnResidual:
            case Kind::FfnResidual:
            case Kind::PleResidual:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Residual::Width, g.hidden,
                                         &count);
                break;

            case Kind::EmbedGather:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Embed::Hidden, g.hidden,
                                         &count);
                break;
            case Kind::PleTokenGather:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Embed::Hidden,
                                         g.n_layers * g.per_layer_emb_dim, &count);
                break;

            default:
                break;
        }
    }
    return count;
}

}  // namespace pie::metal::gemma4
