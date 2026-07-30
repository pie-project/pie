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

/// The KV cache is [n_kv_heads, max_ctx, head_dim] per owning layer, so the head
/// stride depends on this layer's head width -- gemma4's two attention types do
/// not share one.
std::size_t k_head_stride(const Gemma4Geometry& g, int layer) {
    const int hd = layer >= 0 ? g.head_dim_of(layer) : g.head_dim;
    return std::size_t(g.kv_max_ctx) * std::size_t(hd);
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
                       const Gemma4Geometry& g, int rows, bool paged) {
    // `rows` is the token count. Only two kinds of constant depend on it: a
    // GEMM needs to be told M, and an elementwise kernel over a contiguous
    // [rows, width] buffer counts rows*width. Everything else is geometry.
    const std::uint32_t R = std::uint32_t(rows < 1 ? 1 : rows);
    int count = 0;
    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;

        if (const KN kn = qmv_kn(d.kind, g, L); kn.N != 0) {
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::N, kn.N, &count);
            // The GEMM shares Qmv's ordinals 0-6 and appends M. Bound
            // unconditionally: at rows==1 the matvec simply never reads slot 7,
            // and an unbound slot on the prefill path is a row count read out of
            // uninitialized memory.
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmm::M,
                                     std::int32_t(R), &count);
            continue;
        }

        switch (d.kind) {
            // ── norms ──
            case Kind::AttnNorm:
            case Kind::FfnNorm:
            case Kind::FinalRms:
            // The fused norm+residual reads the identical RmsParams at the
            // identical slot; only the buffers after it are new.
            case Kind::PostAttnResidual:
            case Kind::PostFfnResidual:
            case Kind::PleResidualScaled:
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, g.hidden), &count);
                break;
            case Kind::QNorm:
            case Kind::KNorm:
                // Per-head, so the axis is this layer's head_dim, not hidden.
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, hd), &count);
                break;
            // The two PLE norms are NOT the same width. `per_layer_projection_norm`
            // runs over a ple_dim-wide row (once per layer, on the [n_layers,
            // ple_dim] table); `post_per_layer_input_norm` is `RMSNorm(hidden)`
            // and runs on the projection's output, back in the residual stream.
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
                // `base` is log2(theta), not theta -- the kernel raises 2 to it.
                // Bound as theta, `exp2(-d * 1e6)` is 0 for every frequency but
                // the first, which is a rope that does nothing at all.
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Rope::Base,
                                  std::log2(L >= 0 ? g.rope_theta_of(L) : g.rope_theta_global),
                                  &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Rope::HeadDim, hd, &count);
                break;

            // ── attention ──
            //
            // Two ABIs, one kind. The paged shader puts a page table where the
            // contiguous one puts cache strides, so which set is bound is
            // decided by the caller and not guessed from `rows`: the executor
            // may run the paged kernel at one token.
            case Kind::Sdpa: {
                const std::int32_t gqa = g.n_kv_heads > 0 ? g.n_q_heads / g.n_kv_heads : 1;
                if (paged) {
                    using P = bind::SdpaPaged;
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::GqaFactor, gqa, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::PageSize,
                                             g.kv_page_size, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::NKvHeads,
                                             g.n_kv_heads, &count);
                    bind_const<float>(ctx, ord, (std::uint8_t)P::Scale, 1.0f, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::Window,
                                             d.sliding ? g.sliding_window : 0, &count);
                    break;
                }
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Sdpa::GqaFactor, gqa,
                                         &count);
                // 1.0, not 1/sqrt(head_dim): gemma4 folds the attention scale
                // into the q-norm's learned weights, so mlx-lm's `Attention`
                // sets `self.scale = 1.0`. Dividing again is a second scaling
                // of an already-scaled query.
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Sdpa::Scale, 1.0f, &count);
                // The cache layout, [n_kv_heads, max_ctx, head_dim]. Never bound
                // before: four `constant size_t&` slots read out of whatever the
                // argument table held, which is wrong attention, not a crash.
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::KHeadStride,
                                        k_head_stride(g, L), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::KSeqStride,
                                        std::size_t(hd), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::VHeadStride,
                                        k_head_stride(g, L), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::Sdpa::VSeqStride,
                                        std::size_t(hd), &count);
                // Both attention types run an `sdpa_vector_decode_swa`
                // instantiation -- they differ by head width, not by kernel --
                // so BOTH read this slot. Binding it only for sliding layers
                // left layers 4, 9, 14... reading an unbound window, which is
                // wrong attention rather than a crash. 0 means "attend all".
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::SdpaSliding::Window,
                                         d.sliding ? g.sliding_window : 0, &count);
                // Row strides, stated rather than inferred. At decode M==1 and
                // the row index is 0, so these only ever matter for prefill --
                // but an unbound slot would matter there too.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::SdpaSliding::QRowStride,
                                         std::int32_t(g.n_q_heads * hd), &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::SdpaSliding::ORowStride,
                                         std::int32_t(g.n_q_heads * hd), &count);
                break;
            }
            case Kind::KvAppend:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::KvAppend::HeadDim, hd,
                                         &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::KvAppend::KHeadStride,
                                        k_head_stride(g, L), &count);
                bind_const<std::size_t>(ctx, ord, (std::uint8_t)bind::KvAppend::KSeqStride,
                                        std::size_t(hd), &count);
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
            case Kind::GegluTanh: {
                const GegluParams p{R * std::uint32_t(L >= 0 ? g.intermediate_of(L)
                                                                : g.intermediate)};
                bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p, &count);
                break;
            }
            case Kind::PleGeglu: {
                const GegluParams p{R * std::uint32_t(g.per_layer_emb_dim)};
                bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p, &count);
                break;
            }
            case Kind::LayerScalar: {
                const LayerScalarParams p{R * std::uint32_t(g.hidden)};
                bind_const<LayerScalarParams>(ctx, ord, (std::uint8_t)bind::LayerScalar::Params, p,
                                              &count);
                break;
            }
            case Kind::PleCombine: {
                const PleCombineParams p{0.70710678118654752f,
                                         R * std::uint32_t(g.n_layers * g.per_layer_emb_dim)};
                bind_const<PleCombineParams>(ctx, ord, (std::uint8_t)bind::PleCombine::Params, p,
                                             &count);
                break;
            }
            case Kind::FinalSoftcap: {
                const SoftcapParams p{g.final_softcap, std::uint32_t(g.vocab)};
                bind_const<SoftcapParams>(ctx, ord, (std::uint8_t)bind::Softcap::Params, p, &count);
                break;
            }

            // Embed's `Hidden` is a row PITCH, not a count -- the mb kernel
            // indexes (channel, token) -- so it does NOT scale with rows.
            case Kind::EmbedGather:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Embed::Hidden, g.hidden,
                                         &count);
                // Gemma scales its embedding by sqrt(hidden_size). It cannot be
                // folded into the table: the LM head reads the same tied
                // weights, unscaled.
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Embed::Scale,
                                  std::sqrt(float(g.hidden)), &count);
                break;
            case Kind::PleTokenGather:
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Embed::Hidden,
                                         g.n_layers * g.per_layer_emb_dim, &count);
                bind_const<float>(ctx, ord, (std::uint8_t)bind::Embed::Scale,
                                  std::sqrt(float(g.per_layer_emb_dim)), &count);
                break;

            default:
                break;
        }
    }
    return count;
}

}  // namespace pie::metal::gemma4
