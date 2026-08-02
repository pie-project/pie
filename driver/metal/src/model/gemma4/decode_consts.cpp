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

#include "encode.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../../batch/decode_abi.hpp"
#include "kernels.hpp"
#include "../shared_kernels.hpp"
#include "scratch.hpp"

namespace pie::metal::gemma4 {
namespace {

using shared_kernels::RmsParams;

template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, std::uint8_t idx, const V& val, int* count) {
    shared_kernels::bind_const(ctx, ord, idx, val, count, "gemma4");
}

/// Gemma 4 stores plain RMS weights. `plus_one` stays 0 — the `(1 + w)` gain is
/// an earlier Gemma's, and applying it here is an ~80% error per norm that the
/// residual stream compounds.
RmsParams rms_params(const Gemma4Geometry& g, int axis) {
    return RmsParams{
        g.eps, std::uint32_t(axis), 1u, g.norm_plus_one ? 1u : 0u, 1.0f};
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
    // Per layer: the full-attention layers use `num_global_key_value_heads`,
    // which on gemma-4-26B is 2 against the sliding layers' 8.
    const int kv_dim = (layer >= 0 ? g.n_kv_heads_of(layer) : g.n_kv_heads) * hd;
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
        // ── the mixture ──
        case Kind::RouterGemv: return {H, g.n_experts};
        case Kind::ExpertGate: return {H, g.moe_intermediate};
        case Kind::ExpertUp: return {H, g.moe_intermediate};
        case Kind::ExpertDown: return {g.moe_intermediate, H};
        default: return {0, 0};
    }
}

int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const Gemma4Geometry& g, int rows, bool paged, int head_rows) {
    // `rows` is the token count. Only two kinds of constant depend on it: a
    // GEMM needs to be told M, and an elementwise kernel over a contiguous
    // [rows, width] buffer counts rows*width. Everything else is geometry.
    const std::uint32_t R = std::uint32_t(rows < 1 ? 1 : rows);
    // The tail runs on the SAMPLED rows, which `RowGather` compacted. Defaulting
    // to `rows` keeps a caller that samples every row -- the raw tests -- saying
    // one number instead of two.
    const std::uint32_t S =
        head_rows < 1 ? R : std::min(R, std::uint32_t(head_rows));
    int count = 0;
    for (const Dispatch& d : dag) {
        const int ord = d.ordinal;
        const int L = d.layer;
        const int hd = L >= 0 ? g.head_dim_of(L) : g.head_dim;

        if (const KN kn = qmv_kn(d.kind, g, L); kn.N != 0) {
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::K, kn.K, &count);
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmv::N, kn.N, &count);
            if (d.kind == Kind::ExpertGate || d.kind == Kind::ExpertUp ||
                d.kind == Kind::ExpertDown) {
                using Q = bind::GoQmv;
                // The sort collapsed the slot axis: one row per (token, slot)
                // pair, so all three of these are the dense case over a taller
                // batch, and the pair's expert is `row_expert[p]`, not `tid.z`.
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::XSlotStride, 0, &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::XRowStride,
                                         std::int32_t(kn.K), &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::SlotsPerRow, 1, &count);
                bind_const<std::int32_t>(
                    ctx, ord, (std::uint8_t)bind::Qmm::M,
                    std::int32_t(gemma4_moe_sorted_rows(g, int(R))), &count);
                continue;
            }
            // The bounds-checked matvec reads the three stride constants the
            // routed one does -- it is the same kernel with the expert axis
            // switched off -- so a dense projection that lands on it must bind
            // them too. Bound whenever the kind COULD take it rather than only
            // when it does: the alternative is for this file to re-derive the
            // pipeline choice `pso_for` already made.
            {
                using Q = bind::GoQmv;
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::XSlotStride, 0, &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::XRowStride,
                                         std::int32_t(kn.K), &count);
                bind_const<std::int32_t>(ctx, ord, (std::uint8_t)Q::SlotsPerRow, 1, &count);
            }
            // The padded row count, matching `launch_shape_mb`: the GEMM is
            // dispatched over whole tiles, and M is what bounds its inner loop.
            const std::uint32_t m = std::uint32_t(
                gemma4_qmm_rows(g, int(d.kind == Kind::LmHead ? S : R)));
            // The GEMM shares Qmv's ordinals 0-6 and appends M. Bound
            // unconditionally: at rows==1 the matvec simply never reads slot 7,
            // and an unbound slot on the prefill path is a row count read out of
            // uninitialized memory.
            bind_const<std::int32_t>(ctx, ord, (std::uint8_t)bind::Qmm::M,
                                     std::int32_t(m), &count);
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
            // The mixture's three extra hidden-wide norms are the plain norm;
            // only the weight they bind differs, which `shared_kind` answers.
            case Kind::MoeNorm:
            case Kind::DenseBranchNorm:
            case Kind::MoeBranchNorm:
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params,
                                      rms_params(g, g.hidden), &count);
                break;
            // mlx-lm normalizes the router's input with `scale * hidden**-0.5`
            // and the checkpoint stores only `scale`. Without the root every
            // logit is 53x too large, which survives the top-k -- a uniform
            // positive scale does not reorder -- but comes out of the softmax
            // as a one-hot: the mixture degenerates to its single best expert.
            // This is the ONLY norm with a gain; it must not share a case with
            // the others, which is exactly the bug that cost an afternoon.
            case Kind::RouterNorm: {
                RmsParams p = rms_params(g, g.hidden);
                p.gain = 1.0f / std::sqrt(float(g.hidden));
                bind_const<RmsParams>(ctx, ord, (std::uint8_t)bind::Rms::Params, p, &count);
                break;
            }
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
            case Kind::VNorm:
            case Kind::VNormFromK: {
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
                const std::int32_t nkv = g.n_kv_heads_of(L);
                const std::int32_t gqa = nkv > 0 ? g.n_q_heads / nkv : 1;
                if (paged) {
                    using P = bind::SdpaPaged;
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::GqaFactor, gqa, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::PageSize,
                                             g.kv_page_size, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::NKvHeads, nkv, &count);
                    bind_const<float>(ctx, ord, (std::uint8_t)P::Scale, 1.0f, &count);
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::Window,
                                             d.sliding ? g.sliding_window : 0, &count);
                    // N, for the tiled pipeline's partial last tile. Bound
                    // whether or not this fire tiles: the bind table is per
                    // Kind and the pipeline choice is per row count. Unbound,
                    // the tiled kernel reads a stale ordinal for N and decides
                    // which of its rows exist from it -- wrong attention, not
                    // a crash.
                    bind_const<std::int32_t>(ctx, ord, (std::uint8_t)P::Rows,
                                             std::int32_t(R), &count);
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
                                             g.n_kv_heads_of(L), &count);
                }
                break;

            // ── the mixture ──
            case Kind::RouterTopK: {
                const shared_kernels::RouterParams p{std::uint32_t(g.n_experts),
                                                    std::uint32_t(g.experts_per_token)};
                bind_const<shared_kernels::RouterParams>(
                    ctx, ord, (std::uint8_t)bind::GoRouterTopK::Params, p, &count);
                break;
            }
            case Kind::ExpertSort:
            case Kind::ExpertGather: {
                // One params struct for both, so the sort's padding and the
                // gather's bounds cannot disagree. `width` is only read by the
                // gather, which moves hidden-wide rows.
                const shared_kernels::MoeRouteParams p{
                    R * std::uint32_t(g.experts_per_token),
                    std::uint32_t(g.n_experts),
                    std::uint32_t(g.experts_per_token),
                    std::uint32_t(gemma4_moe_tile_rows(g, int(R))),
                    std::uint32_t(gemma4_moe_sorted_rows(g, int(R))),
                    std::uint32_t(g.hidden)};
                bind_const<shared_kernels::MoeRouteParams>(
                    ctx, ord,
                    d.kind == Kind::ExpertSort ? (std::uint8_t)bind::MoeRouteSort::Params
                                               : (std::uint8_t)bind::MoeRouteRows::Params,
                    p, &count);
                break;
            }
            case Kind::ExpertGeglu: {
                // The whole SORTED stack, flat. Sized off the sorted count and
                // not `rows * k`, because the sort pads every expert's run to a
                // tile and the padding rows are real rows of what this reads.
                const GegluParams p{
                    std::uint32_t(gemma4_moe_sorted_rows(g, int(R)) * g.moe_intermediate)};
                bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p, &count);
                break;
            }
            case Kind::ExpertCombine: {
                const shared_kernels::ExpertCombineParams p{
                    std::uint32_t(g.hidden), std::uint32_t(g.experts_per_token)};
                bind_const<shared_kernels::ExpertCombineParams>(
                    ctx, ord, (std::uint8_t)bind::GoExpertCombine::Params, p, &count);
                break;
            }
            // The one add this family does not fuse into a norm.
            case Kind::BranchAdd:
                break;

            // ── elementwise ──
            case Kind::GegluTanh: {
                const GegluParams p{R * std::uint32_t(L >= 0 ? g.intermediate_of(L)
                                                                : g.intermediate)};
                bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p, &count);
                break;
            }
            case Kind::PleGeglu: {
                // At M=1 the flat kernel reads a slice at a byte offset. At M>1
                // the slice strides by the whole table, so the pitches are
                // stated -- and the two params structs share slot 3, which is
                // why `paged` (the prefill's marker) picks between them.
                if (paged) {
                    const GegluStridedParams p{
                        std::uint32_t(g.per_layer_emb_dim), R,
                        std::uint32_t(g.per_layer_emb_dim),
                        std::uint32_t(g.n_layers * g.per_layer_emb_dim),
                        std::uint32_t(g.per_layer_emb_dim)};
                    bind_const<GegluStridedParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p,
                                                   &count);
                } else {
                    const GegluParams p{R * std::uint32_t(g.per_layer_emb_dim)};
                    bind_const<GegluParams>(ctx, ord, (std::uint8_t)bind::Geglu::Params, p,
                                            &count);
                }
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
                // Rows scale it: the cap covers the whole [rows, vocab] block.
                const SoftcapParams p{g.final_softcap, S * std::uint32_t(g.vocab)};
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

            case Kind::RowGather: {
                const RowGatherParams p{std::uint32_t(g.hidden), S};
                bind_const<RowGatherParams>(ctx, ord, (std::uint8_t)bind::RowGather::Params, p,
                                            &count);
                break;
            }

            default:
                break;
        }
    }
    return count;
}


/// Bind the FP16 staging buffer to every projection that reads one.
///
/// Slot 12 is the staged input the FP16 GEMM reads instead of slot 3, and slot
/// 13 is how many elements the staging pass writes. Both are bound on the
/// PROJECTION's table rather than a table of their own, because the staging
/// pass has no DAG entry to own one.
int bind_gemma4_fp16_qmm(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                         const Gemma4Geometry& g, int rows, int head_rows,
                         const SlotHandle& staging, std::vector<SlotHandle>& keep) {
    keep.clear();
    if (!staging.valid()) return 0;
    const int R = rows < 1 ? 1 : rows;
    const int S = head_rows < 1 ? R : std::min(head_rows, R);
    std::vector<std::pair<std::int32_t, SlotHandle>> counts;
    int bound = 0;
    for (const Dispatch& d : dag) {
        const int m = d.kind == Kind::LmHead ? S : R;
        if (!gemma4_fp16_qmm(g, d, m)) continue;
        const std::int32_t elems = std::int32_t(gemma4_qmm_rows(g, m)) *
                                   std::int32_t(qmv_kn(d.kind, g, d.layer).K);
        SlotHandle count;
        for (const auto& entry : counts) {
            if (entry.first == elems) {
                count = entry.second;
                break;
            }
        }
        if (!count.valid()) {
            count = ctx.create_standalone_buffer(sizeof(std::int32_t));
            if (!count.valid()) continue;
            *static_cast<std::int32_t*>(count.contents()) = elems;
            counts.push_back({elems, count});
            keep.push_back(count);
        }
        ctx.arg_bind_ordinal(d.ordinal, 12, staging);
        ctx.arg_bind_ordinal(d.ordinal, 13, count);
        ++bound;
    }
    return bound;
}

}  // namespace pie::metal::gemma4
