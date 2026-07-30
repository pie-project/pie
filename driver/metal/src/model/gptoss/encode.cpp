// GPT-OSS's step encoder: the DAG, walked with a real encoder.
//
// One choice is per-dispatch rather than per kind: the attention window, which
// follows the layer's type. Everything else is uniform — this family has no
// per-layer head width, no KV sharing and no optional stages.
//
// Barriers follow the same rule the other two encoders use: one after every
// dispatch except inside a run of mutually independent ones.

#include "encode.hpp"

#include "../../batch/decode_abi.hpp"
#include "decode_consts.hpp"

namespace pie::metal::gptoss {

namespace {

/// Dispatches that may run together: same layer, mutually independent, all
/// reading something produced before the group starts and writing distinct
/// values. An explicit list rather than something derived, for the same reason
/// the other families' are — the scratch dataflow does not model the KV pages,
/// so "independent" cannot be read off it.
int concurrency_group(Kind k) {
    switch (k) {
        case Kind::QmvQ:
        case Kind::QmvK:
        case Kind::QmvV:
            return 1;  // all three read the attention norm's output
        case Kind::RopeQ:
        case Kind::RopeK:
            return 2;  // q and k, disjoint
        case Kind::ExpertGate:
        case Kind::ExpertUp:
            return 3;  // both read the FFN norm's output and the same expert ids
        default:
            return 0;  // runs alone
    }
}

std::vector<int> concurrent_run_ends(const std::vector<Dispatch>& dag) {
    std::vector<int> ends(dag.size());
    for (std::size_t i = 0; i < dag.size(); ++i) ends[i] = static_cast<int>(i);
    std::size_t i = 0;
    while (i < dag.size()) {
        const int group = concurrency_group(dag[i].kind);
        std::size_t j = i;
        if (group != 0) {
            while (j + 1 < dag.size() && dag[j + 1].layer == dag[i].layer &&
                   concurrency_group(dag[j + 1].kind) == group) {
                ++j;
            }
        }
        for (std::size_t k = i; k <= j; ++k) ends[k] = static_cast<int>(j);
        i = j + 1;
    }
    return ends;
}

}  // namespace

std::vector<int> gptoss_run_ends(const std::vector<Dispatch>& dag) {
    return concurrent_run_ends(dag);
}

Kernel shared_kind(Kind k) {
    switch (k) {
        case Kind::EmbedGather:   return Kernel::GoEmbed;
        case Kind::AttnNorm:      return Kernel::Rms;        // input_layernorm
        case Kind::FfnNorm:       return Kernel::FfnRms;     // post_attention_layernorm
        case Kind::FinalRms:      return Kernel::FinalRms;
        case Kind::QmvQ:          return Kernel::GoQmvQ;
        case Kind::QmvK:          return Kernel::GoQmvK;
        case Kind::QmvV:          return Kernel::GoQmvV;
        case Kind::QmvO:          return Kernel::GoQmvO;
        case Kind::RopeQ:         return Kernel::Rope;
        case Kind::RopeK:         return Kernel::RopeK;
        case Kind::KvAppend:      return Kernel::KvAppend;
        case Kind::SdpaSink:      return Kernel::GoSdpaSink;
        case Kind::AttnResidual:  return Kernel::Residual;
        case Kind::RouterGemv:    return Kernel::GoRouter;
        case Kind::RouterTopK:    return Kernel::GoRouterTopK;
        case Kind::ExpertGate:    return Kernel::GoExpertGate;
        case Kind::ExpertUp:      return Kernel::GoExpertUp;
        case Kind::ExpertSwiGlu:  return Kernel::GoSwiGlu;
        case Kind::ExpertDown:    return Kernel::GoExpertDown;
        case Kind::ExpertCombine: return Kernel::GoExpertCombine;
        case Kind::FfnResidual:   return Kernel::LayerOut;
        case Kind::LmHead:        return Kernel::GoLmHead;
        case Kind::Argmax:        return Kernel::Argmax;
    }
    return Kernel::Argmax;
}

Pso pso_for(const Dispatch& d, const DecodeStepPsos& base, const GptOssPsos& go) {
    switch (d.kind) {
        // Every projection here is K=2880 -- a whole number of quantization
        // groups but not of any reduction block -- so all of them run the
        // tail-handling matvec, and all but the head are biased.
        case Kind::QmvQ: case Kind::QmvK: case Kind::QmvV: case Kind::QmvO:
            return go.qmv_tail_bias;
        case Kind::LmHead:
            return go.qmv_tail;
        case Kind::RouterGemv:
            return go.qmv_u8_bias;
        case Kind::ExpertGate: case Kind::ExpertUp: case Kind::ExpertDown:
            return go.qmv_routed_bias;
        case Kind::RouterTopK:    return go.router_topk;
        case Kind::ExpertSwiGlu:  return go.swiglu;
        case Kind::ExpertCombine: return go.expert_combine;
        case Kind::SdpaSink:      return go.sdpa_sink;
        case Kind::RopeQ: case Kind::RopeK: return go.rope_freqs;
        // The shared table only has entries for qwen3.5's kinds, so the norms
        // and the residual adds map onto those.
        case Kind::AttnNorm:      return base[Kernel::Rms];
        case Kind::FfnNorm:       return base[Kernel::Rms];
        case Kind::FinalRms:      return base[Kernel::FinalRms];
        case Kind::EmbedGather:   return base[Kernel::EmbedGather];
        case Kind::KvAppend:      return base[Kernel::KvAppend];
        case Kind::AttnResidual:  return base[Kernel::Residual];
        case Kind::FfnResidual:   return base[Kernel::Residual];
        case Kind::Argmax:        return base[Kernel::Argmax];
    }
    return Pso{};
}

void launch_shape(const Dispatch& d, const GptOssGeometry& g, Grid& grid, Threadgroup& tg) {
    if (const KN kn = qmv_kn(d.kind, g); kn.N != 0) {
        if (d.kind == Kind::RouterGemv) {
            qmv_u8_dispatch(kn.N, grid, tg);
            return;
        }
        // A routed projection gets one grid plane per selected expert; a dense
        // one gets a single plane. Same kernel, same shape otherwise.
        const bool routed = d.kind == Kind::ExpertGate || d.kind == Kind::ExpertUp ||
                            d.kind == Kind::ExpertDown;
        qmv_dispatch(kn.N, routed ? g.experts_per_token : 1, grid, tg);
        return;
    }

    switch (d.kind) {
        case Kind::EmbedGather:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        case Kind::AttnNorm: case Kind::FfnNorm: case Kind::FinalRms: {
            const int threads = (g.hidden + 3) / 4;
            grid = Grid{std::uint32_t(threads), 1, 1};
            tg = Threadgroup{std::uint32_t(threads), 1, 1};
            return;
        }
        case Kind::RopeQ:
            grid = Grid{std::uint32_t(g.head_dim / 2), std::uint32_t(g.n_q_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim / 2), 1, 1};
            return;
        case Kind::RopeK:
            grid = Grid{std::uint32_t(g.head_dim / 2), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim / 2), 1, 1};
            return;
        case Kind::KvAppend:
            grid = Grid{std::uint32_t(g.head_dim), std::uint32_t(g.n_kv_heads), 1};
            tg = Threadgroup{std::uint32_t(g.head_dim), 1, 1};
            return;
        case Kind::SdpaSink:
            sdpa_sink_dispatch(g.n_q_heads, grid, tg);
            return;
        case Kind::RouterTopK:
            router_topk_dispatch(g.n_experts, grid, tg);
            return;
        case Kind::ExpertSwiGlu:
            elementwise_dispatch(g.experts_per_token * g.intermediate, grid, tg);
            return;
        case Kind::ExpertCombine:
        case Kind::AttnResidual:
        case Kind::FfnResidual:
            elementwise_dispatch(g.hidden, grid, tg);
            return;
        case Kind::Argmax:
            grid = Grid{1024, 1, 1};
            tg = Threadgroup{1024, 1, 1};
            return;
        default:
            grid = Grid{1, 1, 1};
            tg = Threadgroup{1, 1, 1};
            return;
    }
}

Pso pso_for_paged(const Dispatch& d, const DecodeStepPsos& base, const MultiBatchPsos& mb,
                  const GptOssPsos& go) {
    switch (d.kind) {
        case Kind::KvAppend:  return mb.kv_append_paged;
        case Kind::SdpaSink:  return go.sdpa_sink_paged;
        default:              return pso_for(d, base, go);
    }
}

void encode_gptoss_step_paged(StepEncoder& se, const std::vector<Dispatch>& dag,
                              const GptOssGeometry& g, const DecodeStepPsos& base,
                              const MultiBatchPsos& mb, const GptOssPsos& go,
                              int ordinal_base) {
    // The same walk and the same shapes as the contiguous encoder: only the KV
    // layout changed, and a layout is not a launch geometry. Both attention
    // kernels are one threadgroup per query head either way.
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for_paged(d, base, mb, go));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

void encode_gptoss_step(StepEncoder& se, const std::vector<Dispatch>& dag,
                        const GptOssGeometry& g, const DecodeStepPsos& base,
                        const GptOssPsos& go, int ordinal_base) {
    const std::vector<int> run_ends = concurrent_run_ends(dag);
    for (std::size_t i = 0; i < dag.size(); ++i) {
        const Dispatch& d = dag[i];
        Grid grid;
        Threadgroup tg;
        launch_shape(d, g, grid, tg);
        se.set_pso(pso_for(d, base, go));
        se.set_argtable_ordinal(ordinal_base + d.ordinal);
        se.dispatch(grid, tg);
        if (i + 1 >= dag.size() || run_ends[i] == static_cast<int>(i)) se.barrier();
    }
}

}  // namespace pie::metal::gptoss
