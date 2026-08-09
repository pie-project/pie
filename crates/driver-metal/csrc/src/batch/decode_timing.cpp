// decode_timing.cpp — beta's GPU-exec attribution analysis (device-agnostic).
// See decode_timing.hpp. Validated mechanism: files/icb-probes/mtl4_tsattrib.mm.

#include "decode_timing.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace pie::metal {

bool kernel_ablated(Kernel k) {
    // Parsed once into a fixed table: this is asked per dispatch per fire, and
    // a strcmp walk over an env string there would be measuring the ablation.
    static const std::array<bool, kKernelKindCount> table = [] {
        std::array<bool, kKernelKindCount> t{};
        t.fill(false);
        const char* e = std::getenv("PIE_METAL_ABLATE");
        if (e == nullptr || *e == '\0') return t;
        const std::string spec(e);
        for (int i = 0; i < kKernelKindCount; ++i) {
            const char* n = kernel_name(static_cast<Kernel>(i));
            if (n == nullptr) continue;
            // Whole-token match, so `rms` does not also ablate `ffn_rms`.
            std::size_t at = 0;
            while ((at = spec.find(n, at)) != std::string::npos) {
                const bool lok = at == 0 || spec[at - 1] == ',';
                const std::size_t end = at + std::strlen(n);
                const bool rok = end == spec.size() || spec[end] == ',';
                if (lok && rok) { t[std::size_t(i)] = true; break; }
                at = end;
            }
        }
        std::fprintf(stderr,
                     "[ablate] PIE_METAL_ABLATE=%s -- these kinds are NOT "
                     "DISPATCHED. The tokens are wrong on purpose; only the "
                     "wall clock means anything.\n", e);
        // A token that matched nothing skips nothing, and the run then
        // reports the baseline while looking armed -- the banner above still
        // prints. That has cost a whole session's worth of "trace said 46%,
        // ablation says 0%" readings, all of them no-ops, because this takes
        // a kernel KIND and the dispatch trace prints a pipeline HOST NAME:
        // paste `affine_qmv_routed_bfloat16_gs_64_b_4` in here and it matches
        // no kind at all. Say so, loudly, and list what it does take.
        for (std::size_t at = 0; at <= spec.size();) {
            const std::size_t end = std::min(spec.find(',', at), spec.size());
            const std::string token = spec.substr(at, end - at);
            at = end + 1;
            if (token.empty()) continue;
            bool matched = false;
            for (int i = 0; i < kKernelKindCount && !matched; ++i) {
                const char* n = kernel_name(static_cast<Kernel>(i));
                matched = n != nullptr && token == n;
            }
            if (matched) continue;
            std::fprintf(stderr,
                         "[ablate] '%s' IS NOT A KERNEL KIND -- it ablates "
                         "NOTHING and this run will report the baseline. This "
                         "takes kind names, not the pipeline host names the "
                         "dispatch trace prints. Known kinds:\n[ablate]  ",
                         token.c_str());
            for (int i = 0; i < kKernelKindCount; ++i) {
                const char* n = kernel_name(static_cast<Kernel>(i));
                // The enum is sparse -- unassigned ordinals all answer
                // "unknown" -- and printing that seventy times buries the list.
                if (n != nullptr && std::strcmp(n, "unknown") != 0) {
                    std::fprintf(stderr, "%s ", n);
                }
            }
            std::fprintf(stderr, "\n");
        }
        return t;
    }();
    const int i = static_cast<int>(k);
    return i >= 0 && i < kKernelKindCount && table[std::size_t(i)];
}

const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather: return "embed_gather";
        case Kernel::Rms:         return "rms";
        case Kernel::QmvIn:       return "qmv_in";
        case Kernel::QmvInZ:      return "qmv_in_z";
        case Kernel::GdnInA:      return "gdn_in_a";
        case Kernel::GdnInB:      return "gdn_in_b";
        case Kernel::GdnPrep:     return "gdn_prep";
        case Kernel::GdnCore:     return "gdn_core";
        case Kernel::GatedRms:    return "gated_rms";
        case Kernel::QmvOut:      return "qmv_out";
        case Kernel::Residual:    return "residual";
        case Kernel::QmvQ:        return "qmv_q";
        case Kernel::QSplit:      return "q_split";
        case Kernel::QmvK:        return "qmv_k";
        case Kernel::QmvV:        return "qmv_v";
        case Kernel::QNorm:       return "q_norm";
        case Kernel::KNorm:       return "k_norm";
        case Kernel::Rope:        return "rope";
        case Kernel::RopeK:       return "rope_k";
        case Kernel::KvAppend:    return "kv_append";
        case Kernel::Sdpa:        return "sdpa";
        case Kernel::AttnGate:    return "gate";
        case Kernel::QmvO:        return "qmv_o";
        case Kernel::FfnRms:      return "ffn_rms";
        case Kernel::QmvGate:     return "qmv_gate";
        case Kernel::QmvUp:       return "qmv_up";
        case Kernel::SiluMul:     return "silu_mul";
        case Kernel::QmvDown:     return "qmv_down";
        case Kernel::LayerOut:    return "layer_out";
        case Kernel::FinalRms:    return "final_rms";
        case Kernel::QmvLmHead:   return "qmv_lm_head";
        case Kernel::Argmax:      return "argmax";
        case Kernel::KvAppendPaged: return "kv_append_paged";
        case Kernel::SdpaPaged: return "sdpa_paged";
        case Kernel::GdnCoreSlotted: return "gdn_core_slotted";
        case Kernel::GdnPrepSlotted: return "gdn_prep_slotted";
        // The mixture. Unnamed until now, which meant every routed kind
        // reported as `unknown` -- so the attribution report said nothing about
        // the half of a MoE fire that IS the mixture, and `PIE_METAL_ABLATE`
        // could not single any of them out.
        case Kernel::LlRouter:        return "ll_router";
        case Kernel::LlExpertGate:    return "ll_expert_gate";
        case Kernel::LlExpertUp:      return "ll_expert_up";
        case Kernel::LlExpertDown:    return "ll_expert_down";
        case Kernel::LlExpertSiluMul: return "ll_expert_silu_mul";
        case Kernel::LlMoeSort:       return "ll_moe_sort";
        case Kernel::LlMoeGather:     return "ll_moe_gather";
        case Kernel::LlMoeCombine:    return "ll_moe_combine";
        case Kernel::LlSharedGate:      return "ll_shared_gate";
        case Kernel::LlSharedUp:        return "ll_shared_up";
        case Kernel::LlSharedDown:      return "ll_shared_down";
        case Kernel::LlSharedGateProj:  return "ll_shared_gate_proj";
        case Kernel::LlSharedCombine:   return "ll_shared_combine";
        // The remaining half of the enum, which reported as `unknown` -- 50 of
        // 99 kinds had no entry here, so the attribution report was blind to
        // gemma4's mixture and PLE, all of gpt-oss, and both untied-embedding
        // kinds, and `PIE_METAL_ABLATE` could not name any of them. Generated
        // from the enum rather than written by hand, so the mapping is the
        // identity and cannot drift.
        case Kernel::G4AttnPostNorm:         return "g4_attn_post_norm";
        case Kernel::G4FfnPreNorm:           return "g4_ffn_pre_norm";
        case Kernel::G4FfnPostNorm:          return "g4_ffn_post_norm";
        case Kernel::G4VNorm:                return "g4_v_norm";
        case Kernel::G4Geglu:                return "g4_geglu";
        case Kernel::G4LayerScalar:          return "g4_layer_scalar";
        case Kernel::G4Softcap:              return "g4_softcap";
        case Kernel::G4RowGather:            return "g4_row_gather";
        case Kernel::G4SdpaSliding:          return "g4_sdpa_sliding";
        case Kernel::G4PleTokenGather:       return "g4_ple_token_gather";
        case Kernel::G4PleProjGemv:          return "g4_ple_proj_gemv";
        case Kernel::G4PleProjNorm:          return "g4_ple_proj_norm";
        case Kernel::G4PleCombine:           return "g4_ple_combine";
        case Kernel::G4PleGateGemv:          return "g4_ple_gate_gemv";
        case Kernel::G4PleGeglu:             return "g4_ple_geglu";
        case Kernel::G4PleProjLayerGemv:     return "g4_ple_proj_layer_gemv";
        case Kernel::G4PleNorm:              return "g4_ple_norm";
        case Kernel::G4PleResidual:          return "g4_ple_residual";
        case Kernel::G4AttnPostResidual:     return "g4_attn_post_residual";
        case Kernel::G4FfnPostResidual:      return "g4_ffn_post_residual";
        case Kernel::G4PleResidualScaled:    return "g4_ple_residual_scaled";
        case Kernel::EmbedUntied:            return "embed_untied";
        case Kernel::LmHeadUntied:           return "lm_head_untied";
        case Kernel::GoQmvQ:                 return "go_qmv_q";
        case Kernel::GoQmvK:                 return "go_qmv_k";
        case Kernel::GoQmvV:                 return "go_qmv_v";
        case Kernel::GoQmvO:                 return "go_qmv_o";
        case Kernel::GoSdpaSink:             return "go_sdpa_sink";
        case Kernel::GoRouter:               return "go_router";
        case Kernel::GoExpertGate:           return "go_expert_gate";
        case Kernel::GoExpertUp:             return "go_expert_up";
        case Kernel::GoExpertDown:           return "go_expert_down";
        case Kernel::GoRouterTopK:           return "go_router_top_k";
        case Kernel::GoSwiGlu:               return "go_swi_glu";
        case Kernel::GoExpertCombine:        return "go_expert_combine";
        case Kernel::G4Router:               return "g4_router";
        case Kernel::G4RouterNorm:           return "g4_router_norm";
        case Kernel::G4RouterTopK:           return "g4_router_top_k";
        case Kernel::G4MoeNorm:              return "g4_moe_norm";
        case Kernel::G4DenseBranchNorm:      return "g4_dense_branch_norm";
        case Kernel::G4MoeBranchNorm:        return "g4_moe_branch_norm";
        case Kernel::G4ExpertGate:           return "g4_expert_gate";
        case Kernel::G4ExpertUp:             return "g4_expert_up";
        case Kernel::G4ExpertDown:           return "g4_expert_down";
        case Kernel::G4ExpertGeglu:          return "g4_expert_geglu";
        case Kernel::G4MoeSort:              return "g4_moe_sort";
        case Kernel::G4MoeGather:            return "g4_moe_gather";
        case Kernel::G4ExpertCombine:        return "g4_expert_combine";
        case Kernel::G4BranchAdd:            return "g4_branch_add";
        default: break;
    }
    return "unknown";
}

StepAttribution attribute_step(const std::vector<Dispatch>& dag,
                               const uint64_t* boundary_ticks,
                               size_t n_boundaries,
                               double ns_per_tick) {
    StepAttribution a;
    if (boundary_ticks == nullptr || n_boundaries != dag.size() + 1) {
        // boundary/DAG mismatch -> can't attribute; caller checks `valid`.
        return a;
    }
    a.per_dispatch.reserve(dag.size());
    for (size_t i = 0; i < dag.size(); ++i) {
        const uint64_t t0 = boundary_ticks[i];
        const uint64_t t1 = boundary_ticks[i + 1];
        // Monotonic guard: a non-increasing pair (clock wrap / re-order) -> 0, not negative.
        const double ms = (t1 > t0) ? double(t1 - t0) * ns_per_tick / 1e6 : 0.0;

        DispatchAttribution da;
        da.ordinal = dag[i].ordinal;
        da.kind    = dag[i].kind;
        da.layer   = dag[i].layer;
        da.gpu_ms  = ms;
        a.per_dispatch.push_back(da);

        const int ki = static_cast<int>(dag[i].kind);
        a.by_kind[ki]    += ms;
        a.count_kind[ki] += 1;
        a.total_gpu_ms   += ms;
    }
    a.valid = true;
    return a;
}

void print_attribution(const StepAttribution& a, const char* title, int top_n, FILE* out) {
    if (!out) out = stdout;
    if (!a.valid) {
        std::fprintf(out, "[attribution] %s: INVALID (boundary/DAG count mismatch)\n",
                     title ? title : "");
        return;
    }
    std::fprintf(out, "\n==== GPU-exec attribution: %s ====\n", title ? title : "");
    std::fprintf(out, "step gpu-exec total = %.4f ms  (%zu dispatches)\n",
                 a.total_gpu_ms, a.per_dispatch.size());

    // ── per-kernel-kind rollup, sorted DESC by total ms (the fuse/cut targets) ──
    struct KindRow { Kernel k; double ms; int n; };
    std::vector<KindRow> rows;
    for (int i = 0; i < kKernelKindCount; ++i) {
        if (a.count_kind[i] > 0)
            rows.push_back({static_cast<Kernel>(i), a.by_kind[i], a.count_kind[i]});
    }
    std::sort(rows.begin(), rows.end(),
              [](const KindRow& x, const KindRow& y) { return x.ms > y.ms; });

    std::fprintf(out, "\n-- per kernel-kind (sorted by total gpu-exec, the optimization targets) --\n");
    std::fprintf(out, "  %-14s %8s %5s %9s %7s\n", "kind", "total_ms", "n", "ms/disp", "%step");
    for (const auto& r : rows) {
        const double pct = a.total_gpu_ms > 0 ? 100.0 * r.ms / a.total_gpu_ms : 0.0;
        std::fprintf(out, "  %-14s %8.4f %5d %9.5f %6.1f%%\n",
                     kernel_name(r.k), r.ms, r.n, r.ms / r.n, pct);
    }

    // ── top-N hottest individual dispatches ──
    std::vector<const DispatchAttribution*> hot;
    hot.reserve(a.per_dispatch.size());
    for (const auto& d : a.per_dispatch) hot.push_back(&d);
    std::sort(hot.begin(), hot.end(),
              [](const DispatchAttribution* x, const DispatchAttribution* y) {
                  return x->gpu_ms > y->gpu_ms;
              });
    const int n = std::min<int>(top_n, static_cast<int>(hot.size()));
    std::fprintf(out, "\n-- top %d hottest dispatches --\n", n);
    std::fprintf(out, "  %5s %-14s %6s %9s\n", "ord", "kind", "layer", "gpu_ms");
    for (int i = 0; i < n; ++i) {
        const auto* d = hot[i];
        std::fprintf(out, "  %5d %-14s %6d %9.5f\n",
                     d->ordinal, kernel_name(d->kind), d->layer, d->gpu_ms);
    }
    std::fprintf(out, "==== end attribution ====\n\n");
}

}  // namespace pie::metal
