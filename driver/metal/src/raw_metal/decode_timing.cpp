// decode_timing.cpp — beta's GPU-exec attribution analysis (device-agnostic).
// See decode_timing.hpp. Validated mechanism: files/icb-probes/mtl4_tsattrib.mm.

#include "decode_timing.hpp"

#include <algorithm>
#include <cstring>

namespace pie_metal_driver::raw_metal {

const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::EmbedGather: return "embed_gather";
        case Kernel::Rms:         return "rms";
        case Kernel::QmvIn:       return "qmv_in";
        case Kernel::QmvInZ:      return "qmv_in_z";
        case Kernel::GdnInA:      return "gdn_in_a";
        case Kernel::GdnInB:      return "gdn_in_b";
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
        case Kernel::AttnGate:    return "attn_gate";
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

}  // namespace pie_metal_driver::raw_metal
