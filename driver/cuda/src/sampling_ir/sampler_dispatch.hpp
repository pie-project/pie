#pragma once

// sampler_dispatch.hpp — the recognizer (#8) decision core.
//
// The de-hardwiring keystone: instead of a fixed `sampler_type → hardwired
// kernel` coupling, the executor recognizes a request's canonical sampler kind
// and dispatches it to the best execution path per the §2f perf scorecard.
// Three targets:
//   * DedicatedKernel — the hand-tuned kernel beats the general IR lowering
//     (argmax → launch_argmax_bf16 ~1.5×; top-k/top-p → FlashInfer). Keep it.
//   * BakedIR — the IR is a genuine win (temperature: ~2× faster, token-exact)
//     → de-hardwire to the driver-baked `program_for(kind, V)`.
//   * CustomJIT — inferlet-authored programs (mirostat/grammar/EDSL) → JIT.
//
// The kind is recognized from the per-row legacy params (param-inference — zero
// wire change; the explicit canonical-kind tag from foxtrot/golf's SDK/WIT
// swaps in later as just a different dispatch *input*). The dispatch table is
// the scorecard and is intentionally data-driven: a future faster IR top-p
// flips top-p → BakedIR with no executor change.

#include <cstdint>

#include "sampling_ir/pie_standard_samplers.h"

namespace pie_cuda_driver::sampling_ir {

enum class DispatchTarget : std::uint8_t {
    DedicatedKernel,  // legacy hand-tuned kernel (argmax / FlashInfer top-k,p)
    BakedIR,          // driver-baked IR program (temperature / min-p)
    CustomJIT,        // JIT-compiled inferlet-authored program
};

// One row's legacy sampler params (the recognizer's param-inference input).
// `top_k == 0` and `top_p >= 1.0` and `min_p <= 0` mean "unset".
struct SamplerParams {
    float         temperature = 1.0f;
    std::uint32_t top_k       = 0;
    float         top_p       = 1.0f;
    float         min_p       = 0.0f;
};

// Recognize the canonical standard-sampler kind from per-row params.
// Greedy (T<=0) is argmax regardless of a top_k=1; otherwise the active filter
// params select the kind (top-k/top-p/min-p), falling through to plain
// temperature when none are set.
inline StandardSamplerKind infer_sampler_kind(const SamplerParams& p) {
    if (p.temperature <= 0.0f) {
        return StandardSamplerKind::Argmax;
    }
    const bool has_top_k = p.top_k > 0u;
    const bool has_top_p = p.top_p > 0.0f && p.top_p < 1.0f;
    const bool has_min_p = p.min_p > 0.0f;
    if (has_top_k && has_top_p) return StandardSamplerKind::TopKTopP;
    if (has_top_k)              return StandardSamplerKind::TopK;
    if (has_top_p)              return StandardSamplerKind::TopP;
    if (has_min_p)             return StandardSamplerKind::MinP;
    return StandardSamplerKind::Temperature;
}

// The dispatch table = the de-hardwiring scorecard. Configurable: changing a
// row here (e.g. a future faster IR top-p) re-routes with no executor change.
//
// §2f perf (delta de1bb65b, canonical bytecode, bench≡production, ir/sample_temp)
// + manager's scorecard ruling — route each kind to its measured-best target:
//   * temperature 0.52–0.68× → IR ~2× FASTER, token-exact → BakedIR (genuine win).
//   * min-p       ~1.08–1.12× → IR marginally SLOWER (the 0.97× was a degenerate
//     min_p=1.0-vs-0.1 mismatch) → DedicatedKernel (`sample_temp`) to preserve the
//     ~10%. De-hardwiring is UNAFFECTED: the fixed `sampler_type→kernel` enum is
//     still gone, the recognizer still decides — `sample_temp` simply survives as
//     min-p's dispatch *target*. Future option (deferred, one line here, zero
//     executor change): if a faster min-p IR lands, flip min-p → BakedIR.
inline DispatchTarget dispatch_target(StandardSamplerKind kind) {
    switch (kind) {
        // temp: IR ~2× faster + token-exact → the de-hardwiring win.
        case StandardSamplerKind::Temperature:
            return DispatchTarget::BakedIR;
        // Hand-tuned kernel wins (or ties) → keep it as the dispatch target.
        case StandardSamplerKind::MinP:        // sample_temp (~1.1× vs IR)
        case StandardSamplerKind::Argmax:      // launch_argmax_bf16 (~1.5× faster)
        case StandardSamplerKind::TopK:        // FlashInfer
        case StandardSamplerKind::TopP:        // FlashInfer
        case StandardSamplerKind::TopKTopP:    // FlashInfer (joint k+p, neutral)
            return DispatchTarget::DedicatedKernel;
    }
    return DispatchTarget::DedicatedKernel;
}

// Adapt the executor's per-slot sampler params (per_slot_temp / per_slot_top_k /
// per_slot_top_p / per_slot_min_p in build_sample_plan) to the recognizer's
// input. The executor normalizes `top_k == 0` ("unset") to `vocab_size` so the
// dedicated kernels see a concrete bound — but the recognizer's `has_top_k` must
// treat `top_k >= vocab_size` (and <= 0) as UNSET, else every request misclassifies
// as TopK/TopKTopP. This adapter is the single seam the executor and the
// runtime-drift test share so that mapping can never silently regress.
inline SamplerParams params_from_slot(float temperature, std::int32_t top_k_slot,
                                      float top_p, float min_p,
                                      std::int32_t vocab_size) {
    SamplerParams sp;
    sp.temperature = temperature;
    sp.top_k = (top_k_slot <= 0 || top_k_slot >= vocab_size)
                   ? 0u
                   : static_cast<std::uint32_t>(top_k_slot);
    sp.top_p = top_p;
    sp.min_p = min_p;
    return sp;
}

}  // namespace pie_cuda_driver::sampling_ir
