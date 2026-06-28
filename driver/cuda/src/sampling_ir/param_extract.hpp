#pragma once

// #12 phase-1 — extract a recognized standard program's dedicated-kernel params
// from its host-submit inputs, so the DedicatedKernel ladder (FlashInfer /
// sample_temp / argmax) gets the params it needs on a *program* fire (where the
// legacy per-slot `pi.sample_*` are empty — schema.rs: a fire uses EITHER slots
// OR a program). The params relocated into the program; this recovers them.
//
// Binding layout — the carrier's submit_input keys are the program INPUT ORDINAL
// (binding-template slot position), NOT the SDK `TensorKey`. The logits intrinsic
// occupies ordinal 0 (`Binding::Logits`, consumes no `TensorKey`), so the keyed
// scalar params (SDK `TensorKey` 0/1) sit at ordinals 1/2 — the +1 logits offset.
// `param_extract` matches `SubmitInput::key` == the carrier ordinal (golf-probed,
// HW-verified `extract(T=0.8, top_p=0.9)`):
//   T (temperature)      = submit_inputs[ordinal 1]   (every RNG kind)
//   filter (top_p/min_p) = submit_inputs[ordinal 2]
//   k (TopK/TopKTopP)    = baked `RankLe(k)` immediate → foxtrot op-shape (PHASE-2)
//   seed                 = AMBIENT (`pi.sample_seed`, per-row) — NOT a submit input;
//                          the same seed the IR uses, so FlashInfer(philox, seed)
//                          == pre-migration by routing-identity.
//
// The executor writes these into the SAME `per_slot_{temp,top_p,min_p,top_k}`
// buffers a legacy slot-fire fills, so the existing flag-set + `infer_sampler_kind`
// + `dispatch_target` + dedicated ladder run UNCHANGED (behaviour-preserving by
// construction). `extracted_params_agree` is echo's free pre-HW guard.

#include <cstdint>
#include <cstring>
#include <span>

#include "sampling_ir/pie_standard_samplers.h"  // StandardSamplerKind
#include "sampling_ir/runtime.hpp"               // SubmitInput
#include "sampling_ir/sampler_dispatch.hpp"      // infer_sampler_kind, params_from_slot

namespace pie_cuda_driver::sampling_ir {

// Read the f32 scalar carried by a submit input (host LE bytes); `fallback` if
// the pointer is null / too short.
inline float read_submit_f32(const SubmitInput* si, float fallback) {
    if (si == nullptr || si->data == nullptr || si->len_bytes < sizeof(float))
        return fallback;
    float v;
    std::memcpy(&v, si->data, sizeof(float));
    return v;
}

// Dedicated-kernel params for a recognized kind. Defaults = "unset" (temp 1,
// top_p 1 = no nucleus, min_p 0 = no floor, top_k 0 = no rank cut) so a missing
// input is inert.
struct DedicatedParams {
    float        temp  = 1.0f;
    float        top_p = 1.0f;
    float        min_p = 0.0f;
    std::int32_t top_k = 0;  // k-bearing → from op-shape (RankLe), set by the caller
};

// Extract dedicated-kernel params for `kind` by reading the host-submit scalars
// in ASCENDING input-ordinal order — first ordinal = temperature (temp-scale
// precedes softmax), second = filter (top_p/min_p). Reading by ORDER (not a
// hardcoded ordinal 1/2) is immune to the logits-intrinsic offset (the two index
// spaces — TensorKey vs ordinal — differ by exactly that, the #12 recurrence trap).
// k (TopK/TopKTopP) is NOT a submit input — the caller sets `top_k` from the
// op-shape RankLe immediate (`extract_rank_le_k`).
inline DedicatedParams extract_dedicated_params(StandardSamplerKind kind,
                                                std::span<const SubmitInput> ins) {
    DedicatedParams p;
    if (kind == StandardSamplerKind::Argmax) {
        // Bare ReduceArgmax — no inputs. Synthesize greedy so the existing
        // flag-set (`all_rows_greedy`) routes it to `launch_argmax_bf16`.
        p.temp = 0.0f;
        return p;
    }
    // Find the two lowest-ordinal submit scalars (temperature first, filter next).
    const SubmitInput* first_ord = nullptr;   // lowest ordinal  → temperature
    const SubmitInput* second_ord = nullptr;  // next ordinal    → filter
    for (const SubmitInput& si : ins) {
        if (si.data == nullptr || si.len_bytes < sizeof(float)) continue;
        if (first_ord == nullptr || si.key < first_ord->key) {
            second_ord = first_ord;
            first_ord = &si;
        } else if (second_ord == nullptr || si.key < second_ord->key) {
            second_ord = &si;
        }
    }
    p.temp = read_submit_f32(first_ord, 1.0f);  // temperature = first ordinal
    switch (kind) {
        case StandardSamplerKind::TopP:
        case StandardSamplerKind::TopKTopP:
            p.top_p = read_submit_f32(second_ord, 1.0f);  // filter = second ordinal
            break;
        case StandardSamplerKind::MinP:
            p.min_p = read_submit_f32(second_ord, 0.0f);  // filter = second ordinal
            break;
        default:  // Temperature (T only); TopK (T only; k from op-shape, top_p unset)
            break;
    }
    return p;
}

// echo's free agreement guard (#7-step-1 analog): the extracted params MUST
// re-classify (via the PARAM recognizer, with the same #7 normalizations) to the
// kind the HASH recognizer returned. A mis-extract (wrong key → wrong param) →
// reclassify → caught pre-HW. Reuses the exact `infer_sampler_kind`/
// `params_from_slot` the #7 cutover verified.
inline bool extracted_params_agree(StandardSamplerKind hash_kind,
                                   const DedicatedParams& p, std::int32_t vocab) {
    return infer_sampler_kind(
               params_from_slot(p.temp, p.top_k, p.top_p, p.min_p, vocab)) == hash_kind;
}

}  // namespace pie_cuda_driver::sampling_ir
