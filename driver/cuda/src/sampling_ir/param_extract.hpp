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
//   k (TopK/TopKTopP)    = submit_inputs[LAST ordinal] as U32 (#25: k is a host-
//                          submit value-id like top_p/min_p, NOT a baked immediate;
//                          TopK k @ 2nd ordinal, TopKTopP k @ 3rd ordinal)
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

// Read the u32 scalar carried by a submit input (host LE bytes); `fallback` if
// the pointer is null / too short. The #25 top-k `k` is a U32 host-submit scalar.
inline std::uint32_t read_submit_u32(const SubmitInput* si, std::uint32_t fallback) {
    if (si == nullptr || si->data == nullptr || si->len_bytes < sizeof(std::uint32_t))
        return fallback;
    std::uint32_t v;
    std::memcpy(&v, si->data, sizeof(std::uint32_t));
    return v;
}

// Dedicated-kernel params for a recognized kind. Defaults = "unset" (temp 1,
// top_p 1 = no nucleus, min_p 0 = no floor, top_k 0 = no rank cut) so a missing
// input is inert.
struct DedicatedParams {
    float        temp  = 1.0f;
    float        top_p = 1.0f;
    float        min_p = 0.0f;
    std::int32_t top_k = 0;  // #25: from the host-submit binding (last ordinal, U32)
};

// Extract dedicated-kernel params for `kind` by reading the host-submit scalars
// in ASCENDING input-ordinal order — first ordinal = temperature (temp-scale
// precedes softmax), second = filter (top_p/min_p) or k (TopK), third = k
// (TopKTopP). Reading by ORDER (not a hardcoded ordinal 1/2/3) is immune to the
// logits-intrinsic offset (the two index spaces — TensorKey vs ordinal — differ
// by exactly that, the #12 recurrence trap).
// #25: k (TopK/TopKTopP) is the LAST submit ordinal (U32), read like top_p/min_p
// — no longer a baked op-shape immediate.
inline DedicatedParams extract_dedicated_params(StandardSamplerKind kind,
                                                std::span<const SubmitInput> ins) {
    DedicatedParams p;
    if (kind == StandardSamplerKind::Argmax) {
        // Bare ReduceArgmax — no inputs. Synthesize greedy so the existing
        // flag-set (`all_rows_greedy`) routes it to `launch_argmax_bf16`.
        p.temp = 0.0f;
        return p;
    }
    // Find the lowest-ordinal submit scalars in ascending key order:
    //   ord[0] = temperature; ord[1] = filter (top_p/min_p) OR k (TopK);
    //   ord[2] = k (TopKTopP, after top_p).
    const SubmitInput* ord[3] = {nullptr, nullptr, nullptr};
    for (const SubmitInput& si : ins) {
        if (si.data == nullptr || si.len_bytes < sizeof(std::uint32_t)) continue;
        if (ord[0] == nullptr || si.key < ord[0]->key) {
            ord[2] = ord[1]; ord[1] = ord[0]; ord[0] = &si;
        } else if (ord[1] == nullptr || si.key < ord[1]->key) {
            ord[2] = ord[1]; ord[1] = &si;
        } else if (ord[2] == nullptr || si.key < ord[2]->key) {
            ord[2] = &si;
        }
    }
    p.temp = read_submit_f32(ord[0], 1.0f);  // temperature = first ordinal
    switch (kind) {
        case StandardSamplerKind::TopP:
            p.top_p = read_submit_f32(ord[1], 1.0f);  // filter @ 2nd ordinal
            break;
        case StandardSamplerKind::MinP:
            p.min_p = read_submit_f32(ord[1], 0.0f);  // filter @ 2nd ordinal
            break;
        case StandardSamplerKind::TopK:
            // #25: k @ 2nd ordinal (binding [Logits, T, k]); U32.
            p.top_k = static_cast<std::int32_t>(read_submit_u32(ord[1], 0u));
            break;
        case StandardSamplerKind::TopKTopP:
            p.top_p = read_submit_f32(ord[1], 1.0f);  // top_p @ 2nd ordinal
            // #25: k @ 3rd ordinal (binding [Logits, T, top_p, k]); U32.
            p.top_k = static_cast<std::int32_t>(read_submit_u32(ord[2], 0u));
            break;
        default:  // Temperature (T only)
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
