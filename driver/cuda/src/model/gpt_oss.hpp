#pragma once

// GPT-OSS (OpenAI's open-source reference). Architectural pieces:
//
//   * Sparse-MoE block per layer: router with bias → top-K softmax →
//     per-expert SwiGLU MLP. Same shape as Mixtral, except expert
//     weights ship as MXFP4 (E2M1 codes + per-32-element E8M0 scales).
//   * Attention sinks: a per-head learnable scalar `sinks[h]` is added
//     to the softmax denominator, equivalent to a virtual KV slot with
//     logit `sinks[h]` and zero V contribution. flashinfer's
//     `DefaultAttention` variant doesn't support this directly — needs
//     either (a) a fresh `AttentionVariant` subclass with a custom
//     softmax denominator, or (b) injection of the sink as a real KV
//     entry (one extra page row per request) with a logit override.
//   * Alternating attention pattern: even layers run sliding-window
//     attention, odd layers run full causal attention. The forward
//     loop has to dispatch flashinfer with the right `window_left`
//     per layer.
//
// Status: NOT YET WIRED into the runtime. The building blocks live in:
//   - `kernels/dequant_fp4.{hpp,cu}` — MXFP4 → bf16 (one-shot, used
//     during `bind_gpt_oss` to stage expert weights for the existing
//     mixtral_forward_paged path).
//   - `kernels/topk_softmax.{hpp,cu}` + `moe_dispatch.{hpp,cu}` — same
//     routing/scatter primitives Mixtral uses; the GPT-OSS adapter only
//     needs to thread its router-with-bias through them.
//   - `model/mixtral.{hpp,cpp}` — sparse-MoE forward; will need to
//     accept a `LlamaLikeForwardCfg::sliding_window` toggle per layer
//     and a per-head sink-scalar tensor.
//
// Memory caveat: a load-time MXFP4 → bf16 dequant of 8 × 7B expert
// weights at 4-bit storage to 16-bit bf16 inflates expert memory ~4×.
// On a 80GB GPU this fits 8×7B but not larger. The proper path is a
// fused MXFP4-aware GEMM (CUTLASS grouped-gemm with per-block dequant
// in shared mem); flashinfer ships that as a JIT module which we
// don't currently consume — bringing it into the C++ build is the
// next milestone for this model.

#include "engine.hpp"
#include "model/mixtral.hpp"

namespace pie_cuda_driver::model {

// Stub: throws "not yet implemented". Header retained so callers can
// see the planned shape of the API.
MixtralWeights bind_gpt_oss(Engine& engine);

}  // namespace pie_cuda_driver::model
