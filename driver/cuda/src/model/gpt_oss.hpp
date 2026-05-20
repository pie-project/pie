#pragma once

// GPT-OSS (OpenAI's open-source reference). Architectural pieces and the
// way each is realised in this driver:
//
//   * Sparse MoE per layer. Router with bias → top-K softmax →
//     per-expert SwiGLU MLP. Same forward path as Mixtral; the
//     differences (router_bias, expert biases, swiglu_limit clip,
//     +1 on up) are handled by the optional fields on
//     `MixtralLayerWeights` / `MixtralExpertWeights`.
//   * Per-head attention sink. Each attention layer learns one sink
//     scalar per head. We extend the softmax denominator post-hoc by
//     asking flashinfer to write `lse` (log-sum-exp) per (token, head)
//     and then applying `o *= sigmoid(lse - sink_h)` via
//     `kernels/attn_sink`. flashinfer's `params.lse` slot is plumbed
//     for this; no template recompile needed.
//   * Alternating sliding/full attention. `HfConfig.layer_types`
//     drives `LlamaLikeForwardCfg::per_layer_window_left`; the
//     mixtral forward already consumes that table per layer.
//   * Q/K/V/O biases. Loaded from the checkpoint and added via
//     `launch_add_bias_bf16` after each projection.
//   * MXFP4 expert weights. Stored as packed E2M1 nibbles plus per-32
//     E8M0 block scales in the checkpoint. The target-aware load planner can
//     keep them resident as QuantPacked MoE tensors for routed runtime
//     dequant, or lower through an eager BF16 fallback. A true native MXFP4
//     GEMM backend is represented as a separate target policy.
//
// Returns a `MixtralWeights` so the runtime dispatch can reuse
// `mixtral_forward_paged` directly.

#include "model/loaded_model.hpp"
#include "model/mixtral.hpp"

namespace pie_cuda_driver::model {

MixtralWeights bind_gpt_oss(const LoadedModel& engine);

}  // namespace pie_cuda_driver::model
