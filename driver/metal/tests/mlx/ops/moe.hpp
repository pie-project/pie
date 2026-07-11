#pragma once

// Sparse Mixture-of-Experts FFN, ported from driver/cuda's MoE path
// (kernels/topk_softmax + moe_dispatch + flashinfer_moe) onto MLX. Used by
// gemma4 (parallel dense+MoE block) and qwen3.6 (routed MoE + shared expert).
//
// The routed experts are evaluated with MLX `gather_mm` (indexed batched
// matmul) -- the same primitive mlx-lm uses for `SwitchGLU` -- so no custom
// Metal kernel is needed and it runs on GPU and CPU alike.

#include <optional>

#include "ops/tensor.hpp"

namespace pie_metal_driver::ops {

// Router gating style. Softmax = Mixtral / Gemma-style (softmax over logits).
// Sigmoid = DeepSeek / Nemotron-style (per-expert sigmoid, optional additive
// correction bias used only for top-k *selection*).
enum class MoeGate { Softmax, Sigmoid };

// Expert gated-MLP activation. Silu = SwiGLU (qwen3.6); Gelu = GeGLU with the
// tanh approximation (gemma4).
enum class MoeAct { Silu, Gelu };

struct MoeParams {
    int num_experts = 0;        // E: total routed experts
    int experts_per_token = 0;  // K: top-K experts selected per token
    int n_group = 1;            // expert groups for grouped routing (DeepSeek)
    int topk_group = 1;         // groups kept per token when n_group > 1
    MoeGate gate = MoeGate::Softmax;
    MoeAct act = MoeAct::Silu;  // expert activation (SwiGLU / GeGLU-tanh)
    bool norm_topk = true;      // renormalize the top-K routing weights to sum 1
    float routed_scaling = 1.0f;  // routed_scaling_factor applied to weights
};

// Routing decision: pick the top-K experts per token and their weights.
struct MoeRouting {
    Tensor indices;  // [N, K] int32 -- selected expert ids
    Tensor weights;  // [N, K] float32 -- combine weights
};

// Compute the routing from raw router logits `router_logits` [N, E].
//   correction_bias : optional [E] float32, added to the (sigmoid) score for
//                     top-k *selection* only (DeepSeek aux-loss-free bias).
//   per_expert_scale: optional [E] -- multiplies the final weight of each
//                     selected expert (Gemma-4 26B-A4B per-expert gain).
MoeRouting moe_route(const Tensor& router_logits,
                     const MoeParams& params,
                     const std::optional<Tensor>& correction_bias = std::nullopt,
                     const std::optional<Tensor>& per_expert_scale = std::nullopt);

// Evaluate the selected experts (gated MLP) and combine.
//   x          : [N, hidden]
//   routing    : from moe_route (indices [N,K], weights [N,K])
//   gate_up_w  : [E, 2*inter, hidden] -- FUSED gate/up (rows [0:inter)=gate,
//                [inter:2*inter)=up), HF row-major (w @ x.T convention)
//   down_w     : [E, hidden, inter]
//   act        : SwiGLU (Silu) or GeGLU-tanh (Gelu)
// returns [N, hidden].
Tensor moe_experts(const Tensor& x,
                   const MoeRouting& routing,
                   const Tensor& gate_up_w,
                   const Tensor& down_w,
                   MoeAct act);

// Full routed MoE FFN: route then evaluate. `router_logits` [N, E] are
// produced by the caller (x @ gate.T). Expert weights fused as above.
Tensor moe_ffn(const Tensor& x,
               const Tensor& router_logits,
               const Tensor& gate_up_w,
               const Tensor& down_w,
               const MoeParams& params,
               const std::optional<Tensor>& correction_bias = std::nullopt,
               const std::optional<Tensor>& per_expert_scale = std::nullopt);

// Sigmoid-gated shared expert (qwen3.6): a dense SwiGLU FFN whose output is
// scaled by sigmoid(x @ gate_proj.T). Added to the routed MoE output by the
// caller.
//   x          : [N, hidden]
//   gate_w/up_w: [inter, hidden]   down_w: [hidden, inter]
//   shared_gate: [1, hidden]  -- the scalar sigmoid gate projection
Tensor shared_expert(const Tensor& x,
                     const Tensor& gate_w,
                     const Tensor& up_w,
                     const Tensor& down_w,
                     const Tensor& shared_gate);

}  // namespace pie_metal_driver::ops
