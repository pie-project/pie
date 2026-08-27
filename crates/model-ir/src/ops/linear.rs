use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Learned-weight channel mixing and its epilogues: the plain matmuls, the
/// fused MLP activations that consume a projection's packed output, and the MoE
/// routing and grouped-matmul ops that pick which weights a row multiplies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Linear {
    Matmul {
        act: ValueId,
        w: ValueId,
        y: ValueId,
    },
    LmHead {
        act: ValueId,
        w: ValueId,
        y: ValueId,
    },
    MlpSwiglu {
        packed: ValueId,
        intermediate: u32,
        y: ValueId,
    },
    MlpSwigluClamp {
        packed: ValueId,
        intermediate: u32,
        limit: f32,
        y: ValueId,
    },
    MlpSwigluClampAlpha {
        packed: ValueId,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: ValueId,
    },
    MlpGegluTanh {
        gate: ValueId,
        up: ValueId,
        y: ValueId,
    },
    MlpGegluTanhPacked {
        packed: ValueId,
        intermediate: u32,
        y: ValueId,
    },
    MlpSitu {
        packed: ValueId,
        intermediate: u32,
        beta: f32,
        up_cap: Option<f32>,
        y: ValueId,
    },
    MoeTopkSoftmax {
        logits: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    MoeTopkSigmoid {
        logits: ValueId,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
    },
    /// Sigmoid routing with a per-expert bias; weights pass through sqrt-softplus.
    MoeTopkSqrtSoftplus {
        logits: ValueId,
        bias: ValueId,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
    },
    /// Grouped matmul: each routed row multiplies the expert `routes` selects from `bank`.
    MoeMatmulSelect {
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    MoeMatmulSelectBias {
        x: ValueId,
        bank: ValueId,
        bias: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    /// The bias-free twin of `MoeMatmulSelectBias`: a split-plane quantized
    /// bank, nothing added inside the fold. `MoeMatmulSelect` cannot say this —
    /// its `bank` resolves as one dense handle — and the routed bias a rows-cut
    /// expert wants is said after the reduce, by `MoeBiasSum`.
    MoeMatmulSelectQuant {
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    /// Folds the top_k routed rows back to one row per token.
    MoeWeightedSum {
        routed: ValueId,
        weights: ValueId,
        y: ValueId,
    },
    /// The routed bias mixture, stated once on an already-folded activation:
    /// `y[t] = x[t] + Σ_k weights[t, k] · bias[routes[t, k]]`. A replicated
    /// bias folded into a rows-cut expert matmul would be summed once per rank
    /// by the all_reduce that follows it; routing is computed from replicated
    /// inputs, so the mixture can be said after the reduce instead, where it
    /// lands exactly once.
    MoeBiasSum {
        x: ValueId,
        bias: ValueId,
        routes: ValueId,
        weights: ValueId,
        y: ValueId,
    },
    MoeSigmoidGateAdd {
        routed: ValueId,
        shared: ValueId,
        gate: ValueId,
        y: ValueId,
    },
}

impl Operands for Linear {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Matmul { act, w, .. } => sink.extend([*act, *w]),
            Self::LmHead { act, w, .. } => sink.extend([*act, *w]),
            Self::MlpSwiglu { packed, .. } => sink.push(*packed),
            Self::MlpSwigluClamp { packed, .. } => sink.push(*packed),
            Self::MlpSwigluClampAlpha { packed, .. } => sink.push(*packed),
            Self::MlpGegluTanh { gate, up, .. } => sink.extend([*gate, *up]),
            Self::MlpGegluTanhPacked { packed, .. } => sink.push(*packed),
            Self::MlpSitu { packed, .. } => sink.push(*packed),
            Self::MoeTopkSoftmax { logits, .. } => sink.push(*logits),
            Self::MoeTopkSigmoid { logits, .. } => sink.push(*logits),
            Self::MoeTopkSqrtSoftplus { logits, bias, .. } => sink.extend([*logits, *bias]),
            Self::MoeMatmulSelect { x, bank, routes, .. } => sink.extend([*x, *bank, *routes]),
            Self::MoeMatmulSelectBias { x, bank, bias, routes, .. } => {
                sink.extend([*x, *bank, *bias, *routes]);
            }
            Self::MoeMatmulSelectQuant { x, bank, routes, .. } => sink.extend([*x, *bank, *routes]),
            Self::MoeWeightedSum { routed, weights, .. } => sink.extend([*routed, *weights]),
            Self::MoeBiasSum { x, bias, routes, weights, .. } => {
                sink.extend([*x, *bias, *routes, *weights]);
            }
            Self::MoeSigmoidGateAdd { routed, shared, gate, .. } => {
                sink.extend([*routed, *shared, *gate]);
            }
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Matmul { y, .. } => sink.push(*y),
            Self::LmHead { y, .. } => sink.push(*y),
            Self::MlpSwiglu { y, .. } => sink.push(*y),
            Self::MlpSwigluClamp { y, .. } => sink.push(*y),
            Self::MlpSwigluClampAlpha { y, .. } => sink.push(*y),
            Self::MlpGegluTanh { y, .. } => sink.push(*y),
            Self::MlpGegluTanhPacked { y, .. } => sink.push(*y),
            Self::MlpSitu { y, .. } => sink.push(*y),
            Self::MoeTopkSoftmax { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeTopkSigmoid { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeTopkSqrtSoftplus { routes, weights, .. } => sink.extend([*routes, *weights]),
            Self::MoeMatmulSelect { y, .. } => sink.push(*y),
            Self::MoeMatmulSelectBias { y, .. } => sink.push(*y),
            Self::MoeMatmulSelectQuant { y, .. } => sink.push(*y),
            Self::MoeWeightedSum { y, .. } => sink.push(*y),
            Self::MoeBiasSum { y, .. } => sink.push(*y),
            Self::MoeSigmoidGateAdd { y, .. } => sink.push(*y),
        }
    }
    fn aliases(&self, _sink: &mut Vec<(ValueId, ValueId)>) {}
    fn name(&self) -> &'static str {
        match self {
            Self::Matmul { .. } => "linear.matmul",
            Self::LmHead { .. } => "linear.lm_head",
            Self::MlpSwiglu { .. } => "linear.mlp_swiglu",
            Self::MlpSwigluClamp { .. } => "linear.mlp_swiglu_clamp",
            Self::MlpSwigluClampAlpha { .. } => "linear.mlp_swiglu_clamp_alpha",
            Self::MlpGegluTanh { .. } => "linear.mlp_geglu_tanh",
            Self::MlpGegluTanhPacked { .. } => "linear.mlp_geglu_tanh_packed",
            Self::MlpSitu { .. } => "linear.mlp_situ",
            Self::MoeTopkSoftmax { .. } => "linear.moe_topk_softmax",
            Self::MoeTopkSigmoid { .. } => "linear.moe_topk_sigmoid",
            Self::MoeTopkSqrtSoftplus { .. } => "linear.moe_topk_sqrt_softplus",
            Self::MoeMatmulSelect { .. } => "linear.moe_matmul_select",
            Self::MoeMatmulSelectBias { .. } => "linear.moe_matmul_select_bias",
            Self::MoeMatmulSelectQuant { .. } => "linear.moe_matmul_select_quant",
            Self::MoeWeightedSum { .. } => "linear.moe_weighted_sum",
            Self::MoeBiasSum { .. } => "linear.moe_bias_sum",
            Self::MoeSigmoidGateAdd { .. } => "linear.moe_sigmoid_gate_add",
        }
    }
}
