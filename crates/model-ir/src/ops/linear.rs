use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

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
    MlpSwigluClampSplit {
        gate: ValueId,
        up: ValueId,
        limit: f32,
        y: ValueId,
    },
    MlpGegluTanh {
        gate: ValueId,
        up: ValueId,
        y: ValueId,
    },
    MlpGeluTanh {
        x: ValueId,
        y: ValueId,
    },
    MlpGegluTanhPacked {
        packed: ValueId,
        intermediate: u32,
        y: ValueId,
    },
    MatmulGeglu {
        act: ValueId,
        w: ValueId,
        intermediate: u32,
        packed: ValueId,
        y: ValueId,
    },
    LmHeadSoftcap {
        act: ValueId,
        w: ValueId,
        cap: f32,
        y: ValueId,
        y_out: ValueId,
    },
    MlpSitu {
        packed: ValueId,
        intermediate: u32,
        beta: f32,
        up_cap: Option<f32>,
        y: ValueId,
    },
    RelBias {
        x: ValueId,
        w: ValueId,
        heads: u32,
        d_rel: u32,
        extent: u32,
        y: ValueId,
    },
    MoeTopkSoftmax {
        logits: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    MoeTopkSoftmaxScaled {
        logits: ValueId,
        scale: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    MoeTopkSigmoid {
        logits: ValueId,
        bias: Option<ValueId>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
        hint: Option<ValueId>,
    },
    MoeTopkSigmoidSink {
        logits: ValueId,
        bias: Option<ValueId>,
        scale: Option<ValueId>,
        experts: u32,
        top_k: u32,
        sink: u32,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
    },
    MoeTopkSqrtSoftplus {
        logits: ValueId,
        bias: ValueId,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        hint: Option<ValueId>,
        routes: ValueId,
        weights: ValueId,
    },
    MoePredictRoute {
        logits: ValueId,
        bias: ValueId,
        experts: u32,
        top_k: u32,
        routes: ValueId,
        weights: ValueId,
    },
    MoeHashRoute {
        ids: ValueId,
        tid2eid: ValueId,
        logits: ValueId,
        vocab: u32,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: ValueId,
        weights: ValueId,
    },
    GroupRoutes {
        groups: u32,
        routes: ValueId,
    },
    MatmulGrouped {
        x: ValueId,
        w: ValueId,
        routes: ValueId,
        groups: u32,
        y: ValueId,
    },
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
    MoeMatmulSelectQuant {
        x: ValueId,
        bank: ValueId,
        routes: ValueId,
        y: ValueId,
    },
    MoeWeightedSum {
        routed: ValueId,
        weights: ValueId,
        y: ValueId,
    },
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
    LoraCorrect {
        x: ValueId,
        bank_a: ValueId,
        bank_b: ValueId,
        routes: ValueId,
        y: ValueId,
        y_out: ValueId,
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
            Self::MlpSwigluClampSplit { gate, up, .. } => sink.extend([*gate, *up]),
            Self::MlpGegluTanh { gate, up, .. } => sink.extend([*gate, *up]),
            Self::MlpGeluTanh { x, .. } => sink.push(*x),
            Self::MlpGegluTanhPacked { packed, .. } => sink.push(*packed),
            Self::MatmulGeglu { act, w, .. } | Self::LmHeadSoftcap { act, w, .. } => {
                sink.extend([*act, *w]);
            }
            Self::MlpSitu { packed, .. } => sink.push(*packed),
            Self::MoeTopkSoftmax { logits, .. } => sink.push(*logits),
            Self::MoeTopkSoftmaxScaled { logits, scale, .. } => sink.extend([*logits, *scale]),
            Self::MoeTopkSigmoid {
                logits, bias, hint, ..
            } => {
                sink.push(*logits);
                sink.extend(*bias);
                sink.extend(*hint);
            }
            Self::MoeTopkSqrtSoftplus {
                logits, bias, hint, ..
            } => {
                sink.extend([*logits, *bias]);
                sink.extend(*hint);
            }
            Self::MoeTopkSigmoidSink {
                logits,
                bias,
                scale,
                ..
            } => {
                sink.push(*logits);
                sink.extend(*bias);
                sink.extend(*scale);
            }
            Self::RelBias { x, w, .. } => sink.extend([*x, *w]),
            Self::MoePredictRoute { logits, bias, .. } => sink.extend([*logits, *bias]),
            Self::MoeHashRoute {
                ids,
                tid2eid,
                logits,
                ..
            } => sink.extend([*ids, *tid2eid, *logits]),
            Self::GroupRoutes { .. } => {}
            Self::MatmulGrouped { x, w, routes, .. } => sink.extend([*x, *w, *routes]),
            Self::MoeMatmulSelect {
                x, bank, routes, ..
            } => sink.extend([*x, *bank, *routes]),
            Self::MoeMatmulSelectBias {
                x,
                bank,
                bias,
                routes,
                ..
            } => {
                sink.extend([*x, *bank, *bias, *routes]);
            }
            Self::MoeMatmulSelectQuant {
                x, bank, routes, ..
            } => sink.extend([*x, *bank, *routes]),
            Self::MoeWeightedSum {
                routed, weights, ..
            } => sink.extend([*routed, *weights]),
            Self::MoeBiasSum {
                x,
                bias,
                routes,
                weights,
                ..
            } => {
                sink.extend([*x, *bias, *routes, *weights]);
            }
            Self::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                ..
            } => {
                sink.extend([*routed, *shared, *gate]);
            }
            Self::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y,
                ..
            } => {
                sink.extend([*x, *bank_a, *bank_b, *routes, *y]);
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
            Self::MlpSwigluClampSplit { y, .. } => sink.push(*y),
            Self::MlpGegluTanh { y, .. } => sink.push(*y),
            Self::MlpGeluTanh { y, .. } => sink.push(*y),
            Self::MlpGegluTanhPacked { y, .. } => sink.push(*y),
            Self::MatmulGeglu { packed, y, .. } => sink.extend([*packed, *y]),
            Self::LmHeadSoftcap { y, y_out, .. } => sink.extend([*y, *y_out]),
            Self::MlpSitu { y, .. } => sink.push(*y),
            Self::MoeTopkSoftmax {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::MoeTopkSoftmaxScaled {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::MoeTopkSigmoid {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::MoeTopkSqrtSoftplus {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::MoeTopkSigmoidSink {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::RelBias { y, .. } => sink.push(*y),
            Self::MoePredictRoute {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::MoeHashRoute {
                routes, weights, ..
            } => sink.extend([*routes, *weights]),
            Self::GroupRoutes { routes, .. } => sink.push(*routes),
            Self::MatmulGrouped { y, .. } => sink.push(*y),
            Self::MoeMatmulSelect { y, .. } => sink.push(*y),
            Self::MoeMatmulSelectBias { y, .. } => sink.push(*y),
            Self::MoeMatmulSelectQuant { y, .. } => sink.push(*y),
            Self::MoeWeightedSum { y, .. } => sink.push(*y),
            Self::MoeBiasSum { y, .. } => sink.push(*y),
            Self::MoeSigmoidGateAdd { y, .. } => sink.push(*y),
            Self::LoraCorrect { y_out, .. } => sink.push(*y_out),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::LoraCorrect { y, y_out, .. } => sink.push((*y_out, *y)),
            Self::LmHeadSoftcap { y, y_out, .. } => sink.push((*y_out, *y)),
            Self::Matmul { .. }
            | Self::LmHead { .. }
            | Self::MlpSwiglu { .. }
            | Self::MlpSwigluClamp { .. }
            | Self::MlpSwigluClampAlpha { .. }
            | Self::MlpSwigluClampSplit { .. }
            | Self::MlpGegluTanh { .. }
            | Self::MlpGeluTanh { .. }
            | Self::MlpGegluTanhPacked { .. }
            | Self::MatmulGeglu { .. }
            | Self::MlpSitu { .. }
            | Self::MoeTopkSoftmax { .. }
            | Self::MoeTopkSoftmaxScaled { .. }
            | Self::MoeTopkSigmoid { .. }
            | Self::MoeTopkSigmoidSink { .. }
            | Self::RelBias { .. }
            | Self::MoeTopkSqrtSoftplus { .. }
            | Self::MoePredictRoute { .. }
            | Self::MoeHashRoute { .. }
            | Self::GroupRoutes { .. }
            | Self::MatmulGrouped { .. }
            | Self::MoeMatmulSelect { .. }
            | Self::MoeMatmulSelectBias { .. }
            | Self::MoeMatmulSelectQuant { .. }
            | Self::MoeWeightedSum { .. }
            | Self::MoeBiasSum { .. }
            | Self::MoeSigmoidGateAdd { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Matmul { .. } => "linear.matmul",
            Self::LmHead { .. } => "linear.lm_head",
            Self::MlpSwiglu { .. } => "linear.mlp_swiglu",
            Self::MlpSwigluClamp { .. } => "linear.mlp_swiglu_clamp",
            Self::MlpSwigluClampAlpha { .. } => "linear.mlp_swiglu_clamp_alpha",
            Self::MlpSwigluClampSplit { .. } => "linear.mlp_swiglu_clamp_split",
            Self::MlpGegluTanh { .. } => "linear.mlp_geglu_tanh",
            Self::MlpGeluTanh { .. } => "linear.mlp_gelu_tanh",
            Self::MlpGegluTanhPacked { .. } => "linear.mlp_geglu_tanh_packed",
            Self::MatmulGeglu { .. } => "linear.matmul_geglu",
            Self::LmHeadSoftcap { .. } => "linear.lm_head_softcap",
            Self::MlpSitu { .. } => "linear.mlp_situ",
            Self::MoeTopkSoftmax { .. } => "linear.moe_topk_softmax",
            Self::MoeTopkSoftmaxScaled { .. } => "linear.moe_topk_softmax_scaled",
            Self::MoeTopkSigmoid { .. } => "linear.moe_topk_sigmoid",
            Self::MoeTopkSigmoidSink { .. } => "linear.moe_topk_sigmoid_sink",
            Self::RelBias { .. } => "linear.rel_bias",
            Self::MoeTopkSqrtSoftplus { .. } => "linear.moe_topk_sqrt_softplus",
            Self::MoePredictRoute { .. } => "linear.moe_predict_route",
            Self::MoeHashRoute { .. } => "linear.moe_hash_route",
            Self::GroupRoutes { .. } => "linear.group_routes",
            Self::MatmulGrouped { .. } => "linear.matmul_grouped",
            Self::MoeMatmulSelect { .. } => "linear.moe_matmul_select",
            Self::MoeMatmulSelectBias { .. } => "linear.moe_matmul_select_bias",
            Self::MoeMatmulSelectQuant { .. } => "linear.moe_matmul_select_quant",
            Self::MoeWeightedSum { .. } => "linear.moe_weighted_sum",
            Self::MoeBiasSum { .. } => "linear.moe_bias_sum",
            Self::MoeSigmoidGateAdd { .. } => "linear.moe_sigmoid_gate_add",
            Self::LoraCorrect { .. } => "linear.lora_correct",
        }
    }
}
