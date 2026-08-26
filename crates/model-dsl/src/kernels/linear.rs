//! The `Linear` family: the plain matmuls, the fused MLP activations, and the
//! MoE routing/expert path.

use super::*;

pub fn matmul(act: &Value, w: &Weight) -> Value {
    let r = act.rec();
    let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
    r.push(
        Linear::Matmul {
            act: act.id(),
            w: r.weight(w),
            y: y.id(),
        },
        &[act],
    );
    y
}

pub fn lm_head(act: &Value, w: &Weight) -> Value {
    let r = act.rec();
    let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
    r.push(
        Linear::LmHead {
            act: act.id(),
            w: r.weight(w),
            y: y.id(),
        },
        &[act],
    );
    y
}

pub fn attention_landing(act: &Value, w: &Weight, layer: u32) -> Value {
    let r = act.rec();
    let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
    r.push(
        Linear::AttentionLanding {
            act: act.id(),
            w: r.weight(w),
            layer,
            y: y.id(),
        },
        &[act],
    );
    y
}

pub fn mlp_swiglu(packed: &Value, intermediate: u32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::MlpSwiglu {
            packed: packed.id(),
            intermediate,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn mlp_swiglu_clamp(packed: &Value, intermediate: u32, limit: f32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::MlpSwigluClamp {
            packed: packed.id(),
            intermediate,
            limit,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn mlp_swiglu_clamp_alpha(packed: &Value, intermediate: u32, limit: f32, alpha: f32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::MlpSwigluClampAlpha {
            packed: packed.id(),
            intermediate,
            limit,
            alpha,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn mlp_geglu_tanh(gate: &Value, up: &Value) -> Value {
    let r = gate.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Linear::MlpGegluTanh {
            gate: gate.id(),
            up: up.id(),
            y: y.id(),
        },
        &[gate, up],
    );
    y
}

pub fn mlp_geglu_tanh_packed(packed: &Value, intermediate: u32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::MlpGegluTanhPacked {
            packed: packed.id(),
            intermediate,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn mlp_situ(packed: &Value, intermediate: u32, beta: f32, up_cap: Option<f32>) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::MlpSitu {
            packed: packed.id(),
            intermediate,
            beta,
            up_cap,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn moe_topk_softmax(logits: &Value, experts: u32, top_k: u32) -> (Value, Value) {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    r.push(
        Linear::MoeTopkSoftmax {
            logits: logits.id(),
            experts,
            top_k,
            routes: routes.id(),
            weights: weights.id(),
        },
        &[logits],
    );
    (routes, weights)
}

pub fn moe_topk_sigmoid(
    logits: &Value,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
) -> (Value, Value) {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    r.push(
        Linear::MoeTopkSigmoid {
            logits: logits.id(),
            experts,
            top_k,
            renormalize,
            scaling,
            routes: routes.id(),
            weights: weights.id(),
        },
        &[logits],
    );
    (routes, weights)
}

pub fn moe_topk_sqrt_softplus(
    logits: &Value,
    bias: &Weight,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
) -> (Value, Value) {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    r.push(
        Linear::MoeTopkSqrtSoftplus {
            logits: logits.id(),
            bias: r.weight(bias),
            experts,
            top_k,
            renormalize,
            scaling,
            routes: routes.id(),
            weights: weights.id(),
        },
        &[logits],
    );
    (routes, weights)
}

/// The routed rows are `tokens × top_k` — the fold of the old
/// `per(routes)` rule, so `top_k` rides along as a wrapper argument.
pub fn moe_matmul_select(x: &Value, bank: &Weight, routes: &Value, top_k: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(Dim::TokensTimes(top_k), bank.dim(1), x.dtype()));
    r.push(
        Linear::MoeMatmulSelect {
            x: x.id(),
            bank: r.weight(bank),
            routes: routes.id(),
            y: y.id(),
        },
        &[x, routes],
    );
    y
}

pub fn moe_matmul_select_bias(
    x: &Value,
    bank: &Weight,
    bias: &Weight,
    routes: &Value,
    top_k: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(Dim::TokensTimes(top_k), bank.dim(1), x.dtype()));
    r.push(
        Linear::MoeMatmulSelectBias {
            x: x.id(),
            bank: r.weight(bank),
            bias: r.weight(bias),
            routes: routes.id(),
            y: y.id(),
        },
        &[x, routes],
    );
    y
}

pub fn moe_weighted_sum(routed: &Value, weights: &Value) -> Value {
    let r = routed.rec();
    let y = r.fresh(tensor(Dim::Tokens, routed.width(), routed.dtype()));
    r.push(
        Linear::MoeWeightedSum {
            routed: routed.id(),
            weights: weights.id(),
            y: y.id(),
        },
        &[routed, weights],
    );
    y
}

pub fn moe_sigmoid_gate_add(routed: &Value, shared: &Value, gate: &Value) -> Value {
    let r = routed.rec();
    let y = r.fresh(routed.ty().clone());
    r.push(
        Linear::MoeSigmoidGateAdd {
            routed: routed.id(),
            shared: shared.id(),
            gate: gate.id(),
            y: y.id(),
        },
        &[routed, shared, gate],
    );
    y
}
