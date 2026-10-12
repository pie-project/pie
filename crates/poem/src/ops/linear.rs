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

pub fn mlp_swiglu_clamp_split(gate: &Value, up: &Value, limit: f32) -> Value {
    let r = gate.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Linear::MlpSwigluClampSplit {
            gate: gate.id(),
            up: up.id(),
            limit,
            y: y.id(),
        },
        &[gate, up],
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

pub fn mlp_gelu_tanh(x: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Linear::MlpGeluTanh {
            x: x.id(),
            y: y.id(),
        },
        &[x],
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

pub fn moe_topk_softmax_scaled(
    logits: &Value,
    scale: &Weight,
    experts: u32,
    top_k: u32,
) -> (Value, Value) {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    r.push(
        Linear::MoeTopkSoftmaxScaled {
            logits: logits.id(),
            scale: r.weight(scale),
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
            bias: None,
            experts,
            top_k,
            renormalize,
            scaling,
            routes: routes.id(),
            weights: weights.id(),
            hint: None,
        },
        &[logits],
    );
    (routes, weights)
}

pub fn rel_bias(x: &Value, w: &Weight, heads: u32, d_rel: u32, extent: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(
        x.rows(),
        u64::from(heads) * u64::from(extent),
        Dtype::F32,
    ));
    r.push(
        Linear::RelBias {
            x: x.id(),
            w: r.weight(w),
            heads,
            d_rel,
            extent,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn moe_topk_sigmoid_sink(
    logits: &Value,
    bias: &Weight,
    scale: Option<&Weight>,
    experts: u32,
    top_k: u32,
    sink: u32,
    scaling: f32,
) -> (Value, Value) {
    let r = logits.rec();
    let fan = top_k + sink;
    let routes = r.fresh(tensor(Dim::Tokens, fan, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, fan, Dtype::F32));
    r.push(
        Linear::MoeTopkSigmoidSink {
            logits: logits.id(),
            bias: Some(r.weight(bias)),
            scale: scale.map(|s| r.weight(s)),
            experts,
            top_k,
            sink,
            scaling,
            routes: routes.id(),
            weights: weights.id(),
        },
        &[logits],
    );
    (routes, weights)
}

pub fn moe_topk_sigmoid_biased(
    logits: &Value,
    bias: &Weight,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
) -> (Value, Value) {
    moe_topk_sigmoid_biased_hinted(logits, bias, experts, top_k, renormalize, scaling, None)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sigmoid_biased_hinted(
    logits: &Value,
    bias: &Weight,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    hint: Option<&Value>,
) -> (Value, Value) {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    let mut ins: Vec<&Value> = vec![logits];
    ins.extend(hint);
    r.push(
        Linear::MoeTopkSigmoid {
            logits: logits.id(),
            bias: Some(r.weight(bias)),
            experts,
            top_k,
            renormalize,
            scaling,
            routes: routes.id(),
            weights: weights.id(),
            hint: hint.map(Value::id),
        },
        &ins,
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
    moe_topk_sqrt_softplus_hinted(logits, bias, experts, top_k, renormalize, scaling, None)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sqrt_softplus_hinted(
    logits: &Value,
    bias: &Weight,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
    hint: Option<&Value>,
) -> (Value, Value) {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    let mut inputs = vec![logits];
    inputs.extend(hint);
    r.push(
        Linear::MoeTopkSqrtSoftplus {
            logits: logits.id(),
            bias: r.weight(bias),
            experts,
            top_k,
            renormalize,
            scaling,
            hint: hint.map(Value::id),
            routes: routes.id(),
            weights: weights.id(),
        },
        &inputs,
    );
    (routes, weights)
}

pub fn moe_predict_route(logits: &Value, bias: &Weight, experts: u32, top_k: u32) -> Value {
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    r.push(
        Linear::MoePredictRoute {
            logits: logits.id(),
            bias: r.weight(bias),
            experts,
            top_k,
            routes: routes.id(),
            weights: weights.id(),
        },
        &[logits],
    );
    routes
}

#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route(
    ids: &Value,
    tid2eid: &Weight,
    logits: &Value,
    vocab: u32,
    experts: u32,
    top_k: u32,
    renormalize: bool,
    scaling: f32,
) -> (Value, Value) {
    let r = ids.rec();
    let routes = r.fresh(tensor(Dim::Tokens, top_k, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, top_k, Dtype::F32));
    r.push(
        Linear::MoeHashRoute {
            ids: ids.id(),
            tid2eid: r.weight(tid2eid),
            logits: logits.id(),
            vocab,
            experts,
            top_k,
            renormalize,
            scaling,
            routes: routes.id(),
            weights: weights.id(),
        },
        &[ids, logits],
    );
    (routes, weights)
}

pub fn group_routes(x: &Value, groups: u32) -> Value {
    let r = x.rec();
    let routes = r.fresh(tensor(Dim::Tokens, groups, Dtype::I32));
    r.push(
        Linear::GroupRoutes {
            groups,
            routes: routes.id(),
        },
        &[x],
    );
    routes
}

pub fn matmul_grouped(x: &Value, w: &Weight, routes: &Value, groups: u32) -> Value {
    let r = x.rec();
    assert!(
        groups > 0
            && w.dim(0).is_multiple_of(u64::from(groups))
            && x.width().is_multiple_of(u64::from(groups)),
        "`{}` lands {} rows over a {}-wide row, which {groups} groups do not divide",
        w.name,
        w.dim(0),
        x.width(),
    );
    assert_eq!(
        w.dim(1),
        x.width() / u64::from(groups),
        "`{}` contracts over {} and a group's slice of the row is {}",
        w.name,
        w.dim(1),
        x.width() / u64::from(groups),
    );
    let y = r.fresh(tensor(Dim::Tokens, w.dim(0), x.dtype()));
    r.push(
        Linear::MatmulGrouped {
            x: x.id(),
            w: r.weight(w),
            routes: routes.id(),
            groups,
            y: y.id(),
        },
        &[x, routes],
    );
    y
}
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

pub fn moe_matmul_select_quant(x: &Value, bank: &Weight, routes: &Value, top_k: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(Dim::TokensTimes(top_k), bank.dim(1), x.dtype()));
    r.push(
        Linear::MoeMatmulSelectQuant {
            x: x.id(),
            bank: r.weight(bank),
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

pub fn moe_bias_sum(x: &Value, bias: &Weight, routes: &Value, weights: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Linear::MoeBiasSum {
            x: x.id(),
            bias: r.weight(bias),
            routes: routes.id(),
            weights: weights.id(),
            y: y.id(),
        },
        &[x, routes, weights],
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

pub fn lora_correct(
    x: &Value,
    bank_a: &Weight,
    bank_b: &Weight,
    routes: &Value,
    y: &Value,
) -> Value {
    let r = x.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Linear::LoraCorrect {
            x: x.id(),
            bank_a: r.weight(bank_a),
            bank_b: r.weight(bank_b),
            routes: routes.id(),
            y: y.id(),
            y_out: y_out.id(),
        },
        &[x, routes, y],
    );
    y_out.everywhere()
}
