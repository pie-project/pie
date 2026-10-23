use super::*;
use poem_ir::GluAct;

/// `act . w^T`; a weight whose precision quantizes its activations has
/// them quantized first.
pub fn matmul(act: &Value, w: &Weight) -> Value {
    assert!(
        w.act.is_some() || w.dtype.fp8_blocks().is_none(),
        "`{}` is stored block-fp8, which a W8A8 product reads; declare it at fp8b128a8, fp8b32a8g32 or mxfp8a8g32",
        w.name,
    );
    let r = act.rec();
    let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
    match w.act {
        None => r.push(
            Linear::Matmul {
                act: act.id(),
                w: r.weight(w),
                y: y.id(),
                input_scales: None,
                scales: None,
                quant: None,
                logits: false,
            },
            &[act],
        ),
        Some(quant) => r.push(
            Linear::Matmul {
                act: act.id(),
                w: r.weight(w),
                quant: Some(quant),
                input_scales: input_scales(r, w, quant),
                scales: matches!(
                    w.dtype,
                    Dtype::Fp8b128
                        | Dtype::Fp8b32
                        | Dtype::Mxfp8
                        | Dtype::E4m3tile128
                        | Dtype::E4m3g128
                )
                .then(|| r.weight_plane(w, poem_ir::SCALES)),
                y: y.id(),
                logits: false,
            },
            &[act],
        ),
    }
    y
}

/// The static global a format quantizes `w`'s activations under, the
/// weight's own `.input_scales` plane.
fn input_scales(r: &Recorder, w: &Weight, quant: ActQuant) -> Option<ValueId> {
    quant
        .global()
        .then(|| r.weight_plane(w, poem_ir::INPUT_SCALES))
}

/// `w` is read by `op`, which multiplies activations as they are.
fn unquantized(w: &Weight, op: &str) {
    assert!(
        w.act.is_none() && w.dtype.fp8_blocks().is_none(),
        "`{}` quantizes its activations, and {op} reads them at their own dtype; \
         declare it at its stored dtype",
        w.name,
    );
}

/// `act . w^T` landing f32: a router that scores at f32 (a release's
/// `moe_router_dtype`), its logits never rounded to the activation's dtype.
pub fn matmul_f32(act: &Value, w: &Weight) -> Value {
    unquantized(w, "an f32 projection");
    let r = act.rec();
    let y = r.fresh(tensor(act.rows(), w.dim(0), Dtype::F32));
    r.push(
        Linear::Matmul {
            act: act.id(),
            w: r.weight(w),
            y: y.id(),
            input_scales: None,
            scales: None,
            quant: None,
            logits: false,
        },
        &[act],
    );
    y
}

pub fn lm_head(act: &Value, w: &Weight) -> Value {
    unquantized(w, "a head");
    let r = act.rec();
    let y = r.fresh(tensor(act.rows(), w.dim(0), act.dtype()));
    r.push(
        Linear::Matmul {
            act: act.id(),
            w: r.weight(w),
            y: y.id(),
            input_scales: None,
            scales: None,
            quant: None,
            logits: true,
        },
        &[act],
    );
    y
}

pub fn swiglu(packed: &Value, intermediate: u32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::Glu {
            packed: Some(packed.id()),
            gate: None,
            up: None,
            weights: None,
            intermediate,
            act: GluAct::Swiglu,
            limit: None,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn swiglu_clamp(packed: &Value, intermediate: u32, limit: f32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::Glu {
            packed: Some(packed.id()),
            gate: None,
            up: None,
            weights: None,
            intermediate,
            act: GluAct::Swiglu,
            limit: Some(limit),
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn swiglu_clamp_alpha(packed: &Value, intermediate: u32, limit: f32, alpha: f32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::Glu {
            packed: Some(packed.id()),
            gate: None,
            up: None,
            weights: None,
            intermediate,
            act: GluAct::SwigluAlpha { alpha },
            limit: Some(limit),
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn swiglu_clamp_split(gate: &Value, up: &Value, limit: f32) -> Value {
    let r = gate.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Linear::Glu {
            packed: None,
            gate: Some(gate.id()),
            up: Some(up.id()),
            weights: None,
            intermediate: u32::try_from(gate.width()).expect("a gate plane is narrower than 2^32"),
            act: GluAct::Swiglu,
            limit: Some(limit),
            y: y.id(),
        },
        &[gate, up],
    );
    y
}

/// `swiglu_clamp_split` of routed rows, each times its route's weight
/// before it rounds.
pub fn swiglu_clamp_split_routed(gate: &Value, up: &Value, weights: &Value, limit: f32) -> Value {
    let r = gate.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Linear::Glu {
            packed: None,
            gate: Some(gate.id()),
            up: Some(up.id()),
            weights: Some(weights.id()),
            intermediate: u32::try_from(gate.width()).expect("a gate plane is narrower than 2^32"),
            act: GluAct::Swiglu,
            limit: Some(limit),
            y: y.id(),
        },
        &[gate, up, weights],
    );
    y
}

/// `swiglu_clamp` of routed rows (`packed` each row's gate columns,
/// then its up columns), each times its route's weight before it rounds.
pub fn swiglu_clamp_routed(
    packed: &Value,
    intermediate: u32,
    weights: &Value,
    limit: f32,
) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::Glu {
            packed: Some(packed.id()),
            gate: None,
            up: None,
            weights: Some(weights.id()),
            intermediate,
            act: GluAct::Swiglu,
            limit: Some(limit),
            y: y.id(),
        },
        &[packed, weights],
    );
    y
}

pub fn geglu_tanh(gate: &Value, up: &Value) -> Value {
    let r = gate.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Linear::Glu {
            packed: None,
            gate: Some(gate.id()),
            up: Some(up.id()),
            weights: None,
            intermediate: u32::try_from(gate.width()).expect("a gate plane is narrower than 2^32"),
            act: GluAct::GegluTanh,
            limit: None,
            y: y.id(),
        },
        &[gate, up],
    );
    y
}

pub fn gelu_tanh(x: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Linear::GeluTanh {
            x: x.id(),
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn relu2(x: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Linear::Relu2 {
            x: x.id(),
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn geglu_tanh_packed(packed: &Value, intermediate: u32) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::Glu {
            packed: Some(packed.id()),
            gate: None,
            up: None,
            weights: None,
            intermediate,
            act: GluAct::GegluTanh,
            limit: None,
            y: y.id(),
        },
        &[packed],
    );
    y
}

pub fn situ(packed: &Value, intermediate: u32, beta: f32, up_cap: Option<f32>) -> Value {
    let r = packed.rec();
    let y = r.fresh(tensor(packed.rows(), intermediate, packed.dtype()));
    r.push(
        Linear::Glu {
            packed: Some(packed.id()),
            gate: None,
            up: None,
            weights: None,
            intermediate,
            act: GluAct::Situ { beta, up_cap },
            limit: None,
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
    unquantized(w, "a relative bias");
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
    unquantized(w, "a grouped matmul");
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
    unquantized(bank, "a dense bank's select");
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
    unquantized(bank, "a biased select");
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

/// A quantized bank's select; a bank whose precision quantizes its
/// activations has each route's quantized first.
pub fn moe_matmul_select_quant(x: &Value, bank: &Weight, routes: &Value, top_k: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(Dim::TokensTimes(top_k), bank.dim(1), x.dtype()));
    match bank.act {
        None => r.push(
            Linear::MoeMatmulSelectQuant {
                x: x.id(),
                bank: r.weight(bank),
                routes: routes.id(),
                y: y.id(),
            },
            &[x, routes],
        ),
        Some(quant) => r.push(
            Linear::MoeMatmulSelectActQuant {
                x: x.id(),
                bank: r.weight(bank),
                routes: routes.id(),
                quant,
                input_scales: input_scales(r, bank, quant),
                y: y.id(),
            },
            &[x, routes],
        ),
    }
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

/// Each token's weighted routed rows summed, plus its `shared` row, rounded
/// once.
pub fn moe_sum_add(routed: &Value, shared: &Value) -> Value {
    let r = routed.rec();
    let y = r.fresh(tensor(Dim::Tokens, routed.width(), routed.dtype()));
    r.push(
        Linear::MoeSumAdd {
            routed: routed.id(),
            shared: shared.id(),
            y: y.id(),
        },
        &[routed, shared],
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
