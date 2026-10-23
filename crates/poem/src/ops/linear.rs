use super::*;
use poem_ir::{GluAct, RouteScore};

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

/// A gated linear unit, `gating(gate) · up`: over `x`'s `[gate | up]` rows
/// (`intermediate` columns of each), or `x` the gate and `up` its own plane;
/// clamped at `limit` where stated, each row times its route's weight
/// before it rounds where `weights` is given.
pub fn glu(
    x: &Value,
    gating: GluAct,
    intermediate: Option<u32>,
    up: Option<&Value>,
    weights: Option<&Value>,
    limit: Option<f32>,
) -> Value {
    let r = x.rec();
    let (packed, gate, intermediate, y) = match up {
        None => {
            let intermediate =
                intermediate.expect("a packed gated unit states its intermediate width");
            let y = r.fresh(tensor(x.rows(), intermediate, x.dtype()));
            (Some(x.id()), None, intermediate, y)
        }
        Some(_) => {
            let width = u32::try_from(x.width()).expect("a gate plane is narrower than 2^32");
            assert!(
                intermediate.is_none_or(|i| i == width),
                "a gate plane {width} wide is its own intermediate width"
            );
            (None, Some(x.id()), width, r.fresh(x.ty().clone()))
        }
    };
    let mut ins = vec![x];
    ins.extend(up);
    ins.extend(weights);
    r.push(
        Linear::Glu {
            packed,
            gate,
            up: up.map(Value::id),
            weights: weights.map(Value::id),
            intermediate,
            act: gating,
            limit,
            y: y.id(),
        },
        &ins,
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

/// A token's `top_k` of `experts`, scored of `logits` as `routing` says:
/// `bias` added before the pick where given, `scale` the per-expert gain or
/// post-scale, `hint` an earlier pick to keep. A hashed route reads its
/// experts from `table` at the token's `ids`. Its routes and their weights,
/// `top_k` wide, or `top_k + sink` for a route with sink slots.
#[allow(clippy::too_many_arguments)]
pub fn moe_route(
    logits: &Value,
    experts: u32,
    top_k: u32,
    routing: RouteScore,
    bias: Option<&Weight>,
    scale: Option<&Weight>,
    hint: Option<&Value>,
    ids: Option<&Value>,
    table: Option<&Weight>,
) -> (Value, Value) {
    let hashed = matches!(routing, RouteScore::Hash { .. });
    assert!(
        hashed == ids.is_some() && hashed == table.is_some(),
        "a hashed route, and only it, reads a token's ids from a table"
    );
    let fan = match routing {
        RouteScore::SigmoidSink { sink, .. } => top_k + sink,
        _ => top_k,
    };
    let r = logits.rec();
    let routes = r.fresh(tensor(Dim::Tokens, fan, Dtype::I32));
    let weights = r.fresh(tensor(Dim::Tokens, fan, Dtype::F32));
    let mut ins: Vec<&Value> = ids.into_iter().collect();
    ins.push(logits);
    ins.extend(hint);
    r.push(
        Linear::MoeRoute {
            ids: ids.map(Value::id),
            table: table.map(|t| r.weight(t)),
            logits: logits.id(),
            bias: bias.map(|b| r.weight(b)),
            scale: scale.map(|s| r.weight(s)),
            hint: hint.map(Value::id),
            experts,
            top_k,
            score: routing,
            routes: routes.id(),
            weights: weights.id(),
        },
        &ins,
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
        Linear::MoeSelect {
            x: x.id(),
            bank: r.weight(bank),
            routes: routes.id(),
            y: y.id(),
            bias: None,
            input_scales: None,
            quantized: false,
            quant: None,
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
        Linear::MoeSelect {
            x: x.id(),
            bank: r.weight(bank),
            bias: Some(r.weight(bias)),
            routes: routes.id(),
            y: y.id(),
            input_scales: None,
            quantized: false,
            quant: None,
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
            Linear::MoeSelect {
                x: x.id(),
                bank: r.weight(bank),
                routes: routes.id(),
                y: y.id(),
                bias: None,
                input_scales: None,
                quantized: true,
                quant: None,
            },
            &[x, routes],
        ),
        Some(quant) => r.push(
            Linear::MoeSelect {
                x: x.id(),
                bank: r.weight(bank),
                routes: routes.id(),
                quant: Some(quant),
                input_scales: input_scales(r, bank, quant),
                y: y.id(),
                bias: None,
                quantized: true,
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
        Linear::MoeCombine {
            routed: routed.id(),
            weights: Some(weights.id()),
            y: y.id(),
            shared: None,
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
        Linear::MoeCombine {
            routed: routed.id(),
            shared: Some(shared.id()),
            y: y.id(),
            weights: None,
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
