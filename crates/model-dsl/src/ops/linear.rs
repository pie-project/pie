//! The `Linear` family: the plain matmuls, the fused MLP activations, the MoE
//! routing/expert path, and the LoRA correction over a routed adapter bank.

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

/// [`mlp_swiglu_clamp`]'s arithmetic over an unfused pair: the two halves as two values, for a bank whose gate and up cannot be declared as one. Result's type is the gate's.
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

/// The ungated gelu: `gelu_tanh(x)`, no `up` half. Every other gelu builder here multiplies by a gate; a dedicated op avoids baking a zero-`up` bank.
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

/// The same top-k and the same softmax over the selected, times the learned
/// per-expert gain `scale[routes[t, k]]`.
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
        },
        &[logits],
    );
    (routes, weights)
}

/// Sigmoid routing where a per-expert bias steers the choice only: the
/// picked weights are the sigmoid scores, renormalized and scaled.
pub fn moe_topk_sigmoid_biased(
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
        Linear::MoeTopkSigmoid {
            logits: logits.id(),
            bias: Some(r.weight(bias)),
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
    moe_topk_sqrt_softplus_hinted(logits, bias, experts, top_k, renormalize, scaling, None)
}

/// [`moe_topk_sqrt_softplus`] carrying a [`moe_predict_route`] output as its
/// `hint`, so the prediction stays live to this router's segment cut.
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

/// **A ROUTE PREDICTION** — the sqrt-softplus-plus-bias top-`top_k` over
/// `logits` computed early, whose `routes` no select reads. Hand it to the
/// real router as its hint.
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

/// The lookup router: `tid2eid` is a `[vocab, top_k]` I64 table naming this token id's experts outright, at the uniform weight `1/top_k`; no router logit is computed.
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

/// The static routes of a grouped projection: `[tokens, groups]` of `g`.
/// `x` is any token-row value of the fire, named so the routes land in the
/// same row space the projection walks.
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

/// **THE BLOCK-DIAGONAL PROJECTION**: `w` is `[groups · N, K]` and `x` is
/// `[tokens, groups · K]`; group `g` of the row projects through rows
/// `g·N..(g+1)·N` of the plane, and the result is `[tokens, groups · N]`.
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
/// The routed rows are `tokens x top_k`; `top_k` rides along as a wrapper argument.
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

/// The fold over a split-plane quantized bank with nothing added: a routed bias lands after the reduce, through `moe_bias_sum`.
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

/// Adds the routed bias mixture to an already-folded activation. The expert down-projection is rows-cut under tp, so each rank's routed matmul is a partial product and a replicated bias folded into it would be summed tp times; this op adds it once, on the reduced row, after the all_reduce.
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

/// The correction class: add `B[a]*(A[a]*x)` to `y`, in place, where `a` is the adapter each row's lane routes to.
///
/// `x` is the site's input (what the base projection multiplied) and `y` its already-materialised output; the wrapper returns the corrected value, the same arena column — an in-place SSA pair, like `residual_add`'s.
///
/// Runs over the adapter window (rows whose lane's word satisfies the node's guard): a fire no lane carried an adapter into has zero rows for it, so the axis costs nothing there. Which adapter, at what rank, is absorbed by `routes` and the bank's own shape rather than a branch.
pub fn lora_correct(x: &Value, bank_a: &Weight, bank_b: &Weight, routes: &Value, y: &Value) -> Value {
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
    // the value comes back unguarded: the node is narrow (runs only over the adapter window) but its output column is y's, written on every row.
    y_out.everywhere()
}
