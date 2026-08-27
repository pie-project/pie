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

/// The fold over a split-plane quantized bank with nothing added: the routed
/// bias such an expert wants lands after the reduce, through `moe_bias_sum`.
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

/// Adds the routed bias mixture to an already-folded activation. The expert
/// down-projection is rows-cut under tp, so each rank's routed matmul is a
/// partial product and the all_reduce sums the ranks; a replicated bias folded
/// into that matmul would be summed tp times. Routing comes from replicated
/// inputs, so `routes` and `weights` are the same on every rank and the mixture
/// can be said once, here, on the reduced row. At tp = 1 the value is
/// unchanged — the routing weights sum to one — so the model text needs no
/// branch.
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

/// **THE CORRECTION CLASS** (design §8): add `B[a]·(A[a]·x)` to `y`, in place,
/// where `a` is the adapter each row's lane routes to.
///
/// `x` is the site's INPUT (what the base projection multiplied) and `y` its
/// already-materialised output; the wrapper returns the corrected value, which
/// is the same arena column — an in-place SSA pair, like `residual_add`'s.
///
/// **GUARD IT, AND THE GUARD IS THE WINDOW.** Design §8 puts the correction
/// "over the adapter window", and §0 defines a window as the rows of the lanes
/// whose word satisfies the node's cond. So the call site splits on the
/// model's own `has_adapter` fact and this op runs over that arm: a fire no
/// lane carried an adapter into has zero rows for it, `driver::fire::walk`
/// skips a zero-row region outright, and the cost of the axis in that fire is
/// exactly nothing — no launch, no empty grid, no instruction. What §8's table
/// means by "conditional nodes: none" is the CUDA-graph kind (IF/SWITCH), and
/// there are none here: window-split is not a conditional (design §0), and the
/// diversity that would have wanted a branch — WHICH adapter, at what rank —
/// is absorbed by `routes` inside the op and by the bank's own shape.
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
    // **AND THE VALUE COMES BACK UNGUARDED** ([`Value::everywhere`]). The
    // NODE is narrow — it runs over the adapter window and nowhere else — but
    // its output column is `y`'s, written on every row of the fire by whatever
    // produced `y`. A consumer that inherited the guard would carry it down
    // the whole residual stream and the next layer's split would refuse to mix
    // with it, which is the guard leaking out of a window it was never about.
    y_out.everywhere()
}
