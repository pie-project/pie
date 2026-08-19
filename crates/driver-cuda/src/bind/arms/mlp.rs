//! What a trace that states one of `mlp`'s symbols binds to.

use super::Bound;

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    Bound::derived("mlp::sigmoid_gate_inplace_bf16"),

    // `gate_second` is a `Lit`: `ROUTINES` declares one symbol per launcher and
    // nothing declares the `_gate_second` twin, so the `true` branch is
    // unreachable from any trace.
    Bound::derived("mlp::chunked_swiglu_bf16"),
    // The same kernel over a destination the statement places; see the
    // routine for why one mark could not serve both shapes.
    Bound::derived("mlp::chunked_swiglu_into_bf16"),
    Bound::derived("mlp::relu2_bf16"),
    // `gate: In<0, T>` must state its index: `in_place = &[(0, 0)]` spends BOTH
    // pointers on the aliased buffer, the case the alias walk excludes, so a
    // walked column hands `up` a buffer past the end.
    Bound::derived("mlp::geglu_tanh_bf16"),
    // The same `Lit`, and this row declares no alias at all.
    Bound::derived("mlp::chunked_geglu_tanh_bf16"),
    Bound::derived("mlp::gpt_oss_glu_bf16"),
    Bound::derived("mlp::sigmoid_dot_scalar_gate_add_bf16"),
    Bound::derived("mlp::swiglu_bf16"),
    // No `Lit` for the clamp limit: two specs hold the same number in two
    // config fields today and are free to stop.
    Bound {
        symbol: "mlp::swiglu_clamp_bf16",
        arm: None,
        unbound: Some("the join's foreign operands; needs `Facts::aux(i)`"),
    },
    Bound::derived("mlp::chunked_swiglu_clamp_bf16"),
    Bound {
        symbol: "mlp::situ_bf16",
        arm: None,
        unbound: Some("the model's two SITU betas, which `Deployment` states neither of"),
    },
    // `gate_second` is `Lit(Bool(false))`: no `ROUTINES` row declares the
    // `_gate_second` twin, so a method would promise a choice that does not
    // exist.
    Bound {
        symbol: "mlp::chunked_situ_bf16",
        arm: None,
        unbound: Some("the same two SITU betas"),
    },
    // The sparsity threshold travels IN the statement as `ParamF32(0)`: the
    // caller already holds it.
    Bound::derived("mlp::gaussian_topk_bf16"),
];
