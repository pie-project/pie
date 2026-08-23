//! Tensor-parallel CUDA shapes and collectives.

use super::*;

/// `comm::all_reduce_bf16`: the NVLink P2P sum, out of place — a `driver`
/// routine (no generated twin).
#[must_use]
pub fn all_reduce_p2p(x: &Val, hidden: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "comm::all_reduce_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the collective produces its value")
}

/// `comm::all_reduce_residual_rmsnorm_bf16`.
/// Returns `(updated_residual, normed_activation)`.
pub fn all_reduce_residual_rmsnorm(
    x: &Val,
    residual: &Val,
    weight: &NormW,
    hidden: u32,
) -> (Val, Val) {
    let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
    let outs = x.t.with(x.layer, |b| {
        b.launch(
            "comm::all_reduce_residual_rmsnorm_bf16",
            vec![weight.name.clone()],
            None,
            vec![x.id, residual.id],
            vec![shape.clone(), shape],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: x.layer,
    };
    // output 0 aliases `residual`; output 1 is the fresh normed value.
    (mk(outs[0]), mk(outs[1]))
}
