//! Tensor-parallel CUDA shapes and collectives.

use super::*;


builder! {
    /// `kernels::norm::residual_add_rmsnorm`; the epsilon rides the run.
    pub fn residual_add_rmsnorm(
        x: &Val,
        residual: &Val,
        weight: &str,
        hidden: u32,
        eps: f32,
    ) -> Val {
        symbol: "norm::residual_add_rmsnorm",
        on: x,
        weights: [weight],
        params: [eps.to_bits()],
        inputs: [x, residual],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the fused norm produces its value",
    }
}


builder! {
    /// `comm::all_reduce_bf16`: the NVLink P2P sum, out of place.
    pub fn all_reduce_p2p(x: &Val, hidden: u32) -> Val {
        symbol: "comm::all_reduce_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the collective produces its value",
    }


    /// `dist::all_reduce_bf16`.
    pub fn all_reduce(x: &Val, hidden: u32) -> Val {
        symbol: "dist::all_reduce_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the collective produces its value",
    }


    /// `dist::all_reduce_bf16_out`.
    pub fn all_reduce_out(x: &Val, hidden: u32) -> Val {
        symbol: "dist::all_reduce_bf16_out",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the collective produces its value",
    }


    /// `dist::all_gather_bf16`.
    pub fn all_gather(x: &Val, parts: u32, width: u32) -> Val {
        symbol: "dist::all_gather_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width * parts)] as BF16,
        made: "the collective produces its value",
    }
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
