//! GEMMA — 3n's AltUp, and gemma-4.

use super::*;

// ── gemma-4 ────────────────────────────────────────────────────
// Hand-written cases either read result shapes from the tape, use `Trace::with`,
// or choose a symbol from facts/arguments.
pub fn geglu_tanh(x: &Val, intermediate: u32, packed: bool) -> Val {
    record(
        &x.t,
        x.layer,
        if packed {
            "mlp::chunked_geglu_tanh"
        } else {
            "mlp::geglu_tanh"
        },
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

/// `norm::scalar_mul`: multiply by a named load-time constant.
/// The weight name is `scale.<name>`; if present, `params[0]` is `by` as `f32` bits.
/// `None` means this family has not derived the number and falls through to a handwritten arm.
pub fn scalar_mul(x: &Val, scale: &str, by: Option<f32>) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record_with_params(
        &x.t,
        x.layer,
        "norm::scalar_mul",
        vec![format!("scale.{scale}")],
        None,
        by.map(f32::to_bits).into_iter().collect(),
        vec![x.id],
        Some(out),
    )
    .expect("the scale produces its value")
}

/// `moe::flashinfer_cutlass_moe_bf16`: fused decode MoE block — untraced
/// (no generated twin). Consumes `(x, experts, weights)` and both expert
/// banks; output is `[Tokens, hidden]`.
#[must_use]
pub fn moe_fused_cutlass(
    x: &Val,
    experts: &Val,
    weights: &Val,
    gate_up: &MatW,
    down: &MatW,
    hidden: u32,
) -> Val {
    record(
        &x.t,
        gate_up.layer,
        "moe::flashinfer_cutlass_moe_bf16",
        vec![gate_up.name.clone(), down.name.clone()],
        None,
        vec![x.id, experts.id, weights.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the fused MoE produces its value")
}

/// Row RMSNorm variant selected by [`NormW`]. One statement either way:
/// params `[per_head_dim, eps]` per the swept signature, `0` for whole-row.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    crate::rmsnorm(x, w)
}
