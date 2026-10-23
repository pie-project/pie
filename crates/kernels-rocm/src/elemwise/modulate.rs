use super::norm::Next;
use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;
use dtype::Dtype;

pub fn res_blend(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    crate::attn::res_blend(ctx, prefix, blocks, weight, eps, proj, y)
}

/// Muse Spark's post-norm and residual gate (`elemwise.norm_gate`): `x`
/// times `scale` (rounded), RMS-normed unscaled, and with `res` and its f32
/// `gate` the residual's gated sum; with `next`, the next sublayer's input
/// norm over it in the same pass.
#[allow(clippy::too_many_arguments)]
pub fn norm_gate(
    ctx: &Ctx,
    x: Tensor,
    scale: Option<Tensor>,
    res: Option<Tensor>,
    gate: Option<Tensor>,
    eps: f32,
    temperature: f32,
    fan: u32,
    next: Option<Next>,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.norm_gate";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if res.is_some() != gate.is_some() || gate.is_some_and(|g| g.dtype != Dtype::F32) {
        return Err(refuse(OP, "a residual comes with its f32 gate"));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    let absent = |t: Option<&Tensor>| t.map_or(ArgValue::ABSENT, Tensor::arg);
    let tail = next.is_some();
    // A row of whole eights in aligned planes moves in 16-byte packs, held
    // in registers (up to eight a thread) through both norms.
    let chunks = (y.width / VEC_WIDTH).div_ceil(BLOCK);
    let aligned = [
        Some(x),
        scale,
        res,
        gate,
        Some(*y),
        next.as_ref().map(|n| n.weight),
        next.as_ref().map(|n| n.y),
    ]
    .iter()
    .flatten()
    .all(|t| t.ptr.is_multiple_of(16));
    let entry = if y.width.is_multiple_of(VEC_WIDTH) && chunks <= 8 && aligned {
        format!("::pie::elemwise::norm_gate_vec8<{t}, {BLOCK}, {tail}, {chunks}>")
    } else {
        format!("::pie::elemwise::norm_gate<{t}, {BLOCK}, {tail}>")
    };
    ctx.fire(
        OP,
        Fire::at(NORM_UNIT, symbol(&entry)).apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            absent(scale.as_ref()),
            absent(res.as_ref()),
            absent(gate.as_ref()),
            y.arg(),
            stated(OP, y.width)?.arg(),
            eps.arg(),
            temperature.arg(),
            stated(OP, fan)?.arg(),
            absent(next.as_ref().map(|n| &n.weight)),
            next.as_ref().map_or(0.0, |n| n.eps).arg(),
            i32::from(next.as_ref().is_some_and(|n| n.plus_one)).arg(),
            absent(next.as_ref().map(|n| &n.y)),
            ctx.stage(),
        ],
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormKind {
    LayerNormNoAffine { eps: f32 },
    RmsNormNoScale { eps: f32 },
    RmsNorm { weight: Tensor, eps: f32 },
}

crate::error::unsupported! {
    scale_shift = "elemwise.modulate" (x: Tensor, m: Tensor, lane_of_row: Option<Tensor>,
        o: &mut Tensor);
    scale = "elemwise.modulate" (x: Tensor, m: Tensor, lane_of_row: Option<Tensor>,
        o: &mut Tensor);
    tanh_gate = "elemwise.modulate" (x: Tensor, m: Tensor, lane_of_row: Option<Tensor>,
        o: &mut Tensor);
    gated_residual_add = "elemwise.residual_add" (r: Tensor, g: Tensor, y: Tensor,
        lane_of_row: Option<Tensor>, r_out: &mut Tensor);
    norm_modulate = "fused.norm_modulate" (x: Tensor, m: Tensor,
        lane_of_row: Option<Tensor>, norm: NormKind, o: &mut Tensor);
    #[allow(clippy::too_many_arguments)]
    gated_residual_norm_modulate = "fused.gated_residual_norm_modulate" (r: Tensor,
        g: Tensor, y: Tensor, m: Tensor, lane_of_row: Option<Tensor>, norm: NormKind,
        r_out: &mut Tensor, o: &mut Tensor);
}
