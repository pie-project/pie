use std::cmp::Ordering;

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Val};
use crate::tensor::Tensor;

/// `min(max(x, lo), hi)` in f32, bounds splatted over `x`.
fn clip(cx: &mut Cx<'_>, x: Val, lo: Val, hi: Val) -> Result<Val, Error> {
    let dims = cx.dims(x).to_vec();
    let lo = cx.splat(lo, &dims)?;
    let hi = cx.splat(hi, &dims)?;
    let v = cx.max(x, lo)?;
    Ok(cx.min(v, hi)?)
}

/// `x = min(max(x, lo), hi)`; for a bf16 `x` the stated bounds round to bf16
/// first, as kernels-wgpu `norm/clip.wgsl` lands them (an f32 `x` clamps at
/// the f32 bounds, as kernels-metal does).
pub fn clamp(ctx: &Ctx<'_>, lo: f32, hi: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    if !matches!(lo.partial_cmp(&hi), Some(Ordering::Less | Ordering::Equal)) {
        return Err(refuse(
            OP,
            format!("the bounds {lo} and {hi} cross, and a clamp between them is the constant {hi}"),
        ));
    }
    let round = |v: f32| {
        if x.dtype == Dtype::Bf16 {
            f32::from_bits(u32::from(crate::hlo::bf16_bits(v)) << 16)
        } else {
            v
        }
    };
    let (lo, hi) = (round(lo), round(hi));
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let l = cx.const_f(Elem::F32, f64::from(lo), &[]);
        let h = cx.const_f(Elem::F32, f64::from(hi), &[]);
        let v = clip(cx, v, l, h)?;
        cx.write(x, v)
    })
}

/// `x = min(max(x, lo[0]), hi[0])` with one-scalar bound planes in `x`'s
/// element. Reference: kernels-wgpu `clamp_learned_bf16`.
pub fn clamp_learned(ctx: &Ctx<'_>, lo: Tensor, hi: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.clamp_learned";
    expect(OP, x, &[Dtype::Bf16, Dtype::F32])?;
    for (what, bound) in [("lower", lo), ("upper", hi)] {
        if bound.dtype != x.dtype {
            return Err(refuse(
                OP,
                format!("the {what} bound is {:?} and the rows it clamps are {:?}", bound.dtype, x.dtype),
            ));
        }
        if bound.elements() != 1 {
            return Err(refuse(
                OP,
                format!("the {what} bound is a {} x {} plane, and this clamp reads one scalar", bound.rows, bound.width),
            ));
        }
    }
    ctx.emit(&mut |cx| {
        let l = super::norm::scalar_of(cx, lo)?;
        let h = super::norm::scalar_of(cx, hi)?;
        let v = cx.read_f32(x)?;
        let v = clip(cx, v, l, h)?;
        cx.write(x, v)
    })
}
