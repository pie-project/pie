//! Gated MLP activations: a `[gate | up]` packed row (or two planes) in, the
//! gated product out, computed in f32 and rounded once at the store, as
//! kernels-wgpu `mlp/packed.wgsl` and `mlp/gated.wgsl` compute them.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::Val;
use crate::tensor::Tensor;

/// The gate and up halves of a packed `[rows, 2 · intermediate]` row, in f32.
fn halves(op: &'static str, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    expect(op, packed, &[Dtype::Bf16])?;
    if packed.width != intermediate.saturating_mul(2) || y.width != intermediate {
        return Err(refuse(
            op,
            format!(
                "the packed row is {} wide and lands {} where the intermediate is {intermediate}",
                packed.width, y.width
            ),
        ));
    }
    if y.rows != packed.rows {
        return Err(refuse(
            op,
            format!("{} packed rows land {} rows", packed.rows, y.rows),
        ));
    }
    Ok(())
}

fn read_halves(cx: &mut Cx<'_>, packed: Tensor, intermediate: u32) -> Result<(Val, Val), Error> {
    let p = cx.read_f32(packed)?;
    let i = i64::from(intermediate);
    let g = cx.slice_axis(p, 1, 0, i)?;
    let u = cx.slice_axis(p, 1, i, 2 * i)?;
    Ok((g, u))
}

fn split(op: &'static str, gate: Tensor, up: Tensor, y: Tensor) -> Result<(), Error> {
    expect(op, gate, &[Dtype::Bf16])?;
    if (up.rows, up.width) != (gate.rows, gate.width)
        || (y.rows, y.width) != (gate.rows, gate.width)
    {
        return Err(refuse(
            op,
            "the gate, up and output planes are not one rectangle",
        ));
    }
    Ok(())
}

/// `g / (1 + e^-g)`, as the shaders spell silu.
fn silu(cx: &mut Cx<'_>, g: Val) -> Result<Val, Error> {
    let n = cx.neg(g);
    let e = cx.exp(n);
    let d = cx.offset(e, 1.0)?;
    Ok(cx.div(g, d)?)
}

/// `min(g, limit)` and `clamp(u, -limit, limit)`.
fn clamped(cx: &mut Cx<'_>, g: Val, u: Val, limit: f32) -> Result<(Val, Val), Error> {
    let hi = cx.like_f(g, f64::from(limit));
    let g = cx.min(g, hi)?;
    let lo = cx.like_f(u, -f64::from(limit));
    let u = cx.max(u, lo)?;
    let u = cx.min(u, hi)?;
    Ok((g, u))
}

/// `tanh(clamp(x, -16, 16))`, the shaders' `pie_tanh`.
fn pie_tanh(cx: &mut Cx<'_>, x: Val) -> Result<Val, Error> {
    let lo = cx.like_f(x, -16.0);
    let hi = cx.like_f(x, 16.0);
    let x = cx.max(x, lo)?;
    let x = cx.min(x, hi)?;
    Ok(cx.tanh(x))
}

pub fn swiglu(ctx: &Ctx<'_>, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu";
    halves(OP, packed, intermediate, y)?;
    ctx.emit(&mut |cx| {
        let (g, u) = read_halves(cx, packed, intermediate)?;
        let s = silu(cx, g)?;
        let out = cx.mul(s, u)?;
        cx.write(y, out)
    })
}

pub fn swiglu_clamp(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp";
    halves(OP, packed, intermediate, y)?;
    ctx.emit(&mut |cx| {
        let (g, u) = read_halves(cx, packed, intermediate)?;
        let (g, u) = clamped(cx, g, u, limit)?;
        let s = silu(cx, g)?;
        let out = cx.mul(s, u)?;
        cx.write(y, out)
    })
}

/// gpt-oss: `min(g, limit) · σ(alpha · g) · (clamp(u) + 1)`.
pub fn swiglu_clamp_alpha(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    alpha: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_alpha";
    halves(OP, packed, intermediate, y)?;
    ctx.emit(&mut |cx| {
        let (g, u) = read_halves(cx, packed, intermediate)?;
        let (g, u) = clamped(cx, g, u, limit)?;
        let ag = cx.scale(g, f64::from(alpha))?;
        let n = cx.neg(ag);
        let e = cx.exp(n);
        let d = cx.offset(e, 1.0)?;
        let s = cx.div(g, d)?;
        let u1 = cx.offset(u, 1.0)?;
        let out = cx.mul(s, u1)?;
        cx.write(y, out)
    })
}

pub fn swiglu_clamp_split(
    ctx: &Ctx<'_>,
    gate: Tensor,
    up: Tensor,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_swiglu_clamp_split";
    split(OP, gate, up, y)?;
    ctx.emit(&mut |cx| {
        let g = cx.read_f32(gate)?;
        let u = cx.read_f32(up)?;
        let (g, u) = clamped(cx, g, u, limit)?;
        let s = silu(cx, g)?;
        let out = cx.mul(s, u)?;
        cx.write(y, out)
    })
}

pub fn geglu_tanh(ctx: &Ctx<'_>, gate: Tensor, up: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh";
    split(OP, gate, up, y)?;
    ctx.emit(&mut |cx| {
        let g = cx.read_f32(gate)?;
        let u = cx.read_f32(up)?;
        let g = cx.gelu_tanh(g)?;
        let out = cx.mul(g, u)?;
        cx.write(y, out)
    })
}

pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.mlp_gelu_tanh";
    expect(OP, x, &[Dtype::Bf16])?;
    if (y.rows, y.width) != (x.rows, x.width) {
        return Err(refuse(
            OP,
            "the input and output planes are not one rectangle",
        ));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let out = cx.gelu_tanh(v)?;
        cx.write(y, out)
    })
}

pub fn geglu_tanh_packed(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_geglu_tanh_packed";
    halves(OP, packed, intermediate, y)?;
    ctx.emit(&mut |cx| {
        let (g, u) = read_halves(cx, packed, intermediate)?;
        let g = cx.gelu_tanh(g)?;
        let out = cx.mul(g, u)?;
        cx.write(y, out)
    })
}

/// `beta · tanh(g / beta) · σ(g) · u`, with `u` soft-capped to
/// `up_cap · tanh(u / up_cap)` when `up_cap` is positive.
pub fn situ(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    beta: f32,
    up_cap: Option<f32>,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.mlp_situ";
    if beta == 0.0 {
        return Err(refuse(OP, "beta is zero, and the gate divides by it"));
    }
    halves(OP, packed, intermediate, y)?;
    let cap = up_cap.unwrap_or(0.0);
    ctx.emit(&mut |cx| {
        let (g, u) = read_halves(cx, packed, intermediate)?;
        let gb = cx.scale(g, 1.0 / f64::from(beta))?;
        let t = pie_tanh(cx, gb)?;
        let t = cx.scale(t, f64::from(beta))?;
        let n = cx.neg(g);
        let e = cx.exp(n);
        let d = cx.offset(e, 1.0)?;
        let sg = cx.div(t, d)?;
        let u = if cap > 0.0 {
            let uc = cx.scale(u, 1.0 / f64::from(cap))?;
            let t = pie_tanh(cx, uc)?;
            cx.scale(t, f64::from(cap))?
        } else {
            u
        };
        let out = cx.mul(sg, u)?;
        cx.write(y, out)
    })
}
