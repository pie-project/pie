//! Gated MLP activations: a `[gate | up]` packed row (or two planes) in, the
//! gated product out, computed in f32 and rounded once at the store, as
//! kernels-wgpu `mlp/packed.wgsl` and `mlp/gated.wgsl` compute them.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Func, Val};
use crate::tensor::{Landing, Operand, Stored};

/// The gate and up halves of a packed `[rows, 2 · intermediate]` row, in f32.
fn halves(
    op: &'static str,
    packed: impl Stored,
    intermediate: u32,
    y: impl Stored,
) -> Result<(), Error> {
    expect(op, packed, &[Dtype::Bf16])?;
    if packed.width() != intermediate.saturating_mul(2) || y.width() != intermediate {
        return Err(refuse(
            op,
            format!(
                "the packed row is {} wide and lands {} where the intermediate is {intermediate}",
                packed.width(),
                y.width()
            ),
        ));
    }
    if y.rows() != packed.rows() {
        return Err(refuse(
            op,
            format!("{} packed rows land {} rows", packed.rows(), y.rows()),
        ));
    }
    Ok(())
}

fn read_halves(f: &mut Func, packed: Operand, intermediate: u32) -> Result<(Val, Val), Error> {
    let p = packed.f32(f);
    let i = i64::from(intermediate);
    let g = f.slice_axis(p, 1, 0, i)?;
    let u = f.slice_axis(p, 1, i, 2 * i)?;
    Ok((g, u))
}

fn split(
    op: &'static str,
    gate: impl Stored,
    up: impl Stored,
    y: impl Stored,
) -> Result<(), Error> {
    expect(op, gate, &[Dtype::Bf16])?;
    if (up.rows(), up.width()) != (gate.rows(), gate.width())
        || (y.rows(), y.width()) != (gate.rows(), gate.width())
    {
        return Err(refuse(
            op,
            "the gate, up and output planes are not one rectangle",
        ));
    }
    Ok(())
}

/// `g / (1 + e^-g)`, as the shaders spell silu.
fn silu(f: &mut Func, g: Val) -> Result<Val, Error> {
    let n = f.neg(g);
    let e = f.exp(n);
    let d = f.offset(e, 1.0)?;
    Ok(f.div(g, d)?)
}

/// `min(g, limit)` and `clamp(u, -limit, limit)`.
fn clamped(f: &mut Func, g: Val, u: Val, limit: f32) -> Result<(Val, Val), Error> {
    let hi = f.like_f(g, f64::from(limit));
    let g = f.min(g, hi)?;
    let lo = f.like_f(u, -f64::from(limit));
    let u = f.max(u, lo)?;
    let u = f.min(u, hi)?;
    Ok((g, u))
}

/// `tanh(clamp(x, -16, 16))`, the shaders' `pie_tanh`.
fn pie_tanh(f: &mut Func, x: Val) -> Result<Val, Error> {
    let lo = f.like_f(x, -16.0);
    let hi = f.like_f(x, 16.0);
    let x = f.max(x, lo)?;
    let x = f.min(x, hi)?;
    Ok(f.tanh(x))
}

pub fn swiglu(f: &mut Func, packed: Operand, intermediate: u32, y: Landing) -> Result<Val, Error> {
    const OP: &str = "linear.swiglu";
    halves(OP, packed, intermediate, y)?;
    let (g, u) = read_halves(f, packed, intermediate)?;
    let s = silu(f, g)?;
    let out = f.mul(s, u)?;
    Ok(out)
}

pub fn swiglu_clamp(
    f: &mut Func,
    packed: Operand,
    intermediate: u32,
    limit: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.swiglu_clamp";
    halves(OP, packed, intermediate, y)?;
    let (g, u) = read_halves(f, packed, intermediate)?;
    let (g, u) = clamped(f, g, u, limit)?;
    let s = silu(f, g)?;
    let out = f.mul(s, u)?;
    Ok(out)
}

/// gpt-oss: `min(g, limit) · σ(alpha · g) · (clamp(u) + 1)`.
pub fn swiglu_clamp_alpha(
    f: &mut Func,
    packed: Operand,
    intermediate: u32,
    limit: f32,
    alpha: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.swiglu_clamp_alpha";
    halves(OP, packed, intermediate, y)?;
    let (g, u) = read_halves(f, packed, intermediate)?;
    let (g, u) = clamped(f, g, u, limit)?;
    let ag = f.scale(g, f64::from(alpha))?;
    let n = f.neg(ag);
    let e = f.exp(n);
    let d = f.offset(e, 1.0)?;
    let s = f.div(g, d)?;
    let u1 = f.offset(u, 1.0)?;
    let out = f.mul(s, u1)?;
    Ok(out)
}

pub fn swiglu_clamp_split(
    f: &mut Func,
    gate: Operand,
    up: Operand,
    limit: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.swiglu_clamp_split";
    split(OP, gate, up, y)?;
    let g = gate.f32(f);
    let u = up.f32(f);
    let (g, u) = clamped(f, g, u, limit)?;
    let s = silu(f, g)?;
    let out = f.mul(s, u)?;
    Ok(out)
}

pub fn geglu_tanh(f: &mut Func, gate: Operand, up: Operand, y: Landing) -> Result<Val, Error> {
    const OP: &str = "linear.geglu_tanh";
    split(OP, gate, up, y)?;
    let g = gate.f32(f);
    let u = up.f32(f);
    let g = f.gelu_tanh(g)?;
    let out = f.mul(g, u)?;
    Ok(out)
}

pub fn gelu_tanh(f: &mut Func, x: Operand, y: Landing) -> Result<Val, Error> {
    const OP: &str = "linear.gelu_tanh";
    expect(OP, x, &[Dtype::Bf16])?;
    if (y.rows, y.width) != (x.rows, x.width) {
        return Err(refuse(
            OP,
            "the input and output planes are not one rectangle",
        ));
    }
    let v = x.f32(f);
    let out = f.gelu_tanh(v)?;
    Ok(out)
}

pub fn geglu_tanh_packed(
    f: &mut Func,
    packed: Operand,
    intermediate: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.geglu_tanh_packed";
    halves(OP, packed, intermediate, y)?;
    let (g, u) = read_halves(f, packed, intermediate)?;
    let g = f.gelu_tanh(g)?;
    let out = f.mul(g, u)?;
    Ok(out)
}

/// `beta · tanh(g / beta) · σ(g) · u`, with `u` soft-capped to
/// `up_cap · tanh(u / up_cap)` when `up_cap` is positive.
pub fn situ(
    f: &mut Func,
    packed: Operand,
    intermediate: u32,
    beta: f32,
    up_cap: Option<f32>,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.situ";
    if beta == 0.0 {
        return Err(refuse(OP, "beta is zero, and the gate divides by it"));
    }
    halves(OP, packed, intermediate, y)?;
    let cap = up_cap.unwrap_or(0.0);
    let (g, u) = read_halves(f, packed, intermediate)?;
    let gb = f.scale(g, 1.0 / f64::from(beta))?;
    let t = pie_tanh(f, gb)?;
    let t = f.scale(t, f64::from(beta))?;
    let n = f.neg(g);
    let e = f.exp(n);
    let d = f.offset(e, 1.0)?;
    let sg = f.div(t, d)?;
    let u = if cap > 0.0 {
        let uc = f.scale(u, 1.0 / f64::from(cap))?;
        let t = pie_tanh(f, uc)?;
        f.scale(t, f64::from(cap))?
    } else {
        u
    };
    let out = f.mul(sg, u)?;
    Ok(out)
}
