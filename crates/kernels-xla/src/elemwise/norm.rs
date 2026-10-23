#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Fold, Func, Val};
use crate::tensor::{Landing, Operand};

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{what}` is zero")));
    }
    Ok(v)
}

/// `x / rms(x)` over each `axis`-wide run of an f32 `[rows, width]`; the mean
/// square is taken in f32, as the GPU kernels take it.
pub(crate) fn inv_rms_rows(
    f: &mut Func,
    op: &'static str,
    x: Val,
    axis: u32,
    eps: f32,
) -> Result<Val, Error> {
    let dims = f.dims(x).to_vec();
    let (rows, width) = (dims[0], dims[1]);
    let axis = i64::from(nonzero(op, "the normed axis", axis)?);
    if width % axis != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {axis}-wide normed axes"),
        ));
    }
    let heads = width / axis;
    let x3 = f.reshape(x, &[rows, heads, axis])?;
    let sq = f.mul(x3, x3)?;
    let sum = f.reduce(sq, &[2], Fold::Sum)?;
    let mean = f.scale(sum, 1.0 / axis as f64)?;
    let mean = f.offset(mean, f64::from(eps))?;
    let inv = f.rsqrt(mean);
    let inv = f.broadcast(inv, &[rows, heads, axis], &[0, 1])?;
    let y = f.mul(x3, inv)?;
    Ok(f.reshape(y, &[rows, width])?)
}

/// A `[axis]` gain row broadcast over `[rows, width]` (`width / axis` heads
/// share it), optionally `1 + w`.
fn gain(f: &mut Func, weight: Val, rows: i64, width: i64, plus_one: bool) -> Result<Val, Error> {
    let w = f.convert(weight, Elem::F32);
    let n = f.ty(w).elements();
    let w = f.reshape(w, &[n])?;
    let w = if plus_one { f.offset(w, 1.0)? } else { w };
    let heads = width / n;
    let w = f.broadcast(w, &[rows, heads, n], &[2])?;
    Ok(f.reshape(w, &[rows, width])?)
}

fn rms_row(
    f: &mut Func,
    op: &'static str,
    x: Operand,
    weight: Operand,
    eps: f32,
    axis: u32,
    plus_one: bool,
) -> Result<Val, Error> {
    expect(op, x, &[Dtype::Bf16])?;
    if u64::from(weight.rows) * u64::from(weight.width) != u64::from(axis) {
        return Err(refuse(
            op,
            format!(
                "the weight is {}x{}, and it gains a {axis}-wide axis",
                weight.rows, weight.width
            ),
        ));
    }
    let xv = x.f32(f);
    let n = inv_rms_rows(f, op, xv, axis, eps)?;
    let g = gain(
        f,
        weight.val,
        i64::from(x.rows),
        i64::from(x.width),
        plus_one,
    )?;
    Ok(f.mul(n, g)?)
}

pub fn rmsnorm(f: &mut Func, x: Operand, weight: Operand, eps: f32) -> Result<Val, Error> {
    rms_row(f, "elemwise.rmsnorm", x, weight, eps, x.width, false)
}

pub fn rmsnorm_per_head(
    f: &mut Func,
    x: Operand,
    weight: Operand,
    head_dim: u32,
    eps: f32,
) -> Result<Val, Error> {
    rms_row(f, "elemwise.rmsnorm", x, weight, eps, head_dim, false)
}

pub fn rmsnorm_plus_one(f: &mut Func, x: Operand, weight: Operand, eps: f32) -> Result<Val, Error> {
    rms_row(f, "elemwise.rmsnorm", x, weight, eps, x.width, true)
}

pub fn rmsnorm_per_head_plus_one(
    f: &mut Func,
    x: Operand,
    weight: Operand,
    head_dim: u32,
    eps: f32,
) -> Result<Val, Error> {
    rms_row(f, "elemwise.rmsnorm", x, weight, eps, head_dim, true)
}

/// Norms each `group`-wide run on its own, and gains the whole row by a
/// row-wide `1 + w` bank (group `g` reads its own slice of it).
pub fn rmsnorm_grouped_plus_one(
    f: &mut Func,
    x: Operand,
    weight: Operand,
    group: u32,
    eps: f32,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.rmsnorm";
    expect(OP, x, &[Dtype::Bf16])?;
    let bank = weight.width * weight.rows.max(1);
    if bank != x.width {
        return Err(refuse(
            OP,
            format!(
                "the weight bank is {bank} wide and the row it gains is {}",
                x.width
            ),
        ));
    }
    let xv = x.f32(f);
    let n = inv_rms_rows(f, OP, xv, group, eps)?;
    let g = gain(f, weight.val, i64::from(x.rows), i64::from(x.width), true)?;
    Ok(f.mul(n, g)?)
}

pub fn rmsnorm_no_scale(f: &mut Func, x: Operand, head_dim: u32, eps: f32) -> Result<Val, Error> {
    const OP: &str = "elemwise.rmsnorm";
    expect(OP, x, &[Dtype::Bf16])?;
    let xv = x.f32(f);
    inv_rms_rows(f, OP, xv, head_dim, eps)
}

/// `rms(x) * w * gate(z)` per `vd`-wide head of an f32 accumulator, with a
/// `vd`-wide f32 weight; `gate` is `z·σ(z)` or `σ(z)`.
fn gated_rms(
    f: &mut Func,
    op: &'static str,
    x: Operand,
    gate: Operand,
    weight: Operand,
    vd: u32,
    eps: f32,
    sigmoid: bool,
    y: Landing,
) -> Result<Val, Error> {
    expect(op, gate, &[Dtype::Bf16])?;
    if gate.rows != x.rows || gate.width != x.width || y.rows != x.rows || y.width != x.width {
        return Err(refuse(
            op,
            "the gate and the landing ride the normed rectangle",
        ));
    }
    let xv = x.f32(f);
    let n = inv_rms_rows(f, op, xv, vd, eps)?;
    let g = gain(f, weight.val, i64::from(x.rows), i64::from(x.width), false)?;
    let n = f.mul(n, g)?;
    let z = gate.f32(f);
    let s = f.sigmoid(z);
    let gate = if sigmoid { s } else { f.mul(z, s)? };
    Ok(f.mul(n, gate)?)
}

pub fn rmsnorm_gated(
    f: &mut Func,
    x: Operand,
    gate: Operand,
    weight: Operand,
    head_dim: u32,
    eps: f32,
    sigmoid_gate: bool,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.rmsnorm_gated";
    let vd = nonzero(OP, "the stated value-head width", head_dim)?;
    gated_rms(f, OP, x, gate, weight, vd, eps, sigmoid_gate, y)
}

pub fn rmsnorm_gated_by(
    f: &mut Func,
    x: Operand,
    gate: Operand,
    weight: Operand,
    heads: u32,
    eps: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.rmsnorm_gated_by";
    nonzero(OP, "the stated head count", heads)?;
    if x.width == 0 || !x.width.is_multiple_of(heads) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide normed row does not divide by the stated head count {heads}",
                x.width
            ),
        ));
    }
    gated_rms(f, OP, x, gate, weight, x.width / heads, eps, true, y)
}

/// `y + x`, the new `y`.
pub fn residual_add(f: &mut Func, x: Operand, y: Operand) -> Result<Val, Error> {
    const OP: &str = "elemwise.residual_add";
    expect(OP, y, &[Dtype::Bf16])?;
    let a = x.f32(f);
    let b = y.f32(f);
    Ok(f.add(a, b)?)
}

/// A `[width]` row broadcast down `[rows, width]`, in f32.
pub(crate) fn row_of(f: &mut Func, t: Operand, rows: u32, width: u32) -> Result<Val, Error> {
    let v = t.f32(f);
    let v = f.reshape(v, &[i64::from(width)])?;
    Ok(f.broadcast(v, &[i64::from(rows), i64::from(width)], &[1])?)
}

/// `out + bias` per column, the new `out`.
pub fn add_bias(f: &mut Func, bias: Operand, out: Operand) -> Result<Val, Error> {
    const OP: &str = "elemwise.add_bias";
    expect(OP, out, &[Dtype::Bf16, Dtype::F32])?;
    if bias.elements() != u64::from(out.width) {
        return Err(refuse(OP, "the bias is not one value per column"));
    }
    let b = row_of(f, bias, out.rows, out.width)?;
    let o = out.f32(f);
    Ok(f.add(o, b)?)
}

pub fn layernorm(
    f: &mut Func,
    x: Operand,
    weight: Operand,
    bias: Operand,
    eps: f32,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.layernorm";
    expect(OP, x, &[Dtype::Bf16])?;
    let xv = x.f32(f);
    let n = centered(f, xv, eps)?;
    let w = row_of(f, weight, x.rows, x.width)?;
    let b = row_of(f, bias, x.rows, x.width)?;
    let out = f.mul(n, w)?;
    Ok(f.add(out, b)?)
}

/// `(x - mean) / sqrt(var + eps)` per row of an f32 `[rows, width]`.
pub(crate) fn centered(f: &mut Func, x: Val, eps: f32) -> Result<Val, Error> {
    let dims = f.dims(x).to_vec();
    let n = dims[1] as f64;
    let sum = f.reduce(x, &[1], Fold::Sum)?;
    let mean = f.scale(sum, 1.0 / n)?;
    let mean = f.broadcast(mean, &dims, &[0])?;
    let c = f.sub(x, mean)?;
    let sq = f.mul(c, c)?;
    let var = f.reduce(sq, &[1], Fold::Sum)?;
    let var = f.scale(var, 1.0 / n)?;
    let var = f.offset(var, f64::from(eps))?;
    let inv = f.rsqrt(var);
    let inv = f.broadcast(inv, &dims, &[0])?;
    Ok(f.mul(c, inv)?)
}

/// `(out - bias) * scale` per column, the new `out`.
pub fn standardize(
    f: &mut Func,
    bias: Operand,
    scale: Operand,
    out: Operand,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.standardize";
    expect(OP, out, &[Dtype::Bf16])?;
    for (what, plane) in [("bias", bias), ("scale", scale)] {
        if plane.elements() != u64::from(out.width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is not one scalar per column of a {}-wide row",
                    out.width
                ),
            ));
        }
    }
    let b = row_of(f, bias, out.rows, out.width)?;
    let s = row_of(f, scale, out.rows, out.width)?;
    let o = out.f32(f);
    let o = f.sub(o, b)?;
    Ok(f.mul(o, s)?)
}

/// `x * s`, `s` rounded to `x`'s element first, as the stated scalar lands
/// (kernels-cuda `elemwise::mul_scalar<T>`: bf16, f16, f32); the new `x`.
pub fn mul_scalar(f: &mut Func, s: f32, x: Operand) -> Result<Val, Error> {
    const OP: &str = "elemwise.act";
    expect(OP, x, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
    let s = match x.dtype {
        Dtype::Bf16 => f32::from_bits(u32::from(crate::hlo::bf16_bits(s)) << 16),
        Dtype::F16 => f16_value(crate::hlo::f16_bits(s)),
        _ => s,
    };
    let v = x.f32(f);
    Ok(f.scale(v, f64::from(s))?)
}

/// `silu(x * s)`, the new `x`.
pub fn silu_scaled(f: &mut Func, s: f32, x: Operand) -> Result<Val, Error> {
    const OP: &str = "elemwise.act";
    expect(OP, x, &[Dtype::Bf16])?;
    let v = x.f32(f);
    let v = f.scale(v, f64::from(s))?;
    Ok(f.silu(v)?)
}

/// `x * s[0]`, a device scalar; the new `x`.
pub fn scale(f: &mut Func, s: Operand, x: Operand) -> Result<Val, Error> {
    const OP: &str = "elemwise.act";
    expect(OP, x, &[Dtype::Bf16])?;
    let sv = s.f32(f);
    let n = f.ty(sv).elements();
    let sv = f.reshape(sv, &[n])?;
    let sv = f.slice(sv, &[0], &[1], &[1])?;
    let sv = f.reshape(sv, &[])?;
    let sv = f.splat(sv, &[i64::from(x.rows), i64::from(x.width)])?;
    let v = x.f32(f);
    Ok(f.mul(v, sv)?)
}

/// `(x - mean) / sqrt(var + eps)` per row, no gain or bias. Reference:
/// kernels-metal / kernels-cuda `layernorm_no_scale` (wgpu refuses it).
pub fn layernorm_no_scale(f: &mut Func, x: Operand, eps: f32, y: Landing) -> Result<Val, Error> {
    const OP: &str = "elemwise.layernorm";
    expect(OP, x, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
    if x.rows != y.rows || x.width != y.width {
        return Err(refuse(OP, "the normed row lands the rectangle it reads"));
    }
    let xv = x.f32(f);
    centered(f, xv, eps)
}

/// The first element of `s`, an f32 scalar.
pub(crate) fn scalar_of(f: &mut Func, s: Operand) -> Result<Val, Error> {
    let sv = s.f32(f);
    let n = f.ty(sv).elements();
    let sv = f.reshape(sv, &[n])?;
    let sv = f.slice(sv, &[0], &[1], &[1])?;
    Ok(f.reshape(sv, &[])?)
}

pub const MAX_BLEND_BLOCKS: usize = 32;

/// Attn over depth: each candidate (every block, then the prefix) is
/// rms-normed, gained by `weight` and dotted with `proj` into a logit; the
/// softmax over candidates weights their sum. `blocks` are separate
/// `[rows, hidden]` operands (the wgpu engine hands one stacked plane
/// instead). Reference: kernels-slang `elemwise/norm.slang`.
pub fn res_blend(
    f: &mut Func,
    prefix: Operand,
    blocks: &[Operand],
    weight: Operand,
    eps: f32,
    proj: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.res_blend";
    expect(OP, y, &[Dtype::Bf16])?;
    if blocks.len() > MAX_BLEND_BLOCKS {
        return Err(refuse(
            OP,
            format!(
                "{} candidate blocks exceed the bound of {MAX_BLEND_BLOCKS}",
                blocks.len()
            ),
        ));
    }
    let same = |p: Operand| p.rows == y.rows && p.width == y.width;
    if !same(prefix) || !blocks.iter().all(|b| same(*b)) {
        return Err(refuse(
            OP,
            "every candidate is a [rows, hidden] plane like the blend",
        ));
    }
    for (what, p) in [("norm weight", weight), ("projection", proj)] {
        if p.elements() != u64::from(y.width) {
            return Err(refuse(
                OP,
                format!("the {what} is not one value per column"),
            ));
        }
    }
    let (rows, hidden) = (i64::from(y.rows), i64::from(y.width));
    let c = blocks.len() as i64 + 1;
    let mut cands = Vec::with_capacity(c as usize);
    for b in blocks.iter().chain(std::iter::once(&prefix)) {
        let v = b.f32(f);
        cands.push(f.reshape(v, &[rows, 1, hidden])?);
    }
    let all = f.concat(&cands, 1)?;
    let sq = f.mul(all, all)?;
    let ss = f.reduce(sq, &[2], Fold::Sum)?;
    let ms = f.scale(ss, 1.0 / hidden as f64)?;
    let ms = f.offset(ms, f64::from(eps))?;
    let inv = f.rsqrt(ms);
    let inv = f.broadcast(inv, &[rows, c, hidden], &[0, 1])?;
    let n = f.mul(all, inv)?;
    let nw = weight.f32(f);
    let nw = f.reshape(nw, &[hidden])?;
    let nw = f.broadcast(nw, &[rows, c, hidden], &[2])?;
    let pw = proj.f32(f);
    let pw = f.reshape(pw, &[hidden])?;
    let pw = f.broadcast(pw, &[rows, c, hidden], &[2])?;
    let d = f.mul(n, nw)?;
    let d = f.mul(d, pw)?;
    let logits = f.reduce(d, &[2], Fold::Sum)?;
    let p = f.softmax(logits, 1)?;
    let p = f.broadcast(p, &[rows, c, hidden], &[0, 1])?;
    let blend = f.mul(all, p)?;
    Ok(f.reduce(blend, &[1], Fold::Sum)?)
}

/// The value of f16 bits.
fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1F);
    let man = f32::from(bits & 0x3FF);
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1F if man == 0.0 => sign * f32::INFINITY,
        0x1F => f32::NAN,
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}
