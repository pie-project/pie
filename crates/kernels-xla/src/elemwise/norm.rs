#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Fold, Val};
use crate::tensor::Tensor;

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{what}` is zero")));
    }
    Ok(v)
}

/// `x / rms(x)` over each `axis`-wide run of an f32 `[rows, width]`; the mean
/// square is taken in f32, as the GPU kernels take it.
pub(crate) fn inv_rms_rows(
    cx: &mut Cx<'_>,
    op: &'static str,
    x: Val,
    axis: u32,
    eps: f32,
) -> Result<Val, Error> {
    let dims = cx.dims(x).to_vec();
    let (rows, width) = (dims[0], dims[1]);
    let axis = i64::from(nonzero(op, "the normed axis", axis)?);
    if width % axis != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {axis}-wide normed axes"),
        ));
    }
    let heads = width / axis;
    let x3 = cx.reshape(x, &[rows, heads, axis])?;
    let sq = cx.mul(x3, x3)?;
    let sum = cx.reduce(sq, &[2], Fold::Sum)?;
    let mean = cx.scale(sum, 1.0 / axis as f64)?;
    let mean = cx.offset(mean, f64::from(eps))?;
    let inv = cx.rsqrt(mean);
    let inv = cx.broadcast(inv, &[rows, heads, axis], &[0, 1])?;
    let y = cx.mul(x3, inv)?;
    Ok(cx.reshape(y, &[rows, width])?)
}

/// A `[axis]` gain row broadcast over `[rows, width]` (`width / axis` heads
/// share it), optionally `1 + w`.
fn gain(cx: &mut Cx<'_>, weight: Val, rows: i64, width: i64, plus_one: bool) -> Result<Val, Error> {
    let w = cx.convert(weight, Elem::F32);
    let n = cx.ty(w).elements();
    let w = cx.reshape(w, &[n])?;
    let w = if plus_one { cx.offset(w, 1.0)? } else { w };
    let heads = width / n;
    let w = cx.broadcast(w, &[rows, heads, n], &[2])?;
    Ok(cx.reshape(w, &[rows, width])?)
}

fn rms_row(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    y: Tensor,
    eps: f32,
    axis: u32,
    plus_one: bool,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = inv_rms_rows(cx, op, xv, axis, eps)?;
        let w = cx.read(weight)?;
        let g = gain(cx, w, i64::from(x.rows), i64::from(x.width), plus_one)?;
        let out = cx.mul(n, g)?;
        cx.write(y, out)
    })
}

pub fn rmsnorm(ctx: &Ctx<'_>, x: Tensor, weight: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm",
        x,
        weight,
        y,
        eps,
        x.width,
        false,
    )
}

pub fn rmsnorm_per_head(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head",
        x,
        weight,
        y,
        eps,
        head_dim,
        false,
    )
}

pub fn rmsnorm_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_plus_one",
        x,
        weight,
        y,
        eps,
        x.width,
        true,
    )
}

pub fn rmsnorm_per_head_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elementwise.rmsnorm_per_head_plus_one",
        x,
        weight,
        y,
        eps,
        head_dim,
        true,
    )
}

/// Norms each `group`-wide run on its own, and gains the whole row by a
/// row-wide `1 + w` bank (group `g` reads its own slice of it).
pub fn rmsnorm_grouped_plus_one(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_grouped_plus_one";
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
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = inv_rms_rows(cx, OP, xv, group, eps)?;
        let w = cx.read(weight)?;
        let g = gain(cx, w, i64::from(x.rows), i64::from(x.width), true)?;
        let out = cx.mul(n, g)?;
        cx.write(y, out)
    })
}

pub fn rmsnorm_no_scale(
    ctx: &Ctx<'_>,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_no_scale";
    expect(OP, x, &[Dtype::Bf16])?;
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = inv_rms_rows(cx, OP, xv, head_dim, eps)?;
        cx.write(y, n)
    })
}

/// `rms(x) * w * gate(z)` per `vd`-wide head of an f32 accumulator, with a
/// `vd`-wide f32 weight; `gate` is `z·σ(z)` or `σ(z)`.
fn gated_rms(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    vd: u32,
    eps: f32,
    sigmoid: bool,
    y: Tensor,
) -> Result<(), Error> {
    expect(op, gate, &[Dtype::Bf16])?;
    if gate.rows != x.rows || gate.width != x.width || y.rows != x.rows || y.width != x.width {
        return Err(refuse(
            op,
            "the gate and the landing ride the normed rectangle",
        ));
    }
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = inv_rms_rows(cx, op, xv, vd, eps)?;
        let w = cx.read(weight)?;
        let g = gain(cx, w, i64::from(x.rows), i64::from(x.width), false)?;
        let n = cx.mul(n, g)?;
        let z = cx.read_f32(gate)?;
        let s = cx.sigmoid(z);
        let gate = if sigmoid { s } else { cx.mul(z, s)? };
        let out = cx.mul(n, gate)?;
        cx.write(y, out)
    })
}

pub fn rmsnorm_gated(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    sigmoid_gate: bool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated";
    let vd = nonzero(OP, "the stated value-head width", head_dim)?;
    gated_rms(ctx, OP, x, gate, weight, vd, eps, sigmoid_gate, y)
}

pub fn rmsnorm_gated_by(
    ctx: &Ctx<'_>,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    heads: u32,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_gated_by";
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
    gated_rms(ctx, OP, x, gate, weight, x.width / heads, eps, true, y)
}

/// `y += x`.
pub fn residual_add(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    expect(OP, y, &[Dtype::Bf16])?;
    ctx.emit(&mut |cx| {
        let a = cx.read_f32(x)?;
        let b = cx.read_f32(y)?;
        let s = cx.add(a, b)?;
        cx.write(y, s)
    })
}

/// A `[width]` row broadcast down `[rows, width]`, in f32.
pub(crate) fn row_of(cx: &mut Cx<'_>, t: Tensor, rows: u32, width: u32) -> Result<Val, Error> {
    let v = cx.read_f32(t)?;
    let v = cx.reshape(v, &[i64::from(width)])?;
    Ok(cx.broadcast(v, &[i64::from(rows), i64::from(width)], &[1])?)
}

/// `out += bias` per column.
pub fn add_bias(ctx: &Ctx<'_>, bias: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    expect(OP, out, &[Dtype::Bf16, Dtype::F32])?;
    if bias.elements() != u64::from(out.width) {
        return Err(refuse(OP, "the bias is not one value per column"));
    }
    ctx.emit(&mut |cx| {
        let b = row_of(cx, bias, out.rows, out.width)?;
        let o = cx.read_f32(out)?;
        let s = cx.add(o, b)?;
        cx.write(out, s)
    })
}

pub fn layernorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm";
    expect(OP, x, &[Dtype::Bf16])?;
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = centered(cx, xv, eps)?;
        let w = row_of(cx, weight, x.rows, x.width)?;
        let b = row_of(cx, bias, x.rows, x.width)?;
        let out = cx.mul(n, w)?;
        let out = cx.add(out, b)?;
        cx.write(y, out)
    })
}

/// `(x - mean) / sqrt(var + eps)` per row of an f32 `[rows, width]`.
pub(crate) fn centered(cx: &mut Cx<'_>, x: Val, eps: f32) -> Result<Val, Error> {
    let dims = cx.dims(x).to_vec();
    let n = dims[1] as f64;
    let sum = cx.reduce(x, &[1], Fold::Sum)?;
    let mean = cx.scale(sum, 1.0 / n)?;
    let mean = cx.broadcast(mean, &dims, &[0])?;
    let c = cx.sub(x, mean)?;
    let sq = cx.mul(c, c)?;
    let var = cx.reduce(sq, &[1], Fold::Sum)?;
    let var = cx.scale(var, 1.0 / n)?;
    let var = cx.offset(var, f64::from(eps))?;
    let inv = cx.rsqrt(var);
    let inv = cx.broadcast(inv, &dims, &[0])?;
    Ok(cx.mul(c, inv)?)
}

/// `out = (out - bias) * scale` per column.
pub fn standardize(ctx: &Ctx<'_>, bias: Tensor, scale: Tensor, out: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.standardize";
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
    ctx.emit(&mut |cx| {
        let b = row_of(cx, bias, out.rows, out.width)?;
        let s = row_of(cx, scale, out.rows, out.width)?;
        let o = cx.read_f32(out)?;
        let o = cx.sub(o, b)?;
        let o = cx.mul(o, s)?;
        cx.write(out, o)
    })
}

/// `x *= s`, `s` rounded to `x`'s element first, as the stated scalar
/// lands (kernels-cuda `elemwise::mul_scalar<T>`: bf16, f16, f32).
pub fn mul_scalar(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul_scalar";
    expect(OP, x, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
    let s = match x.dtype {
        Dtype::Bf16 => f32::from_bits(u32::from(crate::hlo::bf16_bits(s)) << 16),
        Dtype::F16 => f16_value(crate::hlo::f16_bits(s)),
        _ => s,
    };
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let v = cx.scale(v, f64::from(s))?;
        cx.write(x, v)
    })
}

/// `x = silu(x * s)`.
pub fn silu_scaled(ctx: &Ctx<'_>, s: f32, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu_scaled";
    expect(OP, x, &[Dtype::Bf16])?;
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let v = cx.scale(v, f64::from(s))?;
        let v = cx.silu(v)?;
        cx.write(x, v)
    })
}

/// `x *= s[0]`, a device scalar.
pub fn scale(ctx: &Ctx<'_>, s: Tensor, x: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.scale";
    expect(OP, x, &[Dtype::Bf16])?;
    ctx.emit(&mut |cx| {
        let sv = cx.read_f32(s)?;
        let n = cx.ty(sv).elements();
        let sv = cx.reshape(sv, &[n])?;
        let sv = cx.slice(sv, &[0], &[1], &[1])?;
        let sv = cx.reshape(sv, &[])?;
        let sv = cx.splat(sv, &[i64::from(x.rows), i64::from(x.width)])?;
        let v = cx.read_f32(x)?;
        let v = cx.mul(v, sv)?;
        cx.write(x, v)
    })
}

/// `(x - mean) / sqrt(var + eps)` per row, no gain or bias. Reference:
/// kernels-metal / kernels-cuda `layernorm_no_scale` (wgpu refuses it).
pub fn layernorm_no_scale(ctx: &Ctx<'_>, x: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm_no_scale";
    expect(OP, x, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
    if x.rows != y.rows || x.width != y.width {
        return Err(refuse(OP, "the normed row lands the rectangle it reads"));
    }
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = centered(cx, xv, eps)?;
        cx.write(y, n)
    })
}

/// `y += x` (rounded), then `out = rms(y) · w` (or `1 + w`) over the whole
/// row, normed from the rounded sum. Reference: kernels-cuda
/// `residual_add_rmsnorm` (wgpu composes the same two kernels).
pub fn residual_add_rmsnorm(
    ctx: &Ctx<'_>,
    x: Tensor,
    y: Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add_rmsnorm";
    expect(OP, y, &[Dtype::Bf16, Dtype::F16])?;
    if x.rows != y.rows || x.width != y.width || out.rows != y.rows || out.width != y.width {
        return Err(refuse(
            OP,
            "the residual, the stream and the normed row share one shape",
        ));
    }
    if weight.elements() != u64::from(y.width) {
        return Err(refuse(OP, "the weight is one gain per column"));
    }
    ctx.emit(&mut |cx| {
        let a = cx.read_f32(y)?;
        let b = cx.read_f32(x)?;
        let s = cx.add(a, b)?;
        let s = landed(cx, s, y.dtype)?;
        cx.write(y, s)?;
        let n = inv_rms_rows(cx, OP, s, y.width, eps)?;
        let w = cx.read(weight)?;
        let g = gain(cx, w, i64::from(y.rows), i64::from(y.width), plus_one)?;
        let o = cx.mul(n, g)?;
        cx.write(out, o)
    })
}

/// The second norm [`rmsnorm_residual_add`] may chain.
#[derive(Clone, Copy, Debug)]
pub struct PostNorm {
    pub weight: Tensor,
    pub plus_one: bool,
    pub eps: f32,
    pub out: Tensor,
}

/// `t = rms(x) · w0`; `y += t`; optionally `scaled = y · s[0]`; optionally
/// `out = rms(last) · w1` (or `1 + w1`) where `last` is `scaled` if present,
/// else `y`. Every stored value rounds where the CUDA kernel stores it, and
/// each later step reads the rounded value. Reference: kernels-cuda
/// `rmsnorm_residual_add` (wgpu and Metal refuse it).
pub fn rmsnorm_residual_add(
    ctx: &Ctx<'_>,
    x: Tensor,
    w0: Tensor,
    eps0: f32,
    t: Tensor,
    y: Tensor,
    scale: Option<(Tensor, Tensor)>,
    post: Option<PostNorm>,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_residual_add";
    expect(OP, y, &[Dtype::Bf16, Dtype::F16])?;
    let same = |p: Tensor| p.rows == y.rows && p.width == y.width;
    if !same(x) || !same(t) {
        return Err(refuse(
            OP,
            "the normed row, its scaled copy and the stream share one shape",
        ));
    }
    if let Some((_, scaled)) = scale
        && !same(scaled)
    {
        return Err(refuse(OP, "the scaled stream rides the stream's shape"));
    }
    if let Some(p) = post
        && !same(p.out)
    {
        return Err(refuse(OP, "the post-normed row rides the stream's shape"));
    }
    let (rows, width) = (i64::from(y.rows), i64::from(y.width));
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = inv_rms_rows(cx, OP, xv, y.width, eps0)?;
        let w = cx.read(w0)?;
        let g = gain(cx, w, rows, width, false)?;
        let tv = cx.mul(n, g)?;
        let tv = landed(cx, tv, t.dtype)?;
        cx.write(t, tv)?;
        let yv = cx.read_f32(y)?;
        let folded = cx.add(yv, tv)?;
        let folded = landed(cx, folded, y.dtype)?;
        cx.write(y, folded)?;
        let mut last = folded;
        if let Some((s, scaled)) = scale {
            let sv = scalar_of(cx, s)?;
            let sv = cx.splat(sv, &[rows, width])?;
            let v = cx.mul(folded, sv)?;
            let v = landed(cx, v, scaled.dtype)?;
            cx.write(scaled, v)?;
            last = v;
        }
        if let Some(p) = post {
            let lv = last;
            let n = inv_rms_rows(cx, OP, lv, y.width, p.eps)?;
            let w = cx.read(p.weight)?;
            let g = gain(cx, w, rows, width, p.plus_one)?;
            let o = cx.mul(n, g)?;
            cx.write(p.out, o)?;
        }
        Ok(())
    })
}

/// The first element of `s`, an f32 scalar.
pub(crate) fn scalar_of(cx: &mut Cx<'_>, s: Tensor) -> Result<Val, Error> {
    let sv = cx.read_f32(s)?;
    let n = cx.ty(sv).elements();
    let sv = cx.reshape(sv, &[n])?;
    let sv = cx.slice(sv, &[0], &[1], &[1])?;
    Ok(cx.reshape(sv, &[])?)
}

pub const MAX_BLEND_BLOCKS: usize = 32;

/// Attention over depth: each candidate (every block, then the prefix) is
/// rms-normed, gained by `weight` and dotted with `proj` into a logit; the
/// softmax over candidates weights their sum. `blocks` are separate
/// `[rows, hidden]` handles (the wgpu engine hands one stacked plane instead;
/// here each block is read on its own). Reference: kernels-wgpu
/// `norm/res_blend.wgsl`.
pub fn res_blend(
    ctx: &Ctx<'_>,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.res_blend";
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
    let same = |p: Tensor| p.rows == y.rows && p.width == y.width;
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
    ctx.emit(&mut |cx| {
        let mut cands = Vec::with_capacity(c as usize);
        for b in blocks.iter().chain(std::iter::once(&prefix)) {
            let v = cx.read_f32(*b)?;
            cands.push(cx.reshape(v, &[rows, 1, hidden])?);
        }
        let all = cx.concat(&cands, 1)?;
        let sq = cx.mul(all, all)?;
        let ss = cx.reduce(sq, &[2], Fold::Sum)?;
        let ms = cx.scale(ss, 1.0 / hidden as f64)?;
        let ms = cx.offset(ms, f64::from(eps))?;
        let inv = cx.rsqrt(ms);
        let inv = cx.broadcast(inv, &[rows, c, hidden], &[0, 1])?;
        let n = cx.mul(all, inv)?;
        let nw = cx.read_f32(weight)?;
        let nw = cx.reshape(nw, &[hidden])?;
        let nw = cx.broadcast(nw, &[rows, c, hidden], &[2])?;
        let pw = cx.read_f32(proj)?;
        let pw = cx.reshape(pw, &[hidden])?;
        let pw = cx.broadcast(pw, &[rows, c, hidden], &[2])?;
        let d = cx.mul(n, nw)?;
        let d = cx.mul(d, pw)?;
        let logits = cx.reduce(d, &[2], Fold::Sum)?;
        let p = cx.softmax(logits, 1)?;
        let p = cx.broadcast(p, &[rows, c, hidden], &[0, 1])?;
        let blend = cx.mul(all, p)?;
        let out = cx.reduce(blend, &[1], Fold::Sum)?;
        cx.write(y, out)
    })
}

/// `v` (f32) as a `dtype` handle stores it, back in f32. XLA may skip an
/// f32→bf16→f32 round trip (excess precision), so where a GPU kernel stores a
/// value and a later step reads the stored one, the bf16 rounding is done
/// here in integer ops XLA keeps: round to nearest even on the upper half,
/// NaN passed through.
pub(crate) fn landed(cx: &mut Cx<'_>, v: Val, dtype: Dtype) -> Result<Val, Error> {
    match dtype {
        Dtype::F32 => Ok(v),
        Dtype::Bf16 => {
            let bits = cx.bitcast(v, Elem::U32)?;
            let sixteen = cx.like_i(bits, 16);
            let one = cx.like_i(bits, 1);
            let lsb = cx.shr(bits, sixteen)?;
            let lsb = cx.and(lsb, one)?;
            let bias = cx.like_i(bits, 0x7FFF);
            let r = cx.add(bits, bias)?;
            let r = cx.add(r, lsb)?;
            let mask = cx.like_i(bits, 0xFFFF_0000);
            let r = cx.and(r, mask)?;
            let r = cx.bitcast(r, Elem::F32)?;
            let nan = cx.compare(crate::hlo::Cmp::Ne, v, v)?;
            Ok(cx.select(nan, v, r)?)
        }
        other => {
            let e = crate::cx::elem_of("elementwise.landed", other)?;
            let n = cx.convert(v, e);
            Ok(cx.convert(n, Elem::F32))
        }
    }
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
