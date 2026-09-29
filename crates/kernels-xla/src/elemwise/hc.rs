//! Hyper-connections: `M` residual streams of width `H` ride one
//! `[rows, M·H]` row. Reference: kernels-wgpu `elemwise/hc_*.wgsl`
//! (Metal and CUDA compute the same).
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Fold, Val};
use crate::tensor::Tensor;

const MAX_HC_MULT: u32 = 8;

fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, Error> {
    if hidden == 0 || wide == 0 || !wide.is_multiple_of(hidden) {
        return Err(refuse(
            op,
            format!("the {wide}-wide row is not a whole number of {hidden}-wide hyper-connection streams"),
        ));
    }
    let fan = wide / hidden;
    if fan > MAX_HC_MULT {
        return Err(refuse(
            op,
            format!("the stream count is {fan}, above the {MAX_HC_MULT} the mixers take"),
        ));
    }
    Ok(fan)
}

fn stated_fan(op: &'static str, fan: u32, streams: u32) -> Result<(), Error> {
    if fan != streams {
        return Err(refuse(
            op,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    Ok(())
}

/// A `[rows, M·H]` handle read as f32 `[rows, M, H]`.
fn streams3(cx: &mut Cx<'_>, t: Tensor, m: u32) -> Result<Val, Error> {
    let v = cx.read_f32(t)?;
    Ok(cx.reshape(v, &[i64::from(t.rows), i64::from(m), i64::from(t.width / m)])?)
}

/// A flat f32 plane's first `n` values as `[n]`.
fn flat(cx: &mut Cx<'_>, t: Tensor, n: i64) -> Result<Val, Error> {
    let v = cx.read_f32(t)?;
    let all = cx.ty(v).elements();
    let v = cx.reshape(v, &[all])?;
    Ok(cx.slice(v, &[0], &[n], &[1])?)
}

/// `y[n, s·H + h] = x[n, h]` for each of `streams` streams.
pub fn expand(ctx: &Ctx<'_>, x: Tensor, streams: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_expand";
    let fan = stream_fan(OP, y.width, x.width)?;
    stated_fan(OP, fan, streams)?;
    if y.rows != x.rows {
        return Err(refuse(OP, "the expansion lands one wide row per row"));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read(x)?;
        let (r, h) = (i64::from(x.rows), i64::from(x.width));
        let v = cx.broadcast(v, &[r, i64::from(fan), h], &[0, 2])?;
        cx.write(y, v)
    })
}

/// `y = x / rms(x)` over each whole row, widened to f32.
pub fn rmsnorm_f32(ctx: &Ctx<'_>, streams: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_rmsnorm_f32";
    expect(OP, y, &[Dtype::F32])?;
    if y.rows != streams.rows || y.width != streams.width {
        return Err(refuse(OP, "the normed rectangle is the stream rectangle"));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(streams)?;
        let n = super::norm::inv_rms_rows(cx, OP, v, streams.width, eps)?;
        cx.write(y, n)
    })
}

/// `mixes[n, o] = Σ_d normed[n, d] · hc_fn[o, d]`, all f32.
pub fn project(
    ctx: &Ctx<'_>,
    normed: Tensor,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_project";
    expect(OP, mixes, &[Dtype::F32])?;
    if hc_fn.width != normed.width || normed.width == 0 {
        return Err(refuse(
            OP,
            format!("the dynamic plane contracts {} and the stream row is {} wide", hc_fn.width, normed.width),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(OP, format!("the stream count is {stream_count}")));
    }
    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    if mixes.width != mix_hc || (mix_hc != layer_row && mix_hc != stream_count) || mixes.rows != normed.rows {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide and a collapse row {stream_count}; \
                 the plane lands {} rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let a = cx.read_f32(normed)?;
        let w = cx.read_f32(hc_fn)?;
        let m = cx.matmul_nt(a, w, Elem::F32)?;
        cx.write(mixes, m)
    })
}

/// Pre gates weight the streams into the layer input `x`; post gates and a
/// Sinkhorn-normalized `M × M` comb matrix land for [`fold`].
pub fn gates(
    ctx: &Ctx<'_>,
    normed: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    x: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_gates";
    expect(OP, streams, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, streams.width, x.width)?;
    stated_fan(OP, fan, stream_count)?;
    let m = i64::from(fan);
    let mix_hc = 2 * m + m * m;
    if i64::from(normed.width) != mix_hc
        || post_mix.width != fan
        || comb_mix.width != fan * fan
        || scale.elements() < 3
        || base.elements() < mix_hc as u64
    {
        return Err(refuse(
            OP,
            "the mix row is [2M + M²], the gates [M] and [M, M], the scales three, the bases one per mix",
        ));
    }
    let rows = i64::from(x.rows);
    let h = i64::from(x.width);
    ctx.emit(&mut |cx| {
        let mix = cx.read_f32(normed)?;
        let sc = flat(cx, scale, 3)?;
        let bs = flat(cx, base, mix_hc)?;
        let part = |cx: &mut Cx<'_>, lo: i64, n: i64, k: i64| -> Result<Val, Error> {
            let v = cx.slice_axis(mix, 1, lo, lo + n)?;
            let s = cx.slice(sc, &[k], &[k + 1], &[1])?;
            let s = cx.reshape(s, &[])?;
            let s = cx.splat(s, &[rows, n])?;
            let b = cx.slice(bs, &[lo], &[lo + n], &[1])?;
            let b = cx.broadcast(b, &[rows, n], &[1])?;
            let v = cx.mul(v, s)?;
            Ok(cx.add(v, b)?)
        };
        let pre = part(cx, 0, m, 0)?;
        let pre = cx.sigmoid(pre);
        let pre = cx.offset(pre, f64::from(gate_eps))?;
        let post = part(cx, m, m, 1)?;
        let post = cx.sigmoid(post);
        let post = cx.scale(post, f64::from(alpha))?;
        cx.write(post_mix, post)?;

        let comb = part(cx, 2 * m, m * m, 2)?;
        let comb = cx.reshape(comb, &[rows, m, m])?;
        let comb = cx.softmax(comb, 2)?;
        let mut comb = cx.offset(comb, f64::from(gate_eps))?;
        let norm = |cx: &mut Cx<'_>, c: Val, axis: i64| -> Result<Val, Error> {
            let s = cx.reduce(c, &[axis], Fold::Sum)?;
            let s = cx.offset(s, f64::from(gate_eps))?;
            let keep = if axis == 2 { [0, 1] } else { [0, 2] };
            let s = cx.broadcast(s, &[rows, m, m], &keep)?;
            Ok(cx.div(c, s)?)
        };
        comb = norm(cx, comb, 1)?;
        for _ in 1..sinkhorn.max(1) {
            comb = norm(cx, comb, 2)?;
            comb = norm(cx, comb, 1)?;
        }
        cx.write(comb_mix, comb)?;

        let r = streams3(cx, streams, fan)?;
        let p = cx.broadcast(pre, &[rows, m, h], &[0, 1])?;
        let w = cx.mul(r, p)?;
        let li = cx.reduce(w, &[1], Fold::Sum)?;
        cx.write(x, li)
    })
}

/// `y[n, j, h] = post[n, j] · x[n, h] + Σ_i comb[n, i, j] · streams[n, i, h]`.
pub fn fold(
    ctx: &Ctx<'_>,
    x: Tensor,
    streams: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_fold";
    expect(OP, y, &[Dtype::Bf16])?;
    if y.rows != streams.rows || y.width != streams.width {
        return Err(refuse(OP, "the fold lands the stream rectangle it mixes"));
    }
    let fan = stream_fan(OP, y.width, x.width)?;
    if post_mix.width != fan || comb_mix.width != fan * fan {
        return Err(refuse(OP, "the gate matrices are [N, M] and [N, M, M]"));
    }
    let (rows, m, h) = (i64::from(y.rows), i64::from(fan), i64::from(x.width));
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let r = streams3(cx, streams, fan)?;
        let post = cx.read_f32(post_mix)?;
        let comb = cx.read_f32(comb_mix)?;
        let comb = cx.reshape(comb, &[rows, m, m])?;
        // [rows, i, j] · [rows, i, h] over i → [rows, j, h].
        let mixed = cx.dot_general(comb, r, &[0], &[0], &[1], &[1], Elem::F32)?;
        let px = cx.broadcast(post, &[rows, m, h], &[0, 1])?;
        let xb = cx.broadcast(xv, &[rows, m, h], &[0, 2])?;
        let px = cx.mul(px, xb)?;
        let out = cx.add(px, mixed)?;
        cx.write(y, out)
    })
}

/// `y[n, h] = Σ_i (σ(mixes[n, i]·scale[0] + base[i]) + eps) · streams[n, i, h]`.
pub fn collapse(
    ctx: &Ctx<'_>,
    mixes: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    hc_eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_collapse";
    expect(OP, streams, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, streams.width, y.width)?;
    stated_fan(OP, fan, stream_count)?;
    if mixes.width != fan || mixes.rows != y.rows || base.elements() < u64::from(fan) {
        return Err(refuse(
            OP,
            format!("the collapse folds {fan} streams under a {}-wide mix row", mixes.width),
        ));
    }
    let (rows, m, h) = (i64::from(y.rows), i64::from(fan), i64::from(y.width));
    ctx.emit(&mut |cx| {
        let mix = cx.read_f32(mixes)?;
        let s = flat(cx, scale, 1)?;
        let s = cx.reshape(s, &[])?;
        let s = cx.splat(s, &[rows, m])?;
        let b = flat(cx, base, m)?;
        let b = cx.broadcast(b, &[rows, m], &[1])?;
        let l = cx.mul(mix, s)?;
        let l = cx.add(l, b)?;
        let g = cx.sigmoid(l);
        let g = cx.offset(g, f64::from(hc_eps))?;
        let r = streams3(cx, streams, fan)?;
        let g = cx.broadcast(g, &[rows, m, h], &[0, 1])?;
        let w = cx.mul(r, g)?;
        let out = cx.reduce(w, &[1], Fold::Sum)?;
        cx.write(y, out)
    })
}

/// `y[n, h] = (1/M) Σ_s normed[n, s, h] · σ(gates[n, s, h])`.
pub fn mix(ctx: &Ctx<'_>, gates: Tensor, normed: Tensor, streams: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_mix";
    expect(OP, y, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, normed.width, y.width)?;
    stated_fan(OP, fan, streams)?;
    if gates.width != normed.width || gates.rows != normed.rows || y.rows != normed.rows {
        return Err(refuse(OP, "the gates ride the normed rectangle element for element"));
    }
    ctx.emit(&mut |cx| {
        let g = streams3(cx, gates, fan)?;
        let v = streams3(cx, normed, fan)?;
        let g = cx.sigmoid(g);
        let w = cx.mul(v, g)?;
        let s = cx.reduce(w, &[1], Fold::Sum)?;
        let s = cx.scale(s, 1.0 / f64::from(fan))?;
        cx.write(y, s)
    })
}

/// `hyper[n, s, h] += 2σ(gates[n, s] / M) · o[n, h]`.
pub fn inject(ctx: &Ctx<'_>, o: Tensor, gates: Tensor, streams: u32, hyper: Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_inject";
    expect(OP, hyper, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, hyper.width, o.width)?;
    stated_fan(OP, fan, streams)?;
    if gates.width != fan || o.rows != hyper.rows {
        return Err(refuse(OP, "one gate logit per stream, one output row per wide row"));
    }
    let (rows, m, h) = (i64::from(hyper.rows), i64::from(fan), i64::from(o.width));
    ctx.emit(&mut |cx| {
        let g = cx.read_f32(gates)?;
        let g = cx.slice_axis(g, 0, 0, rows)?;
        let g = cx.scale(g, 1.0 / f64::from(fan))?;
        let g = cx.sigmoid(g);
        let g = cx.scale(g, 2.0)?;
        let g = cx.broadcast(g, &[rows, m, h], &[0, 1])?;
        let ov = cx.read_f32(o)?;
        let ov = cx.broadcast(ov, &[rows, m, h], &[0, 2])?;
        let add = cx.mul(g, ov)?;
        let hv = streams3(cx, hyper, fan)?;
        let out = cx.add(hv, add)?;
        cx.write(hyper, out)
    })
}

/// Per stream, `gate = σ(sign(d)·√max(|d|, 1e-6))` for `d = key·query / √H`
/// (`d = 0` gates by σ(0)); `y[n, s, h] = gate · value[n, h]`.
pub fn ple_gate(
    ctx: &Ctx<'_>,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.ple_gate";
    expect(OP, y, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, y.width, value.width)?;
    stated_fan(OP, fan, streams)?;
    if key.width != y.width || query.width != y.width {
        return Err(refuse(OP, "the key and query ride the stream row they gate"));
    }
    let (rows, m, h) = (i64::from(y.rows), i64::from(fan), i64::from(value.width));
    ctx.emit(&mut |cx| {
        let k = streams3(cx, key, fan)?;
        let q = streams3(cx, query, fan)?;
        let kq = cx.mul(k, q)?;
        let d = cx.reduce(kq, &[2], Fold::Sum)?;
        let isq = cx.const_f(Elem::F32, 1.0 / (h as f64).sqrt(), &[]);
        let isq = cx.splat(isq, &[rows, m])?;
        let d = cx.mul(d, isq)?;
        let a = cx.abs(d);
        let floor = cx.like_f(a, 1e-6);
        let a = cx.max(a, floor)?;
        let mag = cx.sqrt(a);
        let sg = cx.sign(d);
        let damped = cx.mul(sg, mag)?;
        let gate = cx.sigmoid(damped);
        let gate = cx.broadcast(gate, &[rows, m, h], &[0, 1])?;
        let v = cx.read_f32(value)?;
        let v = cx.broadcast(v, &[rows, m, h], &[0, 2])?;
        let out = cx.mul(gate, v)?;
        cx.write(y, out)
    })
}
