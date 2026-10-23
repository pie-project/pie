//! Hyper-connections: `M` residual streams of width `H` ride one
//! `[rows, M·H]` row. Reference: kernels-wgpu `elemwise/hc_*.wgsl`
//! (Metal and CUDA compute the same).
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::Func;
use crate::hlo::{Elem, Fold, Val};
use crate::tensor::{Landing, Operand};

const MAX_HC_MULT: u32 = 8;

fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, Error> {
    if hidden == 0 || wide == 0 || !wide.is_multiple_of(hidden) {
        return Err(refuse(
            op,
            format!(
                "the {wide}-wide row is not a whole number of {hidden}-wide hyper-connection streams"
            ),
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
fn streams3(f: &mut Func, t: Operand, m: u32) -> Result<Val, Error> {
    let v = t.f32(f);
    Ok(f.reshape(
        v,
        &[i64::from(t.rows), i64::from(m), i64::from(t.width / m)],
    )?)
}

/// A flat f32 plane's first `n` values as `[n]`.
fn flat(f: &mut Func, t: Operand, n: i64) -> Result<Val, Error> {
    let v = t.f32(f);
    let all = f.ty(v).elements();
    let v = f.reshape(v, &[all])?;
    Ok(f.slice(v, &[0], &[n], &[1])?)
}

/// `y[n, s·H + h] = x[n, h]` for each of `streams` streams.
pub fn expand(f: &mut Func, x: Operand, streams: u32, y: Landing) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_expand";
    let fan = stream_fan(OP, y.width, x.width)?;
    stated_fan(OP, fan, streams)?;
    if y.rows != x.rows {
        return Err(refuse(OP, "the expansion lands one wide row per row"));
    }
    let v = x.val;
    let (r, h) = (i64::from(x.rows), i64::from(x.width));
    let v = f.broadcast(v, &[r, i64::from(fan), h], &[0, 2])?;
    Ok(v)
}

/// `y = x / rms(x)` over each whole row, widened to f32.
pub fn rmsnorm_f32(f: &mut Func, streams: Operand, eps: f32, y: Landing) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_rmsnorm_f32";
    expect(OP, y, &[Dtype::F32])?;
    if y.rows != streams.rows || y.width != streams.width {
        return Err(refuse(OP, "the normed rectangle is the stream rectangle"));
    }
    let v = streams.f32(f);
    let n = super::norm::inv_rms_rows(f, OP, v, streams.width, eps)?;
    Ok(n)
}

/// `mixes[n, o] = Σ_d normed[n, d] · hc_fn[o, d]`, all f32.
pub fn project(
    f: &mut Func,
    normed: Operand,
    hc_fn: Operand,
    stream_count: u32,
    mixes: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_project";
    expect(OP, mixes, &[Dtype::F32])?;
    if hc_fn.width != normed.width || normed.width == 0 {
        return Err(refuse(
            OP,
            format!(
                "the dynamic plane contracts {} and the stream row is {} wide",
                hc_fn.width, normed.width
            ),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(OP, format!("the stream count is {stream_count}")));
    }
    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    if mixes.width != mix_hc
        || (mix_hc != layer_row && mix_hc != stream_count)
        || mixes.rows != normed.rows
    {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide and a collapse row {stream_count}; \
                 the plane lands {} rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let a = normed.f32(f);
    let w = hc_fn.f32(f);
    let m = f.matmul_nt(a, w, Elem::F32)?;
    Ok(m)
}

/// Pre gates weight the streams into the layer input `x`; post gates and a
/// Sinkhorn-normalized `M × M` comb matrix land for [`fold`]. Returns
/// `(post_mix, comb_mix, x)`, the order they land in.
pub fn gates(
    f: &mut Func,
    normed: Operand,
    streams: Operand,
    scale: Operand,
    base: Operand,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    x: Landing,
    post_mix: Landing,
    comb_mix: Landing,
) -> Result<(Val, Val, Val), Error> {
    const OP: &str = "elemwise.hc_gates";
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
    let mix = normed.f32(f);
    let sc = flat(f, scale, 3)?;
    let bs = flat(f, base, mix_hc)?;
    let part = |f: &mut Func, lo: i64, n: i64, k: i64| -> Result<Val, Error> {
        let v = f.slice_axis(mix, 1, lo, lo + n)?;
        let s = f.slice(sc, &[k], &[k + 1], &[1])?;
        let s = f.reshape(s, &[])?;
        let s = f.splat(s, &[rows, n])?;
        let b = f.slice(bs, &[lo], &[lo + n], &[1])?;
        let b = f.broadcast(b, &[rows, n], &[1])?;
        let v = f.mul(v, s)?;
        Ok(f.add(v, b)?)
    };
    let pre = part(f, 0, m, 0)?;
    let pre = f.sigmoid(pre);
    let pre = f.offset(pre, f64::from(gate_eps))?;
    let post = part(f, m, m, 1)?;
    let post = f.sigmoid(post);
    let post = f.scale(post, f64::from(alpha))?;

    let comb = part(f, 2 * m, m * m, 2)?;
    let comb = f.reshape(comb, &[rows, m, m])?;
    let comb = f.softmax(comb, 2)?;
    let mut comb = f.offset(comb, f64::from(gate_eps))?;
    let norm = |f: &mut Func, c: Val, axis: i64| -> Result<Val, Error> {
        let s = f.reduce(c, &[axis], Fold::Sum)?;
        let s = f.offset(s, f64::from(gate_eps))?;
        let keep = if axis == 2 { [0, 1] } else { [0, 2] };
        let s = f.broadcast(s, &[rows, m, m], &keep)?;
        Ok(f.div(c, s)?)
    };
    comb = norm(f, comb, 1)?;
    for _ in 1..sinkhorn.max(1) {
        comb = norm(f, comb, 2)?;
        comb = norm(f, comb, 1)?;
    }

    let r = streams3(f, streams, fan)?;
    let p = f.broadcast(pre, &[rows, m, h], &[0, 1])?;
    let w = f.mul(r, p)?;
    let li = f.reduce(w, &[1], Fold::Sum)?;
    Ok((post, comb, li))
}

/// `y[n, j, h] = post[n, j] · x[n, h] + Σ_i comb[n, i, j] · streams[n, i, h]`.
pub fn fold(
    f: &mut Func,
    x: Operand,
    streams: Operand,
    post_mix: Operand,
    comb_mix: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_fold";
    expect(OP, y, &[Dtype::Bf16])?;
    if y.rows != streams.rows || y.width != streams.width {
        return Err(refuse(OP, "the fold lands the stream rectangle it mixes"));
    }
    let fan = stream_fan(OP, y.width, x.width)?;
    if post_mix.width != fan || comb_mix.width != fan * fan {
        return Err(refuse(OP, "the gate matrices are [N, M] and [N, M, M]"));
    }
    let (rows, m, h) = (i64::from(y.rows), i64::from(fan), i64::from(x.width));
    let xv = x.f32(f);
    let r = streams3(f, streams, fan)?;
    let post = post_mix.f32(f);
    let comb = comb_mix.f32(f);
    let comb = f.reshape(comb, &[rows, m, m])?;
    // [rows, i, j] · [rows, i, h] over i → [rows, j, h].
    let mixed = f.dot_general(comb, r, &[0], &[0], &[1], &[1], Elem::F32)?;
    let px = f.broadcast(post, &[rows, m, h], &[0, 1])?;
    let xb = f.broadcast(xv, &[rows, m, h], &[0, 2])?;
    let px = f.mul(px, xb)?;
    let out = f.add(px, mixed)?;
    Ok(out)
}

/// `y[n, h] = Σ_i (σ(mixes[n, i]·scale[0] + base[i]) + eps) · streams[n, i, h]`.
pub fn collapse(
    f: &mut Func,
    mixes: Operand,
    streams: Operand,
    scale: Operand,
    base: Operand,
    stream_count: u32,
    hc_eps: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_collapse";
    expect(OP, streams, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, streams.width, y.width)?;
    stated_fan(OP, fan, stream_count)?;
    if mixes.width != fan || mixes.rows != y.rows || base.elements() < u64::from(fan) {
        return Err(refuse(
            OP,
            format!(
                "the collapse folds {fan} streams under a {}-wide mix row",
                mixes.width
            ),
        ));
    }
    let (rows, m, h) = (i64::from(y.rows), i64::from(fan), i64::from(y.width));
    let mix = mixes.f32(f);
    let s = flat(f, scale, 1)?;
    let s = f.reshape(s, &[])?;
    let s = f.splat(s, &[rows, m])?;
    let b = flat(f, base, m)?;
    let b = f.broadcast(b, &[rows, m], &[1])?;
    let l = f.mul(mix, s)?;
    let l = f.add(l, b)?;
    let g = f.sigmoid(l);
    let g = f.offset(g, f64::from(hc_eps))?;
    let r = streams3(f, streams, fan)?;
    let g = f.broadcast(g, &[rows, m, h], &[0, 1])?;
    let w = f.mul(r, g)?;
    let out = f.reduce(w, &[1], Fold::Sum)?;
    Ok(out)
}

/// `y[n, h] = (1/M) Σ_s normed[n, s, h] · σ(gates[n, s, h])`.
pub fn mix(
    f: &mut Func,
    gates: Operand,
    normed: Operand,
    streams: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_mix";
    expect(OP, y, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, normed.width, y.width)?;
    stated_fan(OP, fan, streams)?;
    if gates.width != normed.width || gates.rows != normed.rows || y.rows != normed.rows {
        return Err(refuse(
            OP,
            "the gates ride the normed rectangle element for element",
        ));
    }
    let g = streams3(f, gates, fan)?;
    let v = streams3(f, normed, fan)?;
    let g = f.sigmoid(g);
    let w = f.mul(v, g)?;
    let s = f.reduce(w, &[1], Fold::Sum)?;
    let s = f.scale(s, 1.0 / f64::from(fan))?;
    Ok(s)
}

/// `hyper[n, s, h] += 2σ(gates[n, s] / M) · o[n, h]`.
pub fn inject(
    f: &mut Func,
    o: Operand,
    gates: Operand,
    streams: u32,
    hyper: Operand,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.hc_inject";
    expect(OP, hyper, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, hyper.width, o.width)?;
    stated_fan(OP, fan, streams)?;
    if gates.width != fan || o.rows != hyper.rows {
        return Err(refuse(
            OP,
            "one gate logit per stream, one output row per wide row",
        ));
    }
    let (rows, m, h) = (i64::from(hyper.rows), i64::from(fan), i64::from(o.width));
    let g = gates.f32(f);
    let g = f.slice_axis(g, 0, 0, rows)?;
    let g = f.scale(g, 1.0 / f64::from(fan))?;
    let g = f.sigmoid(g);
    let g = f.scale(g, 2.0)?;
    let g = f.broadcast(g, &[rows, m, h], &[0, 1])?;
    let ov = o.f32(f);
    let ov = f.broadcast(ov, &[rows, m, h], &[0, 2])?;
    let add = f.mul(g, ov)?;
    let hv = streams3(f, hyper, fan)?;
    let out = f.add(hv, add)?;
    Ok(out)
}

/// Per stream, `gate = σ(sign(d)·√max(|d|, 1e-6))` for `d = key·query / √H`
/// (`d = 0` gates by σ(0)); `y[n, s, h] = gate · value[n, h]`.
pub fn ple_gate(
    f: &mut Func,
    key: Operand,
    query: Operand,
    value: Operand,
    streams: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.ple_gate";
    expect(OP, y, &[Dtype::Bf16])?;
    let fan = stream_fan(OP, y.width, value.width)?;
    stated_fan(OP, fan, streams)?;
    if key.width != y.width || query.width != y.width {
        return Err(refuse(
            OP,
            "the key and query ride the stream row they gate",
        ));
    }
    let (rows, m, h) = (i64::from(y.rows), i64::from(fan), i64::from(value.width));
    let k = streams3(f, key, fan)?;
    let q = streams3(f, query, fan)?;
    let kq = f.mul(k, q)?;
    let d = f.reduce(kq, &[2], Fold::Sum)?;
    let isq = f.const_f(Elem::F32, 1.0 / (h as f64).sqrt(), &[]);
    let isq = f.splat(isq, &[rows, m])?;
    let d = f.mul(d, isq)?;
    let a = f.abs(d);
    let floor = f.like_f(a, 1e-6);
    let a = f.max(a, floor)?;
    let mag = f.sqrt(a);
    let sg = f.sign(d);
    let damped = f.mul(sg, mag)?;
    let gate = f.sigmoid(damped);
    let gate = f.broadcast(gate, &[rows, m, h], &[0, 1])?;
    let v = value.f32(f);
    let v = f.broadcast(v, &[rows, m, h], &[0, 2])?;
    let out = f.mul(gate, v)?;
    Ok(out)
}
