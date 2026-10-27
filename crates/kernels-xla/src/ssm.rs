#![allow(clippy::too_many_arguments)]
//! Recurrent state-space scans: the causal depthwise convs (Qwen3.5's
//! `causal_conv1d`, LFM2's short conv), the gated delta rule (Qwen3.5 GDN),
//! Kimi's KDA and Mamba-2's scan (the chunked form with no delta write),
//! each in its decode (one row per lane), chunked (a query CSR of lanes) and
//! committed (a rollback seat's extended window) form.
//!
//! State lives in a [`RecurrentPool`](crate::tensor::RecurrentPool): one slab row per slot. A fire gathers
//! its lanes' rows, computes, and scatters the new rows back; a lane whose
//! write must not land is routed to an out-of-range slot, which XLA's scatter
//! drops.
//!
//! The chunked delta rules run the WY-chunked form (as flash-linear-attention
//! does): every lane is cut into chunks aligned to the lane's start, the
//! intra-chunk `(I + A)⁻¹` is one batched blocked inverse over all chunks,
//! only the chunk-to-chunk state carry is a loop (bounded by the fire's live
//! chunk count, two matmuls a step), and the outputs are formed batched
//! after it.

pub mod dyn_conv;

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};
use crate::lanes::{
    Committed, DROP, LaneTables, Lanes, ci, clamp_i, cmp_i, committed_lanes, count_le,
    decode_lanes, f_reshape_col, flat_f32, flat_i32, full_i, interleave, iota, land_slots,
    loop_upto, nonzero, put_at, ragged_lanes, row_at, slot_rows, take, take_slots,
};
use crate::tensor::{Landing, Operand, RaggedTensor, Stored};

/// The GDN's q/k l2-norm epsilon, as the GPU kernels state it.
const GDN_EPS: f64 = 1e-6;

// --------------------------------------------------------------------- conv

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvOut {
    /// `silu(conv)`: Qwen3.5's `causal_conv1d`.
    Silu,
    /// `conv + x`: the short conv.
    Residual,
}

/// The depthwise causal conv over lanes. `x` `[T, C]` f32, `w` `[C, W]` f32,
/// `past` `[L, hist * C]` f32 (each lane's kept rows, oldest first). Returns
/// `y` `[T, C]` and each lane's next kept rows `[L, hist * C]`: the last
/// `hist` of `past ++ x[begin .. begin + keep]`.
fn conv_lanes(
    f: &mut Func,
    x: Val,
    w: Val,
    past: Val,
    lanes: &Lanes,
    dil: i64,
    out: ConvOut,
    bias: Option<Val>,
) -> Built<(Val, Val)> {
    let (t, c) = (f.dims(x)[0], f.dims(x)[1]);
    let taps = f.dims(w)[1];
    let l = lanes.count(f);
    let hist = f.dims(past)[1] / c;
    let p = f.reshape(past, &[l * hist, c])?;
    // Every row a lane's next window can hold: its kept rows, then x.
    let table = f.concat(&[p, x], 0)?;
    let base = l * hist;

    // Tap k reads x shifted down by its offset; the first rows of each lane
    // (which would read the previous lane) are patched from its kept rows.
    let zero = f.const_f(Elem::F32, 0.0, &[]);
    let mut acc: Option<Val> = None;
    for k in 0..taps {
        let off = (taps - 1 - k) * dil;
        let shifted = if off == 0 {
            x
        } else if off >= t {
            f.const_f(Elem::F32, 0.0, &[t, c])
        } else {
            let xs = f.pad(x, zero, &[off, 0], &[0, 0], &[0, 0])?;
            f.slice_axis(xs, 0, 0, t)?
        };
        let tap = if off == 0 {
            shifted
        } else {
            // Rows begin + s (s < off) of each lane read kept row hist − off + s.
            let o = off.min(hist);
            let s = f.iota(Elem::I32, &[l, o], 1);
            let b = f.broadcast(lanes.begin, &[l, o], &[0])?;
            let row = f.add(b, s)?;
            let sp = f.broadcast(lanes.span, &[l, o], &[0])?;
            let inside = f.compare(Cmp::Lt, s, sp)?;
            // Rows past a lane's span land on a spare row, sliced off after:
            // on a TPU, a scatter into a one-row operand (a decode fire's)
            // of ≥ 512 wide rows does not drop out-of-range updates, it
            // lands them on the row.
            let spare = f.like_i(row, t);
            let row = f.select(inside, row, spare)?;
            let r = f.iota(Elem::I32, &[l, o], 0);
            let hv = f.like_i(r, hist);
            let from = f.mul(r, hv)?;
            let back = f.like_i(s, hist - off);
            let from = f.add(from, back)?;
            let from = f.add(from, s)?;
            let from = f.reshape(from, &[l * o])?;
            let kept = f.take_rows(p, from)?;
            let row = f.reshape(row, &[l * o])?;
            let padded = f.pad(shifted, zero, &[0, 0], &[1, 0], &[0, 0])?;
            let patched = f.put_rows(padded, row, kept, crate::hlo::Combine::Set)?;
            f.slice_axis(patched, 0, 0, t)?
        };
        let wk = f.slice_axis(w, 1, k, k + 1)?;
        let wk = f.reshape(wk, &[c])?;
        let wk = f.broadcast(wk, &[t, c], &[1])?;
        let term = f.mul(tap, wk)?;
        acc = Some(match acc {
            None => term,
            Some(a) => f.add(a, term)?,
        });
    }
    let acc = acc.unwrap_or_else(|| f.const_f(Elem::F32, 0.0, &[t, c]));
    let acc = match bias {
        Some(b) => {
            let b = f.broadcast(b, &[t, c], &[1])?;
            f.add(acc, b)?
        }
        None => acc,
    };
    let y = match out {
        ConvOut::Silu => f.silu(acc)?,
        ConvOut::Residual => f.add(acc, x)?,
    };

    // The kept rows after `keep` tokens.
    let s = f.iota(Elem::I32, &[l, hist], 1);
    let keep = f.broadcast(lanes.keep, &[l, hist], &[0])?;
    let hv = f.const_i(Elem::I32, hist, &[l, hist]);
    let src = f.sub(keep, hv)?;
    let src = f.add(src, s)?;
    let fresh = cmp_i(f, Cmp::Ge, src, 0)?;
    let b = f.broadcast(lanes.begin, &[l, hist], &[0])?;
    let at_x = f.add(b, src)?;
    let bv = f.like_i(at_x, base);
    let at_x = f.add(at_x, bv)?;
    let r = f.iota(Elem::I32, &[l, hist], 0);
    let r = f.mul(r, hv)?;
    let at_p = f.add(src, hv)?;
    let at_p = f.add(at_p, r)?;
    let at = f.select(fresh, at_x, at_p)?;
    let at = f.reshape(at, &[l * hist])?;
    let next = f.take_rows(table, at)?;
    let next = f.reshape(next, &[l, hist * c])?;
    Ok((y, next))
}

/// `(y, conv_state)`: the conv's rows and the slab with each lane's window
/// rolled. The state is rolled in place; the engine refuses a pool that
/// names two banks for it.
fn conv(
    f: &mut Func,
    op: &'static str,
    x: Operand,
    lanes: LaneTables,
    weight: Operand,
    bias: Option<Operand>,
    conv_state: Operand,
    conv_width: u32,
    dilation: u32,
    y: Landing,
    out: ConvOut,
) -> Result<(Val, Val), Error> {
    expect(op, x, &[Dtype::Bf16])?;
    let channels = nonzero(op, "the conv's channel count", x.width)?;
    nonzero(op, "rows", x.rows)?;
    let taps = nonzero(op, "the conv width this statement states", conv_width)?;
    let dil = nonzero(op, "the dilation this statement states", dilation)?;
    if y.rows != x.rows || y.width != x.width {
        return Err(refuse(op, "the conv lands the row it convolves"));
    }
    if weight.elements() != u64::from(channels) * u64::from(taps) {
        return Err(refuse(
            op,
            format!(
                "the weight holds {} taps and the conv states {channels} channels of {taps}",
                weight.elements()
            ),
        ));
    }
    if bias.is_some_and(|b| b.elements() != u64::from(channels)) {
        return Err(refuse(op, "the conv's bias is not one value per channel"));
    }
    let hist = u64::from(taps - 1) * u64::from(dil) + 1;
    let per = slot_rows(
        op,
        &format!(
            "conv (a width of {taps} at dilation {dil} keeps {hist} rows of {channels} channels)"
        ),
        conv_state,
        hist * u64::from(channels),
    )?;
    lanes.check(op, x.rows)?;
    let (t, c, k) = (i64::from(x.rows), i64::from(channels), i64::from(taps));
    let xv = x.f32(f);
    let w = weight.f32(f);
    let w = f.reshape(w, &[c, k])?;
    let bias = bias.map(|b| flat_f32(f, b)).transpose()?;
    let slab = conv_state.val;
    let lanes = lanes.lanes(f, op, t)?;
    let past = take_slots(f, slab, lanes.slot, per)?;
    let past = f.convert(past, Elem::F32);
    let (yv, next) = conv_lanes(f, xv, w, past, &lanes, i64::from(dil), out, bias)?;
    let slab = land_slots(f, slab, lanes.slot, lanes.write, next, per)?;
    Ok((yv, slab))
}

/// One row per lane: `y = silu(conv(state ++ x))`, the window shifts by one.
/// Returns `(y, conv_state)`.
pub fn causal_conv1d(
    f: &mut Func,
    x: Operand,
    weight: Operand,
    bias: Option<Operand>,
    conv_state: Operand,
    slots: Operand,
    conv_width: u32,
    dilation: u32,
    y: Landing,
) -> Result<(Val, Val), Error> {
    conv(
        f,
        "ssm.causal_conv1d",
        x,
        LaneTables::Decode { slots },
        weight,
        bias,
        conv_state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

/// The conv over a query CSR; `slots` is per row (a lane's slot is its first
/// row's).
pub fn causal_conv1d_chunked(
    f: &mut Func,
    x: RaggedTensor<Operand>,
    weight: Operand,
    bias: Option<Operand>,
    conv_state: Operand,
    slots: Operand,
    conv_width: u32,
    dilation: u32,
    y: Landing,
) -> Result<(Val, Val), Error> {
    conv(
        f,
        "ssm.causal_conv1d_chunked",
        x.data,
        LaneTables::Ragged {
            indptr: x.indptr,
            slots,
        },
        weight,
        bias,
        conv_state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

/// The conv over a rollback seat's extended rows: every row is convolved,
/// the window after each lane's committed prefix lands.
pub fn causal_conv1d_committed(
    f: &mut Func,
    x: Operand,
    indptr: Operand,
    seat: Committed<Operand>,
    weight: Operand,
    bias: Option<Operand>,
    conv_state: Operand,
    conv_width: u32,
    dilation: u32,
    y: Landing,
) -> Result<(Val, Val), Error> {
    conv(
        f,
        "ssm.causal_conv1d_committed",
        x,
        LaneTables::Committed { indptr, seat },
        weight,
        bias,
        conv_state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

/// `y = conv(state ++ x) + x`, one row per lane.
/// Reference: kernels-cuda `ssm::conv1d::short_conv` (`ConvOut::Residual`).
pub fn short_conv(
    f: &mut Func,
    x: Operand,
    weight: Operand,
    conv_state: Operand,
    slots: Operand,
    conv_width: u32,
    y: Landing,
) -> Result<(Val, Val), Error> {
    conv(
        f,
        "ssm.short_conv",
        x,
        LaneTables::Decode { slots },
        weight,
        None,
        conv_state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

/// The short conv over a query CSR.
/// Reference: kernels-cuda `ssm::conv1d::short_conv_chunked`.
pub fn short_conv_chunked(
    f: &mut Func,
    x: RaggedTensor<Operand>,
    weight: Operand,
    conv_state: Operand,
    slots: Operand,
    conv_width: u32,
    y: Landing,
) -> Result<(Val, Val), Error> {
    conv(
        f,
        "ssm.short_conv_chunked",
        x.data,
        LaneTables::Ragged {
            indptr: x.indptr,
            slots,
        },
        weight,
        None,
        conv_state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

// ------------------------------------------------------------------ gdn prep

/// `[b | a]` → `[g_log | beta]`: `g = -exp(a_log) · softplus(a + dt_bias)`,
/// `beta = σ(b)`, per value head.
pub fn gdn_prep(
    f: &mut Func,
    ba: Operand,
    dt_bias: Operand,
    a_log: Operand,
    gates: Landing,
) -> Result<Val, Error> {
    const OP: &str = "ssm.gdn_gates";
    expect(OP, ba, &[Dtype::Bf16])?;
    if ba.width == 0 || !ba.width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide `[b | a]` projection does not halve into value heads",
                ba.width
            ),
        ));
    }
    let vh = i64::from(ba.width / 2);
    if gates.rows != ba.rows || gates.width != ba.width {
        return Err(refuse(
            OP,
            "the fused `[g_log | beta]` row rides the projection it is derived from",
        ));
    }
    for (what, t) in [("dt_bias", dt_bias), ("a_log", a_log)] {
        if t.elements() != vh as u64 {
            return Err(refuse(
                OP,
                format!("the {what} bank is not one value per head"),
            ));
        }
    }
    let rows = i64::from(ba.rows);
    let v = ba.f32(f);
    let db = flat_f32(f, dt_bias)?;
    let al = flat_f32(f, a_log)?;
    let fq = &mut *f;
    let b = fq.slice_axis(v, 1, 0, vh)?;
    let a = fq.slice_axis(v, 1, vh, 2 * vh)?;
    let db = fq.broadcast(db, &[rows, vh], &[1])?;
    let z = fq.add(a, db)?;
    let sp = softplus20(fq, z)?;
    let al = fq.exp(al);
    let al = fq.broadcast(al, &[rows, vh], &[1])?;
    let g = fq.mul(al, sp)?;
    let g = fq.neg(g);
    let beta = fq.sigmoid(b);
    let out = fq.concat(&[g, beta], 1)?;
    Ok(out)
}

/// `softplus` with the GPU kernels' linear cut above 20.
fn softplus20(f: &mut Func, z: Val) -> Built<Val> {
    let e = f.exp(z);
    let sp = f.log1p(e);
    let big = f.like_f(z, 20.0);
    let over = f.compare(Cmp::Gt, z, big)?;
    f.select(over, z, sp)
}

// ------------------------------------------------------------- delta rule

/// A delta rule's per-token operands, value heads leading: `q`, `k`
/// `[T, H, dk]` (normed, q scaled), `v` `[T, H, dv]`, the log decay `g`
/// `[T, H, gd]` (`gd` 1 for a scalar decay, `dk` per channel), `beta` `[T, H]`.
#[derive(Clone, Copy)]
struct Heads {
    q: Val,
    k: Val,
    v: Val,
    g: Val,
    beta: Val,
}

/// `x / sqrt(Σx² + eps) · scale` over the last axis of `[T, H, d]`.
fn l2norm(f: &mut Func, x: Val, eps: f64, scale: f64) -> Built<Val> {
    let d = f.dims(x).to_vec();
    let sq = f.mul(x, x)?;
    let s = f.reduce(sq, &[2], Fold::Sum)?;
    let s = f.offset(s, eps)?;
    let inv = f.rsqrt(s);
    let inv = if scale == 1.0 {
        inv
    } else {
        f.scale(inv, scale)?
    };
    let inv = f.broadcast(inv, &d, &[0, 1])?;
    f.mul(x, inv)
}

/// The GDN's heads from the post-conv `[q | k | v]` row and the fused
/// `[g_log | beta]` gates; key heads repeat over their value-head group.
fn gdn_heads(
    f: &mut Func,
    qkv: Val,
    gates: Val,
    hk: i64,
    hv: i64,
    dk: i64,
    dv: i64,
) -> Built<Heads> {
    let t = f.dims(qkv)[0];
    let keys = hk * dk;
    let q = f.slice_axis(qkv, 1, 0, keys)?;
    let k = f.slice_axis(qkv, 1, keys, 2 * keys)?;
    let v = f.slice_axis(qkv, 1, 2 * keys, 2 * keys + hv * dv)?;
    let q = f.reshape(q, &[t, hk, dk])?;
    let k = f.reshape(k, &[t, hk, dk])?;
    let q = l2norm(f, q, GDN_EPS, 1.0 / (dk as f64).sqrt())?;
    let k = l2norm(f, k, GDN_EPS, 1.0)?;
    let group = hv / hk;
    let spread = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.broadcast(x, &[t, hk, group, dk], &[0, 1, 3])?;
        f.reshape(x, &[t, hv, dk])
    };
    let q = spread(f, q)?;
    let k = spread(f, k)?;
    let v = f.reshape(v, &[t, hv, dv])?;
    let g = f.slice_axis(gates, 1, 0, hv)?;
    let g = f.reshape(g, &[t, hv, 1])?;
    let beta = f.slice_axis(gates, 1, hv, 2 * hv)?;
    Ok(Heads { q, k, v, g, beta })
}

/// KDA's heads from `[q | k | v]`, the forget projection and the beta
/// projection: a per-channel log decay `gate_floor · σ(α z)` (or
/// `-α · softplus(z)` without a floor), `z = f + dt_bias`, `α = exp(a_log)`.
fn kda_heads(
    f: &mut Func,
    mixed: Val,
    fp: Val,
    b: Val,
    dt_bias: Val,
    a_log: Val,
    heads: i64,
    d: i64,
    eps: f32,
    floor: f32,
) -> Built<Heads> {
    let t = f.dims(mixed)[0];
    let wide = heads * d;
    let q = f.slice_axis(mixed, 1, 0, wide)?;
    let k = f.slice_axis(mixed, 1, wide, 2 * wide)?;
    let v = f.slice_axis(mixed, 1, 2 * wide, 3 * wide)?;
    let q = f.reshape(q, &[t, heads, d])?;
    let k = f.reshape(k, &[t, heads, d])?;
    let v = f.reshape(v, &[t, heads, d])?;
    let q = l2norm(f, q, f64::from(eps), 1.0 / (d as f64).sqrt())?;
    let k = l2norm(f, k, f64::from(eps), 1.0)?;
    let db = f.reshape(dt_bias, &[wide])?;
    let db = f.broadcast(db, &[t, wide], &[1])?;
    let z = f.add(fp, db)?;
    let z = f.reshape(z, &[t, heads, d])?;
    let alpha = f.exp(a_log);
    let alpha = f.broadcast(alpha, &[t, heads, d], &[1])?;
    let az = f.mul(alpha, z)?;
    let g = if floor != 0.0 {
        let s = f.sigmoid(az);
        f.scale(s, f64::from(floor))?
    } else {
        let sp = softplus20(f, z)?;
        let g = f.mul(alpha, sp)?;
        f.neg(g)
    };
    let beta = f.sigmoid(b);
    Ok(Heads { q, k, v, g, beta })
}

/// Mamba-2's heads from the convolved `xbc` row (`x`, then the groups'
/// `B`, then their `C`) and `dt`: `q = C`, `k = B` (a group's repeated over
/// its heads), `v = x`, `beta = softplus(dt + dt_bias)` and the log decay
/// `g = -exp(a_log) beta`.
fn mamba2_heads(
    f: &mut Func,
    xbc: Val,
    dt: Val,
    a_log: Val,
    dt_bias: Val,
    heads: i64,
    groups: i64,
    n: i64,
    p: i64,
) -> Built<Heads> {
    let t = f.dims(xbc)[0];
    let inner = heads * p;
    let x = f.slice_axis(xbc, 1, 0, inner)?;
    let b = f.slice_axis(xbc, 1, inner, inner + groups * n)?;
    let c = f.slice_axis(xbc, 1, inner + groups * n, inner + 2 * groups * n)?;
    let per = heads / groups;
    let spread = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.reshape(x, &[t, groups, n])?;
        let x = f.broadcast(x, &[t, groups, per, n], &[0, 1, 3])?;
        f.reshape(x, &[t, heads, n])
    };
    let q = spread(f, c)?;
    let k = spread(f, b)?;
    let v = f.reshape(x, &[t, heads, p])?;
    let db = f.broadcast(dt_bias, &[t, heads], &[1])?;
    let z = f.add(dt, db)?;
    let beta = softplus20(f, z)?;
    let a = f.exp(a_log);
    let a = f.broadcast(a, &[t, heads], &[1])?;
    let g = f.mul(a, beta)?;
    let g = f.neg(g);
    let g = f.reshape(g, &[t, heads, 1])?;
    Ok(Heads { q, k, v, g, beta })
}

/// `[.., gd]` → `[.., dk]` along the last axis (a scalar decay repeats).
fn over_keys(f: &mut Func, x: Val, dk: i64) -> Built<Val> {
    let d = f.dims(x).to_vec();
    let r = d.len();
    if d[r - 1] == dk {
        return Ok(x);
    }
    let mut to = d.clone();
    to[r - 1] = dk;
    let lead = f.reshape(x, &d[..r - 1])?;
    let map: Vec<i64> = (0..r as i64 - 1).collect();
    f.broadcast(lead, &to, &map)
}

/// One token per lane: `S ← S·diag(a)`, `δ = β(v − S k)` (`β v` without
/// the delta `rule`), `S ← S + δ kᵀ`, `y = S q`, over `S` `[R, H, dv, dk]`.
fn delta_step(f: &mut Func, h: &Heads, s: Val, rule: bool) -> Built<(Val, Val)> {
    let d = f.dims(s).to_vec();
    let (r, hh, dv, dk) = (d[0], d[1], d[2], d[3]);
    let a = f.exp(h.g);
    let a = over_keys(f, a, dk)?;
    let a = f.broadcast(a, &d, &[0, 1, 3])?;
    let s1 = f.mul(s, a)?;
    let kb = f.broadcast(h.k, &d, &[0, 1, 3])?;
    let delta = if rule {
        let mem = f.mul(s1, kb)?;
        let mem = f.reduce(mem, &[3], Fold::Sum)?;
        f.sub(h.v, mem)?
    } else {
        h.v
    };
    let beta = f.broadcast(h.beta, &[r, hh, dv], &[0, 1])?;
    let delta = f.mul(delta, beta)?;
    let db = f.broadcast(delta, &d, &[0, 1, 2])?;
    let upd = f.mul(db, kb)?;
    let s2 = f.add(s1, upd)?;
    let qb = f.broadcast(h.q, &d, &[0, 1, 3])?;
    let y = f.mul(s2, qb)?;
    let y = f.reduce(y, &[3], Fold::Sum)?;
    Ok((y, s2))
}

/// The pieces of a fire a chunked scan walks: rows `[begin, begin + len)`
/// of segment `s` belong to lane `lane`; the first chunk of a `reset` segment
/// starts from the lane's pool state (else from the state the previous
/// segment left), and a `capture` segment's final state is the lane's.
struct Segs {
    begin: Val,
    len: Val,
    lane: Val,
    reset: Val,
    capture: Val,
}

/// One segment per lane, the lane's whole span.
fn whole_segs(f: &mut Func, lanes: &Lanes) -> Segs {
    let l = lanes.count(f);
    Segs {
        begin: lanes.begin,
        len: lanes.span,
        lane: iota(f, l),
        reset: f.const_i(Elem::Pred, 1, &[l]),
        capture: lanes.write,
    }
}

/// Two segments per lane: the committed prefix (its state captured), then
/// the replayed rest.
fn committed_segs(f: &mut Func, lanes: &Lanes) -> Built<Segs> {
    let l = lanes.count(f);
    let b_begin = f.add(lanes.begin, lanes.keep)?;
    let b_len = f.sub(lanes.span, lanes.keep)?;
    let t = f.const_i(Elem::Pred, 1, &[l]);
    let fresh = cmp_i(f, Cmp::Eq, lanes.keep, 0)?;
    let no = f.const_i(Elem::Pred, 0, &[l]);
    let two = full_i(f, 2, 2 * l);
    let lane = iota(f, 2 * l);
    let lane = f.div(lane, two)?;
    // Interleave preds through i32.
    let tp = |f: &mut Func, a: Val, b: Val| -> Built<Val> {
        let a = f.convert(a, Elem::I32);
        let b = f.convert(b, Elem::I32);
        let ab = interleave(f, a, b)?;
        cmp_i(f, Cmp::Ne, ab, 0)
    };
    Ok(Segs {
        begin: interleave(f, lanes.begin, b_begin)?,
        len: interleave(f, lanes.keep, b_len)?,
        lane,
        reset: tp(f, t, fresh)?,
        capture: tp(f, lanes.write, no)?,
    })
}

/// The chunked delta rule over `segs`, from the lanes' states `init`
/// `[L, H, dv, dk]`. Returns `y` `[T, H, dv]` and the lanes' captured states
/// `[L, H, dv, dk]` (`init` where nothing was captured).
fn delta_chunked(
    f: &mut Func,
    input: &DeltaIn,
    raw: &Raw,
    segs: &Segs,
    init: Val,
    c: i64,
) -> Built<(Val, Val)> {
    let t = f.dims(raw.rows[0])[0];
    let (hh, dk, dv) = input.dims();
    let gd = input.decay_width();
    let rule = input.rule();
    let s = f.dims(segs.begin)[0];
    let nc = (t + c - 1) / c + s;

    // Chunk table: segment s owns chunks [cstart, cend).
    let cm1 = full_i(f, c - 1, s);
    let cv = full_i(f, c, s);
    let nch = f.add(segs.len, cm1)?;
    let nch = f.div(nch, cv)?;
    let cend = f.scan(nch, 0, Fold::Sum)?;
    let cstart = f.sub(cend, nch)?;
    let total = f.reduce(nch, &[0], Fold::Sum)?;
    let ncv = ci(f, nc);
    let total = f.min(total, ncv)?;

    let n = iota(f, nc);
    let seg = count_le(f, cend, n)?;
    let seg = clamp_i(f, seg, 0, s - 1)?;
    let cst = take(f, cstart, seg)?;
    let j = f.sub(n, cst)?;
    let nch_n = take(f, nch, seg)?;
    let len_n = take(f, segs.len, seg)?;
    let beg_n = take(f, segs.begin, seg)?;
    let lane_n = take(f, segs.lane, seg)?;
    let reset = f.convert(segs.reset, Elem::I32);
    let reset_n = take(f, reset, seg)?;
    let capture = f.convert(segs.capture, Elem::I32);
    let cap_n = take(f, capture, seg)?;
    let j0 = cmp_i(f, Cmp::Eq, j, 0)?;
    let r0 = cmp_i(f, Cmp::Ne, reset_n, 0)?;
    let first_n = f.and(j0, r0)?;
    let one = f.like_i(nch_n, 1);
    let lastj = f.sub(nch_n, one)?;
    let jl = f.compare(Cmp::Eq, j, lastj)?;
    let c0 = cmp_i(f, Cmp::Ne, cap_n, 0)?;
    let last_n = f.and(jl, c0)?;

    // Token rows of every chunk position.
    let ic = f.iota(Elem::I32, &[nc, c], 1);
    let jb = f.broadcast(j, &[nc, c], &[0])?;
    let cc = f.const_i(Elem::I32, c, &[nc, c]);
    let local = f.mul(jb, cc)?;
    let local = f.add(local, ic)?;
    let lenb = f.broadcast(len_n, &[nc, c], &[0])?;
    let inside = f.compare(Cmp::Lt, local, lenb)?;
    let tb = f.broadcast(total, &[nc], &[])?;
    let live = f.compare(Cmp::Lt, n, tb)?;
    let live = f.broadcast(live, &[nc, c], &[0])?;
    let valid = f.and(inside, live)?;
    let begb = f.broadcast(beg_n, &[nc, c], &[0])?;
    let row = f.add(begb, local)?;
    let row = clamp_i(f, row, 0, t - 1)?;
    let row = f.reshape(row, &[nc * c])?;

    // Gather the raw rows (bf16 where they arrive so), then form the heads.
    let mut got = Vec::with_capacity(raw.rows.len());
    for &r in &raw.rows {
        got.push(f.take_rows(r, row)?);
    }
    let h = input.heads_from(f, &got, &raw.fixed)?;
    let valid = f.reshape(valid, &[nc * c])?;
    let blank = |f: &mut Func, x: Val, lead: &[i64], perm: &[i64]| -> Built<Val> {
        let d = f.dims(x).to_vec();
        let vb = f.broadcast(valid, &d, &[0])?;
        let zero = f.like_f(x, 0.0);
        let x = f.select(vb, x, zero)?;
        let mut to = lead.to_vec();
        to.extend_from_slice(&d[1..]);
        let x = f.reshape(x, &to)?;
        f.transpose(x, perm)
    };
    let q = blank(f, h.q, &[nc, c], &[0, 2, 1, 3])?;
    let k = blank(f, h.k, &[nc, c], &[0, 2, 1, 3])?;
    let v = blank(f, h.v, &[nc, c], &[0, 2, 1, 3])?;
    let g = blank(f, h.g, &[nc, c], &[0, 2, 1, 3])?;
    let beta = blank(f, h.beta, &[nc, c], &[0, 2, 1])?;

    // Cumulative decay inside each chunk.
    let gcum = f.scan_steps(g, 2, Fold::Sum)?;
    let glast = f.slice_axis(gcum, 2, c - 1, c)?;
    let glast = f.reshape(glast, &[nc, hh, gd])?;
    let e = f.exp(gcum);
    let e = over_keys(f, e, dk)?;
    let qg = f.mul(q, e)?;
    let kg = f.mul(k, e)?;
    let gl = f.broadcast(glast, &[nc, hh, c, gd], &[0, 1, 3])?;
    let tail = f.sub(gl, gcum)?;
    let tail = f.exp(tail);
    let tail = over_keys(f, tail, dk)?;
    let kt = f.mul(k, tail)?;
    let gc = f.exp(glast);

    // Intra-chunk pair terms.
    let ti = f.iota(Elem::I32, &[c, c], 0);
    let si = f.iota(Elem::I32, &[c, c], 1);
    let strict = f.compare(Cmp::Gt, ti, si)?;
    let incl = f.compare(Cmp::Ge, ti, si)?;
    let diag = f.compare(Cmp::Eq, ti, si)?;
    let sq = [nc, hh, c, c];
    let strict4 = f.broadcast(strict, &sq, &[2, 3])?;
    let incl4 = f.broadcast(incl, &sq, &[2, 3])?;
    let zero4 = f.const_f(Elem::F32, 0.0, &sq);
    let (kk, qk) = if gd == 1 {
        let gs = f.reshape(gcum, &[nc, hh, c])?;
        let gt4 = f.broadcast(gs, &sq, &[0, 1, 2])?;
        let gs4 = f.broadcast(gs, &sq, &[0, 1, 3])?;
        let dlt = f.sub(gt4, gs4)?;
        let dlt = f.select(incl4, dlt, zero4)?;
        let ex = f.exp(dlt);
        let qk = f.dot_general(q, k, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
        let kk = if rule {
            let kk = f.dot_general(k, k, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
            f.mul(kk, ex)?
        } else {
            zero4
        };
        (kk, f.mul(qk, ex)?)
    } else {
        let five = [nc, hh, c, c, dk];
        let gt5 = f.broadcast(gcum, &five, &[0, 1, 2, 4])?;
        let gs5 = f.broadcast(gcum, &five, &[0, 1, 3, 4])?;
        let dlt = f.sub(gt5, gs5)?;
        let incl5 = f.broadcast(incl, &five, &[2, 3])?;
        let zero5 = f.const_f(Elem::F32, 0.0, &five);
        let dlt = f.select(incl5, dlt, zero5)?;
        let ex = f.exp(dlt);
        let ks = f.broadcast(k, &five, &[0, 1, 3, 4])?;
        let kt5 = f.broadcast(k, &five, &[0, 1, 2, 4])?;
        let qt5 = f.broadcast(q, &five, &[0, 1, 2, 4])?;
        let kse = f.mul(ks, ex)?;
        let kk = f.mul(kt5, kse)?;
        let kk = f.reduce(kk, &[4], Fold::Sum)?;
        let qk = f.mul(qt5, kse)?;
        let qk = f.reduce(qk, &[4], Fold::Sum)?;
        (kk, qk)
    };
    let p = f.select(incl4, qk, zero4)?;
    let bv = f.broadcast(beta, &[nc, hh, c, dv], &[0, 1, 2])?;
    let rv = f.mul(v, bv)?;
    // Without the delta rule `(I + A)⁻¹` is `I`, and there is no `w`.
    let (ut, w) = if rule {
        let bt = f.broadcast(beta, &sq, &[0, 1, 2])?;
        let a = f.mul(bt, kk)?;
        let a = f.select(strict4, a, zero4)?;
        let eye = f.convert(diag, Elem::F32);
        let eye = f.broadcast(eye, &sq, &[2, 3])?;
        let lower = f.add(eye, a)?;
        let bk = f.broadcast(beta, &[nc, hh, c, dk], &[0, 1, 2])?;
        let rk = f.mul(kg, bk)?;
        let rhs = f.concat(&[rv, rk], 3)?;
        let _ = lower;
        let inv = unit_lower_inverse(f, a)?;
        let x = f.dot_general(inv, rhs, &[0, 1], &[0, 1], &[3], &[2], Elem::F32)?;
        let ut = f.slice_axis(x, 3, 0, dv)?;
        (ut, Some(f.slice_axis(x, 3, dv, dv + dk)?))
    } else {
        (rv, None)
    };

    // The chunk-to-chunk carry: only the state walks sequentially; each
    // chunk's end state is kept, and the outputs are formed after the loop.
    let sz = [hh, dv, dk];
    let s0 = f.const_f(Elem::F32, 0.0, &sz);
    let e0 = f.const_f(Elem::F32, 0.0, &[nc, hh, dv, dk]);
    let first_i = f.convert(first_n, Elem::I32);
    let out = loop_upto(
        f,
        total,
        &[s0, e0],
        &[first_i, lane_n, w.unwrap_or(ut), ut, kt, gc, init],
        |f, i, cr, inv| {
            let (st, ends) = (cr[0], cr[1]);
            let (first_i, lane_n, w, ut, kt, gc, init) =
                (inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6]);
            let first = row_at(f, first_i, i)?;
            let first = cmp_i(f, Cmp::Ne, first, 0)?;
            let lane = row_at(f, lane_n, i)?;
            let from = row_at(f, init, lane)?;
            let s0 = f.select(first, from, st)?;
            let un = row_at(f, ut, i)?;
            let kn = row_at(f, kt, i)?;
            let gn = row_at(f, gc, i)?;
            let u = if rule {
                let wn = row_at(f, w, i)?;
                let ws = f.dot_general(wn, s0, &[0], &[0], &[2], &[2], Elem::F32)?;
                f.sub(un, ws)?
            } else {
                un
            };
            let gn = over_keys(f, gn, dk)?;
            let gn = f.broadcast(gn, &sz, &[0, 2])?;
            let decayed = f.mul(s0, gn)?;
            let upd = f.dot_general(u, kn, &[0], &[0], &[1], &[1], Elem::F32)?;
            let s1 = f.add(decayed, upd)?;
            let ends = put_at(f, ends, s1, i)?;
            Ok(vec![s1, ends])
        },
    )?;
    let ends = out[1];

    // Each chunk's start state: its lane's pool state on a reset, else the
    // previous chunk's end.
    let l = f.dims(init)[0];
    let stride = hh * dv * dk;
    let zero1 = f.const_f(Elem::F32, 0.0, &[1, hh, dv, dk]);
    let prev = f.slice_axis(ends, 0, 0, nc - 1)?;
    let prev = f.concat(&[zero1, prev], 0)?;
    let init2 = f.reshape(init, &[l, stride])?;
    let from = f.take_rows(init2, lane_n)?;
    let from = f.reshape(from, &[nc, hh, dv, dk])?;
    let fb = f.broadcast(first_n, &[nc, hh, dv, dk], &[0])?;
    let starts = f.select(fb, from, prev)?;
    let u = match w {
        Some(w) => {
            let ws = f.dot_general(w, starts, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
            f.sub(ut, ws)?
        }
        None => ut,
    };
    let o1 = f.dot_general(qg, starts, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
    let o2 = f.dot_general(p, u, &[0, 1], &[0, 1], &[3], &[2], Elem::F32)?;
    let o = f.add(o1, o2)?;

    // Each lane's captured state: the end of its capture segment's last chunk.
    let none = f.const_i(Elem::I32, -1, &[l]);
    let drop = f.like_i(lane_n, DROP);
    let at = f.select(last_n, lane_n, drop)?;
    let none = f_reshape_col(f, none)?;
    let chunk_ids = f_reshape_col(f, n)?;
    let fin_at = f.put_rows(none, at, chunk_ids, crate::hlo::Combine::Max)?;
    let fin_at = f.reshape(fin_at, &[l])?;
    let got = cmp_i(f, Cmp::Ge, fin_at, 0)?;
    let ends2 = f.reshape(ends, &[nc, stride])?;
    let fin = f.take_rows(ends2, fin_at)?;
    let gb = f.broadcast(got, &[l, stride], &[0])?;
    let fin = f.select(gb, fin, init2)?;
    let fin = f.reshape(fin, &[l, hh, dv, dk])?;

    // Back to token rows.
    let ti = iota(f, t);
    let sg = count_le(f, segs.begin, ti)?;
    let one = f.like_i(sg, 1);
    let sg = f.sub(sg, one)?;
    let sg = clamp_i(f, sg, 0, s - 1)?;
    let b = take(f, segs.begin, sg)?;
    let local = f.sub(ti, b)?;
    let cvt = f.like_i(local, c);
    let ch = f.div(local, cvt)?;
    let within = f.rem(local, cvt)?;
    let cs = take(f, cstart, sg)?;
    let ch = f.add(cs, ch)?;
    let pos = f.mul(ch, cvt)?;
    let pos = f.add(pos, within)?;
    let pos = clamp_i(f, pos, 0, nc * c - 1)?;
    let o = f.transpose(o, &[0, 2, 1, 3])?;
    let o = f.reshape(o, &[nc * c, hh * dv])?;
    let y = f.take_rows(o, pos)?;
    let y = f.reshape(y, &[t, hh, dv])?;
    Ok((y, fin))
}

/// `(I + A)⁻¹` for a strictly lower-triangular `A` `[n0, n1, C, C]`:
/// forward substitution on the 16-wide diagonal blocks, then exact block
/// merges, `[[X11, 0], [-X22 A21 X11, X22]]`, doubling to `C`. (XLA's own
/// `triangular_solve` walks rows on TPU and is several times slower.)
fn unit_lower_inverse(f: &mut Func, a: Val) -> Built<Val> {
    let d = f.dims(a).to_vec();
    let (n0, n1, c) = (d[0], d[1], d[2]);
    let b0 = c.min(16);
    let m = c / b0;
    let mut blocks = Vec::with_capacity(m as usize);
    for i in 0..m {
        let lo = i * b0;
        let blk = f.slice(
            a,
            &[0, 0, lo, lo],
            &[n0, n1, lo + b0, lo + b0],
            &[1, 1, 1, 1],
        )?;
        blocks.push(f.reshape(blk, &[n0, n1, 1, b0, b0])?);
    }
    let diag = f.concat(&blocks, 2)?;
    // Row i of the inverse: e_i − Σ_{j<i} A[i, j] X[j, :].
    let mut rows: Vec<Val> = Vec::with_capacity(b0 as usize);
    for i in 0..b0 {
        let hot: Vec<f64> = (0..b0).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
        let e = f.const_floats(Elem::F32, &hot, &[b0])?;
        let e = f.broadcast(e, &[n0, n1, m, 1, b0], &[4])?;
        let row = if i == 0 {
            e
        } else {
            let xs = f.concat(&rows, 3)?;
            let ai = f.slice(
                diag,
                &[0, 0, 0, i, 0],
                &[n0, n1, m, i + 1, i],
                &[1, 1, 1, 1, 1],
            )?;
            let ai = f.reshape(ai, &[n0, n1, m, i])?;
            let ai = f.broadcast(ai, &[n0, n1, m, i, b0], &[0, 1, 2, 3])?;
            let prod = f.mul(ai, xs)?;
            let sum = f.reduce(prod, &[3], Fold::Sum)?;
            let sum = f.reshape(sum, &[n0, n1, m, 1, b0])?;
            f.sub(e, sum)?
        };
        rows.push(row);
    }
    let mut inv = f.concat(&rows, 3)?;
    let (mut b, mut mm) = (b0, m);
    while mm > 1 {
        let half = mm / 2;
        let pairs = f.reshape(inv, &[n0, n1, half, 2, b, b])?;
        let x11 = f.slice(
            pairs,
            &[0, 0, 0, 0, 0, 0],
            &[n0, n1, half, 1, b, b],
            &[1; 6],
        )?;
        let x11 = f.reshape(x11, &[n0, n1, half, b, b])?;
        let x22 = f.slice(
            pairs,
            &[0, 0, 0, 1, 0, 0],
            &[n0, n1, half, 2, b, b],
            &[1; 6],
        )?;
        let x22 = f.reshape(x22, &[n0, n1, half, b, b])?;
        let mut a21s = Vec::with_capacity(half as usize);
        for p in 0..half {
            let (r0, c0) = ((2 * p + 1) * b, 2 * p * b);
            let blk = f.slice(a, &[0, 0, r0, c0], &[n0, n1, r0 + b, c0 + b], &[1, 1, 1, 1])?;
            a21s.push(f.reshape(blk, &[n0, n1, 1, b, b])?);
        }
        let a21 = f.concat(&a21s, 2)?;
        let t = f.dot_general(a21, x11, &[0, 1, 2], &[0, 1, 2], &[4], &[3], Elem::F32)?;
        let t = f.dot_general(x22, t, &[0, 1, 2], &[0, 1, 2], &[4], &[3], Elem::F32)?;
        let x21 = f.neg(t);
        let zero = f.like_f(x11, 0.0);
        let top = f.concat(&[x11, zero], 4)?;
        let bot = f.concat(&[x21, x22], 4)?;
        inv = f.concat(&[top, bot], 3)?;
        b *= 2;
        mm = half;
    }
    f.reshape(inv, &[n0, n1, c, c])
}

/// Which lanes a delta-rule entry walks.
#[derive(Clone, Copy)]
/// The operands a delta rule reads: GDN's or KDA's.
enum DeltaIn {
    Gdn {
        qkv: Operand,
        gates: Operand,
        hk: i64,
        hv: i64,
        dk: i64,
        dv: i64,
    },
    Kda {
        mixed: Operand,
        f: Operand,
        b: Operand,
        dt_bias: Operand,
        a_log: Operand,
        heads: i64,
        d: i64,
        eps: f32,
        floor: f32,
    },
    Mamba2 {
        xbc: Operand,
        dt: Operand,
        a_log: Operand,
        dt_bias: Operand,
        heads: i64,
        groups: i64,
        n: i64,
        p: i64,
    },
}

impl DeltaIn {
    fn dims(&self) -> (i64, i64, i64) {
        match *self {
            Self::Gdn { hv, dk, dv, .. } => (hv, dk, dv),
            Self::Kda { heads, d, .. } => (heads, d, d),
            Self::Mamba2 { heads, n, p, .. } => (heads, n, p),
        }
    }

    fn rows(&self) -> u32 {
        match *self {
            Self::Gdn { qkv, .. } => qkv.rows,
            Self::Kda { mixed, .. } => mixed.rows,
            Self::Mamba2 { xbc, .. } => xbc.rows,
        }
    }

    /// The delta rule's `β(v − S k)` write; Mamba-2 writes `β v`.
    fn rule(&self) -> bool {
        !matches!(self, Self::Mamba2 { .. })
    }

    fn chunk(&self, rows: i64, lanes: i64, c: crate::tuning::SsmChunks) -> i64 {
        match self {
            Self::Gdn { .. } | Self::Mamba2 { .. } if rows >= c.long_chunk * lanes.max(1) => {
                c.long_chunk
            }
            Self::Gdn { .. } | Self::Mamba2 { .. } => c.chunk,
            Self::Kda { .. } => c.kda_chunk,
        }
    }

    /// 1 for a scalar decay per head, the key width for a per-channel one.
    fn decay_width(&self) -> i64 {
        match *self {
            Self::Gdn { .. } | Self::Mamba2 { .. } => 1,
            Self::Kda { d, .. } => d,
        }
    }

    /// The per-row operands as they are stored, and the per-head banks.
    fn raw(&self, fq: &mut Func) -> Result<Raw, Error> {
        Ok(match *self {
            Self::Gdn { qkv, gates, .. } => Raw {
                rows: vec![qkv.val, gates.f32(fq)],
                fixed: vec![],
            },
            Self::Kda {
                mixed,
                f,
                b,
                dt_bias,
                a_log,
                ..
            } => Raw {
                rows: vec![mixed.val, f.val, b.val],
                fixed: vec![flat_f32(fq, dt_bias)?, flat_f32(fq, a_log)?],
            },
            Self::Mamba2 {
                xbc,
                dt,
                a_log,
                dt_bias,
                ..
            } => Raw {
                rows: vec![xbc.val, dt.val],
                fixed: vec![flat_f32(fq, a_log)?, flat_f32(fq, dt_bias)?],
            },
        })
    }

    /// The heads of `rows` (the [`Raw::rows`], or rows gathered from them).
    fn heads_from(&self, f: &mut Func, rows: &[Val], fixed: &[Val]) -> Built<Heads> {
        let rows: Vec<Val> = rows.iter().map(|&r| f.convert(r, Elem::F32)).collect();
        match *self {
            Self::Gdn { hk, hv, dk, dv, .. } => gdn_heads(f, rows[0], rows[1], hk, hv, dk, dv),
            Self::Kda {
                heads,
                d,
                eps,
                floor,
                ..
            } => kda_heads(
                f, rows[0], rows[1], rows[2], fixed[0], fixed[1], heads, d, eps, floor,
            ),
            Self::Mamba2 {
                heads,
                groups,
                n,
                p,
                ..
            } => mamba2_heads(f, rows[0], rows[1], fixed[0], fixed[1], heads, groups, n, p),
        }
    }
}

/// A delta rule's operands as read: `rows` are `[T, *]` per-token planes,
/// `fixed` per-head banks.
struct Raw {
    rows: Vec<Val>,
    fixed: Vec<Val>,
}

/// `(y, state)`: the recurrence's value rows and the slab with each lane's
/// state landed.
fn delta(
    f: &mut Func,
    op: &'static str,
    input: DeltaIn,
    lanes: LaneTables,
    state: Operand,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    let (hh, dk, dv) = input.dims();
    let rows = input.rows();
    nonzero(op, "rows", rows)?;
    if y.rows != rows || i64::from(y.width) != hh * dv {
        return Err(refuse(op, "the recurrence lands one value plane per row"));
    }
    let stride = hh * dv * dk;
    let per = slot_rows(
        op,
        &format!("state ({hh} heads of {dv}x{dk})"),
        state,
        stride as u64,
    )?;
    lanes.check(op, rows)?;
    let lanes_n = match lanes {
        LaneTables::Decode { .. } => i64::from(rows),
        LaneTables::Ragged { indptr, .. } | LaneTables::Committed { indptr, .. } => {
            indptr.elements() as i64 - 1
        }
    };
    let t = i64::from(rows);
    let chunk = input.chunk(t, lanes_n, tuning.ssm);
    let chunk = chunk.min(i64::from(rows.next_power_of_two().max(8)));
    let raw = input.raw(f)?;
    let slab = state.val;
    let (yv, slab) = match lanes {
        LaneTables::Decode { slots } => {
            let slots = flat_i32(f, slots)?;
            let fq = &mut *f;
            let lanes = decode_lanes(fq, slots, t)?;
            let heads = input.heads_from(fq, &raw.rows, &raw.fixed)?;
            // Over a slot-blocked bank, rounds of lanes, each gathered,
            // stepped (in f32) and landed before the next, so a round's
            // states stay on chip. A padded lane sharing the sink slot
            // with an earlier round's reads what that round left there,
            // which is garbage either way.
            let round = if per == 1 {
                t
            } else {
                (tuning.decode_round_bytes() / (stride * 4)).clamp(1, t)
            };
            let mut slab = slab;
            let mut ys = Vec::new();
            let mut lo = 0;
            while lo < t {
                let hi = (lo + round).min(t);
                let cut = |fq: &mut Func, x: Val| fq.slice_axis(x, 0, lo, hi);
                let part = if lo == 0 && hi == t {
                    heads
                } else {
                    Heads {
                        q: cut(fq, heads.q)?,
                        k: cut(fq, heads.k)?,
                        v: cut(fq, heads.v)?,
                        g: cut(fq, heads.g)?,
                        beta: cut(fq, heads.beta)?,
                    }
                };
                let slot = cut(fq, lanes.slot)?;
                let write = cut(fq, lanes.write)?;
                let s = take_slots(fq, slab, slot, per)?;
                let s = fq.convert(s, Elem::F32);
                let s = fq.reshape(s, &[hi - lo, hh, dv, dk])?;
                let (yv, s) = delta_step(fq, &part, s, input.rule())?;
                let s = fq.reshape(s, &[hi - lo, stride])?;
                slab = land_slots(fq, slab, slot, write, s, per)?;
                ys.push(yv);
                lo = hi;
            }
            let yv = if ys.len() == 1 {
                ys[0]
            } else {
                fq.concat(&ys, 0)?
            };
            (yv, slab)
        }
        LaneTables::Ragged { indptr, .. } | LaneTables::Committed { indptr, .. } => {
            let ip = flat_i32(f, indptr)?;
            let (lanes, segs) = match lanes {
                LaneTables::Committed { seat, .. } => {
                    let rep = flat_i32(f, seat.replay)?;
                    let com = flat_i32(f, seat.commit)?;
                    let sl = flat_i32(f, seat.slots)?;
                    let lanes = committed_lanes(op, f, ip, rep, com, sl, seat.lane0)?;
                    let segs = committed_segs(f, &lanes)?;
                    (lanes, segs)
                }
                LaneTables::Ragged { slots, .. } | LaneTables::Decode { slots } => {
                    let slots = flat_i32(f, slots)?;
                    let lanes = ragged_lanes(f, ip, slots)?;
                    let segs = whole_segs(f, &lanes);
                    (lanes, segs)
                }
            };
            let fq = &mut *f;
            let l = lanes.count(fq);
            let init = take_slots(fq, slab, lanes.slot, per)?;
            let init = fq.convert(init, Elem::F32);
            let init = fq.reshape(init, &[l, hh, dv, dk])?;
            let (yv, fin) = delta_chunked(fq, &input, &raw, &segs, init, chunk)?;
            let fin = fq.reshape(fin, &[l, stride])?;
            (yv, land_slots(fq, slab, lanes.slot, lanes.write, fin, per)?)
        }
    };
    Ok((yv, slab))
}

fn gdn_in(
    op: &'static str,
    qkv: Operand,
    gates: Operand,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Result<DeltaIn, Error> {
    expect(op, qkv, &[Dtype::Bf16])?;
    nonzero(op, "the key heads this statement states", k_heads)?;
    nonzero(op, "the value heads this statement states", v_heads)?;
    nonzero(op, "the key head width this statement states", k_dim)?;
    nonzero(op, "the value head width this statement states", v_dim)?;
    if !v_heads.is_multiple_of(k_heads) {
        return Err(refuse(
            op,
            format!("the {v_heads} value heads are not a whole number of the {k_heads} key heads"),
        ));
    }
    let (hk, hv, dk, dv) = (
        i64::from(k_heads),
        i64::from(v_heads),
        i64::from(k_dim),
        i64::from(v_dim),
    );
    if i64::from(qkv.width()) != 2 * hk * dk + hv * dv {
        return Err(refuse(
            op,
            "the post-convolution qkv's row is not the four stated head numbers",
        ));
    }
    if gates.rows() != qkv.rows() || i64::from(gates.width()) != 2 * hv {
        return Err(refuse(
            op,
            "the fused `[g_log | beta]` row is not two entries per value head",
        ));
    }
    Ok(DeltaIn::Gdn {
        qkv,
        gates,
        hk,
        hv,
        dk,
        dv,
    })
}

/// One token per lane of the gated delta rule; `y` is the f32 accumulator
/// (`z` gates it later, in the gated rmsnorm). Returns `(y, state)`.
pub fn gated_delta(
    f: &mut Func,
    qkv: Operand,
    gates: Operand,
    state: Operand,
    slots: Operand,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    const OP: &str = "ssm.gated_delta";
    let input = gdn_in(OP, qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
    delta(f, OP, input, LaneTables::Decode { slots }, state, y, tuning)
}

/// The gated delta rule over a query CSR, chunked; `slots` is per row.
pub fn gated_delta_chunked(
    f: &mut Func,
    qkv: RaggedTensor<Operand>,
    gates: Operand,
    state: Operand,
    slots: Operand,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    const OP: &str = "ssm.gated_delta_chunked";
    let input = gdn_in(OP, qkv.data, gates, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = LaneTables::Ragged {
        indptr: qkv.indptr,
        slots,
    };
    delta(f, OP, input, lanes, state, y, tuning)
}

/// The gated delta rule over a rollback seat's extended rows: every row's
/// `y`, and the state after each lane's committed prefix. The GPU's `work`
/// scratch plane is not an operand here.
pub fn gated_delta_committed(
    f: &mut Func,
    qkv: Operand,
    indptr: Operand,
    seat: Committed<Operand>,
    gates: Operand,
    state: Operand,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    const OP: &str = "ssm.gated_delta_committed";
    let input = gdn_in(OP, qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = LaneTables::Committed { indptr, seat };
    delta(f, OP, input, lanes, state, y, tuning)
}

fn kda_in(
    op: &'static str,
    mixed: Operand,
    f: Operand,
    b: Operand,
    dt_bias: Operand,
    a_log: Operand,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
) -> Result<DeltaIn, Error> {
    expect(op, mixed, &[Dtype::Bf16])?;
    nonzero(op, "the KDA heads this statement states", heads)?;
    nonzero(op, "the KDA head width this statement states", head_dim)?;
    let (h, d) = (i64::from(heads), i64::from(head_dim));
    if i64::from(mixed.width()) != 3 * h * d {
        return Err(refuse(
            op,
            "the post-convolution `[q | k | v]` row is not three head planes",
        ));
    }
    if f.rows() != mixed.rows() || i64::from(f.width()) != h * d {
        return Err(refuse(
            op,
            "the forget projection's row is not one head plane",
        ));
    }
    if b.rows() != mixed.rows() || i64::from(b.width()) != h {
        return Err(refuse(
            op,
            "the beta projection's row is not one entry per head",
        ));
    }
    if dt_bias.elements() != (h * d) as u64 || a_log.elements() != h as u64 {
        return Err(refuse(
            op,
            "the decay banks are not one plane / one value per head",
        ));
    }
    Ok(DeltaIn::Kda {
        mixed,
        f,
        b,
        dt_bias,
        a_log,
        heads: h,
        d,
        eps: norm_eps,
        floor: gate_floor,
    })
}

/// One token per lane of KDA (per-channel decay delta rule). Returns
/// `(y, state)`.
pub fn kda_step(
    f: &mut Func,
    mixed: Operand,
    forget: Operand,
    b: Operand,
    dt_bias: Operand,
    a_log: Operand,
    state: Operand,
    slots: Operand,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    const OP: &str = "ssm.kda";
    let input = kda_in(
        OP, mixed, forget, b, dt_bias, a_log, heads, head_dim, norm_eps, gate_floor,
    )?;
    delta(f, OP, input, LaneTables::Decode { slots }, state, y, tuning)
}

/// KDA over a query CSR, chunked; `slots` is per row.
pub fn kda_chunked(
    f: &mut Func,
    mixed: RaggedTensor<Operand>,
    forget: Operand,
    b: Operand,
    dt_bias: Operand,
    a_log: Operand,
    state: Operand,
    slots: Operand,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    const OP: &str = "ssm.kda_chunked";
    let input = kda_in(
        OP, mixed.data, forget, b, dt_bias, a_log, heads, head_dim, norm_eps, gate_floor,
    )?;
    let lanes = LaneTables::Ragged {
        indptr: mixed.indptr,
        slots,
    };
    delta(f, OP, input, lanes, state, y, tuning)
}

/// KDA over a rollback seat's extended rows. The GPU's `work` scratch plane
/// is not an operand here.
pub fn kda_committed(
    f: &mut Func,
    mixed: Operand,
    indptr: Operand,
    seat: Committed<Operand>,
    forget: Operand,
    b: Operand,
    dt_bias: Operand,
    a_log: Operand,
    state: Operand,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    const OP: &str = "ssm.kda_committed";
    let input = kda_in(
        OP, mixed, forget, b, dt_bias, a_log, heads, head_dim, norm_eps, gate_floor,
    )?;
    let lanes = LaneTables::Committed { indptr, seat };
    delta(f, OP, input, lanes, state, y, tuning)
}

// ------------------------------------------------------------------ mamba-2

/// Mamba-2's per-head banks (`a_log`, `dt_bias`, `d`, one value a head) and
/// its gated group norm's row-wide `norm` weight and `eps`.
#[derive(Clone, Copy)]
pub struct Mamba2 {
    pub a_log: Operand,
    pub dt_bias: Operand,
    pub d: Operand,
    pub norm: Operand,
    pub eps: f32,
    pub heads: u32,
    pub groups: u32,
    pub state_dim: u32,
}

/// The scan's operands: `xbc` (`x`, then each group's `B`, then its `C`),
/// `dt` a value a head, `z` and the landing a head's width a head.
fn mamba2_in(
    op: &'static str,
    xbc: Operand,
    dt: Operand,
    z: Operand,
    m: &Mamba2,
    y: Landing,
) -> Result<DeltaIn, Error> {
    for t in [xbc, dt, z] {
        expect(op, t, &[Dtype::Bf16])?;
    }
    let heads = nonzero(op, "the Mamba-2 heads this statement states", m.heads)?;
    let groups = nonzero(op, "the Mamba-2 groups this statement states", m.groups)?;
    let n = nonzero(
        op,
        "the Mamba-2 state width this statement states",
        m.state_dim,
    )?;
    if !heads.is_multiple_of(groups) || z.width == 0 || !z.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "{heads} heads in {groups} groups over a {}-wide z is no whole number of heads a group",
                z.width
            ),
        ));
    }
    let (h, g, n, p) = (
        i64::from(heads),
        i64::from(groups),
        i64::from(n),
        i64::from(z.width / heads),
    );
    if i64::from(xbc.width) != h * p + 2 * g * n
        || dt.width != heads
        || dt.rows != xbc.rows
        || z.rows != xbc.rows
        || y.rows != xbc.rows
        || y.width != z.width
    {
        return Err(refuse(
            op,
            format!(
                "Mamba-2 reads `x` then each group's `B` and `C` and a dt a head, and lands z's \
                 width; this fire states {heads} heads in {groups} groups of a {}-wide state \
                 over a {}-wide xbc, a {}-wide dt, a {}-wide z and a {}-wide output",
                m.state_dim, xbc.width, dt.width, z.width, y.width
            ),
        ));
    }
    for (what, t, want) in [
        ("a_log", m.a_log, h),
        ("dt_bias", m.dt_bias, h),
        ("d", m.d, h),
        ("norm", m.norm, h * p),
    ] {
        if t.elements() != want as u64 {
            return Err(refuse(
                op,
                format!("the {what} bank holds {} values, not {want}", t.elements()),
            ));
        }
    }
    Ok(DeltaIn::Mamba2 {
        xbc,
        dt,
        a_log: m.a_log,
        dt_bias: m.dt_bias,
        heads: h,
        groups: g,
        n,
        p,
    })
}

/// `(C S + D x) * silu(z)` from the scan's `C S` `[T, H, P]`, rounded to
/// bf16 where the GPU lands it, then RMS-normed a group of heads and gained
/// by the row-wide `norm`.
fn mamba2_out(
    f: &mut Func,
    op: &'static str,
    cs: Val,
    xbc: Operand,
    z: Operand,
    m: &Mamba2,
) -> Result<Val, Error> {
    let (t, inner) = (i64::from(xbc.rows), i64::from(z.width));
    let (h, p) = (i64::from(m.heads), inner / i64::from(m.heads));
    let y = f.reshape(cs, &[t, inner])?;
    let x = xbc.f32(f);
    let x = f.slice_axis(x, 1, 0, inner)?;
    let d = flat_f32(f, m.d)?;
    let d = f.broadcast(d, &[t, h, p], &[1])?;
    let d = f.reshape(d, &[t, inner])?;
    let dx = f.mul(x, d)?;
    let y = f.add(y, dx)?;
    let zv = z.f32(f);
    let gate = f.silu(zv)?;
    let y = f.mul(y, gate)?;
    let y = f.convert(y, Elem::Bf16);
    let y = f.convert(y, Elem::F32);
    let normed = crate::elemwise::norm::inv_rms_rows(f, op, y, z.width / m.groups, m.eps)?;
    let w = flat_f32(f, m.norm)?;
    let w = f.broadcast(w, &[t, inner], &[1])?;
    Ok(f.mul(normed, w)?)
}

fn mamba2_run(
    f: &mut Func,
    op: &'static str,
    xbc: Operand,
    dt: Operand,
    z: Operand,
    m: Mamba2,
    lanes: LaneTables,
    state: Operand,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    let input = mamba2_in(op, xbc, dt, z, &m, y)?;
    let (cs, slab) = delta(f, op, input, lanes, state, y, tuning)?;
    Ok((mamba2_out(f, op, cs, xbc, z, &m)?, slab))
}

/// One Mamba-2 step per lane over the convolved `xbc` rows: `dt =
/// softplus(dt + dt_bias)`, `S ← exp(-exp(a_log) dt) S + dt x Bᵀ`, lands
/// `(C S + D x) * silu(z)`, group-normed. Returns `(y, state)`.
/// Reference: kernels-cuda `ssm::mamba2`.
pub fn mamba2(
    f: &mut Func,
    xbc: Operand,
    dt: Operand,
    z: Operand,
    m: Mamba2,
    state: Operand,
    slots: Operand,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    let lanes = LaneTables::Decode { slots };
    mamba2_run(f, "ssm.mamba2", xbc, dt, z, m, lanes, state, y, tuning)
}

/// Mamba-2 over a query CSR, chunked; `slots` is per row.
/// Reference: kernels-cuda `ssm::mamba2_chunked`.
pub fn mamba2_chunked(
    f: &mut Func,
    xbc: RaggedTensor<Operand>,
    dt: Operand,
    z: Operand,
    m: Mamba2,
    state: Operand,
    slots: Operand,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    let lanes = LaneTables::Ragged {
        indptr: xbc.indptr,
        slots,
    };
    mamba2_run(
        f,
        "ssm.mamba2_chunked",
        xbc.data,
        dt,
        z,
        m,
        lanes,
        state,
        y,
        tuning,
    )
}

/// Mamba-2 over a rollback seat's extended rows: every row's `y`, and the
/// state after each lane's committed prefix.
pub fn mamba2_committed(
    f: &mut Func,
    xbc: Operand,
    indptr: Operand,
    seat: Committed<Operand>,
    dt: Operand,
    z: Operand,
    m: Mamba2,
    state: Operand,
    y: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<(Val, Val), Error> {
    let lanes = LaneTables::Committed { indptr, seat };
    mamba2_run(
        f,
        "ssm.mamba2_committed",
        xbc,
        dt,
        z,
        m,
        lanes,
        state,
        y,
        tuning,
    )
}

// ------------------------------------------------------------ dynamic conv
