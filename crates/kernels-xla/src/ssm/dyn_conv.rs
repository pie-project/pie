//! The block draft's dynamic conv: a stateless causal depthwise conv whose
//! taps move per row.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Func, Val};
use crate::lanes::{clamp_i, cmp_i, count_le, csr_lanes, flat_i32, iota, nonzero, take};
use crate::tensor::{Landing, Operand, RaggedTensor};

/// A stateless causal depthwise conv whose taps move per row: row `t` of a
/// lane is `Σ_{k ≤ t} (base[side·taps + k, c] + coeff[t, (side·taps + k)·groups
/// + c / group]) · x[t − k, c]`, the lane's rows before its first reading
/// nothing. Reference: kernels-metal `attn::dynconv::block_dyn_conv`.
pub fn block_dyn_conv(
    f: &mut Func,
    x: RaggedTensor<Operand>,
    coeff: Operand,
    base: Operand,
    side: u32,
    taps: u32,
    group: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "ssm.dyn_conv";
    expect(OP, x.data, &[Dtype::Bf16])?;
    csr_lanes(OP, x.indptr)?;
    let channels = nonzero(OP, "the convolution's channel count", x.data.width)?;
    let taps = nonzero(OP, "the tap count this statement states", taps)?;
    let group = nonzero(OP, "the channels sharing one correction", group)?;
    if side > 1 {
        return Err(refuse(
            OP,
            format!("side {side} is stated, and the projection carries two"),
        ));
    }
    if !channels.is_multiple_of(group) {
        return Err(refuse(
            OP,
            format!("{channels} channels are not a whole number of groups of {group}"),
        ));
    }
    let groups = channels / group;
    if coeff.width != 2 * taps * groups || coeff.rows != x.data.rows {
        return Err(refuse(
            OP,
            "the coefficients are not two sides of taps over groups per row",
        ));
    }
    if base.rows != 2 * taps || base.width != channels {
        return Err(refuse(
            OP,
            "the base kernel is not two sides of taps over channels",
        ));
    }
    if y.rows != x.data.rows || y.width != channels {
        return Err(refuse(OP, "the convolution lands the rows it convolves"));
    }
    let (t, c, k, g, gs) = (
        i64::from(x.data.rows),
        i64::from(channels),
        i64::from(taps),
        i64::from(group),
        i64::from(groups),
    );
    let side = i64::from(side);
    let xv = x.data.f32(f);
    let ip = flat_i32(f, x.indptr)?;
    let co = coeff.f32(f);
    let ba = base.f32(f);
    let fq = &mut *f;
    let l = fq.dims(ip)[0] - 1;
    let lo = fq.slice_axis(ip, 0, 0, l)?;
    let ti = iota(fq, t);
    let lane = count_le(fq, lo, ti)?;
    let one = fq.like_i(lane, 1);
    let lane = fq.sub(lane, one)?;
    let lane = clamp_i(fq, lane, 0, l - 1)?;
    let begin = take(fq, lo, lane)?;
    let j = fq.sub(ti, begin)?;
    let zero = fq.const_f(Elem::F32, 0.0, &[]);
    let mut acc = fq.const_f(Elem::F32, 0.0, &[t, c]);
    for tap in 0..k {
        let at = side * k + tap;
        let xs = fq.pad(xv, zero, &[tap, 0], &[0, 0], &[0, 0])?;
        let xs = fq.slice_axis(xs, 0, 0, t)?;
        let live = cmp_i(fq, Cmp::Ge, j, tap)?;
        let live = fq.broadcast(live, &[t, c], &[0])?;
        let zc = fq.like_f(xs, 0.0);
        let xs = fq.select(live, xs, zc)?;
        let b = fq.slice_axis(ba, 0, at, at + 1)?;
        let b = fq.reshape(b, &[c])?;
        let b = fq.broadcast(b, &[t, c], &[1])?;
        let d = fq.slice_axis(co, 1, at * gs, (at + 1) * gs)?;
        let d = fq.broadcast(d, &[t, gs, g], &[0, 1])?;
        let d = fq.reshape(d, &[t, c])?;
        let coef = fq.add(b, d)?;
        let term = fq.mul(coef, xs)?;
        acc = fq.add(acc, term)?;
    }
    Ok(acc)
}
