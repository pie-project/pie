//! The block draft's dynamic conv: a stateless causal depthwise conv whose
//! taps move per row.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem};
use crate::lanes::{clamp_i, cmp_i, count_le, csr_lanes, flat_i32, iota, nonzero, take};
use crate::tensor::{RaggedTensor, Tensor};

/// A stateless causal depthwise conv whose taps move per row: row `t` of a
/// lane is `Σ_{k ≤ t} (base[side·taps + k, c] + coeff[t, (side·taps + k)·groups
/// + c / group]) · x[t − k, c]`, the lane's rows before its first reading
/// nothing. Reference: kernels-metal `attn::dynconv::block_dyn_conv`.
pub fn block_dyn_conv(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    coeff: Tensor,
    base: Tensor,
    side: u32,
    taps: u32,
    group: u32,
    y: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x.data)?;
        let ip = flat_i32(cx, x.indptr)?;
        let co = cx.read_f32(coeff)?;
        let ba = cx.read_f32(base)?;
        let f = cx.func();
        let l = f.dims(ip)[0] - 1;
        let lo = f.slice_axis(ip, 0, 0, l)?;
        let ti = iota(f, t);
        let lane = count_le(f, lo, ti)?;
        let one = f.like_i(lane, 1);
        let lane = f.sub(lane, one)?;
        let lane = clamp_i(f, lane, 0, l - 1)?;
        let begin = take(f, lo, lane)?;
        let j = f.sub(ti, begin)?;
        let zero = f.const_f(Elem::F32, 0.0, &[]);
        let mut acc = f.const_f(Elem::F32, 0.0, &[t, c]);
        for tap in 0..k {
            let at = side * k + tap;
            let xs = f.pad(xv, zero, &[tap, 0], &[0, 0], &[0, 0])?;
            let xs = f.slice_axis(xs, 0, 0, t)?;
            let live = cmp_i(f, Cmp::Ge, j, tap)?;
            let live = f.broadcast(live, &[t, c], &[0])?;
            let zc = f.like_f(xs, 0.0);
            let xs = f.select(live, xs, zc)?;
            let b = f.slice_axis(ba, 0, at, at + 1)?;
            let b = f.reshape(b, &[c])?;
            let b = f.broadcast(b, &[t, c], &[1])?;
            let d = f.slice_axis(co, 1, at * gs, (at + 1) * gs)?;
            let d = f.broadcast(d, &[t, gs, g], &[0, 1])?;
            let d = f.reshape(d, &[t, c])?;
            let coef = f.add(b, d)?;
            let term = f.mul(coef, xs)?;
            acc = f.add(acc, term)?;
        }
        cx.write(y, acc)
    })
}
