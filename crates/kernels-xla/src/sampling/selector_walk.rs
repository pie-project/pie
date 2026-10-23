//! The block draft's candidate selector walk.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::Ctx;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold};
use crate::lanes::{clamp_i, cmp_i, csr_lanes, flat_i32, nonzero, take};
use crate::tensor::{RaggedTensor, Tensor};

/// The drafter's greedy walk over each lane's candidate rows: row `t` scores
/// its `k` candidates `unary[t, c] + Σ_d pred[prev, d] · hp[t, d] · succ[cand, d]`
/// (the bilinear term only when both ids are in the vocabulary; `hp` 1 when
/// absent), picks the first best, and the pick is the next row's `prev`
/// (the lane's first `prev` is `tokens[begin]`). With `first` 1 the anchor
/// row takes its first candidate unscored. Rows outside every lane keep
/// `picks`. Reference: kernels-metal `attn::selector::walk`.
pub fn selector_walk(
    ctx: &Ctx<'_>,
    cand: RaggedTensor,
    unary: Tensor,
    hp: Option<Tensor>,
    tokens: Tensor,
    pred: Tensor,
    succ: Tensor,
    first: u32,
    picks: Tensor,
) -> Result<(), Error> {
    const OP: &str = "sampling.selector_walk";
    if first > 1 {
        return Err(refuse(
            OP,
            "a span's anchor is row 0 and its first mask row 1",
        ));
    }
    if cand.data.dtype != Dtype::I32 || tokens.dtype != Dtype::I32 || picks.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            "the candidates, the tokens and the picks are i32",
        ));
    }
    if unary.dtype != Dtype::F32 {
        return Err(refuse(OP, "the unary logits are f32"));
    }
    csr_lanes(OP, cand.indptr)?;
    let k = nonzero(OP, "candidates a slot", cand.data.width)?;
    let rank = nonzero(OP, "the codebooks' rank", pred.width)?;
    if succ.width != rank || hp.is_some_and(|h| h.width != rank) || pred.rows != succ.rows {
        return Err(refuse(
            OP,
            "the codebooks and the projected hidden disagree on rank or vocabulary",
        ));
    }
    let vocab = nonzero(OP, "the codebooks' vocabulary", pred.rows)?;
    let rows = cand.data.rows;
    if unary.rows != rows
        || unary.width != k
        || picks.rows != rows
        || tokens.rows != rows
        || hp.is_some_and(|h| h.rows != rows)
    {
        return Err(refuse(
            OP,
            "unary, hp, tokens and picks carry one row per candidate row",
        ));
    }
    let (t, kk, rk, v, first) = (
        i64::from(rows),
        i64::from(k),
        i64::from(rank),
        i64::from(vocab),
        i64::from(first),
    );
    ctx.emit(&mut |cx| {
        let cd = cx.read(cand.data)?;
        let ip = flat_i32(cx, cand.indptr)?;
        let un = cx.read_f32(unary)?;
        let hv = match hp {
            Some(h) => Some(cx.read_f32(h)?),
            None => None,
        };
        let tk = flat_i32(cx, tokens)?;
        let pr = cx.read_f32(pred)?;
        let sc = cx.read_f32(succ)?;
        let pk = flat_i32(cx, picks)?;
        let f = cx.func();
        let l = f.dims(ip)[0] - 1;
        let lo = f.slice_axis(ip, 0, 0, l)?;
        let hi = f.slice_axis(ip, 0, 1, l + 1)?;
        let span = f.sub(hi, lo)?;
        let zero = f.like_i(span, 0);
        let span = f.max(span, zero)?;
        let has = cmp_i(f, Cmp::Gt, span, 0)?;
        let at0 = clamp_i(f, lo, 0, t - 1)?;
        let mut picks0 = pk;
        if first > 0 {
            let c0 = f.slice_axis(cd, 1, 0, 1)?;
            let c0 = f.reshape(c0, &[t])?;
            let anchor = take(f, c0, at0)?;
            let drop = f.like_i(lo, crate::lanes::DROP);
            let at = f.select(has, lo, drop)?;
            let p2 = f.reshape(picks0, &[t, 1])?;
            let a2 = f.reshape(anchor, &[l, 1])?;
            let p2 = f.put_rows(p2, at, a2, crate::hlo::Combine::Set)?;
            picks0 = f.reshape(p2, &[t])?;
        }
        let prev0 = take(f, tk, at0)?;
        let most = f.reduce(span, &[0], Fold::Max)?;
        let fv = crate::lanes::ci(f, first);
        let count = f.sub(most, fv)?;
        let hv = match hv {
            Some(h) => h,
            None => f.const_f(Elem::F32, 1.0, &[t, rk]),
        };
        let out = crate::lanes::loop_upto(
            f,
            count,
            &[picks0, prev0],
            &[cd, un, hv, pr, sc, lo, span],
            |f, i, cr, inv| {
                let (picks, prev) = (cr[0], cr[1]);
                let (cd, un, hv, pr, sc, lo, span) =
                    (inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6]);
                let fv = crate::lanes::ci(f, first);
                let jj = f.add(i, fv)?;
                let jb = f.broadcast(jj, &[l], &[])?;
                let active = f.compare(Cmp::Lt, jb, span)?;
                let row = f.add(lo, jb)?;
                let row_c = clamp_i(f, row, 0, t - 1)?;
                let cids = f.take_rows(cd, row_c)?;
                let u = f.take_rows(un, row_c)?;
                let h = f.take_rows(hv, row_c)?;
                let pc = clamp_i(f, prev, 0, v - 1)?;
                let a = f.take_rows(pr, pc)?;
                let flat = f.reshape(cids, &[l * kk])?;
                let fc = clamp_i(f, flat, 0, v - 1)?;
                let b = f.take_rows(sc, fc)?;
                let b = f.reshape(b, &[l, kk, rk])?;
                let ah = f.mul(a, h)?;
                let ah = f.broadcast(ah, &[l, kk, rk], &[0, 2])?;
                let prod = f.mul(ah, b)?;
                let part = f.reduce(prod, &[2], Fold::Sum)?;
                let p_lo = cmp_i(f, Cmp::Ge, prev, 0)?;
                let p_hi = cmp_i(f, Cmp::Lt, prev, v)?;
                let p_ok = f.and(p_lo, p_hi)?;
                let p_ok = f.broadcast(p_ok, &[l, kk], &[0])?;
                let c_lo = cmp_i(f, Cmp::Ge, cids, 0)?;
                let c_hi = cmp_i(f, Cmp::Lt, cids, v)?;
                let c_ok = f.and(c_lo, c_hi)?;
                let live = f.and(p_ok, c_ok)?;
                let zp = f.like_f(part, 0.0);
                let part = f.select(live, part, zp)?;
                let score = f.add(u, part)?;
                // The first strict maximum; a NaN never wins past slot 0.
                let nan = f.compare(Cmp::Ne, score, score)?;
                let ninf = f.like_f(score, f64::NEG_INFINITY);
                let clean = f.select(nan, ninf, score)?;
                let (best, _) = f.argmax(clean, 1, Elem::I32)?;
                let s0 = f.slice_axis(nan, 1, 0, 1)?;
                let s0 = f.reshape(s0, &[l])?;
                let z = f.like_i(best, 0);
                let best = f.select(s0, z, best)?;
                let ki = f.iota(Elem::I32, &[l, kk], 1);
                let bb = f.broadcast(best, &[l, kk], &[0])?;
                let hot = f.compare(Cmp::Eq, ki, bb)?;
                let low = f.like_i(cids, i64::from(i32::MIN));
                let pick = f.select(hot, cids, low)?;
                let pick = f.reduce(pick, &[1], Fold::Max)?;
                let drop = f.like_i(row, crate::lanes::DROP);
                let at = f.select(active, row, drop)?;
                let p2 = f.reshape(picks, &[t, 1])?;
                let k2 = f.reshape(pick, &[l, 1])?;
                let p2 = f.put_rows(p2, at, k2, crate::hlo::Combine::Set)?;
                let picks = f.reshape(p2, &[t])?;
                let prev = f.select(active, pick, prev)?;
                Ok(vec![picks, prev])
            },
        )?;
        cx.write(picks, out[0])
    })
}
