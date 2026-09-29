#![allow(clippy::too_many_arguments)]
//! Per-layer-embedding n-gram hashing (Gemma 3n PLE, DeepSeek Engram): each
//! token's id and the ids before it (the lane's kept window first) hash to
//! one table row per head.
//!
//! The kept window is a recurrent state of `ngram - 1` i32 cells per slot,
//! each `id + 1` (0: nothing yet, read as `eos`). The GPU kernels read the
//! hash constants from a staged u64 plane; here they are constants of the
//! program, so there is no `hash` operand.

use dtype::Dtype;

use super::ssm::{
    Committed, Lanes, clamp_i, cmp_i, committed_lanes, count_le, csr_lanes, decode_lanes,
    flat_i32, land_rows, nonzero, ragged_lanes, take,
};
use crate::cx::Ctx;
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const MAX_NGRAM: usize = 4;

const MAX_HEADS: usize = 32;

struct Hash<'a> {
    eos: i64,
    mults: &'a [u64],
    primes: &'a [u64],
    offsets: &'a [u64],
    hpn: usize,
}

impl Hash<'_> {
    fn ngram(&self) -> i64 {
        self.mults.len() as i64
    }
    fn span(&self) -> i64 {
        self.ngram() - 1
    }
    fn heads(&self) -> i64 {
        self.primes.len() as i64
    }
}

fn hash<'a>(
    op: &'static str,
    eos: u32,
    mults: &'a [u64],
    primes: &'a [u64],
    offsets: &'a [u64],
    heads_per_ngram: u32,
) -> Result<Hash<'a>, Error> {
    if mults.len() < 2 || mults.len() > MAX_NGRAM {
        return Err(refuse(
            op,
            format!("{} multipliers do not fit the 2..={MAX_NGRAM}-gram range", mults.len()),
        ));
    }
    if primes.len() != offsets.len() || primes.is_empty() || primes.len() > MAX_HEADS {
        return Err(refuse(
            op,
            format!(
                "{} primes against {} offsets do not fit the {MAX_HEADS}-head ceiling",
                primes.len(),
                offsets.len()
            ),
        ));
    }
    nonzero(op, "the heads per n-gram this statement states", heads_per_ngram)?;
    if primes.len() != (mults.len() - 1) * heads_per_ngram as usize {
        return Err(refuse(
            op,
            format!(
                "{} heads against {} n-gram orders of {heads_per_ngram}",
                primes.len(),
                mults.len() - 1
            ),
        ));
    }
    if primes.contains(&0) {
        return Err(refuse(op, "a hash prime is zero"));
    }
    Ok(Hash {
        eos: i64::from(eos as i32),
        mults,
        primes,
        offsets,
        hpn: heads_per_ngram as usize,
    })
}

fn planes(
    op: &'static str,
    h: &Hash<'_>,
    ids: Tensor,
    state: &RecurrentPool,
    map: Option<Tensor>,
    out: Tensor,
) -> Result<(), Error> {
    if ids.dtype != Dtype::I32 || out.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the hasher reads i32 token ids and lands i32 table rows, not {:?} into {:?}",
                ids.dtype, out.dtype
            ),
        ));
    }
    if i64::from(out.width) != h.heads() {
        return Err(refuse(op, "the landing is not one column per hashed head"));
    }
    if state.state.dtype != Dtype::I32 || i64::from(state.state.width) != h.span() {
        return Err(refuse(
            op,
            "the window a lane keeps is the n-gram context, one i32 per trailing id",
        ));
    }
    if let Some(m) = map
        && m.dtype != Dtype::I64
    {
        return Err(refuse(
            op,
            format!("the id map is {:?}, and the hasher maps through i64", m.dtype),
        ));
    }
    nonzero(op, "rows", ids.rows)?;
    Ok(())
}

/// A u64 splat of `x` (built through i64, as MLIR prints ui64 literals
/// unsigned).
fn const_u64(f: &mut Func, x: u64, dims: &[i64]) -> Built<Val> {
    let v = f.const_i(Elem::I64, x as i64, dims);
    f.bitcast(v, Elem::U64)
}

/// Each row's window, hashed: `ids` `[T]`, `past` `[L, span]` kept cells.
/// Returns the head rows `[T, heads]` and each lane's kept cells after
/// `keep` rows.
fn hash_lanes(
    f: &mut Func,
    h: &Hash<'_>,
    ids: Val,
    past: Val,
    lanes: &Lanes,
    map: Option<Val>,
) -> Built<(Val, Val)> {
    let t = f.dims(ids)[0];
    let l = lanes.count(f);
    let span = h.span();
    let base = l * span;
    let cells = f.reshape(past, &[l * span])?;
    // The cells read as ids, then the fire's ids: what a window reads.
    let z = f.like_i(cells, 0);
    let empty = f.compare(Cmp::Eq, cells, z)?;
    let eos = f.like_i(cells, h.eos);
    let one = f.like_i(cells, 1);
    let less = f.sub(cells, one)?;
    let as_ids = f.select(empty, eos, less)?;
    let read = f.concat(&[as_ids, ids], 0)?;
    // The cells, then the fire's ids as cells: what the next window keeps.
    let ione = f.like_i(ids, 1);
    let id_cells = f.add(ids, ione)?;
    let kept = f.concat(&[cells, id_cells], 0)?;

    let ti = f.iota(Elem::I32, &[t], 0);
    let lane = count_le(f, lanes.begin, ti)?;
    let lone = f.like_i(lane, 1);
    let lane = f.sub(lane, lone)?;
    let lane = clamp_i(f, lane, 0, l - 1)?;
    let begin = take(f, lanes.begin, lane)?;
    let j = f.sub(ti, begin)?;
    let lspan = f.like_i(lane, span);
    let lane_row = f.mul(lane, lspan)?;
    let mut window = vec![ids];
    let mut crossed = f.const_i(Elem::Pred, 0, &[t]);
    let eos_t = f.like_i(ids, h.eos);
    for p in 1..=span {
        let pv = f.like_i(j, p);
        let src = f.sub(j, pv)?;
        let fresh = cmp_i(f, Cmp::Ge, src, 0)?;
        let at_x = f.sub(ti, pv)?;
        let bv = f.like_i(at_x, base);
        let at_x = f.add(at_x, bv)?;
        let sv = f.like_i(src, span);
        let at_p = f.add(src, sv)?;
        let at_p = f.add(at_p, lane_row)?;
        let at = f.select(fresh, at_x, at_p)?;
        let w = take(f, read, at)?;
        // Past the first eos, every older id reads as eos.
        let w = f.select(crossed, eos_t, w)?;
        let is_eos = f.compare(Cmp::Eq, w, eos_t)?;
        crossed = f.or(crossed, is_eos)?;
        window.push(w);
    }
    if let Some(m) = map {
        for w in &mut window {
            let v = take(f, m, *w)?;
            *w = f.convert(v, Elem::I32);
        }
    }

    // mixed_o = XOR_{p < o} u64(w_p) · mult_p, one head group per order.
    let mut acc: Option<Val> = None;
    let mut groups = Vec::with_capacity(h.span() as usize);
    for (p, &w) in window.iter().enumerate() {
        let wide = f.convert(w, Elem::I64);
        let wide = f.bitcast(wide, Elem::U64)?;
        let m = const_u64(f, h.mults[p], &[t])?;
        let term = f.mul(wide, m)?;
        let mixed = match acc {
            None => term,
            Some(a) => f.xor(a, term)?,
        };
        acc = Some(mixed);
        if p == 0 {
            continue;
        }
        let order = p + 1;
        let lo = (order - 2) * h.hpn;
        let hpn = h.hpn as i64;
        let mb = f.broadcast(mixed, &[t, hpn], &[0])?;
        let primes: Vec<i64> = h.primes[lo..lo + h.hpn].iter().map(|&x| x as i64).collect();
        let offsets: Vec<i64> = h.offsets[lo..lo + h.hpn].iter().map(|&x| x as i64).collect();
        let pr = f.const_ints(Elem::I64, &primes, &[hpn])?;
        let pr = f.bitcast(pr, Elem::U64)?;
        let pr = f.broadcast(pr, &[t, hpn], &[1])?;
        let of = f.const_ints(Elem::I64, &offsets, &[hpn])?;
        let of = f.bitcast(of, Elem::U64)?;
        let of = f.broadcast(of, &[t, hpn], &[1])?;
        let r = f.rem(mb, pr)?;
        let r = f.add(r, of)?;
        // The low 32 bits, as the GPU's `(int)` keeps them.
        let low = const_u64(f, 0xFFFF_FFFF, &[t, hpn])?;
        let r = f.and(r, low)?;
        let r = f.convert(r, Elem::U32);
        groups.push(f.bitcast(r, Elem::I32)?);
    }
    let out = f.concat(&groups, 1)?;

    // The kept cells after `keep` rows.
    let s = f.iota(Elem::I32, &[l, span], 1);
    let keep = f.broadcast(lanes.keep, &[l, span], &[0])?;
    let sv = f.const_i(Elem::I32, span, &[l, span]);
    let src = f.sub(keep, sv)?;
    let src = f.add(src, s)?;
    let fresh = cmp_i(f, Cmp::Ge, src, 0)?;
    let b = f.broadcast(lanes.begin, &[l, span], &[0])?;
    let at_x = f.add(b, src)?;
    let bv = f.like_i(at_x, base);
    let at_x = f.add(at_x, bv)?;
    let r = f.iota(Elem::I32, &[l, span], 0);
    let r = f.mul(r, sv)?;
    let at_p = f.add(src, sv)?;
    let at_p = f.add(at_p, r)?;
    let at = f.select(fresh, at_x, at_p)?;
    let next = take(f, kept, at)?;
    Ok((out, next))
}

#[derive(Clone, Copy)]
enum PleLanes {
    Decode,
    Ragged(Tensor),
    Committed(Tensor, Committed),
}

fn ngram(
    ctx: &Ctx<'_>,
    op: &'static str,
    ids: Tensor,
    lanes: PleLanes,
    state: &RecurrentPool,
    h: &Hash<'_>,
    map: Option<Tensor>,
    out: Tensor,
) -> Result<(), Error> {
    planes(op, h, ids, state, map, out)?;
    match lanes {
        PleLanes::Decode => {
            if state.slots.elements() < u64::from(ids.rows) || out.rows != ids.rows {
                return Err(refuse(op, "the slots and the landing ride the fire's rows"));
            }
        }
        PleLanes::Ragged(indptr) => {
            csr_lanes(op, indptr)?;
            if out.rows != ids.rows {
                return Err(refuse(op, "the landing rides the fire's rows"));
            }
        }
        PleLanes::Committed(indptr, _) => csr_lanes(op, indptr)?,
    }
    let t = i64::from(ids.rows);
    ctx.emit(&mut |cx| {
        let idv = flat_i32(cx, ids)?;
        let slab = cx.read(state.state)?;
        let m = match map {
            Some(m) => {
                let v = cx.read(m)?;
                let n = cx.ty(v).elements();
                Some(cx.reshape(v, &[n])?)
            }
            None => None,
        };
        let (lanes, own) = match lanes {
            PleLanes::Decode => {
                let slots = flat_i32(cx, state.slots)?;
                (decode_lanes(cx.func(), slots, t)?, None)
            }
            PleLanes::Ragged(indptr) => {
                let ip = flat_i32(cx, indptr)?;
                let slots = flat_i32(cx, state.slots)?;
                (ragged_lanes(cx.func(), ip, slots)?, None)
            }
            PleLanes::Committed(indptr, cm) => {
                let ip = flat_i32(cx, indptr)?;
                let rep = flat_i32(cx, cm.replay)?;
                let com = flat_i32(cx, cm.commit)?;
                let sl = flat_i32(cx, cm.slots)?;
                let f = cx.func();
                let lanes = committed_lanes(op, f, ip, rep, com, sl, cm.lane0)?;
                let l = lanes.count(f);
                let lane0 = i64::from(cm.lane0);
                let rep = f.slice_axis(rep, 0, lane0, lane0 + l)?;
                (lanes, Some((ip, rep)))
            }
        };
        let f = cx.func();
        let past = f.take_rows(slab, lanes.slot)?;
        let (rows, next) = hash_lanes(f, h, idv, past, &lanes, m)?;
        let slab = land_rows(f, slab, lanes.slot, lanes.write, next)?;
        let rows = match own {
            None => rows,
            Some((ip, rep)) => {
                // Own row o of lane r is extended row
                // ext_begin[r] + replay[r] + (o - indptr[r]).
                let l = lanes.count(f);
                let n = i64::from(out.rows);
                let o = f.iota(Elem::I32, &[n], 0);
                let lo = f.slice_axis(ip, 0, 0, l)?;
                let lane = count_le(f, lo, o)?;
                let one = f.like_i(lane, 1);
                let lane = f.sub(lane, one)?;
                let lane = clamp_i(f, lane, 0, l - 1)?;
                let own0 = take(f, lo, lane)?;
                let ext0 = take(f, lanes.begin, lane)?;
                let skip = take(f, rep, lane)?;
                let at = f.sub(o, own0)?;
                let at = f.add(at, ext0)?;
                let at = f.add(at, skip)?;
                let at = clamp_i(f, at, 0, t - 1)?;
                f.take_rows(rows, at)?
            }
        };
        cx.write(out, rows)?;
        cx.write(state.state, slab)
    })
}

/// One row per lane: hash `[id, window..]`, then push `id` into the window.
pub fn ngram_ids(
    ctx: &Ctx<'_>,
    ids: Tensor,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Tensor>,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids";
    let h = hash(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    ngram(ctx, OP, ids, PleLanes::Decode, state, &h, map, ngram_ids)
}

/// Every row of a query CSR's lanes; `state.slots` is per row.
pub fn ngram_ids_chunked(
    ctx: &Ctx<'_>,
    ids: RaggedTensor,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Tensor>,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids_chunked";
    let h = hash(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    ngram(
        ctx,
        OP,
        ids.data,
        PleLanes::Ragged(ids.indptr),
        state,
        &h,
        map,
        ngram_ids,
    )
}

/// A rollback seat's extended ids: the own rows (past each lane's replay)
/// land in `ngram_ids` at their own positions, the window after each lane's
/// committed prefix lands in the state.
pub fn ngram_ids_committed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    indptr: Tensor,
    committed: &Committed,
    state: &RecurrentPool,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Tensor>,
    ngram_ids: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ple_ngram_ids_committed";
    let h = hash(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    ngram(
        ctx,
        OP,
        ids,
        PleLanes::Committed(indptr, *committed),
        state,
        &h,
        map,
        ngram_ids,
    )
}

// ------------------------------------------------------------ selector walk

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
    const OP: &str = "attention.selector_walk";
    if first > 1 {
        return Err(refuse(OP, "a span's anchor is row 0 and its first mask row 1"));
    }
    if cand.data.dtype != Dtype::I32 || tokens.dtype != Dtype::I32 || picks.dtype != Dtype::I32 {
        return Err(refuse(OP, "the candidates, the tokens and the picks are i32"));
    }
    if unary.dtype != Dtype::F32 {
        return Err(refuse(OP, "the unary logits are f32"));
    }
    csr_lanes(OP, cand.indptr)?;
    let k = nonzero(OP, "candidates a slot", cand.data.width)?;
    let rank = nonzero(OP, "the codebooks' rank", pred.width)?;
    if succ.width != rank || hp.is_some_and(|h| h.width != rank) || pred.rows != succ.rows {
        return Err(refuse(OP, "the codebooks and the projected hidden disagree on rank or vocabulary"));
    }
    let vocab = nonzero(OP, "the codebooks' vocabulary", pred.rows)?;
    let rows = cand.data.rows;
    if unary.rows != rows
        || unary.width != k
        || picks.rows != rows
        || tokens.rows != rows
        || hp.is_some_and(|h| h.rows != rows)
    {
        return Err(refuse(OP, "unary, hp, tokens and picks carry one row per candidate row"));
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
            let drop = f.like_i(lo, super::ssm::DROP);
            let at = f.select(has, lo, drop)?;
            let p2 = f.reshape(picks0, &[t, 1])?;
            let a2 = f.reshape(anchor, &[l, 1])?;
            let p2 = f.put_rows(p2, at, a2, crate::hlo::Combine::Set)?;
            picks0 = f.reshape(p2, &[t])?;
        }
        let prev0 = take(f, tk, at0)?;
        let most = f.reduce(span, &[0], Fold::Max)?;
        let fv = super::ssm::ci(f, first);
        let count = f.sub(most, fv)?;
        let hv = match hv {
            Some(h) => h,
            None => f.const_f(Elem::F32, 1.0, &[t, rk]),
        };
        let out = super::ssm::loop_upto(
            f,
            count,
            &[picks0, prev0],
            &[cd, un, hv, pr, sc, lo, span],
            |f, i, cr, inv| {
                let (picks, prev) = (cr[0], cr[1]);
                let (cd, un, hv, pr, sc, lo, span) =
                    (inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6]);
                let fv = super::ssm::ci(f, first);
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
                let drop = f.like_i(row, super::ssm::DROP);
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
