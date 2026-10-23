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

use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Elem, Func, Val};
use crate::lanes::{
    Committed, LaneTables, Lanes, clamp_i, cmp_i, committed_lanes, count_le, csr_lanes,
    decode_lanes, flat_i32, land_rows, nonzero, ragged_lanes, take,
};
use crate::tensor::{Landing, Operand, RaggedTensor, Stored};

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
            format!(
                "{} multipliers do not fit the 2..={MAX_NGRAM}-gram range",
                mults.len()
            ),
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
    nonzero(
        op,
        "the heads per n-gram this statement states",
        heads_per_ngram,
    )?;
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
    ids: impl Stored,
    state: impl Stored,
    map: Option<impl Stored>,
    out: impl Stored,
) -> Result<(), Error> {
    if ids.dtype() != Dtype::I32 || out.dtype() != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the hasher reads i32 token ids and lands i32 table rows, not {:?} into {:?}",
                ids.dtype(),
                out.dtype()
            ),
        ));
    }
    if i64::from(out.width()) != h.heads() {
        return Err(refuse(op, "the landing is not one column per hashed head"));
    }
    if state.dtype() != Dtype::I32 || i64::from(state.width()) != h.span() {
        return Err(refuse(
            op,
            "the window a lane keeps is the n-gram context, one i32 per trailing id",
        ));
    }
    if let Some(m) = map
        && m.dtype() != Dtype::I64
    {
        return Err(refuse(
            op,
            format!(
                "the id map is {:?}, and the hasher maps through i64",
                m.dtype()
            ),
        ));
    }
    nonzero(op, "rows", ids.rows())?;
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
        let offsets: Vec<i64> = h.offsets[lo..lo + h.hpn]
            .iter()
            .map(|&x| x as i64)
            .collect();
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

/// `(out, state)`: the hashed rows, and the slab with each lane's window
/// pushed.
fn ngram(
    f: &mut Func,
    op: &'static str,
    ids: Operand,
    lanes: LaneTables,
    state: Operand,
    h: &Hash<'_>,
    map: Option<Operand>,
    out: Landing,
) -> Result<(Val, Val), Error> {
    planes(op, h, ids, state, map, out)?;
    match lanes {
        LaneTables::Decode { slots } => {
            if slots.elements() < u64::from(ids.rows) || out.rows != ids.rows {
                return Err(refuse(op, "the slots and the landing ride the fire's rows"));
            }
        }
        LaneTables::Ragged { indptr, .. } => {
            csr_lanes(op, indptr)?;
            if out.rows != ids.rows {
                return Err(refuse(op, "the landing rides the fire's rows"));
            }
        }
        LaneTables::Committed { indptr, .. } => csr_lanes(op, indptr)?,
    }
    let t = i64::from(ids.rows);
    let idv = flat_i32(f, ids)?;
    let slab = state.val;
    let m = match map {
        Some(m) => {
            let v = m.val;
            let n = f.ty(v).elements();
            Some(f.reshape(v, &[n])?)
        }
        None => None,
    };
    let (lanes, own) = match lanes {
        LaneTables::Decode { slots } => {
            let slots = flat_i32(f, slots)?;
            (decode_lanes(f, slots, t)?, None)
        }
        LaneTables::Ragged { indptr, slots } => {
            let ip = flat_i32(f, indptr)?;
            let slots = flat_i32(f, slots)?;
            (ragged_lanes(f, ip, slots)?, None)
        }
        LaneTables::Committed { indptr, seat } => {
            let ip = flat_i32(f, indptr)?;
            let rep = flat_i32(f, seat.replay)?;
            let com = flat_i32(f, seat.commit)?;
            let sl = flat_i32(f, seat.slots)?;
            let lanes = committed_lanes(op, f, ip, rep, com, sl, seat.lane0)?;
            let l = lanes.count(f);
            let lane0 = i64::from(seat.lane0);
            let rep = f.slice_axis(rep, 0, lane0, lane0 + l)?;
            (lanes, Some((ip, rep)))
        }
    };
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
    Ok((rows, slab))
}

/// One row per lane: hash `[id, window..]`, then push `id` into the window.
/// Returns `(ngram_ids, state)`: the hashed rows and the new slab.
pub fn ngram_ids(
    f: &mut Func,
    ids: Operand,
    state: Operand,
    slots: Operand,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Operand>,
    ngram_ids: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "gather.ngram_ids";
    let h = hash(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    ngram(
        f,
        OP,
        ids,
        LaneTables::Decode { slots },
        state,
        &h,
        map,
        ngram_ids,
    )
}

/// Every row of a query CSR's lanes; `slots` is per row.
pub fn ngram_ids_chunked(
    f: &mut Func,
    ids: RaggedTensor<Operand>,
    state: Operand,
    slots: Operand,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Operand>,
    ngram_ids: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "gather.ngram_ids_chunked";
    let h = hash(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    let lanes = LaneTables::Ragged {
        indptr: ids.indptr,
        slots,
    };
    ngram(f, OP, ids.data, lanes, state, &h, map, ngram_ids)
}

/// A rollback seat's extended ids: the own rows (past each lane's replay)
/// land in `ngram_ids` at their own positions, the window after each lane's
/// committed prefix lands in the state.
pub fn ngram_ids_committed(
    f: &mut Func,
    ids: Operand,
    indptr: Operand,
    seat: Committed<Operand>,
    state: Operand,
    eos: u32,
    mults: &[u64],
    primes: &[u64],
    offsets: &[u64],
    heads_per_ngram: u32,
    map: Option<Operand>,
    ngram_ids: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "gather.ngram_ids_committed";
    let h = hash(OP, eos, mults, primes, offsets, heads_per_ngram)?;
    let lanes = LaneTables::Committed { indptr, seat };
    ngram(f, OP, ids, lanes, state, &h, map, ngram_ids)
}

// ------------------------------------------------------------ selector walk
