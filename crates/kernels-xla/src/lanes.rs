//! Lanes and their recurrent rows, in HLO: the lane tables of a decode, a
//! ragged (CSR) and a rollback (committed) fire, the slab-row gathers and
//! scatters a recurrent state moves through, and the i32 helpers they are
//! written in. Shared by `ssm`, `attn` (index, pool), `gather` (ngram) and
//! `sampling` (selector walk).

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Combine, Elem, Fold, Func, GatherDims, ScatterDims, Val};
use crate::tensor::{Operand, Stored};

/// The slot a write that must not land is routed to.
pub(crate) const DROP: i64 = i32::MAX as i64;

/// A rollback seat's per-lane tables: `replay` rows buffered ahead of each
/// lane's own rows, the `commit` prefix whose state lands, the lane's `slots`
/// (negative: no state), all indexed from `lane0`.
#[derive(Clone, Copy, Debug)]
pub struct Committed<T> {
    pub replay: T,

    pub commit: T,

    pub slots: T,

    pub lane0: u32,
}

// ------------------------------------------------------------------ helpers

pub(crate) fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{what}` is zero")));
    }
    Ok(v)
}

/// A handle's elements as a flat i32 vector.
pub(crate) fn flat_i32(f: &mut Func, t: Operand) -> Result<Val, Error> {
    let v = t.val;
    let v = f.convert(v, Elem::I32);
    let n = f.ty(v).elements();
    Ok(f.reshape(v, &[n])?)
}

/// A handle's elements as a flat f32 vector.
pub(crate) fn flat_f32(f: &mut Func, t: Operand) -> Result<Val, Error> {
    let v = t.f32(f);
    let n = f.ty(v).elements();
    Ok(f.reshape(v, &[n])?)
}

pub(crate) fn ci(f: &mut Func, x: i64) -> Val {
    f.const_i(Elem::I32, x, &[])
}

pub(crate) fn iota(f: &mut Func, n: i64) -> Val {
    f.iota(Elem::I32, &[n], 0)
}

pub(crate) fn full_i(f: &mut Func, x: i64, n: i64) -> Val {
    f.const_i(Elem::I32, x, &[n])
}

/// `v[idx]` for a rank-1 `v` and i32 `idx` of any rank; out-of-range
/// indices clamp.
pub(crate) fn take(f: &mut Func, v: Val, idx: Val) -> Built<Val> {
    let n = f.dims(v)[0];
    let shape = f.dims(idx).to_vec();
    let m: i64 = shape.iter().product();
    let t = f.reshape(v, &[n, 1])?;
    let i = f.reshape(idx, &[m])?;
    let r = f.take_rows(t, i)?;
    f.reshape(r, &shape)
}

/// `#{s : starts[s] <= probe[n]}` for each `n`.
pub(crate) fn count_le(f: &mut Func, starts: Val, probe: Val) -> Built<Val> {
    let s = f.dims(starts)[0];
    let n = f.dims(probe)[0];
    let a = f.broadcast(starts, &[n, s], &[1])?;
    let b = f.broadcast(probe, &[n, s], &[0])?;
    let le = f.compare(Cmp::Le, a, b)?;
    let c = f.convert(le, Elem::I32);
    f.reduce(c, &[1], Fold::Sum)
}

pub(crate) fn excl_cumsum(f: &mut Func, v: Val) -> Built<Val> {
    let inc = f.scan(v, 0, Fold::Sum)?;
    f.sub(inc, v)
}

/// `x[i]` along axis 0, the axis dropped.
pub(crate) fn row_at(f: &mut Func, x: Val, i: Val) -> Built<Val> {
    let d = f.dims(x).to_vec();
    let zero = ci(f, 0);
    let mut starts = vec![i];
    starts.extend(std::iter::repeat_n(zero, d.len() - 1));
    let mut sizes = d.clone();
    sizes[0] = 1;
    let s = f.dynamic_slice(x, &starts, &sizes)?;
    f.reshape(s, &d[1..])
}

/// A rank-1 vector as a one-column matrix.
pub(crate) fn f_reshape_col(f: &mut Func, v: Val) -> Built<Val> {
    let n = f.dims(v)[0];
    f.reshape(v, &[n, 1])
}

/// `x[i] = upd` along axis 0.
pub(crate) fn put_at(f: &mut Func, x: Val, upd: Val, i: Val) -> Built<Val> {
    let d = f.dims(x).to_vec();
    let zero = ci(f, 0);
    let mut starts = vec![i];
    starts.extend(std::iter::repeat_n(zero, d.len() - 1));
    let mut sizes = d.clone();
    sizes[0] = 1;
    let u = f.reshape(upd, &sizes)?;
    f.dynamic_update_slice(x, u, &starts)
}

pub(crate) fn clamp_i(f: &mut Func, v: Val, lo: i64, hi: i64) -> Built<Val> {
    let lo = ci(f, lo);
    let hi = ci(f, hi);
    f.clamp(lo, v, hi)
}

pub(crate) fn cmp_i(f: &mut Func, dir: Cmp, v: Val, x: i64) -> Built<Val> {
    let c = f.like_i(v, x);
    f.compare(dir, v, c)
}

/// `[a0, b0, a1, b1, ..]` of two equal rank-1 vectors.
pub(crate) fn interleave(f: &mut Func, a: Val, b: Val) -> Built<Val> {
    let n = f.dims(a)[0];
    let a = f.reshape(a, &[n, 1])?;
    let b = f.reshape(b, &[n, 1])?;
    let ab = f.concat(&[a, b], 1)?;
    f.reshape(ab, &[2 * n])
}

/// `for i in 0..count` with a traced `count`: `body` gets the counter, the
/// carried values and the loop invariants (passed through the loop rather
/// than captured), and yields the next carried values.
pub(crate) fn loop_upto(
    f: &mut Func,
    count: Val,
    carried: &[Val],
    invariant: &[Val],
    body: impl FnOnce(&mut Func, Val, &[Val], &[Val]) -> Built<Vec<Val>>,
) -> Built<Vec<Val>> {
    let zero = ci(f, 0);
    let (nc, ni) = (carried.len(), invariant.len());
    let mut inits = vec![zero, count];
    inits.extend_from_slice(carried);
    inits.extend_from_slice(invariant);
    let out = f.while_loop(
        &inits,
        |f, a| f.compare(Cmp::Lt, a[0], a[1]),
        |f, a| {
            let one = ci(f, 1);
            let next = f.add(a[0], one)?;
            let inv = a[2 + nc..2 + nc + ni].to_vec();
            let rest = body(f, a[0], &a[2..2 + nc], &inv)?;
            let mut v = vec![next, a[1]];
            v.extend(rest);
            v.extend(inv);
            Ok(v)
        },
    )?;
    Ok(out[2..2 + nc].to_vec())
}

/// Scatters `rows` (`[lanes, stride]`) over the slab at `slot`, dropping the
/// lanes `write` is false for.
pub(crate) fn land_rows(f: &mut Func, slab: Val, slot: Val, write: Val, rows: Val) -> Built<Val> {
    let drop = f.like_i(slot, DROP);
    let at = f.select(write, slot, drop)?;
    let elem = f.elem(slab);
    let rows = f.convert(rows, elem);
    f.put_rows(slab, at, rows, crate::hlo::Combine::Set)
}

/// How many rows of its bank one slot of `stride` elements holds: 1 for a
/// `[slots, stride]` bank, `stride / width` for a bank whose slots are
/// blocks of `[stride / width, width]` rows.
///
/// On a TPU the second is what makes a slot contiguous: a 2-D array is tiled
/// `(8, 128)`, so a row of a `[slots, stride]` bank is strided across tiles
/// shared by eight slots, and gathering or scattering it is several times
/// slower than moving a block of whole tiles.
pub(crate) fn slot_rows(
    op: &'static str,
    what: &str,
    bank: impl Stored,
    stride: u64,
) -> Result<i64, Error> {
    let width = u64::from(bank.width());
    if width == stride {
        return Ok(1);
    }
    if width != 0 && stride.is_multiple_of(width) {
        let per = stride / width;
        if u64::from(bank.rows()).is_multiple_of(per) {
            return Ok(per as i64);
        }
    }
    Err(refuse(
        op,
        format!(
            "the {what} bank is {}x{}, and a slot keeps {stride}: neither one row per slot nor \
             whole blocks of rows",
            bank.rows(),
            bank.width()
        ),
    ))
}

/// The most lanes `take_slots` slices out of a one-row-per-slot bank row by
/// row. A TPU gathers rows as wide as a GDN state (a megabyte) by cutting the
/// whole bank into column blocks first: a copy of every slot, per layer, per
/// fire.
const SLICED_LANES: i64 = 8;

/// Each slot's state as one row, `[L, per · width]`, from a bank of `per`
/// rows per slot; out-of-range slots clamp (they read garbage, harmlessly).
pub(crate) fn take_slots(f: &mut Func, slab: Val, slot: Val, per: i64) -> Built<Val> {
    if per == 1 {
        let l = f.dims(slot)[0];
        if l > SLICED_LANES {
            return f.take_rows(slab, slot);
        }
        let width = f.dims(slab)[1];
        let zero = ci(f, 0);
        let mut rows = Vec::with_capacity(l as usize);
        for i in 0..l {
            let at = f.slice_axis(slot, 0, i, i + 1)?;
            let at = f.reshape(at, &[])?;
            rows.push(f.dynamic_slice(slab, &[at, zero], &[1, width])?);
        }
        return if l == 1 {
            Ok(rows[0])
        } else {
            f.concat(&rows, 0)
        };
    }
    let (rows, width) = (f.dims(slab)[0], f.dims(slab)[1]);
    let l = f.dims(slot)[0];
    let blocks = f.reshape(slab, &[rows / per, per, width])?;
    let at = f.reshape(slot, &[l, 1])?;
    let got = f.gather(
        blocks,
        at,
        &GatherDims {
            offset_dims: vec![1, 2],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            ..GatherDims::default()
        },
        &[1, per, width],
    )?;
    f.reshape(got, &[l, per * width])
}

/// Scatters `rows` (`[L, per · width]`) over the slots of a bank of `per`
/// rows per slot, dropping the lanes `write` is false for (and any slot out
/// of range).
pub(crate) fn land_slots(
    f: &mut Func,
    slab: Val,
    slot: Val,
    write: Val,
    rows: Val,
    per: i64,
) -> Built<Val> {
    if per == 1 {
        return land_rows(f, slab, slot, write, rows);
    }
    let drop = f.like_i(slot, DROP);
    let at = f.select(write, slot, drop)?;
    let elem = f.elem(slab);
    let rows = f.convert(rows, elem);
    let (n, width) = (f.dims(slab)[0], f.dims(slab)[1]);
    let l = f.dims(slot)[0];
    let blocks = f.reshape(slab, &[n / per, per, width])?;
    let at = f.reshape(at, &[l, 1])?;
    let rows = f.reshape(rows, &[l, per, width])?;
    let out = f.scatter(
        blocks,
        at,
        rows,
        &ScatterDims {
            update_window_dims: vec![1, 2],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
            ..ScatterDims::default()
        },
        Combine::Set,
    )?;
    f.reshape(out, &[n, width])
}

// -------------------------------------------------------------------- lanes

/// Where each lane of a fire sits: rows `[begin, begin + span)`, the state
/// after its first `keep` rows lands in `slot` when `write`.
pub(crate) struct Lanes {
    pub(crate) begin: Val,
    pub(crate) span: Val,
    pub(crate) keep: Val,
    pub(crate) slot: Val,
    pub(crate) write: Val,
}

impl Lanes {
    pub(crate) fn count(&self, f: &Func) -> i64 {
        f.dims(self.begin)[0]
    }
}

/// One row per lane, the slot of each row in `slots`.
/// How a fire's rows fall into lanes, with the tables that say so: one row
/// a lane (`slots` its state row), a query CSR (`slots` per row), or a
/// rollback seat's extended lanes.
#[derive(Clone, Copy, Debug)]
pub enum LaneTables {
    Decode {
        slots: Operand,
    },
    Ragged {
        indptr: Operand,
        slots: Operand,
    },
    Committed {
        indptr: Operand,
        seat: Committed<Operand>,
    },
}

impl LaneTables {
    /// Refuses tables that cannot cover `rows` rows.
    pub(crate) fn check(&self, op: &'static str, rows: u32) -> Result<(), Error> {
        match self {
            Self::Decode { slots } => {
                if slots.elements() < u64::from(rows) {
                    return Err(refuse(
                        op,
                        "the pool names fewer slots than the fire has rows",
                    ));
                }
                Ok(())
            }
            Self::Ragged { indptr, .. } | Self::Committed { indptr, .. } => csr_lanes(op, *indptr),
        }
    }

    /// The lanes of `rows` rows, the tables read in order (indptr first).
    pub(crate) fn lanes(&self, f: &mut Func, op: &'static str, rows: i64) -> Result<Lanes, Error> {
        match *self {
            Self::Decode { slots } => {
                let slots = flat_i32(f, slots)?;
                Ok(decode_lanes(f, slots, rows)?)
            }
            Self::Ragged { indptr, slots } => {
                let ip = flat_i32(f, indptr)?;
                let slots = flat_i32(f, slots)?;
                Ok(ragged_lanes(f, ip, slots)?)
            }
            Self::Committed { indptr, seat } => {
                let ip = flat_i32(f, indptr)?;
                let rep = flat_i32(f, seat.replay)?;
                let com = flat_i32(f, seat.commit)?;
                let sl = flat_i32(f, seat.slots)?;
                committed_lanes(op, f, ip, rep, com, sl, seat.lane0)
            }
        }
    }
}

pub(crate) fn decode_lanes(f: &mut Func, slots: Val, rows: i64) -> Built<Lanes> {
    let slot = f.slice_axis(slots, 0, 0, rows)?;
    let t = f.const_i(Elem::Pred, 1, &[rows]);
    Ok(Lanes {
        begin: iota(f, rows),
        span: full_i(f, 1, rows),
        keep: full_i(f, 1, rows),
        slot,
        write: t,
    })
}

/// A query CSR's lanes; `slots` is per row, a lane's slot the slot of its
/// first row (as the GPU kernels read it).
pub(crate) fn ragged_lanes(f: &mut Func, indptr: Val, slots: Val) -> Built<Lanes> {
    let l = f.dims(indptr)[0] - 1;
    let lo = f.slice_axis(indptr, 0, 0, l)?;
    let hi = f.slice_axis(indptr, 0, 1, l + 1)?;
    let span = f.sub(hi, lo)?;
    let zero = f.like_i(span, 0);
    let span = f.max(span, zero)?;
    let n = f.dims(slots)[0];
    let at = clamp_i(f, lo, 0, n - 1)?;
    let slot = take(f, slots, at)?;
    let write = cmp_i(f, Cmp::Gt, span, 0)?;
    Ok(Lanes {
        begin: lo,
        span,
        keep: span,
        slot,
        write,
    })
}

/// A rollback seat's extended lanes: lane `r` holds its `replay` rows ahead
/// of its own, and the state after its first `commit` rows lands.
pub(crate) fn committed_lanes(
    op: &'static str,
    f: &mut Func,
    indptr: Val,
    replay: Val,
    commit: Val,
    slots: Val,
    lane0: u32,
) -> Result<Lanes, Error> {
    let l = f.dims(indptr)[0] - 1;
    let lane0 = i64::from(lane0);
    for (what, v) in [("replay", replay), ("commit", commit), ("slot", slots)] {
        if f.dims(v)[0] < lane0 + l {
            return Err(refuse(
                op,
                format!(
                    "the seat's {what} table holds {} lanes and this window reads {l} from lane {lane0}",
                    f.dims(v)[0]
                ),
            ));
        }
    }
    let rep = f.slice_axis(replay, 0, lane0, lane0 + l)?;
    let com = f.slice_axis(commit, 0, lane0, lane0 + l)?;
    let slot = f.slice_axis(slots, 0, lane0, lane0 + l)?;
    let lo = f.slice_axis(indptr, 0, 0, l)?;
    let hi = f.slice_axis(indptr, 0, 1, l + 1)?;
    let shift = excl_cumsum(f, rep)?;
    let begin = f.add(lo, shift)?;
    let own = f.sub(hi, lo)?;
    let span = f.add(own, rep)?;
    let zero = f.like_i(span, 0);
    let span = f.max(span, zero)?;
    let keep = f.min(com, span)?;
    let keep = f.max(keep, zero)?;
    let a = cmp_i(f, Cmp::Gt, span, 0)?;
    let b = cmp_i(f, Cmp::Ge, slot, 0)?;
    let c = cmp_i(f, Cmp::Gt, keep, 0)?;
    let write = f.and(a, b)?;
    let write = f.and(write, c)?;
    Ok(Lanes {
        begin,
        span,
        keep,
        slot,
        write,
    })
}

pub(crate) fn csr_lanes(op: &'static str, indptr: impl Stored) -> Result<(), Error> {
    if indptr.dtype() != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the query CSR's boundaries are {:?}, and this scan walks an i32 indptr",
                indptr.dtype()
            ),
        ));
    }
    if indptr.elements() < 2 {
        return Err(refuse(op, "the query CSR this fire names spans no request"));
    }
    Ok(())
}
