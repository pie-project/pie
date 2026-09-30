#![allow(clippy::too_many_arguments)]
//! Recurrent state-space scans: the causal depthwise convs (Qwen3.5's
//! `causal_conv1d`, LFM2's short conv), the gated delta rule (Qwen3.5 GDN)
//! and Kimi's KDA, each in its decode (one row per lane), chunked (a query
//! CSR of lanes) and committed (a rollback seat's extended window) form.
//!
//! State lives in a [`RecurrentPool`]: one slab row per slot. A fire gathers
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

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Combine, Elem, Fold, Func, GatherDims, ScatterDims, Val};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

/// The slot a write that must not land is routed to.
pub(crate) const DROP: i64 = i32::MAX as i64;

/// Tokens per chunk of the scalar-decay (GDN) chunked scan: `LONG_CHUNK`
/// when the fire's lanes average at least that many rows, else `CHUNK`
/// (every lane pads its last chunk).
const CHUNK: i64 = 64;

const LONG_CHUNK: i64 = 128;

/// Tokens per chunk of the per-channel-decay (KDA) chunked scan: its
/// intra-chunk decay is a `[C, C, d]` product, so it chunks finer.
const KDA_CHUNK: i64 = 16;

/// The GDN's q/k l2-norm epsilon, as the GPU kernels state it.
const GDN_EPS: f64 = 1e-6;

/// A rollback seat's per-lane tables: `replay` rows buffered ahead of each
/// lane's own rows, the `commit` prefix whose state lands, the lane's `slots`
/// (negative: no state), all indexed from `lane0`.
#[derive(Clone, Copy, Debug)]
pub struct Committed {
    pub replay: Tensor,

    pub commit: Tensor,

    pub slots: Tensor,

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
pub(crate) fn flat_i32(cx: &mut Cx<'_>, t: Tensor) -> Result<Val, Error> {
    let v = cx.read(t)?;
    let v = cx.convert(v, Elem::I32);
    let n = cx.ty(v).elements();
    Ok(cx.reshape(v, &[n])?)
}

/// A handle's elements as a flat f32 vector.
pub(crate) fn flat_f32(cx: &mut Cx<'_>, t: Tensor) -> Result<Val, Error> {
    let v = cx.read_f32(t)?;
    let n = cx.ty(v).elements();
    Ok(cx.reshape(v, &[n])?)
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
fn f_reshape_col(f: &mut Func, v: Val) -> Built<Val> {
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
fn interleave(f: &mut Func, a: Val, b: Val) -> Built<Val> {
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
    bank: Tensor,
    stride: u64,
) -> Result<i64, Error> {
    let width = u64::from(bank.width);
    if width == stride {
        return Ok(1);
    }
    if width != 0 && stride.is_multiple_of(width) {
        let per = stride / width;
        if u64::from(bank.rows).is_multiple_of(per) {
            return Ok(per as i64);
        }
    }
    Err(refuse(
        op,
        format!(
            "the {what} bank is {}x{}, and a slot keeps {stride}: neither one row per slot nor \
             whole blocks of rows",
            bank.rows, bank.width
        ),
    ))
}

/// Each slot's state as one row, `[L, per · width]`, from a bank of `per`
/// rows per slot; out-of-range slots clamp (they read garbage, harmlessly).
pub(crate) fn take_slots(f: &mut Func, slab: Val, slot: Val, per: i64) -> Built<Val> {
    if per == 1 {
        return f.take_rows(slab, slot);
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

/// Lanes a decode-form delta rule moves per gather → step → scatter round:
/// small enough that a round's gathered and updated states stay on chip
/// (a whole 64-lane fire of 1 MiB states does not, and every intermediate
/// then goes through HBM).
const DECODE_ROUND_BYTES: i64 = 32 << 20;

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

pub(crate) fn csr_lanes(op: &'static str, indptr: Tensor) -> Result<(), Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the query CSR's boundaries are {:?}, and this scan walks an i32 indptr",
                indptr.dtype
            ),
        ));
    }
    if indptr.elements() < 2 {
        return Err(refuse(op, "the query CSR this fire names spans no request"));
    }
    Ok(())
}

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

#[derive(Clone, Copy)]
enum ConvLanes {
    Decode,
    Ragged(Tensor),
    Committed(Tensor, Committed),
}

fn conv(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    lanes: ConvLanes,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    out: ConvOut,
) -> Result<(), Error> {
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
    if state.conv_state.buf != state.new_conv_state.buf {
        return Err(refuse(
            op,
            "this plane rolls the conv state in place, and the pool names two banks",
        ));
    }
    let hist = u64::from(taps - 1) * u64::from(dil) + 1;
    let per = slot_rows(
        op,
        &format!(
            "conv (a width of {taps} at dilation {dil} keeps {hist} rows of {channels} channels)"
        ),
        state.conv_state,
        hist * u64::from(channels),
    )?;
    match lanes {
        ConvLanes::Decode => {
            if state.slots.elements() < u64::from(x.rows) {
                return Err(refuse(
                    op,
                    "the pool names fewer slots than the fire has rows",
                ));
            }
        }
        ConvLanes::Ragged(indptr) | ConvLanes::Committed(indptr, _) => csr_lanes(op, indptr)?,
    }
    let (t, c, k) = (i64::from(x.rows), i64::from(channels), i64::from(taps));
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let w = cx.read_f32(weight)?;
        let w = cx.reshape(w, &[c, k])?;
        let slab = cx.read(state.conv_state)?;
        let lanes = match lanes {
            ConvLanes::Decode => {
                let slots = flat_i32(cx, state.slots)?;
                decode_lanes(cx.func(), slots, t)?
            }
            ConvLanes::Ragged(indptr) => {
                let ip = flat_i32(cx, indptr)?;
                let slots = flat_i32(cx, state.slots)?;
                ragged_lanes(cx.func(), ip, slots)?
            }
            ConvLanes::Committed(indptr, cm) => {
                let ip = flat_i32(cx, indptr)?;
                let rep = flat_i32(cx, cm.replay)?;
                let com = flat_i32(cx, cm.commit)?;
                let sl = flat_i32(cx, cm.slots)?;
                committed_lanes(op, cx.func(), ip, rep, com, sl, cm.lane0)?
            }
        };
        let f = cx.func();
        let past = take_slots(f, slab, lanes.slot, per)?;
        let past = f.convert(past, Elem::F32);
        let (yv, next) = conv_lanes(f, xv, w, past, &lanes, i64::from(dil), out)?;
        let slab = land_slots(f, slab, lanes.slot, lanes.write, next, per)?;
        cx.write(y, yv)?;
        cx.write(state.new_conv_state, slab)
    })
}

/// One row per lane: `y = silu(conv(state ++ x))`, the window shifts by one.
pub fn causal_conv1d(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    conv(
        ctx,
        "attention.ssm_causal_conv1d",
        x,
        ConvLanes::Decode,
        weight,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

/// The conv over a query CSR; `state.slots` is per row (a lane's slot is its
/// first row's).
pub fn causal_conv1d_chunked(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    conv(
        ctx,
        "attention.ssm_causal_conv1d_chunked",
        x.data,
        ConvLanes::Ragged(x.indptr),
        weight,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

/// The conv over a rollback seat's extended rows: every row is convolved,
/// the window after each lane's committed prefix lands.
pub fn causal_conv1d_committed(
    ctx: &Ctx<'_>,
    x: Tensor,
    indptr: Tensor,
    committed: &Committed,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    conv(
        ctx,
        "attention.ssm_causal_conv1d_committed",
        x,
        ConvLanes::Committed(indptr, *committed),
        weight,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

/// `y = conv(state ++ x) + x`, one row per lane.
/// Reference: kernels-cuda `attn::ssm::short_conv` (`ConvOut::Residual`).
pub fn short_conv(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: Tensor,
) -> Result<(), Error> {
    conv(
        ctx,
        "attention.short_conv",
        x,
        ConvLanes::Decode,
        weight,
        state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

/// The short conv over a query CSR.
/// Reference: kernels-cuda `attn::ssm::short_conv_chunked`.
pub fn short_conv_chunked(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: Tensor,
) -> Result<(), Error> {
    conv(
        ctx,
        "attention.short_conv_chunked",
        x.data,
        ConvLanes::Ragged(x.indptr),
        weight,
        state,
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
    ctx: &Ctx<'_>,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gdn_prep";
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
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(ba)?;
        let db = flat_f32(cx, dt_bias)?;
        let al = flat_f32(cx, a_log)?;
        let f = cx.func();
        let b = f.slice_axis(v, 1, 0, vh)?;
        let a = f.slice_axis(v, 1, vh, 2 * vh)?;
        let db = f.broadcast(db, &[rows, vh], &[1])?;
        let z = f.add(a, db)?;
        let sp = softplus20(f, z)?;
        let al = f.exp(al);
        let al = f.broadcast(al, &[rows, vh], &[1])?;
        let g = f.mul(al, sp)?;
        let g = f.neg(g);
        let beta = f.sigmoid(b);
        let out = f.concat(&[g, beta], 1)?;
        cx.write(gates, out)
    })
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

/// One token per lane: `S ← S·diag(a)`, `δ = β(v − S k)`, `S ← S + δ kᵀ`,
/// `y = S q`, over `S` `[R, H, dv, dk]`.
fn delta_step(f: &mut Func, h: &Heads, s: Val) -> Built<(Val, Val)> {
    let d = f.dims(s).to_vec();
    let (r, hh, dv, dk) = (d[0], d[1], d[2], d[3]);
    let a = f.exp(h.g);
    let a = over_keys(f, a, dk)?;
    let a = f.broadcast(a, &d, &[0, 1, 3])?;
    let s1 = f.mul(s, a)?;
    let kb = f.broadcast(h.k, &d, &[0, 1, 3])?;
    let mem = f.mul(s1, kb)?;
    let mem = f.reduce(mem, &[3], Fold::Sum)?;
    let delta = f.sub(h.v, mem)?;
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
    let gcum = f.scan(g, 2, Fold::Sum)?;
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
        let kk = f.dot_general(k, k, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
        let qk = f.dot_general(q, k, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
        (f.mul(kk, ex)?, f.mul(qk, ex)?)
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
    let bt = f.broadcast(beta, &sq, &[0, 1, 2])?;
    let a = f.mul(bt, kk)?;
    let a = f.select(strict4, a, zero4)?;
    let p = f.select(incl4, qk, zero4)?;
    let eye = f.convert(diag, Elem::F32);
    let eye = f.broadcast(eye, &sq, &[2, 3])?;
    let lower = f.add(eye, a)?;
    let bv = f.broadcast(beta, &[nc, hh, c, dv], &[0, 1, 2])?;
    let bk = f.broadcast(beta, &[nc, hh, c, dk], &[0, 1, 2])?;
    let rv = f.mul(v, bv)?;
    let rk = f.mul(kg, bk)?;
    let rhs = f.concat(&[rv, rk], 3)?;
    let _ = lower;
    let inv = unit_lower_inverse(f, a)?;
    let x = f.dot_general(inv, rhs, &[0, 1], &[0, 1], &[3], &[2], Elem::F32)?;
    let ut = f.slice_axis(x, 3, 0, dv)?;
    let w = f.slice_axis(x, 3, dv, dv + dk)?;

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
        &[first_i, lane_n, w, ut, kt, gc, init],
        |f, i, cr, inv| {
            let (st, ends) = (cr[0], cr[1]);
            let (first_i, lane_n, w, ut, kt, gc, init) =
                (inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6]);
            let first = row_at(f, first_i, i)?;
            let first = cmp_i(f, Cmp::Ne, first, 0)?;
            let lane = row_at(f, lane_n, i)?;
            let from = row_at(f, init, lane)?;
            let s0 = f.select(first, from, st)?;
            let wn = row_at(f, w, i)?;
            let un = row_at(f, ut, i)?;
            let kn = row_at(f, kt, i)?;
            let gn = row_at(f, gc, i)?;
            let ws = f.dot_general(wn, s0, &[0], &[0], &[2], &[2], Elem::F32)?;
            let u = f.sub(un, ws)?;
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
    let ws = f.dot_general(w, starts, &[0, 1], &[0, 1], &[3], &[3], Elem::F32)?;
    let u = f.sub(ut, ws)?;
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
enum DeltaLanes {
    Decode,
    Ragged(Tensor),
    Committed(Tensor, Committed),
}

/// The operands a delta rule reads: GDN's or KDA's.
enum DeltaIn {
    Gdn {
        qkv: Tensor,
        gates: Tensor,
        hk: i64,
        hv: i64,
        dk: i64,
        dv: i64,
    },
    Kda {
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        dt_bias: Tensor,
        a_log: Tensor,
        heads: i64,
        d: i64,
        eps: f32,
        floor: f32,
    },
}

impl DeltaIn {
    fn dims(&self) -> (i64, i64, i64) {
        match *self {
            Self::Gdn { hv, dk, dv, .. } => (hv, dk, dv),
            Self::Kda { heads, d, .. } => (heads, d, d),
        }
    }

    fn rows(&self) -> u32 {
        match *self {
            Self::Gdn { qkv, .. } => qkv.rows,
            Self::Kda { mixed, .. } => mixed.rows,
        }
    }

    fn chunk(&self, rows: i64, lanes: i64) -> i64 {
        match self {
            Self::Gdn { .. } if rows >= LONG_CHUNK * lanes.max(1) => LONG_CHUNK,
            Self::Gdn { .. } => CHUNK,
            Self::Kda { .. } => KDA_CHUNK,
        }
    }

    /// 1 for a scalar decay per head, the key width for a per-channel one.
    fn decay_width(&self) -> i64 {
        match *self {
            Self::Gdn { .. } => 1,
            Self::Kda { d, .. } => d,
        }
    }

    /// The per-row operands as they are stored, and the per-head banks.
    fn raw(&self, cx: &mut Cx<'_>) -> Result<Raw, Error> {
        Ok(match *self {
            Self::Gdn { qkv, gates, .. } => Raw {
                rows: vec![cx.read(qkv)?, cx.read_f32(gates)?],
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
                rows: vec![cx.read(mixed)?, cx.read(f)?, cx.read(b)?],
                fixed: vec![flat_f32(cx, dt_bias)?, flat_f32(cx, a_log)?],
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
        }
    }
}

/// A delta rule's operands as read: `rows` are `[T, *]` per-token planes,
/// `fixed` per-head banks.
struct Raw {
    rows: Vec<Val>,
    fixed: Vec<Val>,
}

fn delta(
    ctx: &Ctx<'_>,
    op: &'static str,
    input: DeltaIn,
    lanes: DeltaLanes,
    state: &RecurrentPool,
    y: Tensor,
) -> Result<(), Error> {
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
        state.state,
        stride as u64,
    )?;
    match lanes {
        DeltaLanes::Decode => {
            if state.slots.elements() < u64::from(rows) {
                return Err(refuse(
                    op,
                    "the pool names fewer slots than the fire has rows",
                ));
            }
        }
        DeltaLanes::Ragged(indptr) | DeltaLanes::Committed(indptr, _) => csr_lanes(op, indptr)?,
    }
    let lanes_n = match lanes {
        DeltaLanes::Decode => i64::from(rows),
        DeltaLanes::Ragged(ip) | DeltaLanes::Committed(ip, _) => ip.elements() as i64 - 1,
    };
    let chunk = input.chunk(i64::from(rows), lanes_n);
    let chunk = chunk.min(i64::from(rows.next_power_of_two().max(8)));
    let t = i64::from(rows);
    ctx.emit(&mut |cx| {
        let raw = input.raw(cx)?;
        let slab = cx.read(state.state)?;
        let (yv, slab) = match lanes {
            DeltaLanes::Decode => {
                let slots = flat_i32(cx, state.slots)?;
                let f = cx.func();
                let lanes = decode_lanes(f, slots, t)?;
                let heads = input.heads_from(f, &raw.rows, &raw.fixed)?;
                // Over a slot-blocked bank, rounds of lanes, each gathered,
                // stepped (in f32) and landed before the next, so a round's
                // states stay on chip. A padded lane sharing the sink slot
                // with an earlier round's reads what that round left there,
                // which is garbage either way.
                let round = if per == 1 {
                    t
                } else {
                    (DECODE_ROUND_BYTES / (stride * 4)).clamp(1, t)
                };
                let mut slab = slab;
                let mut ys = Vec::new();
                let mut lo = 0;
                while lo < t {
                    let hi = (lo + round).min(t);
                    let cut = |f: &mut Func, x: Val| f.slice_axis(x, 0, lo, hi);
                    let part = if lo == 0 && hi == t {
                        heads
                    } else {
                        Heads {
                            q: cut(f, heads.q)?,
                            k: cut(f, heads.k)?,
                            v: cut(f, heads.v)?,
                            g: cut(f, heads.g)?,
                            beta: cut(f, heads.beta)?,
                        }
                    };
                    let slot = cut(f, lanes.slot)?;
                    let write = cut(f, lanes.write)?;
                    let s = take_slots(f, slab, slot, per)?;
                    let s = f.convert(s, Elem::F32);
                    let s = f.reshape(s, &[hi - lo, hh, dv, dk])?;
                    let (yv, s) = delta_step(f, &part, s)?;
                    let s = f.reshape(s, &[hi - lo, stride])?;
                    slab = land_slots(f, slab, slot, write, s, per)?;
                    ys.push(yv);
                    lo = hi;
                }
                let yv = if ys.len() == 1 {
                    ys[0]
                } else {
                    f.concat(&ys, 0)?
                };
                (yv, slab)
            }
            DeltaLanes::Ragged(indptr) | DeltaLanes::Committed(indptr, _) => {
                let ip = flat_i32(cx, indptr)?;
                let (lanes, segs) = if let DeltaLanes::Committed(_, cm) = lanes {
                    let rep = flat_i32(cx, cm.replay)?;
                    let com = flat_i32(cx, cm.commit)?;
                    let sl = flat_i32(cx, cm.slots)?;
                    let f = cx.func();
                    let lanes = committed_lanes(op, f, ip, rep, com, sl, cm.lane0)?;
                    let segs = committed_segs(f, &lanes)?;
                    (lanes, segs)
                } else {
                    let slots = flat_i32(cx, state.slots)?;
                    let f = cx.func();
                    let lanes = ragged_lanes(f, ip, slots)?;
                    let segs = whole_segs(f, &lanes);
                    (lanes, segs)
                };
                let f = cx.func();
                let l = lanes.count(f);
                let init = take_slots(f, slab, lanes.slot, per)?;
                let init = f.convert(init, Elem::F32);
                let init = f.reshape(init, &[l, hh, dv, dk])?;
                let (yv, fin) = delta_chunked(f, &input, &raw, &segs, init, chunk)?;
                let fin = f.reshape(fin, &[l, stride])?;
                (yv, land_slots(f, slab, lanes.slot, lanes.write, fin, per)?)
            }
        };
        cx.write(y, yv)?;
        cx.write(state.state, slab)
    })
}

fn gdn_in(
    op: &'static str,
    qkv: Tensor,
    gates: Tensor,
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
    if i64::from(qkv.width) != 2 * hk * dk + hv * dv {
        return Err(refuse(
            op,
            "the post-convolution qkv's row is not the four stated head numbers",
        ));
    }
    if gates.rows != qkv.rows || i64::from(gates.width) != 2 * hv {
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
/// (`z` gates it later, in the gated rmsnorm).
pub fn gated_delta(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta";
    let _ = z;
    let input = gdn_in(OP, qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
    delta(ctx, OP, input, DeltaLanes::Decode, state, y)
}

/// The gated delta rule over a query CSR, chunked; `state.slots` is per row.
pub fn gated_delta_chunked(
    ctx: &Ctx<'_>,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta_chunked";
    let _ = z;
    let input = gdn_in(OP, qkv.data, gates, k_heads, v_heads, k_dim, v_dim)?;
    delta(ctx, OP, input, DeltaLanes::Ragged(qkv.indptr), state, y)
}

/// The gated delta rule over a rollback seat's extended rows: every row's
/// `y`, and the state after each lane's committed prefix. The GPU's `work`
/// scratch plane is not an operand here.
pub fn gated_delta_committed(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta_committed";
    let input = gdn_in(OP, qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
    delta(
        ctx,
        OP,
        input,
        DeltaLanes::Committed(indptr, *committed),
        state,
        y,
    )
}

fn kda_in(
    op: &'static str,
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
) -> Result<DeltaIn, Error> {
    expect(op, mixed, &[Dtype::Bf16])?;
    nonzero(op, "the KDA heads this statement states", heads)?;
    nonzero(op, "the KDA head width this statement states", head_dim)?;
    let (h, d) = (i64::from(heads), i64::from(head_dim));
    if i64::from(mixed.width) != 3 * h * d {
        return Err(refuse(
            op,
            "the post-convolution `[q | k | v]` row is not three head planes",
        ));
    }
    if f.rows != mixed.rows || i64::from(f.width) != h * d {
        return Err(refuse(
            op,
            "the forget projection's row is not one head plane",
        ));
    }
    if b.rows != mixed.rows || i64::from(b.width) != h {
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

/// One token per lane of KDA (per-channel decay delta rule).
pub fn kda_step(
    ctx: &Ctx<'_>,
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_step";
    let input = kda_in(
        OP, mixed, f, b, dt_bias, a_log, heads, head_dim, norm_eps, gate_floor,
    )?;
    delta(ctx, OP, input, DeltaLanes::Decode, state, y)
}

/// KDA over a query CSR, chunked; `state.slots` is per row.
pub fn kda_chunked(
    ctx: &Ctx<'_>,
    mixed: RaggedTensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_chunked";
    let input = kda_in(
        OP, mixed.data, f, b, dt_bias, a_log, heads, head_dim, norm_eps, gate_floor,
    )?;
    delta(ctx, OP, input, DeltaLanes::Ragged(mixed.indptr), state, y)
}

/// KDA over a rollback seat's extended rows. The GPU's `work` scratch plane
/// is not an operand here.
pub fn kda_committed(
    ctx: &Ctx<'_>,
    mixed: Tensor,
    indptr: Tensor,
    committed: &Committed,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_committed";
    let input = kda_in(
        OP, mixed, f, b, dt_bias, a_log, heads, head_dim, norm_eps, gate_floor,
    )?;
    delta(
        ctx,
        OP,
        input,
        DeltaLanes::Committed(indptr, *committed),
        state,
        y,
    )
}

// ------------------------------------------------------------ dynamic conv

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
    const OP: &str = "attention.block_dyn_conv";
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
