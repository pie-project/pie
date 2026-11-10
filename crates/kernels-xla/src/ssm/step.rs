//! A decode fire's gated delta rule as one Mosaic kernel: each lane's state
//! is copied out of the bank into VMEM (the next lane's copy in flight while
//! this one steps), stepped there and copied back in place. The XLA form
//! gathers 16 lanes' states a round into a slab, steps the slab and scatters
//! it back: three passes over every state a layer.

use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};
use crate::lanes::{clamp_i, cmp_i};
use crate::mosaic::{Body, Grid, ICmp, Kernel, MTy, Operand, Sem, Space, V, Window};

use super::Heads;

/// `(y, slab)` for one token a lane, or `None` where the kernel does not
/// apply: heads 128 wide on both sides, a scalar decay a head, the bank bf16
/// with one slot of `H · dk` rows a lane.
pub(super) fn gdn_step(
    f: &mut Func,
    h: &Heads,
    slot: Val,
    slab: Val,
    per: i64,
) -> Built<Option<(Val, Val)>> {
    let qd = f.dims(h.q).to_vec();
    let vd = f.dims(h.v).to_vec();
    let (t, hv, dk, dv) = (qd[0], qd[1], qd[2], vd[2]);
    let rows = f.dims(slab)[0];
    if dk != 128
        || dv != 128
        || hv % 8 != 0
        || per != hv * dk
        || f.dims(h.g) != [t, hv, 1]
        || f.elem(slab) != Elem::Bf16
        || f.dims(slab)[1] != dv
        || rows % per != 0
    {
        return Ok(None);
    }
    let g = if t % 8 == 0 { 8 } else { t };
    let slots = rows / per;
    // a lane routed out of the bank (either way) lands on the sink, the last slot
    let sink = f.like_i(slot, slots - 1);
    let neg = cmp_i(f, Cmp::Lt, slot, 0)?;
    let slot = f.select(neg, sink, slot)?;
    let slot = clamp_i(f, slot, 0, slots - 1)?;
    // Per head, along the value lanes: the decay, `β` and `q·k`.
    let lanes = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.reshape(x, &[t, hv])?;
        f.broadcast(x, &[t, hv, dv], &[0, 1])
    };
    let a = f.exp(h.g);
    let a = lanes(f, a)?;
    let beta = lanes(f, h.beta)?;
    let qk = f.mul(h.q, h.k)?;
    let qk = f.reduce(qk, &[2], Fold::Sum)?;
    let qk = lanes(f, qk)?;
    let compiled = kernel(t, g, hv, rows, per)?;
    let out = compiled.call_aliased(f, &[slot, h.q, h.k, h.v, a, beta, qk, slab], &[(1, 7)])?;
    Ok(Some((out[0], out[1])))
}

fn kernel(t: i64, g: i64, hv: i64, rows: i64, per: i64) -> Built<crate::mosaic::Compiled> {
    let (dk, dv) = (128, 128);
    let lane = |w: i64| Operand {
        array: vec![t, hv, w],
        elem: Elem::F32,
        block: vec![g, hv, w],
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, gr: &Grid| {
            let z = b.i32(0);
            Ok(vec![gr.ids[0], z, z])
        }),
    };
    let bank = || Operand {
        array: vec![rows, dv],
        elem: Elem::Bf16,
        block: vec![rows, dv],
        window: Window::Any,
        index: Box::new(|b: &mut Body, _: &Grid| Ok(vec![b.i32(0), b.i32(0)])),
    };
    let state = per * dv;
    // the two state buffers and a few f32 copies of a state as it steps
    let nb: i64 = super::step_bufs_n();
    let heads = super::step_heads_on();
    let mxu = super::step_mxu_on();
    let vmem = nb * state * 2 + 6 * state * 4 + 2 * 6 * g * hv * 128 * 4 + (8 << 20);
    let k = Kernel {
        name: "gdn_step".into(),
        grid: vec![t / g],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![t]],
        ins: vec![
            lane(dk),
            lane(dk),
            lane(dv),
            lane(dv),
            lane(dv),
            lane(dv),
            bank(),
        ],
        outs: vec![lane(dv), bank()],
        vmem_limit: Some(u64::try_from(vmem).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![nb, per, dv], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![2, nb]),
        ],
    };
    k.build(|b, blk| {
        let n = blk.grid.ids[0];
        let slot_of = blk.grid.prefetch[0];
        let (q_r, k_r, v_r, a_r, b_r, qk_r, bank_in) = (
            blk.ins[0], blk.ins[1], blk.ins[2], blk.ins[3], blk.ins[4], blk.ins[5], blk.ins[6],
        );
        let (y_r, bank_out) = (blk.outs[0], blk.outs[1]);
        let (buf, sems) = (blk.scratch[0], blk.scratch[1]);
        // Row `row`'s state between its slot of the bank and buffer `s`:
        // loaded or stored, started or waited for.
        let copy = |b: &mut Body, row: V, s: V, store: bool, wait: bool| -> Built<()> {
            let sl = b.load(slot_of, &[row])?;
            let pv = b.i32(per);
            let at = b.muli(sl, pv)?;
            let z = b.i32(0);
            let hbm = b.ref_slice(if store { bank_out } else { bank_in }, &[at, z], &[per, dv])?;
            let vm = b.ref_slice(buf, &[s, z, z], &[1, per, dv])?;
            let vm = b.ref_squeeze(vm)?;
            let kind = b.i32(i64::from(store));
            let sem = b.ref_slice(sems, &[kind, s], &[1, 1])?;
            let sem = b.ref_squeeze(sem)?;
            let sem = b.ref_squeeze(sem)?;
            let (src, dst) = if store { (vm, hbm) } else { (hbm, vm) };
            if wait {
                b.dma_wait(sem, src, dst)
            } else {
                b.dma_start(src, dst, sem)
            }
        };
        let gv = b.i32(g);
        let base = b.muli(n, gv)?;
        let zero = b.i32(0);
        let first = b.cmpi(ICmp::Eq, n, zero)?;
        b.when(first, |b| {
            let z = b.i32(0);
            copy(b, z, z, false, false)
        })?;
        let (z, one) = (b.i32(0), b.i32(1));
        b.for_range(z, gv, one, |b, r| {
            let row = b.addi(base, r)?;
            let nbv = b.i32(nb);
            let s = b.remsi(row, nbv)?;
            let one = b.i32(1);
            let next = b.addi(row, one)?;
            // the next row's buffer: the one the row `nb - 1` before stored
            // out of, waited for before it takes the next row
            let other = b.remsi(next, nbv)?;
            let back = b.i32(nb - 1);
            let past = b.cmpi(ICmp::Sge, row, back)?;
            b.when(past, |b| {
                let back = b.i32(nb - 1);
                let prev = b.subi(row, back)?;
                copy(b, prev, other, true, true)
            })?;
            let tv = b.i32(t);
            let more = b.cmpi(ICmp::Slt, next, tv)?;
            b.when(more, |b| copy(b, next, other, false, false))?;
            copy(b, row, s, false, true)?;

            let ri = b.as_index(r)?;
            let si = b.as_index(s)?;
            let i0 = b.index(0);
            let at = |b: &mut Body, x: V, w: i64| -> Built<V> {
                let x = b.vload(x, &[ri, i0, i0], &[1, hv, w])?;
                b.shape_cast(x, &[hv, w])
            };
            if mxu {
                // a head at a time through the MXU: the state is bf16, so one
                // bf16 pass over q and k split into high and low halves reads
                // `q·S` and `k·S` near f32, and `k ⊗ delta` comes as a
                // transposed product of their halves
                let a = at(b, a_r, dv)?;
                let q = at(b, q_r, dk)?;
                let k = at(b, k_r, dk)?;
                let v = at(b, v_r, dv)?;
                let beta = at(b, b_r, dv)?;
                let qk = at(b, qk_r, dv)?;
                let split = |b: &mut Body, x: V| -> Built<(V, V)> {
                    let hi = b.convert_f(x, Elem::Bf16)?;
                    let hi = b.convert_f(hi, Elem::F32)?;
                    let lo = b.subf(x, hi)?;
                    Ok((hi, lo))
                };
                let (qh, ql) = split(b, q)?;
                let (kh, kl) = split(b, k)?;
                let zk = b.splat_f(&[12, dk], Elem::F32, 0.0)?;
                let zk5 = b.splat_f(&[13, dk], Elem::F32, 0.0)?;
                let zv5 = b.splat_f(&[13, dv], Elem::F32, 0.0)?;
                let acc8 = b.splat_f(&[16, dv], Elem::F32, 0.0)?;
                let acc_s = b.splat_f(&[dk, dv], Elem::F32, 0.0)?;
                for h in 0..hv {
                    let off = b.index(h * dk);
                    let st = b.vload(buf, &[si, off, i0], &[1, dk, dv])?;
                    let st = b.shape_cast(st, &[dk, dv])?;
                    let row = |b: &mut Body, x: V, w: i64| b.slice(x, &[h, 0], &[1, w]);
                    let pieces = [row(b, qh, dk)?, row(b, ql, dk)?, row(b, kh, dk)?, row(b, kl, dk)?, zk];
                    let lhs = b.concat(&pieces, 0)?;
                    let lhs = b.convert_f(lhs, Elem::Bf16)?;
                    let p = b.matmul(lhs, st, acc8, false)?;
                    let p0 = b.rows(p, 0, 1)?;
                    let p1 = b.rows(p, 1, 1)?;
                    let p2 = b.rows(p, 2, 1)?;
                    let p3 = b.rows(p, 3, 1)?;
                    let ah = row(b, a, dv)?;
                    let sq = b.addf(p0, p1)?;
                    let sq = b.mulf(sq, ah)?;
                    let mem = b.addf(p2, p3)?;
                    let mem = b.mulf(mem, ah)?;
                    let vh = row(b, v, dv)?;
                    let bh = row(b, beta, dv)?;
                    let qkh = row(b, qk, dv)?;
                    let delta = b.subf(vh, mem)?;
                    let delta = b.mulf(bh, delta)?;
                    let dy = b.mulf(delta, qkh)?;
                    let y = b.addf(sq, dy)?;
                    let y = b.shape_cast(y, &[1, 1, dv])?;
                    let hi = b.index(h);
                    b.vstore(y, y_r, &[ri, hi, i0])?;
                    let (dh, dl) = split(b, delta)?;
                    let kh1 = row(b, kh, dk)?;
                    let kl1 = row(b, kl, dk)?;
                    let lk = b.concat(&[kh1, kh1, kl1, zk5], 0)?;
                    let lk = b.convert_f(lk, Elem::Bf16)?;
                    let rd = b.concat(&[dh, dl, dh, zv5], 0)?;
                    let rd = b.convert_f(rd, Elem::Bf16)?;
                    let upd = b.matmul_tl(lk, rd, acc_s)?;
                    let st32 = b.convert_f(st, Elem::F32)?;
                    let a2 = b.broadcast(ah, &[dk, dv])?;
                    let s1 = b.mulf(st32, a2)?;
                    let s2 = b.addf(s1, upd)?;
                    let s2 = b.convert_f(s2, Elem::Bf16)?;
                    let s2 = b.shape_cast(s2, &[1, dk, dv])?;
                    b.vstore(s2, buf, &[si, off, i0])?;
                }
            } else if heads {
                // a head at a time: each head's state, its columns of q and
                // k and its products stay a few vregs, not whole-state
                // temporaries in VMEM
                let a = at(b, a_r, dv)?;
                let q = at(b, q_r, dk)?;
                let k = at(b, k_r, dk)?;
                let v = at(b, v_r, dv)?;
                let beta = at(b, b_r, dv)?;
                let qk = at(b, qk_r, dv)?;
                let full = [1, dk, dv];
                for h in 0..hv {
                    let off = b.index(h * dk);
                    let st = b.vload(buf, &[si, off, i0], &[1, dk, dv])?;
                    let st = b.convert_f(st, Elem::F32)?;
                    let row = |b: &mut Body, x: V, w: i64| b.slice(x, &[h, 0], &[1, w]);
                    let column = |b: &mut Body, x: V| -> Built<V> {
                        let x = row(b, x, dk)?;
                        let x = b.shape_cast(x, &[1, dk, 1])?;
                        b.broadcast(x, &full)
                    };
                    let qb = column(b, q)?;
                    let kb = column(b, k)?;
                    let ah = row(b, a, dv)?;
                    let sq = b.mulf(st, qb)?;
                    let sq = b.reduce_sum(sq, 1)?;
                    let sq = b.mulf(sq, ah)?;
                    let mem = b.mulf(st, kb)?;
                    let mem = b.reduce_sum(mem, 1)?;
                    let mem = b.mulf(mem, ah)?;
                    let vh = row(b, v, dv)?;
                    let bh = row(b, beta, dv)?;
                    let qkh = row(b, qk, dv)?;
                    let delta = b.subf(vh, mem)?;
                    let delta = b.mulf(bh, delta)?;
                    let dy = b.mulf(delta, qkh)?;
                    let y = b.addf(sq, dy)?;
                    let y = b.shape_cast(y, &[1, 1, dv])?;
                    let hi = b.index(h);
                    b.vstore(y, y_r, &[ri, hi, i0])?;
                    let a3 = b.shape_cast(ah, &[1, 1, dv])?;
                    let a3 = b.broadcast(a3, &full)?;
                    let s1 = b.mulf(st, a3)?;
                    let d3 = b.shape_cast(delta, &[1, 1, dv])?;
                    let d3 = b.broadcast(d3, &full)?;
                    let upd = b.mulf(kb, d3)?;
                    let s2 = b.addf(s1, upd)?;
                    let s2 = b.convert_f(s2, Elem::Bf16)?;
                    b.vstore(s2, buf, &[si, off, i0])?;
                }
            } else {
                let st = b.vload(buf, &[si, i0, i0], &[1, per, dv])?;
                let st = b.convert_f(st, Elem::F32)?;
                let st = b.shape_cast(st, &[hv, dk, dv])?;
                let full = [hv, dk, dv];
                let a = at(b, a_r, dv)?;
                let a = b.shape_cast(a, &[hv, 1, dv])?;
                let a = b.broadcast(a, &full)?;
                let s1 = b.mulf(st, a)?;
                let column = |b: &mut Body, x: V| -> Built<V> {
                    let x = at(b, x, dk)?;
                    let x = b.shape_cast(x, &[hv, dk, 1])?;
                    b.broadcast(x, &full)
                };
                let qb = column(b, q_r)?;
                let kb = column(b, k_r)?;
                let sq = b.mulf(s1, qb)?;
                let sq = b.reduce_sum(sq, 1)?;
                let mem = b.mulf(s1, kb)?;
                let mem = b.reduce_sum(mem, 1)?;
                let v = at(b, v_r, dv)?;
                let beta = at(b, b_r, dv)?;
                let qk = at(b, qk_r, dv)?;
                let delta = b.subf(v, mem)?;
                let delta = b.mulf(beta, delta)?;
                let dy = b.mulf(delta, qk)?;
                let y = b.addf(sq, dy)?;
                let y = b.shape_cast(y, &[1, hv, dv])?;
                b.vstore(y, y_r, &[ri, i0, i0])?;
                let d = b.shape_cast(delta, &[hv, 1, dv])?;
                let d = b.broadcast(d, &full)?;
                let upd = b.mulf(kb, d)?;
                let s2 = b.addf(s1, upd)?;
                let s2 = b.convert_f(s2, Elem::Bf16)?;
                let s2 = b.shape_cast(s2, &[1, per, dv])?;
                b.vstore(s2, buf, &[si, i0, i0])?;
            }
            copy(b, row, s, true, false)?;
            let last = b.i32(t - 1);
            let fin = b.cmpi(ICmp::Eq, row, last)?;
            b.when(fin, |b| {
                // every store still in flight: the last `nb - 1` rows'
                for back in (1..nb - 1).rev() {
                    if t - 1 - back >= 0 {
                        let bk = b.i32(back);
                        let prev = b.subi(row, bk)?;
                        let nbv = b.i32(nb);
                        let ps = b.remsi(prev, nbv)?;
                        copy(b, prev, ps, true, true)?;
                    }
                }
                copy(b, row, s, true, true)
            })
        })
    })
}
