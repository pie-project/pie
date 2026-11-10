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
        || hv % HG != 0
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

/// Heads a lane steps at a time.
const HG: i64 = 2;

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
    let vmem = 2 * state * 2 + 6 * state * 4 + 2 * 6 * g * hv * 128 * 4 + (8 << 20);
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
            MTy::Ref(vec![2, per, dv], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![2, 2]),
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
            let two = b.i32(2);
            let s = b.remsi(row, two)?;
            let one = b.i32(1);
            let other = b.subi(one, s)?;
            // the row before stored out of the other buffer before that
            // buffer takes the next row
            let past = b.cmpi(ICmp::Sge, row, one)?;
            b.when(past, |b| {
                let one = b.i32(1);
                let prev = b.subi(row, one)?;
                copy(b, prev, other, true, true)
            })?;
            let next = b.addi(row, one)?;
            let tv = b.i32(t);
            let more = b.cmpi(ICmp::Slt, next, tv)?;
            b.when(more, |b| copy(b, next, other, false, false))?;
            copy(b, row, s, false, true)?;

            let ri = b.as_index(r)?;
            let si = b.as_index(s)?;
            let i0 = b.index(0);
            // A few heads at a time, so a group's state and its products stay
            // in vector registers rather than round VMEM between ops.
            let mut ys = Vec::with_capacity((hv / HG) as usize);
            for h0 in (0..hv).step_by(HG as usize) {
                let hi = b.index(h0);
                let at = |b: &mut Body, x: V, w: i64| -> Built<V> {
                    let x = b.vload(x, &[ri, hi, i0], &[1, HG, w])?;
                    b.shape_cast(x, &[HG, w])
                };
                let ro = b.index(h0 * dk);
                let st = b.vload(buf, &[si, ro, i0], &[1, HG * dk, dv])?;
                let st = b.convert_f(st, Elem::F32)?;
                let st = b.shape_cast(st, &[HG, dk, dv])?;
                let full = [HG, dk, dv];
                let a = at(b, a_r, dv)?;
                let a = b.shape_cast(a, &[HG, 1, dv])?;
                let a = b.broadcast(a, &full)?;
                let s1 = b.mulf(st, a)?;
                let column = |b: &mut Body, x: V| -> Built<V> {
                    let x = at(b, x, dk)?;
                    let x = b.shape_cast(x, &[HG, dk, 1])?;
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
                ys.push(b.addf(sq, dy)?);
                let d = b.shape_cast(delta, &[HG, 1, dv])?;
                let d = b.broadcast(d, &full)?;
                let upd = b.mulf(kb, d)?;
                let s2 = b.addf(s1, upd)?;
                let s2 = b.convert_f(s2, Elem::Bf16)?;
                let s2 = b.shape_cast(s2, &[1, HG * dk, dv])?;
                b.vstore(s2, buf, &[si, ro, i0])?;
            }
            let y = b.concat(&ys, 0)?;
            let y = b.shape_cast(y, &[1, hv, dv])?;
            b.vstore(y, y_r, &[ri, i0, i0])?;
            copy(b, row, s, true, false)?;
            let last = b.i32(t - 1);
            let fin = b.cmpi(ICmp::Eq, row, last)?;
            b.when(fin, |b| copy(b, row, s, true, true))
        })
    })
}
