//! A batched decode's rows reading their own lanes' keys as one Mosaic
//! kernel. The per-row walk (`paged::per_row`) gathers every row's chunk up
//! to the batch's longest context into a slab in HBM and reads the slab back:
//! at 256 lanes three times the keys' own bytes, all rows paying for the
//! longest. Here each row copies only the pool pages its `[lo, hi)` range
//! covers into VMEM (a DMA a page), the next row's copies in flight while
//! this row computes, and walks them in chunks with an online softmax.
//!
//! A row's scores come as `[keys, q heads]`: the key chunk (all kv heads,
//! `[c, kvh·d]`) times a block-diagonal query (`[qh, kvh·d]`, each query
//! head's row zero outside its kv head's columns), so the small operand is
//! the one the MXU holds and the keys stream through it; the values meet the
//! probabilities as `[kvh·dv, qh]` (transposed values times probabilities),
//! each query head's column read off its own kv head's rows at the end.

#![allow(clippy::too_many_arguments)]

use crate::hlo::{Built, Elem, Func, Malformed, Val};
use crate::mosaic::{Body, Compiled, Grid, ICmp, Kernel, MTy, Operand, Sem, Space, V, Window};

use super::paged::{Flashed, Heads};

fn bad<T>(detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op: "attn.pages",
        detail: detail.into(),
    })
}

/// Rows a grid step takes.
const G: i64 = 8;

/// Keys a chunk of a row's walk.
const CHUNK: i64 = 128;

/// One kernel's static shape.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Rows, padded to whole steps.
    rp: i64,
    /// Keys a row's buffer holds (the lane span).
    span: i64,
    ps: i64,
    /// Entries of the dense page table (`lanes · pages`).
    table: i64,
    pages: i64,
    slots: i64,
    h: Heads,
    scale: f64,
}

impl Shape {
    fn vmem(span: i64, h: &Heads) -> i64 {
        let (qh, w, wv) = (h.qh(), h.kvh * h.d, h.kvh * h.dv);
        // both slots of keys and values, the state, the blocks, room for
        // a chunk's temporaries
        2 * span * (w + wv) * 2
            + wv * qh.max(128) * 4
            + 2 * G * (qh * h.d * 2 + wv * qh.max(128) * 4)
            + 8 * CHUNK * (w + wv) * 4
    }
}

/// Whether the kernel takes rows of these heads reading `span` keys.
pub(crate) fn fits(span: i64, h: &Heads, vmem_cap: i64) -> bool {
    h.d % 128 == 0
        && h.dv % 128 == 0
        && h.qh() <= 128
        && span % CHUNK == 0
        && Shape::vmem(span, h) <= vmem_cap
}

fn kernel(s: &Shape, vmem_cap: i64) -> Built<Compiled> {
    let Shape {
        rp,
        span,
        ps,
        table,
        pages,
        slots,
        h,
        scale,
    } = *s;
    let (qh, gq, d, dv, kvh) = (h.qh(), h.g, h.d, h.dv, h.kvh);
    let (w, wv) = (kvh * d, kvh * dv);
    let steps = rp / G;
    let rows = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let zero = b.i32(0);
        Ok(vec![gr.ids[0], zero])
    };
    let block = |array: Vec<i64>, elem: Elem, block: Vec<i64>| Operand {
        array,
        elem,
        block,
        window: Window::Blocked,
        index: Box::new(rows),
    };
    let whole = |array: Vec<i64>| Operand {
        block: array.clone(),
        array,
        elem: Elem::Bf16,
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    };
    let k = Kernel {
        name: "decode_pages".into(),
        grid: vec![steps],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![table], vec![rp], vec![rp], vec![rp]],
        ins: vec![
            block(vec![rp * qh, d], Elem::Bf16, vec![G * qh, d]),
            whole(vec![slots, w]),
            whole(vec![slots, wv]),
        ],
        outs: vec![
            block(vec![rp * dv, qh], Elem::F32, vec![G * dv, qh]),
            block(vec![rp, qh], Elem::F32, vec![G, qh]),
            block(vec![rp, qh], Elem::F32, vec![G, qh]),
        ],
        vmem_limit: Some(u64::try_from(vmem_cap).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![2, span, w], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![2, span, wv], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![2, 2]),
            MTy::Ref(vec![1, qh], Elem::F32, Space::Vmem),
            MTy::Ref(vec![1, qh], Elem::F32, Space::Vmem),
            MTy::Ref(vec![wv, qh], Elem::F32, Space::Vmem),
        ],
    };
    k.build(|b, blk| {
        let i = blk.grid.ids[0];
        let (pt, lane, lo, hi) = (
            blk.grid.prefetch[0],
            blk.grid.prefetch[1],
            blk.grid.prefetch[2],
            blk.grid.prefetch[3],
        );
        let (q_ref, kp, vp) = (blk.ins[0], blk.ins[1], blk.ins[2]);
        let (o_ref, m_ref, l_ref) = (blk.outs[0], blk.outs[1], blk.outs[2]);
        let (kbuf, vbuf, sems) = (blk.scratch[0], blk.scratch[1], blk.scratch[2]);
        let (m_sc, l_sc, a_sc) = (blk.scratch[3], blk.scratch[4], blk.scratch[5]);
        // Every page of row `row`'s range, from its first page on, into
        // `slot` (`wait`: waited for).
        let copies = |b: &mut Body, row: V, slot: V, wait: bool| -> Built<()> {
            let ln = b.load(lane, &[row])?;
            let l0 = b.load(lo, &[row])?;
            let l1 = b.load(hi, &[row])?;
            let psv = b.i32(ps);
            let p0 = b.divsi(l0, psv)?;
            let top = b.i32(ps - 1);
            let p1 = b.addi(l1, top)?;
            let p1 = b.divsi(p1, psv)?;
            let p1 = b.maxsi(p1, p0)?;
            let one = b.i32(1);
            b.for_range(p0, p1, one, |b, p| {
                let pv = b.i32(pages);
                let at = b.muli(ln, pv)?;
                let at = b.addi(at, p)?;
                let page = b.load(pt, &[at])?;
                let psv = b.i32(ps);
                let src_row = b.muli(page, psv)?;
                let dp = b.subi(p, p0)?;
                let dst_row = b.muli(dp, psv)?;
                let z = b.i32(0);
                for (plane, buf, width, which) in [(kp, kbuf, w, 0), (vp, vbuf, wv, 1)] {
                    let src = b.ref_slice(plane, &[src_row, z], &[ps, width])?;
                    let dst = b.ref_slice(buf, &[slot, dst_row, z], &[1, ps, width])?;
                    let dst = b.ref_squeeze(dst)?;
                    let wh = b.i32(which);
                    let sem = b.ref_slice(sems, &[wh, slot], &[1, 1])?;
                    let sem = b.ref_squeeze(sem)?;
                    let sem = b.ref_squeeze(sem)?;
                    if wait {
                        b.dma_wait(sem, src, dst)?;
                    } else {
                        b.dma_start(src, dst, sem)?;
                    }
                }
                Ok(())
            })
        };
        let gv = b.i32(G);
        let base = b.muli(i, gv)?;
        let zero = b.i32(0);
        let first = b.cmpi(ICmp::Eq, i, zero)?;
        b.when(first, |b| {
            let z = b.i32(0);
            copies(b, z, z, false)
        })?;

        // the block-diagonal query's column heads and row heads
        let dvec = b.splat_i(&[qh, w], d)?;
        let io = b.iota(&[qh, w], 1);
        let colh = b.divsi(io, dvec)?;
        let gvec = b.splat_i(&[qh, w], gq)?;
        let io = b.iota(&[qh, w], 0);
        let rowh = b.divsi(io, gvec)?;
        let diag = b.cmpi(ICmp::Eq, colh, rowh)?;
        let zq = b.splat_f(&[qh, w], Elem::Bf16, 0.0)?;
        // a query head's kv head, along the lanes of `[dv, qh]`
        let gvo = b.splat_i(&[dv, qh], gq)?;
        let io = b.iota(&[dv, qh], 1);
        let head_o = b.divsi(io, gvo)?;
        let key_s = b.iota(&[CHUNK, qh], 0);
        let key_v = b.iota(&[CHUNK, wv], 0);
        let ninf_s = b.splat_f(&[CHUNK, qh], Elem::F32, f64::NEG_INFINITY)?;
        let ninf1 = b.splat_f(&[1, qh], Elem::F32, f64::NEG_INFINITY)?;
        let z1 = b.splat_f(&[1, qh], Elem::F32, 0.0)?;
        let o1 = b.splat_f(&[1, qh], Elem::F32, 1.0)?;
        let zs = b.splat_f(&[CHUNK, qh], Elem::F32, 0.0)?;
        let za = b.splat_f(&[wv, qh], Elem::F32, 0.0)?;
        let zo = b.splat_f(&[dv, qh], Elem::F32, 0.0)?;
        let zv = b.splat_f(&[CHUNK, wv], Elem::Bf16, 0.0)?;
        let sc = b.splat_f(&[CHUNK, qh], Elem::F32, scale)?;

        let (z, gn, step1) = (b.i32(0), b.i32(G), b.i32(1));
        b.for_range(z, gn, step1, |b, r| {
            let row = b.addi(base, r)?;
            let two = b.i32(2);
            let slot = b.remsi(row, two)?;
            // the next row's pages fly while this row computes
            let one = b.i32(1);
            let next = b.addi(row, one)?;
            let rpv = b.i32(rp);
            let more = b.cmpi(ICmp::Slt, next, rpv)?;
            b.when(more, |b| {
                let one = b.i32(1);
                let other = b.subi(one, slot)?;
                copies(b, next, other, false)
            })?;
            copies(b, row, slot, true)?;

            let l0 = b.load(lo, &[row])?;
            let l1 = b.load(hi, &[row])?;
            let psv = b.i32(ps);
            let p0 = b.divsi(l0, psv)?;
            let k0 = b.muli(p0, psv)?;
            // buffer key `j` is position `k0 + j`
            let lo_k = b.subi(l0, k0)?;
            let hi_k = b.subi(l1, k0)?;
            let hi_k = b.maxsi(hi_k, lo_k)?;
            let qhv = b.i32(qh);
            let at = b.muli(r, qhv)?;
            let at = b.as_index(at)?;
            let i0 = b.index(0);
            let q = b.vload(q_ref, &[at, i0], &[qh, d])?;
            let reps: Vec<V> = (0..kvh).map(|_| q).collect();
            let qt = b.concat(&reps, 1)?;
            let qbd = b.select(diag, qt, zq)?;
            let sidx = b.as_index(slot)?;
            b.vstore(ninf1, m_sc, &[i0, i0])?;
            b.vstore(z1, l_sc, &[i0, i0])?;
            b.vstore(za, a_sc, &[i0, i0])?;
            // the chunks of keys the row's range covers
            let cv = b.i32(CHUNK);
            let c0 = b.divsi(lo_k, cv)?;
            let top = b.i32(CHUNK - 1);
            let c1 = b.addi(hi_k, top)?;
            let c1 = b.divsi(c1, cv)?;
            let one = b.i32(1);
            b.for_range(c0, c1, one, |b, c| {
                let cv = b.i32(CHUNK);
                let kb = b.muli(c, cv)?;
                let kb = b.assume_multiple(kb, CHUNK)?;
                let bidx = b.as_index(kb)?;
                let i0 = b.index(0);
                let kc = b.vload(kbuf, &[sidx, bidx, i0], &[1, CHUNK, w])?;
                let kc = b.shape_cast(kc, &[CHUNK, w])?;
                let s = b.matmul(kc, qbd, zs, true)?;
                let s = b.mulf(s, sc)?;
                // keys in `[lo_k, hi_k)` only
                let off = b.broadcast_i(kb, &[CHUNK, qh])?;
                let kj = b.addi(key_s, off)?;
                let lb = b.broadcast_i(lo_k, &[CHUNK, qh])?;
                let hb = b.broadcast_i(hi_k, &[CHUNK, qh])?;
                let ge = b.cmpi(ICmp::Sge, kj, lb)?;
                let lt = b.cmpi(ICmp::Slt, kj, hb)?;
                let valid = b.andi(ge, lt)?;
                let s = b.select(valid, s, ninf_s)?;
                let m0 = b.vload(m_sc, &[i0, i0], &[1, qh])?;
                let mx = b.max_rows(s)?;
                let mn = b.maxf(m0, mx)?;
                let alive = b.cmpf(true, mn, ninf1)?;
                let ms = b.select(alive, mn, z1)?;
                let msb = b.broadcast(ms, &[CHUNK, qh])?;
                let zz = b.subf(s, msb)?;
                let p = b.exp(zz)?;
                let dm = b.subf(m0, ms)?;
                let corr = b.exp(dm)?;
                let l0 = b.vload(l_sc, &[i0, i0], &[1, qh])?;
                let l = b.mulf(l0, corr)?;
                let ps_ = b.sum_rows(p)?;
                let l = b.addf(l, ps_)?;
                b.vstore(l, l_sc, &[i0, i0])?;
                b.vstore(mn, m_sc, &[i0, i0])?;
                let vc = b.vload(vbuf, &[sidx, bidx, i0], &[1, CHUNK, wv])?;
                let vc = b.shape_cast(vc, &[CHUNK, wv])?;
                // a buffer row past the range may hold anything
                let offv = b.broadcast_i(kb, &[CHUNK, wv])?;
                let vj = b.addi(key_v, offv)?;
                let lbv = b.broadcast_i(lo_k, &[CHUNK, wv])?;
                let hbv = b.broadcast_i(hi_k, &[CHUNK, wv])?;
                let ge = b.cmpi(ICmp::Sge, vj, lbv)?;
                let lt = b.cmpi(ICmp::Slt, vj, hbv)?;
                let held = b.andi(ge, lt)?;
                let vc = b.select(held, vc, zv)?;
                let pb = b.convert_f(p, Elem::Bf16)?;
                let ov = b.matmul_tl(vc, pb, za)?;
                let a0 = b.vload(a_sc, &[i0, i0], &[wv, qh])?;
                let cb = b.broadcast(corr, &[wv, qh])?;
                let a = b.mulf(a0, cb)?;
                let a = b.addf(a, ov)?;
                b.vstore(a, a_sc, &[i0, i0])
            })?;
            // each query head's column off its own kv head's rows
            let a = b.vload(a_sc, &[i0, i0], &[wv, qh])?;
            let mut o = zo;
            for kh in 0..kvh {
                let part = b.slice(a, &[kh * dv, 0], &[dv, qh])?;
                let khv = b.splat_i(&[dv, qh], kh)?;
                let mine = b.cmpi(ICmp::Eq, head_o, khv)?;
                o = b.select(mine, part, o)?;
            }
            let m = b.vload(m_sc, &[i0, i0], &[1, qh])?;
            let l = b.vload(l_sc, &[i0, i0], &[1, qh])?;
            let pos = b.cmpf(true, l, z1)?;
            let inv = b.divf(o1, l)?;
            let inv = b.select(pos, inv, z1)?;
            let ib = b.broadcast(inv, &[dv, qh])?;
            let o = b.mulf(o, ib)?;
            let dvv = b.i32(dv);
            let ot = b.muli(r, dvv)?;
            let ot = b.as_index(ot)?;
            b.vstore(o, o_ref, &[ot, i0])?;
            let ri = b.as_index(r)?;
            b.vstore(m, m_ref, &[ri, i0])?;
            b.vstore(l, l_ref, &[ri, i0])
        })
    })
}

/// The rows' reading: `q` `[r, kvh, g, d]` bf16, row `i` reading positions
/// `[lo[i], hi[i])` of lane `lane[i]` (whose pages `pt[lane · pages + p]`
/// name) from the `keys`/`values` planes (`[slots, kvh·d]`, pages of `ps`).
pub(crate) fn attend(
    f: &mut Func,
    q: Val,
    keys: Val,
    values: Val,
    pt: Val,
    lane: Val,
    lo: Val,
    hi: Val,
    pages: i64,
    ps: i64,
    h: &Heads,
    scale: f64,
    vmem_cap: i64,
) -> Built<Flashed> {
    let r = f.dims(q)[0];
    let span = pages * ps;
    if !fits(span, h, vmem_cap) {
        return bad(format!("{span} keys of {h:?} do not fit"));
    }
    let rp = (r + G - 1) / G * G;
    let s = Shape {
        rp,
        span,
        ps,
        table: f.dims(pt)[0],
        pages,
        slots: f.dims(keys)[0],
        h: *h,
        scale,
    };
    let qh = h.qh();
    let pad_i = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.convert(x, Elem::I32);
        let x = f.reshape(x, &[r])?;
        if rp == r {
            return Ok(x);
        }
        let z = f.const_i(Elem::I32, 0, &[]);
        f.pad(x, z, &[0], &[rp - r], &[0])
    };
    let lane = pad_i(f, lane)?;
    let lo = pad_i(f, lo)?;
    let hi = pad_i(f, hi)?;
    let q = f.convert(q, Elem::Bf16);
    let q = f.reshape(q, &[r * qh, h.d])?;
    let q = if rp == r {
        q
    } else {
        let z = f.const_f(Elem::Bf16, 0.0, &[]);
        f.pad(q, z, &[0, 0], &[(rp - r) * qh, 0], &[0, 0])?
    };
    let keys = f.convert(keys, Elem::Bf16);
    let values = f.convert(values, Elem::Bf16);
    let kern = kernel(&s, vmem_cap)?;
    let out = kern.call(f, &[pt, lane, lo, hi, q, keys, values])?;
    // `[rp·dv, qh]` as `[r, kvh, g, dv]`
    let o = f.reshape(out[0], &[rp, h.dv, qh])?;
    let o = f.transpose(o, &[0, 2, 1])?;
    let o = f.slice_axis(o, 0, 0, r)?;
    let o = f.reshape(o, &[r, h.kvh, h.g, h.dv])?;
    let stat = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.slice_axis(x, 0, 0, r)?;
        f.reshape(x, &[r, h.kvh, h.g])
    };
    Ok(Flashed {
        o,
        m: stat(f, out[1])?,
        l: stat(f, out[2])?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_copies_each_rows_pages() {
        let h = Heads {
            kvh: 8,
            g: 2,
            d: 128,
            dv: 128,
        };
        let s = Shape {
            rp: 256,
            span: 512,
            ps: 16,
            table: 256 * 32,
            pages: 32,
            slots: 20000 * 16,
            h,
            scale: 0.1,
        };
        let c = kernel(&s, 100 << 20).unwrap();
        assert!(c.module.contains("tpu.enqueue_dma"));
        assert!(c.module.contains("scratch_operands = 6 : i64"));
        assert_eq!(c.operands.len(), 7);
        if let Some(p) = std::env::var_os("PAGES_DUMP") {
            std::fs::write(p, &c.module).unwrap();
        }
    }
}
