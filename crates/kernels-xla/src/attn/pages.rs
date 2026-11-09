//! A decode's rows reading their lanes' keys as one Mosaic kernel: each row
//! copies the pool pages its `[lo, hi)` range covers straight into VMEM (a
//! DMA a page, the next rows' copies in flight while these rows compute)
//! and reads every head there. The per-row walk gathers those pages into a
//! slab in HBM first and reads the slab back, chunk by chunk: at a batch of
//! decode rows, most of a step.
//!
//! A grid step takes `G` rows; a row's buffer holds the lane's whole span.
//! Each query row scores against its own kv head's keys (one product per kv
//! head over all the row's query heads, the other heads' rows dropped), and
//! probabilities meet values in bf16, as the walk's do.

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

/// One kernel's static shape.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Rows, padded to whole steps.
    rp: i64,
    /// Rows a step.
    g: i64,
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

/// Keys a row reads a turn (its range only, in chunks).
const CHUNK: i64 = 128;

/// Pages one copy may move, when the pool holds them in a row.
const RUN: i64 = 4;

/// Buffer bytes a step may hold (both slots, keys and values).
const BUFFER_BYTES: i64 = 32 << 20;

impl Shape {
    fn rows_a_step(span: i64, h: &Heads) -> i64 {
        let per_row = 2 * span * h.kvh * (h.d + h.dv) * 2;
        (BUFFER_BYTES / per_row.max(1)).clamp(0, 8)
    }

    fn vmem(&self) -> i64 {
        let (g, h) = (self.g, self.h);
        let qh = h.qh();
        let buffers = 2 * g * self.span * h.kvh * (h.d + h.dv) * 2;
        let blocks = 2 * (g * qh * h.d * 2 + g * qh * h.dv * 4 + 2 * g * qh * 128 * 4);
        buffers + blocks + 8 * qh.max(8) * self.span * 4 + 4 * self.span * (h.d + h.dv) * 2
    }
}

/// Whether the kernel takes rows of these heads reading `span` keys.
pub(crate) fn fits(span: i64, h: &Heads, vmem_cap: i64) -> bool {
    let g = Shape::rows_a_step(span, h);
    h.d % 128 == 0 && h.dv % 128 == 0 && g >= 1 && {
        let s = Shape {
            rp: g,
            g,
            span,
            ps: 1,
            table: 1,
            pages: 1,
            slots: 1,
            h: *h,
            scale: 1.0,
        };
        s.vmem() <= vmem_cap
    }
}

fn kernel(s: &Shape, vmem_cap: i64) -> Built<Compiled> {
    let Shape {
        rp,
        g,
        span,
        ps,
        table,
        pages,
        slots,
        h,
        scale,
    } = *s;
    let (qh, kvh, gq, d, dv) = (h.qh(), h.kvh, h.g, h.d, h.dv);
    let (w, wv) = (kvh * d, kvh * dv);
    let steps = rp / g;
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
        name: "pages".into(),
        grid: vec![steps],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![table], vec![rp], vec![rp], vec![rp]],
        ins: vec![
            block(vec![rp * qh, d], Elem::Bf16, vec![g * qh, d]),
            whole(vec![slots, w]),
            whole(vec![slots, wv]),
        ],
        outs: vec![
            block(vec![rp * qh, dv], Elem::F32, vec![g * qh, dv]),
            block(vec![rp * qh, 1], Elem::F32, vec![g * qh, 1]),
            block(vec![rp * qh, 1], Elem::F32, vec![g * qh, 1]),
        ],
        // the buffers, and room for the compiler's own temporaries
        vmem_limit: Some(u64::try_from(vmem_cap).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![2, g, span, w], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![2, g, span, wv], Elem::Bf16, Space::Vmem),
            MTy::Sems(vec![2, 2]),
            MTy::Ref(vec![qh, 1], Elem::F32, Space::Vmem),
            MTy::Ref(vec![qh, 1], Elem::F32, Space::Vmem),
            MTy::Ref(vec![qh, dv], Elem::F32, Space::Vmem),
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
        // Every page of a step's rows, copied (`wait`: waited for).
        let copies = |b: &mut Body, step: V, slot: V, wait: bool| -> Built<()> {
            let (z, gn, one) = (b.i32(0), b.i32(g), b.i32(1));
            b.for_range(z, gn, one, |b, r| {
                let gv = b.i32(g);
                let row = b.muli(step, gv)?;
                let row = b.addi(row, r)?;
                let ln = b.load(lane, &[row])?;
                let l0 = b.load(lo, &[row])?;
                let l1 = b.load(hi, &[row])?;
                let psv = b.i32(ps);
                let p0 = b.divsi(l0, psv)?;
                let top = b.i32(ps - 1);
                let p1 = b.addi(l1, top)?;
                let p1 = b.divsi(p1, psv)?;
                let p1 = b.maxsi(p1, p0)?;
                // `n` pages from table entry `at` (consecutive in the pool
                // when `n > 1`) into the buffer from page `dp` on
                let copy = |b: &mut Body, page: V, dp: V, n: i64| -> Built<()> {
                    let psv = b.i32(ps);
                    let src_row = b.muli(page, psv)?;
                    let dst_row = b.muli(dp, psv)?;
                    let z = b.i32(0);
                    for (plane, buf, width, which) in [(kp, kbuf, w, 0), (vp, vbuf, wv, 1)] {
                        let src = b.slice_ref(plane, &[src_row, z], &[n * ps, width])?;
                        let dst =
                            b.slice_ref(buf, &[slot, r, dst_row, z], &[1, 1, n * ps, width])?;
                        let dst = b.squeeze(dst)?;
                        let wh = b.i32(which);
                        let sem = b.slice_ref(sems, &[wh, slot], &[1, 1])?;
                        let sem = b.squeeze(sem)?;
                        if wait {
                            b.dma_wait(src, dst, sem)?;
                        } else {
                            b.dma(src, dst, sem)?;
                        }
                    }
                    Ok(())
                };
                // Runs of RUN pages a copy where the pool holds them in a
                // row (a prompt's pages mostly), else a copy a page.
                let span_p = b.subi(p1, p0)?;
                let rv = b.i32(RUN - 1);
                let runs = b.addi(span_p, rv)?;
                let rv = b.i32(RUN);
                let runs = b.divsi(runs, rv)?;
                let (z, one) = (b.i32(0), b.i32(1));
                b.for_range(z, runs, one, |b, c| {
                    let rv = b.i32(RUN);
                    let off = b.muli(c, rv)?;
                    let base = b.addi(p0, off)?;
                    let pv = b.i32(pages);
                    let at = b.muli(ln, pv)?;
                    let at = b.addi(at, base)?;
                    let left = b.subi(p1, base)?;
                    let whole = b.cmpi(ICmp::Sge, left, rv)?;
                    let part = b.cmpi(ICmp::Slt, left, rv)?;
                    b.when(whole, |b| {
                        let first = b.load(pt, &[at])?;
                        let z = b.i32(0);
                        let no = b.cmpi(ICmp::Ne, z, z)?;
                        let mut row = b.cmpi(ICmp::Eq, z, z)?;
                        for i in 1..RUN {
                            let iv = b.i32(i);
                            let ai = b.addi(at, iv)?;
                            let pi = b.load(pt, &[ai])?;
                            let want = b.addi(first, iv)?;
                            let eq = b.cmpi(ICmp::Eq, pi, want)?;
                            row = b.andi(row, eq)?;
                        }
                        let scattered = b.cmpi(ICmp::Eq, row, no)?;
                        b.when(row, |b| copy(b, first, off, RUN))?;
                        b.when(scattered, |b| {
                            for i in 0..RUN {
                                let iv = b.i32(i);
                                let ai = b.addi(at, iv)?;
                                let pi = b.load(pt, &[ai])?;
                                let di = b.addi(off, iv)?;
                                copy(b, pi, di, 1)?;
                            }
                            Ok(())
                        })
                    })?;
                    b.when(part, |b| {
                        for i in 0..RUN - 1 {
                            let iv = b.i32(i);
                            let has = b.cmpi(ICmp::Slt, iv, left)?;
                            b.when(has, |b| {
                                let ai = b.addi(at, iv)?;
                                let pi = b.load(pt, &[ai])?;
                                let di = b.addi(off, iv)?;
                                copy(b, pi, di, 1)
                            })?;
                        }
                        Ok(())
                    })
                })
            })
        };
        let two = b.i32(2);
        let slot = b.remsi(i, two)?;
        let zero = b.i32(0);
        let first = b.cmpi(ICmp::Eq, i, zero)?;
        b.when(first, |b| {
            let z = b.i32(0);
            copies(b, z, z, false)
        })?;
        let one = b.i32(1);
        let next = b.addi(i, one)?;
        let nsteps = b.i32(steps);
        let more = b.cmpi(ICmp::Slt, next, nsteps)?;
        b.when(more, |b| {
            let one = b.i32(1);
            let other = b.subi(one, slot)?;
            copies(b, next, other, false)
        })?;
        copies(b, i, slot, true)?;

        let (m_sc, l_sc, a_sc) = (blk.scratch[3], blk.scratch[4], blk.scratch[5]);
        let ninf = b.splat_f(&[qh, CHUNK], Elem::F32, f64::NEG_INFINITY)?;
        let sc = b.splat_f(&[qh, CHUNK], Elem::F32, scale)?;
        let zs = b.splat_f(&[qh, CHUNK], Elem::F32, 0.0)?;
        let zo = b.splat_f(&[qh, dv], Elem::F32, 0.0)?;
        let zv = b.splat_f(&[CHUNK, dv], Elem::Bf16, 0.0)?;
        let ninf1 = b.splat_f(&[qh, 1], Elem::F32, f64::NEG_INFINITY)?;
        let z1 = b.splat_f(&[qh, 1], Elem::F32, 0.0)?;
        let o1 = b.splat_f(&[qh, 1], Elem::F32, 1.0)?;
        let gqv = b.i32(gq);
        let gqs = b.broadcast(gqv, &[qh, CHUNK])?;
        let io = b.iota(&[qh, CHUNK], 0);
        let head_s = b.divsi(io, gqs)?;
        let gqo = b.broadcast(gqv, &[qh, dv])?;
        let io = b.iota(&[qh, dv], 0);
        let head_o = b.divsi(io, gqo)?;
        let key_s = b.iota(&[qh, CHUNK], 1);
        let key_v = b.iota(&[CHUNK, dv], 0);
        let (z, gn, step1) = (b.i32(0), b.i32(g), b.i32(1));
        b.for_range(z, gn, step1, |b, r| {
            let gv = b.i32(g);
            let row = b.muli(i, gv)?;
            let row = b.addi(row, r)?;
            let l0 = b.load(lo, &[row])?;
            let l1 = b.load(hi, &[row])?;
            let psv = b.i32(ps);
            let p0 = b.divsi(l0, psv)?;
            let k0 = b.muli(p0, psv)?;
            // key `j` of the buffer is position `k0 + j`
            let lo_k = b.subi(l0, k0)?;
            let hi_k = b.subi(l1, k0)?;
            let hi_k = b.maxsi(hi_k, lo_k)?;
            let qhv = b.i32(qh);
            let at = b.muli(r, qhv)?;
            let at = b.as_index(at)?;
            let i0 = b.index(0);
            let q = b.vload(q_ref, &[at, i0], &[qh, d])?;
            let sidx = b.as_index(slot)?;
            let ridx = b.as_index(r)?;
            b.vstore(ninf1, m_sc, &[i0, i0])?;
            b.vstore(z1, l_sc, &[i0, i0])?;
            b.vstore(zo, a_sc, &[i0, i0])?;
            // the chunks of keys the row's range covers, an online softmax
            let cv = b.i32(CHUNK);
            let c0 = b.divsi(lo_k, cv)?;
            let top = b.i32(CHUNK - 1);
            let c1 = b.addi(hi_k, top)?;
            let c1 = b.divsi(c1, cv)?;
            let one = b.i32(1);
            b.for_range(c0, c1, one, |b, c| {
                let cv = b.i32(CHUNK);
                let base = b.muli(c, cv)?;
                let base = b.assume_multiple(base, CHUNK)?;
                let bidx = b.as_index(base)?;
                let bs = b.broadcast(base, &[qh, CHUNK])?;
                let kj = b.addi(key_s, bs)?;
                let lb = b.broadcast(lo_k, &[qh, CHUNK])?;
                let hb = b.broadcast(hi_k, &[qh, CHUNK])?;
                let ge = b.cmpi(ICmp::Sge, kj, lb)?;
                let lt = b.cmpi(ICmp::Slt, kj, hb)?;
                let valid = b.andi(ge, lt)?;
                let bv = b.broadcast(base, &[CHUNK, dv])?;
                let vj = b.addi(key_v, bv)?;
                let lbv = b.broadcast(lo_k, &[CHUNK, dv])?;
                let hbv = b.broadcast(hi_k, &[CHUNK, dv])?;
                let ge = b.cmpi(ICmp::Sge, vj, lbv)?;
                let lt = b.cmpi(ICmp::Slt, vj, hbv)?;
                let held = b.andi(ge, lt)?;
                let mut s = ninf;
                for kh in 0..kvh {
                    let col = b.index(kh * d);
                    let kb = b.vload(kbuf, &[sidx, ridx, bidx, col], &[1, 1, CHUNK, d])?;
                    let kb = b.shape_cast(kb, &[CHUNK, d])?;
                    let sh = b.matmul(q, kb, zs, true)?;
                    let khv = b.i32(kh);
                    let khb = b.broadcast(khv, &[qh, CHUNK])?;
                    let mine = b.cmpi(ICmp::Eq, head_s, khb)?;
                    s = b.select(mine, sh, s)?;
                }
                let s = b.mulf(s, sc)?;
                let s = b.select(valid, s, ninf)?;
                let m0 = b.vload(m_sc, &[i0, i0], &[qh, 1])?;
                let mx = b.reduce_rows(s, true)?;
                let mn = b.maxf(m0, mx)?;
                let alive = b.gtf(mn, ninf1)?;
                let ms = b.select(alive, mn, z1)?;
                let msb = b.broadcast(ms, &[qh, CHUNK])?;
                let zz = b.subf(s, msb)?;
                let p = b.exp(zz)?;
                let dm = b.subf(m0, ms)?;
                let corr = b.exp(dm)?;
                let l0 = b.vload(l_sc, &[i0, i0], &[qh, 1])?;
                let l = b.mulf(l0, corr)?;
                let ps = b.reduce_rows(p, false)?;
                let l = b.addf(l, ps)?;
                b.vstore(l, l_sc, &[i0, i0])?;
                b.vstore(mn, m_sc, &[i0, i0])?;
                let pb = b.convert_f(p, Elem::Bf16)?;
                let mut o = zo;
                for kh in 0..kvh {
                    let col = b.index(kh * dv);
                    let vb = b.vload(vbuf, &[sidx, ridx, bidx, col], &[1, 1, CHUNK, dv])?;
                    let vb = b.shape_cast(vb, &[CHUNK, dv])?;
                    // a buffer row past the range may hold anything
                    let vb = b.select(held, vb, zv)?;
                    let ov = b.matmul(pb, vb, zo, false)?;
                    let khv = b.i32(kh);
                    let khb = b.broadcast(khv, &[qh, dv])?;
                    let mine = b.cmpi(ICmp::Eq, head_o, khb)?;
                    o = b.select(mine, ov, o)?;
                }
                let a0 = b.vload(a_sc, &[i0, i0], &[qh, dv])?;
                let cb = b.broadcast(corr, &[qh, dv])?;
                let a = b.mulf(a0, cb)?;
                let a = b.addf(a, o)?;
                b.vstore(a, a_sc, &[i0, i0])
            })?;
            let m = b.vload(m_sc, &[i0, i0], &[qh, 1])?;
            let l = b.vload(l_sc, &[i0, i0], &[qh, 1])?;
            let a = b.vload(a_sc, &[i0, i0], &[qh, dv])?;
            let pos = b.gtf(l, z1)?;
            let inv = b.divf(o1, l)?;
            let inv = b.select(pos, inv, z1)?;
            let ib = b.broadcast(inv, &[qh, dv])?;
            let o = b.mulf(a, ib)?;
            b.vstore(o, o_ref, &[at, i0])?;
            b.vstore(m, m_ref, &[at, i0])?;
            b.vstore(l, l_ref, &[at, i0])
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
    let g = Shape::rows_a_step(span, h).min(r.max(1));
    let rp = (r + g - 1) / g * g;
    let s = Shape {
        rp,
        g,
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
    let cut = |f: &mut Func, x: Val, last: &[i64]| -> Built<Val> {
        let x = f.slice_axis(x, 0, 0, r * qh)?;
        let mut dims = vec![r, h.kvh, h.g];
        dims.extend_from_slice(last);
        f.reshape(x, &dims)
    };
    Ok(Flashed {
        o: cut(f, out[0], &[h.dv])?,
        m: cut(f, out[1], &[])?,
        l: cut(f, out[2], &[])?,
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
        let g = Shape::rows_a_step(1024, &h);
        let s = Shape {
            rp: 256,
            g,
            span: 1024,
            ps: 16,
            table: 256 * 64,
            pages: 64,
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
