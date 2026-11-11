//! A prefill's causal (and sliding-window) attention over the keys the fire
//! itself appends, as one Mosaic flash kernel: the rows' own `k`, `v` (as
//! the fire hands them to `kv_append`) are the key rows, so a block of query
//! rows reads its keys as whole row blocks of that rectangle, never through
//! the page table, and the scores, the running max and sum and the
//! accumulator stay in VMEM (the paged walk writes each chunk's f32 scores
//! to HBM and reads them back).
//!
//! Row `r` reads key rows `[klo[r], khi[r])` of the rectangle: its segment
//! (its lane's consecutive positions in this fire) up to itself, cut by the
//! window. Keys a lane holds from before the fire (a cached prefix) are read
//! too, from a rectangle of their pool pages gathered whole (at most
//! `OLD_PAGES`), walked before the fresh blocks. The kernel stands in for
//! the paged walk only when the fire's older keys fit that rectangle and no
//! row carries a custom mask, every row filed in its lane's slot: the fire
//! decides at run time (`stablehlo.case`), the walk being the other branch.
//!
//! Grid: one step per row block, its key blocks walked by a loop with the
//! next block's `k`/`v` copying in (two buffers) while the current one is
//! read. A turn of the block takes one kv head and `gf` of its query heads,
//! stacked along the rows (`[gf · bq, d]`), so one score tile serves them
//! all. The scores are kept transposed, `sᵀ = k · qᵀ` (`[bk, gf · bq]`, keys
//! on the sublanes), as splash attention keeps them: the softmax reduces
//! over the sublanes (plain vector maxima and sums, not cross-lane work), the
//! running max and sum are lane-dense rows, and the accumulator is `vᵀ · pᵀ`
//! (`[dv, gf · bq]`), turned once at the end. Probabilities meet values in
//! bf16, as the walk's do. The turns are unrolled: identical kernels compile
//! once a program (libtpu caches the module), well under a second.

#![allow(clippy::too_many_arguments)]

use crate::error::Error;
use crate::hlo::{Branch, Built, Cmp, Elem, Fold, Func, Malformed, Ty, Val};
use crate::mosaic::{Body, Compiled, Grid, ICmp, Kernel, MTy, Sem, Space, V, Window};
use crate::tensor::{Operand, PageMap};

use super::paged::{Flashed, Geom, Heads, cmp_i, read_rows_i32, round_up, with_i};

/// Keys per block.
const BK: i64 = 256;
/// Query rows per block.
const BQ: i64 = 256;
/// The most stacked rows one score tile holds.
const STACK: i64 = 1024;
/// The most pool pages of older keys (a lane's cached prefix) a fire's rows
/// may read beside their own: whole pages, gathered into one rectangle (a
/// fire that reads more walks the pool). One size, so a layer carries one
/// kernel: every variant is a copy of its code in each program.
const OLD_PAGES: i64 = 16;

fn bad<T>(detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op: "attn.flash",
        detail: detail.into(),
    })
}

/// The paged walk a fire falls back to.
pub(crate) type Walk<'a> = Box<dyn FnOnce(&mut Func) -> Built<Flashed> + 'a>;

/// What the fire appended: `k`, `v` (`[r, kvh · d]` bf16) and each row's
/// write page and offset (`[r]` i32).
#[derive(Clone, Copy, Debug)]
pub struct Appended {
    pub k: Val,
    pub v: Val,
    pub page: Val,
    pub off: Val,
}

impl Appended {
    /// Whether `k`'s rows are as many as the kernel reads.
    #[must_use]
    pub fn worth(k: Operand) -> bool {
        i64::from(k.rows) >= BK
    }

    /// What `kv_append` read: its rows and their slots.
    pub fn of(
        f: &mut Func,
        k: Operand,
        v: Operand,
        page: Operand,
        off: Operand,
    ) -> Result<Self, Error> {
        let page = read_rows_i32(f, "attn.kv_append", page, k.rows)?;
        let off = read_rows_i32(f, "attn.kv_append", off, k.rows)?;
        Ok(Self {
            k: k.val,
            v: v.val,
            page,
            off,
        })
    }
}

/// One kernel's static shape.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Key rows (the rectangle's).
    r: i64,
    /// Query rows rounded up to whole row blocks (the grid's reach).
    rp: i64,
    bq: i64,
    bk: i64,
    h: Heads,
    /// Query heads stacked a turn, and turns per kv head.
    gf: i64,
    gc: i64,
    scale: f64,
}

impl Shape {
    fn of(r: i64, h: &Heads, scale: f64, bq: i64) -> Self {
        let mut gf = h.g;
        while gf > 1 && (gf * bq > STACK || h.g % gf != 0) {
            gf -= 1;
        }
        Self {
            r,
            rp: round_up(r, bq),
            bq,
            bk: BK,
            h: *h,
            gf,
            gc: h.g / gf,
            scale,
        }
    }

    const fn turns(&self) -> i64 {
        self.h.kvh * self.gc
    }

    const fn n(&self) -> i64 {
        self.gf * self.bq
    }

    /// The VMEM the kernel holds: two buffers of each block, the scratch,
    /// and the score tile's temporaries.
    fn vmem(&self) -> i64 {
        let Shape { bq, bk, h, .. } = *self;
        let (qh, n, t) = (h.qh(), self.n(), self.turns());
        let blocks =
            bq * qh * h.d * 2 + bq * qh * h.dv * 2 + 2 * 8 * bq * 4 + 2 * round_up(qh, 8) * bq * 4;
        let bufs = 2 * bk * h.kvh * (h.d + h.dv) * 2;
        let scratch = bq * bk * 4 + t * (2 * 8 * n + h.dv * n) * 4;
        let temps = 6 * n * bk * 4 + 3 * n * h.dv * 4 + n * h.d * 2;
        2 * blocks + bufs + scratch + temps
    }
}

/// The shape the kernel takes for `r` rows of these heads within
/// `vmem_cap`, if any: heads a whole number of lane tiles wide, the
/// rectangle at least a key block tall and tile-aligned at its end.
fn shape(r: i64, h: &Heads, scale: f64, vmem_cap: i64) -> Option<Shape> {
    if h.d % 128 != 0 || h.dv % 128 != 0 || r < BK || r % 16 != 0 || h.g < 1 {
        return None;
    }
    // Whole row blocks only: a partial last block's copies reach past the
    // fire's rows (gemma-4-31B's 640-row fires faulted the chip).
    [BQ, BQ / 2]
        .into_iter()
        .filter(|&bq| r % bq == 0)
        .map(|bq| Shape::of(r, h, scale, bq))
        .find(|s| s.vmem() <= vmem_cap)
}

/// Whether the kernel takes this fire: the static side of the decision
/// (`PIE_XLA_FLASH_PREFILL=0` turns it off).
pub(crate) fn fits(f: &Func, ap: &Appended, r: i64, h: &Heads, vmem_cap: i64) -> bool {
    if std::env::var("PIE_XLA_FLASH_PREFILL").is_ok_and(|v| v == "0") {
        return false;
    }
    let kd = f.dims(ap.k);
    let vd = f.dims(ap.v);
    kd.len() == 2
        && kd[0] == r
        && kd[1] == h.kvh * h.d
        && vd == [r, h.kvh * h.dv]
        && f.elem(ap.k) == Elem::Bf16
        && f.elem(ap.v) == Elem::Bf16
        && f.dims(ap.page) == [r]
        && f.dims(ap.off) == [r]
        && shape(r, h, 1.0, vmem_cap).is_some()
}

/// The kernel over a fire's own key rows and an older-keys rectangle of
/// `mo` rows (operands 5 and 6 its keys and values, 7 and 8 each row's range
/// in it, prefetched 2 and 3 each row block's blocks of it): one walk over
/// the older blocks and then the fresh ones, one copy of the arithmetic.
fn kernel(s: &Shape, vmem_cap: i64, mo: i64) -> Built<Compiled> {
    let Shape {
        r,
        rp,
        bq,
        bk,
        h,
        gf,
        gc,
        scale,
    } = *s;
    let (qh, kvh, g, d, dv) = (h.qh(), h.kvh, h.g, h.d, h.dv);
    let (wk, wv) = (kvh * d, kvh * dv);
    let nb = rp / bq;
    let n = s.n();
    let turns = s.turns();
    let rows = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let zero = b.i32(0);
        Ok(vec![gr.ids[0], zero])
    };
    let cols = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let zero = b.i32(0);
        Ok(vec![zero, gr.ids[0]])
    };
    let blocked = |array: Vec<i64>, elem: Elem, block: Vec<i64>, index: crate::mosaic::IndexMap| {
        crate::mosaic::Operand {
            array,
            elem,
            block,
            window: Window::Blocked,
            index,
        }
    };
    let any = |array: Vec<i64>| crate::mosaic::Operand {
        array,
        elem: Elem::Bf16,
        block: vec![],
        window: Window::Any,
        index: Box::new(|_: &mut Body, _: &Grid| Ok(vec![])),
    };
    let k = Kernel {
        name: "flash_prefill".into(),
        grid: vec![nb],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![nb], vec![nb], vec![nb], vec![nb]],
        ins: vec![
            blocked(
                vec![r, qh * d],
                Elem::Bf16,
                vec![bq, qh * d],
                Box::new(rows),
            ),
            any(vec![r, wk]),
            any(vec![r, wv]),
            blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
            blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
            any(vec![mo, wk]),
            any(vec![mo, wv]),
            blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
            blocked(vec![1, r], Elem::I32, vec![1, bq], Box::new(cols)),
        ],
        outs: vec![
            blocked(
                vec![r, qh * dv],
                Elem::Bf16,
                vec![bq, qh * dv],
                Box::new(rows),
            ),
            blocked(vec![qh, r], Elem::F32, vec![qh, bq], Box::new(cols)),
            blocked(vec![qh, r], Elem::F32, vec![qh, bq], Box::new(cols)),
        ],
        vmem_limit: Some(u64::try_from((s.vmem() * 3 / 2 + (8 << 20)).min(vmem_cap)).unwrap_or(0)),
        scratch: vec![
            MTy::Ref(vec![2, bk, wk], Elem::Bf16, Space::Vmem),
            MTy::Ref(vec![2, bk, wv], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![4]),
            MTy::Ref(vec![bk, bq], Elem::F32, Space::Vmem),
            MTy::Ref(vec![turns, 1, n], Elem::F32, Space::Vmem),
            MTy::Ref(vec![turns, 1, n], Elem::F32, Space::Vmem),
            MTy::Ref(vec![turns, dv, n], Elem::F32, Space::Vmem),
        ],
    };
    k.build(|b, blk| {
        let i = blk.grid.ids[0];
        let q_ref = blk.ins[0];
        let (fresh_k, fresh_v, klo_ref, khi_ref) = (blk.ins[1], blk.ins[2], blk.ins[3], blk.ins[4]);
        let (old_k, old_v, okl_ref, okh_ref) = (blk.ins[5], blk.ins[6], blk.ins[7], blk.ins[8]);
        let (o_ref, mo_ref, lo_ref) = (blk.outs[0], blk.outs[1], blk.outs[2]);
        let (kbuf, vbuf, sems, bias_ref, m_s, l_s, acc_s) = (
            blk.scratch[0],
            blk.scratch[1],
            blk.scratch[2],
            blk.scratch[3],
            blk.scratch[4],
            blk.scratch[5],
            blk.scratch[6],
        );
        let pre = &blk.grid.prefetch;
        let lo = b.load(pre[0], &[i])?;
        let hi = b.load(pre[1], &[i])?;
        let olo = b.load(pre[2], &[i])?;
        let ohi = b.load(pre[3], &[i])?;
        let i0 = b.index(0);
        let zero = b.i32(0);
        let one = b.i32(1);

        // Every turn's state starts empty.
        for t in 0..turns {
            let at = b.index(t);
            let ninf = b.splat_f(&[1, 1, n], Elem::F32, f64::NEG_INFINITY)?;
            b.vstore(ninf, m_s, &[at, i0, i0])?;
            let z = b.splat_f(&[1, 1, n], Elem::F32, 0.0)?;
            b.vstore(z, l_s, &[at, i0, i0])?;
            let z = b.splat_f(&[1, dv, n], Elem::F32, 0.0)?;
            b.vstore(z, acc_s, &[at, i0, i0])?;
        }

        // Step `t` reads older block `olo + t` while `t` is below their
        // count, then fresh block `lo + t - count`.
        let count = |b: &mut Body, lo: V, hi: V| -> Built<V> {
            let c = b.subi(hi, lo)?;
            let c = b.addi(c, one)?;
            b.maxsi(c, zero)
        };
        let nold = count(b, olo, ohi)?;
        let nfresh = count(b, lo, hi)?;
        let steps = b.addi(nold, nfresh)?;
        let older = |b: &mut Body, t: V| -> Built<V> { b.cmpi(ICmp::Slt, t, nold) };
        let block_of = |b: &mut Body, t: V| -> Built<V> {
            let o = b.addi(olo, t)?;
            let f0 = b.subi(t, nold)?;
            let f0 = b.addi(lo, f0)?;
            let is_old = b.cmpi(ICmp::Slt, t, nold)?;
            b.select(is_old, o, f0)
        };
        // A block's first row in its rectangle: a fresh block clamped so it
        // stays inside the fire's rows (the mask skips the rows a clamped
        // block repeats); the older rectangle is whole blocks.
        let block_row = |b: &mut Body, t: V| -> Built<V> {
            let j = block_of(b, t)?;
            let bkv = b.i32(bk);
            let at = b.muli(j, bkv)?;
            let top = b.i32(r - bk);
            let clamped = b.minsi(at, top)?;
            let is_old = b.cmpi(ICmp::Slt, t, nold)?;
            b.select(is_old, at, clamped)
        };
        // Starts (`wait` false) or waits for the copies of step `t` into
        // buffer `slot`, from whichever rectangle the step reads.
        let copies = |b: &mut Body, t: V, slot: V, wait: bool| -> Built<()> {
            let kb = block_row(b, t)?;
            let is_old = older(b, t)?;
            let is_fresh = b.cmpi(ICmp::Sge, t, nold)?;
            for (cond, src_k, src_v) in [(is_old, old_k, old_v), (is_fresh, fresh_k, fresh_v)] {
                b.when(cond, |b| {
                    let z = b.i32(0);
                    let two = b.i32(2);
                    let vslot = b.addi(slot, two)?;
                    for (src, buf, w, sidx) in [(src_k, kbuf, wk, slot), (src_v, vbuf, wv, vslot)] {
                        let s = b.ref_slice(src, &[kb, z], &[bk, w])?;
                        let d = b.ref_slice(buf, &[slot, z, z], &[1, bk, w])?;
                        let d = b.ref_squeeze(d)?;
                        let sem = b.ref_slice(sems, &[sidx], &[1])?;
                        let sem = b.ref_squeeze(sem)?;
                        if wait {
                            b.dma_wait(sem, s, d)?;
                        } else {
                            b.dma_start(s, d, sem)?;
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(())
        };
        let any_steps = b.cmpi(ICmp::Sgt, steps, zero)?;
        b.when(any_steps, |b| copies(b, zero, zero, false))?;
        b.for_range(zero, steps, one, |b, t| {
            let two = b.i32(2);
            let slot = b.remsi(t, two)?;
            let other = b.subi(one, slot)?;
            let next = b.addi(t, one)?;
            let more = b.cmpi(ICmp::Slt, next, steps)?;
            b.when(more, |b| copies(b, next, other, false))?;
            copies(b, t, slot, true)?;

            // The block's admission, a bias of 0 or -inf per (key, row): a
            // row admits the key rows of its range in the block's
            // rectangle, from the block's own start.
            let kb = block_row(b, t)?;
            let j = block_of(b, t)?;
            let is_old = older(b, t)?;
            let sq = [bk, bq];
            let kr = b.broadcast_i(kb, &sq)?;
            let io = b.iota(&sq, 0);
            let kr = b.addi(kr, io)?;
            let bkv = b.i32(bk);
            let start = b.muli(j, bkv)?;
            let start = b.broadcast_i(start, &sq)?;
            let flo = b.vload_all(klo_ref)?;
            let olo_v = b.vload_all(okl_ref)?;
            let klo = b.select(is_old, olo_v, flo)?;
            let klo = b.broadcast(klo, &sq)?;
            let klo = b.maxsi(klo, start)?;
            let fhi = b.vload_all(khi_ref)?;
            let ohi_v = b.vload_all(okh_ref)?;
            let khi = b.select(is_old, ohi_v, fhi)?;
            let khi = b.broadcast(khi, &sq)?;
            let ge = b.cmpi(ICmp::Sge, kr, klo)?;
            let lt = b.cmpi(ICmp::Slt, kr, khi)?;
            let ok = b.andi(ge, lt)?;
            let zs = b.splat_f(&sq, Elem::F32, 0.0)?;
            let ninf = b.splat_f(&sq, Elem::F32, f64::NEG_INFINITY)?;
            let bias = b.select(ok, zs, ninf)?;
            b.vstore_all(bias, bias_ref)?;

            let z = b.i32(0);
            let kslab = b.ref_slice(kbuf, &[slot, z, z], &[1, bk, wk])?;
            let kslab = b.ref_squeeze(kslab)?;
            let vslab = b.ref_slice(vbuf, &[slot, z, z], &[1, bk, wv])?;
            let vslab = b.ref_squeeze(vslab)?;
            for t2 in 0..turns {
                let (kh, gi) = (t2 / gc, t2 % gc);
                let col = b.index(kh * d);
                let kv = b.vload(kslab, &[i0, col], &[bk, d])?;
                let col = b.index(kh * dv);
                let vv = b.vload(vslab, &[i0, col], &[bk, dv])?;
                // The turn's `gf` query heads, stacked along the rows.
                let mut parts = Vec::with_capacity(gf as usize);
                for x in 0..gf {
                    let qi = kh * g + gi * gf + x;
                    let col = b.index(qi * d);
                    parts.push(b.vload(q_ref, &[i0, col], &[bq, d])?);
                }
                let qf = if gf == 1 {
                    parts[0]
                } else {
                    b.concat(&parts, 0)?
                };
                let zs = b.splat_f(&[bk, n], Elem::F32, 0.0)?;
                let st = b.matmul(kv, qf, zs, true)?;
                let st = if scale == 1.0 {
                    st
                } else {
                    let sc = b.splat_f(&[bk, n], Elem::F32, scale)?;
                    b.mulf(st, sc)?
                };
                let bias = b.vload_all(bias_ref)?;
                let bias = if gf == 1 {
                    bias
                } else {
                    let copies = vec![bias; gf as usize];
                    b.concat(&copies, 1)?
                };
                let st = b.addf(st, bias)?;
                let at = b.index(t2);
                let m0 = b.vload(m_s, &[at, i0, i0], &[1, 1, n])?;
                let m0 = b.shape_cast(m0, &[1, n])?;
                let l0 = b.vload(l_s, &[at, i0, i0], &[1, 1, n])?;
                let l0 = b.shape_cast(l0, &[1, n])?;
                let mc = b.reduce_cols(st, true)?;
                let mn = b.maxf(m0, mc)?;
                let ninf1 = b.splat_f(&[1, n], Elem::F32, f64::NEG_INFINITY)?;
                let alive = b.cmpf(true, mn, ninf1)?;
                let z1 = b.splat_f(&[1, n], Elem::F32, 0.0)?;
                let ms = b.select(alive, mn, z1)?;
                let msb = b.broadcast(ms, &[bk, n])?;
                let e = b.subf(st, msb)?;
                let p = b.exp(e)?;
                let ps = b.reduce_cols(p, false)?;
                let dm = b.subf(m0, ms)?;
                let al = b.exp(dm)?;
                let l = b.mulf(l0, al)?;
                let l = b.addf(l, ps)?;
                let mn3 = b.shape_cast(mn, &[1, 1, n])?;
                b.vstore(mn3, m_s, &[at, i0, i0])?;
                let l3 = b.shape_cast(l, &[1, 1, n])?;
                b.vstore(l3, l_s, &[at, i0, i0])?;
                let a0 = b.vload(acc_s, &[at, i0, i0], &[1, dv, n])?;
                let a0 = b.shape_cast(a0, &[dv, n])?;
                let ab = b.broadcast(al, &[dv, n])?;
                let a = b.mulf(a0, ab)?;
                let pb = b.convert_f(p, Elem::Bf16)?;
                let zo = b.splat_f(&[dv, n], Elem::F32, 0.0)?;
                let pv = b.matmul_tl(vv, pb, zo)?;
                let a = b.addf(a, pv)?;
                let a3 = b.shape_cast(a, &[1, dv, n])?;
                b.vstore(a3, acc_s, &[at, i0, i0])?;
            }
            Ok(())
        })?;

        // Each head's rows normalized and turned into their columns; the
        // max and sum a row per head.
        let mut mrows: Vec<(i64, V)> = Vec::with_capacity(qh as usize);
        let mut lrows: Vec<(i64, V)> = Vec::with_capacity(qh as usize);
        for t2 in 0..turns {
            let at = b.index(t2);
            let l = b.vload(l_s, &[at, i0, i0], &[1, 1, n])?;
            let l = b.shape_cast(l, &[1, n])?;
            let m = b.vload(m_s, &[at, i0, i0], &[1, 1, n])?;
            let m = b.shape_cast(m, &[1, n])?;
            let z1 = b.splat_f(&[1, n], Elem::F32, 0.0)?;
            let one1 = b.splat_f(&[1, n], Elem::F32, 1.0)?;
            let pos = b.cmpf(true, l, z1)?;
            let inv = b.divf(one1, l)?;
            let inv = b.select(pos, inv, z1)?;
            let a = b.vload(acc_s, &[at, i0, i0], &[1, dv, n])?;
            let a = b.shape_cast(a, &[dv, n])?;
            let ib = b.broadcast(inv, &[dv, n])?;
            let ot = b.mulf(a, ib)?;
            let o = b.transpose(ot)?;
            let o = b.convert_f(o, Elem::Bf16)?;
            let (kh, gi) = (t2 / gc, t2 % gc);
            for x in 0..gf {
                let qi = kh * g + gi * gf + x;
                let og = b.rows(o, x * bq, bq)?;
                let col = b.index(qi * dv);
                b.vstore(og, o_ref, &[i0, col])?;
                mrows.push((qi, b.slice(m, &[0, x * bq], &[1, bq])?));
                lrows.push((qi, b.slice(l, &[0, x * bq], &[1, bq])?));
            }
        }
        mrows.sort_by_key(|x| x.0);
        lrows.sort_by_key(|x| x.0);
        let ms: Vec<V> = mrows.iter().map(|x| x.1).collect();
        let ls: Vec<V> = lrows.iter().map(|x| x.1).collect();
        let mo = b.concat(&ms, 0)?;
        b.vstore_all(mo, mo_ref)?;
        let lsum = b.concat(&ls, 0)?;
        b.vstore_all(lsum, lo_ref)
    })
}

/// Each row's segment: the first row of the run of rows that are one lane's
/// consecutive positions, as `(start [r], p0 [r])`, the run's first row and
/// that row's position. A padding row (position below zero) stands alone.
fn segments(f: &mut Func, lane: Val, qpos: Val) -> Built<(Val, Val)> {
    let r = f.dims(lane)[0];
    let io = f.iota(Elem::I32, &[r], 0);
    let prev = |f: &mut Func, x: Val, fill: i64| -> Built<Val> {
        let head = f.slice_axis(x, 0, 0, r - 1)?;
        let c = f.const_i(Elem::I32, fill, &[]);
        f.pad(head, c, &[1], &[0], &[0])
    };
    let pl = prev(f, lane, -1)?;
    let pp = prev(f, qpos, -2)?;
    let pp = with_i(f, pp, 1, Func::add)?;
    let same = f.compare(Cmp::Eq, lane, pl)?;
    let next = f.compare(Cmp::Eq, qpos, pp)?;
    let same = f.and(same, next)?;
    let live = cmp_i(f, Cmp::Ge, qpos, 0)?;
    let same = f.and(same, live)?;
    let z = f.like_i(io, 0);
    let cand = f.select(same, z, io)?;
    let start = f.scan_steps(cand, 0, Fold::Max)?;
    let back = f.sub(io, start)?;
    let p0 = f.sub(qpos, back)?;
    Ok((start, p0))
}

/// The attention of `q4` (`[r, kvh, g, d]` bf16) rows over the fire's
/// appended keys where that reads the same keys the paged walk would, else
/// `walk`: `lo`/`hi` bound each row's admitted positions, `lane` its lane
/// (clamped), `cap` the lane's capacity, `enabled` its custom-mask flag,
/// `slots` the pool plane's rows. `o` is bf16.
pub(crate) fn attend(
    f: &mut Func,
    ap: &Appended,
    h: &Heads,
    geom: &Geom,
    slots: i64,
    planes: (Val, Val),
    q4: Val,
    lane: Val,
    qpos: Val,
    lo: Val,
    hi: Val,
    window: i64,
    cap: Val,
    enabled: Option<Val>,
    scale: f64,
    vmem_cap: i64,
    walk: Walk<'_>,
    maps: [PageMap; 2],
) -> Built<Flashed> {
    let r = f.dims(q4)[0];
    let Some(s) = shape(r, h, scale, vmem_cap) else {
        return bad(format!("{r} rows of {h:?} do not fit"));
    };
    let (start, p0) = segments(f, lane, qpos)?;
    let live = cmp_i(f, Cmp::Ge, qpos, 0)?;
    let old = older(
        f, geom, slots, planes, lane, qpos, lo, p0, start, live, window, maps,
    )?;
    // Filed in its lane's slot (not the sink, the plane's last page).
    let ps = geom.ps;
    let pix = with_i(f, qpos, ps, Func::div)?;
    let at = with_i(f, lane, geom.pages, Func::mul)?;
    let at = f.add(at, pix)?;
    let page = super::paged::take(f, geom.pt, at)?;
    let want = with_i(f, page, ps, Func::mul)?;
    let off = with_i(f, qpos, ps, Func::rem)?;
    let want = f.add(want, off)?;
    let got = with_i(f, ap.page, ps, Func::mul)?;
    let got = f.add(got, ap.off)?;
    let filed = f.compare(Cmp::Eq, got, want)?;
    let inside = cmp_i(f, Cmp::Ge, ap.page, 0)?;
    // the sink is the keys plane's last page, in its own pages
    let plane_page = maps[0].apply(f, ap.page)?;
    let below = cmp_i(f, Cmp::Lt, plane_page, slots / ps - 1)?;
    let filed = f.and(filed, inside)?;
    let filed = f.and(filed, below)?;
    let upto = with_i(f, qpos, 1, Func::add)?;
    let held = f.compare(Cmp::Le, upto, cap)?;
    let mut ok = f.and(filed, held)?;
    if let Some(en) = enabled {
        let off = cmp_i(f, Cmp::Eq, en, 0)?;
        ok = f.and(ok, off)?;
    }
    let dead = f.not(live);
    let ok = f.or(ok, dead)?;
    let all = f.reduce(ok, &[0], Fold::And)?;
    // 0: the walk; 1: the kernel, the older keys a rectangle holds (none
    // where every row's keys are fresh).
    let fits_old = cmp_i(f, Cmp::Le, old.total, OLD_PAGES)?;
    let all = f.and(all, fits_old)?;
    let which = f.convert(all, Elem::I32);

    // Each row's key rows `[klo, khi)` of the rectangle: its segment's,
    // from its window's start (the older keys are the other rectangle's).
    let klo = f.max(lo, p0)?;
    let klo = f.sub(klo, p0)?;
    let klo = f.add(klo, start)?;
    let khi = f.sub(hi, p0)?;
    let khi = f.add(khi, start)?;
    let dead_lo = f.like_i(klo, 0);
    let klo = f.select(live, klo, dead_lo)?;
    let khi = f.select(live, khi, dead_lo)?;

    let outs = vec![
        Ty::new(Elem::Bf16, &[r, h.kvh, h.g, h.dv]),
        Ty::new(Elem::F32, &[r, h.kvh, h.g]),
        Ty::new(Elem::F32, &[r, h.kvh, h.g]),
    ];
    let walked: Branch<'_> = Box::new(move |f: &mut Func| {
        let fl = walk(f)?;
        let o = f.convert(fl.o, Elem::Bf16);
        Ok(vec![o, fl.m, fl.l])
    });
    let flashed: Branch<'_> = Box::new(move |f: &mut Func| {
        let (o, m, l) = run(f, &s, ap, q4, klo, khi, vmem_cap, &old)?;
        Ok(vec![o, m, l])
    });
    let out = f.case(which, &outs, vec![walked, flashed])?;
    Ok(Flashed {
        o: out[0],
        m: out[1],
        l: out[2],
    })
}

/// The older keys a fire's rows read (their lanes' cached prefixes): the
/// pool pages that hold them, laid one after another as a rectangle of
/// `OLD_PAGES` whole pages, and each row's range `[klo, khi)` in it.
#[derive(Clone, Copy)]
struct Older {
    /// The keys plane's pages, `[OLD_PAGES]` (the sink page past the ones
    /// read).
    pages: Val,
    /// The values plane's (its own `PageMap`; `pages` when the maps agree).
    values_pages: Val,
    /// How many pages the rows read (i32 scalar).
    total: Val,
    klo: Val,
    khi: Val,
    planes: (Val, Val),
    ps: i64,
    /// The pages unmapped (`blocks::SINK` where the sink is read), and each
    /// plane's map and sink: on a pool the DMA gather maps them itself.
    raw: Val,
    maps: [PageMap; 2],
    sink: i64,
}

impl Older {
    /// The rectangle's keys and values, `[OLD_PAGES · ps, w]` each.
    fn gather(&self, f: &mut Func) -> Built<(Val, Val)> {
        let n = OLD_PAGES;
        // Both planes of the chunk pool by DMA, each plane's map applied in
        // the kernel (the unmapped ids one computation a fire), only with
        // `PIE_XLA_OLDER_DMA=1`: the kernel call falls between the queries'
        // projection and the flash, and XLA evicts the queries from fast
        // memory around it (v6e qwen3-0.6b, 2048-token prefill: a 5.6 us
        // copy and a flash 7 us slower a layer, against the XLA gather's
        // 7 us). The XLA gather reads the pool's pages as blocks.
        if let (Some((pool, Elem::Bf16)), Some((pool_v, Elem::Bf16))) =
            (f.view_parts(self.planes.0), f.view_parts(self.planes.1))
            && pool == pool_v
            && crate::blocks::on()
            && std::env::var("PIE_XLA_OLDER_DMA").is_ok_and(|v| v == "1")
        {
            let mut planes = Vec::with_capacity(2);
            for (plane, map) in [(self.planes.0, self.maps[0]), (self.planes.1, self.maps[1])] {
                let (slots, w) = (f.dims(plane)[0], f.dims(plane)[1]);
                let bytes = self.ps * w * 2;
                if bytes % 512 == 0 && (bytes / 256) % 16 == 0 {
                    planes.push((bytes / 512, map, self.sink, slots / self.ps - 1));
                }
            }
            let mut out = Vec::with_capacity(2);
            if planes.len() == 2 {
                // both planes in one kernel call
                let got =
                    crate::blocks::gather_paged_planes(f, pool, self.raw, Elem::Bf16, &planes)?;
                for (g, plane) in got.into_iter().zip([self.planes.0, self.planes.1]) {
                    let w = f.dims(plane)[1];
                    out.push(f.reshape(g, &[n * self.ps, w])?);
                }
            }
            if out.len() == 2 {
                return Ok((out[0], out[1]));
            }
        }
        let at = f.reshape(self.pages, &[n, 1])?;
        let dims = crate::hlo::GatherDims {
            offset_dims: vec![1, 2],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            ..crate::hlo::GatherDims::default()
        };
        let at_v = if self.values_pages == self.pages {
            at
        } else {
            f.reshape(self.values_pages, &[n, 1])?
        };
        let one = |f: &mut Func, plane: Val, at: Val| -> Built<Val> {
            let (slots, w) = (f.dims(plane)[0], f.dims(plane)[1]);
            let paged = f.reshape(plane, &[slots / self.ps, self.ps, w])?;
            let g = f.gather(paged, at, &dims, &[1, self.ps, w])?;
            f.reshape(g, &[n * self.ps, w])
        };
        f.with_xla_reads(|f| Ok((one(f, self.planes.0, at)?, one(f, self.planes.1, at_v)?)))
    }
}

/// Each segment's older keys: positions `[lo, p0)` of its lane (`lo` at its
/// first row; a window's rows start later, never earlier), read through the
/// lane's page table as whole pages.
fn older(
    f: &mut Func,
    geom: &Geom,
    slots: i64,
    planes: (Val, Val),
    lane: Val,
    qpos: Val,
    lo: Val,
    p0: Val,
    start: Val,
    live: Val,
    window: i64,
    maps: [PageMap; 2],
) -> Built<Older> {
    let r = f.dims(qpos)[0];
    let ps = geom.ps;
    // `lo` at the segment's first row, from its position.
    let lo0 = if window > 0 {
        let s = with_i(f, p0, window - 1, Func::sub)?;
        let z = f.like_i(s, 0);
        f.max(s, z)?
    } else {
        f.like_i(p0, 0)
    };
    let first = with_i(f, lo0, ps, Func::div)?;
    let end = with_i(f, p0, ps - 1, Func::add)?;
    let end = with_i(f, end, ps, Func::div)?;
    let has = f.compare(Cmp::Lt, lo0, p0)?;
    let has = f.and(has, live)?;
    let span = f.sub(end, first)?;
    let none = f.like_i(span, 0);
    let np = f.select(has, span, none)?;
    // A page count at each segment's first row, summed in row order: a
    // segment's pages start where the ones before it end.
    let io = f.iota(Elem::I32, &[r], 0);
    let head = f.compare(Cmp::Eq, start, io)?;
    let at_head = f.select(head, np, none)?;
    let cum = f.scan_steps(at_head, 0, Fold::Sum)?;
    let total = f.reduce(at_head, &[0], Fold::Sum)?;
    let base = f.sub(cum, np)?;
    // Each row's range in the rectangle.
    let row0 = with_i(f, first, ps, Func::mul)?;
    let b0 = with_i(f, base, ps, Func::mul)?;
    let a = f.sub(lo, row0)?;
    let klo = f.add(b0, a)?;
    let c = f.sub(p0, row0)?;
    let khi = f.add(b0, c)?;
    let reads = f.compare(Cmp::Lt, lo, p0)?;
    let reads = f.and(reads, live)?;
    let klo = f.select(reads, klo, none)?;
    let khi = f.select(reads, khi, none)?;
    // Page `j` of the rectangle: the segment whose run holds it.
    let grid = [OLD_PAGES, r];
    let j = f.iota(Elem::I32, &grid, 0);
    let bb = f.broadcast(base, &grid, &[1])?;
    let eb = f.broadcast(cum, &grid, &[1])?;
    let hb = f.broadcast(head, &grid, &[1])?;
    let ge = f.compare(Cmp::Ge, j, bb)?;
    let lt = f.compare(Cmp::Lt, j, eb)?;
    let mine = f.and(ge, lt)?;
    let mine = f.and(mine, hb)?;
    let pick = |f: &mut Func, v: Val| -> Built<Val> {
        let vb = f.broadcast(v, &grid, &[1])?;
        let m1 = f.like_i(vb, -1);
        let x = f.select(mine, vb, m1)?;
        f.reduce(x, &[1], Fold::Max)
    };
    let owner_lane = pick(f, lane)?;
    let owner_first = pick(f, first)?;
    let owner_base = pick(f, base)?;
    let jj = f.iota(Elem::I32, &[OLD_PAGES], 0);
    let off = f.sub(jj, owner_base)?;
    let pix = f.add(owner_first, off)?;
    let pix = super::paged::clamp_i(f, pix, 0, geom.pages - 1)?;
    let ol = super::paged::clamp_i(f, owner_lane, 0, geom.lanes - 1)?;
    let at = with_i(f, ol, geom.pages, Func::mul)?;
    let at = f.add(at, pix)?;
    let raw = super::paged::take(f, geom.pt, at)?;
    let used = cmp_i(f, Cmp::Ge, owner_lane, 0)?;
    let sink = f.like_i(raw, slots / ps - 1);
    // each plane's own pages (the sink is either plane's last page)
    let plane_pages = |f: &mut Func, map: PageMap| -> Built<Val> {
        let page = map.apply(f, raw)?;
        let page = f.select(used, page, sink)?;
        super::paged::clamp_i(f, page, 0, slots / ps - 1)
    };
    let page = plane_pages(f, maps[0])?;
    let values_pages = if maps[1] == maps[0] {
        page
    } else {
        plane_pages(f, maps[1])?
    };
    let sentinel = f.like_i(raw, i64::from(crate::blocks::SINK));
    let raw = f.select(used, raw, sentinel)?;
    Ok(Older {
        pages: page,
        values_pages,
        total,
        klo,
        khi,
        planes,
        ps,
        raw,
        maps,
        sink: slots / ps - 1,
    })
}

/// The kernel over `q4`, each row reading key rows `[klo, khi)`: `(o [r,
/// kvh, g, dv] bf16, m [r, kvh, g], l [r, kvh, g])`.
fn run(
    f: &mut Func,
    s: &Shape,
    ap: &Appended,
    q4: Val,
    klo: Val,
    khi: Val,
    vmem_cap: i64,
    old: &Older,
) -> Built<(Val, Val, Val)> {
    let Shape {
        r, rp, bq, bk, h, ..
    } = *s;
    let nb = rp / bq;
    let q2 = f.reshape(q4, &[r, h.qh() * h.d])?;
    // A power-of-two scale (1/16 at d = 256) turns `q` exactly, before the
    // kernel: its score tiles then take no multiply.
    let unit = Shape { scale: 1.0, ..*s };
    let (q2, s) = if s.scale != 1.0 && s.scale.log2().fract() == 0.0 {
        (f.scale(q2, s.scale)?, &unit)
    } else {
        (q2, s)
    };
    let pad_rows = |f: &mut Func, x: Val, fill: i64| -> Built<Val> {
        if rp == r {
            return Ok(x);
        }
        let c = if f.elem(x).is_float() {
            f.const_f(f.elem(x), fill as f64, &[])
        } else {
            f.const_i(f.elem(x), fill, &[])
        };
        let rank = f.dims(x).len();
        let mut hi = vec![0; rank];
        hi[0] = rp - r;
        f.pad(x, c, &vec![0; rank], &hi, &vec![0; rank])
    };
    // Each row block's first and last key block (none: last below first).
    let blocks = |f: &mut Func, klo: Val, khi: Val| -> Built<(Val, Val)> {
        let empty = f.compare(Cmp::Ge, klo, khi)?;
        let far = f.like_i(klo, i64::from(i32::MAX));
        let lo_rows = f.select(empty, far, klo)?;
        let none = f.like_i(khi, 0);
        let hi_rows = f.select(empty, none, khi)?;
        let lo_rows = pad_rows(f, lo_rows, i64::from(i32::MAX))?;
        let hi_rows = pad_rows(f, hi_rows, 0)?;
        let lo_b = f.reshape(lo_rows, &[nb, bq])?;
        let lo_b = f.reduce(lo_b, &[1], Fold::Min)?;
        let hi_b = f.reshape(hi_rows, &[nb, bq])?;
        let hi_b = f.reduce(hi_b, &[1], Fold::Max)?;
        let any = cmp_i(f, Cmp::Gt, hi_b, 0)?;
        let lo_blk = with_i(f, lo_b, bk, Func::div)?;
        let zero = f.like_i(lo_blk, 0);
        let lo_blk = f.select(any, lo_blk, zero)?;
        let hi_blk = with_i(f, hi_b, 1, Func::sub)?;
        let hi_blk = with_i(f, hi_blk, bk, Func::div)?;
        let minus = f.like_i(hi_blk, -1);
        Ok((lo_blk, f.select(any, hi_blk, minus)?))
    };
    let (lo_blk, hi_blk) = blocks(f, klo, khi)?;
    // The rows are whole row blocks (`shape`), so none is padded.
    let klo = f.reshape(klo, &[1, r])?;
    let khi = f.reshape(khi, &[1, r])?;
    let (olo, ohi) = blocks(f, old.klo, old.khi)?;
    let okl = f.reshape(old.klo, &[1, r])?;
    let okh = f.reshape(old.khi, &[1, r])?;
    let (ok, ov) = old.gather(f)?;
    let kern = kernel(s, vmem_cap, OLD_PAGES * old.ps)?;
    let out = kern.call(
        f,
        &[
            lo_blk, hi_blk, olo, ohi, q2, ap.k, ap.v, klo, khi, ok, ov, okl, okh,
        ],
    )?;
    let o = f.reshape(out[0], &[r, h.kvh, h.g, h.dv])?;
    // The max and sum come a row per head.
    let heads = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.transpose(x, &[1, 0])?;
        f.reshape(x, &[r, h.kvh, h.g])
    };
    let m = heads(f, out[1])?;
    let l = heads(f, out[2])?;
    Ok((o, m, l))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kernel_stacks_a_kv_heads_query_heads() {
        let h = Heads {
            kvh: 8,
            g: 2,
            d: 128,
            dv: 128,
        };
        let s = shape(1792, &h, 0.1, 110 << 20).unwrap();
        assert_eq!((s.rp, s.bq, s.bk, s.gf, s.gc), (1792, 256, 256, 2, 1));
        let c = kernel(&s, 110 << 20, 256).unwrap();
        assert!(c.module.contains("iteration_bounds = array<i64: 7>"));
        assert_eq!(
            c.module.matches("tpu.matmul").count(),
            2 * 8,
            "one copy of each turn"
        );
        assert!(c.module.contains("scratch_operands = 7 : i64"));
        assert!(
            c.module.contains("vector<256x512xf32>"),
            "scores keyed on the sublanes"
        );
        assert_eq!(c.operands.len(), 13);
        assert_eq!(c.results.len(), 3);
        if let Some(p) = std::env::var_os("FLASH_DUMP") {
            std::fs::write(p, &c.module).unwrap();
        }
    }

    #[test]
    fn wide_groups_split_into_turns() {
        let h = Heads {
            kvh: 1,
            g: 8,
            d: 256,
            dv: 256,
        };
        let s = shape(2048, &h, 0.1, 110 << 20).unwrap();
        assert_eq!((s.bq, s.gf, s.gc), (256, 4, 2));
        assert!(shape(200, &h, 0.1, 110 << 20).is_none());
        assert!(shape(1000, &h, 0.1, 110 << 20).is_none());
    }
}
