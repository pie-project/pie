//! Causal attention of a prefill's rows over the keys the fire itself
//! writes, as one Mosaic flash kernel: the rows' own `k`, `v` (as the fire
//! appends them) are the key rows, so a block of query rows reads its keys
//! as whole blocks of that rectangle, never through the page table, and the
//! scores, the running max and sum and the accumulator stay in VMEM (the
//! paged walk writes each chunk's f32 scores to HBM and reads them back).
//!
//! Row `r` reads key rows `[first[r], r]`: the rows of its segment (its
//! lane's consecutive positions in this fire) inside its window. The keys
//! before the segment come from the pool through the paged walk, and the two
//! readings fold by their log-sum-exps (`paged::attend`).
//!
//! Grid `(row block, key block)`: a row block walks the key blocks from its
//! first row's segment start to its diagonal, every head in one step (a step
//! per head and block costs more than its work at prefill widths); steps past
//! a block's last key block repeat its index (no DMA) and skip their work.
//! Probabilities meet values in bf16, as the paged walk's do.

#![allow(clippy::too_many_arguments)]

use crate::hlo::{Built, Cmp, Elem, Fold, Func, Malformed, Val};
use crate::mosaic::{Body, Compiled, Grid, ICmp, Kernel, Operand, Sem, V, Window};

use super::paged::{Flashed, Heads, with_i};

/// Query rows per block.
const BR: i64 = 256;
/// Key rows per block.
const BK: i64 = 512;

fn bad<T>(detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op: "attn.fresh",
        detail: detail.into(),
    })
}

/// One kernel's static shape.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Rows, padded to whole key blocks.
    rp: i64,
    br: i64,
    bk: i64,
    h: Heads,
    scale: f64,
}

impl Shape {
    fn of(r: i64, h: &Heads, scale: f64) -> Self {
        let bk = if r <= BR { BR } else { BK };
        Self {
            rp: round_up(r, bk),
            br: BR,
            bk,
            h: *h,
            scale,
        }
    }

    /// The VMEM the kernel holds: two buffers of each block, the scratch,
    /// and a few `[br, bk]` f32 temporaries.
    fn vmem(&self) -> i64 {
        let (br, bk, h) = (self.br, self.bk, self.h);
        let qh = h.qh();
        let blocks = br * qh * h.d * 2
            + bk * h.kvh * (h.d + h.dv) * 2
            + br * 128 * 4
            + br * qh * h.dv * 4
            + br * 128 * 4;
        let scratch = 2 * qh * br * 128 * 4 + qh * br * h.dv * 4;
        2 * blocks + scratch + 8 * br * bk * 4
    }
}

const fn round_up(x: i64, m: i64) -> i64 {
    (x + m - 1) / m * m
}

/// Whether the kernel takes `r` rows of these heads within `vmem_cap`.
pub(crate) fn fits(r: i64, h: &Heads, vmem_cap: i64) -> bool {
    h.d % 128 == 0 && h.dv % 128 == 0 && r >= 2 && Shape::of(r, h, 1.0).vmem() <= vmem_cap
}

fn kernel(s: &Shape, vmem_cap: i64) -> Built<Compiled> {
    let Shape {
        rp,
        br,
        bk,
        h,
        scale,
        ..
    } = *s;
    let (qh, kvh, g, d, dv) = (h.qh(), h.kvh, h.g, h.d, h.dv);
    let nb = rp / br;
    let nj = rp / bk;
    let rows = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let zero = b.i32(0);
        Ok(vec![gr.ids[0], zero])
    };
    let keys = |b: &mut Body, gr: &Grid| -> Built<Vec<V>> {
        let lo = b.load(gr.prefetch[0], &[gr.ids[0]])?;
        let hi = b.load(gr.prefetch[1], &[gr.ids[0]])?;
        let at = b.addi(lo, gr.ids[1])?;
        let at = b.minsi(at, hi)?;
        let zero = b.i32(0);
        Ok(vec![at, zero])
    };
    let operand =
        |array: Vec<i64>, elem: Elem, block: Vec<i64>, index: crate::mosaic::IndexMap| Operand {
            array,
            elem,
            block,
            window: Window::Blocked,
            index,
        };
    let k = Kernel {
        name: "fresh".into(),
        grid: vec![nb, nj],
        semantics: vec![Sem::Parallel, Sem::Arbitrary],
        prefetch: vec![vec![nb], vec![nb]],
        ins: vec![
            operand(
                vec![rp, qh * d],
                Elem::Bf16,
                vec![br, qh * d],
                Box::new(rows),
            ),
            operand(
                vec![rp, kvh * d],
                Elem::Bf16,
                vec![bk, kvh * d],
                Box::new(keys),
            ),
            operand(
                vec![rp, kvh * dv],
                Elem::Bf16,
                vec![bk, kvh * dv],
                Box::new(keys),
            ),
            operand(vec![rp, 1], Elem::I32, vec![br, 1], Box::new(rows)),
        ],
        outs: vec![
            operand(
                vec![rp, qh * dv],
                Elem::F32,
                vec![br, qh * dv],
                Box::new(rows),
            ),
            operand(vec![rp, qh], Elem::F32, vec![br, qh], Box::new(rows)),
        ],
        vmem_limit: Some(u64::try_from(s.vmem().min(vmem_cap)).unwrap_or(0)),
        scratch: vec![
            (vec![qh, br, 1], Elem::F32),
            (vec![qh, br, 1], Elem::F32),
            (vec![qh, br, dv], Elem::F32),
        ],
    };
    k.build(|b, blk| {
        let (i, j) = (blk.grid.ids[0], blk.grid.ids[1]);
        let (q_ref, k_ref, v_ref, first_ref) = (blk.ins[0], blk.ins[1], blk.ins[2], blk.ins[3]);
        let (o_ref, lse_ref) = (blk.outs[0], blk.outs[1]);
        let (m_ref, l_ref, acc_ref) = (blk.scratch[0], blk.scratch[1], blk.scratch[2]);
        let lo = b.load(blk.grid.prefetch[0], &[i])?;
        let hi = b.load(blk.grid.prefetch[1], &[i])?;
        let zero = b.i32(0);
        let start = b.cmpi(ICmp::Eq, j, zero)?;
        b.when(start, |b| {
            let ninf = b.splat_f(&[qh, br, 1], Elem::F32, f64::NEG_INFINITY)?;
            b.vstore_all(ninf, m_ref)?;
            let z = b.splat_f(&[qh, br, 1], Elem::F32, 0.0)?;
            b.vstore_all(z, l_ref)?;
            let z = b.splat_f(&[qh, br, dv], Elem::F32, 0.0)?;
            b.vstore_all(z, acc_ref)
        })?;
        let jj = b.addi(lo, j)?;
        let live = b.cmpi(ICmp::Sle, jj, hi)?;
        b.when(live, |b| {
            let brv = b.i32(br);
            let row0 = b.muli(i, brv)?;
            let io = b.iota(&[br, bk], 0);
            let r0 = b.broadcast(row0, &[br, bk])?;
            let qrow = b.addi(r0, io)?;
            let bkv = b.i32(bk);
            let key0 = b.muli(jj, bkv)?;
            let io = b.iota(&[br, bk], 1);
            let k0 = b.broadcast(key0, &[br, bk])?;
            let krow = b.addi(k0, io)?;
            let causal = b.cmpi(ICmp::Sle, krow, qrow)?;
            let first = b.vload_all(first_ref)?;
            let first = b.broadcast(first, &[br, bk])?;
            let inside = b.cmpi(ICmp::Sge, krow, first)?;
            let ok = b.andi(causal, inside)?;
            let ninf = b.splat_f(&[br, bk], Elem::F32, f64::NEG_INFINITY)?;
            let sc = b.splat_f(&[br, bk], Elem::F32, scale)?;
            let zs = b.splat_f(&[br, bk], Elem::F32, 0.0)?;
            let zo = b.splat_f(&[br, dv], Elem::F32, 0.0)?;
            let ninf1 = b.splat_f(&[br, 1], Elem::F32, f64::NEG_INFINITY)?;
            let z1 = b.splat_f(&[br, 1], Elem::F32, 0.0)?;
            let i0 = b.index(0);
            for kh in 0..kvh {
                let col = b.index(kh * d);
                let kb = b.vload(k_ref, &[i0, col], &[bk, d])?;
                let col = b.index(kh * dv);
                let vb = b.vload(v_ref, &[i0, col], &[bk, dv])?;
                for gi in 0..g {
                    let qi = kh * g + gi;
                    let col = b.index(qi * d);
                    let qb = b.vload(q_ref, &[i0, col], &[br, d])?;
                    let s = b.matmul(qb, kb, zs, true)?;
                    let s = b.mulf(s, sc)?;
                    let s = b.select(ok, s, ninf)?;
                    let at = b.index(qi);
                    let m0 = b.vload(m_ref, &[at, i0, i0], &[1, br, 1])?;
                    let m0 = b.shape_cast(m0, &[br, 1])?;
                    let mx = b.reduce_rows(s, true)?;
                    let mn = b.maxf(m0, mx)?;
                    let alive = b.gtf(mn, ninf1)?;
                    let ms = b.select(alive, mn, z1)?;
                    let msb = b.broadcast(ms, &[br, bk])?;
                    let z = b.subf(s, msb)?;
                    let p = b.exp(z)?;
                    let dm = b.subf(m0, ms)?;
                    let c = b.exp(dm)?;
                    let l0 = b.vload(l_ref, &[at, i0, i0], &[1, br, 1])?;
                    let l0 = b.shape_cast(l0, &[br, 1])?;
                    let l = b.mulf(l0, c)?;
                    let ps = b.reduce_rows(p, false)?;
                    let l = b.addf(l, ps)?;
                    let l = b.shape_cast(l, &[1, br, 1])?;
                    b.vstore(l, l_ref, &[at, i0, i0])?;
                    let a0 = b.vload(acc_ref, &[at, i0, i0], &[1, br, dv])?;
                    let a0 = b.shape_cast(a0, &[br, dv])?;
                    let pb = b.convert_f(p, Elem::Bf16)?;
                    let pv = b.matmul(pb, vb, zo, false)?;
                    let cb = b.broadcast(c, &[br, dv])?;
                    let a = b.mulf(a0, cb)?;
                    let a = b.addf(a, pv)?;
                    let a = b.shape_cast(a, &[1, br, dv])?;
                    b.vstore(a, acc_ref, &[at, i0, i0])?;
                    let mn = b.shape_cast(mn, &[1, br, 1])?;
                    b.vstore(mn, m_ref, &[at, i0, i0])?;
                }
            }
            Ok(())
        })?;
        let lastj = b.i32(nj - 1);
        let last = b.cmpi(ICmp::Eq, j, lastj)?;
        b.when(last, |b| {
            let z1 = b.splat_f(&[br, 1], Elem::F32, 0.0)?;
            let one = b.splat_f(&[br, 1], Elem::F32, 1.0)?;
            let ninf1 = b.splat_f(&[br, 1], Elem::F32, f64::NEG_INFINITY)?;
            let lanes = b.iota(&[br, qh], 1);
            let mut lse = b.splat_f(&[br, qh], Elem::F32, f64::NEG_INFINITY)?;
            let i0 = b.index(0);
            for qi in 0..qh {
                let at = b.index(qi);
                let l = b.vload(l_ref, &[at, i0, i0], &[1, br, 1])?;
                let l = b.shape_cast(l, &[br, 1])?;
                let m = b.vload(m_ref, &[at, i0, i0], &[1, br, 1])?;
                let m = b.shape_cast(m, &[br, 1])?;
                let pos = b.gtf(l, z1)?;
                let inv = b.divf(one, l)?;
                let inv = b.select(pos, inv, z1)?;
                let a = b.vload(acc_ref, &[at, i0, i0], &[1, br, dv])?;
                let a = b.shape_cast(a, &[br, dv])?;
                let ib = b.broadcast(inv, &[br, dv])?;
                let o = b.mulf(a, ib)?;
                let col = b.index(qi * dv);
                b.vstore(o, o_ref, &[i0, col])?;
                let ll = b.log(l)?;
                let x = b.addf(m, ll)?;
                let x = b.select(pos, x, ninf1)?;
                let x = b.broadcast(x, &[br, qh])?;
                let c = b.i32(qi);
                let c = b.broadcast(c, &[br, qh])?;
                let here = b.cmpi(ICmp::Eq, lanes, c)?;
                lse = b.select(here, x, lse)?;
            }
            b.vstore_all(lse, lse_ref)
        })
    })
}

/// The reading of `q4` (`[r, kvh, g, d]` bf16) over the key rows `k`
/// (`[r, kvh·d]`) and `v` (`[r, kvh·dv]`): row `r` reads rows
/// `[first[r], r]`. `(o [r, kvh, g, dv] f32, lse [r, kvh, g] f32)`, the
/// natural log-sum-exp (`-inf` where a row read nothing).
pub(crate) fn attend(
    f: &mut Func,
    q4: Val,
    k: Val,
    v: Val,
    first: Val,
    h: &Heads,
    scale: f64,
    vmem_cap: i64,
) -> Built<(Val, Val)> {
    let r = f.dims(q4)[0];
    if !fits(r, h, vmem_cap) {
        return bad(format!("{r} rows of {h:?} do not fit"));
    }
    let s = Shape::of(r, h, scale);
    let (rp, br, bk) = (s.rp, s.br, s.bk);
    let nb = rp / br;
    let pad = |f: &mut Func, x: Val| -> Built<Val> {
        let x = f.convert(x, Elem::Bf16);
        let w = f.dims(x).iter().skip(1).product::<i64>();
        let x = f.reshape(x, &[r, w])?;
        if rp == r {
            return Ok(x);
        }
        let z = f.const_f(Elem::Bf16, 0.0, &[]);
        f.pad(x, z, &[0, 0], &[rp - r, 0], &[0, 0])
    };
    let q2 = pad(f, q4)?;
    let k2 = pad(f, k)?;
    let v2 = pad(f, v)?;
    if f.dims(k2)[1] != h.kvh * h.d || f.dims(v2)[1] != h.kvh * h.dv {
        return bad("the key rows are not the pool's width");
    }
    // Padding rows read themselves.
    let own = f.iota(Elem::I32, &[rp], 0);
    let first = if rp == r {
        first
    } else {
        let tail = f.slice_axis(own, 0, r, rp)?;
        f.concat(&[first, tail], 0)?
    };
    let fb = f.reshape(first, &[nb, br])?;
    let fb = f.reduce(fb, &[1], Fold::Min)?;
    let blo = with_i(f, fb, bk, Func::div)?;
    let last = f.iota(Elem::I32, &[nb], 0);
    let last = with_i(f, last, br, Func::mul)?;
    let last = with_i(f, last, br - 1, Func::add)?;
    let bhi = with_i(f, last, bk, Func::div)?;
    let first = f.reshape(first, &[rp, 1])?;
    let kern = kernel(&s, vmem_cap)?;
    let out = kern.call(f, &[blo, bhi, q2, k2, v2, first])?;
    let o = f.slice_axis(out[0], 0, 0, r)?;
    let o = f.reshape(o, &[r, h.kvh, h.g, h.dv])?;
    let lse = f.slice_axis(out[1], 0, 0, r)?;
    let lse = f.reshape(lse, &[r, h.kvh, h.g])?;
    Ok((o, lse))
}

/// Folds the fresh reading `(o, lse)` into the walk's `pool` reading.
pub(crate) fn fold(f: &mut Func, pool: Flashed, o: Val, lse: Val) -> Built<Flashed> {
    let zero = f.like_f(pool.l, 0.0);
    let has = f.compare(Cmp::Gt, pool.l, zero)?;
    let ll = f.log(pool.l);
    let lse2 = f.add(pool.m, ll)?;
    let ninf = f.like_f(lse2, f64::NEG_INFINITY);
    let lse2 = f.select(has, lse2, ninf)?;
    let mx = f.max(lse, lse2)?;
    let finite = f.compare(Cmp::Gt, mx, ninf)?;
    let mx = f.select(finite, mx, zero)?;
    let d1 = f.sub(lse, mx)?;
    let w1 = f.exp(d1);
    let d2 = f.sub(lse2, mx)?;
    let w2 = f.exp(d2);
    let den = f.add(w1, w2)?;
    let pos = f.compare(Cmp::Gt, den, zero)?;
    let one = f.like_f(den, 1.0);
    let inv = f.div(one, den)?;
    let inv = f.select(pos, inv, zero)?;
    let w1 = f.mul(w1, inv)?;
    let w2 = f.mul(w2, inv)?;
    let dims = f.dims(o).to_vec();
    let w1 = f.broadcast(w1, &dims, &[0, 1, 2])?;
    let w2 = f.broadcast(w2, &dims, &[0, 1, 2])?;
    let a = f.mul(o, w1)?;
    let b = f.mul(pool.o, w2)?;
    Ok(Flashed {
        o: f.add(a, b)?,
        m: mx,
        l: den,
    })
}

/// Each row's segment: the first row of the run of rows that are one lane's
/// consecutive positions (a row written to the sink stands alone), as
/// `(start [r], p0 [r])`, the run's first row and that row's position.
pub(crate) fn segments(f: &mut Func, req: Val, qpos: Val, alone: Val) -> Built<(Val, Val)> {
    let r = f.dims(req)[0];
    let io = f.iota(Elem::I32, &[r], 0);
    let prev = |f: &mut Func, x: Val, fill: i64| -> Built<Val> {
        let head = f.slice_axis(x, 0, 0, r - 1)?;
        let c = f.const_i(f.elem(x), fill, &[]);
        f.pad(head, c, &[1], &[0], &[0])
    };
    let pr = prev(f, req, -1)?;
    let pp = prev(f, qpos, -2)?;
    let pp = with_i(f, pp, 1, Func::add)?;
    let pa = prev(f, alone, 1)?;
    let same = f.compare(Cmp::Eq, req, pr)?;
    let next = f.compare(Cmp::Eq, qpos, pp)?;
    let same = f.and(same, next)?;
    let lone = f.or(alone, pa)?;
    let lone = f.not(lone);
    let same = f.and(same, lone)?;
    let z = f.like_i(io, 0);
    let cand = f.select(same, z, io)?;
    let start = f.scan_steps(cand, 0, Fold::Max)?;
    let back = f.sub(io, start)?;
    let p0 = f.sub(qpos, back)?;
    Ok((start, p0))
}
