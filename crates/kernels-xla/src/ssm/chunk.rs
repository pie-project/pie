//! A prefill fire's gated delta rule as one Mosaic kernel over its chunks:
//! each grid step takes one chunk of every value head, forms the chunk's
//! intra-chunk terms, carries the state from the chunk before (or reads the
//! lane's state out of the bank when the chunk opens a lane) and writes the
//! lane's state back when the chunk closes it. The XLA form lays the same
//! chunks out as `[chunks, heads, rows, ..]` planes and walks them with a
//! `while` loop; on a TPU the planes' layout copies and the loop's slices
//! cost more than the chunks' arithmetic.

use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};
use crate::lanes::{Lanes, ci, clamp_i, cmp_i, count_le, full_i, iota, take};
use crate::mosaic::{Grid, ICmp, Kernel, MTy, Operand, Sem, Space, Window};

use super::{DeltaIn, GDN_EPS, Raw, Segs};

/// `(y, slab)` over the lanes of a ragged fire, or `None` where the kernel
/// does not apply: the GDN's heads must be 128 wide on both sides and the
/// bank bf16, one slot per lane's `H · dk` rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_chunks(
    f: &mut Func,
    input: &DeltaIn,
    raw: &Raw,
    segs: &Segs,
    lanes: &Lanes,
    slab: Val,
    per: i64,
    c: i64,
) -> Built<Option<(Val, Val)>> {
    let DeltaIn::Gdn { hk, hv, dk, dv, .. } = *input else {
        return Ok(None);
    };
    if dk != 128
        || dv != 128
        || per != hv * dk
        || f.elem(slab) != Elem::Bf16
        || c % 8 != 0
        || c > 128
    {
        return Ok(None);
    }
    let t = f.dims(raw.rows[0])[0];
    let s = f.dims(segs.begin)[0];
    let rows = f.dims(slab)[0];
    let slots = rows / per;

    // The chunks in segment order, as `delta_chunked` lays them out.
    let cm1 = full_i(f, c - 1, s);
    let cv = full_i(f, c, s);
    let nch = f.add(segs.len, cm1)?;
    let nch = f.div(nch, cv)?;
    let capture = f.convert(segs.capture, Elem::I32);
    let nc = (t + c - 1) / c + s;
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
    let cap_n = take(f, capture, seg)?;
    let tb = f.broadcast(total, &[nc], &[])?;
    let live_n = f.compare(Cmp::Lt, n, tb)?;
    let j0 = cmp_i(f, Cmp::Eq, j, 0)?;
    let r0 = cmp_i(f, Cmp::Ne, reset_n, 0)?;
    let first_n = f.and(j0, r0)?;
    let first_n = f.and(first_n, live_n)?;
    let one = f.like_i(nch_n, 1);
    let lastj = f.sub(nch_n, one)?;
    let jl = f.compare(Cmp::Eq, j, lastj)?;
    let c0 = cmp_i(f, Cmp::Ne, cap_n, 0)?;
    let last_n = f.and(jl, c0)?;
    let last_n = f.and(last_n, live_n)?;
    // Each chunk's rows of its lane (none past the fire's live chunks).
    let cn = full_i(f, c, nc);
    let done = f.mul(j, cn)?;
    let left = f.sub(len_n, done)?;
    let left = clamp_i(f, left, 0, c)?;
    let none = full_i(f, 0, nc);
    let nvalid = f.select(live_n, left, none)?;
    let first_i = f.convert(first_n, Elem::I32);
    let last_i = f.convert(last_n, Elem::I32);
    let slot_n = take(f, lanes.slot, lane_n)?;
    let slot_n = clamp_i(f, slot_n, 0, slots - 1)?;

    let ic = f.iota(Elem::I32, &[nc, c], 1);
    let jb = f.broadcast(j, &[nc, c], &[0])?;
    let cc = f.const_i(Elem::I32, c, &[nc, c]);
    let local = f.mul(jb, cc)?;
    let local = f.add(local, ic)?;
    let lenb = f.broadcast(len_n, &[nc, c], &[0])?;
    let inside = f.compare(Cmp::Lt, local, lenb)?;
    let live = f.broadcast(live_n, &[nc, c], &[0])?;
    let valid = f.and(inside, live)?;
    let begb = f.broadcast(beg_n, &[nc, c], &[0])?;
    let row = f.add(begb, local)?;
    let row = clamp_i(f, row, 0, t - 1)?;
    let sr = nc * c;
    let row = f.reshape(row, &[sr])?;
    let valid = f.reshape(valid, &[sr])?;

    // The chunk rows' `q | k | v` as they are stored (the kernel widens
    // them, and zeroes the `q` of the rows past its chunk's lane: f32 rows
    // blanked here cost a v5e Qwen3.5-0.8B 1792-row fire 18% of its GDN),
    // laid as the rows are: the kernel takes each head off its rows (normed,
    // `q` scaled), so no head-major plane is laid out for it (v5e: six
    // layout copies a layer, 1.1 ms of a 640-row fire beside the kernel's
    // 2.2), and the decay and `β` per value head, blank rows zero.
    let qkv = f.take_rows(raw.rows[0], row)?;
    let gates = f.take_rows(raw.rows[1], row)?;
    let gates = f.convert(gates, Elem::F32);
    let blank = |f: &mut Func, x: Val| -> Built<Val> {
        let d = f.dims(x).to_vec();
        let vb = f.broadcast(valid, &d, &[0])?;
        let zero = f.like_f(x, 0.0);
        f.select(vb, x, zero)
    };
    let keys = hk * dk;
    let width = 2 * keys + hv * dv;
    let qkv = f.slice_axis(qkv, 1, 0, width)?;
    let stored = f.elem(qkv);
    let g = f.slice_axis(gates, 1, 0, hv)?;
    let g = blank(f, g)?;
    let beta = f.slice_axis(gates, 1, hv, 2 * hv)?;
    let beta = blank(f, beta)?;
    // The decay summed within each chunk.
    let g3 = f.reshape(g, &[nc, c, hv])?;
    let gcum = f.scan_steps(g3, 1, Fold::Sum)?;
    let gcum = f.reshape(gcum, &[sr, hv])?;

    let compiled = kernel(hk, hv, dk, dv, c, nc, rows, stored)?;
    let out = compiled.call_aliased(
        f,
        &[first_i, last_i, slot_n, nvalid, qkv, gcum, beta, slab],
        &[(1, 7)],
    )?;
    let o = out[0];

    // Token `ti` sits in its segment's chunk `cstart + local / c`.
    let ti = iota(f, t);
    let sg = count_le(f, segs.begin, ti)?;
    let one_sg = f.like_i(sg, 1);
    let sg = f.sub(sg, one_sg)?;
    let sg = clamp_i(f, sg, 0, s - 1)?;
    let b_tok = take(f, segs.begin, sg)?;
    let local_tok = f.sub(ti, b_tok)?;
    let cvt = f.like_i(local_tok, c);
    let ch = f.div(local_tok, cvt)?;
    let within = f.rem(local_tok, cvt)?;
    let cs = take(f, cstart, sg)?;
    let ch = f.add(cs, ch)?;
    let pos = f.mul(ch, cvt)?;
    let pos = f.add(pos, within)?;
    let pos = clamp_i(f, pos, 0, sr - 1)?;
    let y = f.take_rows(o, pos)?;
    let y = f.reshape(y, &[t, hv, dv])?;
    Ok(Some((y, out[1])))
}

/// The kernel: grid over the `nc` chunks, the state carried in VMEM.
fn kernel(
    hk: i64,
    hv: i64,
    dk: i64,
    dv: i64,
    c: i64,
    nc: i64,
    rows: i64,
    stored: Elem,
) -> Built<crate::mosaic::Compiled> {
    let sr = nc * c;
    let per = hv * dk;
    // A chunk's rows at a step, `w` lanes a row.
    let rowed = |w: i64, elem: Elem| Operand {
        array: vec![sr, w],
        elem,
        block: vec![c, w],
        window: Window::Blocked,
        index: Box::new(|b, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
    };
    let bank = || Operand {
        array: vec![rows, dv],
        elem: Elem::Bf16,
        block: vec![rows, dv],
        window: Window::Any,
        index: Box::new(|b, _g: &Grid| Ok(vec![b.i32(0), b.i32(0)])),
    };
    // A value per token and head, the chunk's rows at a step.
    let tokens = || Operand {
        array: vec![sr, hv],
        elem: Elem::F32,
        block: vec![c, hv],
        window: Window::Blocked,
        index: Box::new(|b, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
    };
    let kernel = Kernel {
        name: "gdn_chunks".to_string(),
        grid: vec![nc],
        semantics: vec![Sem::Arbitrary],
        prefetch: vec![vec![nc], vec![nc], vec![nc], vec![nc]],
        ins: vec![
            rowed(2 * hk * dk + hv * dv, stored),
            tokens(),
            tokens(),
            bank(),
        ],
        outs: vec![rowed(hv * dv, Elem::F32), bank()],
        vmem_limit: Some(96 << 20),
        scratch: vec![
            MTy::Ref(vec![hv, dk, dv], Elem::F32, Space::Vmem),
            MTy::Ref(vec![per, dv], Elem::Bf16, Space::Vmem),
            MTy::Sem(vec![1]),
        ],
    };
    let rep = hv / hk;
    kernel.build(|b, blk| {
        let (x_r, g_r, b_r, bank_r) = (blk.ins[0], blk.ins[1], blk.ins[2], blk.ins[3]);
        let (o_r, bank_o) = (blk.outs[0], blk.outs[1]);
        let (st, sb, sems) = (blk.scratch[0], blk.scratch[1], blk.scratch[2]);
        let n = blk.grid.ids[0];
        let z = b.i32(0);
        let sem = b.ref_slice(sems, &[z], &[1])?;
        let sem = b.ref_squeeze(sem)?;
        let per_v = b.i32(per);
        let slot = b.load(blk.grid.prefetch[2], &[n])?;
        let at = b.muli(slot, per_v)?;
        let first = b.load(blk.grid.prefetch[0], &[n])?;
        let first = b.cmpi(ICmp::Ne, first, z)?;
        let last = b.load(blk.grid.prefetch[1], &[n])?;
        let last = b.cmpi(ICmp::Ne, last, z)?;

        // A lane's first chunk starts from its slot of the bank.
        b.when(first, |b| {
            let src = b.ref_slice(bank_r, &[at, z], &[per, dv])?;
            b.dma_start(src, sb, sem)?;
            b.dma_wait(sem, src, sb)?;
            let s = b.vload_all(sb)?;
            let s = b.convert_f(s, Elem::F32)?;
            let s = b.shape_cast(s, &[hv, dk, dv])?;
            b.vstore_all(s, st)
        })?;

        let s0 = b.vload_all(st)?;
        // Each head off the chunk's rows; `q`, `k` normed (each row's sum
        // of squares across its lanes through the MXU, against ones, as a
        // bf16 head and its bf16 rest: two passes, not full precision's
        // six), `q` scaled.
        let x = b.vload_all(x_r)?;
        let x = b.convert_f(x, Elem::F32)?;
        // A row past the chunk's lane reads its `q` as zero, so its output
        // is zero (its `β` and decay come blank: it moves no state).
        let nv = b.load(blk.grid.prefetch[3], &[n])?;
        let nv = b.broadcast_i(nv, &[c, dk])?;
        let rid = b.iota(&[c, dk], 0);
        let in_lane = b.cmpi(ICmp::Slt, rid, nv)?;
        let ones = b.splat_f(&[dk, dk], Elem::Bf16, 1.0)?;
        let zero_cd = b.splat_f(&[c, dk], Elem::F32, 0.0)?;
        let eps = b.splat_f(&[c, dk], Elem::F32, GDN_EPS)?;
        let qs = b.splat_f(&[c, dk], Elem::F32, 1.0 / (dk as f64).sqrt())?;
        let heads = |b: &mut crate::mosaic::Body,
                     n: i64,
                     at: i64,
                     d: i64,
                     norm: bool,
                     scale: Option<crate::mosaic::V>|
         -> Built<crate::mosaic::V> {
            let mut hs = Vec::with_capacity(n as usize);
            for h in 0..n {
                let mut xh = b.slice(x, &[0, at + h * d], &[c, d])?;
                if scale.is_some() {
                    xh = b.select(in_lane, xh, zero_cd)?;
                }
                if norm {
                    let sq = b.mulf(xh, xh)?;
                    let hi = b.convert_f(sq, Elem::Bf16)?;
                    let wide = b.convert_f(hi, Elem::F32)?;
                    let lo = b.subf(sq, wide)?;
                    let lo = b.convert_f(lo, Elem::Bf16)?;
                    let ss = b.matmul_g(hi, ones, zero_cd, 1, 0, false)?;
                    let ss = b.matmul_g(lo, ones, ss, 1, 0, false)?;
                    let ss = b.addf(ss, eps)?;
                    let inv = b.rsqrt(ss)?;
                    xh = b.mulf(xh, inv)?;
                }
                if let Some(sc) = scale {
                    xh = b.mulf(xh, sc)?;
                }
                hs.push(b.shape_cast(xh, &[1, c, d])?);
            }
            b.concat(&hs, 0)
        };
        let q = heads(b, hk, 0, dk, true, Some(qs))?;
        let k = heads(b, hk, hk * dk, dk, true, None)?;
        let v = heads(b, hv, 2 * hk * dk, dv, false, None)?;
        // The chunk's decay sums and `β`, a row per token and a lane per
        // head, turned to a row per head: as a lane-wide column per token
        // (`[H, C, 1]` stretched) and as the row itself.
        let gt = b.vload_all(g_r)?;
        let gt = b.transpose(gt)?;
        let grow = b.shape_cast(gt, &[hv, 1, c])?;
        let gcol = b.shape_cast(gt, &[hv, c, 1])?;
        let gb = b.broadcast(gcol, &[hv, c, 128])?;
        let bt = b.vload_all(b_r)?;
        let bt = b.transpose(bt)?;
        let bcol = b.shape_cast(bt, &[hv, c, 1])?;
        let bb = b.broadcast(bcol, &[hv, c, 128])?;
        // Key heads spread over their value heads.
        let spread =
            |b: &mut crate::mosaic::Body, x: crate::mosaic::V| -> Built<crate::mosaic::V> {
                if rep == 1 {
                    return Ok(x);
                }
                let x = b.shape_cast(x, &[hk, 1, c, dk])?;
                let x = b.broadcast(x, &[hk, rep, c, dk])?;
                b.shape_cast(x, &[hv, c, dk])
            };
        let q = spread(b, q)?;
        let k = spread(b, k)?;

        let sq = [hv, c, c];
        let ri = b.iota(&sq, 1);
        let cj = b.iota(&sq, 2);
        let lower = b.cmpi(ICmp::Sge, ri, cj)?;
        let strict = b.cmpi(ICmp::Sgt, ri, cj)?;
        let diag = b.cmpi(ICmp::Eq, ri, cj)?;
        let zero_sq = b.splat_f(&sq, Elem::F32, 0.0)?;
        let one_sq = b.splat_f(&sq, Elem::F32, 1.0)?;
        let eye = b.select(diag, one_sq, zero_sq)?;

        // Decay from row `j` to row `i` of the chunk, `exp(g_i − g_j)`.
        let gc = b.broadcast(gcol, &sq)?;
        let gr = b.broadcast(grow, &sq)?;
        let dlt = b.subf(gc, gr)?;
        let dlt = b.select(lower, dlt, zero_sq)?;
        let dm = b.exp(dlt)?;
        let e = b.exp(gb)?;
        let beta = bb;

        // `T = I + β · dm ∘ k kᵀ`, unit lower-triangular, and `T⁻¹ [β v | β e k]`
        // by block forward substitution, as the XLA form solves it.
        let kb = b.mulf(k, beta)?;
        let kk = b.matmul_g(kb, k, zero_sq, 2, 2, false)?;
        let a = b.mulf(dm, kk)?;
        let a = b.select(strict, a, zero_sq)?;
        let tm = b.addf(a, eye)?;
        let vb = b.mulf(v, beta)?;
        let kbe = b.mulf(kb, e)?;
        let rhs = b.concat(&[vb, kbe], 2)?;
        let uw = if 128 % c == 0 && c >= 16 && hv % (128 / c) == 0 {
            solve_lanes(b, tm, rhs, hv, c)?
        } else {
            solve_unit_lower(b, tm, rhs, hv, c)?
        };
        let u = b.slice(uw, &[0, 0, 0], &[hv, c, dv])?;
        let w = b.slice(uw, &[0, 0, dv], &[hv, c, dk])?;
        let zero_cv = b.splat_f(&[hv, c, dv], Elem::F32, 0.0)?;
        let ws = b.matmul_g(w, s0, zero_cv, 2, 1, false)?;
        let u = b.subf(u, ws)?;
        let qe = b.mulf(q, e)?;
        let oq = b.matmul_g(qe, s0, zero_cv, 2, 1, false)?;
        let qk = b.matmul_g(q, k, zero_sq, 2, 2, false)?;
        let p = b.mulf(qk, dm)?;
        let p = b.select(lower, p, zero_sq)?;
        let o2 = b.matmul_g(p, u, zero_cv, 2, 1, false)?;
        let o = b.addf(oq, o2)?;
        for h in 0..hv {
            let oh = b.slice(o, &[h, 0, 0], &[1, c, dv])?;
            let oh = b.shape_cast(oh, &[c, dv])?;
            let (r0, col) = (b.index(0), b.index(h * dv));
            b.vstore(oh, o_r, &[r0, col])?;
        }

        // The state after the chunk: decayed to its end, plus `kᵀ u` with
        // each row's decay to the end.
        let glast = b.slice(gb, &[0, c - 1, 0], &[hv, 1, 128])?;
        let glast_c = b.broadcast(glast, &[hv, c, 128])?;
        let tail = b.subf(glast_c, gb)?;
        let tail = b.exp(tail)?;
        let kt = b.mulf(k, tail)?;
        let zero_s = b.splat_f(&[hv, dk, dv], Elem::F32, 0.0)?;
        let upd = b.matmul_g(kt, u, zero_s, 1, 1, false)?;
        let gl = b.exp(glast)?;
        let gl = b.broadcast(gl, &[hv, dk, dv])?;
        let decayed = b.mulf(s0, gl)?;
        let s1 = b.addf(decayed, upd)?;
        b.vstore_all(s1, st)?;

        // A lane's last chunk lands its state in the bank.
        b.when(last, |b| {
            let s = b.shape_cast(s1, &[per, dv])?;
            let s = b.convert_f(s, Elem::Bf16)?;
            b.vstore_all(s, sb)?;
            let dst = b.ref_slice(bank_o, &[at, z], &[per, dv])?;
            b.dma_start(sb, dst, sem)?;
            b.dma_wait(sem, sb, dst)
        })?;
        Ok(())
    })
}

/// `X = T⁻¹ R` for a unit lower-triangular `T` `[H, C, C]` and `R` `[H, C, N]`
/// (f32): forward substitution over 16-row blocks, each block's rows one
/// at a time on the VPU and the blocks before it taken off through the MXU
/// at full precision (the XLA form's `unit_lower_inverse`, applied).
fn solve_unit_lower(
    b: &mut crate::mosaic::Body,
    t: crate::mosaic::V,
    r: crate::mosaic::V,
    h: i64,
    c: i64,
) -> Built<crate::mosaic::V> {
    let n = match b.ty(r) {
        MTy::Vector(d, _) => d[2],
        _ => unreachable!("a vector"),
    };
    let bs = c.min(16);
    let mut blocks: Vec<crate::mosaic::V> = Vec::with_capacity((c / bs) as usize);
    for i in 0..c / bs {
        let lo = i * bs;
        // This block's rows of `R`, less what the solved blocks above
        // contribute through `T`'s off-diagonal part.
        let mut target = b.slice(r, &[0, lo, 0], &[h, bs, n])?;
        if i > 0 {
            let above = b.concat(&blocks, 1)?;
            let coupling = b.slice(t, &[0, lo, 0], &[h, bs, lo])?;
            let zero = b.splat_f(&[h, bs, n], Elem::F32, 0.0)?;
            let carried = b.matmul_g(coupling, above, zero, 2, 1, true)?;
            target = b.subf(target, carried)?;
        }
        let lt = b.slice(t, &[0, lo, lo], &[h, bs, bs])?;
        let mut rows: Vec<crate::mosaic::V> = Vec::with_capacity(bs as usize);
        for q in 0..bs {
            let mut row = b.slice(target, &[0, q, 0], &[h, 1, n])?;
            if q > 0 {
                let solved = b.concat(&rows, 1)?;
                let coef = b.slice(lt, &[0, q, 0], &[h, 1, q])?;
                let coef = b.shape_cast(coef, &[h, q, 1])?;
                let coef = b.broadcast(coef, &[h, q, n])?;
                let prod = b.mulf(coef, solved)?;
                let sum = b.reduce_sum(prod, 1)?;
                let sum = b.shape_cast(sum, &[h, 1, n])?;
                row = b.subf(row, sum)?;
            }
            rows.push(row);
        }
        blocks.push(b.concat(&rows, 1)?);
    }
    if blocks.len() == 1 {
        Ok(blocks[0])
    } else {
        b.concat(&blocks, 1)
    }
}

/// `X = T⁻¹ R` for a unit lower-triangular `T` `[H, C, C]` and `R`
/// `[H, C, N]` (f32), `C` 16, 32, 64 or 128 and `H` a multiple of `128 / C`.
/// Every head's 16-row diagonal blocks are inverted at once, side by side in
/// the lanes (lane `16 i + n` holds a block's column `n`; narrow chunks put
/// `128 / C` heads in a row), a column of the inverses at a pass of
/// lane-wise products over all their rows; then `R' = D⁻¹ R` and
/// `M = D⁻¹ E` (`E` the part of `T` below the blocks) through the MXU, and
/// `X = (I + M)⁻¹ R'` block by block, as vLLM-TPU's GDN kernel forms its
/// inverse (its products at one bf16 pass, as theirs). The row-at-a-time
/// substitution (`solve_unit_lower`) relaid a `[H, 1, q]` row of
/// coefficients per row: a GDN layer of a 1792-row fire, with the rows'
/// bf16 input, v5e Qwen3.5-0.8B 539 -> 292 us, Qwen3.6-27B's 48 heads 1183
/// -> 703; v6e 335 -> 186 (pf256-512 135.5k -> 173.7k tok/s), 777 -> 386.
fn solve_lanes(
    b: &mut crate::mosaic::Body,
    t: crate::mosaic::V,
    r: crate::mosaic::V,
    h: i64,
    c: i64,
) -> Built<crate::mosaic::V> {
    use crate::mosaic::V;
    let n = match b.ty(r) {
        MTy::Vector(d, _) => d[2],
        _ => unreachable!("a vector"),
    };
    let bs = 16;
    let nb = c / bs;
    let sl = [h, bs, c];
    // rsum[h, q, 16 i + j] = T[h, 16 i + q, 16 i + j]
    let lane3 = b.iota(&sl, 2);
    let zero3 = b.splat_f(&sl, Elem::F32, 0.0)?;
    let sixteen3 = b.splat_i(&sl, bs)?;
    let grp3 = b.divsi(lane3, sixteen3)?;
    let mut rsum: Option<V> = None;
    for i in 0..nb {
        let slab = b.slice(t, &[0, i * bs, 0], &sl)?;
        let gi = b.splat_i(&sl, i)?;
        let m = b.cmpi(ICmp::Eq, grp3, gi)?;
        let part = b.select(m, slab, zero3)?;
        rsum = Some(match rsum {
            None => part,
            Some(acc) => b.addf(acc, part)?,
        });
    }
    let rsum = rsum.expect("blocks");
    // Narrow chunks pack `hs` heads side by side in the lanes.
    let hs = 128 / c;
    let hp = h / hs;
    let rsum = if hs == 1 {
        rsum
    } else {
        let r4 = b.shape_cast(rsum, &[hp, hs, bs, c])?;
        let mut pieces = Vec::with_capacity(hs as usize);
        for si in 0..hs {
            let pc = b.slice(r4, &[0, si, 0, 0], &[hp, 1, bs, c])?;
            pieces.push(b.shape_cast(pc, &[hp, bs, c])?);
        }
        b.concat(&pieces, 2)?
    };
    let (h_all, c_all) = (h, c);
    let (h, c) = (hp, 128);
    // The blocks' inverses, every row at once (`[h, 16, c]`, a row of
    // every block a sublane, lanes `16 i + n`), column by column: once row
    // `j` is done, the rows below it less column `j`'s coefficients (spread
    // over their blocks' lanes) times row `j`.
    let sl2 = [h, bs, c];
    let lane = b.iota(&sl2, 2);
    let row = b.iota(&sl2, 1);
    let sixteen = b.splat_i(&sl2, bs)?;
    let lg = b.remsi(lane, sixteen)?;
    let zero = b.splat_f(&sl2, Elem::F32, 0.0)?;
    let one = b.splat_f(&sl2, Elem::F32, 1.0)?;
    let diag = b.cmpi(ICmp::Eq, lg, row)?;
    let mut x3 = b.select(diag, one, zero)?;
    for j in 0..bs - 1 {
        let jj = b.splat_i(&sl2, j)?;
        let at_j = b.cmpi(ICmp::Eq, lg, jj)?;
        let below = b.cmpi(ICmp::Sgt, row, jj)?;
        let v = b.select(at_j, rsum, zero)?;
        let mut v = if j == 0 {
            v
        } else {
            let by = b.i32(c - j);
            b.rotate(v, by, 2)?
        };
        for k in [1, 2, 4, 8] {
            let by = b.i32(k);
            let w = b.rotate(v, by, 2)?;
            v = b.addf(v, w)?;
        }
        let v = b.select(below, v, zero)?;
        let xj = b.slice(x3, &[0, j, 0], &[h, 1, c])?;
        let xj = b.broadcast(xj, &sl2)?;
        let prod = b.mulf(v, xj)?;
        x3 = b.subf(x3, prod)?;
    }
    let (h, c) = (h_all, c_all);
    let x3 = if hs == 1 {
        x3
    } else {
        let mut pieces = Vec::with_capacity(hs as usize);
        for si in 0..hs {
            let pc = b.slice(x3, &[0, 0, si * c], &[hp, bs, c])?;
            pieces.push(b.shape_cast(pc, &[hp, 1, bs, c])?);
        }
        let x4 = b.concat(&pieces, 1)?;
        b.shape_cast(x4, &[h, bs, c])?
    };
    let mut dparts = Vec::with_capacity(nb as usize);
    for i in 0..nb {
        let gi = b.splat_i(&sl, i)?;
        let m = b.cmpi(ICmp::Eq, grp3, gi)?;
        dparts.push(b.select(m, x3, zero3)?);
    }
    let dinv = b.concat(&dparts, 1)?;
    // E: T below the diagonal blocks.
    let sq = [h, c, c];
    let ri = b.iota(&sq, 1);
    let cj = b.iota(&sq, 2);
    let s16 = b.splat_i(&sq, bs)?;
    let rb = b.divsi(ri, s16)?;
    let cb = b.divsi(cj, s16)?;
    let below = b.cmpi(ICmp::Sgt, rb, cb)?;
    let zsq = b.splat_f(&sq, Elem::F32, 0.0)?;
    let e = b.select(below, t, zsq)?;
    let zr = b.splat_f(&[h, c, n], Elem::F32, 0.0)?;
    let rp = b.matmul_g(dinv, r, zr, 2, 1, false)?;
    let m = b.matmul_g(dinv, e, zsq, 2, 1, false)?;
    let mut blocks: Vec<V> = Vec::with_capacity(nb as usize);
    for i in 0..nb {
        let lo = i * bs;
        let mut x = b.slice(rp, &[0, lo, 0], &[h, bs, n])?;
        if i > 0 {
            let above = b.concat(&blocks, 1)?;
            let coupling = b.slice(m, &[0, lo, 0], &[h, bs, lo])?;
            let z = b.splat_f(&[h, bs, n], Elem::F32, 0.0)?;
            let carried = b.matmul_g(coupling, above, z, 2, 1, false)?;
            x = b.subf(x, carried)?;
        }
        blocks.push(x);
    }
    b.concat(&blocks, 1)
}
