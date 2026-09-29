#![allow(clippy::too_many_arguments)]
//! Compressed (pooled) attention, DeepSeek-V4 style: every `ratio`-th
//! position closes a block; a compressor pools the block's gated state into
//! one entry, filed in a paged pool at the closing position; queries attend
//! over their request's closed entries (all, or a selected few).
//!
//! In kernels-wgpu this module lives inline in `attn.rs` (`pub mod pool`).

use dtype::Dtype;

use super::ssm::{DROP, ci, clamp_i, cmp_i, flat_i32, loop_upto, nonzero, take};
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Combine, Elem, Fold, Func, ScatterDims, Val};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

/// Keys per step of the pooled readers' flash loop.
const KEY_BLOCK: i64 = 256;

const LOG2E: f64 = std::f64::consts::LOG2_E;

// ------------------------------------------------------------------ paging

/// The pool cell holding `pos` of request `req`:
/// `indices[indptr[req] + pos / ps] * ps + pos % ps`. `req` and `pos` share
/// a shape; a negative `pos` reads position 0 (callers mask it).
pub(crate) fn cell_of(
    f: &mut Func,
    indices: Val,
    indptr: Val,
    req: Val,
    pos: Val,
    ps: i64,
) -> Built<Val> {
    let zero = f.like_i(pos, 0);
    let pos = f.max(pos, zero)?;
    let first = take(f, indptr, req)?;
    let psv = f.like_i(pos, ps);
    let page = f.div(pos, psv)?;
    let within = f.rem(pos, psv)?;
    let at = f.add(first, page)?;
    let page = take(f, indices, at)?;
    let cell = f.mul(page, psv)?;
    f.add(cell, within)
}

/// Writes `rows` `[n, w]` over columns `[0, w)` of `table` `[N, P]` at row
/// `at[i]`; out-of-range rows are dropped.
pub(crate) fn put_cols(f: &mut Func, table: Val, at: Val, rows: Val) -> Built<Val> {
    let (p, w) = (f.dims(table)[1], f.dims(rows)[1]);
    let elem = f.elem(table);
    let rows = f.convert(rows, elem);
    if p == w {
        return f.put_rows(table, at, rows, Combine::Set);
    }
    let n = f.dims(at)[0];
    let at = f.reshape(at, &[n, 1])?;
    let zero = f.const_i(Elem::I32, 0, &[n, 1]);
    let idx = f.concat(&[at, zero], 1)?;
    f.scatter(
        table,
        idx,
        rows,
        &ScatterDims {
            update_window_dims: vec![1],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0, 1],
            index_vector_dim: 1,
            ..ScatterDims::default()
        },
        Combine::Set,
    )
}

/// A KV pool's page tables as flat i32 vectors, and its page size.
pub(crate) fn paging(
    cx: &mut crate::cx::Cx<'_>,
    op: &'static str,
    pool: &KvPool,
) -> Result<(Val, Val, i64), Error> {
    if pool.page_size <= 0 {
        return Err(refuse(op, "the pool's page size is zero"));
    }
    let indices = flat_i32(cx, pool.page_indices)?;
    let indptr = flat_i32(cx, pool.page_indptr)?;
    Ok((indices, indptr, i64::from(pool.page_size)))
}

/// The first `n` elements of a flat vector.
pub(crate) fn first(f: &mut Func, v: Val, n: i64) -> Built<Val> {
    f.slice_axis(v, 0, 0, n)
}

fn long_enough(op: &'static str, what: &str, t: Tensor, n: u32) -> Result<(), Error> {
    if t.elements() < u64::from(n) {
        return Err(refuse(
            op,
            format!("the {what} holds {} entries and this op reads {n}", t.elements()),
        ));
    }
    Ok(())
}

// --------------------------------------------------------------- boundaries

fn boundary(
    ctx: &Ctx<'_>,
    op: &'static str,
    positions: Tensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    let n = nonzero(op, "rows", boundary_pos.rows)?;
    let ratio = i64::from(nonzero(op, "the pooling ratio", ratio)?);
    for t in [boundary_pos, boundary_req, boundary_rope] {
        expect(op, t, &[Dtype::I32])?;
        if t.rows != n {
            return Err(refuse(op, "the boundary tables are one entry per token row"));
        }
    }
    long_enough(op, "position table", positions, n)?;
    long_enough(op, "owning-request table", request_of_token, n)?;
    long_enough(op, "row-valid table", row_valid, n)?;
    let n = i64::from(n);
    ctx.emit(&mut |cx| {
        let p = flat_i32(cx, positions)?;
        let r = flat_i32(cx, request_of_token)?;
        let v = flat_i32(cx, row_valid)?;
        let f = cx.func();
        let p = first(f, p, n)?;
        let r = first(f, r, n)?;
        let v = first(f, v, n)?;
        let valid = cmp_i(f, Cmp::Ne, v, 0)?;
        let one = f.like_i(p, 1);
        let rv = f.like_i(p, ratio);
        let next = f.add(p, one)?;
        let m = f.rem(next, rv)?;
        let closes = cmp_i(f, Cmp::Eq, m, 0)?;
        let is_b = f.and(valid, closes)?;
        let none = f.like_i(p, -1);
        let bpos = f.select(is_b, p, none)?;
        let blk = f.div(p, rv)?;
        let blk = f.mul(blk, rv)?;
        let zero = f.like_i(p, 0);
        let rope = f.select(is_b, blk, zero)?;
        cx.write(boundary_pos, bpos)?;
        cx.write(boundary_req, r)?;
        cx.write(boundary_rope, rope)
    })
}

/// Marks the rows whose position closes a `ratio` block: `boundary_pos` the
/// position (or -1), `boundary_rope` the block's first position, and
/// `boundary_req` each row's request in the fire.
pub fn boundary_decode(
    ctx: &Ctx<'_>,
    positions: Tensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    boundary(
        ctx,
        "attention.pool_boundary_decode",
        positions,
        request_of_token,
        row_valid,
        ratio,
        boundary_pos,
        boundary_req,
        boundary_rope,
    )
}

pub fn boundary_prefill(
    ctx: &Ctx<'_>,
    positions: RaggedTensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    boundary(
        ctx,
        "attention.pool_boundary_prefill",
        positions.data,
        request_of_token,
        row_valid,
        ratio,
        boundary_pos,
        boundary_req,
        boundary_rope,
    )
}

// -------------------------------------------------------------- compressor

/// Files each row's compressor projections (`kv`, gate logits `score`) into
/// the state planes at `write_page · page_size + write_offset`; a negative
/// page or an offset outside the page drops the row.
pub fn state_write(
    ctx: &Ctx<'_>,
    kv: Tensor,
    score: Tensor,
    pages: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
    head_dim: u32,
    ratio: u32,
    state_kv: Tensor,
    state_score: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_state_write";
    expect(OP, kv, &[Dtype::Bf16])?;
    if pages.page_size <= 0 {
        return Err(refuse(OP, "the source pool's page size is zero"));
    }
    let head_dim = nonzero(OP, "the head width this compressor states", head_dim)?;
    nonzero(OP, "the pooling ratio", ratio)?;
    let width = kv.width;
    if (width != head_dim && width != 2 * head_dim) || score.width != width || score.rows != kv.rows
    {
        return Err(refuse(
            OP,
            format!(
                "a compressor projects head width {head_dim} or twice it; the pair handed over \
                 is {} and {}",
                kv.width, score.width
            ),
        ));
    }
    if state_score.width != state_kv.width || state_kv.width < width {
        return Err(refuse(
            OP,
            "the two state slabs are one plane laid at one pitch at least the row wide",
        ));
    }
    let rows = nonzero(OP, "rows", kv.rows)?;
    long_enough(OP, "write-page table", write_page, rows)?;
    long_enough(OP, "write-offset table", write_offset, rows)?;
    let (n, ps) = (i64::from(rows), i64::from(pages.page_size));
    ctx.emit(&mut |cx| {
        let wp = flat_i32(cx, write_page)?;
        let wo = flat_i32(cx, write_offset)?;
        let k = cx.read(kv)?;
        let s = cx.read(score)?;
        let sk = cx.read(state_kv)?;
        let ss = cx.read(state_score)?;
        let f = cx.func();
        let at = write_cell(f, wp, wo, n, ps)?;
        let sk = put_cols(f, sk, at, k)?;
        let ss = put_cols(f, ss, at, s)?;
        cx.write(state_kv, sk)?;
        cx.write(state_score, ss)
    })
}

/// `page · ps + off` of each row, or [`DROP`] when the page is negative or
/// the offset is outside the page.
pub(crate) fn write_cell(f: &mut Func, page: Val, off: Val, n: i64, ps: i64) -> Built<Val> {
    let page = first(f, page, n)?;
    let off = first(f, off, n)?;
    let a = cmp_i(f, Cmp::Ge, page, 0)?;
    let b = cmp_i(f, Cmp::Ge, off, 0)?;
    let c = cmp_i(f, Cmp::Lt, off, ps)?;
    let ok = f.and(a, b)?;
    let ok = f.and(ok, c)?;
    let psv = f.like_i(page, ps);
    let cell = f.mul(page, psv)?;
    let cell = f.add(cell, off)?;
    let drop = f.like_i(cell, DROP);
    f.select(ok, cell, drop)
}

/// The compressor's pooled entry of each boundary row: a softmax over the
/// block's `coff · ratio` positions (per column, gated by the state's
/// logits plus the absolute-position bias `ape`) of the state's kv. A
/// `coff` of 2 reads the previous block through the second column half.
pub fn gather(
    ctx: &Ctx<'_>,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pages: &KvPool,
    head_dim: u32,
    ratio: u32,
    state_kv: Tensor,
    state_score: Tensor,
    ape: Option<Tensor>,
    entries: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_gather";
    expect(OP, entries, &[Dtype::Bf16])?;
    let hd = nonzero(OP, "the head width this gather states", head_dim)?;
    if hd != entries.width {
        return Err(refuse(OP, "the stated head width is not the entry's width"));
    }
    let rows = nonzero(OP, "rows", boundary_pos.rows)?;
    let ratio = nonzero(OP, "the pooling ratio", ratio)?;
    let coff: u32 = match ape {
        None => {
            if ratio == 4 {
                2
            } else {
                1
            }
        }
        Some(a) if a.width == hd => 1,
        Some(a) if a.width == 2 * hd => 2,
        Some(a) => {
            return Err(refuse(
                OP,
                format!("an ape {} wide is neither one head width ({hd}) nor two", a.width),
            ));
        }
    };
    let width = hd * coff;
    if state_score.width != state_kv.width || state_kv.width < width {
        return Err(refuse(
            OP,
            "the two state slabs are one plane laid at one pitch at least coff heads wide",
        ));
    }
    if let Some(a) = ape
        && (a.dtype != Dtype::F32 || a.width != width || a.rows != ratio)
    {
        return Err(refuse(OP, "the absolute-position plane is not an f32 [ratio, width]"));
    }
    if pages.page_size <= 0 {
        return Err(refuse(OP, "the pooled space's page size is zero"));
    }
    if entries.rows != rows || boundary_req.rows != rows {
        return Err(refuse(OP, "the boundary tables and entries are one row per token row"));
    }
    let (n, hd, r, coff) = (
        i64::from(rows),
        i64::from(hd),
        i64::from(ratio),
        i64::from(coff),
    );
    let window = coff * r;
    ctx.emit(&mut |cx| {
        let (indices, indptr, ps) = paging(cx, OP, pages)?;
        let bpos = flat_i32(cx, boundary_pos)?;
        let breq = flat_i32(cx, boundary_req)?;
        let sk = cx.read(state_kv)?;
        let ss = cx.read(state_score)?;
        let ape = match ape {
            Some(a) => Some(cx.read_f32(a)?),
            None => None,
        };
        let f = cx.func();
        let i = f.iota(Elem::I32, &[n, window], 1);
        let bp = f.broadcast(bpos, &[n, window], &[0])?;
        let back = f.const_i(Elem::I32, window - 1, &[n, window]);
        let pos = f.add(bp, i)?;
        let pos = f.sub(pos, back)?;
        let live = cmp_i(f, Cmp::Ge, pos, 0)?;
        let req = f.broadcast(breq, &[n, window], &[0])?;
        let cell = cell_of(f, indices, indptr, req, pos, ps)?;
        let cell = f.reshape(cell, &[n * window])?;
        let pitch = f.dims(sk)[1];
        let kvr = f.take_rows(sk, cell)?;
        let scr = f.take_rows(ss, cell)?;
        let kvr = f.reshape(kvr, &[n, window, pitch])?;
        let scr = f.reshape(scr, &[n, window, pitch])?;
        // Position i < ratio reads columns [0, hd), the rest [hd, 2hd).
        // (A select, not a concat of the two cuts: libtpu merges
        // `concat(slice(x), slice(x))` along the window axis as if the cuts
        // were adjacent, dropping the second one's column offset.)
        let second = f.iota(Elem::I32, &[n, window, hd], 1);
        let second = cmp_i(f, Cmp::Ge, second, r)?;
        let halves = |f: &mut Func, x: Val| -> Built<Val> {
            let lo = f.slice_axis(x, 2, 0, hd)?;
            if coff == 1 {
                return Ok(lo);
            }
            let hi = f.slice_axis(x, 2, hd, 2 * hd)?;
            f.select(second, hi, lo)
        };
        let kvr = halves(f, kvr)?;
        let kvr = f.convert(kvr, Elem::F32);
        let scr = halves(f, scr)?;
        let mut scr = f.convert(scr, Elem::F32);
        if let Some(a) = ape {
            // ape[pos % ratio, column].
            let rv = f.like_i(pos, r);
            let zero = f.like_i(pos, 0);
            let pz = f.max(pos, zero)?;
            let m = f.rem(pz, rv)?;
            let m = f.reshape(m, &[n * window])?;
            let rowsa = f.take_rows(a, m)?;
            let rowsa = f.reshape(rowsa, &[n, window, coff * hd])?;
            let rowsa = halves(f, rowsa)?;
            scr = f.add(scr, rowsa)?;
        }
        let dims = [n, window, hd];
        let live3 = f.broadcast(live, &dims, &[0, 1])?;
        let ninf = f.const_f(Elem::F32, f64::NEG_INFINITY, &dims);
        let scr = f.select(live3, scr, ninf)?;
        let mx = f.reduce(scr, &[1], Fold::Max)?;
        let dead = f.like_f(mx, -3.0e38);
        let dead = f.compare(Cmp::Gt, mx, dead)?;
        let zero2 = f.like_f(mx, 0.0);
        let mx = f.select(dead, mx, zero2)?;
        let mb = f.broadcast(mx, &dims, &[0, 2])?;
        let e = f.sub(scr, mb)?;
        let e = f.exp(e);
        let zero3 = f.const_f(Elem::F32, 0.0, &dims);
        let e = f.select(live3, e, zero3)?;
        let z = f.reduce(e, &[1], Fold::Sum)?;
        let acc = f.mul(e, kvr)?;
        let acc = f.reduce(acc, &[1], Fold::Sum)?;
        let pos_z = cmp_f(f, Cmp::Gt, z, 0.0)?;
        let one = f.like_f(z, 1.0);
        let zs = f.select(pos_z, z, one)?;
        let out = f.div(acc, zs)?;
        let ok = f.and(pos_z, dead)?;
        let out = f.select(ok, out, zero2)?;
        let has = cmp_i(f, Cmp::Ge, bpos, 0)?;
        let has = f.broadcast(has, &[n, hd], &[0])?;
        let out = f.select(has, out, zero2)?;
        cx.write(entries, out)
    })
}

fn cmp_f(f: &mut Func, dir: Cmp, v: Val, x: f64) -> Built<Val> {
    let c = f.like_f(v, x);
    f.compare(dir, v, c)
}

/// Files each boundary row's entry in the compressed pool at the block's
/// closing position. (`write_page`/`write_offset` are not read: the entry
/// lands at its boundary position's cell, as the GPU kernels file it.)
pub fn kv_append(
    ctx: &Ctx<'_>,
    entries: Tensor,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_kv_append";
    let _ = (write_page, write_offset);
    expect(OP, entries, &[Dtype::Bf16])?;
    let rows = nonzero(OP, "rows", entries.rows)?;
    if pool.keys.width < entries.width {
        return Err(refuse(OP, "the compressed pool's row is narrower than the entry"));
    }
    long_enough(OP, "boundary position table", boundary_pos, rows)?;
    long_enough(OP, "boundary request table", boundary_req, rows)?;
    let n = i64::from(rows);
    ctx.emit(&mut |cx| {
        let (indices, indptr, ps) = paging(cx, OP, pool)?;
        let bpos = flat_i32(cx, boundary_pos)?;
        let breq = flat_i32(cx, boundary_req)?;
        let e = cx.read(entries)?;
        let keys = cx.read(pool.keys)?;
        let f = cx.func();
        let at = boundary_cell(f, indices, indptr, bpos, breq, n, ps)?;
        let keys = put_cols(f, keys, at, e)?;
        cx.write(pool.keys, keys)
    })
}

/// The pool cell of each boundary row, [`DROP`] for a row closing no block.
fn boundary_cell(
    f: &mut Func,
    indices: Val,
    indptr: Val,
    bpos: Val,
    breq: Val,
    n: i64,
    ps: i64,
) -> Built<Val> {
    let bpos = first(f, bpos, n)?;
    let breq = first(f, breq, n)?;
    let cell = cell_of(f, indices, indptr, breq, bpos, ps)?;
    let has = cmp_i(f, Cmp::Ge, bpos, 0)?;
    let drop = f.like_i(cell, DROP);
    f.select(has, cell, drop)
}

// ----------------------------------------------------------------- readers

/// Which compressed rows a reader walks.
#[derive(Clone, Copy)]
enum Keys {
    /// Every closed block before the query: `0 .. (pos + 1) / ratio`.
    All,
    /// The query's selection row (ids outside the closed range skip).
    Selected(Val),
}

/// Flash attention of `q` `[R, H, D]` (bf16) over each row's compressed
/// entries (`values = keys`). Returns `o` `[R, H, D]` f32 and the base-2
/// log-sum-exp `[R, H]` (`-inf` for a row that sees nothing).
fn pooled_flash(
    f: &mut Func,
    q: Val,
    table: Val,
    indices: Val,
    indptr: Val,
    req: Val,
    visible: Val,
    keys: Keys,
    nk: i64,
    ratio: i64,
    ps: i64,
    scale: f32,
) -> Built<(Val, Val)> {
    let qd = f.dims(q).to_vec();
    let (r, h, d) = (qd[0], qd[1], qd[2]);
    let kb = KEY_BLOCK.min(nk.max(1));
    let nb = (nk + kb - 1) / kb;
    let (count, sel) = match keys {
        Keys::All => {
            // Blocks up to the furthest-seeing row.
            let mx = f.reduce(visible, &[0], Fold::Max)?;
            let kbm = ci(f, kb - 1);
            let mx = f.add(mx, kbm)?;
            let kbv = ci(f, kb);
            let c = f.div(mx, kbv)?;
            let c = clamp_i(f, c, 0, nb)?;
            (c, f.const_i(Elem::I32, 0, &[r, 1]))
        }
        Keys::Selected(s) => {
            let w = f.dims(s)[1];
            let neg = ci(f, -1);
            let s = f.pad(s, neg, &[0, 0], &[0, nb * kb - w], &[0, 0])?;
            (ci(f, nb), s)
        }
    };
    let selected = matches!(keys, Keys::Selected(_));
    let m0 = f.const_f(Elem::F32, f64::NEG_INFINITY, &[r, h]);
    let l0 = f.const_f(Elem::F32, 0.0, &[r, h]);
    let a0 = f.const_f(Elem::F32, 0.0, &[r, h, d]);
    let out = loop_upto(
        f,
        count,
        &[m0, l0, a0],
        &[q, table, indices, indptr, req, visible, sel],
        |f, b, cr, inv| {
            let (m, l, acc) = (cr[0], cr[1], cr[2]);
            let (q, table, indices, indptr, req, visible, sel) =
                (inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], inv[6]);
            let kbv = ci(f, kb);
            let start = f.mul(b, kbv)?;
            let c = if selected {
                let zero = ci(f, 0);
                f.dynamic_slice(sel, &[zero, start], &[r, kb])?
            } else {
                let i = f.iota(Elem::I32, &[r, kb], 1);
                let s = f.broadcast(start, &[r, kb], &[])?;
                f.add(i, s)?
            };
            let vis = f.broadcast(visible, &[r, kb], &[0])?;
            let a = cmp_i(f, Cmp::Ge, c, 0)?;
            let bb = f.compare(Cmp::Lt, c, vis)?;
            let valid = f.and(a, bb)?;
            let one = f.like_i(c, 1);
            let rv = f.like_i(c, ratio);
            let pos = f.add(c, one)?;
            let pos = f.mul(pos, rv)?;
            let pos = f.sub(pos, one)?;
            let rq = f.broadcast(req, &[r, kb], &[0])?;
            let cell = cell_of(f, indices, indptr, rq, pos, ps)?;
            let cell = f.reshape(cell, &[r * kb])?;
            let k = f.take_rows(table, cell)?;
            let k = f.slice_axis(k, 1, 0, d)?;
            let k = f.reshape(k, &[r, kb, d])?;
            let s = f.dot_general(q, k, &[0], &[0], &[2], &[2], Elem::F32)?;
            let s = f.scale(s, f64::from(scale))?;
            let v3 = f.broadcast(valid, &[r, h, kb], &[0, 2])?;
            let ninf = f.const_f(Elem::F32, f64::NEG_INFINITY, &[r, h, kb]);
            let s = f.select(v3, s, ninf)?;
            let mb = f.reduce(s, &[2], Fold::Max)?;
            let mn = f.max(m, mb)?;
            let is_inf = f.like_f(mn, f64::NEG_INFINITY);
            let is_inf = f.compare(Cmp::Eq, mn, is_inf)?;
            let zero = f.like_f(mn, 0.0);
            let mu = f.select(is_inf, zero, mn)?;
            let corr = f.sub(m, mu)?;
            let corr = f.exp(corr);
            let mub = f.broadcast(mu, &[r, h, kb], &[0, 1])?;
            let p = f.sub(s, mub)?;
            let p = f.exp(p);
            let ps_ = f.reduce(p, &[2], Fold::Sum)?;
            let l = f.mul(l, corr)?;
            let l = f.add(l, ps_)?;
            let kf = f.convert(k, Elem::F32);
            let pv = f.dot_general(p, kf, &[0], &[0], &[2], &[1], Elem::F32)?;
            let cb = f.broadcast(corr, &[r, h, d], &[0, 1])?;
            let acc = f.mul(acc, cb)?;
            let acc = f.add(acc, pv)?;
            Ok(vec![mn, l, acc])
        },
    )?;
    let (m, l, acc) = (out[0], out[1], out[2]);
    let live = cmp_f(f, Cmp::Gt, l, 0.0)?;
    let one = f.like_f(l, 1.0);
    let ls = f.select(live, l, one)?;
    let lb = f.broadcast(ls, &[r, h, d], &[0, 1])?;
    let o = f.div(acc, lb)?;
    let lv = f.broadcast(live, &[r, h, d], &[0, 1])?;
    let zero = f.const_f(Elem::F32, 0.0, &[r, h, d]);
    let o = f.select(lv, o, zero)?;
    let lg = f.log(ls);
    let lse = f.add(lg, m)?;
    let lse = f.scale(lse, LOG2E)?;
    let ninf = f.like_f(lse, f64::NEG_INFINITY);
    let lse = f.select(live, lse, ninf)?;
    Ok((o, lse))
}

fn reader(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Option<(Tensor, u32)>,
    entries: &KvPool,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    expect(op, q, &[Dtype::Bf16])?;
    expect(op, lse, &[Dtype::F32])?;
    if entries.page_size <= 0 {
        return Err(refuse(op, "the compressed cache page size is zero"));
    }
    let h = nonzero(op, "heads", heads)?;
    let d = nonzero(op, "the head width", head_dim)?;
    let rows = nonzero(op, "rows", o.rows)?;
    let ratio = nonzero(op, "the pooling ratio", ratio)?;
    if q.rows < rows || q.width != h * d || o.width != h * d || lse.rows != rows || lse.width != h
    {
        return Err(refuse(op, "q, o and lse are not [rows, heads x head width] / [rows, heads]"));
    }
    if entries.keys.width < d {
        return Err(refuse(op, "the compressed pool's row is narrower than a head"));
    }
    long_enough(op, "position table", positions, rows)?;
    long_enough(op, "owning-request table", request_of_token, rows)?;
    if let Some((s, k)) = selection {
        nonzero(op, "the selection budget this reader states", k)?;
        if s.width != k || s.rows < rows {
            return Err(refuse(op, "the selection is not [rows, top_k]"));
        }
    }
    let (n, hh, dd, rt) = (i64::from(rows), i64::from(h), i64::from(d), i64::from(ratio));
    let nk = match selection {
        Some((_, k)) => i64::from(k),
        None => i64::from(entries.max_pages) * i64::from(entries.page_size) / rt,
    };
    if nk == 0 {
        // No lane holds enough pages to close a block: nothing is visible,
        // so every row reads nothing (o = 0, lse = -inf), as the walk over
        // an empty key set answers.
        return ctx.emit(&mut |cx| {
            let zero = cx.const_f(Elem::F32, 0.0, &[n, hh * dd]);
            cx.write(o, zero)?;
            let ninf = cx.const_f(Elem::F32, f64::NEG_INFINITY, &[n, hh]);
            cx.write(lse, ninf)
        });
    }
    ctx.emit(&mut |cx| {
        let (indices, indptr, ps) = paging(cx, op, entries)?;
        let qv = cx.read(q)?;
        let pos = flat_i32(cx, positions)?;
        let req = flat_i32(cx, request_of_token)?;
        let table = cx.read(entries.keys)?;
        let sel = match selection {
            Some((s, _)) => {
                let v = cx.read(s)?;
                let v = cx.convert(v, Elem::I32);
                Some(cx.slice_axis(v, 0, 0, n)?)
            }
            None => None,
        };
        let f = cx.func();
        let qv = f.slice_axis(qv, 0, 0, n)?;
        let qv = f.reshape(qv, &[n, hh, dd])?;
        let pos = first(f, pos, n)?;
        let req = first(f, req, n)?;
        let one = f.like_i(pos, 1);
        let rv = f.like_i(pos, rt);
        let vis = f.add(pos, one)?;
        let vis = f.div(vis, rv)?;
        let keys = match sel {
            Some(s) => Keys::Selected(s),
            None => Keys::All,
        };
        let (ov, lv) = pooled_flash(
            f, qv, table, indices, indptr, req, vis, keys, nk, rt, ps, sm_scale,
        )?;
        cx.write(o, ov)?;
        cx.write(lse, lv)
    })
}

/// Each query row attends over every closed block of its request; `o` is
/// normalized, `lse` the base-2 log-sum-exp (for a later merge).
pub fn attention_lse(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    entries: &KvPool,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    reader(
        ctx,
        "attention.pool_lse",
        q,
        positions,
        request_of_token,
        None,
        entries,
        ratio,
        heads,
        head_dim,
        sm_scale,
        o,
        lse,
    )
}

/// As [`attention_lse`], over the row's selected block ids only.
pub fn attention_lse_selected(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Tensor,
    entries: &KvPool,
    ratio: u32,
    top_k: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    reader(
        ctx,
        "attention.pool_lse_selected",
        q,
        positions,
        request_of_token,
        Some((selection, top_k)),
        entries,
        ratio,
        heads,
        head_dim,
        sm_scale,
        o,
        lse,
    )
}
