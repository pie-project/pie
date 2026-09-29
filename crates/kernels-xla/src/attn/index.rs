#![allow(clippy::too_many_arguments)]
//! The sparse-attention indexer (DeepSeek DSA / V4): a small per-token key,
//! layer-normed and roped, cached in its own paged pool; per query, a
//! ReLU-gated multi-head score against every cached key (or every
//! `ratio`-th), and the `top_k` keys by a bisected threshold.
//!
//! In kernels-wgpu this module lives inline in `attn.rs` (`pub mod index`).

use dtype::Dtype;

use super::pool::{cell_of, first, paging, put_cols, write_cell};
use super::ssm::{DROP, ci, clamp_i, cmp_i, flat_i32, loop_upto, nonzero};
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Combine, Elem, Fold, Func, ScatterDims, Val};
use crate::tensor::{KvPool, Tensor};

/// Keys per step of the scoring loop.
const KEY_BLOCK: i64 = 512;

/// Bisection steps of the top-k threshold, as the GPU kernels take them.
const BISECT: i64 = 40;

fn rotated(op: &'static str, rope_dim: u32, head_dim: u32) -> Result<(), Error> {
    if !rope_dim.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the rotated prefix {rope_dim} is odd, and this rotation turns pairs"),
        ));
    }
    if rope_dim > head_dim {
        return Err(refuse(
            op,
            format!("the rotated prefix {rope_dim} is wider than the {head_dim}-wide row"),
        ));
    }
    Ok(())
}

/// Rotates the first `rope_dim` lanes of every `hd`-wide head of `x`
/// (`[rows, heads, hd]` f32) by interleaved pairs, `θ^(-2i/rope_dim)` per
/// pair, at each row's position.
fn rope_heads(f: &mut Func, x: Val, pos: Val, rope_dim: i64, theta: f32) -> Built<Val> {
    if rope_dim == 0 {
        return Ok(x);
    }
    let d = f.dims(x).to_vec();
    let (rows, heads, hd) = (d[0], d[1], d[2]);
    let pairs = rope_dim / 2;
    let freqs: Vec<f64> = (0..pairs)
        .map(|i| f64::from(theta.powf(-2.0 * i as f32 / rope_dim as f32)))
        .collect();
    let fr = f.const_floats(Elem::F32, &freqs, &[pairs])?;
    let fr = f.broadcast(fr, &[rows, pairs], &[1])?;
    let p = f.convert(pos, Elem::F32);
    let p = f.broadcast(p, &[rows, pairs], &[0])?;
    let ang = f.mul(p, fr)?;
    let c = f.cos(ang);
    let s = f.sin(ang);
    let c = f.broadcast(c, &[rows, heads, pairs], &[0, 2])?;
    let s = f.broadcast(s, &[rows, heads, pairs], &[0, 2])?;
    let head = f.slice_axis(x, 2, 0, rope_dim)?;
    let head = f.reshape(head, &[rows, heads, pairs, 2])?;
    let a = f.slice_axis(head, 3, 0, 1)?;
    let a = f.reshape(a, &[rows, heads, pairs])?;
    let b = f.slice_axis(head, 3, 1, 2)?;
    let b = f.reshape(b, &[rows, heads, pairs])?;
    let ac = f.mul(a, c)?;
    let bs = f.mul(b, s)?;
    let a2 = f.sub(ac, bs)?;
    let bc = f.mul(b, c)?;
    let as_ = f.mul(a, s)?;
    let b2 = f.add(bc, as_)?;
    let a2 = f.reshape(a2, &[rows, heads, pairs, 1])?;
    let b2 = f.reshape(b2, &[rows, heads, pairs, 1])?;
    let turned = f.concat(&[a2, b2], 3)?;
    let turned = f.reshape(turned, &[rows, heads, rope_dim])?;
    if rope_dim == hd {
        return Ok(turned);
    }
    let rest = f.slice_axis(x, 2, rope_dim, hd)?;
    f.concat(&[turned, rest], 2)
}

/// The index key, in place: layernorm (`w`, `b`), rounded to its bf16
/// store, then its first `rope_dim` lanes roped at the row's position.
pub fn layernorm_rope(
    ctx: &Ctx<'_>,
    k: Tensor,
    positions: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "attention.index_layernorm_rope";
    expect(OP, k, &[Dtype::Bf16])?;
    let hd = nonzero(OP, "the index key row's width", k.width)?;
    rotated(OP, rope_dim, hd)?;
    let rows = nonzero(OP, "rows", k.rows)?;
    if weight.elements() != u64::from(hd) || bias.elements() != u64::from(hd) {
        return Err(refuse(OP, "the norm's weight and bias are not one value per lane"));
    }
    if positions.elements() < u64::from(rows) {
        return Err(refuse(OP, "the position table is shorter than the fire"));
    }
    let (n, hd) = (i64::from(rows), i64::from(hd));
    ctx.emit(&mut |cx| {
        let x = cx.read_f32(k)?;
        let w = super::ssm::flat_f32(cx, weight)?;
        let b = super::ssm::flat_f32(cx, bias)?;
        let pos = flat_i32(cx, positions)?;
        let f = cx.func();
        let pos = first(f, pos, n)?;
        let sum = f.reduce(x, &[1], Fold::Sum)?;
        let mean = f.scale(sum, 1.0 / hd as f64)?;
        let mean = f.broadcast(mean, &[n, hd], &[0])?;
        let c = f.sub(x, mean)?;
        let sq = f.mul(c, c)?;
        let var = f.reduce(sq, &[1], Fold::Sum)?;
        let var = f.scale(var, 1.0 / hd as f64)?;
        let var = f.offset(var, f64::from(eps))?;
        let inv = f.rsqrt(var);
        let inv = f.broadcast(inv, &[n, hd], &[0])?;
        let y = f.mul(c, inv)?;
        let w = f.broadcast(w, &[n, hd], &[1])?;
        let b = f.broadcast(b, &[n, hd], &[1])?;
        let y = f.mul(y, w)?;
        let y = f.add(y, b)?;
        // The normed key is stored before it is roped.
        let y = f.convert(y, Elem::Bf16);
        let y = f.convert(y, Elem::F32);
        let y = f.reshape(y, &[n, 1, hd])?;
        let y = rope_heads(f, y, pos, i64::from(rope_dim), theta)?;
        cx.write(k, y)
    })
}

/// The index query, in place: the first `rope_dim` lanes of each head roped.
pub fn rope(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "attention.index_rope";
    expect(OP, q, &[Dtype::Bf16])?;
    let h = nonzero(OP, "the head count this rotation states", heads)?;
    let hd = nonzero(OP, "the head width this rotation states", head_dim)?;
    rotated(OP, rope_dim, hd)?;
    let rows = nonzero(OP, "rows", q.rows)?;
    if q.width != h * hd {
        return Err(refuse(OP, "the query row is not heads x head width"));
    }
    if positions.elements() < u64::from(rows) {
        return Err(refuse(OP, "the position table is shorter than the fire"));
    }
    let n = i64::from(rows);
    ctx.emit(&mut |cx| {
        let x = cx.read_f32(q)?;
        let pos = flat_i32(cx, positions)?;
        let f = cx.func();
        let pos = first(f, pos, n)?;
        let x = f.reshape(x, &[n, i64::from(h), i64::from(hd)])?;
        let y = rope_heads(f, x, pos, i64::from(rope_dim), theta)?;
        cx.write(q, y)
    })
}

/// Files each row's index key at `write_page · page_size + write_offset`
/// (a negative page or an out-of-page offset drops the row).
pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    keys: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_kv_append";
    expect(OP, k, &[Dtype::Bf16])?;
    if keys.page_size <= 0 {
        return Err(refuse(OP, "the kv page size is zero"));
    }
    if keys.seq_stride != u64::from(k.width) || keys.keys.width < k.width {
        return Err(refuse(
            OP,
            format!(
                "the pool's token pitch {} is not the {}-wide row this index writes",
                keys.seq_stride, k.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", k.rows)?;
    if write_page.elements() < u64::from(rows) || write_offset.elements() < u64::from(rows) {
        return Err(refuse(OP, "the write tables are shorter than the fire"));
    }
    let (n, ps) = (i64::from(rows), i64::from(keys.page_size));
    ctx.emit(&mut |cx| {
        let wp = flat_i32(cx, write_page)?;
        let wo = flat_i32(cx, write_offset)?;
        let kv = cx.read(k)?;
        let table = cx.read(keys.keys)?;
        let f = cx.func();
        let at = write_cell(f, wp, wo, n, ps)?;
        let table = put_cols(f, table, at, kv)?;
        cx.write(keys.keys, table)
    })
}

/// Each boundary row's mean of the `ratio` cached keys ending at its
/// position (positions before 0 add nothing; the divisor stays `ratio`);
/// 0 for a row closing no block.
pub fn block_mean(
    ctx: &Ctx<'_>,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    keys: &KvPool,
    head_dim: u32,
    ratio: u32,
    entries: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_block_mean";
    expect(OP, entries, &[Dtype::Bf16])?;
    if keys.page_size <= 0 {
        return Err(refuse(OP, "the index cache page size is zero"));
    }
    let hd = nonzero(OP, "the key width this mean states", head_dim)?;
    if entries.width != hd || keys.keys.width < hd {
        return Err(refuse(OP, "the stated head width is not the entry's (or the pool's) width"));
    }
    let ratio = i64::from(nonzero(OP, "the block width this mean pools over", ratio)?);
    let rows = nonzero(OP, "rows", boundary_pos.rows)?;
    if boundary_req.elements() < u64::from(rows) || entries.rows != rows {
        return Err(refuse(OP, "the boundary tables and entries are one row per token row"));
    }
    let (n, hd) = (i64::from(rows), i64::from(hd));
    ctx.emit(&mut |cx| {
        let (indices, indptr, ps) = paging(cx, OP, keys)?;
        let bpos = flat_i32(cx, boundary_pos)?;
        let breq = flat_i32(cx, boundary_req)?;
        let table = cx.read(keys.keys)?;
        let f = cx.func();
        let bpos = first(f, bpos, n)?;
        let breq = first(f, breq, n)?;
        let i = f.iota(Elem::I32, &[n, ratio], 1);
        let bp = f.broadcast(bpos, &[n, ratio], &[0])?;
        let back = f.const_i(Elem::I32, ratio - 1, &[n, ratio]);
        let pos = f.add(bp, i)?;
        let pos = f.sub(pos, back)?;
        let live = cmp_i(f, Cmp::Ge, pos, 0)?;
        let req = f.broadcast(breq, &[n, ratio], &[0])?;
        let cell = cell_of(f, indices, indptr, req, pos, ps)?;
        let cell = f.reshape(cell, &[n * ratio])?;
        let got = f.take_rows(table, cell)?;
        let got = f.slice_axis(got, 1, 0, hd)?;
        let got = f.convert(got, Elem::F32);
        let got = f.reshape(got, &[n, ratio, hd])?;
        let lb = f.broadcast(live, &[n, ratio, hd], &[0, 1])?;
        let zero = f.like_f(got, 0.0);
        let got = f.select(lb, got, zero)?;
        let sum = f.reduce(got, &[1], Fold::Sum)?;
        let mean = f.scale(sum, 1.0 / ratio as f64)?;
        let has = cmp_i(f, Cmp::Ge, bpos, 0)?;
        let has = f.broadcast(has, &[n, hd], &[0])?;
        let z2 = f.like_f(mean, 0.0);
        let out = f.select(has, mean, z2)?;
        cx.write(entries, out)
    })
}

/// Per query row: scores `Σ_h max(q_h · k_j, 0) · w_h` against the cached
/// keys `j < (pos + 1) / ratio` (key `j` at position `(j + 1) · ratio − 1`),
/// then the `top_k` keys: every key when there are no more than `top_k`,
/// else, in key order, those scoring at or above a threshold bisected 40
/// times between the row's min and max (as the GPU kernels pick them);
/// unfilled slots are -1. The GPU's `scores` scratch plane is not an
/// operand: the score row's bound is the pool's `max_pages · page_size /
/// ratio`.
pub fn topk(
    ctx: &Ctx<'_>,
    q: Tensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    selection: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_topk";
    expect(OP, q, &[Dtype::Bf16])?;
    expect(OP, selection, &[Dtype::I32])?;
    let h = nonzero(OP, "the head count this ranking states", heads)?;
    let d = nonzero(OP, "the key width this ranking states", head_dim)?;
    let k = nonzero(OP, "the selection budget this ranking states", top_k)?;
    let stride = i64::from(ratio.max(1));
    if keys.page_size <= 0 {
        return Err(refuse(OP, "the index cache page size is zero"));
    }
    if q.width != h * d || keys.keys.width < d {
        return Err(refuse(OP, "the index query is not heads x key width"));
    }
    if weights.is_some_and(|w| w.width != h) {
        return Err(refuse(OP, "the index head weights are not one per stated head"));
    }
    if selection.width != k {
        return Err(refuse(OP, "the selection is not the budget it states"));
    }
    let rows = nonzero(OP, "rows", selection.rows)?;
    if q.rows < rows
        || positions.elements() < u64::from(rows)
        || request_of_token.elements() < u64::from(rows)
    {
        return Err(refuse(OP, "q and the fire tables are shorter than the selection"));
    }
    let nk = i64::from(keys.max_pages) * i64::from(keys.page_size) / stride;
    if nk == 0 {
        return Err(refuse(OP, "the pool's page bound holds no key"));
    }
    let (n, hh, dd, kk) = (i64::from(rows), i64::from(h), i64::from(d), i64::from(k));
    ctx.emit(&mut |cx| {
        let (indices, indptr, ps) = paging(cx, OP, keys)?;
        let qv = cx.read(q)?;
        let w = match weights {
            Some(w) => Some(cx.read_f32(w)?),
            None => None,
        };
        let pos = flat_i32(cx, positions)?;
        let req = flat_i32(cx, request_of_token)?;
        let table = cx.read(keys.keys)?;
        let f = cx.func();
        let qv = f.slice_axis(qv, 0, 0, n)?;
        let qv = f.reshape(qv, &[n, hh, dd])?;
        let w = match w {
            Some(w) => f.slice_axis(w, 0, 0, n)?,
            None => f.const_f(Elem::F32, 1.0, &[n, hh]),
        };
        let pos = first(f, pos, n)?;
        let req = first(f, req, n)?;
        let one = f.like_i(pos, 1);
        let total = f.add(pos, one)?;
        let sv = f.like_i(pos, stride);
        let nkeys = f.div(total, sv)?;
        let zero = f.like_i(pos, 0);
        let pos_t = cmp_i(f, Cmp::Gt, total, 0)?;
        let nkeys = f.select(pos_t, nkeys, zero)?;
        let nkeys = clamp_i(f, nkeys, 0, nk)?;
        let scores = index_scores(f, qv, w, table, indices, indptr, req, nkeys, nk, stride, ps)?;
        let sel = bisect_select(f, scores, nkeys, kk)?;
        cx.write(selection, sel)
    })
}

/// The `[n, nk]` score rows (entries past a row's `nkeys` are garbage).
fn index_scores(
    f: &mut Func,
    q: Val,
    w: Val,
    table: Val,
    indices: Val,
    indptr: Val,
    req: Val,
    nkeys: Val,
    nk: i64,
    stride: i64,
    ps: i64,
) -> Built<Val> {
    let qd = f.dims(q).to_vec();
    let (n, hh, dd) = (qd[0], qd[1], qd[2]);
    let kb = KEY_BLOCK.min(nk);
    let nb = (nk + kb - 1) / kb;
    let mx = f.reduce(nkeys, &[0], Fold::Max)?;
    let kbm = ci(f, kb - 1);
    let mx = f.add(mx, kbm)?;
    let kbv = ci(f, kb);
    let count = f.div(mx, kbv)?;
    let count = clamp_i(f, count, 0, nb)?;
    let s0 = f.const_f(Elem::F32, 0.0, &[n, nb * kb]);
    let out = loop_upto(
        f,
        count,
        &[s0],
        &[q, w, table, indices, indptr, req],
        |f, b, cr, inv| {
            let scores = cr[0];
            let (q, w, table, indices, indptr, req) = (inv[0], inv[1], inv[2], inv[3], inv[4], inv[5]);
            let kbv = ci(f, kb);
            let start = f.mul(b, kbv)?;
            let j = f.iota(Elem::I32, &[n, kb], 1);
            let sb = f.broadcast(start, &[n, kb], &[])?;
            let j = f.add(j, sb)?;
            let one = f.like_i(j, 1);
            let sv = f.like_i(j, stride);
            let p = f.add(j, one)?;
            let p = f.mul(p, sv)?;
            let p = f.sub(p, one)?;
            let rq = f.broadcast(req, &[n, kb], &[0])?;
            let cell = cell_of(f, indices, indptr, rq, p, ps)?;
            let cell = f.reshape(cell, &[n * kb])?;
            let k = f.take_rows(table, cell)?;
            let k = f.slice_axis(k, 1, 0, dd)?;
            let k = f.reshape(k, &[n, kb, dd])?;
            let dot = f.dot_general(q, k, &[0], &[0], &[2], &[2], Elem::F32)?;
            let zero = f.like_f(dot, 0.0);
            let dot = f.max(dot, zero)?;
            let wb = f.broadcast(w, &[n, hh, kb], &[0, 1])?;
            let dot = f.mul(dot, wb)?;
            let s = f.reduce(dot, &[1], Fold::Sum)?;
            let z = ci(f, 0);
            let scores = f.dynamic_update_slice(scores, s, &[z, start])?;
            Ok(vec![scores])
        },
    )?;
    f.slice_axis(out[0], 1, 0, nk)
}

/// The GPU's bisected top-k over `[n, nk]` scores with `nkeys` live per row.
fn bisect_select(f: &mut Func, s: Val, nkeys: Val, k: i64) -> Built<Val> {
    let d = f.dims(s).to_vec();
    let (n, nk) = (d[0], d[1]);
    let j = f.iota(Elem::I32, &[n, nk], 1);
    let nkb = f.broadcast(nkeys, &[n, nk], &[0])?;
    let valid = f.compare(Cmp::Lt, j, nkb)?;
    let big = f.const_f(Elem::F32, 3.0e38, &[n, nk]);
    let small = f.const_f(Elem::F32, -3.0e38, &[n, nk]);
    let lo_in = f.select(valid, s, big)?;
    let hi_in = f.select(valid, s, small)?;
    let lo = f.reduce(lo_in, &[1], Fold::Min)?;
    let hi = f.reduce(hi_in, &[1], Fold::Max)?;
    let out = f.for_loop(BISECT, &[lo, hi], |f, _, a| {
        let (lo, hi) = (a[0], a[1]);
        let sum = f.add(lo, hi)?;
        let mid = f.scale(sum, 0.5)?;
        let mb = f.broadcast(mid, &[n, nk], &[0])?;
        let ge = f.compare(Cmp::Ge, s, mb)?;
        let ge = f.and(ge, valid)?;
        let c = f.convert(ge, Elem::I32);
        let cnt = f.reduce(c, &[1], Fold::Sum)?;
        let over = cmp_i(f, Cmp::Gt, cnt, k)?;
        let lo = f.select(over, mid, lo)?;
        let hi = f.select(over, hi, mid)?;
        Ok(vec![lo, hi])
    })?;
    let thr = out[1];
    let tb = f.broadcast(thr, &[n, nk], &[0])?;
    let ge = f.compare(Cmp::Ge, s, tb)?;
    let few = cmp_i(f, Cmp::Le, nkeys, k)?;
    let few = f.broadcast(few, &[n, nk], &[0])?;
    let take = f.or(few, ge)?;
    let take = f.and(take, valid)?;
    let ti = f.convert(take, Elem::I32);
    let rank = f.scan(ti, 1, Fold::Sum)?;
    let one = f.like_i(rank, 1);
    let rank = f.sub(rank, one)?;
    let fits = cmp_i(f, Cmp::Lt, rank, k)?;
    let keep = f.and(take, fits)?;
    let drop = f.like_i(rank, DROP);
    let slot = f.select(keep, rank, drop)?;
    let r = f.iota(Elem::I32, &[n, nk, 1], 0);
    let slot = f.reshape(slot, &[n, nk, 1])?;
    let idx = f.concat(&[r, slot], 2)?;
    let init = f.const_i(Elem::I32, -1, &[n, k]);
    f.scatter(
        init,
        idx,
        j,
        &ScatterDims {
            update_window_dims: vec![],
            inserted_window_dims: vec![0, 1],
            scatter_dims_to_operand_dims: vec![0, 1],
            index_vector_dim: 2,
            ..ScatterDims::default()
        },
        Combine::Set,
    )
}
