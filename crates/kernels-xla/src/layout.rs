//! Row and column movement: embeddings, splits, row gathers and scatters,
//! folds, argmax / top-k. Every entry mirrors `kernels_wgpu::layout` (the
//! WGSL under `kernels-wgpu/kernels/layout` is the reference); the ones only
//! Metal or CUDA serve say where they were read.
//!
//! GPU-only constraints (even widths so a bf16 pair moves as a word, the
//! lanes a workgroup holds) are dropped: a traced slice has no word size.
//! Pure moves take any plain dtype, and refuse only when source and landing
//! disagree, since a move never converts.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, elem_of};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Val};
use crate::tensor::{Bank, Tensor};

/// The index a best-of reduction carries for "nothing picked yet".
const NONE: i64 = i32::MAX as i64;

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("{what} is zero")));
    }
    Ok(v)
}

fn same_dtype(op: &'static str, a: Tensor, b: Tensor) -> Result<(), Error> {
    if a.dtype != b.dtype {
        return Err(refuse(
            op,
            format!(
                "the source is {:?} and the landing {:?}; a move does not convert",
                a.dtype, b.dtype
            ),
        ));
    }
    Ok(())
}

fn i32_ids(op: &'static str, what: &str, t: Tensor) -> Result<(), Error> {
    if t.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!("the {what} are {:?}, and this op reads i32", t.dtype),
        ));
    }
    Ok(())
}

/// `t` as a flat i32 vector of its `n` first elements.
fn index_vec(cx: &mut Cx<'_>, op: &'static str, t: Tensor, n: u32) -> Result<Val, Error> {
    if t.elements() < u64::from(n) {
        return Err(refuse(
            op,
            format!(
                "{}x{} indices name fewer than the {n} rows moved",
                t.rows, t.width
            ),
        ));
    }
    let v = cx.read(t)?;
    let flat = cx.reshape(v, &[t.elements() as i64])?;
    let flat = cx.slice_axis(flat, 0, 0, i64::from(n))?;
    Ok(cx.convert(flat, Elem::I32))
}

/// `0 <= ids < vocab`, elementwise.
fn in_vocab(cx: &mut Cx<'_>, ids: Val, vocab: u32) -> Result<Val, Error> {
    let zero = cx.like_i(ids, 0);
    let top = cx.like_i(ids, i64::from(vocab));
    let lo = cx.compare(Cmp::Ge, ids, zero)?;
    let hi = cx.compare(Cmp::Lt, ids, top)?;
    Ok(cx.and(lo, hi)?)
}

/// `ids` with every id outside `[0, vocab)` sent to row 0, as the embed
/// kernels guard their read.
fn guarded(cx: &mut Cx<'_>, ids: Val, vocab: u32) -> Result<(Val, Val), Error> {
    let ok = in_vocab(cx, ids, vocab)?;
    let zero = cx.like_i(ids, 0);
    Ok((cx.select(ok, ids, zero)?, ok))
}

/// Zeroes the rows of `[n, width]` whose `ok[n]` is false.
fn zero_rows(cx: &mut Cx<'_>, rows: Val, ok: Val) -> Result<Val, Error> {
    let dims = cx.dims(rows).to_vec();
    let ok = cx.broadcast(ok, &dims, &[0])?;
    let zero = cx.like_f(rows, 0.0);
    Ok(cx.select(ok, rows, zero)?)
}

/// The first `rows` rows of a table value.
fn top_rows(cx: &mut Cx<'_>, v: Val, rows: u32) -> Result<Val, Error> {
    Ok(cx.slice_axis(v, 0, 0, i64::from(rows))?)
}

/// Writes `v` over the first rows of `y`, keeping the rest as they were.
fn write_head(cx: &mut Cx<'_>, y: Tensor, v: Val) -> Result<(), Error> {
    let n = cx.dims(v)[0];
    if n == i64::from(y.rows) {
        return cx.write(y, v);
    }
    let elem = elem_of("layout", y.dtype)?;
    let v = cx.convert(v, elem);
    let old = cx.read(y)?;
    let zero = cx.const_i(Elem::I32, 0, &[]);
    let merged = cx.dynamic_update_slice(old, v, &[zero, zero])?;
    cx.write(y, merged)
}

// ------------------------------------------------------------------ embeds

pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    i32_ids(OP, "token ids", ids)?;
    same_dtype(OP, table, y)?;
    nonzero(OP, "the row count this embedding table states", vocab)?;
    if table.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the table rows are {} wide and the landing's {}",
                table.width, y.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let ids = index_vec(cx, OP, ids, y.rows)?;
        let (ids, _) = guarded(cx, ids, vocab.min(table.rows))?;
        let t = cx.read(table)?;
        let rows = cx.take_rows(t, ids)?;
        cx.write(y, rows)
    })
}

/// The vocab-parallel embed: this rank holds table rows `[rank * local,
/// (rank + 1) * local)`, answers them, and zeroes every other id so an
/// all-reduce sums the shards. Read from kernels-cuda
/// `layout::embed_vocab_shard` (`kernels/layout/layout.cuh`), whose rank comes
/// from the communicator; here it is stated.
pub fn embed_vocab_shard(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    rank: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_vocab_shard";
    i32_ids(OP, "token ids", ids)?;
    same_dtype(OP, table, y)?;
    let local = nonzero(OP, "this rank's band of the embedding table", table.rows)?;
    if table.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "the table rows are {} wide and the landing's {}",
                table.width, y.width
            ),
        ));
    }
    let offset = i64::from(rank) * i64::from(local);
    if offset > i64::from(i32::MAX) {
        return Err(refuse(
            OP,
            format!("rank {rank}'s band starts past any i32 id"),
        ));
    }
    ctx.emit(&mut |cx| {
        let ids = index_vec(cx, OP, ids, y.rows)?;
        let off = cx.like_i(ids, offset);
        let local_ids = cx.sub(ids, off)?;
        let (local_ids, ok) = guarded(cx, local_ids, local)?;
        let t = cx.read(table)?;
        let rows = cx.take_rows(t, local_ids)?;
        let rows = zero_rows(cx, rows, ok)?;
        cx.write(y, rows)
    })
}

fn concat_shape(op: &'static str, ids: Tensor, y: Tensor) -> Result<(u32, u32), Error> {
    let heads = nonzero(op, "the ids per row", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide landing is not {heads} table rows side by side",
                y.width
            ),
        ));
    }
    if ids.rows != y.rows {
        return Err(refuse(
            op,
            format!("{} rows of ids and {} rows landed", ids.rows, y.rows),
        ));
    }
    Ok((heads, y.width / heads))
}

/// `y[n] = table[ids[n, 0]] ‖ table[ids[n, 1]] ‖ …`; an id outside the
/// vocabulary lands a zero slice.
pub fn embed_concat(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    i32_ids(OP, "token ids", ids)?;
    same_dtype(OP, table, y)?;
    nonzero(OP, "the row count this embedding table states", vocab)?;
    let (heads, width) = concat_shape(OP, ids, y)?;
    if table.width != width {
        return Err(refuse(
            OP,
            format!(
                "the table rows are {} wide and each landed slice {width}",
                table.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let n = y.rows * heads;
        let ids = index_vec(cx, OP, ids, n)?;
        let (ids, ok) = guarded(cx, ids, vocab.min(table.rows))?;
        let t = cx.read(table)?;
        let rows = cx.take_rows(t, ids)?;
        let rows = zero_rows(cx, rows, ok)?;
        cx.write(y, rows)
    })
}

/// Checks a quantized table and answers `(bits per code, storage elem)`.
fn bank_shape(op: &'static str, table: &Bank, width: u32) -> Result<Elem, Error> {
    if table.biases.is_none() {
        return Err(refuse(
            op,
            format!(
                "the table is a symmetric {}-bit bank in groups of {}, and the gather decodes \
                 the affine form alone",
                table.bits, table.group
            ),
        ));
    }
    if !matches!(table.bits, 2 | 4 | 8) || !matches!(table.group, 32 | 64 | 128) {
        return Err(refuse(
            op,
            format!(
                "no gather at group size {}, {} bits (the GPU set is 32/64/128 x 2/4/8)",
                table.group, table.bits
            ),
        ));
    }
    if !width.is_multiple_of(table.group) {
        return Err(refuse(
            op,
            format!(
                "the {width}-wide row is not a whole number of {}-code groups",
                table.group
            ),
        ));
    }
    if crate::pack::code_elem(table.codes.dtype).is_some() {
        // A bank's codes (landed one per element, `crate::pack`, or as the
        // raw bytes of their rows: the read value tells).
        if crate::pack::code_bits(table.codes.dtype) != Some(table.bits) {
            return Err(refuse(
                op,
                format!(
                    "the code plane is {:?} {} wide, and a {width}-wide row of {}-bit codes \
                     wants {width} codes",
                    table.codes.dtype, table.codes.width, table.bits
                ),
            ));
        }
        return scale_planes(
            op,
            table,
            width,
            crate::pack::code_elem(table.codes.dtype).unwrap_or(Elem::U8),
        );
    }
    let elem = elem_of(op, table.codes.dtype)?;
    if !elem.is_int() {
        return Err(refuse(
            op,
            format!(
                "the code plane is {:?}; codes are packed in an integer plane",
                table.codes.dtype
            ),
        ));
    }
    let bits = elem.bits();
    let fold = u64::from(folding(table, width));
    if u64::from(table.codes.width) * u64::from(bits)
        != fold * u64::from(width) * u64::from(table.bits)
    {
        return Err(refuse(
            op,
            format!(
                "the code plane rows are {} x {bits}-bit, and a {width}-wide row of {}-bit codes \
                 packs into {} bits",
                table.codes.width,
                table.bits,
                u64::from(width) * u64::from(table.bits)
            ),
        ));
    }
    scale_planes(op, table, width, elem)
}

/// How many table rows each stored row of a byte-plane table holds: a
/// table every reader of which gathers rows may land folded, `[V/f, f·row]`
/// for its codes (packed bytes), its scales and biases each by their own
/// `f`, so its minor axis tiles evenly and its major one does not (the TPU
/// lays a 2-D array out with the axis that tiles evenly minor: a
/// `[V, row]` table would land column-major and every gather would first
/// relayout all of it). 1 when not folded.
pub(crate) fn folding(table: &Bank, width: u32) -> u32 {
    let row = u64::from(width) * u64::from(table.bits) / 8;
    if table.codes.dtype != Dtype::U8 || row == 0 {
        return 1;
    }
    let w = u64::from(table.codes.width);
    if w > row && w.is_multiple_of(row) {
        u32::try_from(w / row).unwrap_or(1)
    } else {
        1
    }
}

/// The table rows a quantized table holds (its stored rows unfolded).
#[must_use]
pub fn table_rows(table: &Bank, width: u32) -> u32 {
    table.codes.rows.saturating_mul(folding(table, width))
}

/// Rows `ids` (i32 `[n]`) of a table `[R, f·w]` that folds `f` rows of `w`
/// into each stored row: `[n, w]`. Takes each id's whole stored row (a
/// plain row gather; a gather of `[1, w]` windows at two start indices
/// runs one slice per id on TPU), then keeps its `w`-wide part.
fn take_folded(cx: &mut Cx<'_>, plane: Val, ids: Val, fold: i64, w: i64) -> Result<Val, Error> {
    if fold == 1 {
        return Ok(cx.take_rows(plane, ids)?);
    }
    let n = cx.dims(ids)[0];
    let f = cx.like_i(ids, fold);
    let hi = cx.div(ids, f)?;
    let lo = cx.rem(ids, f)?;
    let whole = cx.take_rows(plane, hi)?;
    let whole = cx.reshape(whole, &[n, fold, w])?;
    let at = cx.iota(Elem::I32, &[n, fold, w], 1);
    let lo = cx.broadcast(lo, &[n, fold, w], &[0])?;
    let hit = cx.compare(Cmp::Eq, at, lo)?;
    let elem = cx.elem(whole);
    let zero = if elem.is_int() {
        cx.const_i(elem, 0, &[n, fold, w])
    } else {
        cx.const_f(elem, 0.0, &[n, fold, w])
    };
    let kept = cx.select(hit, whole, zero)?;
    // Exactly one term per output is kept: an integer or float sum of it
    // and zeros is the kept value.
    Ok(cx.reduce(kept, &[1], crate::hlo::Fold::Sum)?)
}

/// Checks the scale and bias planes hold one value per group per row.
fn scale_planes(op: &'static str, table: &Bank, width: u32, elem: Elem) -> Result<Elem, Error> {
    let groups = u64::from(table_rows(table, width)) * u64::from(width / table.group);
    for (what, plane) in [("scale", Some(table.scales)), ("bias", table.biases)] {
        let plane = plane.unwrap_or(table.scales);
        if plane.elements() != groups {
            return Err(refuse(
                op,
                format!(
                    "the {what} plane holds {} values, and {} rows of {} groups want {groups}",
                    plane.elements(),
                    table.codes.rows,
                    width / table.group
                ),
            ));
        }
    }
    Ok(elem)
}

/// The rows `ids` of an affine bank, decoded to f32 `[n, width]`: codes run
/// from the low bits of each little-endian storage element, and a code `c`
/// in group `g` is `scale[g] * c + bias[g]` (`common/affine.inc.wgsl`).
pub(crate) fn affine_rows(
    cx: &mut Cx<'_>,
    table: &Bank,
    ids: Val,
    width: u32,
) -> Result<Val, Error> {
    let n = cx.dims(ids)[0];
    let fold = i64::from(folding(table, width));
    let rows = i64::from(table.codes.rows) * fold;
    let groups = i64::from(width / table.group);
    let codes = cx.read(table.codes)?;
    if cx.ty(codes).elements() == rows * i64::from(width) {
        // One code per element: gather, convert, scale.
        let codes = cx.reshape(codes, &[rows, i64::from(width)])?;
        let code = cx.take_rows(codes, ids)?;
        let code = cx.convert(code, Elem::F32);
        let code = cx.reshape(code, &[n, groups, i64::from(table.group)])?;
        return affine_scale(cx, table, ids, code, rows, groups, n, width);
    }
    let units = cx.ty(codes).elements() / rows;
    let codes = cx.reshape(codes, &[rows / fold, fold * units])?;
    let codes = take_folded(cx, codes, ids, fold, units)?;
    let elem = cx.elem(codes);
    let unsigned = match elem {
        Elem::I8 => Elem::U8,
        Elem::I16 => Elem::U16,
        Elem::I32 => Elem::U32,
        Elem::I64 => Elem::U64,
        e => e,
    };
    let codes = if unsigned == elem {
        codes
    } else {
        cx.bitcast(codes, unsigned)?
    };
    let per = i64::from(unsigned.bits() / table.bits);
    let dims = [n, units, per];
    let codes = cx.broadcast(codes, &dims, &[0, 1])?;
    let lane = cx.iota(unsigned, &dims, 2);
    let step = cx.const_i(unsigned, i64::from(table.bits), &dims);
    let shift = cx.mul(lane, step)?;
    let shifted = cx.shr(codes, shift)?;
    let mask = cx.const_i(unsigned, (1i64 << table.bits) - 1, &dims);
    let code = cx.and(shifted, mask)?;
    let code = cx.convert(code, Elem::F32);
    let code = cx.reshape(code, &[n, groups, i64::from(table.group)])?;
    affine_scale(cx, table, ids, code, rows, groups, n, width)
}

/// `scale · code + bias` for gathered codes `[n, G, group]` (f32).
#[allow(clippy::too_many_arguments)]
fn affine_scale(
    cx: &mut Cx<'_>,
    table: &Bank,
    ids: Val,
    code: Val,
    rows: i64,
    groups: i64,
    n: i64,
    width: u32,
) -> Result<Val, Error> {
    let factor = |cx: &mut Cx<'_>, plane: Tensor| -> Result<Val, Error> {
        // Each plane states its own fold (its stored row over the groups).
        let fold = if plane.width > 0
            && i64::from(plane.width) % groups == 0
            && i64::from(plane.rows) * i64::from(plane.width) == rows * groups
        {
            i64::from(plane.width) / groups
        } else {
            1
        };
        let v = cx.read(plane)?;
        let v = cx.reshape(v, &[rows / fold, fold * groups])?;
        let v = take_folded(cx, v, ids, fold, groups)?;
        let v = cx.convert(v, Elem::F32);
        Ok(cx.broadcast(v, &[n, groups, i64::from(table.group)], &[0, 1])?)
    };
    let s = factor(cx, table.scales)?;
    let b = factor(cx, table.biases.unwrap_or(table.scales))?;
    let v = cx.mul(code, s)?;
    let v = cx.add(v, b)?;
    Ok(cx.reshape(v, &[n, i64::from(width)])?)
}

fn gather_affine(
    ctx: &Ctx<'_>,
    op: &'static str,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
    slices: u32,
    width: u32,
) -> Result<(), Error> {
    i32_ids(op, "token ids", ids)?;
    nonzero(op, "the row count this embedding table states", vocab)?;
    if y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op, dtype: y.dtype });
    }
    bank_shape(op, &table, width)?;
    ctx.emit(&mut |cx| {
        let ids = index_vec(cx, op, ids, slices)?;
        let (ids, ok) = guarded(cx, ids, vocab.min(table_rows(&table, width)))?;
        let rows = affine_rows(cx, &table, ids, width)?;
        let rows = zero_rows(cx, rows, ok)?;
        cx.write(y, rows)
    })
}

/// [`embed`] over an affine-quantized table (the `Bank`'s codes are the
/// checkpoint's packed words, handed as a `U32` or `U8` plane — or any plain
/// integer plane whose row packs `width * bits` bits; scales and biases are
/// one value per group per row). An id outside the vocabulary lands zeros, as
/// `embed_gather.wgsl` writes them.
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    gather_affine(ctx, OP, ids, table, vocab, y, y.rows, y.width)
}

pub fn embed_concat_mb_4bit(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    let (heads, width) = concat_shape(OP, ids, y)?;
    gather_affine(ctx, OP, ids, table, vocab, y, y.rows * heads, width)
}

/// `y[n] = Σ_t weights[n, t] · table[ids[n, t]]`, accumulated in f32 in tap
/// order; an id outside the vocabulary reads row 0.
pub fn embed_weighted(
    ctx: &Ctx<'_>,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_weighted";
    i32_ids(OP, "taps", ids)?;
    same_dtype(OP, table, y)?;
    if weights.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the interpolation weights are {:?}, and this gather reads f32",
                weights.dtype
            ),
        ));
    }
    if ids.rows != weights.rows || ids.width != weights.width {
        return Err(refuse(
            OP,
            format!(
                "{} x {} taps and {} x {} weights; every tap is weighted and every weight taps",
                ids.rows, ids.width, weights.rows, weights.width
            ),
        ));
    }
    if ids.rows != y.rows || table.width != y.width {
        return Err(refuse(
            OP,
            format!(
                "{} rows of taps into a {}x{} landing from {}-wide table rows",
                ids.rows, y.rows, y.width, table.width
            ),
        ));
    }
    let taps = nonzero(OP, "the taps per row", ids.width)?;
    nonzero(OP, "the table's row count", vocab)?;
    ctx.emit(&mut |cx| {
        let (n, width) = (i64::from(y.rows), i64::from(y.width));
        let t = cx.read(table)?;
        let id = cx.read(ids)?;
        let (id, _) = guarded(cx, id, vocab.min(table.rows))?;
        let w = cx.read(weights)?;
        let mut acc = cx.const_f(Elem::F32, 0.0, &[n, width]);
        for tap in 0..i64::from(taps) {
            let i = cx.slice_axis(id, 1, tap, tap + 1)?;
            let i = cx.reshape(i, &[n])?;
            let row = cx.take_rows(t, i)?;
            let row = cx.convert(row, Elem::F32);
            let wt = cx.slice_axis(w, 1, tap, tap + 1)?;
            let wt = cx.reshape(wt, &[n])?;
            let wt = cx.broadcast(wt, &[n, width], &[0])?;
            let term = cx.mul(row, wt)?;
            acc = cx.add(acc, term)?;
        }
        cx.write(y, acc)
    })
}

// ------------------------------------------------------------------ splits

pub fn split_qkv(
    ctx: &Ctx<'_>,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_qkv";
    if u64::from(q_width) + 2 * u64::from(kv_width) != u64::from(packed.width) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide packed row is not q {q_width} + 2 x kv {kv_width}",
                packed.width
            ),
        ));
    }
    for (what, t, w) in [("q", q, q_width), ("k", k, kv_width), ("v", v, kv_width)] {
        same_dtype(OP, packed, t)?;
        if t.width != w || t.rows != packed.rows {
            return Err(refuse(
                OP,
                format!(
                    "the {what} landing is {}x{}, and the cut is {}x{w}",
                    t.rows, t.width, packed.rows
                ),
            ));
        }
    }
    ctx.emit(&mut |cx| {
        let p = cx.read(packed)?;
        let (qw, kw) = (i64::from(q_width), i64::from(kv_width));
        let qv = cx.slice_axis(p, 1, 0, qw)?;
        let kv = cx.slice_axis(p, 1, qw, qw + kw)?;
        let vv = cx.slice_axis(p, 1, qw + kw, qw + 2 * kw)?;
        cx.write(q, qv)?;
        cx.write(k, kv)?;
        cx.write(v, vv)
    })
}

/// Cuts a `[q_h ‖ gate_h]`-per-head row into its query and gate halves.
pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_q_gate";
    nonzero(OP, "the head width this cut walks", head_dim)?;
    if q.width == 0 || !q.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide query half does not divide by the stated head width {head_dim}",
                q.width
            ),
        ));
    }
    if u64::from(packed.width) != 2 * u64::from(q.width) {
        return Err(refuse(
            OP,
            format!(
                "the packed row is {} wide, and q + gate is {}",
                packed.width,
                2 * q.width
            ),
        ));
    }
    for t in [q, gate] {
        same_dtype(OP, packed, t)?;
        if t.width != q.width || t.rows != packed.rows {
            return Err(refuse(OP, "the query and gate landings are the cut's half"));
        }
    }
    ctx.emit(&mut |cx| {
        let rows = i64::from(packed.rows);
        let hd = i64::from(head_dim);
        let heads = i64::from(q.width / head_dim);
        let p = cx.read(packed)?;
        let p = cx.reshape(p, &[rows, heads, 2 * hd])?;
        let qv = cx.slice_axis(p, 2, 0, hd)?;
        let gv = cx.slice_axis(p, 2, hd, 2 * hd)?;
        cx.write(q, qv)?;
        cx.write(gate, gv)
    })
}

pub fn split_rows(
    ctx: &Ctx<'_>,
    x: Tensor,
    width: u32,
    left: Tensor,
    right: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_rows";
    nonzero(OP, "the left half of this cut", left.width)?;
    nonzero(OP, "the right half of this cut", right.width)?;
    if left.width != width {
        return Err(refuse(
            OP,
            format!(
                "the left half is {} wide, and the cut states {width}",
                left.width
            ),
        ));
    }
    if u64::from(left.width) + u64::from(right.width) != u64::from(x.width) {
        return Err(refuse(
            OP,
            format!(
                "the halves {} + {} do not cover the {}-wide packed row",
                left.width, right.width, x.width
            ),
        ));
    }
    for t in [left, right] {
        same_dtype(OP, x, t)?;
        if t.rows != x.rows {
            return Err(refuse(
                OP,
                format!("{} rows cut into a {}-row half", x.rows, t.rows),
            ));
        }
    }
    ctx.emit(&mut |cx| {
        let v = cx.read(x)?;
        let w = i64::from(width);
        let l = cx.slice_axis(v, 1, 0, w)?;
        let r = cx.slice_axis(v, 1, w, i64::from(x.width))?;
        cx.write(left, l)?;
        cx.write(right, r)
    })
}

/// `y = table[:, layer * width .. (layer + 1) * width]` over `y`'s rows.
pub fn select(
    ctx: &Ctx<'_>,
    table: Tensor,
    layer: u32,
    width: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.select";
    nonzero(OP, "the slice width this select states", width)?;
    same_dtype(OP, table, y)?;
    if y.width != width {
        return Err(refuse(
            OP,
            format!(
                "the landing is {} wide, and the select states {width}",
                y.width
            ),
        ));
    }
    let offset = u64::from(layer) * u64::from(width);
    if offset + u64::from(width) > u64::from(table.width) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide relayed row does not reach layer {layer}'s slice at {offset}",
                table.width
            ),
        ));
    }
    if y.rows > table.rows {
        return Err(refuse(
            OP,
            format!("{} rows selected from {}", y.rows, table.rows),
        ));
    }
    ctx.emit(&mut |cx| {
        let t = cx.read(table)?;
        let t = top_rows(cx, t, y.rows)?;
        let s = cx.slice_axis(t, 1, offset as i64, (offset + u64::from(width)) as i64)?;
        cx.write(y, s)
    })
}

// ------------------------------------------------------------- row moves

fn move_shape(op: &'static str, wide: Tensor, tight: Tensor, index: Tensor) -> Result<(), Error> {
    i32_ids(op, "row map", index)?;
    if index.elements() < u64::from(tight.rows) {
        return Err(refuse(
            op,
            format!(
                "{} rows to move and {} rows named",
                tight.rows,
                index.elements()
            ),
        ));
    }
    if wide.dtype != tight.dtype || wide.width != tight.width {
        return Err(refuse(
            op,
            format!(
                "the fire-wide rectangle is {} x {:?} and the compacted one {} x {:?}; a row copy \
                 does not reshape",
                wide.width, wide.dtype, tight.width, tight.dtype
            ),
        ));
    }
    Ok(())
}

/// `tight[i] = wide[index[i]]`. An index outside `wide` is clamped into it
/// (XLA's gather rule); the GPU reads past the plane there.
pub fn gather_rows(ctx: &Ctx<'_>, wide: Tensor, index: Tensor, tight: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.gather_rows";
    move_shape(OP, wide, tight, index)?;
    ctx.emit(&mut |cx| {
        let ids = index_vec(cx, OP, index, tight.rows)?;
        let w = cx.read(wide)?;
        let rows = cx.take_rows(w, ids)?;
        cx.write(tight, rows)
    })
}

fn scatter_into(
    cx: &mut Cx<'_>,
    op: &'static str,
    src: Tensor,
    index: Tensor,
    y: Tensor,
    live_only: bool,
) -> Result<(), Error> {
    let ids = index_vec(cx, op, index, src.rows)?;
    let ids = if live_only {
        // A negative route is a dead row: send it out of range, where the
        // scatter drops it.
        let zero = cx.like_i(ids, 0);
        let live = cx.compare(Cmp::Ge, ids, zero)?;
        let far = cx.like_i(ids, NONE);
        cx.select(live, ids, far)?
    } else {
        ids
    };
    let s = cx.read(src)?;
    let old = cx.read(y)?;
    let out = cx.put_rows(old, ids, s, crate::hlo::Combine::Set)?;
    cx.write(y, out)
}

/// `wide[index[i]] = tight[i]`; `wide` keeps every row not named. An index
/// outside `wide` is dropped.
pub fn scatter_rows(
    ctx: &Ctx<'_>,
    tight: Tensor,
    index: Tensor,
    wide: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_rows";
    move_shape(OP, wide, tight, index)?;
    ctx.emit(&mut |cx| scatter_into(cx, OP, tight, index, wide, false))
}

/// [`scatter_rows`] where a negative route skips its row.
pub fn scatter_live_rows(
    ctx: &Ctx<'_>,
    src: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_live_rows";
    move_shape(OP, y, src, routes)?;
    ctx.emit(&mut |cx| scatter_into(cx, OP, src, routes, y, true))
}

/// `y[i] = x[perm[i]]` for every row of `y`. Read from kernels-metal
/// `layout::pack_rows` (`row_gather.metal`).
pub fn pack_rows(ctx: &Ctx<'_>, x: Tensor, perm: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.pack_rows";
    move_shape(OP, x, y, perm)?;
    nonzero(OP, "rows to move", y.rows)?;
    ctx.emit(&mut |cx| {
        let ids = index_vec(cx, OP, perm, y.rows)?;
        let v = cx.read(x)?;
        let rows = cx.take_rows(v, ids)?;
        cx.write(y, rows)
    })
}

/// `y[perm[i]] = x[i]` for every row of `x`; `y`'s other rows keep what they
/// held. Read from kernels-metal `layout::unpack_rows` (`row_scatter`).
pub fn unpack_rows(ctx: &Ctx<'_>, x: Tensor, perm: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.unpack_rows";
    move_shape(OP, y, x, perm)?;
    nonzero(OP, "rows to move", x.rows)?;
    ctx.emit(&mut |cx| scatter_into(cx, OP, x, perm, y, false))
}

// ------------------------------------------------------------------ folds

fn fold_extent(op: &'static str, x: Tensor, side: u32) -> Result<(u32, u32), Error> {
    nonzero(op, "the folding square's side", side)?;
    nonzero(op, "the folded row's width", x.width)?;
    let block = side
        .checked_mul(side)
        .ok_or_else(|| refuse(op, format!("a {side}-wide folding square overflows u32")))?;
    if x.rows < block {
        return Err(refuse(
            op,
            format!("{} rows do not fill one {side}x{side} fold", x.rows),
        ));
    }
    Ok((block, x.rows / block))
}

/// `y[o] = mean(x[o * side² .. (o + 1) * side²])`, in f32; `y`'s rows past the
/// pooled count keep what they held.
pub fn pool_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.pool_rows";
    same_dtype(OP, x, y)?;
    let (block, out) = fold_extent(OP, x, side)?;
    if y.width != x.width || y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} rows of {} pool into {out}x{}, and the landing is {}x{}",
                x.rows, x.width, x.width, y.rows, y.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let (b, o, w) = (i64::from(block), i64::from(out), i64::from(x.width));
        let v = cx.slice_axis(v, 0, 0, o * b)?;
        let v = cx.reshape(v, &[o, b, w])?;
        let s = cx.reduce(v, &[1], crate::hlo::Fold::Sum)?;
        let s = cx.scale(s, f64::from(1.0 / block as f32))?;
        write_head(cx, y, s)
    })
}

/// `y[o] = x[o * side²] ‖ … ‖ x[(o + 1) * side² - 1]`.
pub fn merge_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.merge_rows";
    same_dtype(OP, x, y)?;
    let (block, out) = fold_extent(OP, x, side)?;
    let merged = u64::from(block) * u64::from(x.width);
    if u64::from(y.width) != merged || y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{block} rows of {} merge into {out}x{merged}, and the landing is {}x{}",
                x.width, y.rows, y.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read(x)?;
        let o = i64::from(out);
        let v = cx.slice_axis(v, 0, 0, o * i64::from(block))?;
        let v = cx.reshape(v, &[o, merged as i64])?;
        write_head(cx, y, v)
    })
}

// ---------------------------------------------------------------- ranking

/// The best column of each row of an f32 `[rows, width]` — largest value,
/// lowest index on a tie, NaN never picked, nor any column `skip` marks —
/// as `(index, value)`, `[rows]` each. A row with nothing to pick answers
/// index [`NONE`].
fn best(cx: &mut Cx<'_>, x: Val, skip: Option<Val>) -> Result<(Val, Val), Error> {
    let dims = cx.dims(x).to_vec();
    let col = cx.iota(Elem::I32, &dims, 1);
    let mut ok = cx.compare(Cmp::Eq, x, x)?;
    if let Some(skip) = skip {
        let keep = cx.not(skip);
        ok = cx.and(ok, keep)?;
    }
    let none = cx.like_i(col, NONE);
    let idx = cx.select(ok, col, none)?;
    let lo = cx.const_f(Elem::F32, f64::NEG_INFINITY, &[]);
    let nil = cx.const_i(Elem::I32, NONE, &[]);
    let out = cx.reduce_with(&[x, idx], &[lo, nil], &[1], |f, a, b| {
        let (av, ai, bv, bi) = (a[0], a[1], b[0], b[1]);
        let n = f.const_i(Elem::I32, NONE, &[]);
        let a_ok = f.compare(Cmp::Ne, ai, n)?;
        let b_none = f.compare(Cmp::Eq, bi, n)?;
        let gt = f.compare(Cmp::Gt, av, bv)?;
        let eq = f.compare(Cmp::Eq, av, bv)?;
        let lt = f.compare(Cmp::Lt, ai, bi)?;
        let tie = f.and(eq, lt)?;
        let wins = f.or(gt, tie)?;
        let wins = f.or(b_none, wins)?;
        let take = f.and(a_ok, wins)?;
        let v = f.select(take, av, bv)?;
        let i = f.select(take, ai, bi)?;
        Ok(vec![v, i])
    })?;
    Ok((out[1], out[0]))
}

/// Writes the index of each row's largest value into column `column` of the
/// i32 `y`; ties go to the lowest index, NaN is never picked, an all-NaN row
/// answers 0.
pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.argmax";
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::F16) {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    if y.dtype != Dtype::I32 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: y.dtype,
        });
    }
    nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    if x.rows != y.rows {
        return Err(refuse(
            OP,
            format!("{} rows ranked into {} rows", x.rows, y.rows),
        ));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let (i, _) = best(cx, v, None)?;
        let none = cx.like_i(i, NONE);
        let hit = cx.compare(Cmp::Ne, i, none)?;
        let zero = cx.like_i(i, 0);
        let i = cx.select(hit, i, zero)?;
        let i = cx.reshape(i, &[i64::from(x.rows), 1])?;
        if y.width == 1 {
            return cx.write(y, i);
        }
        let old = cx.read(y)?;
        let r = cx.const_i(Elem::I32, 0, &[]);
        let c = cx.const_i(Elem::I32, i64::from(column), &[]);
        let merged = cx.dynamic_update_slice(old, i, &[r, c])?;
        cx.write(y, merged)
    })
}

/// The `k` largest values of each row, descending, with their columns; ties
/// go to the lower column, NaN is never picked, and a slot no value fills
/// answers value 0 at column 0. Values land f32, indices i32. Read from
/// kernels-metal `layout::topk` (`topk.metal`); the GPU stamps k = 8 and 16,
/// this takes any `k` up to the row width.
pub fn topk(
    ctx: &Ctx<'_>,
    x: Tensor,
    k: u32,
    values: Tensor,
    indices: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.topk";
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F32 | Dtype::F16) {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    nonzero(OP, "k", k)?;
    if k > x.width {
        return Err(refuse(
            OP,
            format!("k = {k} is wider than the {}-wide row", x.width),
        ));
    }
    if values.rows != rows || values.width != k || values.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!("the values plane is not [{rows}, {k}] f32"),
        ));
    }
    if indices.rows != rows || indices.width != k || indices.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!("the indices plane is not [{rows}, {k}] i32"),
        ));
    }
    ctx.emit(&mut |cx| {
        let r = i64::from(rows);
        let dims = [r, i64::from(x.width)];
        let v = cx.read_f32(x)?;
        let col = cx.iota(Elem::I32, &dims, 1);
        let mut taken = cx.const_i(Elem::Pred, 0, &dims);
        let (mut vs, mut is) = (Vec::new(), Vec::new());
        for _ in 0..k {
            let (i, m) = best(cx, v, Some(taken))?;
            let wide = cx.broadcast(i, &dims, &[0])?;
            let hit = cx.compare(Cmp::Eq, col, wide)?;
            taken = cx.or(taken, hit)?;
            let none = cx.like_i(i, NONE);
            let found = cx.compare(Cmp::Ne, i, none)?;
            let zi = cx.like_i(i, 0);
            let zv = cx.like_f(m, 0.0);
            let i = cx.select(found, i, zi)?;
            let m = cx.select(found, m, zv)?;
            is.push(cx.reshape(i, &[r, 1])?);
            vs.push(cx.reshape(m, &[r, 1])?);
        }
        let vs = cx.concat(&vs, 1)?;
        let is = cx.concat(&is, 1)?;
        cx.write(values, vs)?;
        cx.write(indices, is)
    })
}
