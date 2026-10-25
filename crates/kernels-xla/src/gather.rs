//! Row and column movement: embeddings, splits, row gathers and scatters,
//! folds. Argmax and top-k are `sampling`; n-gram ids are [`ngram`].
//!
//! GPU-only constraints (even widths so a bf16 pair moves as a word, the
//! lanes a workgroup holds) are dropped: a traced slice has no word size.
//! Pure moves take any plain dtype, and refuse only when source and landing
//! disagree, since a move never converts.

#![allow(clippy::too_many_arguments)]

pub mod ngram;

use dtype::Dtype;

use crate::check::elem_of;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Func, Val};
use crate::tensor::{Bank, Landing, Operand, Stored};

/// The index a best-of reduction carries for "nothing picked yet".
pub(crate) const NONE: i64 = i32::MAX as i64;

pub(crate) fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("{what} is zero")));
    }
    Ok(v)
}

fn same_dtype(op: &'static str, a: impl Stored, b: impl Stored) -> Result<(), Error> {
    if a.dtype() != b.dtype() {
        return Err(refuse(
            op,
            format!(
                "the source is {:?} and the landing {:?}; a move does not convert",
                a.dtype(),
                b.dtype()
            ),
        ));
    }
    Ok(())
}

fn i32_ids(op: &'static str, what: &str, t: impl Stored) -> Result<(), Error> {
    if t.dtype() != Dtype::I32 {
        return Err(refuse(
            op,
            format!("the {what} are {:?}, and this op reads i32", t.dtype()),
        ));
    }
    Ok(())
}

/// `t` as a flat i32 vector of its `n` first elements.
fn index_vec(f: &mut Func, op: &'static str, t: Operand, n: u32) -> Result<Val, Error> {
    if t.elements() < u64::from(n) {
        return Err(refuse(
            op,
            format!(
                "{}x{} indices name fewer than the {n} rows moved",
                t.rows, t.width
            ),
        ));
    }
    let v = t.val;
    let flat = f.reshape(v, &[t.elements() as i64])?;
    let flat = f.slice_axis(flat, 0, 0, i64::from(n))?;
    Ok(f.convert(flat, Elem::I32))
}

/// `0 <= ids < vocab`, elementwise.
fn in_vocab(f: &mut Func, ids: Val, vocab: u32) -> Result<Val, Error> {
    let zero = f.like_i(ids, 0);
    let top = f.like_i(ids, i64::from(vocab));
    let lo = f.compare(Cmp::Ge, ids, zero)?;
    let hi = f.compare(Cmp::Lt, ids, top)?;
    Ok(f.and(lo, hi)?)
}

/// `ids` with every id outside `[0, vocab)` sent to row 0, as the embed
/// kernels guard their read.
fn guarded(f: &mut Func, ids: Val, vocab: u32) -> Result<(Val, Val), Error> {
    let ok = in_vocab(f, ids, vocab)?;
    let zero = f.like_i(ids, 0);
    Ok((f.select(ok, ids, zero)?, ok))
}

/// Zeroes the rows of `[n, width]` whose `ok[n]` is false.
fn zero_rows(f: &mut Func, rows: Val, ok: Val) -> Result<Val, Error> {
    let dims = f.dims(rows).to_vec();
    let ok = f.broadcast(ok, &dims, &[0])?;
    let zero = f.like_f(rows, 0.0);
    Ok(f.select(ok, rows, zero)?)
}

/// The first `rows` rows of a table value.
fn top_rows(f: &mut Func, v: Val, rows: u32) -> Result<Val, Error> {
    Ok(f.slice_axis(v, 0, 0, i64::from(rows))?)
}

// ------------------------------------------------------------------ embeds

pub fn embed(
    f: &mut Func,
    ids: Operand,
    table: Operand,
    vocab: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.embed";
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
    let ids = index_vec(f, OP, ids, y.rows)?;
    let (ids, _) = guarded(f, ids, vocab.min(table.rows))?;
    let t = table.val;
    let rows = f.take_rows(t, ids)?;
    Ok(rows)
}

/// The vocab-parallel embed: this rank holds table rows `[rank * local,
/// (rank + 1) * local)`, answers them, and zeroes every other id so an
/// all-reduce sums the shards. Read from kernels-cuda
/// `gather::embed_vocab_shard` (`kernels/gather/embed.cu`), whose rank comes
/// from the communicator; here it is stated.
pub fn embed_vocab_shard(
    f: &mut Func,
    ids: Operand,
    table: Operand,
    rank: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.embed_vocab_shard";
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
    let ids = index_vec(f, OP, ids, y.rows)?;
    let off = f.like_i(ids, offset);
    let local_ids = f.sub(ids, off)?;
    let (local_ids, ok) = guarded(f, local_ids, local)?;
    let t = table.val;
    let rows = f.take_rows(t, local_ids)?;
    let rows = zero_rows(f, rows, ok)?;
    Ok(rows)
}

fn concat_shape(op: &'static str, ids: impl Stored, y: impl Stored) -> Result<(u32, u32), Error> {
    let heads = nonzero(op, "the ids per row", ids.width())?;
    if y.width() == 0 || !y.width().is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide landing is not {heads} table rows side by side",
                y.width()
            ),
        ));
    }
    if ids.rows() != y.rows() {
        return Err(refuse(
            op,
            format!("{} rows of ids and {} rows landed", ids.rows(), y.rows()),
        ));
    }
    Ok((heads, y.width() / heads))
}

/// `y[n] = table[ids[n, 0]] ‖ table[ids[n, 1]] ‖ …`; an id outside the
/// vocabulary lands a zero slice.
pub fn embed_concat(
    f: &mut Func,
    ids: Operand,
    table: Operand,
    vocab: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.embed";
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
    let n = y.rows * heads;
    let ids = index_vec(f, OP, ids, n)?;
    let (ids, ok) = guarded(f, ids, vocab.min(table.rows))?;
    let t = table.val;
    let rows = f.take_rows(t, ids)?;
    let rows = zero_rows(f, rows, ok)?;
    Ok(rows)
}

/// Checks a quantized table and answers `(bits per code, storage elem)`.
fn bank_shape(op: &'static str, table: &Bank<impl Stored>, width: u32) -> Result<Elem, Error> {
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
    if crate::pack::code_elem(table.codes.dtype()).is_some() {
        // A bank's codes (landed one per element, `crate::pack`, or as the
        // raw bytes of their rows: the read value tells).
        if crate::pack::code_bits(table.codes.dtype()) != Some(table.bits) {
            return Err(refuse(
                op,
                format!(
                    "the code plane is {:?} {} wide, and a {width}-wide row of {}-bit codes \
                     wants {width} codes",
                    table.codes.dtype(),
                    table.codes.width(),
                    table.bits
                ),
            ));
        }
        return scale_planes(
            op,
            table,
            width,
            crate::pack::code_elem(table.codes.dtype()).unwrap_or(Elem::U8),
        );
    }
    let elem = elem_of(op, table.codes.dtype())?;
    if !elem.is_int() {
        return Err(refuse(
            op,
            format!(
                "the code plane is {:?}; codes are packed in an integer plane",
                table.codes.dtype()
            ),
        ));
    }
    let bits = elem.bits();
    let fold = u64::from(folding(table, width));
    if u64::from(table.codes.width()) * u64::from(bits)
        != fold * u64::from(width) * u64::from(table.bits)
    {
        return Err(refuse(
            op,
            format!(
                "the code plane rows are {} x {bits}-bit, and a {width}-wide row of {}-bit codes \
                 packs into {} bits",
                table.codes.width(),
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
pub(crate) fn folding(table: &Bank<impl Stored>, width: u32) -> u32 {
    let row = u64::from(width) * u64::from(table.bits) / 8;
    if table.codes.dtype() != Dtype::U8 || row == 0 {
        return 1;
    }
    let w = u64::from(table.codes.width());
    if w > row && w.is_multiple_of(row) {
        u32::try_from(w / row).unwrap_or(1)
    } else {
        1
    }
}

/// The table rows a quantized table holds (its stored rows unfolded).
#[must_use]
pub fn table_rows(table: &Bank<impl Stored>, width: u32) -> u32 {
    table.codes.rows().saturating_mul(folding(table, width))
}

/// Rows `ids` (i32 `[n]`) of a table `[R, f·w]` that folds `f` rows of `w`
/// into each stored row: `[n, w]`. Takes each id's whole stored row (a
/// plain row gather; a gather of `[1, w]` windows at two start indices
/// runs one slice per id on TPU), then keeps its `w`-wide part.
fn take_folded(f: &mut Func, plane: Val, ids: Val, fold: i64, w: i64) -> Result<Val, Error> {
    if fold == 1 {
        return Ok(f.take_rows(plane, ids)?);
    }
    let n = f.dims(ids)[0];
    let fq = f.like_i(ids, fold);
    let hi = f.div(ids, fq)?;
    let lo = f.rem(ids, fq)?;
    let whole = f.take_rows(plane, hi)?;
    let whole = f.reshape(whole, &[n, fold, w])?;
    let at = f.iota(Elem::I32, &[n, fold, w], 1);
    let lo = f.broadcast(lo, &[n, fold, w], &[0])?;
    let hit = f.compare(Cmp::Eq, at, lo)?;
    let elem = f.elem(whole);
    let zero = if elem.is_int() {
        f.const_i(elem, 0, &[n, fold, w])
    } else {
        f.const_f(elem, 0.0, &[n, fold, w])
    };
    let kept = f.select(hit, whole, zero)?;
    // Exactly one term per output is kept: an integer or float sum of it
    // and zeros is the kept value.
    Ok(f.reduce(kept, &[1], crate::hlo::Fold::Sum)?)
}

/// Checks the scale and bias planes hold one value per group per row.
fn scale_planes(
    op: &'static str,
    table: &Bank<impl Stored>,
    width: u32,
    elem: Elem,
) -> Result<Elem, Error> {
    let groups = u64::from(table_rows(table, width)) * u64::from(width / table.group);
    for (what, plane) in [("scale", Some(table.scales)), ("bias", table.biases)] {
        let plane = plane.unwrap_or(table.scales);
        if plane.elements() != groups {
            return Err(refuse(
                op,
                format!(
                    "the {what} plane holds {} values, and {} rows of {} groups want {groups}",
                    plane.elements(),
                    table.codes.rows(),
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
    f: &mut Func,
    table: &Bank<Operand>,
    ids: Val,
    width: u32,
) -> Result<Val, Error> {
    let n = f.dims(ids)[0];
    let fold = i64::from(folding(table, width));
    let rows = i64::from(table.codes.rows) * fold;
    let groups = i64::from(width / table.group);
    let codes = table.codes.val;
    if f.ty(codes).elements() == rows * i64::from(width) {
        // One code per element: gather, convert, scale.
        let codes = f.reshape(codes, &[rows, i64::from(width)])?;
        let code = f.take_rows(codes, ids)?;
        let code = f.convert(code, Elem::F32);
        let code = f.reshape(code, &[n, groups, i64::from(table.group)])?;
        return affine_scale(f, table, ids, code, rows, groups, n, width);
    }
    let units = f.ty(codes).elements() / rows;
    let codes = f.reshape(codes, &[rows / fold, fold * units])?;
    let codes = take_folded(f, codes, ids, fold, units)?;
    let elem = f.elem(codes);
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
        f.bitcast(codes, unsigned)?
    };
    let per = i64::from(unsigned.bits() / table.bits);
    let dims = [n, units, per];
    let codes = f.broadcast(codes, &dims, &[0, 1])?;
    let lane = f.iota(unsigned, &dims, 2);
    let step = f.const_i(unsigned, i64::from(table.bits), &dims);
    let shift = f.mul(lane, step)?;
    let shifted = f.shr(codes, shift)?;
    let mask = f.const_i(unsigned, (1i64 << table.bits) - 1, &dims);
    let code = f.and(shifted, mask)?;
    let code = f.convert(code, Elem::F32);
    let code = f.reshape(code, &[n, groups, i64::from(table.group)])?;
    affine_scale(f, table, ids, code, rows, groups, n, width)
}

/// `scale · code + bias` for gathered codes `[n, G, group]` (f32).
#[allow(clippy::too_many_arguments)]
fn affine_scale(
    f: &mut Func,
    table: &Bank<Operand>,
    ids: Val,
    code: Val,
    rows: i64,
    groups: i64,
    n: i64,
    width: u32,
) -> Result<Val, Error> {
    let factor = |f: &mut Func, plane: Operand| -> Result<Val, Error> {
        // Each plane states its own fold (its stored row over the groups).
        let fold = if plane.width > 0
            && i64::from(plane.width) % groups == 0
            && i64::from(plane.rows) * i64::from(plane.width) == rows * groups
        {
            i64::from(plane.width) / groups
        } else {
            1
        };
        let v = plane.val;
        let v = f.reshape(v, &[rows / fold, fold * groups])?;
        let v = take_folded(f, v, ids, fold, groups)?;
        let v = f.convert(v, Elem::F32);
        Ok(f.broadcast(v, &[n, groups, i64::from(table.group)], &[0, 1])?)
    };
    let s = factor(f, table.scales)?;
    let b = factor(f, table.biases.unwrap_or(table.scales))?;
    let v = f.mul(code, s)?;
    let v = f.add(v, b)?;
    Ok(f.reshape(v, &[n, i64::from(width)])?)
}

fn gather_affine(
    f: &mut Func,
    op: &'static str,
    ids: Operand,
    table: Bank<Operand>,
    vocab: u32,
    y: Landing,
    slices: u32,
    width: u32,
) -> Result<Val, Error> {
    i32_ids(op, "token ids", ids)?;
    nonzero(op, "the row count this embedding table states", vocab)?;
    if y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op, dtype: y.dtype });
    }
    bank_shape(op, &table, width)?;
    let ids = index_vec(f, op, ids, slices)?;
    let (ids, ok) = guarded(f, ids, vocab.min(table_rows(&table, width)))?;
    let rows = affine_rows(f, &table, ids, width)?;
    let rows = zero_rows(f, rows, ok)?;
    Ok(rows)
}

/// [`embed`] over an affine-quantized table (the `Bank`'s codes are the
/// checkpoint's packed words, handed as a `U32` or `U8` plane — or any plain
/// integer plane whose row packs `width * bits` bits; scales and biases are
/// one value per group per row). An id outside the vocabulary lands zeros, as
/// `embed_gather.wgsl` writes them.
pub fn embed_gather_mb_4bit(
    f: &mut Func,
    ids: Operand,
    table: Bank<Operand>,
    vocab: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.embed";
    gather_affine(f, OP, ids, table, vocab, y, y.rows, y.width)
}

pub fn embed_concat_mb_4bit(
    f: &mut Func,
    ids: Operand,
    table: Bank<Operand>,
    vocab: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.embed";
    let (heads, width) = concat_shape(OP, ids, y)?;
    gather_affine(f, OP, ids, table, vocab, y, y.rows * heads, width)
}

/// `y[n] = Σ_t weights[n, t] · table[ids[n, t]]`, accumulated in f32 in tap
/// order; an id outside the vocabulary reads row 0.
pub fn embed_weighted(
    f: &mut Func,
    ids: Operand,
    weights: Operand,
    table: Operand,
    vocab: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.embed";
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
    let (n, width) = (i64::from(y.rows), i64::from(y.width));
    let t = table.val;
    let id = ids.val;
    let (id, _) = guarded(f, id, vocab.min(table.rows))?;
    let w = weights.val;
    let mut acc = f.const_f(Elem::F32, 0.0, &[n, width]);
    for tap in 0..i64::from(taps) {
        let i = f.slice_axis(id, 1, tap, tap + 1)?;
        let i = f.reshape(i, &[n])?;
        let row = f.take_rows(t, i)?;
        let row = f.convert(row, Elem::F32);
        let wt = f.slice_axis(w, 1, tap, tap + 1)?;
        let wt = f.reshape(wt, &[n])?;
        let wt = f.broadcast(wt, &[n, width], &[0])?;
        let term = f.mul(row, wt)?;
        acc = f.add(acc, term)?;
    }
    Ok(acc)
}

// ------------------------------------------------------------------ splits

pub fn split_qkv(
    f: &mut Func,
    packed: Operand,
    q_width: u32,
    kv_width: u32,
    q: Landing,
    k: Landing,
    v: Landing,
) -> Result<(Val, Val, Val), Error> {
    const OP: &str = "gather.split_qkv";
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
    let p = packed.val;
    let (qw, kw) = (i64::from(q_width), i64::from(kv_width));
    let qv = f.slice_axis(p, 1, 0, qw)?;
    let kv = f.slice_axis(p, 1, qw, qw + kw)?;
    let vv = f.slice_axis(p, 1, qw + kw, qw + 2 * kw)?;
    Ok((qv, kv, vv))
}

/// Cuts a `[q_h ‖ gate_h]`-per-head row into its query and gate halves.
pub fn split_q_gate(
    f: &mut Func,
    packed: Operand,
    head_dim: u32,
    q: Landing,
    gate: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "gather.split_q_gate";
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
    let rows = i64::from(packed.rows);
    let hd = i64::from(head_dim);
    let heads = i64::from(q.width / head_dim);
    let p = packed.val;
    let p = f.reshape(p, &[rows, heads, 2 * hd])?;
    let qv = f.slice_axis(p, 2, 0, hd)?;
    let gv = f.slice_axis(p, 2, hd, 2 * hd)?;
    Ok((qv, gv))
}

pub fn split_rows(
    f: &mut Func,
    x: Operand,
    width: u32,
    left: Landing,
    right: Landing,
) -> Result<(Val, Val), Error> {
    const OP: &str = "gather.split_rows";
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
    let v = x.val;
    let w = i64::from(width);
    let l = f.slice_axis(v, 1, 0, w)?;
    let r = f.slice_axis(v, 1, w, i64::from(x.width))?;
    Ok((l, r))
}

/// `xs` side by side, a row each's columns after the one before's.
/// kernels-cuda `gather::concat_cols`.
pub fn concat_cols(f: &mut Func, xs: &[Operand], y: Landing) -> Result<Val, Error> {
    const OP: &str = "gather.concat_cols";
    let mut width = 0u64;
    for x in xs {
        same_dtype(OP, *x, y)?;
        if x.rows != y.rows {
            return Err(refuse(
                OP,
                format!("a {}-row input beside {} rows", x.rows, y.rows),
            ));
        }
        width += u64::from(x.width);
    }
    if width != u64::from(y.width) {
        return Err(refuse(
            OP,
            format!(
                "the inputs are {width} wide together and the output {}",
                y.width
            ),
        ));
    }
    let vals: Vec<Val> = xs.iter().map(|x| x.val).collect();
    Ok(f.concat(&vals, 1)?)
}

/// `y = table[:, layer * width .. (layer + 1) * width]` over `y`'s rows.
pub fn select(
    f: &mut Func,
    table: Operand,
    layer: u32,
    width: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.select";
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
    let t = table.val;
    let t = top_rows(f, t, y.rows)?;
    let s = f.slice_axis(t, 1, offset as i64, (offset + u64::from(width)) as i64)?;
    Ok(s)
}

// ------------------------------------------------------------- row moves

fn move_shape(
    op: &'static str,
    wide: impl Stored,
    tight: impl Stored,
    index: impl Stored,
) -> Result<(), Error> {
    i32_ids(op, "row map", index)?;
    if index.elements() < u64::from(tight.rows()) {
        return Err(refuse(
            op,
            format!(
                "{} rows to move and {} rows named",
                tight.rows(),
                index.elements()
            ),
        ));
    }
    if wide.dtype() != tight.dtype() || wide.width() != tight.width() {
        return Err(refuse(
            op,
            format!(
                "the fire-wide rectangle is {} x {:?} and the compacted one {} x {:?}; a row copy \
                 does not reshape",
                wide.width(),
                wide.dtype(),
                tight.width(),
                tight.dtype()
            ),
        ));
    }
    Ok(())
}

/// `tight[i] = wide[index[i]]`. An index outside `wide` is clamped into it
/// (XLA's gather rule); the GPU reads past the plane there.
pub fn gather_rows(
    f: &mut Func,
    wide: Operand,
    index: Operand,
    tight: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.rows";
    move_shape(OP, wide, tight, index)?;
    let ids = index_vec(f, OP, index, tight.rows)?;
    let w = wide.val;
    let rows = f.take_rows(w, ids)?;
    Ok(rows)
}

fn scatter_into(
    f: &mut Func,
    op: &'static str,
    src: Operand,
    index: Operand,
    y: Operand,
    live_only: bool,
) -> Result<Val, Error> {
    let ids = index_vec(f, op, index, src.rows)?;
    let ids = if live_only {
        // A negative route is a dead row: send it out of range, where the
        // scatter drops it.
        let zero = f.like_i(ids, 0);
        let live = f.compare(Cmp::Ge, ids, zero)?;
        let far = f.like_i(ids, NONE);
        f.select(live, ids, far)?
    } else {
        ids
    };
    let s = src.val;
    let old = y.val;
    Ok(f.put_rows(old, ids, s, crate::hlo::Combine::Set)?)
}

/// `wide[index[i]] = tight[i]`; `wide` keeps every row not named. An index
/// outside `wide` is dropped.
pub fn scatter_rows(
    f: &mut Func,
    tight: Operand,
    index: Operand,
    wide: Operand,
) -> Result<Val, Error> {
    const OP: &str = "gather.scatter_rows";
    move_shape(OP, wide, tight, index)?;
    scatter_into(f, OP, tight, index, wide, false)
}

/// [`scatter_rows`] where a negative route skips its row.
pub fn scatter_live_rows(
    f: &mut Func,
    src: Operand,
    routes: Operand,
    y: Operand,
) -> Result<Val, Error> {
    const OP: &str = "gather.scatter_live_rows";
    move_shape(OP, y, src, routes)?;
    scatter_into(f, OP, src, routes, y, true)
}

/// `y[i] = x[perm[i]]` for every row of `y`. Read from kernels-metal
/// `gather::pack_rows`.
pub fn pack_rows(f: &mut Func, x: Operand, perm: Operand, y: Landing) -> Result<Val, Error> {
    const OP: &str = "gather.pack_rows";
    move_shape(OP, x, y, perm)?;
    nonzero(OP, "rows to move", y.rows)?;
    let ids = index_vec(f, OP, perm, y.rows)?;
    let v = x.val;
    let rows = f.take_rows(v, ids)?;
    Ok(rows)
}

/// `y[perm[i]] = x[i]` for every row of `x`; `y`'s other rows keep what they
/// held. Read from kernels-metal `gather::unpack_rows`.
pub fn unpack_rows(f: &mut Func, x: Operand, perm: Operand, y: Operand) -> Result<Val, Error> {
    const OP: &str = "gather.unpack_rows";
    move_shape(OP, y, x, perm)?;
    nonzero(OP, "rows to move", x.rows)?;
    scatter_into(f, OP, x, perm, y, false)
}

// ------------------------------------------------------------------ folds

fn fold_extent(op: &'static str, x: impl Stored, side: u32) -> Result<(u32, u32), Error> {
    nonzero(op, "the folding square's side", side)?;
    nonzero(op, "the folded row's width", x.width())?;
    let block = side
        .checked_mul(side)
        .ok_or_else(|| refuse(op, format!("a {side}-wide folding square overflows u32")))?;
    if x.rows() < block {
        return Err(refuse(
            op,
            format!("{} rows do not fill one {side}x{side} fold", x.rows()),
        ));
    }
    Ok((block, x.rows() / block))
}

/// `y[o] = mean(x[o * side² .. (o + 1) * side²])`, in f32; `y`'s rows past the
/// pooled count keep what they held.
pub fn pool_rows(f: &mut Func, x: Operand, side: u32, y: Landing) -> Result<Val, Error> {
    const OP: &str = "gather.pool_rows";
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
    let v = x.f32(f);
    let (b, o, w) = (i64::from(block), i64::from(out), i64::from(x.width));
    let v = f.slice_axis(v, 0, 0, o * b)?;
    let v = f.reshape(v, &[o, b, w])?;
    let s = f.reduce(v, &[1], crate::hlo::Fold::Sum)?;
    let s = f.scale(s, f64::from(1.0 / block as f32))?;
    Ok(s)
}

/// `y[o] = x[o * side²] ‖ … ‖ x[(o + 1) * side² - 1]`.
pub fn merge_rows(f: &mut Func, x: Operand, side: u32, y: Landing) -> Result<Val, Error> {
    const OP: &str = "gather.merge_rows";
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
    let v = x.val;
    let o = i64::from(out);
    let v = f.slice_axis(v, 0, 0, o * i64::from(block))?;
    let v = f.reshape(v, &[o, merged as i64])?;
    Ok(v)
}

/// Patch rows of raw RGB bytes to the tower's rows: `(v / 255 - mean[c]) /
/// std[c]`, channel-major or pixel-major, the frame repeated `temporal` times.
#[allow(clippy::too_many_arguments)]
pub fn pixels(
    f: &mut Func,
    x: Operand,
    patch: u32,
    mean: [f32; 3],
    std: [f32; 3],
    channel_major: bool,
    temporal: u32,
    y: Landing,
) -> Result<Option<Val>, Error> {
    const OP: &str = "gather.pixels";
    if !matches!(y.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: y.dtype,
        });
    }
    if x.dtype != Dtype::U8 {
        return Err(refuse(
            OP,
            format!("pixel rows are u8 bytes, not {:?}", x.dtype),
        ));
    }
    let in_width = 3 * u64::from(patch) * u64::from(patch);
    if u64::from(x.width) != in_width
        || u64::from(y.width) != u64::from(temporal) * in_width
        || y.rows < x.rows
    {
        return Err(refuse(
            OP,
            format!(
                "{} x {} pixel rows of {patch} x {patch} patches do not land as {} x {} rows \
                 repeated {temporal} times",
                x.rows, x.width, y.rows, y.width
            ),
        ));
    }
    if std.contains(&0.0) {
        return Err(refuse(OP, "a channel's std is zero"));
    }
    nonzero(OP, "the frame count", temporal)?;
    if x.rows == 0 {
        return Ok(None);
    }
    let (n, px, t) = (
        i64::from(x.rows),
        i64::from(patch) * i64::from(patch),
        i64::from(temporal),
    );
    let v = x.f32(f);
    let v = f.reshape(v, &[n, px, 3])?;
    let (v, dims, ch) = if channel_major {
        (f.transpose(v, &[0, 2, 1])?, [n, 3, px], 1)
    } else {
        (v, [n, px, 3], 2)
    };
    let m = f.const_floats(Elem::F32, &mean.map(f64::from), &[3])?;
    let m = f.broadcast(m, &dims, &[ch])?;
    let s = f.const_floats(Elem::F32, &std.map(f64::from), &[3])?;
    let s = f.broadcast(s, &dims, &[ch])?;
    let k = f.const_f(Elem::F32, 255.0, &dims);
    let v = f.div(v, k)?;
    let v = f.sub(v, m)?;
    let v = f.div(v, s)?;
    let v = if channel_major {
        f.broadcast(v, &[n, 3, t, px], &[0, 1, 3])?
    } else {
        f.broadcast(v, &[n, t, px, 3], &[0, 2, 3])?
    };
    let v = f.reshape(v, &[n, i64::from(y.width)])?;
    Ok(Some(v))
}

/// Position-table taps for each patch: bilinear over its image's grid
/// stretched onto a `side x side` table (4 taps), or axes (2 unit taps).
#[allow(clippy::too_many_arguments)]
///
/// `grids` and `segments` are the planes' shapes; their values are read only
/// for the bilinear form (`bilinear`). Returns `(ids, weights)`, or nothing
/// when there is no row or image to tap.
pub fn grid_taps(
    f: &mut Func,
    positions: Operand,
    grids: impl Stored,
    segments: impl Stored,
    bilinear: Option<Bilinear>,
    side: u32,
    ids: Landing,
    weights: Landing,
) -> Result<Option<(Val, Val)>, Error> {
    const OP: &str = "gather.grid_taps";
    let taps: u32 = if bilinear.is_some() { 4 } else { 2 };
    let (grids, segments) = (grids.shape(), segments.shape());
    if positions.dtype != Dtype::I32 || grids.dtype != Dtype::I32 || segments.dtype != Dtype::I32 {
        return Err(refuse(OP, "positions, grids and segments are i32"));
    }
    if ids.dtype != Dtype::I32 || weights.dtype != Dtype::F32 {
        return Err(refuse(OP, "the taps land as i32 ids and f32 weights"));
    }
    if positions.width != 3 || grids.width != 3 {
        return Err(refuse(OP, "positions and grids are three wide"));
    }
    if ids.width != taps
        || weights.width != taps
        || ids.rows < positions.rows
        || weights.rows < positions.rows
    {
        return Err(refuse(
            OP,
            format!(
                "{taps} taps a row, and the destinations are {} x {} and {} x {}",
                ids.rows, ids.width, weights.rows, weights.width
            ),
        ));
    }
    nonzero(OP, "the table side", side)?;
    let images = (segments.elements().saturating_sub(1)).min(u64::from(grids.rows));
    if images == 0 || positions.rows == 0 {
        return Ok(None);
    }
    let (n, img, side) = (i64::from(positions.rows), images as i64, i64::from(side));
    let p = positions.val;
    let row = f.slice_axis(p, 1, 1, 2)?;
    let row = f.reshape(row, &[n])?;
    let col = f.slice_axis(p, 1, 2, 3)?;
    let col = f.reshape(col, &[n])?;
    let (id_cols, w_cols) = if let Some(planes) = bilinear {
        let seg = planes.segments.val;
        let seg = f.reshape(seg, &[segments.elements() as i64])?;
        let seg = f.slice_axis(seg, 0, 0, img)?;
        let seg = f.broadcast(seg, &[n, img], &[1])?;
        let at = f.iota(Elem::I32, &[n, img], 0);
        let past = f.compare(Cmp::Ge, at, seg)?;
        let past = f.convert(past, Elem::I32);
        let which = f.reduce(past, &[1], crate::hlo::Fold::Sum)?;
        let one = f.const_i(Elem::I32, 1, &[n]);
        let which = f.sub(which, one)?;
        let lo = f.const_i(Elem::I32, 0, &[]);
        let hi = f.const_i(Elem::I32, img - 1, &[]);
        let which = f.clamp(lo, which, hi)?;
        let g = planes.grids.val;
        let g = f.take_rows(g, which)?;
        let gh = f.slice_axis(g, 1, 1, 2)?;
        let gh = f.reshape(gh, &[n])?;
        let gw = f.slice_axis(g, 1, 2, 3)?;
        let gw = f.reshape(gw, &[n])?;
        let (ht, hw) = axis_taps(f, row, gh, side)?;
        let (wt, ww) = axis_taps(f, col, gw, side)?;
        let stride = f.const_i(Elem::I32, side, &[n]);
        let (mut id_cols, mut w_cols) = (Vec::new(), Vec::new());
        for a in 0..2 {
            for b in 0..2 {
                let id = f.mul(ht[a], stride)?;
                id_cols.push(f.add(id, wt[b])?);
                w_cols.push(f.mul(hw[a], ww[b])?);
            }
        }
        (id_cols, w_cols)
    } else {
        let last = f.const_i(Elem::I32, side - 1, &[n]);
        let c = f.min(col, last)?;
        let r = f.min(row, last)?;
        let off = f.const_i(Elem::I32, side, &[n]);
        let r = f.add(r, off)?;
        let unit = f.const_f(Elem::F32, 1.0, &[n]);
        (vec![c, r], vec![unit, unit])
    };
    let columns = |f: &mut Func, vs: Vec<Val>| -> Result<Val, Error> {
        let vs = vs
            .into_iter()
            .map(|v| f.reshape(v, &[n, 1]))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(f.concat(&vs, 1)?)
    };
    let id = columns(f, id_cols)?;
    let w = columns(f, w_cols)?;
    Ok(Some((id, w)))
}

/// The planes a bilinear [`grid_taps`] reads: each image's grid and the
/// images' segment bounds.
#[derive(Clone, Copy, Debug)]
pub struct Bilinear {
    pub grids: Operand,
    pub segments: Operand,
}

/// The two bilinear taps of `index` on a `size`-long axis stretched onto
/// `side`: `(taps, weights)`, `[n]` each.
fn axis_taps(
    f: &mut Func,
    index: Val,
    size: Val,
    side: i64,
) -> Result<([Val; 2], [Val; 2]), Error> {
    let n = f.dims(index)[0];
    let one_i = f.const_i(Elem::I32, 1, &[n]);
    let span = f.sub(size, one_i)?;
    let span = f.max(span, one_i)?;
    let span = f.convert(span, Elem::F32);
    let i = f.convert(index, Elem::F32);
    let reach = f.const_f(Elem::F32, (side - 1) as f64, &[n]);
    let src = f.mul(i, reach)?;
    let src = f.div(src, span)?;
    let fl = f.floor(src);
    let base = f.convert(fl, Elem::I32);
    let frac = f.sub(src, fl)?;
    let lo = f.const_i(Elem::I32, 0, &[]);
    let hi = f.const_i(Elem::I32, side - 1, &[]);
    let one = f.const_f(Elem::F32, 1.0, &[n]);
    let zero = f.const_f(Elem::F32, 0.0, &[n]);
    let mut taps = [base; 2];
    let mut weights = [one; 2];
    for t in 0..2i64 {
        let k = f.const_i(Elem::I32, t, &[n]);
        let tap = f.add(base, k)?;
        taps[t as usize] = f.clamp(lo, tap, hi)?;
        let kf = f.const_f(Elem::F32, t as f64, &[n]);
        let d = f.sub(frac, kf)?;
        let d = f.abs(d);
        let w = f.sub(one, d)?;
        weights[t as usize] = f.max(w, zero)?;
    }
    Ok((taps, weights))
}

// ---------------------------------------------------------------- ranking
