//! Bidirectional attention inside image segments (vision encoders).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::paged::{Heads, Planes, Rows, Rule, blocked, flat_i32, nonzero, row_heads, with_i};
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};
use crate::tensor::Tensor;

/// For each row, the segment `[lo, hi)` of `indptr` (`segments + 1` entries)
/// holding it, `lo = hi = 0` for a row outside every segment, and the
/// segment index (`-1` outside).
pub(crate) fn segment_of(
    f: &mut Func,
    indptr: Val,
    segments: i64,
    rows: i64,
) -> Built<(Val, Val, Val)> {
    let starts = f.slice(indptr, &[0], &[segments], &[1])?;
    let first = f.slice(indptr, &[0], &[1], &[1])?;
    let first = f.reshape(first, &[])?;
    let total = f.slice(indptr, &[segments], &[segments + 1], &[1])?;
    let total = f.reshape(total, &[])?;
    let r = f.iota(Elem::I32, &[rows], 0);
    let rb = f.broadcast(r, &[rows, segments], &[0])?;
    let sb = f.broadcast(starts, &[rows, segments], &[1])?;
    let le = f.compare(Cmp::Le, sb, rb)?;
    let one = f.const_i(Elem::I32, 1, &[rows, segments]);
    let zero = f.const_i(Elem::I32, 0, &[rows, segments]);
    let n = f.select(le, one, zero)?;
    let n = f.reduce(n, &[1], Fold::Sum)?;
    let s = with_i(f, n, 1, Func::sub)?;
    let s = super::paged::clamp_i(f, s, 0, segments - 1)?;
    let fb = f.splat(first, &[rows])?;
    let tb = f.splat(total, &[rows])?;
    let a = f.compare(Cmp::Ge, r, fb)?;
    let b = f.compare(Cmp::Lt, r, tb)?;
    let inside = f.and(a, b)?;
    let lo = super::paged::take(f, indptr, s)?;
    let s1 = with_i(f, s, 1, Func::add)?;
    let hi = super::paged::take(f, indptr, s1)?;
    let z = f.like_i(lo, 0);
    let lo = f.select(inside, lo, z)?;
    let hi = f.select(inside, hi, z)?;
    let neg = f.like_i(s, -1);
    let s = f.select(inside, s, neg)?;
    Ok((lo, hi, s))
}

/// Every row attends, bidirectionally, to the key rows of its own segment
/// (`segments` is the `[images + 1]` i32 boundary vector over the shared row
/// numbering); a row outside every segment answers zeros.
pub fn bidirectional(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    segments: Tensor,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.dense";
    for t in [q, k, v, o] {
        expect(OP, t, &[Dtype::Bf16])?;
    }
    let qh = row_heads(OP, q.width, head_dim)?;
    let kvh = row_heads(OP, k.width, head_dim)?;
    if !qh.is_multiple_of(kvh) || v.width != k.width || v.rows != k.rows {
        return Err(refuse(
            OP,
            format!("{qh} query heads do not group over {kvh} kv heads of one k/v rectangle"),
        ));
    }
    if o.rows != q.rows || o.width != q.width {
        return Err(refuse(OP, "the answer is one row per query row"));
    }
    if segments.dtype != Dtype::I32 || segments.elements() < 2 {
        return Err(refuse(
            OP,
            "the segment list is an i32 boundary vector naming at least one image",
        ));
    }
    nonzero(OP, "rows", q.rows)?;
    let heads = Heads {
        kvh: i64::from(kvh),
        g: i64::from(qh / kvh),
        d: i64::from(head_dim),
        dv: i64::from(head_dim),
    };
    let r = i64::from(q.rows);
    let images = segments.elements() as i64 - 1;
    ctx.emit(&mut |cx| {
        let seg = cx.read(segments)?;
        let seg = flat_i32(cx, seg)?;
        let qv = cx.read(q)?;
        let kv = cx.read(k)?;
        let vv = cx.read(v)?;
        let f = cx.func();
        let (lo, hi, _) = segment_of(f, seg, images, r)?;
        let q4 = f.reshape(qv, &[r, heads.kvh, heads.g, heads.d])?;
        let rows = Rows::new(q4, lo, hi);
        let planes = Planes::Plain { k: kv, v: vv };
        let fl = blocked(
            f,
            &planes,
            &heads,
            &Rule::default(),
            &rows,
            i64::from(k.rows),
            f64::from(sm_scale),
        )?;
        cx.write(o, fl.o)
    })
}
