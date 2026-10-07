//! Attention over ragged (segmented) q and kv rectangles, no cache.
//! Reference: kernels-metal `attn::ragged` (scalar arm) and kernels-cuda
//! `attn_ragged` (its mask variants).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::dense::segment_of;
use super::paged::{Heads, Planes, Rows, Rule, blocked, flat_i32, head, nonzero, row_heads};
use crate::cx::{Ctx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func};
use crate::tensor::Tensor;

/// What masks or biases a ragged attention besides its segments.
#[derive(Clone, Copy, Debug)]
pub enum RaggedMask {
    /// Segments only (poem_ir `None` and `GroupBlockDiagonal`).
    Segments,
    /// A row with tag `t >= 0` reads only keys tagged `t`; a negative tag
    /// reads its whole segment. One i32 per query row / key row.
    ReferenceTags { q_tags: Tensor, kv_tags: Tensor },
    /// `q_classes[q] < 0 || kv_classes[k] < 0 || table[q_class · count +
    /// kv_class] != 0` admits a pair (kernels-cuda's lane class table).
    ClassTable {
        q_classes: Tensor,
        kv_classes: Tensor,
        table: Tensor,
        count: u32,
    },
    /// Adds `table[h, clamp(k − q + max_len − 1)]` (f32, `[q_heads, 2·max_len
    /// − 1]`) to the scaled logit, `k`/`q` indexed within the segment.
    RelativeBias { table: Tensor, max_len: u32 },
}

/// Row `r` of segment `s` (`q_indptr[s] <= r < q_indptr[s+1]`) attends to
/// key rows `kv_indptr[s] .. kv_indptr[s+1]`, not causally; a row outside
/// every segment, or with nothing admitted, answers zeros.
pub fn forward(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    mask: &RaggedMask,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ragged";
    for t in [q, k, v, o] {
        expect(OP, t, &[Dtype::Bf16])?;
    }
    let qh = row_heads(OP, q.width, head_dim)?;
    let kvh = nonzero(OP, "kv heads", kv_heads)?;
    if k.width != kvh * head_dim || v.width != k.width || v.rows != k.rows {
        return Err(refuse(
            OP,
            format!("the key and value rectangles are {kvh} x {head_dim} wide, one row each"),
        ));
    }
    if !qh.is_multiple_of(kvh) {
        return Err(refuse(
            OP,
            format!("{qh} query heads do not group over {kvh} kv heads"),
        ));
    }
    if o.rows != q.rows || o.width != q.width {
        return Err(refuse(OP, "the answer is one row per query row"));
    }
    for t in [q_indptr, kv_indptr] {
        if t.dtype != Dtype::I32 {
            return Err(refuse(OP, "a segment table is i32"));
        }
    }
    let entries = q_indptr.elements().min(kv_indptr.elements());
    if entries < 2 {
        return Err(refuse(
            OP,
            format!("a CSR of {entries} entries names no segment"),
        ));
    }
    let segments = entries as i64 - 1;
    nonzero(OP, "rows", q.rows)?;
    let span = match *mask {
        RaggedMask::RelativeBias { table, max_len } => {
            let max_len = nonzero(OP, "the bias table's reach", max_len)?;
            let span = 2 * u64::from(max_len) - 1;
            if table.dtype != Dtype::F32 || table.elements() < u64::from(qh) * span {
                return Err(refuse(
                    OP,
                    format!("the bias table is not {qh} f32 rows of {span}"),
                ));
            }
            span as i64
        }
        RaggedMask::ReferenceTags { q_tags, kv_tags } => {
            for (t, n) in [(q_tags, q.rows), (kv_tags, k.rows)] {
                if t.dtype != Dtype::I32 || t.elements() < u64::from(n) {
                    return Err(refuse(OP, "a tag table is one i32 per row"));
                }
            }
            0
        }
        RaggedMask::ClassTable {
            q_classes,
            kv_classes,
            table,
            count,
        } => {
            for (t, n) in [(q_classes, q.rows), (kv_classes, k.rows)] {
                if t.dtype != Dtype::I32 || t.elements() < u64::from(n) {
                    return Err(refuse(OP, "a class table is one i32 per row"));
                }
            }
            nonzero(OP, "the class count", count)?;
            if !matches!(table.dtype, Dtype::U8 | Dtype::Bool)
                || table.elements() < u64::from(count) * u64::from(count)
            {
                return Err(refuse(OP, "the class table is count x count u8"));
            }
            0
        }
        RaggedMask::Segments => 0,
    };
    let heads = Heads {
        kvh: i64::from(kvh),
        g: i64::from(qh / kvh),
        d: i64::from(head_dim),
        dv: i64::from(head_dim),
    };
    let (r, nk) = (i64::from(q.rows), i64::from(k.rows));
    let mask = *mask;
    ctx.emit(&mut |cx| {
        let qi = cx.read(q_indptr)?;
        let qi = flat_i32(cx, qi)?;
        let qi = head(cx, qi, segments + 1)?;
        let ki = cx.read(kv_indptr)?;
        let ki = flat_i32(cx, ki)?;
        let ki = head(cx, ki, segments + 1)?;
        let mut rule = Rule::default();
        let mut tags = None;
        let mut classes = None;
        match mask {
            RaggedMask::Segments => {}
            RaggedMask::ReferenceTags { q_tags, kv_tags } => {
                let a = cx.read(q_tags)?;
                let a = flat_i32(cx, a)?;
                tags = Some(head(cx, a, r)?);
                let b = cx.read(kv_tags)?;
                let b = flat_i32(cx, b)?;
                rule.kv_tags = Some(head(cx, b, nk)?);
            }
            RaggedMask::ClassTable {
                q_classes,
                kv_classes,
                table,
                count,
            } => {
                let a = cx.read(q_classes)?;
                let a = flat_i32(cx, a)?;
                classes = Some(head(cx, a, r)?);
                let b = cx.read(kv_classes)?;
                let b = flat_i32(cx, b)?;
                let b = head(cx, b, nk)?;
                let t = cx.read(table)?;
                let t = flat_i32(cx, t)?;
                rule.classes = Some((b, t, i64::from(count)));
            }
            RaggedMask::RelativeBias { table, max_len } => {
                let t = cx.read(table)?;
                let n = cx.ty(t).elements();
                let t = cx.reshape(t, &[n])?;
                let t = head(cx, t, i64::from(qh) * span)?;
                rule.rbias = Some((t, i64::from(max_len)));
            }
        }
        let qv = cx.read(q)?;
        let kv = cx.read(k)?;
        let vv = cx.read(v)?;
        let f = cx.func();
        let (qlo, _, s) = segment_of(f, qi, segments, r)?;
        let sc = super::paged::clamp_i(f, s, 0, segments - 1)?;
        let inside = super::paged::cmp_i(f, crate::hlo::Cmp::Ge, s, 0)?;
        let lo = super::paged::take(f, ki, sc)?;
        let s1 = super::paged::with_i(f, sc, 1, Func::add)?;
        let hi = super::paged::take(f, ki, s1)?;
        let z = f.like_i(lo, 0);
        let lo = f.select(inside, lo, z)?;
        let hi = f.select(inside, hi, z)?;
        let q4 = f.reshape(qv, &[r, heads.kvh, heads.g, heads.d])?;
        let mut rows = Rows::new(q4, lo, hi);
        rows.tag = tags;
        rows.class = classes;
        if rule.rbias.is_some() {
            let io = f.iota(Elem::I32, &[r], 0);
            rows.qi = Some(f.sub(io, qlo)?);
            rows.begin = Some(lo);
        }
        let planes = Planes::Plain { k: kv, v: vv };
        let fl = blocked(f, &planes, &heads, &rule, &rows, nk, f64::from(sm_scale))?;
        cx.write(o, fl.o)
    })
}
