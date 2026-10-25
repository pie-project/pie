//! Attn over ragged (segmented) q and kv rectangles, no cache.
//! Reference: kernels-metal `attn::ragged` (scalar arm) and kernels-cuda
//! `attn_ragged` (its mask variants).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::dense::segment_of;
use super::paged::{Heads, Planes, Rows, Rule, blocked, flat_i32, head, nonzero, row_heads};
use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::tensor::{Landing, Operand, Stored};

/// What masks or biases a ragged attention besides its segments.
#[derive(Clone, Copy, Debug)]
pub enum RaggedMask<T> {
    /// Segments only (poem_ir `None` and `GroupBlockDiagonal`).
    Segments,
    /// Segment `s`'s pair `(q, k)` (indexed within the segment) is admitted
    /// where bit `q · kv_len + k` of `bits` (u8), from byte `indptr[s]`
    /// (i32), is set (kernels-cuda's custom-mask arm).
    Custom { bits: T, indptr: T },
    /// Adds `table[h, clamp(k − q + max_len − 1)]` (f32, `[q_heads, 2·max_len
    /// − 1]`) to the scaled logit, `k`/`q` indexed within the segment.
    RelativeBias { table: T, max_len: u32 },
}

impl<T> RaggedMask<T> {
    /// The mask with each plane mapped by `f`, in the order [`forward`]
    /// consumes them.
    pub fn try_map<U, E>(self, mut f: impl FnMut(T) -> Result<U, E>) -> Result<RaggedMask<U>, E> {
        Ok(match self {
            Self::Segments => RaggedMask::Segments,
            Self::Custom { bits, indptr } => RaggedMask::Custom {
                bits: f(bits)?,
                indptr: f(indptr)?,
            },
            Self::RelativeBias { table, max_len } => RaggedMask::RelativeBias {
                table: f(table)?,
                max_len,
            },
        })
    }
}

/// Row `r` of segment `s` (`q_indptr[s] <= r < q_indptr[s+1]`) attends to
/// key rows `kv_indptr[s] .. kv_indptr[s+1]`, not causally; a row outside
/// every segment, or with nothing admitted, answers zeros.
pub fn forward(
    f: &mut Func,
    q: Operand,
    k: Operand,
    v: Operand,
    q_indptr: Operand,
    kv_indptr: Operand,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    mask: &RaggedMask<Operand>,
    o: Landing,
    tuning: crate::tuning::Tuning,
) -> Result<Val, Error> {
    const OP: &str = "attn.ragged";
    for t in [q.shape(), k.shape(), v.shape(), o] {
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
        RaggedMask::Custom { bits, indptr } => {
            if !matches!(bits.dtype, Dtype::U8 | Dtype::Bool) || bits.elements() == 0 {
                return Err(refuse(OP, "a custom mask's bits are u8"));
            }
            if indptr.dtype != Dtype::I32 || indptr.elements() < entries {
                return Err(refuse(
                    OP,
                    "a custom mask states one i32 byte offset per segment",
                ));
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
    let qi = q_indptr.val;
    let qi = flat_i32(f, qi)?;
    let qi = head(f, qi, segments + 1)?;
    let ki = kv_indptr.val;
    let ki = flat_i32(f, ki)?;
    let ki = head(f, ki, segments + 1)?;
    let mut rule = Rule::default();
    let mut starts = None;
    match mask {
        RaggedMask::Segments => {}
        RaggedMask::Custom { bits, indptr } => {
            let b = bits.val;
            rule.bits = Some(flat_i32(f, b)?);
            let m = indptr.val;
            let m = flat_i32(f, m)?;
            starts = Some(head(f, m, segments + 1)?);
        }
        RaggedMask::RelativeBias { table, max_len } => {
            let t = table.val;
            let n = f.ty(t).elements();
            let t = f.reshape(t, &[n])?;
            let t = head(f, t, i64::from(qh) * span)?;
            rule.rbias = Some((t, i64::from(max_len)));
        }
    }
    let qv = q.val;
    let kv = k.val;
    let vv = v.val;
    let ch = tuning.chunks;
    let fq = &mut *f;
    let (qlo, _, s) = segment_of(fq, qi, segments, r)?;
    let sc = super::paged::clamp_i(fq, s, 0, segments - 1)?;
    let inside = super::paged::cmp_i(fq, crate::hlo::Cmp::Ge, s, 0)?;
    let lo = super::paged::take(fq, ki, sc)?;
    let s1 = super::paged::with_i(fq, sc, 1, Func::add)?;
    let hi = super::paged::take(fq, ki, s1)?;
    let z = fq.like_i(lo, 0);
    let lo = fq.select(inside, lo, z)?;
    let hi = fq.select(inside, hi, z)?;
    let q4 = fq.reshape(qv, &[r, heads.kvh, heads.g, heads.d])?;
    let mut rows = Rows::new(q4, lo, hi);
    if rule.rbias.is_some() {
        let io = fq.iota(Elem::I32, &[r], 0);
        rows.qi = Some(fq.sub(io, qlo)?);
        rows.begin = Some(lo);
    }
    if let Some(starts) = starts {
        // A row's first key stands at bit `8 · indptr[s] + qo · kv_len`.
        let io = fq.iota(Elem::I32, &[r], 0);
        let qo = fq.sub(io, qlo)?;
        let len = fq.sub(hi, lo)?;
        let at = fq.mul(qo, len)?;
        let byte = super::paged::take(fq, starts, sc)?;
        let byte = super::paged::with_i(fq, byte, 8, Func::mul)?;
        rows.bit = Some(fq.add(byte, at)?);
        rows.begin = Some(lo);
    }
    let planes = Planes::Plain { k: kv, v: vv };
    let fl = blocked(
        fq,
        &planes,
        &heads,
        &rule,
        &rows,
        nk,
        f64::from(sm_scale),
        ch,
    )?;
    Ok(fl.o)
}
