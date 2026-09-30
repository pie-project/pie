//! Paged attention (kernels-wgpu's `attn::` root), and the flash core every
//! attention family here shares.
//!
//! The core reads keys through a *flat key index*: for a paged pool, key
//! position `kp` of lane `l` is flat index `l · span + kp` (`span` =
//! `max_pages · page_size`), resolved through a dense `[lanes, max_pages]`
//! page table built in-graph from the fire's CSR; for a plain key rectangle
//! the flat index is the key row. Two drivers walk it with an online softmax
//! in f32:
//!
//! - [`per_row`]: every query row gathers its own lane's keys, chunk by chunk
//!   (the decode shape, and any per-row key list such as a selection);
//! - [`blocked`]: query rows in blocks, each block walking only the key
//!   chunks its rows' `[lo, hi)` ranges span (a data-dependent trip count), as
//!   one dense `dot_general` per chunk (the prefill shape).
//!
//! Scores are `q · k` with bf16 operands and f32 accumulation (every product
//! exact), then scaled; probabilities meet values as a two-term bf16 split
//! (`p = hi + lo`), which keeps the value reduction f32-accurate at two MXU
//! passes. A row with no admitted key answers zeros (and a `-inf` log-sum-exp),
//! never NaN.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::{DecodePlan, PrefillPlan};
use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Built, Cmp, Elem, Fold, Func, GatherDims, Malformed, Val};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

// ------------------------------------------------------------------ helpers

pub(crate) fn bad<T>(detail: &str) -> Built<T> {
    Err(Malformed {
        op: "attention",
        detail: detail.to_string(),
    })
}

/// `v op x` for an int (or float) scalar `x` splat to `v`'s shape.
pub(crate) fn with_i(
    f: &mut Func,
    v: Val,
    x: i64,
    op: fn(&mut Func, Val, Val) -> Built<Val>,
) -> Built<Val> {
    let c = if f.elem(v).is_float() {
        f.like_f(v, x as f64)
    } else {
        f.like_i(v, x)
    };
    op(f, v, c)
}

pub(crate) fn cmp_i(f: &mut Func, dir: Cmp, v: Val, x: i64) -> Built<Val> {
    let c = f.like_i(v, x);
    f.compare(dir, v, c)
}

pub(crate) fn clamp_i(f: &mut Func, v: Val, lo: i64, hi: i64) -> Built<Val> {
    let elem = f.elem(v);
    let lo = f.const_i(elem, lo, &[]);
    let hi = f.const_i(elem, hi, &[]);
    f.clamp(lo, v, hi)
}

pub(crate) fn scalar_i(f: &mut Func, x: i64) -> Val {
    f.const_i(Elem::I32, x, &[])
}

/// Any int table as a flat i32 vector.
pub(crate) fn flat_i32(f: &mut Func, v: Val) -> Built<Val> {
    let n = f.ty(v).elements();
    let v = f.convert(v, Elem::I32);
    f.reshape(v, &[n])
}

/// The first `n` entries of a flat vector (the table may be longer).
pub(crate) fn head(f: &mut Func, v: Val, n: i64) -> Built<Val> {
    f.slice(v, &[0], &[n], &[1])
}

/// `table[idx]` elementwise for a flat `table`; `idx` any shape (clamped).
pub(crate) fn take(f: &mut Func, table: Val, idx: Val) -> Built<Val> {
    let dims = f.dims(idx).to_vec();
    let n = f.dims(table)[0];
    let t = f.reshape(table, &[n, 1])?;
    let m: i64 = dims.iter().product();
    let i = f.reshape(idx, &[m])?;
    let g = f.take_rows(t, i)?;
    f.reshape(g, &dims)
}

/// Rows `[start, start + n)` of `v` along axis 0 (`start` an i32 scalar).
pub(crate) fn rows_at(f: &mut Func, v: Val, start: Val, n: i64) -> Built<Val> {
    let dims = f.dims(v).to_vec();
    let zero = scalar_i(f, 0);
    let mut starts = vec![start];
    starts.extend(std::iter::repeat_n(zero, dims.len() - 1));
    let mut sizes = dims;
    sizes[0] = n;
    f.dynamic_slice(v, &starts, &sizes)
}

/// `v` grown along axis 0 by `extra` rows of zero.
fn pad_rows(f: &mut Func, v: Val, extra: i64) -> Built<Val> {
    if extra == 0 {
        return Ok(v);
    }
    let r = f.ty(v).rank();
    let elem = f.elem(v);
    let z = if elem.is_float() {
        f.const_f(elem, 0.0, &[])
    } else {
        f.const_i(elem, 0, &[])
    };
    let mut hi = vec![0; r];
    hi[0] = extra;
    f.pad(v, z, &vec![0; r], &hi, &vec![0; r])
}

/// `⌈x / m⌉` for positive `m`.
pub(crate) const fn cdiv(x: i64, m: i64) -> i64 {
    (x + m - 1) / m
}

pub(crate) const fn round_up(x: i64, m: i64) -> i64 {
    cdiv(x, m) * m
}

fn pow2_floor(x: i64) -> i64 {
    if x <= 1 {
        1
    } else {
        1 << (63 - x.leading_zeros())
    }
}

// ----------------------------------------------------------------- geometry

/// A fire's paging, laid dense: `pt[l · pages + p]` is lane `l`'s `p`-th
/// page (clamped garbage past its count), `count[l]` its page count.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geom {
    pub lanes: i64,
    pub pages: i64,
    pub ps: i64,
    pub span: i64,
    pub pt: Val,
    pub count: Val,
}

pub(crate) fn geometry(
    f: &mut Func,
    indptr: Val,
    indices: Val,
    ps: i64,
    pages: i64,
) -> Built<Geom> {
    let lanes = f.dims(indptr)[0] - 1;
    let start = f.slice(indptr, &[0], &[lanes], &[1])?;
    let end = f.slice(indptr, &[1], &[lanes + 1], &[1])?;
    let count = f.sub(end, start)?;
    let p = f.iota(Elem::I32, &[lanes, pages], 1);
    let s = f.broadcast(start, &[lanes, pages], &[0])?;
    let at = f.add(s, p)?;
    let pt = take(f, indices, at)?;
    let pt = f.reshape(pt, &[lanes * pages])?;
    Ok(Geom {
        lanes,
        pages,
        ps,
        span: pages * ps,
        pt,
        count,
    })
}

impl Geom {
    /// Lane `lane`'s kv capacity (`pages · page_size`), clamped to `span`.
    pub(crate) fn capacity(&self, f: &mut Func, lane: Val) -> Built<Val> {
        let c = take(f, self.count, lane)?;
        let c = with_i(f, c, self.ps, Func::mul)?;
        clamp_i(f, c, 0, self.span)
    }

    /// The owning lane of each row, clamped into the fire's lanes.
    pub(crate) fn lane_of(&self, f: &mut Func, req: Val) -> Built<Val> {
        clamp_i(f, req, 0, self.lanes - 1)
    }
}

// ------------------------------------------------------------------- planes

/// Where keys and values come from.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Planes {
    /// A paged pool. `latent`: the MLA layout (keys plane = the compressed
    /// latent, which is also the value; values plane = the rope tail, joined
    /// to the key).
    Paged {
        keys: Val,
        values: Val,
        geom: Geom,
        latent: bool,
    },
    /// Plain `[n, kv_heads · d]` key and value rectangles.
    Plain { k: Val, v: Val },
}

/// Head geometry of one attention: `g` query heads per kv head.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Heads {
    pub kvh: i64,
    pub g: i64,
    pub d: i64,
    pub dv: i64,
}

impl Heads {
    pub(crate) const fn qh(&self) -> i64 {
        self.kvh * self.g
    }
}

/// Keys `[n, kvh, d]` and values `[n, kvh, dv]` at flat indices `x` (`[n]`).
pub(crate) fn fetch(f: &mut Func, planes: &Planes, h: &Heads, x: Val) -> Built<(Val, Val)> {
    let n = f.dims(x)[0];
    match *planes {
        Planes::Paged {
            keys,
            values,
            geom,
            latent,
        } => {
            let x = clamp_i(f, x, 0, geom.lanes * geom.span - 1)?;
            let lane = with_i(f, x, geom.span, Func::div)?;
            let kp = with_i(f, x, geom.span, Func::rem)?;
            let pix = with_i(f, kp, geom.ps, Func::div)?;
            let off = with_i(f, kp, geom.ps, Func::rem)?;
            let at = with_i(f, lane, geom.pages, Func::mul)?;
            let at = f.add(at, pix)?;
            let page = take(f, geom.pt, at)?;
            let slot = with_i(f, page, geom.ps, Func::mul)?;
            let slot = f.add(slot, off)?;
            let kr = f.take_rows(keys, slot)?;
            if latent {
                let v = f.reshape(kr, &[n, 1, h.dv])?;
                let k = if h.d > h.dv {
                    let pe = f.take_rows(values, slot)?;
                    f.concat(&[kr, pe], 1)?
                } else {
                    kr
                };
                let k = f.reshape(k, &[n, 1, h.d])?;
                Ok((k, v))
            } else {
                let vr = f.take_rows(values, slot)?;
                let k = f.reshape(kr, &[n, h.kvh, h.d])?;
                let v = f.reshape(vr, &[n, h.kvh, h.dv])?;
                Ok((k, v))
            }
        }
        Planes::Plain { k, v } => {
            let nk = f.dims(k)[0];
            let x = clamp_i(f, x, 0, nk - 1)?;
            let kr = f.take_rows(k, x)?;
            let vr = f.take_rows(v, x)?;
            let k = f.reshape(kr, &[n, h.kvh, h.d])?;
            let v = f.reshape(vr, &[n, h.kvh, h.dv])?;
            Ok((k, v))
        }
    }
}

/// Keys `[m · ps, kvh, d]` and values `[m · ps, kvh, dv]` of whole pool
/// pages `page` (`[m]` i32), each read as one `ps`-row slab: the keys
/// [`fetch`] returns for those pages' positions, without a per-key
/// page-table lookup or a per-key row gather. `None` for plain planes.
pub(crate) fn fetch_pages(
    f: &mut Func,
    planes: &Planes,
    h: &Heads,
    page: Val,
) -> Built<Option<(Val, Val)>> {
    let Planes::Paged {
        keys,
        values,
        geom,
        latent,
    } = *planes
    else {
        return Ok(None);
    };
    let m = f.dims(page)[0];
    let n = m * geom.ps;
    let page = f.reshape(page, &[m, 1])?;
    let blocks = |f: &mut Func, plane: Val| -> Built<Val> {
        let (slots, w) = (f.dims(plane)[0], f.dims(plane)[1]);
        // `[slots, w]` as `[pages, ps, w]`: whole pages along the major
        // axis, so the gather moves full `ps · w` slabs.
        let paged = f.reshape(plane, &[slots / geom.ps, geom.ps, w])?;
        let b = f.gather(
            paged,
            page,
            &GatherDims {
                offset_dims: vec![1, 2],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                ..GatherDims::default()
            },
            &[1, geom.ps, w],
        )?;
        f.reshape(b, &[n, w])
    };
    let kr = blocks(f, keys)?;
    if latent {
        let v = f.reshape(kr, &[n, 1, h.dv])?;
        let k = if h.d > h.dv {
            let pe = blocks(f, values)?;
            f.concat(&[kr, pe], 1)?
        } else {
            kr
        };
        let k = f.reshape(k, &[n, 1, h.d])?;
        Ok(Some((k, v)))
    } else {
        let vr = blocks(f, values)?;
        let k = f.reshape(kr, &[n, h.kvh, h.d])?;
        let v = f.reshape(vr, &[n, h.kvh, h.dv])?;
        Ok(Some((k, v)))
    }
}

/// A walk's page table laid for chunks of whole pages: per row (`[r,
/// pages + np]`, each row its lane's pages) or flat (`[lanes · pages +
/// np]`), padded by one chunk so a chunk's window never clamps.
#[derive(Clone, Copy, Debug)]
struct PageWalk {
    table: Val,
    np: i64,
    ps: i64,
}

impl PageWalk {
    /// For a chunk of `c` keys when it covers whole pages of a paged pool:
    /// per row when `lane` (`[r]` i32) is given, else flat.
    fn of(f: &mut Func, planes: &Planes, lane: Option<Val>, c: i64) -> Built<Option<Self>> {
        let Planes::Paged { geom, .. } = *planes else {
            return Ok(None);
        };
        if geom.ps <= 0 || c % geom.ps != 0 {
            return Ok(None);
        }
        let np = c / geom.ps;
        let zero = f.const_i(Elem::I32, 0, &[]);
        let table = match lane {
            Some(lane) => {
                let pt = f.reshape(geom.pt, &[geom.lanes, geom.pages])?;
                let lane = clamp_i(f, lane, 0, geom.lanes - 1)?;
                let rows = f.take_rows(pt, lane)?;
                f.pad(rows, zero, &[0, 0], &[0, np], &[0, 0])?
            }
            None => f.pad(geom.pt, zero, &[0], &[np], &[0])?,
        };
        Ok(Some(Self {
            table,
            np,
            ps: geom.ps,
        }))
    }

    /// The pool pages of the chunk at key `base` (an i32 scalar, a multiple
    /// of the chunk): `[r · np]` or `[np]`, in key order.
    fn chunk(&self, f: &mut Func, base: Val) -> Built<Val> {
        let p0 = with_i(f, base, self.ps, Func::div)?;
        let dims = f.dims(self.table).to_vec();
        if dims.len() == 2 {
            let zero = scalar_i(f, 0);
            let w = f.dynamic_slice(self.table, &[zero, p0], &[dims[0], self.np])?;
            f.reshape(w, &[dims[0] * self.np])
        } else {
            f.dynamic_slice(self.table, &[p0], &[self.np])
        }
    }
}

// --------------------------------------------------------------------- rows

/// Per-query-row operands, each with the rows on axis 0. `lo`/`hi` bound the
/// admitted keys: key positions for [`per_row`], flat key indices for
/// [`blocked`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rows {
    /// `[R, kvh, g, d]` bf16.
    pub q: Val,
    pub lo: Val,
    pub hi: Val,
    /// Owning lane, i32 (the per-row driver's key base).
    pub lane: Option<Val>,
    /// Absolute query position, i32 (relative bias).
    pub qpos: Option<Val>,
    /// Custom-mask flag per row (0 off, else on), i32.
    pub enabled: Option<Val>,
    /// `[R, pitch]` u8 custom-mask plane, indexed by key position.
    pub mask: Option<Val>,
    /// `[R, qh · extent]` f32 relative-bias rows.
    pub rel: Option<Val>,
    /// Reference tag per row, i32.
    pub tag: Option<Val>,
    /// Attention class per row, i32.
    pub class: Option<Val>,
    /// Row index within its segment, i32 (ragged bias).
    pub qi: Option<Val>,
    /// The segment's first key row, i32 (ragged bias).
    pub begin: Option<Val>,
}

impl Rows {
    pub(crate) fn new(q: Val, lo: Val, hi: Val) -> Self {
        Self {
            q,
            lo,
            hi,
            lane: None,
            qpos: None,
            enabled: None,
            mask: None,
            rel: None,
            tag: None,
            class: None,
            qi: None,
            begin: None,
        }
    }

    fn map(&self, f: &mut Func, g: &mut dyn FnMut(&mut Func, Val) -> Built<Val>) -> Built<Self> {
        let mut o = |f: &mut Func, v: Option<Val>| v.map(|v| g(f, v)).transpose();
        Ok(Self {
            q: o(f, Some(self.q))?.unwrap_or(self.q),
            lo: o(f, Some(self.lo))?.unwrap_or(self.lo),
            hi: o(f, Some(self.hi))?.unwrap_or(self.hi),
            lane: o(f, self.lane)?,
            qpos: o(f, self.qpos)?,
            enabled: o(f, self.enabled)?,
            mask: o(f, self.mask)?,
            rel: o(f, self.rel)?,
            tag: o(f, self.tag)?,
            class: o(f, self.class)?,
            qi: o(f, self.qi)?,
            begin: o(f, self.begin)?,
        })
    }
}

/// What admits a (row, key) pair besides its range, and what it adds to the
/// logit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Rule {
    /// Column bound of the custom mask (keys at or past it are masked out
    /// for an enabled row).
    pub bound: i64,
    /// Blocked paged walk: key position = flat index mod this span.
    pub kp_span: Option<i64>,
    /// Relative bias rows: `(extent, log_floor, log_alpha)`.
    pub rel: Option<(i64, u32, f32)>,
    /// Per-key reference tags `[n]`.
    pub kv_tags: Option<Val>,
    /// `(kv_classes [n], table [count²] i32, count)`.
    pub classes: Option<(Val, Val, i64)>,
    /// `(table [qh · (2·max_len − 1)] f32, max_len)`.
    pub rbias: Option<(Val, i64)>,
    /// The most keys one row admits (a sliding window), when bounded: the
    /// per-row walk then reads chunks no wider than it.
    pub window: Option<i64>,
}

struct Logits {
    valid: Val,
    /// `[rb, qh, c]` f32 added after scaling.
    bias: Option<Val>,
    /// `[rb]` f32 multiplier after the bias.
    mult: Option<Val>,
}

/// The admission and logit terms of `rows` (`rb` of them) against keys at
/// flat index `x` and position `kp`, both `[rb, c]`; `range` is the driver's
/// `[lo, hi)` test.
fn logits(
    f: &mut Func,
    rule: &Rule,
    h: &Heads,
    rows: &Rows,
    x: Val,
    kp: Val,
    kbase: Option<Val>,
    range: Val,
) -> Built<Logits> {
    let dims = f.dims(x).to_vec();
    let (rb, c) = (dims[0], dims[1]);
    let mut valid = range;
    if let (Some(en), Some(plane)) = (rows.enabled, rows.mask) {
        let pitch = f.dims(plane)[1];
        let on = cmp_i(f, Cmp::Ne, en, 0)?;
        let on = f.broadcast(on, &dims, &[0])?;
        let ge = cmp_i(f, Cmp::Ge, kp, 0)?;
        let lt = cmp_i(f, Cmp::Lt, kp, rule.bound.min(pitch))?;
        let inb = f.and(ge, lt)?;
        let cell = match kbase {
            // Every row reads columns `base .. base + c`: one window of the
            // plane (zero past its pitch) instead of a gather per cell.
            Some(base) => {
                let zero = f.const_i(f.elem(plane), 0, &[]);
                let wide = f.pad(plane, zero, &[0, 0], &[0, c], &[0, 0])?;
                let start = clamp_i(f, base, 0, pitch)?;
                let top = scalar_i(f, 0);
                f.dynamic_slice(wide, &[top, start], &[rb, c])?
            }
            None => {
                let col = clamp_i(f, kp, 0, pitch - 1)?;
                let r = f.iota(Elem::I32, &dims, 0);
                let at = with_i(f, r, pitch, Func::mul)?;
                let at = f.add(at, col)?;
                let flat = f.reshape(plane, &[rb * pitch])?;
                take(f, flat, at)?
            }
        };
        let keep = cmp_i(f, Cmp::Ne, cell, 0)?;
        let keep = f.and(inb, keep)?;
        let off = f.not(on);
        let ok = f.or(off, keep)?;
        valid = f.and(valid, ok)?;
    }
    if let (Some(kv_tags), Some(tag)) = (rule.kv_tags, rows.tag) {
        let n = f.dims(kv_tags)[0];
        let xc = clamp_i(f, x, 0, n - 1)?;
        let kt = take(f, kv_tags, xc)?;
        let qt = f.broadcast(tag, &dims, &[0])?;
        let untagged = cmp_i(f, Cmp::Lt, qt, 0)?;
        let same = f.compare(Cmp::Eq, kt, qt)?;
        let ok = f.or(untagged, same)?;
        valid = f.and(valid, ok)?;
    }
    if let (Some((kv_classes, table, count)), Some(class)) = (rule.classes, rows.class) {
        let n = f.dims(kv_classes)[0];
        let xc = clamp_i(f, x, 0, n - 1)?;
        let kc = take(f, kv_classes, xc)?;
        let qc = f.broadcast(class, &dims, &[0])?;
        let at = with_i(f, qc, count, Func::mul)?;
        let at = f.add(at, kc)?;
        let cells = f.dims(table)[0];
        let at = clamp_i(f, at, 0, cells - 1)?;
        let cell = take(f, table, at)?;
        let q_free = cmp_i(f, Cmp::Lt, qc, 0)?;
        let k_free = cmp_i(f, Cmp::Lt, kc, 0)?;
        let allowed = cmp_i(f, Cmp::Ne, cell, 0)?;
        let ok = f.or(q_free, k_free)?;
        let ok = f.or(ok, allowed)?;
        valid = f.and(valid, ok)?;
    }
    let qh = h.qh();
    let d3 = [rb, qh, c];
    let mut bias = None;
    let mut mult = None;
    if let (Some((extent, log_floor, log_alpha)), Some(plane), Some(qpos)) =
        (rule.rel, rows.rel, rows.qpos)
    {
        let qp = f.broadcast(qpos, &dims, &[0])?;
        let d = f.sub(qp, kp)?;
        let ge = cmp_i(f, Cmp::Ge, d, 0)?;
        let lt = cmp_i(f, Cmp::Lt, d, extent)?;
        let inr = f.and(ge, lt)?;
        let dc = clamp_i(f, d, 0, extent - 1)?;
        let r = f.iota(Elem::I32, &d3, 0);
        let hh = f.iota(Elem::I32, &d3, 1);
        let at = with_i(f, r, qh, Func::mul)?;
        let at = f.add(at, hh)?;
        let at = with_i(f, at, extent, Func::mul)?;
        let dc = f.broadcast(dc, &d3, &[0, 2])?;
        let at = f.add(at, dc)?;
        let flat = f.reshape(plane, &[rb * qh * extent])?;
        let b = take(f, flat, at)?;
        let b = f.convert(b, Elem::F32);
        let inr = f.broadcast(inr, &d3, &[0, 2])?;
        let z = f.like_f(b, 0.0);
        bias = Some(f.select(inr, b, z)?);
        if log_alpha != 0.0 && log_floor != 0 {
            let n = with_i(f, qpos, 1, Func::add)?;
            let n = f.convert(n, Elem::F32);
            let ratio = f.scale(n, 1.0 / f64::from(log_floor))?;
            let one = f.like_f(ratio, 1.0);
            let above = f.compare(Cmp::Gt, ratio, one)?;
            let lg = f.log(ratio);
            let m = f.scale(lg, f64::from(log_alpha))?;
            let m = f.offset(m, 1.0)?;
            mult = Some(f.select(above, m, one)?);
        }
    }
    if let (Some((table, max_len)), Some(qi), Some(begin)) = (rule.rbias, rows.qi, rows.begin) {
        let span = 2 * max_len - 1;
        let b = f.broadcast(begin, &dims, &[0])?;
        let q = f.broadcast(qi, &dims, &[0])?;
        let at = f.sub(x, b)?;
        let at = f.sub(at, q)?;
        let at = with_i(f, at, max_len - 1, Func::add)?;
        let at = clamp_i(f, at, 0, span - 1)?;
        let at = f.broadcast(at, &d3, &[0, 2])?;
        let hh = f.iota(Elem::I32, &d3, 1);
        let hh = with_i(f, hh, span, Func::mul)?;
        let at = f.add(hh, at)?;
        let b = take(f, table, at)?;
        bias = Some(f.convert(b, Elem::F32));
    }
    Ok(Logits { valid, bias, mult })
}

// ------------------------------------------------------------ online softmax

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// Scores `[r, kvh, g, c]`; values `[r, c, kvh, dv]`.
    PerRow,
    /// Scores `[kvh, rb, g, c]`; values `[c, kvh, dv]`.
    Blocked,
}

/// `x` as `hi + lo`, both bf16: `hi` is `x` with its low 16 mantissa bits
/// cleared (exact in bf16), `lo` the rest rounded (`|lo| < 2⁻⁷|x|`), so
/// `hi + lo` holds `x` to ~2⁻¹⁶. The split is by bits, not by an f32→bf16→f32
/// round trip, which XLA's excess-precision rewrite folds back to `x`.
pub(crate) fn split_bf16(f: &mut Func, x: Val) -> Built<(Val, Val)> {
    let bits = f.bitcast(x, Elem::U32)?;
    let keep = f.like_i(bits, 0xFFFF_0000);
    let bits = f.and(bits, keep)?;
    let hi32 = f.bitcast(bits, Elem::F32)?;
    let lo = f.sub(x, hi32)?;
    Ok((f.convert(hi32, Elem::Bf16), f.convert(lo, Elem::Bf16)))
}

/// `p · v` in f32 from bf16 values: `p` split as `hi + lo`, two bf16 passes.
fn pv(f: &mut Func, lay: Layout, p: Val, v: Val) -> Built<Val> {
    let (hi, lo) = split_bf16(f, p)?;
    let split = f.ty(v).rank() == 3;
    let dot = |f: &mut Func, x: Val| match lay {
        Layout::PerRow if split => by_head(f, x, v, 1),
        Layout::PerRow => f.dot_general_at(x, v, &[0, 1], &[0, 2], &[3], &[1], Elem::F32, false),
        Layout::Blocked => f.dot_general_at(x, v, &[0], &[1], &[3], &[0], Elem::F32, false),
    };
    let a = dot(f, hi)?;
    let b = dot(f, lo)?;
    f.add(a, b)
}

/// Whether the per-row walk reads kv heads as 128-aligned column slices of
/// `[r, c, kvh · d]` (no relayout of the gathered keys) rather than as a
/// `[r, c, kvh, d]` batch axis.
fn heads_by_slice(h: &Heads) -> bool {
    h.kvh > 1 && (h.d % 128 == 0 && h.dv % 128 == 0 || heads_by_pair(h.kvh, h.d, h.dv))
}

/// Whether kv heads 64 wide go through [`by_head`] two at a time, as one
/// 128-wide head (gpt-oss: 8 kv heads of 64): the gathered keys stay
/// `[r, c, kvh · 64]` with their 128-lane tiles whole, where a `[.., kvh,
/// 64]` batch axis relayouts every chunk of them (v6e: ~20 us a layer at
/// 64 rows).
fn heads_by_pair(kvh: i64, d: i64, dv: i64) -> bool {
    kvh > 1 && kvh % 2 == 0 && d == 64 && dv == 64
}

/// A per-row contraction one kv head at a time. `a` is `[r, kvh, g, x]`
/// and `b` `[r, c, kvh · w]`: with `contract` 2 (`x = w = d`, scores) the
/// result is `[r, kvh, g, c]`; with `contract` 1 (`x = c`, `w = dv`, the
/// value reduction) it is `[r, kvh, g, dv]`. f32 out.
fn by_head(f: &mut Func, a: Val, b: Val, contract: i64) -> Built<Val> {
    let ad = f.dims(a).to_vec();
    let bd = f.dims(b).to_vec();
    let (r, kvh, g, x) = (ad[0], ad[1], ad[2], ad[3]);
    let w = bd[2] / kvh;
    if w == 64 && kvh % 2 == 0 && (contract == 1 || x == 64) {
        return by_pair(f, a, b, contract);
    }
    let mut parts = Vec::with_capacity(kvh as usize);
    for hh in 0..kvh {
        let ah = f.slice(a, &[0, hh, 0, 0], &[r, hh + 1, g, x], &[1, 1, 1, 1])?;
        let ah = f.reshape(ah, &[r, g, x])?;
        let bh = f.slice(b, &[0, 0, hh * w], &[r, bd[1], (hh + 1) * w], &[1, 1, 1])?;
        let o = f.dot_general_at(ah, bh, &[0], &[0], &[2], &[contract], Elem::F32, false)?;
        let od = f.dims(o).to_vec();
        parts.push(f.reshape(o, &[r, 1, g, od[2]])?);
    }
    f.concat(&parts, 1)
}

/// [`by_head`] for 64-wide heads, two at a time: heads `2p` and `2p+1` are
/// one 128-wide head `p` of `b` (`[r, c, kvh/2, 128]`, its tiles whole).
/// Scores: head `2p`'s query rows padded with zeros on the right, `2p+1`'s
/// on the left, so each row meets only its own head's key columns (the
/// zeros add exact zeros). Values: every row against both heads' value
/// columns, each keeping its own head's half. f32 out, as [`by_head`].
fn by_pair(f: &mut Func, a: Val, b: Val, contract: i64) -> Built<Val> {
    let ad = f.dims(a).to_vec();
    let bd = f.dims(b).to_vec();
    let (r, kvh, g, x) = (ad[0], ad[1], ad[2], ad[3]);
    let (c, hp) = (bd[1], kvh / 2);
    let bw = f.reshape(b, &[r, c, hp, 128])?;
    // `[.., 2 (j), .., 2 (j'), ..]`: true where the row's head `j` is the
    // column half `j'`.
    let own = |f: &mut Func, dims: &[i64], j: i64, jj: i64| -> Built<Val> {
        let a = f.iota(Elem::I32, dims, j);
        let b = f.iota(Elem::I32, dims, jj);
        f.compare(Cmp::Eq, a, b)
    };
    if contract == 2 {
        let a2 = f.reshape(a, &[r, hp, 2, g, x])?;
        let dims = [r, hp, 2, g, 2, x];
        let wide = f.broadcast(a2, &dims, &[0, 1, 2, 3, 5])?;
        let keep = own(f, &dims, 2, 4)?;
        let zero = f.const_f(f.elem(a), 0.0, &dims);
        let q = f.select(keep, wide, zero)?;
        let q = f.reshape(q, &[r, hp, 2 * g, 128])?;
        let s = f.dot_general_at(q, bw, &[0, 1], &[0, 2], &[3], &[3], Elem::F32, false)?;
        return f.reshape(s, &[r, kvh, g, c]);
    }
    let p = f.reshape(a, &[r, hp, 2 * g, x])?;
    let o = f.dot_general_at(p, bw, &[0, 1], &[0, 2], &[3], &[1], Elem::F32, false)?;
    let dims = [r, hp, 2, g, 2, 64];
    let o = f.reshape(o, &dims)?;
    let keep = own(f, &dims, 2, 4)?;
    let zero = f.const_f(Elem::F32, 0.0, &dims);
    let o = f.select(keep, o, zero)?;
    let o = f.reduce(o, &[4], Fold::Sum)?;
    f.reshape(o, &[r, kvh, g, 64])
}

fn init(f: &mut Func, stat: &[i64], dv: i64) -> [Val; 3] {
    let m = f.const_f(Elem::F32, f64::NEG_INFINITY, stat);
    let l = f.const_f(Elem::F32, 0.0, stat);
    let mut ad = stat.to_vec();
    ad.push(dv);
    let acc = f.const_f(Elem::F32, 0.0, &ad);
    [m, l, acc]
}

/// One chunk of the online softmax: raw scores `s` (4-d, keys last),
/// admission `[rb, c]`, value chunk `v`, state `(m, l, acc)`.
fn step(
    f: &mut Func,
    lay: Layout,
    h: &Heads,
    s: Val,
    lg: &Logits,
    scale: f64,
    v: Val,
    st: &[Val],
) -> Built<Vec<Val>> {
    let sd = f.dims(s).to_vec();
    let mut s = f.scale(s, scale)?;
    if let Some(b) = lg.bias {
        let rb = f.dims(b)[0];
        let c = f.dims(b)[2];
        let b = f.reshape(b, &[rb, h.kvh, h.g, c])?;
        let b = if lay == Layout::Blocked {
            f.transpose(b, &[1, 0, 2, 3])?
        } else {
            b
        };
        s = f.add(s, b)?;
    }
    if let Some(m) = lg.mult {
        let at = if lay == Layout::Blocked { 1 } else { 0 };
        let m = f.broadcast(m, &sd, &[at])?;
        s = f.mul(s, m)?;
    }
    let vmap: [i64; 2] = if lay == Layout::Blocked {
        [1, 3]
    } else {
        [0, 3]
    };
    let valid = f.broadcast(lg.valid, &sd, &vmap)?;
    let ninf = f.like_f(s, f64::NEG_INFINITY);
    let sm = f.select(valid, s, ninf)?;
    let cm = f.reduce(sm, &[3], Fold::Max)?;
    let (m0, l0, acc0) = (st[0], st[1], st[2]);
    let m = f.max(m0, cm)?;
    let floor = f.like_f(m, f64::NEG_INFINITY);
    let live = f.compare(Cmp::Gt, m, floor)?;
    let zero = f.like_f(m, 0.0);
    let ms = f.select(live, m, zero)?;
    let mb = f.broadcast(ms, &sd, &[0, 1, 2])?;
    let z = f.sub(s, mb)?;
    let e = f.exp(z);
    let zs = f.like_f(e, 0.0);
    let p = f.select(valid, e, zs)?;
    let dm = f.sub(m0, ms)?;
    let corr = f.exp(dm);
    let l = f.mul(l0, corr)?;
    let ps = f.reduce(p, &[3], Fold::Sum)?;
    let l = f.add(l, ps)?;
    let o = pv(f, lay, p, v)?;
    let ad = f.dims(acc0).to_vec();
    let cb = f.broadcast(corr, &ad, &[0, 1, 2])?;
    let acc = f.mul(acc0, cb)?;
    let acc = f.add(acc, o)?;
    Ok(vec![m, l, acc])
}

/// A finished reading: `o [R, kvh, g, dv]` f32 (zeros where nothing was
/// admitted), with the running max `m` and sum `l` (`[R, kvh, g]`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Flashed {
    pub o: Val,
    pub m: Val,
    pub l: Val,
}

fn finish(f: &mut Func, lay: Layout, st: &[Val]) -> Built<Flashed> {
    let (m, l, acc) = (st[0], st[1], st[2]);
    let zero = f.like_f(l, 0.0);
    let pos = f.compare(Cmp::Gt, l, zero)?;
    let one = f.like_f(l, 1.0);
    let inv = f.div(one, l)?;
    let inv = f.select(pos, inv, zero)?;
    let ad = f.dims(acc).to_vec();
    let ib = f.broadcast(inv, &ad, &[0, 1, 2])?;
    let o = f.mul(acc, ib)?;
    if lay == Layout::Blocked {
        Ok(Flashed {
            o: f.transpose(o, &[1, 0, 2, 3])?,
            m: f.transpose(m, &[1, 0, 2])?,
            l: f.transpose(l, &[1, 0, 2])?,
        })
    } else {
        Ok(Flashed { o, m, l })
    }
}

impl Flashed {
    /// The base-2 log-sum-exp `[R, kvh·g]` the GPU kernels publish (`-inf`
    /// where nothing was admitted).
    pub(crate) fn lse2(&self, f: &mut Func) -> Built<Val> {
        let zero = f.like_f(self.l, 0.0);
        let pos = f.compare(Cmp::Gt, self.l, zero)?;
        let lm = f.scale(self.m, std::f64::consts::LOG2_E)?;
        let ll = f.log(self.l);
        let ll = f.scale(ll, std::f64::consts::LOG2_E)?;
        let v = f.add(lm, ll)?;
        let ninf = f.like_f(v, f64::NEG_INFINITY);
        let v = f.select(pos, v, ninf)?;
        let d = f.dims(v).to_vec();
        f.reshape(v, &[d[0], d[1] * d[2]])
    }
}

// ------------------------------------------------------------------ drivers

/// Element budget of one chunk's largest intermediate.
const BUDGET: i64 = 1 << 24;

/// Query rows per block of the blocked driver.
const BLOCK_ROWS: i64 = 512;

/// Keys per chunk of the blocked driver.
const BLOCK_KEYS: i64 = 512;

/// The keys a per-row walk visits.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KeyList {
    /// Positions `0..span` of the row's lane, cut short at the largest `hi`.
    Range,
    /// An explicit `[R, n]` i32 position list (negative entries are skipped).
    Table(Val),
}

/// Every row gathers its own lane's keys (paged planes only).
pub(crate) fn per_row(
    f: &mut Func,
    planes: &Planes,
    h: &Heads,
    rule: &Rule,
    rows: &Rows,
    keys: KeyList,
    scale: f64,
) -> Built<Flashed> {
    let Planes::Paged { geom, .. } = *planes else {
        return bad("the per-row walk reads a paged pool");
    };
    let Some(lane) = rows.lane else {
        return bad("the per-row walk needs each row's lane");
    };
    let r = f.dims(rows.q)[0];
    let total = match keys {
        KeyList::Range => geom.span,
        KeyList::Table(t) => f.dims(t)[1],
    };
    let width = (h.kvh * (h.d + h.dv)).max(h.qh()).max(1);
    let c = pow2_floor((BUDGET / (r * width)).max(8)).min(round_up(total, 8));
    // A windowed row admits at most `window` keys: chunks of about that
    // width (whole pages), walked from the lowest admitted one, read the
    // window and not the whole span.
    let c = match (keys, rule.window, geom.ps) {
        (KeyList::Range, Some(w), ps) if ps > 0 => {
            let cw = round_up((w.max(1) as u64).next_power_of_two() as i64, ps);
            if cw < c && c % cw == 0 { cw } else { c }
        }
        _ => c,
    };
    if let (KeyList::Range, Some(w)) = (keys, rule.window)
        && let Some(walk) = PageWalk::of(f, planes, Some(lane), c)?
    {
        return per_row_window(f, planes, h, rule, rows, lane, &walk, c, w, scale);
    }
    let n = cdiv(total, c);
    let table = match keys {
        KeyList::Range => None,
        KeyList::Table(t) if n * c > total => {
            let neg = f.const_i(Elem::I32, -1, &[]);
            Some(f.pad(t, neg, &[0, 0], &[0, n * c - total], &[0, 0])?)
        }
        KeyList::Table(t) => Some(t),
    };
    let stat = [r, h.kvh, h.g];
    let st0 = init(f, &stat, h.dv);
    let span = geom.span;
    let walk = match table {
        None => PageWalk::of(f, planes, Some(lane), c)?,
        Some(_) => None,
    };
    let body = |f: &mut Func, ci: Val, st: &[Val]| -> Built<Vec<Val>> {
        let base = with_i(f, ci, c, Func::mul)?;
        let kp = match table {
            None => {
                let io = f.iota(Elem::I32, &[r, c], 1);
                let b = f.splat(base, &[r, c])?;
                f.add(io, b)?
            }
            Some(t) => {
                let zero = scalar_i(f, 0);
                f.dynamic_slice(t, &[zero, base], &[r, c])?
            }
        };
        let kc = clamp_i(f, kp, 0, span - 1)?;
        let lb = f.broadcast(lane, &[r, c], &[0])?;
        let x = with_i(f, lb, span, Func::mul)?;
        let x = f.add(x, kc)?;
        let paged = match walk {
            Some(w) => {
                let page = w.chunk(f, base)?;
                fetch_pages(f, planes, h, page)?
            }
            None => None,
        };
        let (k, v) = match paged {
            Some(kv) => kv,
            None => {
                let xf = f.reshape(x, &[r * c])?;
                fetch(f, planes, h, xf)?
            }
        };
        let (s, v) = if heads_by_slice(h) {
            let k = f.reshape(k, &[r, c, h.kvh * h.d])?;
            let v = f.reshape(v, &[r, c, h.kvh * h.dv])?;
            (by_head(f, rows.q, k, 2)?, v)
        } else {
            let k = f.reshape(k, &[r, c, h.kvh, h.d])?;
            let v = f.reshape(v, &[r, c, h.kvh, h.dv])?;
            let s = f.dot_general_at(rows.q, k, &[0, 1], &[0, 2], &[3], &[3], Elem::F32, false)?;
            (s, v)
        };
        let lo = f.broadcast(rows.lo, &[r, c], &[0])?;
        let hi = f.broadcast(rows.hi, &[r, c], &[0])?;
        let ge = f.compare(Cmp::Ge, kp, lo)?;
        let lt = f.compare(Cmp::Lt, kp, hi)?;
        let range = f.and(ge, lt)?;
        let kbase = match table {
            None => Some(base),
            Some(_) => None,
        };
        let lg = logits(f, rule, h, rows, x, kp, kbase, range)?;
        step(f, Layout::PerRow, h, s, &lg, scale, v, st)
    };
    let st = if n == 1 {
        let zero = scalar_i(f, 0);
        body(f, zero, &st0)?
    } else if table.is_none() {
        // Visit only the chunks between the lowest and the largest admitted
        // positions.
        let top = f.reduce(rows.hi, &[0], Fold::Max)?;
        let top = with_i(f, top, c - 1, Func::add)?;
        let top = with_i(f, top, c, Func::div)?;
        let bound = clamp_i(f, top, 0, n)?;
        let has = f.compare(Cmp::Lt, rows.lo, rows.hi)?;
        let big = f.like_i(rows.lo, i64::from(i32::MAX));
        let lo = f.select(has, rows.lo, big)?;
        let first = f.reduce(lo, &[0], Fold::Min)?;
        let first = with_i(f, first, c, Func::div)?;
        let first = clamp_i(f, first, 0, n)?;
        let mut carried = vec![first];
        carried.extend_from_slice(&st0);
        let out = f.while_loop(
            &carried,
            |f, a| f.compare(Cmp::Lt, a[0], bound),
            |f, a| {
                let next = with_i(f, a[0], 1, Func::add)?;
                let mut st = body(f, a[0], &a[1..])?;
                st.insert(0, next);
                Ok(st)
            },
        )?;
        out[1..].to_vec()
    } else {
        f.for_loop(n, &st0, body)?
    };
    finish(f, Layout::PerRow, &st)
}

/// The per-row walk of a sliding window: each row walks its own window,
/// from the page its window starts in, a fixed number of chunks (the window
/// and a page, in chunks of `c`), so rows at different positions read their
/// windows and not the span between the lowest and the highest of them.
#[allow(clippy::too_many_arguments)]
fn per_row_window(
    f: &mut Func,
    planes: &Planes,
    h: &Heads,
    rule: &Rule,
    rows: &Rows,
    lane: Val,
    walk: &PageWalk,
    c: i64,
    window: i64,
    scale: f64,
) -> Built<Flashed> {
    let Planes::Paged { geom, .. } = *planes else {
        return bad("the per-row walk reads a paged pool");
    };
    let r = f.dims(rows.q)[0];
    let span = geom.span;
    let ps = geom.ps;
    let np = walk.np;
    // Each row's first page of its window.
    let lo = clamp_i(f, rows.lo, 0, span - 1)?;
    let p0 = with_i(f, lo, ps, Func::div)?;
    let base = with_i(f, p0, ps, Func::mul)?;
    let n = cdiv(window + ps, c);
    // Every row's window pages, `[r, n · np]`, read out of the dense table
    // by index (the same expression in every layer over this pool, so the
    // program computes it once).
    let pages = geom.pages;
    let j = f.iota(Elem::I32, &[r, n * np], 1);
    let p0b = f.broadcast(p0, &[r, n * np], &[0])?;
    let at = f.add(j, p0b)?;
    let at = clamp_i(f, at, 0, pages - 1)?;
    let lb = f.broadcast(lane, &[r, n * np], &[0])?;
    let lb = with_i(f, lb, pages, Func::mul)?;
    let at = f.add(lb, at)?;
    let at = f.reshape(at, &[r * n * np])?;
    let table = take(f, geom.pt, at)?;
    let table = f.reshape(table, &[r, n * np])?;
    let mut st = init(f, &[r, h.kvh, h.g], h.dv).to_vec();
    for ci in 0..n {
        let start = with_i(f, base, ci * c, Func::add)?;
        let io = f.iota(Elem::I32, &[r, c], 1);
        let sb = f.broadcast(start, &[r, c], &[0])?;
        let kp = f.add(io, sb)?;
        let kc = clamp_i(f, kp, 0, span - 1)?;
        let lbc = f.broadcast(lane, &[r, c], &[0])?;
        let x = with_i(f, lbc, span, Func::mul)?;
        let x = f.add(x, kc)?;
        let page = f.slice(table, &[0, ci * np], &[r, (ci + 1) * np], &[1, 1])?;
        let page = f.reshape(page, &[r * np])?;
        let (k, v) = match fetch_pages(f, planes, h, page)? {
            Some(kv) => kv,
            None => return bad("the per-row window walk reads whole pages"),
        };
        let (s, v) = if heads_by_slice(h) {
            let k = f.reshape(k, &[r, c, h.kvh * h.d])?;
            let v = f.reshape(v, &[r, c, h.kvh * h.dv])?;
            (by_head(f, rows.q, k, 2)?, v)
        } else {
            let k = f.reshape(k, &[r, c, h.kvh, h.d])?;
            let v = f.reshape(v, &[r, c, h.kvh, h.dv])?;
            let s = f.dot_general_at(rows.q, k, &[0, 1], &[0, 2], &[3], &[3], Elem::F32, false)?;
            (s, v)
        };
        let lo = f.broadcast(rows.lo, &[r, c], &[0])?;
        let hi = f.broadcast(rows.hi, &[r, c], &[0])?;
        let ge = f.compare(Cmp::Ge, kp, lo)?;
        let lt = f.compare(Cmp::Lt, kp, hi)?;
        let range = f.and(ge, lt)?;
        let lg = logits(f, rule, h, rows, x, kp, None, range)?;
        st = step(f, Layout::PerRow, h, s, &lg, scale, v, &st)?;
    }
    finish(f, Layout::PerRow, &st)
}

/// Row blocks against the flat key axis `0..keys`, each block walking only
/// the chunks its rows' `[lo, hi)` span.
pub(crate) fn blocked(
    f: &mut Func,
    planes: &Planes,
    h: &Heads,
    rule: &Rule,
    rows: &Rows,
    keys: i64,
    scale: f64,
) -> Built<Flashed> {
    let r = f.dims(rows.q)[0];
    let qh = h.qh().max(1);
    let br = BLOCK_ROWS.min(round_up(r, 8));
    let nb = cdiv(r, br);
    let rp = nb * br;
    let rows_p = if rp > r {
        rows.map(f, &mut |f, v| pad_rows(f, v, rp - r))?
    } else {
        *rows
    };
    let width = (h.kvh * (h.d + h.dv)).max(1);
    let fit = pow2_floor((BUDGET / (br * qh)).min(BUDGET / width).max(8));
    let c = BLOCK_KEYS.min(fit).min(round_up(keys.max(1), 8));
    // Lanes narrower than a chunk: walk them a span at a time, so no chunk
    // straddles two lanes (the mask is then read as a window, below).
    let c = match rule.kp_span {
        Some(sp) if sp < c && sp % 8 == 0 => sp,
        _ => c,
    };
    let walk = PageWalk::of(f, planes, None, c)?;
    // The keys and values of the chunk at flat key `base`, `[c, kvh, d]`.
    let fetch_chunk = |f: &mut Func, base: Val| -> Built<(Val, Val)> {
        let paged = match walk {
            Some(w) => {
                let page = w.chunk(f, base)?;
                fetch_pages(f, planes, h, page)?
            }
            None => None,
        };
        match paged {
            Some(kv) => Ok(kv),
            None => {
                let io = f.iota(Elem::I32, &[c], 0);
                let b = f.splat(base, &[c])?;
                let x = f.add(io, b)?;
                fetch(f, planes, h, x)
            }
        }
    };
    // One chunk of the online softmax for the rows `rb` (a block of `br`).
    let attend_chunk = |f: &mut Func, rb: &Rows, base: Val, st: &[Val]| -> Built<Vec<Val>> {
        let (k, v) = fetch_chunk(f, base)?;
        let io = f.iota(Elem::I32, &[c], 0);
        let b = f.splat(base, &[c])?;
        let x = f.add(io, b)?;
        let xb = f.broadcast(x, &[br, c], &[1])?;
        let lo = f.broadcast(rb.lo, &[br, c], &[0])?;
        let hi = f.broadcast(rb.hi, &[br, c], &[0])?;
        let ge = f.compare(Cmp::Ge, xb, lo)?;
        let lt = f.compare(Cmp::Lt, xb, hi)?;
        let range = f.and(ge, lt)?;
        let (kp, kbase) = match rule.kp_span {
            // A chunk that never straddles two lanes' spans reads
            // positions `base mod span ..` of one lane: the mask is then
            // one window of its plane, not a gather per cell (rows of
            // other lanes are out of range anyway).
            Some(sp) if sp % c == 0 => {
                let kb = with_i(f, base, sp, Func::rem)?;
                let io = f.iota(Elem::I32, &[br, c], 1);
                let kbb = f.splat(kb, &[br, c])?;
                (f.add(io, kbb)?, Some(kb))
            }
            Some(sp) => (with_i(f, xb, sp, Func::rem)?, None),
            None => (xb, Some(base)),
        };
        let lg = logits(f, rule, h, rb, xb, kp, kbase, range)?;
        let s = f.dot_general_at(rb.q, k, &[1], &[1], &[3], &[2], Elem::F32, false)?;
        step(f, Layout::Blocked, h, s, &lg, scale, v, st)
    };
    let block = |f: &mut Func, rb: &Rows| -> Built<Flashed> {
        let has = f.compare(Cmp::Lt, rb.lo, rb.hi)?;
        let big = f.like_i(rb.lo, i64::from(i32::MAX));
        let lo = f.select(has, rb.lo, big)?;
        let klo = f.reduce(lo, &[0], Fold::Min)?;
        let none = f.like_i(rb.hi, 0);
        let hi = f.select(has, rb.hi, none)?;
        let khi = f.reduce(hi, &[0], Fold::Max)?;
        let c0 = with_i(f, klo, c, Func::div)?;
        let c1 = with_i(f, khi, c - 1, Func::add)?;
        let c1 = with_i(f, c1, c, Func::div)?;
        // Each row's chunk range `[lo / c, ⌈hi / c⌉)`: the walk steps to the
        // next chunk some row admits, skipping the gaps between the lanes
        // of a block (another lane's unused span, a window's past).
        let row_c0 = with_i(f, lo, c, Func::div)?;
        let row_c1 = with_i(f, hi, c - 1, Func::add)?;
        let row_c1 = with_i(f, row_c1, c, Func::div)?;
        let mut carried = vec![c0];
        carried.extend_from_slice(&init(f, &[h.kvh, br, h.g], h.dv));
        let out = f.while_loop(
            &carried,
            |f, a| f.compare(Cmp::Lt, a[0], c1),
            |f, a| {
                let ci = a[0];
                let base = with_i(f, ci, c, Func::mul)?;
                let mut st = attend_chunk(f, rb, base, &a[1..])?;
                let next = with_i(f, ci, 1, Func::add)?;
                let nb = f.splat(next, &[br])?;
                let cand = f.max(row_c0, nb)?;
                let open = f.compare(Cmp::Lt, cand, row_c1)?;
                let end = f.splat(c1, &[br])?;
                let cand = f.select(open, cand, end)?;
                let next = f.reduce(cand, &[0], Fold::Min)?;
                st.insert(0, next);
                Ok(st)
            },
        )?;
        finish(f, Layout::Blocked, &out[1..])
    };
    let whole = if nb == 1 {
        block(f, &rows_p)?
    } else {
        let st0 = [
            f.const_f(Elem::F32, 0.0, &[rp, h.kvh, h.g, h.dv]),
            f.const_f(Elem::F32, 0.0, &[rp, h.kvh, h.g]),
            f.const_f(Elem::F32, 0.0, &[rp, h.kvh, h.g]),
        ];
        let out = f.for_loop(nb, &st0, |f, bi, st| {
            let start = with_i(f, bi, br, Func::mul)?;
            let rb = rows_p.map(f, &mut |f, v| rows_at(f, v, start, br))?;
            let fl = block(f, &rb)?;
            let zero = scalar_i(f, 0);
            let o = f.dynamic_update_slice(st[0], fl.o, &[start, zero, zero, zero])?;
            let m = f.dynamic_update_slice(st[1], fl.m, &[start, zero, zero])?;
            let l = f.dynamic_update_slice(st[2], fl.l, &[start, zero, zero])?;
            Ok(vec![o, m, l])
        })?;
        Flashed {
            o: out[0],
            m: out[1],
            l: out[2],
        }
    };
    if rp == r {
        return Ok(whole);
    }
    Ok(Flashed {
        o: f.slice_axis(whole.o, 0, 0, r)?,
        m: f.slice_axis(whole.m, 0, 0, r)?,
        l: f.slice_axis(whole.l, 0, 0, r)?,
    })
}

// ------------------------------------------------------------ paged entries

pub(crate) fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("`{what}` is zero")));
    }
    Ok(v)
}

fn tables_agree(
    op: &'static str,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
) -> Result<(), Error> {
    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's position table is {:?}; this plan carries i32 positions",
                positions.dtype
            ),
        ));
    }
    if request_of_token.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire's owning-request table is {:?}; this plan carries an i32 request per token",
                request_of_token.dtype
            ),
        ));
    }
    for (what, t) in [("mask plane", mask), ("mask-enabled flags", mask_enabled)] {
        if !matches!(t.dtype, Dtype::U8 | Dtype::Bool) {
            return Err(refuse(
                op,
                format!("the fire's {what} are {:?}; this plan carries u8", t.dtype),
            ));
        }
    }
    if positions.elements() != request_of_token.elements() {
        return Err(refuse(
            op,
            format!(
                "the fire tables disagree: {} positions beside {} owning requests",
                positions.elements(),
                request_of_token.elements()
            ),
        ));
    }
    Ok(())
}

/// Validates the fire tables; the plan carries them into the attention.
/// `kv_len` is not read: the pool's CSR and the positions carry the lengths.
pub fn plan_decode(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<DecodePlan, Error> {
    let _ = (ctx, kv_len);
    tables_agree(
        "attention.plan_decode",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(DecodePlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

pub fn plan_prefill(
    ctx: &Ctx<'_>,
    kv_len: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    mask: Tensor,
    mask_enabled: Tensor,
    mask_stride: u32,
) -> Result<PrefillPlan, Error> {
    let _ = (ctx, kv_len);
    tables_agree(
        "attention.plan_prefill",
        positions,
        request_of_token,
        mask,
        mask_enabled,
    )?;
    Ok(PrefillPlan {
        positions,
        request_of_token,
        mask,
        mask_enabled,
        mask_stride,
    })
}

/// The fire tables an attention reads, whichever plan carried them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tables {
    pub positions: Tensor,
    pub request_of_token: Tensor,
    pub mask: Tensor,
    pub mask_enabled: Tensor,
    pub mask_stride: u32,
}

impl Tables {
    pub(crate) const fn decode(p: &DecodePlan) -> Self {
        Self {
            positions: p.positions,
            request_of_token: p.request_of_token,
            mask: p.mask,
            mask_enabled: p.mask_enabled,
            mask_stride: p.mask_stride,
        }
    }

    pub(crate) const fn prefill(p: &PrefillPlan) -> Self {
        Self {
            positions: p.positions,
            request_of_token: p.request_of_token,
            mask: p.mask,
            mask_enabled: p.mask_enabled,
            mask_stride: p.mask_stride,
        }
    }

    pub(crate) fn with_mask(mut self, mask: Tensor) -> Self {
        self.mask = mask;
        self
    }
}

pub(crate) fn window_extent(op: &'static str, window: Option<u32>) -> Result<i64, Error> {
    match window {
        None => Ok(0),
        Some(w) => Ok(i64::from(nonzero(op, "the sliding extent", w)?)),
    }
}

pub(crate) fn row_heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width", head_dim)?;
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide query row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

pub(crate) fn pool_heads(op: &'static str, pool: &KvPool, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width", head_dim)?;
    if pool.head_stride != u64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the head width {head_dim} is not the pool's head stride {}",
                pool.head_stride
            ),
        ));
    }
    if pool.seq_stride == 0 || !pool.seq_stride.is_multiple_of(pool.head_stride) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not a whole number of {head_dim}-wide kv heads",
                pool.seq_stride
            ),
        ));
    }
    u32::try_from(pool.seq_stride / pool.head_stride)
        .map_err(|_| refuse(op, "the pool's kv head count does not fit a u32"))
}

pub(crate) fn kv_heads_agree(
    op: &'static str,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
) -> Result<(), Error> {
    let spelled = pool_heads(op, pool, head_dim)?;
    if kv_heads != spelled {
        return Err(refuse(
            op,
            format!(
                "the stated kv head count {kv_heads} is not the {spelled} the pool's strides spell"
            ),
        ));
    }
    Ok(())
}

/// Refuses a pool this backend cannot page through.
pub(crate) fn pool_paging(op: &'static str, pool: &KvPool) -> Result<(), Error> {
    if pool.page_size <= 0 {
        return Err(refuse(op, "the kv page size is zero"));
    }
    nonzero(
        op,
        "the pool's per-lane page bound (max_pages)",
        pool.max_pages,
    )?;
    if pool.page_indptr.elements() < 2 {
        return Err(refuse(op, "the page CSR names no lane"));
    }
    if pool.page_indices.elements() == 0 {
        return Err(refuse(op, "the page CSR holds no page"));
    }
    for t in [pool.page_indptr, pool.page_indices] {
        if !matches!(t.dtype, Dtype::I32 | Dtype::U32) {
            return Err(refuse(
                op,
                format!("a page table is {:?}; it is i32", t.dtype),
            ));
        }
    }
    Ok(())
}

/// Reads the pool's paging into a dense page table.
pub(crate) fn read_geometry(cx: &mut Cx<'_>, pool: &KvPool) -> Result<Geom, Error> {
    let indptr = cx.read(pool.page_indptr)?;
    let indptr = flat_i32(cx, indptr)?;
    let indices = cx.read(pool.page_indices)?;
    let indices = flat_i32(cx, indices)?;
    Ok(geometry(
        cx,
        indptr,
        indices,
        i64::from(pool.page_size),
        i64::from(pool.max_pages),
    )?)
}

/// A per-row i32 table cut to the fire's `r` rows.
pub(crate) fn read_rows_i32(
    cx: &mut Cx<'_>,
    op: &'static str,
    t: Tensor,
    r: u32,
) -> Result<Val, Error> {
    if t.elements() < u64::from(r) {
        return Err(refuse(
            op,
            format!(
                "a per-row table holds {} entries for {r} rows",
                t.elements()
            ),
        ));
    }
    let v = cx.read(t)?;
    let v = flat_i32(cx, v)?;
    Ok(head(cx, v, i64::from(r))?)
}

/// How a paged attention walks its keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Walk {
    /// Each row gathers its own lane's keys.
    PerRow,
    /// Row blocks over the flat key axis.
    Blocked,
    /// Per-row when rows are few per lane (at most two), blocked otherwise.
    Auto,
}

/// One paged attention call, all semantics stated.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Paged {
    pub op: &'static str,
    pub q: Tensor,
    pub tables: Tables,
    /// Read the custom mask (rel reads none).
    pub masked: bool,
    pub window: Option<u32>,
    pub causal: bool,
    pub head_dim: u32,
    pub sm_scale: f32,
    pub o: Tensor,
    pub lse: Option<Tensor>,
    pub walk: Walk,
    /// `(selection [rows, top_k] i32 block ids, ratio)`.
    pub selection: Option<(Tensor, u32)>,
    /// `(bias [rows, qh · extent] f32, extent, log_floor, log_alpha)`.
    pub rel: Option<(Tensor, u32, u32, f32)>,
}

pub(crate) fn attend(ctx: &Ctx<'_>, call: Paged, pool: &KvPool) -> Result<(), Error> {
    let op = call.op;
    expect(op, call.q, &[Dtype::Bf16])?;
    expect(op, pool.keys, &[Dtype::Bf16])?;
    expect(op, pool.values, &[Dtype::Bf16])?;
    pool_paging(op, pool)?;
    let kv_heads = pool_heads(op, pool, call.head_dim)?;
    let q_heads = row_heads(op, call.q.width, call.head_dim)?;
    if !q_heads.is_multiple_of(kv_heads) {
        return Err(refuse(
            op,
            format!(
                "{q_heads} query heads are not a whole number of the pool's {kv_heads} kv heads"
            ),
        ));
    }
    let width = kv_heads * call.head_dim;
    if pool.keys.width != width || pool.values.width != width {
        return Err(refuse(
            op,
            format!(
                "the pool planes are {} and {} wide; {kv_heads} kv heads of {} want {width}",
                pool.keys.width, pool.values.width, call.head_dim
            ),
        ));
    }
    let rows = nonzero(op, "rows", call.q.rows)?;
    if call.o.rows != rows || call.o.width != call.q.width {
        return Err(refuse(
            op,
            "the answer is not one row per query row, as wide as the query",
        ));
    }
    if let Some(lse) = call.lse
        && (lse.dtype != Dtype::F32 || lse.rows != rows || lse.width != q_heads)
    {
        return Err(refuse(
            op,
            "the log-sum-exp plane is one f32 per head per row",
        ));
    }
    let window = window_extent(op, call.window)?;
    let t = call.tables;
    if call.masked {
        if !matches!(t.mask.dtype, Dtype::U8 | Dtype::Bool) {
            return Err(refuse(
                op,
                format!("the mask is {:?}; a mask plane is u8", t.mask.dtype),
            ));
        }
        if t.mask.rows < rows || t.mask_stride > t.mask.width {
            return Err(refuse(
                op,
                format!(
                    "the mask plane is {}x{}; this fire reads {rows} rows of {} columns",
                    t.mask.rows, t.mask.width, t.mask_stride
                ),
            ));
        }
    }
    if let Some((sel, ratio)) = call.selection {
        if call.window.is_some() {
            return Err(refuse(
                op,
                "a selection and a sliding window both answer which keys a row reads",
            ));
        }
        nonzero(op, "the block width this reader expands", ratio)?;
        nonzero(op, "the selection budget", sel.width)?;
        if sel.dtype != Dtype::I32 || sel.rows < rows {
            return Err(refuse(
                op,
                "the selection is one i32 block-id row per query row",
            ));
        }
    }
    if let Some((bias, extent, _, _)) = call.rel {
        nonzero(op, "the relative extent", extent)?;
        if bias.dtype != Dtype::F32
            || bias.rows < rows
            || u64::from(bias.width) != u64::from(q_heads) * u64::from(extent)
        {
            return Err(refuse(
                op,
                format!(
                    "the relative-bias table is [{}, {}]; this fire wants [{rows}, {q_heads} x {extent}]",
                    bias.rows, bias.width
                ),
            ));
        }
    }
    let heads = Heads {
        kvh: i64::from(kv_heads),
        g: i64::from(q_heads / kv_heads),
        d: i64::from(call.head_dim),
        dv: i64::from(call.head_dim),
    };
    let r = i64::from(rows);
    ctx.emit(&mut |cx| {
        let geom = read_geometry(cx, pool)?;
        let walk = match call.walk {
            _ if call.selection.is_some() => Walk::PerRow,
            Walk::Auto if r <= 2 * geom.lanes => Walk::PerRow,
            Walk::Auto => Walk::Blocked,
            w => w,
        };
        let qpos = read_rows_i32(cx, op, t.positions, rows)?;
        let req = read_rows_i32(cx, op, t.request_of_token, rows)?;
        let enabled = if call.masked {
            Some(read_rows_i32(cx, op, t.mask_enabled, rows)?)
        } else {
            None
        };
        let mask = if call.masked {
            let m = cx.read(t.mask)?;
            Some(cx.slice_axis(m, 0, 0, r)?)
        } else {
            None
        };
        let q = cx.read(call.q)?;
        let keys = cx.read(pool.keys)?;
        let values = cx.read(pool.values)?;
        let f = cx.func();
        let q4 = f.reshape(q, &[r, heads.kvh, heads.g, heads.d])?;
        let lane = geom.lane_of(f, req)?;
        let cap = geom.capacity(f, lane)?;
        // Window start.
        let lo = if window > 0 {
            let past = cmp_i(f, Cmp::Ge, qpos, window)?;
            let s = with_i(f, qpos, window - 1, Func::sub)?;
            let z = f.like_i(qpos, 0);
            f.select(past, s, z)?
        } else {
            f.like_i(qpos, 0)
        };
        // Causal end (exclusive), or the lane's whole capacity for a wide
        // row (non-causal, or a row whose mask flag is 2).
        let upto = with_i(f, qpos, 1, Func::add)?;
        let causal_hi = f.min(upto, cap)?;
        let hi = match (call.causal, enabled) {
            (false, _) => cap,
            (true, Some(en)) => {
                let wide = cmp_i(f, Cmp::Eq, en, 2)?;
                f.select(wide, cap, causal_hi)?
            }
            (true, None) => causal_hi,
        };
        let hi = f.max(hi, lo)?;
        let mut rule = Rule {
            bound: i64::from(t.mask_stride),
            window: (window > 0).then_some(window),
            ..Rule::default()
        };
        let planes = Planes::Paged {
            keys,
            values,
            geom,
            latent: false,
        };
        let (lo_w, hi_w) = if walk == Walk::Blocked {
            let base = with_i(f, lane, geom.span, Func::mul)?;
            rule.kp_span = Some(geom.span);
            (f.add(base, lo)?, f.add(base, hi)?)
        } else {
            (lo, hi)
        };
        let mut rs = Rows::new(q4, lo_w, hi_w);
        rs.lane = Some(lane);
        if call.masked {
            rs.enabled = enabled;
            rs.mask = mask;
        }
        let rel_plane = match call.rel {
            Some((bias, extent, floor, alpha)) => {
                rule.rel = Some((i64::from(extent), floor, alpha));
                rs.qpos = Some(qpos);
                Some(bias)
            }
            None => None,
        };
        let selection = call.selection;
        let _ = f;
        if let Some(bias) = rel_plane {
            let b = cx.read(bias)?;
            rs.rel = Some(cx.slice_axis(b, 0, 0, r)?);
        }
        let sel = match selection {
            Some((sel, ratio)) => {
                let s = cx.read(sel)?;
                let s = cx.slice_axis(s, 0, 0, r)?;
                Some((cx.convert(s, Elem::I32), i64::from(ratio)))
            }
            None => None,
        };
        let f = cx.func();
        let scale = f64::from(call.sm_scale);
        let flashed = match (walk, sel) {
            (_, Some((sel, ratio))) => {
                let table = selected_positions(f, sel, qpos, ratio)?;
                per_row(f, &planes, &heads, &rule, &rs, KeyList::Table(table), scale)?
            }
            (Walk::Blocked, None) => blocked(
                f,
                &planes,
                &heads,
                &rule,
                &rs,
                geom.lanes * geom.span,
                scale,
            )?,
            _ => per_row(f, &planes, &heads, &rule, &rs, KeyList::Range, scale)?,
        };
        let lse = match call.lse {
            Some(_) => Some(flashed.lse2(f)?),
            None => None,
        };
        cx.write(call.o, flashed.o)?;
        if let (Some(t), Some(v)) = (call.lse, lse) {
            cx.write(t, v)?;
        }
        Ok(())
    })
}

/// The key positions a block selection names, per row: the named blocks
/// (`ratio` cells each; ids outside `[0, (q_pos+1)/ratio)` skipped as `-1`),
/// then the open block the selection cannot name.
fn selected_positions(f: &mut Func, sel: Val, qpos: Val, ratio: i64) -> Built<Val> {
    let d = f.dims(sel).to_vec();
    let (r, top_k) = (d[0], d[1]);
    let n = top_k * ratio + ratio - 1;
    let sel = f.broadcast(sel, &[r, top_k, ratio], &[0, 1])?;
    let sel = f.reshape(sel, &[r, top_k * ratio])?;
    let neg = f.const_i(Elem::I32, -1, &[]);
    let sel = f.pad(sel, neg, &[0, 0], &[0, ratio - 1], &[0, 0])?;
    let upto = with_i(f, qpos, 1, Func::add)?;
    let nblocks = with_i(f, upto, ratio, Func::div)?;
    let tk = f.like_i(nblocks, top_k);
    let blocks = f.min(tk, nblocks)?;
    let sel_end = with_i(f, blocks, ratio, Func::mul)?;
    let dims = [r, n];
    let nb = f.broadcast(nblocks, &dims, &[0])?;
    let se = f.broadcast(sel_end, &dims, &[0])?;
    let idx = f.iota(Elem::I32, &dims, 1);
    let within = with_i(f, idx, ratio, Func::rem)?;
    let ge = cmp_i(f, Cmp::Ge, sel, 0)?;
    let lt = f.compare(Cmp::Lt, sel, nb)?;
    let named = f.and(ge, lt)?;
    let kp = with_i(f, sel, ratio, Func::mul)?;
    let kp = f.add(kp, within)?;
    let skip = f.like_i(kp, -1);
    let kp_sel = f.select(named, kp, skip)?;
    let tail = with_i(f, nb, ratio, Func::mul)?;
    let past = f.sub(idx, se)?;
    let tail = f.add(tail, past)?;
    let in_sel = f.compare(Cmp::Lt, idx, se)?;
    f.select(in_sel, kp_sel, tail)
}

fn plain(
    op: &'static str,
    q: Tensor,
    tables: Tables,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    walk: Walk,
) -> Paged {
    Paged {
        op,
        q,
        tables,
        masked: true,
        window,
        causal: true,
        head_dim,
        sm_scale,
        o,
        lse,
        walk,
        selection: None,
        rel: None,
    }
}

pub fn decode(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    let call = plain(
        "attention.decode",
        q,
        Tables::decode(plan),
        window,
        head_dim,
        sm_scale,
        o,
        None,
        Walk::PerRow,
    );
    attend(ctx, call, pool)
}

pub fn decode_lse(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    let call = plain(
        "attention.decode_lse",
        q,
        Tables::decode(plan),
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        Walk::PerRow,
    );
    attend(ctx, call, pool)
}

pub fn prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let call = plain(
        OP,
        q.data,
        Tables::prefill(plan),
        window,
        head_dim,
        sm_scale,
        o,
        None,
        Walk::Auto,
    );
    attend(ctx, call, pool)
}

pub fn prefill_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let call = plain(
        OP,
        q.data,
        Tables::prefill(plan),
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        Walk::Auto,
    );
    attend(ctx, call, pool)
}

/// A prefill whose custom mask is `mask` rather than the plan's (the plan's
/// per-row enable flags still gate it). Causal; see
/// [`super::arbiter::masked`] for the non-causal form.
pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    let tables = Tables::prefill(plan).with_mask(mask);
    let call = plain(
        "attention.masked",
        q.data,
        tables,
        window,
        head_dim,
        sm_scale,
        o,
        None,
        Walk::Auto,
    );
    attend(ctx, call, pool)
}

/// Selected-block decode (DeepSeek sparse attention): `selection` names
/// `ratio`-wide key blocks per row.
pub fn decode_selected(
    ctx: &Ctx<'_>,
    q: Tensor,
    plan: &DecodePlan,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    ratio: u32,
    o: Tensor,
) -> Result<(), Error> {
    let mut call = plain(
        "attention.decode_selected",
        q,
        Tables::decode(plan),
        window,
        head_dim,
        sm_scale,
        o,
        None,
        Walk::PerRow,
    );
    call.selection = Some((selection, ratio));
    attend(ctx, call, pool)
}

pub fn prefill_selected(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    ratio: u32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_selected";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    let mut call = plain(
        OP,
        q.data,
        Tables::prefill(plan),
        window,
        head_dim,
        sm_scale,
        o,
        None,
        Walk::PerRow,
    );
    call.selection = Some((selection, ratio));
    attend(ctx, call, pool)
}

/// Folds a per-head attention sink into an lse-carrying reading, in place:
/// `o *= σ(lse · ln 2 − sink[h])` where the lse is finite.
pub fn sink(
    ctx: &Ctx<'_>,
    o: Tensor,
    lse: Tensor,
    sink: Tensor,
    head_dim: u32,
) -> Result<(), Error> {
    const OP: &str = "attention.sink";
    expect(OP, o, &[Dtype::Bf16])?;
    let heads = row_heads(OP, o.width, head_dim)?;
    if lse.dtype != Dtype::F32 || lse.rows != o.rows || lse.width != heads {
        return Err(refuse(
            OP,
            "the log-sum-exp plane is one f32 per head per row",
        ));
    }
    if sink.elements() < u64::from(heads) {
        return Err(refuse(OP, "the sink bank is not one value per head"));
    }
    let (r, h, d) = (i64::from(o.rows), i64::from(heads), i64::from(head_dim));
    ctx.emit(&mut |cx| {
        let ov = cx.read_f32(o)?;
        let lv = cx.read_f32(lse)?;
        let sv = cx.read_f32(sink)?;
        let n = cx.ty(sv).elements();
        let sv = cx.reshape(sv, &[n])?;
        let sv = cx.slice(sv, &[0], &[h], &[1])?;
        let sv = cx.broadcast(sv, &[r, h], &[1])?;
        let x = cx.scale(lv, std::f64::consts::LN_2)?;
        let x = cx.sub(x, sv)?;
        let s = cx.sigmoid(x);
        let fin = cx.is_finite(lv);
        let one = cx.like_f(s, 1.0);
        let s = cx.select(fin, s, one)?;
        let s = cx.broadcast(s, &[r, h, d], &[0, 1])?;
        let ov = cx.reshape(ov, &[r, h, d])?;
        let y = cx.mul(ov, s)?;
        cx.write(o, y)
    })
}

pub fn merge_lse(
    ctx: &Ctx<'_>,
    o1: Tensor,
    lse1: Tensor,
    o2: Tensor,
    lse2: Tensor,
    heads: u32,
    head_dim: u32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    super::merge::merge_lse(ctx, o1, lse1, o2, lse2, heads, head_dim, o, lse)
}

/// `x = cap · tanh(x / cap)`, in place.
pub fn logit_softcap(ctx: &Ctx<'_>, x: Tensor, cap: f32) -> Result<(), Error> {
    const OP: &str = "attention.logit_softcap";
    expect(OP, x, &[Dtype::Bf16])?;
    if cap == 0.0 {
        return Err(refuse(OP, "the cap is zero"));
    }
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let v = cx.scale(v, 1.0 / f64::from(cap))?;
        let v = cx.tanh(v);
        let v = cx.scale(v, f64::from(cap))?;
        cx.write(x, v)
    })
}

/// Scatters `src` rows into `plane` at `write_page · page_size +
/// write_offset`; a row whose page or offset is out of range (a padded row)
/// is dropped.
pub(crate) fn scatter_slots(
    cx: &mut Cx<'_>,
    op: &'static str,
    pairs: &[(Tensor, Tensor)],
    page_size: i32,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    let Some(&(first, _)) = pairs.first() else {
        return Ok(());
    };
    let n = first.rows;
    let page = read_rows_i32(cx, op, write_page, n)?;
    let off = read_rows_i32(cx, op, write_offset, n)?;
    for &(src, plane) in pairs {
        let ps = i64::from(page_size);
        let slots = i64::from(plane.rows);
        let pages = slots / ps;
        let f = cx.func();
        let a = cmp_i(f, Cmp::Ge, page, 0)?;
        let b = cmp_i(f, Cmp::Lt, page, pages)?;
        let c = cmp_i(f, Cmp::Ge, off, 0)?;
        let d = cmp_i(f, Cmp::Lt, off, ps)?;
        let ok = f.and(a, b)?;
        let ok = f.and(ok, c)?;
        let ok = f.and(ok, d)?;
        let s = with_i(f, page, ps, Func::mul)?;
        let s = f.add(s, off)?;
        let out = f.like_i(s, slots);
        let slot = f.select(ok, s, out)?;
        let rows = cx.read(src)?;
        let table = cx.read(plane)?;
        let elem = cx.elem(table);
        let rows = cx.convert(rows, elem);
        let new = cx.put_rows(table, slot, rows, crate::hlo::Combine::Set)?;
        cx.write(plane, new)?;
    }
    Ok(())
}

fn head_split(op: &'static str, pool: &KvPool, row: u32) -> Result<(), Error> {
    let hd = u32::try_from(pool.head_stride)
        .ok()
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "the pool's head stride {} spells no head width",
                    pool.head_stride
                ),
            )
        })?;
    if row == 0 || !row.is_multiple_of(hd) {
        return Err(refuse(
            op,
            format!("the {row}-wide appended row does not divide by the head stride {hd}"),
        ));
    }
    if pool.seq_stride != u64::from(row) {
        return Err(refuse(
            op,
            format!(
                "the pool's sequence stride {} is not the {row}-wide row this appender writes",
                pool.seq_stride
            ),
        ));
    }
    Ok(())
}

fn append(
    ctx: &Ctx<'_>,
    op: &'static str,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    expect(op, k, &[Dtype::Bf16])?;
    if pool.page_size <= 0 {
        return Err(refuse(op, "the kv page size is zero"));
    }
    if v.rows != k.rows || v.width != k.width {
        return Err(refuse(
            op,
            "the value plane is appended beside the key plane, one rectangle",
        ));
    }
    head_split(op, pool, k.width)?;
    if pool.keys.width != k.width || pool.values.width != v.width {
        return Err(refuse(
            op,
            "the appended rows are not as wide as the pool planes",
        ));
    }
    ctx.emit(&mut |cx| {
        scatter_slots(
            cx,
            op,
            &[(k, pool.keys), (v, pool.values)],
            pool.page_size,
            write_page,
            write_offset,
        )
    })
}

pub fn kv_append(
    ctx: &Ctx<'_>,
    k: Tensor,
    v: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append(
        ctx,
        "attention.kv_append",
        k,
        v,
        pool,
        write_page,
        write_offset,
    )
}

pub fn kv_append_shared(
    ctx: &Ctx<'_>,
    plane: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    append(
        ctx,
        "attention.kv_append_shared",
        plane,
        plane,
        pool,
        write_page,
        write_offset,
    )
}
