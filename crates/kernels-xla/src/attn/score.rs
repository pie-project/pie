//! Attn-score capture: the mean softmax row of each request's last
//! observed query rows, per head, into an f32 slab (eviction policies and
//! interpretability read it).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::paged::{
    Heads, PageWalk, Planes, by_head, cdiv, clamp_i, fetch, fetch_pages, flat_i32, geometry, head,
    heads_by_slice, kv_heads_agree, nonzero, pool_paging, round_up, row_heads, take, with_i,
};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, Val};
use crate::tensor::{KvPool, Operand, RaggedTensor, Stored};

/// For request `r < requests` (lane `r` of the pool and segment `r` of
/// `q.indptr`) and query head `h`, slab row
/// `(lane_offset + r) · plane_stride + plane + h` becomes the mean, over the
/// request's last `min(observe, q_len)` rows, of the causal softmax over its
/// keys (keys past `kv_max` are in the denominator but not written; the rest
/// of the row is zero).
pub fn capture(
    f: &mut Func,
    q: RaggedTensor<Operand>,
    positions: Operand,
    keys: Operand,
    page_indptr: Operand,
    page_indices: Operand,
    pool: &KvPool<impl Stored>,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    observe: u32,
    lane_offset: u32,
    plane_stride: u32,
    plane: u32,
    kv_max: u32,
    requests: u32,
    scores: Operand,
    tuning: crate::tuning::Tuning,
) -> Result<Val, Error> {
    const OP: &str = "attn.score_capture";
    if window.is_some() {
        return Err(refuse(
            OP,
            "a sliding window is stated, and a windowed row is not the softmax over the request's keys",
        ));
    }
    if q.data.dtype != Dtype::Bf16 || keys.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: q.data.dtype,
        });
    }
    if scores.dtype != Dtype::F32 || scores.width != kv_max {
        return Err(refuse(
            OP,
            "the score slab is an f32 rectangle one kv ceiling wide",
        ));
    }
    if positions.dtype != Dtype::I32 || q.indptr.dtype != Dtype::I32 {
        return Err(refuse(OP, "the positions and the qo indptr are i32"));
    }
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    pool_paging(OP, pool)?;
    let qh = row_heads(OP, q.data.width, head_dim)?;
    if !qh.is_multiple_of(kv_heads) {
        return Err(refuse(
            OP,
            format!("{qh} query heads do not group over {kv_heads} kv heads"),
        ));
    }
    nonzero(OP, "the observation window", observe)?;
    nonzero(OP, "the kv ceiling", kv_max)?;
    nonzero(OP, "the requests", requests)?;
    if plane + qh > plane_stride
        || u64::from(lane_offset + requests) * u64::from(plane_stride) > u64::from(scores.rows)
    {
        return Err(refuse(OP, "the slab does not seat these planes"));
    }
    if q.indptr.elements() < u64::from(requests) + 1
        || page_indptr.elements() < u64::from(requests) + 1
    {
        return Err(refuse(
            OP,
            "the indptr tables name fewer segments than the requests",
        ));
    }
    let h = Heads {
        kvh: i64::from(kv_heads),
        g: i64::from(qh / kv_heads),
        d: i64::from(head_dim),
        dv: i64::from(head_dim),
    };
    let (nr, ob, rows) = (
        i64::from(requests),
        i64::from(observe),
        i64::from(q.data.rows),
    );
    let kmax = i64::from(kv_max);
    let scale = f64::from(sm_scale);
    let ids: Vec<i64> = (0..nr)
        .flat_map(|r| {
            (0..i64::from(qh)).map(move |hh| {
                (i64::from(lane_offset) + r) * i64::from(plane_stride) + i64::from(plane) + hh
            })
        })
        .collect();
    let indptr = page_indptr.val;
    let indptr = flat_i32(f, indptr)?;
    let indptr = head(f, indptr, nr + 1)?;
    let indices = page_indices.val;
    let indices = flat_i32(f, indices)?;
    let qo = q.indptr.val;
    let qo = flat_i32(f, qo)?;
    let pos = positions.val;
    let pos = flat_i32(f, pos)?;
    let qv = q.data.val;
    let keys = keys.val;
    let slab = scores.val;
    let budget = tuning.chunks.budget;
    let fq = &mut *f;
    let geom = geometry(
        fq,
        indptr,
        indices,
        i64::from(pool.page_size),
        i64::from(pool.max_pages),
    )?;
    let span = geom.span;
    let qo_lo = fq.slice(qo, &[0], &[nr], &[1])?;
    let qo_hi = fq.slice(qo, &[1], &[nr + 1], &[1])?;
    let qlen = fq.sub(qo_hi, qo_lo)?;
    let nrows = clamp_i(fq, qlen, 0, ob)?;
    let d2 = [nr, ob];
    let w = fq.iota(Elem::I32, &d2, 1);
    let hib = fq.broadcast(qo_hi, &d2, &[0])?;
    let nb = fq.broadcast(nrows, &d2, &[0])?;
    let at = fq.sub(hib, nb)?;
    let at = fq.add(at, w)?;
    let live = fq.compare(Cmp::Lt, w, nb)?;
    let at = clamp_i(fq, at, 0, rows - 1)?;
    let qpos = take(fq, pos, at)?;
    let lanes = fq.iota(Elem::I32, &[nr], 0);
    let cap = geom.capacity(fq, lanes)?;
    let capb = fq.broadcast(cap, &d2, &[0])?;
    let upto = with_i(fq, qpos, 1, Func::add)?;
    let lim = fq.min(upto, capb)?;
    let zero = fq.like_i(lim, 0);
    let lim = fq.select(live, lim, zero)?;
    let lim = fq.max(lim, zero)?;
    // Observed q rows, [nr, kvh, ob * g, d].
    let af = fq.reshape(at, &[nr * ob])?;
    let qr = fq.take_rows(qv, af)?;
    let q4 = if ob == 1 {
        fq.reshape(qr, &[nr, h.kvh, h.g, h.d])?
    } else {
        let q5 = fq.reshape(qr, &[nr, ob, h.kvh, h.g, h.d])?;
        let qt = fq.transpose(q5, &[0, 2, 1, 3, 4])?;
        fq.reshape(qt, &[nr, h.kvh, ob * h.g, h.d])?
    };
    let planes = Planes::Paged {
        keys,
        values: keys,
        geom,
        latent: false,
    };
    let c = {
        let width = (h.kvh * h.d).max(ob * h.qh());
        let c = (budget / (nr * width)).max(8);
        let c = if c <= 1 {
            1
        } else {
            1i64 << (63 - c.leading_zeros())
        };
        c.min(round_up(span, 8))
    };
    let walk = PageWalk::of(fq, &planes, Some(lanes), c)?;
    let by_slice = heads_by_slice(&h);
    // Scores of chunk `ci`: [nr, kvh, ob, g, c] and admission.
    let chunk = |fq: &mut Func,
                 ci: crate::hlo::Val|
     -> crate::hlo::Built<(crate::hlo::Val, crate::hlo::Val)> {
        let base = with_i(fq, ci, c, Func::mul)?;
        let io = fq.iota(Elem::I32, &[nr, c], 1);
        let bb = fq.splat(base, &[nr, c])?;
        let kp = fq.add(io, bb)?;
        let paged = match walk {
            Some(w) => {
                let page = w.chunk(fq, base)?;
                fetch_pages(fq, &planes, &h, page)?
            }
            None => None,
        };
        let (k, _) = match paged {
            Some(kv) => kv,
            None => {
                let kc = clamp_i(fq, kp, 0, span - 1)?;
                let lb = fq.iota(Elem::I32, &[nr, c], 0);
                let x = with_i(fq, lb, span, Func::mul)?;
                let x = fq.add(x, kc)?;
                let xf = fq.reshape(x, &[nr * c])?;
                fetch(fq, &planes, &h, xf)?
            }
        };
        let s = if by_slice {
            let k = fq.reshape(k, &[nr, c, h.kvh * h.d])?;
            by_head(fq, q4, k, 2)?
        } else {
            let k = fq.reshape(k, &[nr, c, h.kvh, h.d])?;
            fq.dot_general_at(q4, k, &[0, 1], &[0, 2], &[3], &[3], Elem::F32, false)?
        };
        let sd = [nr, h.kvh, ob, h.g, c];
        let s = fq.reshape(s, &sd)?;
        let s = fq.scale(s, scale)?;
        let kpb = fq.broadcast(kp, &[nr, ob, c], &[0, 2])?;
        let lb = fq.broadcast(lim, &[nr, ob, c], &[0, 1])?;
        let ok = fq.compare(Cmp::Lt, kpb, lb)?;
        let ok = fq.broadcast(ok, &sd, &[0, 2, 4])?;
        Ok((s, ok))
    };
    let stat = [nr, h.kvh, ob, h.g];
    let n1 = cdiv(span, c);
    // Row weights: 1 / rows over the observed rows.
    let nf = fq.convert(nrows, Elem::F32);
    let one_n = fq.like_f(nf, 1.0);
    let nf = fq.max(nf, one_n)?;
    let wr = fq.div(one_n, nf)?;
    let wr = fq.broadcast(wr, &d2, &[0])?;
    let zw = fq.like_f(wr, 0.0);
    let wr = fq.select(live, wr, zw)?;
    let kc = kmax.min(span);
    let mean = if n1 == 1 {
        let zero = fq.const_i(Elem::I32, 0, &[]);
        let (s, ok) = chunk(fq, zero)?;
        let ninf = fq.like_f(s, f64::NEG_INFINITY);
        let sm = fq.select(ok, s, ninf)?;
        let m = fq.reduce(sm, &[4], Fold::Max)?;
        let floor = fq.like_f(m, f64::NEG_INFINITY);
        let fin = fq.compare(Cmp::Gt, m, floor)?;
        let z = fq.like_f(m, 0.0);
        let ms = fq.select(fin, m, z)?;
        let sd = fq.dims(s).to_vec();
        let mb = fq.broadcast(ms, &sd, &[0, 1, 2, 3])?;
        let e = fq.sub(s, mb)?;
        let e = fq.exp(e);
        let ze = fq.like_f(e, 0.0);
        let p = fq.select(ok, e, ze)?;
        let l = fq.reduce(p, &[4], Fold::Sum)?;
        let zl = fq.like_f(l, 0.0);
        let pos_l = fq.compare(Cmp::Gt, l, zl)?;
        let one = fq.like_f(l, 1.0);
        let inv = fq.div(one, l)?;
        let inv = fq.select(pos_l, inv, zl)?;
        let wb = fq.broadcast(wr, &stat, &[0, 2])?;
        let iw = fq.mul(inv, wb)?;
        let iwb = fq.broadcast(iw, &sd, &[0, 1, 2, 3])?;
        let p = fq.mul(p, iwb)?;
        let mean = if ob == 1 {
            fq.reshape(p, &[nr, h.kvh, h.g, c])?
        } else {
            fq.reduce(p, &[2], Fold::Sum)?
        };
        fq.slice_axis(mean, 3, 0, kc)?
    } else {
        // Pass 1: the softmax max and denominator over every admitted key.
        let m0 = fq.const_f(Elem::F32, f64::NEG_INFINITY, &stat);
        let l0 = fq.const_f(Elem::F32, 0.0, &stat);
        let st = fq.for_loop(n1, &[m0, l0], |fq, ci, st| {
            let (s, ok) = chunk(fq, ci)?;
            let ninf = fq.like_f(s, f64::NEG_INFINITY);
            let sm = fq.select(ok, s, ninf)?;
            let cm = fq.reduce(sm, &[4], Fold::Max)?;
            let m = fq.max(st[0], cm)?;
            let floor = fq.like_f(m, f64::NEG_INFINITY);
            let fin = fq.compare(Cmp::Gt, m, floor)?;
            let z = fq.like_f(m, 0.0);
            let ms = fq.select(fin, m, z)?;
            let sd = fq.dims(s).to_vec();
            let mb = fq.broadcast(ms, &sd, &[0, 1, 2, 3])?;
            let e = fq.sub(s, mb)?;
            let e = fq.exp(e);
            let ze = fq.like_f(e, 0.0);
            let p = fq.select(ok, e, ze)?;
            let ps = fq.reduce(p, &[4], Fold::Sum)?;
            let dm = fq.sub(st[0], ms)?;
            let corr = fq.exp(dm);
            let l = fq.mul(st[1], corr)?;
            let l = fq.add(l, ps)?;
            Ok(vec![m, l])
        })?;
        let (m, l) = (st[0], st[1]);
        let floor = fq.like_f(m, f64::NEG_INFINITY);
        let fin = fq.compare(Cmp::Gt, m, floor)?;
        let z = fq.like_f(m, 0.0);
        let m = fq.select(fin, m, z)?;
        let zl = fq.like_f(l, 0.0);
        let pos_l = fq.compare(Cmp::Gt, l, zl)?;
        let one = fq.like_f(l, 1.0);
        let inv = fq.div(one, l)?;
        let inv = fq.select(pos_l, inv, zl)?;
        let wb = fq.broadcast(wr, &stat, &[0, 2])?;
        let iw = fq.mul(inv, wb)?;
        // Pass 2: the mean probability of the first `kv_max` keys.
        let n2 = cdiv(kc, c);
        let acc0 = fq.const_f(Elem::F32, 0.0, &[nr, h.kvh, h.g, n2 * c]);
        let out = fq.for_loop(n2, &[acc0], |fq, ci, st| {
            let (s, ok) = chunk(fq, ci)?;
            let sd = fq.dims(s).to_vec();
            let mb = fq.broadcast(m, &sd, &[0, 1, 2, 3])?;
            let e = fq.sub(s, mb)?;
            let e = fq.exp(e);
            let iwb = fq.broadcast(iw, &sd, &[0, 1, 2, 3])?;
            let p = fq.mul(e, iwb)?;
            let zp = fq.like_f(p, 0.0);
            let p = fq.select(ok, p, zp)?;
            let mean = if ob == 1 {
                fq.reshape(p, &[nr, h.kvh, h.g, c])?
            } else {
                fq.reduce(p, &[2], Fold::Sum)?
            };
            let base = with_i(fq, ci, c, Func::mul)?;
            let zero = fq.const_i(Elem::I32, 0, &[]);
            Ok(vec![fq.dynamic_update_slice(
                st[0],
                mean,
                &[zero, zero, zero, base],
            )?])
        })?;
        fq.slice_axis(out[0], 3, 0, kc)?
    };
    let mean = fq.reshape(mean, &[nr * h.qh(), kc])?;
    let mean = if kc < kmax {
        let z = fq.const_f(Elem::F32, 0.0, &[]);
        fq.pad(mean, z, &[0, 0], &[0, kmax - kc], &[0, 0])?
    } else {
        mean
    };
    if lane_offset == 0 && plane == 0 && plane_stride == qh && nr * h.qh() == i64::from(scores.rows)
    {
        return Ok(mean);
    }
    let idv = fq.const_ints(Elem::I32, &ids, &[ids.len() as i64])?;
    let new = fq.put_rows(slab, idv, mean, crate::hlo::Combine::Set)?;
    Ok(new)
}
