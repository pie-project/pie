//! Attention-score capture: the mean softmax row of each request's last
//! observed query rows, per head, into an f32 slab (eviction policies and
//! interpretability read it).

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::PrefillPlan;
use super::paged::{
    Heads, Planes, cdiv, clamp_i, fetch, flat_i32, geometry, head, kv_heads_agree, nonzero,
    pool_paging, round_up, row_heads, take, with_i,
};
use crate::cx::Ctx;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const BUDGET: i64 = 1 << 24;

/// For request `r < requests` (lane `r` of the pool and segment `r` of
/// `q.indptr`) and query head `h`, slab row
/// `(lane_offset + r) · plane_stride + plane + h` becomes the mean, over the
/// request's last `min(observe, q_len)` rows, of the causal softmax over its
/// keys (keys past `kv_max` are in the denominator but not written; the rest
/// of the row is zero).
pub fn capture(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
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
    scores: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.score_capture";
    if window.is_some() {
        return Err(refuse(
            OP,
            "a sliding window is stated, and a windowed row is not the softmax over the request's keys",
        ));
    }
    if q.data.dtype != Dtype::Bf16 || pool.keys.dtype != Dtype::Bf16 {
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
    if plan.positions.dtype != Dtype::I32 || q.indptr.dtype != Dtype::I32 {
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
        || pool.page_indptr.elements() < u64::from(requests) + 1
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
    ctx.emit(&mut |cx| {
        let indptr = cx.read(pool.page_indptr)?;
        let indptr = flat_i32(cx, indptr)?;
        let indptr = head(cx, indptr, nr + 1)?;
        let indices = cx.read(pool.page_indices)?;
        let indices = flat_i32(cx, indices)?;
        let qo = cx.read(q.indptr)?;
        let qo = flat_i32(cx, qo)?;
        let pos = cx.read(plan.positions)?;
        let pos = flat_i32(cx, pos)?;
        let qv = cx.read(q.data)?;
        let keys = cx.read(pool.keys)?;
        let slab = cx.read(scores)?;
        let f = cx.func();
        let geom = geometry(
            f,
            indptr,
            indices,
            i64::from(pool.page_size),
            i64::from(pool.max_pages),
        )?;
        let span = geom.span;
        let qo_lo = f.slice(qo, &[0], &[nr], &[1])?;
        let qo_hi = f.slice(qo, &[1], &[nr + 1], &[1])?;
        let qlen = f.sub(qo_hi, qo_lo)?;
        let nrows = clamp_i(f, qlen, 0, ob)?;
        let d2 = [nr, ob];
        let w = f.iota(Elem::I32, &d2, 1);
        let hib = f.broadcast(qo_hi, &d2, &[0])?;
        let nb = f.broadcast(nrows, &d2, &[0])?;
        let at = f.sub(hib, nb)?;
        let at = f.add(at, w)?;
        let live = f.compare(Cmp::Lt, w, nb)?;
        let at = clamp_i(f, at, 0, rows - 1)?;
        let qpos = take(f, pos, at)?;
        let lanes = f.iota(Elem::I32, &[nr], 0);
        let cap = geom.capacity(f, lanes)?;
        let capb = f.broadcast(cap, &d2, &[0])?;
        let upto = with_i(f, qpos, 1, Func::add)?;
        let lim = f.min(upto, capb)?;
        let zero = f.like_i(lim, 0);
        let lim = f.select(live, lim, zero)?;
        let lim = f.max(lim, zero)?;
        // Observed q rows, [nr, ob, kvh, g, d].
        let af = f.reshape(at, &[nr * ob])?;
        let qr = f.take_rows(qv, af)?;
        let q5 = f.reshape(qr, &[nr, ob, h.kvh, h.g, h.d])?;
        let planes = Planes::Paged {
            keys,
            values: keys,
            geom,
            latent: false,
        };
        let c = {
            let width = (h.kvh * h.d).max(ob * h.qh());
            let c = (BUDGET / (nr * width)).max(8);
            let c = if c <= 1 {
                1
            } else {
                1i64 << (63 - c.leading_zeros())
            };
            c.min(round_up(span, 8))
        };
        // Scores of chunk `ci`: [nr, kvh, ob, g, c] and admission.
        let chunk = |f: &mut Func,
                     ci: crate::hlo::Val|
         -> crate::hlo::Built<(crate::hlo::Val, crate::hlo::Val)> {
            let base = with_i(f, ci, c, Func::mul)?;
            let io = f.iota(Elem::I32, &[nr, c], 1);
            let bb = f.splat(base, &[nr, c])?;
            let kp = f.add(io, bb)?;
            let kc = clamp_i(f, kp, 0, span - 1)?;
            let lb = f.iota(Elem::I32, &[nr, c], 0);
            let x = with_i(f, lb, span, Func::mul)?;
            let x = f.add(x, kc)?;
            let xf = f.reshape(x, &[nr * c])?;
            let (k, _) = fetch(f, &planes, &h, xf)?;
            let k = f.reshape(k, &[nr, c, h.kvh, h.d])?;
            let s = f.dot_general_at(q5, k, &[0, 2], &[0, 2], &[4], &[3], Elem::F32, false)?;
            let s = f.scale(s, scale)?;
            let sd = [nr, h.kvh, ob, h.g, c];
            let kpb = f.broadcast(kp, &[nr, ob, c], &[0, 2])?;
            let lb = f.broadcast(lim, &[nr, ob, c], &[0, 1])?;
            let ok = f.compare(Cmp::Lt, kpb, lb)?;
            let ok = f.broadcast(ok, &sd, &[0, 2, 4])?;
            Ok((s, ok))
        };
        // Pass 1: the softmax max and denominator over every admitted key.
        let stat = [nr, h.kvh, ob, h.g];
        let m0 = f.const_f(Elem::F32, f64::NEG_INFINITY, &stat);
        let l0 = f.const_f(Elem::F32, 0.0, &stat);
        let n1 = cdiv(span, c);
        let st = f.for_loop(n1, &[m0, l0], |f, ci, st| {
            let (s, ok) = chunk(f, ci)?;
            let ninf = f.like_f(s, f64::NEG_INFINITY);
            let sm = f.select(ok, s, ninf)?;
            let cm = f.reduce(sm, &[4], Fold::Max)?;
            let m = f.max(st[0], cm)?;
            let floor = f.like_f(m, f64::NEG_INFINITY);
            let fin = f.compare(Cmp::Gt, m, floor)?;
            let z = f.like_f(m, 0.0);
            let ms = f.select(fin, m, z)?;
            let sd = f.dims(s).to_vec();
            let mb = f.broadcast(ms, &sd, &[0, 1, 2, 3])?;
            let e = f.sub(s, mb)?;
            let e = f.exp(e);
            let ze = f.like_f(e, 0.0);
            let p = f.select(ok, e, ze)?;
            let ps = f.reduce(p, &[4], Fold::Sum)?;
            let dm = f.sub(st[0], ms)?;
            let corr = f.exp(dm);
            let l = f.mul(st[1], corr)?;
            let l = f.add(l, ps)?;
            Ok(vec![m, l])
        })?;
        let (m, l) = (st[0], st[1]);
        let floor = f.like_f(m, f64::NEG_INFINITY);
        let fin = f.compare(Cmp::Gt, m, floor)?;
        let z = f.like_f(m, 0.0);
        let m = f.select(fin, m, z)?;
        let zl = f.like_f(l, 0.0);
        let pos_l = f.compare(Cmp::Gt, l, zl)?;
        let one = f.like_f(l, 1.0);
        let inv = f.div(one, l)?;
        let inv = f.select(pos_l, inv, zl)?;
        // Row weights: 1 / rows over the observed rows.
        let nf = f.convert(nrows, Elem::F32);
        let one_n = f.like_f(nf, 1.0);
        let nf = f.max(nf, one_n)?;
        let wr = f.div(one_n, nf)?;
        let wr = f.broadcast(wr, &d2, &[0])?;
        let zw = f.like_f(wr, 0.0);
        let wr = f.select(live, wr, zw)?;
        // Pass 2: the mean probability of the first `kv_max` keys.
        let kc = kmax.min(span);
        let n2 = cdiv(kc, c);
        let acc0 = f.const_f(Elem::F32, 0.0, &[nr, h.kvh, h.g, n2 * c]);
        let out = f.for_loop(n2, &[acc0], |f, ci, st| {
            let (s, ok) = chunk(f, ci)?;
            let sd = f.dims(s).to_vec();
            let mb = f.broadcast(m, &sd, &[0, 1, 2, 3])?;
            let e = f.sub(s, mb)?;
            let e = f.exp(e);
            let ib = f.broadcast(inv, &sd, &[0, 1, 2, 3])?;
            let wb = f.broadcast(wr, &sd, &[0, 2])?;
            let p = f.mul(e, ib)?;
            let p = f.mul(p, wb)?;
            let zp = f.like_f(p, 0.0);
            let p = f.select(ok, p, zp)?;
            let mean = f.reduce(p, &[2], Fold::Sum)?;
            let base = with_i(f, ci, c, Func::mul)?;
            let zero = f.const_i(Elem::I32, 0, &[]);
            Ok(vec![f.dynamic_update_slice(
                st[0],
                mean,
                &[zero, zero, zero, base],
            )?])
        })?;
        let mean = f.slice_axis(out[0], 3, 0, kc)?;
        let mean = f.reshape(mean, &[nr * h.qh(), kc])?;
        let mean = if kc < kmax {
            let z = f.const_f(Elem::F32, 0.0, &[]);
            f.pad(mean, z, &[0, 0], &[0, kmax - kc], &[0, 0])?
        } else {
            mean
        };
        let idv = f.const_ints(Elem::I32, &ids, &[ids.len() as i64])?;
        let new = f.put_rows(slab, idv, mean, crate::hlo::Combine::Set)?;
        cx.write(scores, new)
    })
}
