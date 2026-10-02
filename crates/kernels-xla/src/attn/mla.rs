//! Multi-head latent attention (DeepSeek MLA), absorbed form: the pool holds
//! the normed latent `c_kv` (keys plane, `[slots, kv_lora_rank]`) and the
//! rotated rope tail `k_pe` (values plane, `[slots, rope]`); a query reads
//! `score = (q_latent · c_kv + q_pe · k_pe) · sm_scale` and answers the
//! softmax-weighted latent.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use super::paged::{
    Heads, KeyList, Planes, Rows, Rule, blocked, per_row, pool_paging, read_geometry,
    read_rows_i32, scatter_slots, with_i,
};
use crate::cx::{Ctx, Cx, expect};
use crate::elemwise::norm::inv_rms_rows;
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    super::paged::nonzero(op, what, v)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaPlan;

/// Nothing to plan: the attention reads the pool's CSR and the fire tables.
pub fn plan(
    _ctx: &Ctx<'_>,
    _kv_indptr: Tensor,
    _kv_indices: Tensor,
    _last_page_len: Tensor,
    _kv_len: Tensor,
) -> Result<MlaPlan, Error> {
    Ok(MlaPlan)
}

/// `kv_c = rmsnorm(kv_a[:, ..rank]) · weight`, `k_pe = kv_a[:, rank ..
/// rank + rope]` (`rope` = `k_pe`'s width).
pub fn latents(
    ctx: &Ctx<'_>,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_lora_rank: u32,
    kv_c: Tensor,
    k_pe: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_latents";
    split_checks(OP, kv_a, weight, kv_lora_rank, kv_c, k_pe)?;
    ctx.emit(&mut |cx| {
        let pe = split_kv_a(cx, OP, kv_a, weight, eps, kv_lora_rank, kv_c, k_pe)?;
        if let Some(pe) = pe {
            cx.write(k_pe, pe)?;
        }
        Ok(())
    })
}

/// [`latents`], then the neox rotation of `k_pe` (one `rope_dim`-wide head,
/// `θ_i = pos · theta^(−2i / rope_dim)`, halves rotated), as kernels-wgpu's
/// `rope::partial_q(k_pe, positions, rope_dim, rope_dim, theta)`.
pub fn latents_rope(
    ctx: &Ctx<'_>,
    kv_a: Tensor,
    positions: Tensor,
    weight: Tensor,
    eps: f32,
    kv_lora_rank: u32,
    rope_dim: u32,
    theta: f32,
    kv_c: Tensor,
    k_pe: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_latents_rope";
    split_checks(OP, kv_a, weight, kv_lora_rank, kv_c, k_pe)?;
    nonzero(OP, "the rope width", rope_dim)?;
    if k_pe.width != rope_dim || !rope_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the rope tail is {} wide; the rotation turns an even {rope_dim}",
                k_pe.width
            ),
        ));
    }
    if positions.dtype != Dtype::I32 || positions.elements() < u64::from(kv_a.rows) {
        return Err(refuse(OP, "the position stream is one i32 per row"));
    }
    ctx.emit(&mut |cx| {
        let Some(pe) = split_kv_a(cx, OP, kv_a, weight, eps, kv_lora_rank, kv_c, k_pe)? else {
            return Ok(());
        };
        // k_pe lands in bf16 before it rotates, as the two GPU passes store it.
        let pe = cx.convert(pe, Elem::Bf16);
        let pe = cx.convert(pe, Elem::F32);
        let pos = read_rows_i32(cx, OP, positions, kv_a.rows)?;
        let out = neox(cx, pe, pos, i64::from(rope_dim), theta)?;
        cx.write(k_pe, out)
    })
}

fn split_checks(
    op: &'static str,
    kv_a: Tensor,
    weight: Tensor,
    rank: u32,
    kv_c: Tensor,
    k_pe: Tensor,
) -> Result<(), Error> {
    expect(op, kv_a, &[Dtype::Bf16])?;
    nonzero(op, "the latent rank", rank)?;
    if kv_c.width != rank || kv_c.rows != kv_a.rows || k_pe.rows != kv_a.rows {
        return Err(refuse(
            op,
            "the latent is the stated rank wide and both outputs are one row per source row",
        ));
    }
    if kv_a.width < rank + k_pe.width {
        return Err(refuse(
            op,
            format!(
                "the {}-wide source row does not hold the {rank}-wide latent and the {}-wide rope tail",
                kv_a.width, k_pe.width
            ),
        ));
    }
    if weight.elements() < u64::from(rank) {
        return Err(refuse(
            op,
            "the norm weight is not one value per latent column",
        ));
    }
    Ok(())
}

/// Writes `kv_c`; answers the rope tail (f32) when it is not empty.
fn split_kv_a(
    cx: &mut Cx<'_>,
    op: &'static str,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    rank: u32,
    kv_c: Tensor,
    k_pe: Tensor,
) -> Result<Option<Val>, Error> {
    let (r, rk) = (i64::from(kv_a.rows), i64::from(rank));
    let a = cx.read_f32(kv_a)?;
    let lat = cx.slice_axis(a, 1, 0, rk)?;
    let n = inv_rms_rows(cx, op, lat, rank, eps)?;
    let w = cx.read_f32(weight)?;
    let wn = cx.ty(w).elements();
    let w = cx.reshape(w, &[wn])?;
    let w = cx.slice(w, &[0], &[rk], &[1])?;
    let w = cx.broadcast(w, &[r, rk], &[1])?;
    let c = cx.mul(n, w)?;
    cx.write(kv_c, c)?;
    if k_pe.width == 0 {
        return Ok(None);
    }
    let pe = cx.slice_axis(a, 1, rk, rk + i64::from(k_pe.width))?;
    Ok(Some(pe))
}

/// Neox rotation of one `dim`-wide head per row of an f32 `[r, dim]`.
fn neox(f: &mut Func, x: Val, pos: Val, dim: i64, theta: f32) -> Result<Val, Error> {
    let r = f.dims(x)[0];
    let half = dim / 2;
    let i = f.iota(Elem::F32, &[half], 0);
    // inv_freq_i = 2^(−(2i / dim) · log2 θ)
    let e = f.scale(
        i,
        -2.0 / dim as f64 * f64::from(theta.log2()) * std::f64::consts::LN_2,
    )?;
    let inv = f.exp(e);
    let p = f.convert(pos, Elem::F32);
    let pb = f.broadcast(p, &[r, half], &[0])?;
    let ib = f.broadcast(inv, &[r, half], &[1])?;
    let ang = f.mul(pb, ib)?;
    let (c, s) = (f.cos(ang), f.sin(ang));
    let x1 = f.slice_axis(x, 1, 0, half)?;
    let x2 = f.slice_axis(x, 1, half, dim)?;
    let a = f.mul(x1, c)?;
    let b = f.mul(x2, s)?;
    let y1 = f.sub(a, b)?;
    let a = f.mul(x1, s)?;
    let b = f.mul(x2, c)?;
    let y2 = f.add(a, b)?;
    Ok(f.concat(&[y1, y2], 1)?)
}

/// Cuts each head of `q_b` (`[rows, heads · (nope + rope)]`) into its nope
/// and rope parts.
pub fn split_q_b(
    ctx: &Ctx<'_>,
    q_b: Tensor,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: Tensor,
    q_pe: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_split_q_b";
    expect(OP, q_b, &[Dtype::Bf16])?;
    let heads = nonzero(OP, "heads", heads)?;
    let per = nope_dim + rope_dim;
    if q_b.width != heads * per
        || q_nope.width != heads * nope_dim
        || q_pe.width != heads * rope_dim
    {
        return Err(refuse(
            OP,
            "the nope and rope planes are the per-head cut of q_b's row",
        ));
    }
    let (r, h) = (i64::from(q_b.rows), i64::from(heads));
    let (n, p) = (i64::from(nope_dim), i64::from(rope_dim));
    ctx.emit(&mut |cx| {
        let v = cx.read(q_b)?;
        let v = cx.reshape(v, &[r, h, n + p])?;
        if n > 0 {
            let a = cx.slice_axis(v, 2, 0, n)?;
            cx.write(q_nope, a)?;
        }
        if p > 0 {
            let b = cx.slice_axis(v, 2, n, n + p)?;
            cx.write(q_pe, b)?;
        }
        Ok(())
    })
}

/// `kv_b` as `[heads, nope + v_dim, rank]` bf16.
fn kv_b_heads(cx: &mut Cx<'_>, kv_b: Tensor, h: i64, per: i64, rank: i64) -> Result<Val, Error> {
    let w = cx.read(kv_b)?;
    let w = cx.convert(w, Elem::Bf16);
    Ok(cx.reshape(w, &[h, per, rank])?)
}

fn kv_b_checks(
    op: &'static str,
    kv_b: Tensor,
    heads: u32,
    rank: u32,
    nope: u32,
    v_dim: u32,
) -> Result<(), Error> {
    expect(op, kv_b, &[Dtype::Bf16])?;
    let want = u64::from(heads) * u64::from(nope + v_dim) * u64::from(rank);
    if kv_b.elements() != want {
        return Err(refuse(
            op,
            format!(
                "kv_b holds {} values; {heads} heads of ({nope} + {v_dim}) x {rank} are {want}",
                kv_b.elements()
            ),
        ));
    }
    Ok(())
}

/// `q_latent[t, h] = q_nope[t, h] · W_UK[h]` with `W_UK[h] = kv_b[h, ..nope,
/// :]` (`[nope, rank]`).
pub fn absorb_q(
    ctx: &Ctx<'_>,
    q_nope: Tensor,
    kv_b: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
    v_head_dim: u32,
    q_latent: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_absorb_q";
    expect(OP, q_nope, &[Dtype::Bf16])?;
    nonzero(OP, "heads", heads)?;
    nonzero(OP, "the latent rank", kv_lora_rank)?;
    nonzero(OP, "the nope width", nope_dim)?;
    kv_b_checks(OP, kv_b, heads, kv_lora_rank, nope_dim, v_head_dim)?;
    if q_nope.width != heads * nope_dim
        || q_latent.width != heads * kv_lora_rank
        || q_latent.rows != q_nope.rows
    {
        return Err(refuse(
            OP,
            "the absorbed q is heads x rank wide, one row per token",
        ));
    }
    let (t, h, rk) = (
        i64::from(q_nope.rows),
        i64::from(heads),
        i64::from(kv_lora_rank),
    );
    let (n, v) = (i64::from(nope_dim), i64::from(v_head_dim));
    ctx.emit(&mut |cx| {
        let q = cx.read(q_nope)?;
        let q = cx.reshape(q, &[t, h, n])?;
        let w = kv_b_heads(cx, kv_b, h, n + v, rk)?;
        let wk = cx.slice_axis(w, 1, 0, n)?;
        // [h, t, rank]
        let y = cx.dot_general_at(q, wk, &[1], &[0], &[2], &[1], Elem::F32, false)?;
        let y = cx.transpose(y, &[1, 0, 2])?;
        cx.write(q_latent, y)
    })
}

/// `o[t, h] = latent[t, h] · W_UV[h]ᵀ` with `W_UV[h] = kv_b[h, nope.., :]`
/// (`[v_dim, rank]`).
pub fn absorb_out(
    ctx: &Ctx<'_>,
    latent: Tensor,
    kv_b: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    nope_dim: u32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_absorb_out";
    expect(OP, latent, &[Dtype::Bf16])?;
    nonzero(OP, "heads", heads)?;
    nonzero(OP, "the latent rank", kv_lora_rank)?;
    nonzero(OP, "the value head dim", v_head_dim)?;
    kv_b_checks(OP, kv_b, heads, kv_lora_rank, nope_dim, v_head_dim)?;
    if latent.width != heads * kv_lora_rank
        || o.width != heads * v_head_dim
        || o.rows != latent.rows
    {
        return Err(refuse(
            OP,
            "the latent is heads x rank wide and the answer heads x v_dim, one row per token",
        ));
    }
    let (t, h, rk) = (
        i64::from(latent.rows),
        i64::from(heads),
        i64::from(kv_lora_rank),
    );
    let (n, v) = (i64::from(nope_dim), i64::from(v_head_dim));
    ctx.emit(&mut |cx| {
        let l = cx.read(latent)?;
        let l = cx.reshape(l, &[t, h, rk])?;
        let w = kv_b_heads(cx, kv_b, h, n + v, rk)?;
        let wv = cx.slice_axis(w, 1, n, n + v)?;
        // [h, t, v]
        let y = cx.dot_general_at(l, wv, &[1], &[0], &[2], &[2], Elem::F32, false)?;
        let y = cx.transpose(y, &[1, 0, 2])?;
        cx.write(o, y)
    })
}

/// Appends `kv_c` to the keys plane and `k_pe` to the values plane at each
/// row's write slot; out-of-range slots (padded rows) are dropped.
pub fn kv_append(
    ctx: &Ctx<'_>,
    kv_c: Tensor,
    k_pe: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.mla_kv_append";
    expect(OP, kv_c, &[Dtype::Bf16])?;
    if pool.page_size <= 0 {
        return Err(refuse(OP, "the kv page size is zero"));
    }
    if pool.keys.width != kv_c.width
        || (k_pe.width != 0 && (pool.values.width != k_pe.width || k_pe.rows != kv_c.rows))
    {
        return Err(refuse(
            OP,
            "the latent and rope planes are as wide as the pool's keys and values planes",
        ));
    }
    ctx.emit(&mut |cx| {
        let mut pairs = vec![(kv_c, pool.keys)];
        if k_pe.width != 0 {
            pairs.push((k_pe, pool.values));
        }
        scatter_slots(cx, OP, &pairs, pool.page_size, write_page, write_offset)
    })
}

fn flash(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    q_pe: Tensor,
    selection: Option<Tensor>,
    pool: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: Tensor,
    decode: bool,
) -> Result<(), Error> {
    expect(op, q, &[Dtype::Bf16])?;
    expect(op, pool.keys, &[Dtype::Bf16])?;
    pool_paging(op, pool)?;
    let heads = nonzero(op, "heads", heads)?;
    let rank = nonzero(op, "the latent rank", kv_lora_rank)?;
    let rows = nonzero(op, "rows", q.rows)?;
    if q.width != heads * rank || pool.keys.width != rank {
        return Err(refuse(
            op,
            "the absorbed q is heads x rank wide over a rank-wide latent pool",
        ));
    }
    if !q_pe.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide rotated q plane does not divide by the {heads} heads",
                q_pe.width
            ),
        ));
    }
    let kpe = q_pe.width / heads;
    if kpe != 0 {
        expect(op, q_pe, &[Dtype::Bf16])?;
        expect(op, pool.values, &[Dtype::Bf16])?;
        if pool.values.width != kpe || q_pe.rows != rows {
            return Err(refuse(
                op,
                "the rope plane is heads x rope wide over a rope-wide pool plane",
            ));
        }
    }
    if o.rows != rows || o.width != heads * rank {
        return Err(refuse(
            op,
            "the latent reading is heads x rank wide, one row per query row",
        ));
    }
    if positions.dtype != Dtype::I32 || request_of_token.dtype != Dtype::I32 {
        return Err(refuse(op, "the position and owning-request tables are i32"));
    }
    if let Some(sel) = selection {
        if sel.dtype != Dtype::I32 || sel.rows != rows {
            return Err(refuse(
                op,
                "the selection is one i32 key-index row per query row",
            ));
        }
        nonzero(op, "the selection budget", sel.width)?;
    }
    let h = Heads {
        kvh: 1,
        g: i64::from(heads),
        d: i64::from(rank + kpe),
        dv: i64::from(rank),
    };
    let r = i64::from(rows);
    ctx.emit(&mut |cx| {
        let geom = read_geometry(cx, pool)?;
        let qpos = read_rows_i32(cx, op, positions, rows)?;
        let req = read_rows_i32(cx, op, request_of_token, rows)?;
        let qv = cx.read(q)?;
        let qp = if kpe != 0 { Some(cx.read(q_pe)?) } else { None };
        let keys = cx.read(pool.keys)?;
        let values = if kpe != 0 {
            cx.read(pool.values)?
        } else {
            keys
        };
        let sel = match selection {
            Some(s) => {
                let v = cx.read(s)?;
                Some(cx.convert(v, Elem::I32))
            }
            None => None,
        };
        let f = cx.func();
        let q3 = f.reshape(qv, &[r, h.g, i64::from(rank)])?;
        let q3 = match qp {
            Some(p) => {
                let p = f.reshape(p, &[r, h.g, i64::from(kpe)])?;
                f.concat(&[q3, p], 2)?
            }
            None => q3,
        };
        let q4 = f.reshape(q3, &[r, 1, h.g, h.d])?;
        let lane = geom.lane_of(f, req)?;
        let cap = geom.capacity(f, lane)?;
        let upto = with_i(f, qpos, 1, Func::add)?;
        let hi = f.min(upto, cap)?;
        let zero = f.like_i(hi, 0);
        let hi = f.max(hi, zero)?;
        let planes = Planes::Paged {
            keys,
            values,
            geom,
            latent: true,
        };
        let scale = f64::from(sm_scale);
        let per = sel.is_some() || decode || r <= 2 * geom.lanes;
        let fl = if per {
            let mut rows_v = Rows::new(q4, zero, hi);
            rows_v.lane = Some(lane);
            let keys = match sel {
                Some(t) => KeyList::Table(t),
                None => KeyList::Range,
            };
            per_row(f, &planes, &h, &Rule::default(), &rows_v, keys, scale)?
        } else {
            let base = with_i(f, lane, geom.span, Func::mul)?;
            let hi = f.add(base, hi)?;
            let rows_v = Rows::new(q4, base, hi);
            let rule = Rule {
                kp_span: Some(geom.span),
                ..Rule::default()
            };
            blocked(
                f,
                &planes,
                &h,
                &rule,
                &rows_v,
                geom.lanes * geom.span,
                scale,
            )?
        };
        cx.write(o, fl.o)
    })
}

pub fn attention_decode(
    ctx: &Ctx<'_>,
    q: Tensor,
    q_pe: Tensor,
    pool: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    flash(
        ctx,
        "attention.mla_decode",
        q,
        q_pe,
        None,
        pool,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        true,
    )
}

pub fn attention_prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    q_pe: Tensor,
    pool: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    flash(
        ctx,
        "attention.mla_prefill",
        q.data,
        q_pe,
        None,
        pool,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        false,
    )
}

/// `selection` is `[rows, top_k]` i32 key positions; entries outside
/// `[0, q_pos]` are skipped, repeats count again.
pub fn attention_decode_selected(
    ctx: &Ctx<'_>,
    q: Tensor,
    q_pe: Tensor,
    selection: Tensor,
    pool: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    flash(
        ctx,
        "attention.mla_decode_selected",
        q,
        q_pe,
        Some(selection),
        pool,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        true,
    )
}

pub fn attention_prefill_selected(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    q_pe: Tensor,
    selection: Tensor,
    pool: &KvPool,
    positions: Tensor,
    request_of_token: Tensor,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    flash(
        ctx,
        "attention.mla_prefill_selected",
        q.data,
        q_pe,
        Some(selection),
        pool,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        false,
    )
}
