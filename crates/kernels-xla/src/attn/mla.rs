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
use crate::check::expect;
use crate::elemwise::norm::inv_rms_rows;
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::tensor::{KvPool, Landing, Operand, RaggedTensor, Stored};
use crate::tuning::Tuning;

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    super::paged::nonzero(op, what, v)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaPlan;

/// Nothing to plan: the attention reads the pool's CSR and the fire tables.
pub fn plan<T>(
    _kv_indptr: T,
    _kv_indices: T,
    _last_page_len: T,
    _kv_len: T,
) -> Result<MlaPlan, Error> {
    Ok(MlaPlan)
}

/// `kv_c = rmsnorm(kv_a[:, ..rank]) · weight`, `k_pe = kv_a[:, rank ..
/// rank + rope]` (`rope` = `k_pe`'s width): `(kv_c, k_pe)`, no `k_pe` when
/// it is empty.
pub fn latents(
    f: &mut Func,
    kv_a: Operand,
    weight: Operand,
    eps: f32,
    kv_lora_rank: u32,
    kv_c: Landing,
    k_pe: Landing,
) -> Result<(Val, Option<Val>), Error> {
    const OP: &str = "attn.mla_latents";
    split_checks(OP, kv_a, weight, kv_lora_rank, kv_c, k_pe)?;
    split_kv_a(f, OP, kv_a, weight, eps, kv_lora_rank, k_pe)
}

/// [`latents`], then the neox rotation of `k_pe` (one `rope_dim`-wide head,
/// `θ_i = pos · theta^(−2i / rope_dim)`, halves rotated), as kernels-slang's
/// `rope::partial_q(k_pe, positions, rope_dim, rope_dim, theta)`.
pub fn latents_rope(
    f: &mut Func,
    kv_a: Operand,
    weight: Operand,
    positions: Operand,
    eps: f32,
    kv_lora_rank: u32,
    rope_dim: u32,
    theta: f32,
    kv_c: Landing,
    k_pe: Landing,
) -> Result<(Val, Option<Val>), Error> {
    const OP: &str = "attn.mla_latents_rope";
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
    let (c, pe) = split_kv_a(f, OP, kv_a, weight, eps, kv_lora_rank, k_pe)?;
    let Some(pe) = pe else {
        return Ok((c, None));
    };
    // k_pe lands in bf16 before it rotates, as the two GPU passes store it.
    let pe = f.convert(pe, Elem::Bf16);
    let pe = f.convert(pe, Elem::F32);
    let pos = read_rows_i32(f, OP, positions, kv_a.rows)?;
    Ok((c, Some(neox(f, pe, pos, i64::from(rope_dim), theta)?)))
}

fn split_checks(
    op: &'static str,
    kv_a: impl Stored,
    weight: impl Stored,
    rank: u32,
    kv_c: impl Stored,
    k_pe: impl Stored,
) -> Result<(), Error> {
    expect(op, kv_a, &[Dtype::Bf16])?;
    nonzero(op, "the latent rank", rank)?;
    if kv_c.width() != rank || kv_c.rows() != kv_a.rows() || k_pe.rows() != kv_a.rows() {
        return Err(refuse(
            op,
            "the latent is the stated rank wide and both outputs are one row per source row",
        ));
    }
    if kv_a.width() < rank + k_pe.width() {
        return Err(refuse(
            op,
            format!(
                "the {}-wide source row does not hold the {rank}-wide latent and the {}-wide rope tail",
                kv_a.width(),
                k_pe.width()
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

/// `(kv_c, k_pe)`: the normed latent, and the rope tail (f32) when it is not
/// empty.
fn split_kv_a(
    f: &mut Func,
    op: &'static str,
    kv_a: Operand,
    weight: Operand,
    eps: f32,
    rank: u32,
    k_pe: Landing,
) -> Result<(Val, Option<Val>), Error> {
    let (r, rk) = (i64::from(kv_a.rows), i64::from(rank));
    let a = kv_a.f32(f);
    let lat = f.slice_axis(a, 1, 0, rk)?;
    let n = inv_rms_rows(f, op, lat, rank, eps)?;
    let w = weight.f32(f);
    let wn = f.ty(w).elements();
    let w = f.reshape(w, &[wn])?;
    let w = f.slice(w, &[0], &[rk], &[1])?;
    let w = f.broadcast(w, &[r, rk], &[1])?;
    let c = f.mul(n, w)?;
    if k_pe.width == 0 {
        return Ok((c, None));
    }
    let pe = f.slice_axis(a, 1, rk, rk + i64::from(k_pe.width))?;
    Ok((c, Some(pe)))
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
/// and rope parts: `(q_nope, q_pe)`, either absent when it is empty.
pub fn split_q_b(
    f: &mut Func,
    q_b: Operand,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: Landing,
    q_pe: Landing,
) -> Result<(Option<Val>, Option<Val>), Error> {
    const OP: &str = "attn.mla_split_q_b";
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
    let v = f.reshape(q_b.val, &[r, h, n + p])?;
    let a = if n > 0 {
        Some(f.slice_axis(v, 2, 0, n)?)
    } else {
        None
    };
    let b = if p > 0 {
        Some(f.slice_axis(v, 2, n, n + p)?)
    } else {
        None
    };
    Ok((a, b))
}

/// `kv_b` as `[heads, nope + v_dim, rank]` bf16.
fn kv_b_heads(f: &mut Func, kv_b: Operand, h: i64, per: i64, rank: i64) -> Result<Val, Error> {
    let w = f.convert(kv_b.val, Elem::Bf16);
    Ok(f.reshape(w, &[h, per, rank])?)
}

fn kv_b_checks(
    op: &'static str,
    kv_b: impl Stored,
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
    f: &mut Func,
    q_nope: Operand,
    kv_b: Operand,
    heads: u32,
    kv_lora_rank: u32,
    nope_dim: u32,
    v_head_dim: u32,
    q_latent: Landing,
) -> Result<Val, Error> {
    const OP: &str = "attn.mla_absorb_q";
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
    let q = f.reshape(q_nope.val, &[t, h, n])?;
    let w = kv_b_heads(f, kv_b, h, n + v, rk)?;
    let wk = f.slice_axis(w, 1, 0, n)?;
    // [h, t, rank]
    let y = f.dot_general_at(q, wk, &[1], &[0], &[2], &[1], Elem::F32, false)?;
    Ok(f.transpose(y, &[1, 0, 2])?)
}

/// `o[t, h] = latent[t, h] · W_UV[h]ᵀ` with `W_UV[h] = kv_b[h, nope.., :]`
/// (`[v_dim, rank]`).
pub fn absorb_out(
    f: &mut Func,
    latent: Operand,
    kv_b: Operand,
    heads: u32,
    kv_lora_rank: u32,
    v_head_dim: u32,
    nope_dim: u32,
    o: Landing,
) -> Result<Val, Error> {
    const OP: &str = "attn.mla_absorb_out";
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
    let l = f.reshape(latent.val, &[t, h, rk])?;
    let w = kv_b_heads(f, kv_b, h, n + v, rk)?;
    let wv = f.slice_axis(w, 1, n, n + v)?;
    // [h, t, v]
    let y = f.dot_general_at(l, wv, &[1], &[0], &[2], &[2], Elem::F32, false)?;
    Ok(f.transpose(y, &[1, 0, 2])?)
}

/// Appends `kv_c` to the keys plane and `k_pe` (when there is one) to the
/// values plane at each row's write slot; out-of-range slots (padded rows)
/// are dropped. `pool` gives the paging; its page tables are not read. The
/// new `(keys, values)`.
pub fn kv_append(
    f: &mut Func,
    kv_c: Operand,
    keys: Operand,
    rope: Option<(Operand, Operand)>,
    pool: &KvPool<impl Stored>,
    write_page: Operand,
    write_offset: Operand,
) -> Result<(Val, Option<Val>), Error> {
    const OP: &str = "attn.mla_kv_append";
    expect(OP, kv_c, &[Dtype::Bf16])?;
    if pool.page_size <= 0 {
        return Err(refuse(OP, "the kv page size is zero"));
    }
    if keys.width != kv_c.width
        || rope.is_some_and(|(k_pe, values)| values.width != k_pe.width || k_pe.rows != kv_c.rows)
    {
        return Err(refuse(
            OP,
            "the latent and rope planes are as wide as the pool's keys and values planes",
        ));
    }
    let mut pairs = vec![(kv_c, keys)];
    if let Some(r) = rope {
        pairs.push(r);
    }
    let planes = scatter_slots(f, OP, &pairs, pool.page_size, write_page, write_offset)?;
    Ok((planes[0], planes.get(1).copied()))
}

/// One MLA attention call: a description over handles (`T`) and landings
/// (`L`), read by [`MlaCall::try_map`] into the operands [`flash`] takes.
#[derive(Clone, Copy, Debug)]
pub struct MlaCall<T, L = T> {
    pub op: &'static str,
    pub q: T,
    pub q_pe: T,
    pub selection: Option<T>,
    pub positions: T,
    pub request_of_token: T,
    pub heads: u32,
    pub kv_lora_rank: u32,
    pub sm_scale: f32,
    pub o: L,
    pub decode: bool,
}

impl<T: Stored> MlaCall<T> {
    /// The call and its pool with every plane [`flash`] consumes read by
    /// `read`, in the order it consumes them: the page tables, the row
    /// tables, the query, its rope part, the latent keys, the rope plane,
    /// the selection. With no rope (`q_pe` zero-wide) neither rope plane is
    /// read: they keep their shapes over the query's value, never read.
    pub fn try_map<O: Copy, E>(
        self,
        pool: &KvPool<T>,
        mut read: impl FnMut(T) -> Result<O, E>,
    ) -> Result<(MlaCall<O, Landing>, KvPool<O>), E> {
        let roped = self.heads != 0 && self.q_pe.width() / self.heads != 0;
        let page_indptr = read(pool.page_indptr)?;
        let page_indices = read(pool.page_indices)?;
        let positions = read(self.positions)?;
        let request_of_token = read(self.request_of_token)?;
        let q = read(self.q)?;
        let q_pe = if roped { read(self.q_pe)? } else { q };
        let keys = read(pool.keys)?;
        let values = if roped { read(pool.values)? } else { keys };
        let selection = match self.selection {
            Some(s) => Some(read(s)?),
            None => None,
        };
        Ok((
            MlaCall {
                op: self.op,
                q,
                q_pe,
                selection,
                positions,
                request_of_token,
                heads: self.heads,
                kv_lora_rank: self.kv_lora_rank,
                sm_scale: self.sm_scale,
                o: self.o.shape(),
                decode: self.decode,
            },
            KvPool {
                keys,
                values,
                page_indices,
                page_indptr,
                page_size: pool.page_size,
                max_pages: pool.max_pages,
                seq_stride: pool.seq_stride,
                head_stride: pool.head_stride,
            },
        ))
    }
}

/// The latent attention an [`MlaCall`] states, its operands read: `o`.
/// The query's rope part and the pool's values plane are read only when the
/// rope is not empty.
pub fn flash(
    f: &mut Func,
    call: MlaCall<Operand, Landing>,
    pool: KvPool<Operand>,
    tuning: Tuning,
) -> Result<Val, Error> {
    let MlaCall {
        op,
        q,
        q_pe,
        selection,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        decode,
    } = call;
    let pool = &pool;
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
    let geom = read_geometry(f, pool)?;
    let qpos = read_rows_i32(f, op, positions, rows)?;
    let req = read_rows_i32(f, op, request_of_token, rows)?;
    let qv = q.val;
    let qp = if kpe != 0 { Some(q_pe.val) } else { None };
    let keys = pool.keys.val;
    let values = if kpe != 0 { pool.values.val } else { keys };
    let sel = selection.map(|s| f.convert(s.val, Elem::I32));
    let (ch, per_row_max) = (tuning.chunks, tuning.attn_per_row_max);
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
    let per = sel.is_some() || decode || r <= per_row_max * geom.lanes;
    let fl = if per {
        let mut rows_v = Rows::new(q4, zero, hi);
        rows_v.lane = Some(lane);
        let keys = match sel {
            Some(t) => KeyList::Table(t),
            None => KeyList::Range,
        };
        per_row(f, &planes, &h, &Rule::default(), &rows_v, keys, scale, ch)?
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
            ch,
        )?
    };
    Ok(fl.o)
}

pub fn attention_decode<T: Stored>(
    q: T,
    q_pe: T,
    positions: T,
    request_of_token: T,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: T,
) -> MlaCall<T> {
    MlaCall {
        op: "attn.mla_decode",
        q,
        q_pe,
        selection: None,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        decode: true,
    }
}

pub fn attention_prefill<T: Stored>(
    q: RaggedTensor<T>,
    q_pe: T,
    positions: T,
    request_of_token: T,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: T,
) -> MlaCall<T> {
    MlaCall {
        op: "attn.mla_prefill",
        q: q.data,
        q_pe,
        selection: None,
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        decode: false,
    }
}

/// `selection` is `[rows, top_k]` i32 key positions; entries outside
/// `[0, q_pos]` are skipped, repeats count again.
pub fn attention_decode_selected<T: Stored>(
    q: T,
    q_pe: T,
    selection: T,
    positions: T,
    request_of_token: T,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: T,
) -> MlaCall<T> {
    MlaCall {
        op: "attn.mla_decode",
        q,
        q_pe,
        selection: Some(selection),
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        decode: true,
    }
}

pub fn attention_prefill_selected<T: Stored>(
    q: RaggedTensor<T>,
    q_pe: T,
    selection: T,
    positions: T,
    request_of_token: T,
    heads: u32,
    kv_lora_rank: u32,
    sm_scale: f32,
    o: T,
) -> MlaCall<T> {
    MlaCall {
        op: "attn.mla_prefill",
        q: q.data,
        q_pe,
        selection: Some(selection),
        positions,
        request_of_token,
        heads,
        kv_lora_rank,
        sm_scale,
        o,
        decode: false,
    }
}
