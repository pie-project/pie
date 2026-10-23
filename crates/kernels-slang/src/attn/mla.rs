//! Multi-head latent attention (`attn.mla_*`) from `kernels/attn/mla.slang`:
//! the latent prologue (latents, the q split, the absorbed projections, the
//! paged append) and the paged attention over latents, causal or over a
//! selection.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const MLA: &str = "attn/mla.slang";

const PREP_THREADS: u32 = 256;

const LANES: u32 = 32;

const MAX_CKV: u32 = 16 * LANES;
const MAX_KPE: u32 = 4 * LANES;

const ABSORB_GROUP: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaPlan;

pub fn plan(
    _ctx: &Ctx<'_>,
    _kv_indptr: Tensor,
    _kv_indices: Tensor,
    _last_page_len: Tensor,
    _kv_len: Tensor,
) -> Result<MlaPlan, Error> {
    Ok(MlaPlan)
}

pub fn latents(
    ctx: &Ctx<'_>,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_lora_rank: u32,
    kv_c: Tensor,
    k_pe: Tensor,
) -> Result<(), Error> {
    split_kv_a_norm(
        ctx,
        "attn.mla_latents",
        kv_a,
        weight,
        eps,
        kv_lora_rank,
        kv_c,
        k_pe,
    )
}

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
    split_kv_a_norm(
        ctx,
        "attn.mla_latents_rope",
        kv_a,
        weight,
        eps,
        kv_lora_rank,
        kv_c,
        k_pe,
    )?;
    crate::elemwise::rope::partial_q(
        ctx,
        k_pe,
        positions,
        rope_dim,
        rope_dim,
        theta,
        rope_dim / 2,
    )
}

fn split_kv_a_norm(
    ctx: &Ctx<'_>,
    op: &'static str,
    kv_a: Tensor,
    weight: Tensor,
    eps: f32,
    kv_lora_rank: u32,
    kv_c: Tensor,
    k_pe: Tensor,
) -> Result<(), Error> {
    let entry = dtype_dispatch!(op, kv_a.dtype, { Bf16 => "attn_mla_latents_bf16" });
    debug_assert!(
        kv_c.width == kv_lora_rank && kv_c.rows == kv_a.rows,
        "the latent output is the stated rank wide, one row per source row"
    );
    debug_assert!(
        k_pe.rows == kv_a.rows,
        "the rope tail is one row per source row"
    );
    let kv_lora = stated(op, nonzero(op, "the latent rank", kv_lora_rank)?)?;
    let rope = stated(op, k_pe.width)?;
    let src_row_stride = stated(op, kv_a.width)?;
    if kv_a.width < kv_lora_rank + k_pe.width {
        return Err(refuse(
            op,
            format!(
                "the {}-wide source row does not hold the {kv_lora_rank}-wide latent beside \
                 the {}-wide rope tail",
                kv_a.width, k_pe.width
            ),
        ));
    }
    let rows = nonzero(op, "rows", kv_a.rows)?;
    super::even_lanes(op, "latent rank", kv_lora_rank)?;
    super::even_lanes(op, "source row", kv_a.width)?;
    if k_pe.width != 0 {
        super::even_lanes(op, "rope tail", k_pe.width)?;
    }
    ctx.fire(
        Fire::at(MLA, entry).apply(Grid::of([PREP_THREADS * rows, 1, 1], [PREP_THREADS, 1, 1])),
        &[
            kv_a.arg(),
            weight.arg(),
            kv_c.arg_mut(),
            k_pe.arg_mut(),
            kv_lora.arg(),
            rope.arg(),
            src_row_stride.arg(),
            eps.arg(),
        ],
    )
}

pub fn split_q_b(
    ctx: &Ctx<'_>,
    q_b: Tensor,
    heads: u32,
    nope_dim: u32,
    rope_dim: u32,
    q_nope: Tensor,
    q_pe: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_split_q_b";
    let entry = dtype_dispatch!(OP, q_b.dtype, { Bf16 => "attn_mla_split_q_b_bf16" });
    let heads = stated(OP, nonzero(OP, "heads", heads)?)?;
    let nope = stated(OP, nope_dim)?;
    let rope = stated(OP, rope_dim)?;
    let per = i64::from(nope) + i64::from(rope);
    let total = i64::from(q_b.rows) * i64::from(heads) * per;
    let total = i32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} split elements do not fit the shader's int"),
        )
    })?;
    let lanes = nonzero(OP, "split elements", u32::try_from(total).unwrap_or(0))?;
    if !nope_dim.is_multiple_of(2) || !rope_dim.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!("a {nope_dim} + {rope_dim} head splits in bf16 pairs, so both widths are even"),
        ));
    }
    let lanes = lanes.div_ceil(2);
    debug_assert!(
        q_nope.width == u32::try_from(i64::from(heads) * i64::from(nope)).unwrap_or(u32::MAX)
            && q_pe.width == u32::try_from(i64::from(heads) * i64::from(rope)).unwrap_or(u32::MAX),
        "the nope and rope planes are the per-head cut of q_b's row"
    );
    ctx.fire(
        Fire::at(MLA, entry).apply(Grid::of([lanes, 1, 1], [256, 1, 1])),
        &[
            q_b.arg(),
            q_nope.arg_mut(),
            q_pe.arg_mut(),
            total.arg(),
            heads.arg(),
            nope.arg(),
            rope.arg(),
        ],
    )
}

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
    const OP: &str = "attn.mla_absorb_q";
    let entry = dtype_dispatch!(OP, q_nope.dtype, { Bf16 => "attn_mla_absorb_q_bf16" });
    let heads_i = stated(OP, nonzero(OP, "heads", heads)?)?;
    let rank = stated(OP, nonzero(OP, "the latent rank", kv_lora_rank)?)?;
    let nope = stated(OP, nope_dim)?;
    let v_dim = stated(OP, v_head_dim)?;
    let rows = nonzero(OP, "rows", q_nope.rows)?;
    let pairs = super::even_lanes(OP, "latent rank", kv_lora_rank)?;
    super::even_lanes(OP, "nope width", nope_dim)?;
    debug_assert!(
        q_latent.width == heads * kv_lora_rank && q_latent.rows == q_nope.rows,
        "the absorbed q is `heads · rank` wide, one row per token"
    );
    ctx.fire(
        Fire::at(MLA, entry).apply(Grid::of([pairs, heads, rows], [ABSORB_GROUP, 1, 1])),
        &[
            q_nope.arg(),
            kv_b.arg(),
            q_latent.arg_mut(),
            heads_i.arg(),
            rank.arg(),
            nope.arg(),
            v_dim.arg(),
        ],
    )
}

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
    const OP: &str = "attn.mla_absorb_out";
    let entry = dtype_dispatch!(OP, latent.dtype, { Bf16 => "attn_mla_absorb_out_bf16" });
    let heads_i = stated(OP, nonzero(OP, "heads", heads)?)?;
    let rank = stated(OP, nonzero(OP, "the latent rank", kv_lora_rank)?)?;
    let v_dim = stated(OP, nonzero(OP, "the value head dim", v_head_dim)?)?;
    let nope = stated(OP, nope_dim)?;
    let rows = nonzero(OP, "rows", latent.rows)?;
    let pairs = super::even_lanes(OP, "value head width", v_head_dim)?;
    super::even_lanes(OP, "latent rank", kv_lora_rank)?;
    debug_assert!(
        latent.width == heads * kv_lora_rank,
        "the latent reading is `heads · rank` wide, one row per token"
    );
    debug_assert!(
        o.width == heads * v_head_dim && o.rows == latent.rows,
        "the value-space output is `heads · v_dim` wide, one row per token"
    );
    ctx.fire(
        Fire::at(MLA, entry).apply(Grid::of([pairs, heads, rows], [ABSORB_GROUP, 1, 1])),
        &[
            latent.arg(),
            kv_b.arg(),
            o.arg_mut(),
            heads_i.arg(),
            rank.arg(),
            v_dim.arg(),
            nope.arg(),
        ],
    )
}

pub fn kv_append(
    ctx: &Ctx<'_>,
    kv_c: Tensor,
    k_pe: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.mla_kv_append";
    let entry = dtype_dispatch!(OP, kv_c.dtype, { Bf16 => "attn_mla_kv_append_bf16" });
    if pool.page_size <= 0 {
        return Err(refuse(OP, "the kv page size is zero"));
    }
    debug_assert!(
        k_pe.rows == kv_c.rows,
        "the rope plane is appended beside the latent plane, one row each"
    );
    debug_assert!(
        write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
        "the write tables are u32: one destination page and one in-page slot per row"
    );
    let rows = nonzero(OP, "rows", kv_c.rows)?;
    let width = nonzero(OP, "the appended latent width", kv_c.width.max(k_pe.width))?;
    super::even_lanes(OP, "appended latent", kv_c.width)?;
    if k_pe.width != 0 {
        super::even_lanes(OP, "appended rope tail", k_pe.width)?;
    }
    let width = width.div_ceil(2);

    let values = if k_pe.width == 0 {
        pool.keys
    } else {
        pool.values
    };
    ctx.fire(
        Fire::at(MLA, entry).apply(Grid::of([width, rows, 1], [256, 1, 1])),
        &[
            kv_c.arg(),
            k_pe.arg(),
            pool.keys.arg_mut(),
            values.arg_mut(),
            write_page.arg(),
            write_offset.arg(),
            pool.page_size.arg(),
            stated(OP, kv_c.width)?.arg(),
            stated(OP, k_pe.width)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
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
) -> Result<(), Error> {
    let entry = match selection {
        None => dtype_dispatch!(op, q.dtype, { Bf16 => "attn_mla_paged_bf16" }),
        Some(_) => dtype_dispatch!(op, q.dtype, { Bf16 => "attn_mla_paged_selected_bf16" }),
    };
    if pool.page_size <= 0 {
        return Err(refuse(op, "the kv page size is zero"));
    }
    let heads = nonzero(op, "the head count this attention states", heads)?;
    let rows = nonzero(op, "rows", q.rows)?;
    let ckv = latent_strip(op, "latent rank", kv_lora_rank, MAX_CKV)?;
    if !q_pe.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide rotated q plane does not divide by the {heads} heads",
                q_pe.width
            ),
        ));
    }

    let (kpe, q_pe, values) = if q_pe.width == 0 {
        (0, q, pool.keys)
    } else {
        (
            latent_strip(op, "rope width", q_pe.width / heads, MAX_KPE)?,
            q_pe,
            pool.values,
        )
    };
    debug_assert!(
        positions.dtype == Dtype::I32 && request_of_token.dtype == Dtype::I32,
        "the fire's position and owning-request tables are i32, one entry per row"
    );
    debug_assert!(
        o.rows == q.rows && o.width == heads * kv_lora_rank,
        "the latent reading is `heads · rank` wide, one row per query row"
    );
    let mut args = vec![
        q.arg(),
        q_pe.arg(),
        pool.keys.arg(),
        values.arg(),
        o.arg_mut(),
        positions.arg(),
        request_of_token.arg(),
        pool.page_indices.arg(),
        pool.page_indptr.arg(),
        pool.page_size.arg(),
        stated(op, heads)?.arg(),
        ckv.arg(),
        kpe.arg(),
        sm_scale.arg(),
    ];
    if let Some(selection) = selection {
        debug_assert!(
            selection.dtype == Dtype::I32,
            "the selection is an i32 key-index row"
        );
        if selection.rows != o.rows {
            return Err(refuse(
                op,
                "the selection does not carry one row per query row",
            ));
        }
        args.push(selection.arg());
        args.push(stated(op, nonzero(op, "the selection budget", selection.width)?)?.arg());
    }
    ctx.fire(
        Fire::at(MLA, entry).apply(Grid::of([heads * LANES, rows, 1], [LANES, 1, 1])),
        &args,
    )
}

fn latent_strip(op: &'static str, what: &str, width: u32, max: u32) -> Result<i32, Error> {
    if width == 0 || !width.is_multiple_of(2 * LANES) || width > max {
        return Err(refuse(
            op,
            format!(
                "the {what} {width} is not one this kernel can lane-split in pairs \
                 (a nonzero multiple of {}, at most {max})",
                2 * LANES
            ),
        ));
    }
    stated(op, width)
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
        "attn.mla_decode",
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
        "attn.mla_prefill",
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
    )
}

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
        "attn.mla_decode_selected",
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
        "attn.mla_prefill_selected",
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
    )
}
