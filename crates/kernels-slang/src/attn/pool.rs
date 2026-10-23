//! The compressed KV pool (`attn.pool_*`) from `kernels/attn/pool.slang`:
//! block boundaries, the compressor's state pages, the pooled entries and
//! their pages, and the attention over pooled entries with its LSE.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const POOL: &str = "attn/pool.slang";

const META_BLOCK: u32 = 128;

const ATTN_BLOCK: u32 = 128;

const POOL_HEAD_MAX: u32 = 512;

const LANE_GROUP: u32 = 256;

const fn compressor_coff(ratio: u32) -> i32 {
    if ratio == 4 { 2 } else { 1 }
}

fn boundary_tables(op: &'static str, boundary_pos: &Tensor, boundary_req: &Tensor) {
    debug_assert_eq!(
        boundary_pos.dtype,
        Dtype::I32,
        "`{op}` reads i32 boundary positions"
    );
    debug_assert_eq!(
        boundary_req.dtype,
        Dtype::I32,
        "`{op}` reads i32 boundary requests"
    );
    debug_assert_eq!(
        boundary_pos.rows, boundary_req.rows,
        "`{op}`'s boundary tables are one entry per token row"
    );
}

fn boundary_rope_table(op: &'static str, boundary_pos: &Tensor, boundary_rope: &Tensor) {
    debug_assert_eq!(
        boundary_rope.dtype,
        Dtype::I32,
        "`{op}` writes i32 compressed-row rope positions"
    );
    debug_assert_eq!(
        boundary_pos.rows, boundary_rope.rows,
        "`{op}`'s rope column is one entry per token row"
    );
}

#[allow(clippy::too_many_arguments)]
pub fn boundary_decode(
    ctx: &Ctx<'_>,
    positions: Tensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_boundary_decode";
    boundary_tables(OP, &boundary_pos, &boundary_req);
    boundary_rope_table(OP, &boundary_pos, &boundary_rope);
    let n = stated(OP, nonzero(OP, "rows", boundary_pos.rows)?)?;
    let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
    ctx.fire(
        Fire::at(POOL, "attn_pool_boundary")
            .apply(Grid::of([boundary_pos.rows, 1, 1], [META_BLOCK, 1, 1])),
        &[
            positions.arg(),
            boundary_pos.arg_mut(),
            boundary_req.arg_mut(),
            boundary_rope.arg_mut(),
            row_valid.arg(),
            request_of_token.arg(),
            n.arg(),
            ratio.arg(),
        ],
    )
}

pub fn boundary_prefill(
    ctx: &Ctx<'_>,
    positions: RaggedTensor,
    request_of_token: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    boundary_rope: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_boundary_prefill";
    boundary_tables(OP, &boundary_pos, &boundary_req);
    boundary_rope_table(OP, &boundary_pos, &boundary_rope);
    let n = stated(OP, nonzero(OP, "rows", boundary_pos.rows)?)?;
    let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
    ctx.fire(
        Fire::at(POOL, "attn_pool_boundary")
            .apply(Grid::of([boundary_pos.rows, 1, 1], [META_BLOCK, 1, 1])),
        &[
            positions.data.arg(),
            boundary_pos.arg_mut(),
            boundary_req.arg_mut(),
            boundary_rope.arg_mut(),
            row_valid.arg(),
            request_of_token.arg(),
            n.arg(),
            ratio.arg(),
        ],
    )
}

pub fn state_write(
    ctx: &Ctx<'_>,
    kv: Tensor,
    score: Tensor,
    pages: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
    head_dim: u32,
    ratio: u32,
    state_kv: Tensor,
    state_score: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_state_write";
    let entry = dtype_dispatch!(OP, kv.dtype, { Bf16 => "attn_pool_state_write_bf16" });
    debug_assert_eq!(
        score.dtype, kv.dtype,
        "`{OP}` writes both projections in one element"
    );
    debug_assert!(
        state_kv.dtype == kv.dtype && state_score.dtype == kv.dtype,
        "`{OP}` lands the state plane in the projections' own element"
    );
    debug_assert!(
        write_page.dtype == Dtype::U32 && write_offset.dtype == Dtype::U32,
        "the write tables are u32: one destination page and one in-page slot per row"
    );
    if pages.page_size <= 0 {
        return Err(refuse(OP, "the source pool's page size is zero"));
    }
    let head_dim = nonzero(OP, "the head width this compressor states", head_dim)?;
    let ratio = nonzero(OP, "the pooling ratio", ratio)?;
    let width = nonzero(OP, "the compressor's row width", kv.width)?;
    if (width != head_dim && width != 2 * head_dim) || score.width != width {
        return Err(refuse(
            OP,
            format!(
                "a ratio-{ratio} compressor projects head width {head_dim} or twice it; the \
                 pair handed over is {} and {}",
                kv.width, score.width
            ),
        ));
    }
    let coff = width / head_dim;
    debug_assert_eq!(
        score.rows, kv.rows,
        "the two projections are one row per token row"
    );
    let pitch = state_kv.width;
    if state_score.width != pitch {
        return Err(refuse(
            OP,
            format!(
                "`state_kv` is {pitch} wide and `state_score` is {} — the two state slabs \
                 are one plane laid at one pitch",
                state_score.width
            ),
        ));
    }
    if pitch < width {
        return Err(refuse(
            OP,
            format!(
                "the state slabs are {pitch} wide and this compressor writes coff {coff} x \
                 head width {head_dim} = {width} columns of every row"
            ),
        ));
    }
    let rows = nonzero(OP, "rows", kv.rows)?;
    let pairs = super::even_lanes(OP, "compressor row", width)?;
    super::even_lanes(OP, "state pitch", pitch)?;
    ctx.fire(
        Fire::at(POOL, entry).apply(Grid::of([pairs, rows, 1], [LANE_GROUP, 1, 1])),
        &[
            kv.arg(),
            score.arg(),
            state_kv.arg_mut(),
            state_score.arg_mut(),
            write_page.arg(),
            write_offset.arg(),
            stated(OP, width)?.arg(),
            pages.page_size.arg(),
            stated(OP, pitch)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}

pub fn gather(
    ctx: &Ctx<'_>,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pages: &KvPool,
    head_dim: u32,
    ratio: u32,
    state_kv: Tensor,
    state_score: Tensor,
    ape: Option<Tensor>,
    entries: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_gather";
    let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "attn_pool_gather_bf16" });
    boundary_tables(OP, &boundary_pos, &boundary_req);
    debug_assert_eq!(
        state_kv.dtype, entries.dtype,
        "`{OP}` pools the state plane into an entry of its own element"
    );
    debug_assert_eq!(
        state_score.dtype, entries.dtype,
        "`{OP}` reads the gate logits in the state plane's element"
    );
    if pages.page_size <= 0 {
        return Err(refuse(OP, "the pooled space's page size is zero"));
    }
    let head_dim = nonzero(OP, "the head width this gather states", head_dim)?;
    if head_dim != entries.width {
        return Err(refuse(
            OP,
            format!(
                "the stated head width {head_dim} is not the {}-wide entry it sized",
                entries.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", boundary_pos.rows)?;
    let ratio = nonzero(OP, "the pooling ratio", ratio)?;
    let coff = match ape {
        None => compressor_coff(ratio),
        Some(ape) if ape.width == head_dim => 1,
        Some(ape) if ape.width == 2 * head_dim => 2,
        Some(ape) => {
            return Err(refuse(
                OP,
                format!(
                    "an ape {} wide is neither one head width ({head_dim}) nor two",
                    ape.width
                ),
            ));
        }
    };
    let width = head_dim.saturating_mul(coff.unsigned_abs());
    let pitch = state_kv.width;
    if state_score.width != pitch {
        return Err(refuse(
            OP,
            format!(
                "`state_kv` is {pitch} wide and `state_score` is {} — the two state slabs \
                 are one plane laid at one pitch",
                state_score.width
            ),
        ));
    }
    if pitch < width {
        return Err(refuse(
            OP,
            format!(
                "the state slabs are {pitch} wide and this gather reads coff {coff} x head \
                 width {head_dim} = {width} columns of every row"
            ),
        ));
    }
    if let Some(ape) = ape {
        if ape.dtype != Dtype::F32 {
            return Err(refuse(OP, "the absolute-position plane is read as f32"));
        }
        if ape.width != width || ape.rows != ratio {
            return Err(refuse(
                OP,
                format!(
                    "the absolute-position plane is {} x {} and this gather reads it at \
                     [ratio {ratio}, {width}]",
                    ape.rows, ape.width
                ),
            ));
        }
    }
    let pairs = super::even_lanes(OP, "pooled head", head_dim)?;
    super::even_lanes(OP, "state pitch", pitch)?;
    ctx.fire(
        Fire::at(POOL, entry).apply(Grid::of([pairs, rows, 1], [LANE_GROUP, 1, 1])),
        &[
            state_kv.arg(),
            state_score.arg(),
            ape.unwrap_or(state_score).arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            pages.page_indices.arg(),
            pages.page_indptr.arg(),
            entries.arg_mut(),
            stated(OP, head_dim)?.arg(),
            stated(OP, ratio)?.arg(),
            coff.arg(),
            pages.page_size.arg(),
            i32::from(ape.is_some()).arg(),
            stated(OP, pitch)?.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}

pub fn kv_append(
    ctx: &Ctx<'_>,
    entries: Tensor,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_kv_append";
    let _ = (write_page, write_offset);
    let entry = dtype_dispatch!(OP, entries.dtype, { Bf16 => "attn_pool_store_bf16" });
    boundary_tables(OP, &boundary_pos, &boundary_req);
    if pool.page_size <= 0 {
        return Err(refuse(OP, "the compressed cache page size is zero"));
    }
    let head_dim = nonzero(OP, "the pooled entry's width", entries.width)?;
    let rows = nonzero(OP, "rows", entries.rows)?;
    let pairs = super::even_lanes(OP, "pooled entry", head_dim)?;
    ctx.fire(
        Fire::at(POOL, entry).apply(Grid::of([pairs, rows, 1], [LANE_GROUP, 1, 1])),
        &[
            entries.arg(),
            pool.keys.arg_mut(),
            boundary_pos.arg(),
            boundary_req.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            stated(OP, head_dim)?.arg(),
            pool.page_size.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}

fn reader_head(op: &'static str, heads: u32, head_dim: u32) -> Result<(i32, i32), Error> {
    let num_q_heads = nonzero(op, "heads", heads)?;
    let head_dim = nonzero(op, "the head width", head_dim)?;
    super::even_lanes(op, "head", head_dim)?;
    if head_dim > POOL_HEAD_MAX {
        return Err(refuse(
            op,
            format!(
                "the head width {head_dim} is above the {POOL_HEAD_MAX} this flash reader tiles in workgroup memory"
            ),
        ));
    }
    Ok((stated(op, num_q_heads)?, stated(op, head_dim)?))
}

pub fn attention_lse(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    entries: &KvPool,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_lse";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{OP}` lands an f32 log-sum-exp plane"
    );
    debug_assert_eq!(
        request_of_token.dtype,
        Dtype::I32,
        "`{OP}` reads an i32 owning request per token"
    );
    if entries.page_size <= 0 {
        return Err(refuse(OP, "the compressed cache page size is zero"));
    }
    let (num_q_heads, head_dim) = reader_head(OP, heads, head_dim)?;
    let rows = nonzero(OP, "rows", o.rows)?;
    let ratio = stated(OP, nonzero(OP, "the pooling ratio", ratio)?)?;
    ctx.fire(
        Fire::at(POOL, "attn_pool_lse_bf16")
            .apply(Grid::of([ATTN_BLOCK, rows, heads], [ATTN_BLOCK, 1, 1])),
        &[
            q.arg(),
            entries.keys.arg(),
            o.arg_mut(),
            lse.arg_mut(),
            positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            request_of_token.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
            ratio.arg(),
            entries.page_size.arg(),
            sm_scale.arg(),
        ],
    )
}

pub fn attention_lse_selected(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Tensor,
    entries: &KvPool,
    ratio: u32,
    top_k: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.pool_lse_selected";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{OP}` lands an f32 log-sum-exp plane"
    );
    debug_assert_eq!(
        selection.dtype,
        Dtype::I32,
        "`{OP}` walks i32 compressed-row ids"
    );
    debug_assert_eq!(
        request_of_token.dtype,
        Dtype::I32,
        "`{OP}` reads an i32 owning request per token"
    );
    if entries.page_size <= 0 {
        return Err(refuse(OP, "the compressed cache page size is zero"));
    }
    let (num_q_heads, head_dim) = reader_head(OP, heads, head_dim)?;
    let rows = nonzero(OP, "rows", o.rows)?;
    let ratio = nonzero(OP, "the pooling ratio", ratio)?;
    let top_k = nonzero(OP, "the selection budget this reader states", top_k)?;
    if selection.width != top_k {
        return Err(refuse(
            OP,
            format!(
                "the selection is {} wide and this reader walks the {top_k} ids it states",
                selection.width
            ),
        ));
    }
    if selection.rows < rows {
        return Err(refuse(
            OP,
            format!(
                "the selection seats {} rows and this reader launches {rows}",
                selection.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at(POOL, "attn_pool_lse_selected_bf16")
            .apply(Grid::of([ATTN_BLOCK, rows, heads], [ATTN_BLOCK, 1, 1])),
        &[
            q.arg(),
            entries.keys.arg(),
            selection.arg(),
            o.arg_mut(),
            lse.arg_mut(),
            positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            request_of_token.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
            stated(OP, ratio)?.arg(),
            stated(OP, top_k)?.arg(),
            entries.page_size.arg(),
            sm_scale.arg(),
        ],
    )
}
