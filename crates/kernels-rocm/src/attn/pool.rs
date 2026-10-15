use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::attn::mla::{latent, naive};
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, nonzero, refuse, stated,
};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/pool.hip";

const META_BLOCK: u32 = 128;

const WAVE: u32 = 64;

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WAVE)
            .max(1)
            .saturating_mul(WAVE)
            .min(MAX_BLOCK),
    )
}

const fn compressor_coff(ratio: i32) -> i32 {
    if ratio == 4 { 2 } else { 1 }
}

fn pooling_ratio(op: &'static str, ratio: u32) -> Result<i32, Error> {
    count(op, "the pooling ratio this statement states", ratio)
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

#[allow(clippy::too_many_arguments)]
pub fn boundary_decode(
    ctx: &Ctx,
    positions: Tensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: &mut Tensor,
    boundary_req: &mut Tensor,
    boundary_rope: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_boundary_decode";
    boundary_tables(OP, boundary_pos, boundary_req);
    boundary_rope_table(OP, boundary_pos, boundary_rope);
    let ratio = pooling_ratio(OP, ratio)?;
    let rows = count(OP, "rows", boundary_pos.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_boundary_decode<::pie::i32>")
            .apply(Launch::flat(boundary_pos.rows, META_BLOCK)),
        &[
            positions.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            boundary_rope.arg(),
            rows.arg(),
            ratio.arg(),
            row_valid.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn boundary_prefill(
    ctx: &Ctx,
    positions: RaggedTensor,
    row_valid: Tensor,
    ratio: u32,
    boundary_pos: &mut Tensor,
    boundary_req: &mut Tensor,
    boundary_rope: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_boundary_prefill";
    boundary_tables(OP, boundary_pos, boundary_req);
    boundary_rope_table(OP, boundary_pos, boundary_rope);
    let ratio = pooling_ratio(OP, ratio)?;
    let rows = count(OP, "rows", boundary_pos.rows)?;
    let num_requests = kv::lanes_of(OP, positions.indptr)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_boundary_prefill<::pie::i32>")
            .apply(Launch::flat(boundary_pos.rows, META_BLOCK)),
        &[
            positions.data.arg(),
            positions.indptr.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            boundary_rope.arg(),
            rows.arg(),
            num_requests.arg(),
            ratio.arg(),
            row_valid.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn state_write(
    ctx: &Ctx,
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
    const OP: &str = "attention.pool_state_write";
    dtype_dispatch!(OP, kv.dtype, { Bf16 => () });
    let ratio = pooling_ratio(OP, ratio)?;
    let head_dim = count(OP, "the head width this compressor states", head_dim)?;
    let width = count(OP, "the compressor's row width", kv.width)?;
    if (width != head_dim && width != 2 * head_dim) || score.width != kv.width {
        return Err(refuse(
            OP,
            format!(
                "a ratio-{ratio:?} compressor projects head width {head_dim} or twice it; \
                 the pair handed over is {} and {}",
                kv.width, score.width
            ),
        ));
    }
    let pitch = count(OP, "the state plane's row pitch", state_kv.width)?;
    if state_score.width != state_kv.width {
        return Err(refuse(
            OP,
            "the two state slabs are one plane laid at one pitch",
        ));
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_state_write<::pie::bf16>")
            .apply(route_rows(kv.rows, width.unsigned_abs())),
        &[
            kv.arg(),
            score.arg(),
            state_kv.arg(),
            state_score.arg(),
            write_page.arg(),
            write_offset.arg(),
            width.arg(),
            pages.page_size.arg(),
            pitch.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gather(
    ctx: &Ctx,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pages: &KvPool,
    head_dim: u32,
    ratio: u32,
    state_kv: Tensor,
    state_score: Tensor,
    ape: Tensor,
    entries: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_gather";
    dtype_dispatch!(OP, entries.dtype, { Bf16 => () });
    boundary_tables(OP, &boundary_pos, &boundary_req);
    if entries.width != head_dim {
        return Err(refuse(
            OP,
            format!(
                "the stated head width {head_dim} is not the {}-wide entry it sized",
                entries.width
            ),
        ));
    }
    let ratio = pooling_ratio(OP, ratio)?;
    let head_dim = count(OP, "the head width this gather states", head_dim)?;
    let coff = if ape.is_absent() {
        compressor_coff(ratio)
    } else {
        match count(OP, "the ape's row width", ape.width)? {
            w if w == head_dim => 1,
            w if w == 2 * head_dim => 2,
            w => {
                return Err(refuse(
                    OP,
                    format!("an ape {w} wide is neither one head width ({head_dim}) nor two"),
                ));
            }
        }
    };
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_gather_paged<::pie::bf16>")
            .apply(route_rows(boundary_pos.rows, head_dim.unsigned_abs())),
        &[
            state_kv.arg(),
            state_score.arg(),
            ape.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            pages.page_indices.arg(),
            pages.page_indptr.arg(),
            entries.arg(),
            head_dim.arg(),
            ratio.arg(),
            coff.arg(),
            pages.page_size.arg(),
            count(OP, "the state plane's row pitch", state_kv.width)?.arg(),
        ],
    )
}

pub fn kv_append(
    ctx: &Ctx,
    entries: Tensor,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pool: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_kv_append";
    let _ = (write_page, write_offset);
    dtype_dispatch!(OP, entries.dtype, { Bf16 => () });
    boundary_tables(OP, &boundary_pos, &boundary_req);
    let head_dim = count(OP, "the pooled entry's width", entries.width)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_store_entries<::pie::bf16>")
            .apply(route_rows(entries.rows, entries.width)),
        &[
            entries.arg(),
            pool.keys.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            head_dim.arg(),
            pool.page_size.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
/// What an attention over a pool reads, before the walk is picked.
struct PoolRead<'a> {
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Option<Tensor>,
    entries: &'a KvPool,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
}

/// The attention over a pool: the latent walk for 512-wide entries, the
/// reference walk (a wavefront per row and key stride) for any other width.
fn attend_pool(
    ctx: &Ctx,
    op: &'static str,
    read: &PoolRead<'_>,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, read.q.dtype, { Bf16 => () });
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{op}` lands an f32 log-sum-exp plane"
    );
    debug_assert_eq!(
        read.request_of_token.dtype,
        Dtype::I32,
        "`{op}` reads an i32 owning request per token"
    );
    let entries = read.entries;
    if entries.page_size <= 0 {
        return Err(refuse(op, "the compressed cache page size is zero"));
    }
    let ratio = pooling_ratio(op, read.ratio)?;
    let num_q_heads = count(op, "the head count this attention states", read.heads)?;
    let head_dim = count(op, "the head width this attention states", read.head_dim)?;
    if entries.layout != 0 || entries.seq_stride != i64::from(head_dim) {
        return Err(refuse(
            op,
            format!(
                "the pool's entries are not contiguous {head_dim}-wide rows (pitch {})",
                entries.seq_stride
            ),
        ));
    }
    let rows = nonzero(op, "rows", o.rows)?;
    let (selection, top_k) = match read.selection {
        Some(sel) => (sel.ptr, count(op, "the selection budget", sel.width)?),
        None => (0, 0),
    };

    if head_dim == latent::RANK && !ctx.tuning().mla_naive {
        let max_list = if selection != 0 {
            i64::from(top_k)
        } else {
            i64::from(entries.max_pages_per_request) * i64::from(entries.page_size)
                / i64::from(ratio)
        };
        return latent::gather(
            ctx,
            op,
            super::mla::latent::RANK,
            0,
            rows,
            latent::Gather {
                q_nope: read.q.ptr,
                ckv: entries.keys.ptr,
                page_indices: entries.page_indices.ptr,
                page_indptr: entries.page_indptr.ptr,
                positions: read.positions.ptr,
                req_of_token: read.request_of_token.ptr,
                selection,
                o: o.ptr,
                lse: lse.ptr,
                top_k,
                num_heads: num_q_heads,
                page_size: entries.page_size,
                ratio,
                splits: latent::splits(ctx, rows, num_q_heads, max_list),
                scale_log2: read.sm_scale * core::f32::consts::LOG2_E,
                ..latent::Gather::default()
            },
        );
    }

    if head_dim > naive::MAX_LATENT {
        return Err(refuse(
            op,
            format!(
                "the head width {head_dim} is wider than the {} a lane-split walk holds",
                naive::MAX_LATENT
            ),
        ));
    }
    let g = naive::wave_group(num_q_heads, stated(op, rows)?);
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::grid(
        [rows, (num_q_heads / g).max(1) as u32, 1],
        [naive::NAIVE_BLOCK, 1, 1],
    )
    .smem(naive::naive_smem_bytes(head_dim));
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::attn::pool_lse_paged").apply(launch),
        &[
            read.q.arg(),
            entries.keys.arg(),
            o.arg(),
            lse.arg(),
            read.positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            read.request_of_token.arg(),
            ArgValue::Ptr(selection),
            top_k.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
            ratio.arg(),
            entries.page_size.arg(),
            read.sm_scale.arg(),
            g.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_lse(
    ctx: &Ctx,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    entries: &KvPool,
    ratio: u32,
    heads: u32,
    head_dim: u32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    attend_pool(
        ctx,
        "attention.pool_lse",
        &PoolRead {
            q,
            positions,
            request_of_token,
            selection: None,
            entries,
            ratio,
            heads,
            head_dim,
            sm_scale,
        },
        o,
        lse,
    )
}

/// The attention over the pool entries an indexer selected: row t reads
/// the entries its selection row lists (−1 and entries it cannot see yet
/// skipped), as `attention.pool_lse` reads all it sees.
#[allow(clippy::too_many_arguments)]
pub fn attention_lse_selected(
    ctx: &Ctx,
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
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_lse_selected";
    debug_assert_eq!(
        selection.dtype,
        Dtype::I32,
        "`{OP}` walks the i32 compressed-row ids `attention.index_topk` published"
    );
    if selection.ptr == 0 {
        return Err(refuse(OP, "the selection this attention reads is null"));
    }
    if selection.width != top_k {
        return Err(refuse(
            OP,
            "the selection this statement allocated is not the budget it stated",
        ));
    }
    if selection.rows != o.rows {
        return Err(refuse(
            OP,
            "the selection does not carry one row per query row",
        ));
    }
    attend_pool(
        ctx,
        OP,
        &PoolRead {
            q,
            positions,
            request_of_token,
            selection: Some(selection),
            entries,
            ratio,
            heads,
            head_dim,
            sm_scale,
        },
        o,
        lse,
    )
}
