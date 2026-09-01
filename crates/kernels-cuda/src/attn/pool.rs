//! `Pool`: pooled (compressed) attention — every `ratio` tokens close a
//! boundary whose pooled entry lands in its own cache. Transcribed from the
//! old POOL claims. The pooled compressor state slabs still have no IR seat and arrive
//! as explicit seam arguments the engine binds from fire state; the
//! geometry the ops once smuggled (`row_valid`, `request_of_token`) is
//! op-named now (see the remaining MENLO-SEAM notes per entry).

use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, nonzero, refuse};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/pool.cuh";

const META_BLOCK: u32 = 128;

const ATTN_BLOCK: u32 = 128;

const WARP: u32 = 32;

/// One block per row, sized to the row in whole warps.
fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

const fn compressor_coff(ratio: i32) -> i32 {
    if ratio == 4 { 2 } else { 1 }
}

fn pooling_ratio(op: &'static str, ratio: u32) -> Result<i32, Error> {
    count(op, "the pooling ratio this statement states", ratio)
}

/// The boundary kernels' third column: the compressed row's rope position,
/// `(p / ratio) * ratio` — the block's FIRST token, which is where
/// `compressor_prefill` ropes the pooled entry (`rows = arange(0, cutoff,
/// ratio)`), and NOT `boundary_pos`'s closing cell. This plane computed it
/// from the day the kernels landed and dropped it into scratch, because the
/// op published no seat for it; it publishes one now, so the column is an
/// operand and the model text ropes at it.
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

/// Marks which decode rows close a pooling boundary. `row_valid` (the
/// CUDA-graph padding mask) is the op's own named input now — the seam that
/// used to smuggle it in from fire state is closed.
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

/// The prefill twin: boundaries within each request's ragged span, the
/// op-named `row_valid` masking as in `boundary_decode`; the fire indptr
/// rides in `positions`.
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

/// **THE ROLLING STATE'S WRITER.** `kv` is the compressor's `wkv · x` and
/// `score` its `wgate · x`, both `[rows, coff * head_dim]`; each row is
/// scattered into the cell `write_page`/`write_offset` name for it — the
/// SOURCE cache's own slot, which is the cell the latent appender writes in
/// the same fire.
///
/// **THIS IS THE OP THAT MAKES [`gather`] POOL SOMETHING**: the two state
/// slabs are a seam the shell owns and no IR value names, and until this
/// entry existed nothing wrote a byte of either.
///
// MENLO-SEAM: as `gather` below — the slabs stay seam arguments, because
// they are addressed by the cache's cell and not by the fire's row.
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
    let coff = compressor_coff(ratio);
    let head_dim = count(OP, "the head width this compressor states", head_dim)?;
    let width = head_dim * coff;
    if kv.width != width.unsigned_abs() || score.width != width.unsigned_abs() {
        return Err(refuse(
            OP,
            format!(
                "a ratio-{ratio:?} compressor projects coff x head width = {width} \
                 columns; the pair handed over is {} and {}",
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

/// Pools the closing window out of the kv cache into per-boundary entries.
///
// MENLO-SEAM: the pooled compressor state (`state_kv`, `state_score`) has no
// IR seat; the engine binds the slabs it staged for this cache. `ape` DOES
// have one now — it is a checkpoint plane and not shell scratch — and the
// arm hands over the shell's absent seat when the compressor states none.
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
    let coff = compressor_coff(ratio);
    let head_dim = count(OP, "the head width this gather states", head_dim)?;
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
            // The plane's ROW PITCH, which is not always this gather's own
            // `coff * head_dim` — see the shader's note.
            count(OP, "the state plane's row pitch", state_kv.width)?.arg(),
        ],
    )
}

/// Stores pooled entries into the compressed cache. The compressed pages
/// are the pool row's storage plane (`pool.keys`).
///
// MENLO-SEAM: the op states its write geometry (`write_page`/
// `write_offset`), but the pooled store still re-derives each entry's cell
// from the boundary tables and the pool's read-side page tables — the
// stated pair goes unread until the store takes explicit descriptors.
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

/// Attention over the compressed entries, with the log-sum-exp plane a
/// later `attention.merge_lse` folds against the dense pass.
/// `request_of_token` (the owning request per token row) is the op's own
/// named input now — the fire-table seam that used to carry it is closed.
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
    const OP: &str = "attention.pool_lse";
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
    let ratio = pooling_ratio(OP, ratio)?;
    let num_q_heads = count(OP, "the head count this attention states", heads)?;
    let head_dim = count(OP, "the head width this attention states", head_dim)?;
    nonzero(OP, "rows", o.rows)?;

    let smem = head_dim
        .unsigned_abs()
        .saturating_add(ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));

    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_lse_paged").apply(
            Launch::grid([o.rows, num_q_heads.unsigned_abs(), 1], [ATTN_BLOCK, 1, 1]).smem(smem),
        ),
        &[
            q.arg(),
            entries.keys.arg(),
            o.arg(),
            lse.arg(),
            positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            request_of_token.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
            ratio.arg(),
            entries.page_size.arg(),
            sm_scale.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
