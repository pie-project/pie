use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, nonzero, refuse, symbol,
};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/pool.cuh";

const META_BLOCK: u32 = 128;

const ATTN_BLOCK: u32 = 128;

const WARP: u32 = 32;

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
    let pitch = count(OP, "the state plane's row pitch", state_kv.width)?;
    // One pass over 16-byte runs, the window split across the block.
    let run = head_dim / 8;
    if ctx.tuning().rows_vec
        && head_dim % 8 == 0
        && run > 0
        && 256 % run == 0
        && pitch % 8 == 0
        && [state_kv.ptr, state_score.ptr, entries.ptr]
            .iter()
            .all(|&at| aligned16(at))
        && (ape.is_absent() || aligned16(ape.ptr))
    {
        let window = (coff * ratio).unsigned_abs();
        let smem = (window * 8).next_multiple_of(16)
            + 256 / run.unsigned_abs() * 3 * head_dim.unsigned_abs() * 4;
        return ctx.fire(
            OP,
            Fire::at(FILE, "::pie::attn::pool_gather_vec<::pie::bf16>")
                .apply(Launch::grid([boundary_pos.rows, 1, 1], [256, 1, 1]).smem(smem)),
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
                pitch.arg(),
                ctx.stage(),
            ],
        );
    }
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
            pitch.arg(),
            ctx.stage(),
        ],
    )
}

/// Whether `pool` holds its compressed entries compact, a row each: its
/// positions `row / ratio` wide (in its plane's elements), a page of
/// `page_size` positions `page_size / ratio` entries `row` wide. Else an
/// entry is a position's row at its group's last position.
#[must_use]
pub fn compact(pool: &KvPool, ratio: u32, row: i64) -> bool {
    ratio > 1
        && pool.layout == 0
        && pool.page_size > 0
        && pool.page_size % i32::try_from(ratio).unwrap_or(i32::MAX) == 0
        && pool.seq_stride.saturating_mul(i64::from(ratio)) == row
}

/// The ratio a writer packs `row`-wide entries into `pool` by: the times its
/// positions' rows divide one (a compact pool's ratio), 1 where an entry is a
/// position's row.
pub fn packing(op: &'static str, pool: &KvPool, row: i64) -> Result<i32, Error> {
    let stride = pool.seq_stride;
    if stride <= 0 || stride >= row {
        return Ok(1);
    }
    let div = row / stride;
    if row % stride != 0
        || pool.page_size <= 0
        || pool.page_size % i32::try_from(div).unwrap_or(i32::MAX) != 0
    {
        return Err(refuse(
            op,
            format!(
                "a {row}-wide entry does not pack into a pool {stride} wide a position at {} a page",
                pool.page_size
            ),
        ));
    }
    Ok(i32::try_from(div).unwrap_or(1))
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
    let div = packing(OP, pool, i64::from(head_dim))?;
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
            div.arg(),
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

    if pool_mma_takes(ctx, head_dim) {
        return pool_mma(
            ctx,
            OP,
            q,
            positions,
            request_of_token,
            None,
            entries,
            ratio,
            0,
            num_q_heads,
            head_dim,
            sm_scale,
            o,
            lse,
        );
    }

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
            i32::from(compact(entries, ratio.unsigned_abs(), i64::from(head_dim))).arg(),
            ctx.stage(),
        ],
    )
}

/// Every query row reads only the compressed rows `selection` names for it:
/// `top_k` i32 ids a row, ended early by a negative one.
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
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(
        lse.dtype,
        Dtype::F32,
        "`{OP}` lands an f32 log-sum-exp plane"
    );
    debug_assert_eq!(
        selection.dtype,
        Dtype::I32,
        "`{OP}` walks the i32 compressed-row ids `attention.index_topk` published"
    );
    debug_assert_eq!(
        request_of_token.dtype,
        Dtype::I32,
        "`{OP}` reads an i32 owning request per token"
    );
    let ratio = pooling_ratio(OP, ratio)?;
    let top_k = count(OP, "the selection budget this reader states", top_k)?;
    let num_q_heads = count(OP, "the head count this attention states", heads)?;
    let head_dim = count(OP, "the head width this attention states", head_dim)?;
    nonzero(OP, "rows", o.rows)?;
    if head_dim.unsigned_abs() > 8 * ATTN_BLOCK {
        return Err(refuse(
            OP,
            format!(
                "the head width {head_dim} is above the {} a block's eight accumulators per lane hold",
                8 * ATTN_BLOCK
            ),
        ));
    }
    if selection.width != top_k.unsigned_abs() {
        return Err(refuse(
            OP,
            format!(
                "the selection is {} wide and this reader walks the {top_k} ids it states",
                selection.width
            ),
        ));
    }
    if selection.rows < o.rows {
        return Err(refuse(
            OP,
            format!(
                "the selection seats {} rows and this reader launches {}",
                selection.rows, o.rows
            ),
        ));
    }

    if pool_mma_takes(ctx, head_dim) {
        return pool_mma(
            ctx,
            OP,
            q,
            positions,
            request_of_token,
            Some(selection),
            entries,
            ratio,
            top_k,
            num_q_heads,
            head_dim,
            sm_scale,
            o,
            lse,
        );
    }

    let smem = head_dim
        .unsigned_abs()
        .saturating_add(ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));

    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::pool_lse_selected_paged").apply(
            Launch::grid([o.rows, num_q_heads.unsigned_abs(), 1], [ATTN_BLOCK, 1, 1]).smem(smem),
        ),
        &[
            q.arg(),
            entries.keys.arg(),
            selection.arg(),
            o.arg(),
            lse.arg(),
            positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            request_of_token.arg(),
            num_q_heads.arg(),
            head_dim.arg(),
            ratio.arg(),
            top_k.arg(),
            entries.page_size.arg(),
            sm_scale.arg(),
            i32::from(compact(entries, ratio.unsigned_abs(), i64::from(head_dim))).arg(),
            ctx.stage(),
        ],
    )
}

/// The head widths the tensor-core pool walk is built for.
const POOL_MMA_WIDTHS: [i32; 3] = [128, 256, 512];

fn pool_mma_takes(ctx: &Ctx, head_dim: i32) -> bool {
    ctx.tuning().pool_mma && POOL_MMA_WIDTHS.contains(&head_dim)
}

/// `pool_lse` (or its selected form, with `selection`) on tensor cores: a
/// block a row's 16 heads (`attn/pool_mma.cuh`).
#[allow(clippy::too_many_arguments)]
fn pool_mma(
    ctx: &Ctx,
    op: &'static str,
    q: Tensor,
    positions: Tensor,
    request_of_token: Tensor,
    selection: Option<Tensor>,
    entries: &KvPool,
    ratio: i32,
    top_k: i32,
    num_q_heads: i32,
    head_dim: i32,
    sm_scale: f32,
    o: &mut Tensor,
    lse: &mut Tensor,
) -> Result<(), Error> {
    const HEADS: u32 = 16;
    const KEYS: u32 = 16;
    const WARPS: u32 = 4;
    let d = head_dim.unsigned_abs();
    let ld = d + 8;
    let smem = HEADS * ld * 2 + 2 * KEYS * ld * 2 + WARPS * HEADS * KEYS * 4;
    let selected = selection.is_some();
    ctx.fire(
        op,
        Fire::at(
            "attn/pool_mma.cuh",
            symbol(&format!(
                "::pie::attn::pool_mma::pool_lse_mma<{d}, {selected}>"
            )),
        )
        .apply(
            Launch::grid(
                [o.rows, num_q_heads.unsigned_abs().div_ceil(HEADS), 1],
                [WARPS * 32, 1, 1],
            )
            .smem(smem),
        ),
        &[
            q.arg(),
            entries.keys.arg(),
            selection.map_or(ArgValue::ABSENT, |s| s.arg()),
            o.arg(),
            lse.arg(),
            positions.arg(),
            entries.page_indices.arg(),
            entries.page_indptr.arg(),
            request_of_token.arg(),
            num_q_heads.arg(),
            ratio.arg(),
            top_k.arg(),
            entries.page_size.arg(),
            sm_scale.arg(),
            i32::from(compact(entries, ratio.unsigned_abs(), i64::from(head_dim))).arg(),
            ctx.stage(),
        ],
    )
}

/// DeepSeek-V4's compressor at its boundaries in one launch
/// (`pool_compress`): the gather `gather` fires, the entry's RMS norm times
/// `weight`, its tail rotated forward (interleaved) at `rope_pos`, and the
/// rotated entry written into `out` as `kv_append` (or, an fp8 pool,
/// `dsv4::store_entries`) writes it. Whether it fired: only under
/// `kernel.elemwise.rows_vec`, for the shapes the one pass is built for;
/// nothing is fired where it did not.
#[allow(clippy::too_many_arguments)]
pub fn compress(
    ctx: &Ctx,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    pages: &KvPool,
    head_dim: u32,
    ratio: u32,
    state_kv: Tensor,
    state_score: Tensor,
    ape: Tensor,
    weight: Tensor,
    eps: f32,
    rope_pos: Tensor,
    rotary_dim: u32,
    theta: f32,
    forward_interleaved: bool,
    yarn: Option<crate::elemwise::rope::Yarn>,
    out: &KvPool,
    state: Option<StateRows>,
) -> Result<bool, Error> {
    const OP: &str = "attention.pool_compress";
    // V4's fp8 rows, or a bf16 pool; either a row a position or compact.
    let fp8 = crate::attn::dsv4::rows_packed(out)
        .is_some_and(|(rows, _)| rows == crate::attn::dsv4::Rows::Ds);
    let out_div = if fp8 {
        crate::attn::dsv4::rows_packed(out).map_or(1, |(_, div)| div)
    } else {
        packing(OP, out, i64::from(head_dim))?
    };
    let run = head_dim / 8;
    let pitch = state_kv.width;
    let fits = ctx.tuning().rows_vec
        && forward_interleaved
        && head_dim.is_multiple_of(8)
        && run > 0
        && 256 % run == 0
        && rotary_dim > 0
        && rotary_dim <= head_dim
        && rotary_dim.is_multiple_of(2)
        && pitch.is_multiple_of(8)
        && state_kv.dtype == Dtype::Bf16
        && weight.dtype == Dtype::Bf16
        && [state_kv.ptr, state_score.ptr, out.keys.ptr]
            .iter()
            .all(|&at| aligned16(at))
        && (ape.is_absent() || aligned16(ape.ptr))
        && (out_div == 1 || u32::try_from(out_div) == Ok(ratio))
        && if fp8 {
            head_dim == 512
        } else {
            out.keys.dtype == Dtype::Bf16
        };
    // The rows' own state written first, where the fire holds one row a
    // request (the caller's to say): its rows as `state_write` reads them.
    let state_fits = state.as_ref().is_none_or(|s| {
        s.kv.dtype == Dtype::Bf16
            && s.score.dtype == Dtype::Bf16
            && s.kv.width == s.score.width
            && s.kv.width <= pitch
            && s.kv.rows >= boundary_pos.rows
    });
    if !fits || !state_fits {
        return Ok(false);
    }
    boundary_tables(OP, &boundary_pos, &boundary_req);
    let ratio = pooling_ratio(OP, ratio)?;
    let head_dim_i = count(OP, "the head width this compressor states", head_dim)?;
    let coff = if ape.is_absent() {
        compressor_coff(ratio)
    } else {
        match count(OP, "the ape's row width", ape.width)? {
            w if w == head_dim_i => 1,
            w if w == 2 * head_dim_i => 2,
            w => {
                return Err(refuse(
                    OP,
                    format!("an ape {w} wide is neither one head width ({head_dim}) nor two"),
                ));
            }
        }
    };
    let (factor, low_dim, high_dim) = match yarn {
        Some(yn) => {
            let max_position = nonzero(
                OP,
                "the position span this layer's YaRN ramp states",
                yn.original_max_position,
            )?;
            let (low, high) = crate::elemwise::rope::ramp_bounds(
                i32::try_from(rotary_dim).unwrap_or(i32::MAX),
                theta,
                yn.beta_fast,
                yn.beta_slow,
                i32::try_from(max_position).unwrap_or(i32::MAX),
            );
            (yn.factor, low, high)
        }
        None => (1.0, 0.0, 0.0),
    };
    let window = (coff * ratio).unsigned_abs();
    let groups = 256 / run;
    let smem = (window * 8).next_multiple_of(16) + groups * 3 * head_dim * 4 + head_dim * 2;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::attn::pool_compress<{fp8}>")))
            .apply(Launch::grid([boundary_pos.rows, 1, 1], [256, 1, 1]).smem(smem)),
        &[
            state_kv.arg(),
            state_score.arg(),
            ape.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            pages.page_indices.arg(),
            pages.page_indptr.arg(),
            weight.arg(),
            eps.arg(),
            rope_pos.arg(),
            count(OP, "the rotated tail's width", rotary_dim)?.arg(),
            theta.arg(),
            factor.arg(),
            low_dim.arg(),
            high_dim.arg(),
            out.keys.arg(),
            out.page_indices.arg(),
            out.page_indptr.arg(),
            out.page_size.arg(),
            out_div.arg(),
            head_dim_i.arg(),
            ratio.arg(),
            coff.arg(),
            pages.page_size.arg(),
            count(OP, "the state plane's row pitch", pitch)?.arg(),
            state.as_ref().map_or(ArgValue::ABSENT, |s| s.kv.arg()),
            state.as_ref().map_or(ArgValue::ABSENT, |s| s.score.arg()),
            state
                .as_ref()
                .map_or(ArgValue::ABSENT, |s| s.write_page.arg()),
            state
                .as_ref()
                .map_or(ArgValue::ABSENT, |s| s.write_offset.arg()),
            state.as_ref().map_or(0, |s| s.kv.width).arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}

/// A fire's compressor projections and where each row's state lands
/// (`state_write`'s), for [`compress`] to write before it gathers, where
/// every request holds one row of the fire.
pub struct StateRows {
    pub kv: Tensor,
    pub score: Tensor,
    pub write_page: Tensor,
    pub write_offset: Tensor,
}
