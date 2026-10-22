use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const POOL: &str = "attn/compressed/pool.cu";
const INDEX: &str = "attn/index/index.cu";
#[cfg(feature = "cuda")]
const LOGITS_SM100: &str = "attn/index/logits_sm100.cu";

const K_BLOCK: u32 = 256;

const POOL_BLOCK: u32 = 256;

#[must_use]
fn q_rope_block(n_heads: i32) -> u32 {
    (n_heads.unsigned_abs().div_ceil(32) * 32).max(32)
}

fn pool_pitch(op: &'static str, pool: &KvPool, row: i32) -> Result<(), Error> {
    // A row of bf16 keys written into a narrower plane would spill into the
    // next cell's; an index's keys stay bf16 at any kv.
    if pool.keys.dtype != Dtype::Bf16 {
        return Err(refuse(
            op,
            format!(
                "an index's key rows are bf16, and this pool holds {:?} ones; declare the \
                 index's kv space bf16",
                pool.keys.dtype
            ),
        ));
    }
    if pool.layout != 0 {
        return Err(refuse(
            op,
            "a contiguous index-key row cannot land in an HND pool: a token step there is \
             one head wide and the row would have to be scattered",
        ));
    }
    if row <= 0 {
        return Err(refuse(op, "the index key row is zero-wide"));
    }
    if pool.seq_stride != i64::from(row) {
        return Err(refuse(
            op,
            format!(
                "the pool's token pitch {} is not the {row}-wide row this index writes",
                pool.seq_stride
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn layernorm_rope(
    ctx: &Ctx,
    k: &mut Tensor,
    positions: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "attention.index_layernorm_rope";
    dtype_dispatch!(OP, k.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let head_dim = count(OP, "the index key row's width", k.width)?;
    let rope_dim = stated(OP, rope_dim)?;
    ctx.fire(
        OP,
        // A warp a row where the profile says (`kernel.elemwise.rows_vec`).
        if ctx.tuning().rows_vec {
            Fire::at(INDEX, "::pie::attn::index_knorm_rope_warp<::pie::bf16>")
                .apply(Launch::per_row(k.rows, 32))
        } else {
            Fire::at(INDEX, "::pie::attn::index_knorm_rope<::pie::bf16>")
                .apply(Launch::per_row(k.rows, K_BLOCK))
        },
        &[
            k.arg(),
            weight.arg(),
            bias.arg(),
            positions.arg(),
            head_dim.arg(),
            rope_dim.arg(),
            theta.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub fn rope(
    ctx: &Ctx,
    q: &mut Tensor,
    positions: Tensor,
    heads: u32,
    head_dim: u32,
    rope_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "attention.index_rope";
    dtype_dispatch!(OP, q.dtype, { Bf16 => () });
    debug_assert_eq!(positions.dtype, Dtype::I32, "`{OP}` reads i32 positions");
    let n_heads = count(OP, "the head count this rotation states", heads)?;
    let head_dim = stated(OP, head_dim)?;
    let rope_dim = stated(OP, rope_dim)?;
    ctx.fire(
        OP,
        // A thread a (head, pair) where the profile says
        // (`kernel.elemwise.rows_vec`).
        if ctx.tuning().rows_vec {
            Fire::at(INDEX, "::pie::attn::index_q_rope_pairs<::pie::bf16>")
                .apply(Launch::per_row(q.rows, K_BLOCK))
        } else {
            Fire::at(INDEX, "::pie::attn::index_q_rope<::pie::bf16>")
                .apply(Launch::per_row(q.rows, q_rope_block(n_heads)))
        },
        &[
            q.arg(),
            positions.arg(),
            n_heads.arg(),
            head_dim.arg(),
            rope_dim.arg(),
            theta.arg(),
            ctx.stage(),
        ],
    )
}

/// The bytes of one DeepSeek-V4.1 FP4 index key (`attn/index.cuh`'s
/// `index_fp4`): 128 dims as 64 bytes of e2m1 pairs and 4 ue8m0 exponents.
pub const FP4_ROW: u32 = 68;

/// Whether `keys` holds FP4 index keys: a u8 plane [`FP4_ROW`] wide.
#[must_use]
pub fn fp4_keys(keys: &KvPool) -> bool {
    keys.keys.dtype == Dtype::U8
        && keys.layout == 0
        && keys.seq_stride > 0
        && i64::from(FP4_ROW) % keys.seq_stride == 0
}

/// The index keys (bf16, 128 wide) quantized into FP4 rows at their
/// boundaries' positions, as DeepSeek-V4.1's reference stores them:
/// `pool_kv_append` onto an FP4 key pool.
pub fn store_fp4(
    ctx: &Ctx,
    entries: Tensor,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    keys: &KvPool,
) -> Result<(), Error> {
    const OP: &str = "attention.pool_kv_append";
    if !fp4_keys(keys) {
        return Err(refuse(OP, "the pool does not hold FP4 index keys"));
    }
    if entries.dtype != Dtype::Bf16 || entries.width != 128 {
        return Err(refuse(
            OP,
            "an FP4 index key is written from a bf16 key of 128",
        ));
    }
    if entries.rows == 0 {
        return Ok(());
    }
    let div = crate::attn::pool::packing(OP, keys, i64::from(FP4_ROW))?;
    ctx.fire(
        OP,
        Fire::at(INDEX, "::pie::attn::index_store_fp4")
            .apply(Launch::grid([entries.rows.div_ceil(4), 1, 1], [128, 1, 1])),
        &[
            entries.arg(),
            keys.keys.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            stated(OP, entries.rows)?.arg(),
            keys.page_size.arg(),
            div.arg(),
        ],
    )
}

/// `q`'s rows read at FP4 (each 32-block quantized under a ue8m0 exponent
/// and read back), as DeepSeek-V4.1's reference reads its index queries
/// against FP4 keys, into a scratch plane of their own.
fn fp4_queries(ctx: &Ctx, op: &'static str, q: RaggedTensor) -> Result<RaggedTensor, Error> {
    let width = q.data.width;
    if !width.is_multiple_of(32) {
        return Err(refuse(op, "an FP4 query is whole 32-wide blocks"));
    }
    let bytes = q.data.rows as usize * width as usize * 2;
    let out = ctx.scratch_shared(op, "attn.index_fp4_queries", bytes)? as usize as u64;
    let runs = u64::from(q.data.rows) * u64::from(width / 8);
    if runs > 0 {
        let lanes = u32::try_from(runs).map_err(|_| refuse(op, "the queries overflow a launch"))?;
        ctx.fire(
            op,
            Fire::at(INDEX, "::pie::attn::index_q_fp4").apply(Launch::flat(lanes, 256)),
            &[
                q.data.arg(),
                ArgValue::Ptr(out),
                ArgValue::I64(i64::try_from(runs).unwrap_or(i64::MAX)),
                ctx.stage(),
                stated(op, width)?.arg(),
            ],
        )?;
    }
    Ok(RaggedTensor {
        data: Tensor::new(out, q.data.rows, width, Dtype::Bf16),
        ..q
    })
}

/// FP4 index-key pages, every one the table names, widened into a bf16
/// plane in table order: the pool a bf16 ranking reads instead, its table
/// the identity.
fn stage_fp4(ctx: &Ctx, op: &'static str, keys: &KvPool) -> Result<KvPool, Error> {
    // A compact pool's keys a row each, `page_size / div` a page: staged a
    // row a key, the staged pool as compact (128 / div wide a position).
    let div = crate::attn::pool::packing(op, keys, i64::from(FP4_ROW))?;
    let pages = keys.page_indices.rows;
    let page = u32::try_from(keys.page_size / div).unwrap_or(0);
    let cells = pages.saturating_mul(page);
    let plane =
        ctx.scratch_shared(op, "attn.index_fp4_stage", cells as usize * 128 * 2)? as usize as u64;
    let table =
        ctx.scratch_shared(op, "attn.index_fp4_stage_table", pages as usize * 4)? as usize as u64;
    if pages > 0 {
        ctx.fire(
            op,
            Fire::at(INDEX, "::pie::attn::index_stage_fp4")
                .apply(Launch::grid([pages, 1, 1], [256, 1, 1])),
            &[
                keys.keys.arg(),
                ArgValue::Ptr(plane),
                keys.page_indices.arg(),
                ArgValue::Ptr(table),
                ArgValue::I32(i32::try_from(pages).unwrap_or(i32::MAX)),
                ArgValue::I32(i32::try_from(page).unwrap_or(i32::MAX)),
            ],
        )?;
    }
    let width = u32::try_from(128 / div).unwrap_or(128);
    let positions = cells.saturating_mul(div.unsigned_abs());
    Ok(KvPool {
        keys: Tensor::new(plane, positions, width, Dtype::Bf16),
        values: Tensor::new(plane, positions, width, Dtype::Bf16),
        page_indices: Tensor::new(table, pages, 1, Dtype::U32),
        seq_stride: i64::from(width),
        head_stride: i64::from(width),
        ..*keys
    })
}

pub fn kv_append(
    ctx: &Ctx,
    k: RaggedTensor,
    keys: &KvPool,
    write_page: Tensor,
    write_offset: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_kv_append";
    let _ = (write_page, write_offset);
    dtype_dispatch!(OP, k.data.dtype, { Bf16 => () });
    pool_pitch(OP, keys, stated(OP, k.data.width)?)?;
    kv::write_mla_to_pages(
        ctx,
        OP,
        k.data,
        ArgValue::ABSENT,
        ArgValue::ABSENT,
        k.indptr,
        keys,
        stated(OP, k.data.width)?,
        0,
    )
}

pub fn block_mean(
    ctx: &Ctx,
    boundary_pos: Tensor,
    boundary_req: Tensor,
    keys: &KvPool,
    head_dim: u32,
    ratio: u32,
    entries: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_block_mean";
    dtype_dispatch!(OP, entries.dtype, { Bf16 => () });
    debug_assert!(
        boundary_pos.dtype == Dtype::I32 && boundary_req.dtype == Dtype::I32,
        "`{OP}` reads the i32 block tables `attention.pool_boundary_*` published"
    );
    let head_dim = count(OP, "the key width this mean states", head_dim)?;
    pool_pitch(OP, keys, head_dim)?;
    if stated(OP, entries.width)? != head_dim {
        return Err(refuse(
            OP,
            "the stated head width is not the entry this mean sized",
        ));
    }
    let ratio = count(OP, "the block width this mean pools over", ratio)?;
    let rows = count(OP, "rows", boundary_pos.rows)?;
    ctx.fire(
        OP,
        Fire::at(POOL, "::pie::attn::index_block_mean_paged<::pie::bf16>")
            .apply(Launch::per_row(boundary_pos.rows, POOL_BLOCK)),
        &[
            keys.keys.arg(),
            entries.arg(),
            boundary_pos.arg(),
            boundary_req.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            head_dim.arg(),
            ratio.arg(),
            keys.page_size.arg(),
            ctx.stage(),
        ],
    )?;
    let _ = rows;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn topk(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    selection: &mut Tensor,
) -> Result<(), Error> {
    ranked(
        ctx, q, weights, keys, heads, head_dim, top_k, ratio, None, selection,
    )
    .map(|_| ())
}

/// The turn of an index's queries' tail the ranking lands as it loads them:
/// `rope_partial_last` forward and interleaved, at `positions`.
#[derive(Clone, Copy, Debug)]
pub struct QueryTurn {
    pub positions: Tensor,
    pub dim: u32,
    pub theta: f32,
    pub yarn: Option<crate::elemwise::rope::Yarn>,
}

/// [`topk`] of queries whose tail `turn` rotates first, turned as the
/// tensor-core scorer loads them: whether it fired (nothing is fired where
/// the scorer is another).
#[allow(clippy::too_many_arguments)]
pub fn topk_turned(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    turn: QueryTurn,
    selection: &mut Tensor,
) -> Result<bool, Error> {
    ranked(
        ctx,
        q,
        weights,
        keys,
        heads,
        head_dim,
        top_k,
        ratio,
        Some(turn),
        selection,
    )
}

/// [`topk`]'s blocks read back as the tokens each stands for, then the
/// row's own open block's (`top_k * ratio + ratio - 1` a row, -1 unused):
/// `attention.index_topk` scored `pooled_tokens`.
#[allow(clippy::too_many_arguments)]
pub fn topk_tokens(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    selection: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_topk";
    let ratio = ratio.max(1);
    let width = top_k
        .checked_mul(ratio)
        .and_then(|w| w.checked_add(ratio - 1))
        .ok_or_else(|| refuse(OP, "the token selection's width overflows"))?;
    if selection.width != width {
        return Err(refuse(
            OP,
            format!(
                "a {top_k}-block selection at ratio {ratio} reads back as {width} tokens, not {}",
                selection.width
            ),
        ));
    }
    let blocks_at = ctx.scratch_shared(
        OP,
        "attn.index_topk_blocks",
        (selection.rows as usize).saturating_mul(top_k as usize) * 4,
    )? as usize as u64;
    let mut blocks = Tensor::new(blocks_at, selection.rows, top_k, Dtype::I32);
    ranked(
        ctx,
        q,
        weights,
        keys,
        heads,
        head_dim,
        top_k,
        ratio,
        None,
        &mut blocks,
    )?;
    let num_requests = kv::lanes_of(OP, q.indptr)?;
    ctx.fire(
        OP,
        Fire::at(INDEX, "::pie::attn::index_pool_tokens")
            .apply(Launch::per_row(selection.rows, K_BLOCK)),
        &[
            ArgValue::Ptr(blocks_at),
            q.indptr.arg(),
            keys.page_indptr.arg(),
            keys.last_page_lens.arg(),
            selection.arg(),
            num_requests.arg(),
            keys.page_size.arg(),
            stated(OP, top_k)?.arg(),
            stated(OP, ratio)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn ranked(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    ratio: u32,
    turn: Option<QueryTurn>,
    selection: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "attention.index_topk";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(selection.dtype, Dtype::I32, "`{OP}` writes i32 page ids");
    let heads = count(OP, "the head count this ranking states", heads)?;
    let head_dim = count(OP, "the key width this ranking states", head_dim)?;
    let top_k = count(OP, "the selection budget this ranking states", top_k)?;

    // DeepSeek-V4.1's FP4 keys, read against its queries read at FP4: by
    // the tensor-core scorer as they are, where it takes the shape and the
    // tcgen05 one does not; else widened to bf16 pages for the others.
    let fp4 = fp4_keys(keys);
    let q = if fp4 {
        if turn.is_some() || head_dim != 128 {
            return Ok(false);
        }
        let q = fp4_queries(ctx, OP, q)?;
        #[cfg(feature = "cuda")]
        let tc5 = ctx.tuning().index_tc5 && selection.rows >= TC5_FROM;
        #[cfg(not(feature = "cuda"))]
        let tc5 = false;
        if !(ctx.tuning().index_mma && heads <= 64) || tc5 {
            let staged = stage_fp4(ctx, OP, keys)?;
            return ranked(
                ctx,
                q,
                weights,
                &staged,
                heads.unsigned_abs(),
                head_dim.unsigned_abs(),
                top_k.unsigned_abs(),
                ratio,
                None,
                selection,
            );
        }
        q
    } else {
        if !crate::attn::pool::compact(keys, ratio.max(1), i64::from(head_dim)) {
            pool_pitch(OP, keys, head_dim)?;
        }
        q
    };
    // The keys a row each (`page_size / ratio` a page) in a compact pool.
    let compact = crate::attn::pool::compact(
        keys,
        ratio.max(1),
        if fp4 {
            i64::from(FP4_ROW)
        } else {
            i64::from(head_dim)
        },
    );
    let q_width = stated(OP, q.data.width)?;
    if q_width != heads.saturating_mul(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {q_width}-wide index query does not divide by the stated head count \
                 and width"
            ),
        ));
    }
    if let Some(w) = weights
        && stated(OP, w.width)? != heads
    {
        return Err(refuse(
            OP,
            "the index head weights are not one per stated head",
        ));
    }
    if stated(OP, selection.width)? != top_k {
        return Err(refuse(
            OP,
            "the selection this statement allocated is not the budget it stated",
        ));
    }
    let num_requests = kv::lanes_of(OP, q.indptr)?;

    let max_kv = keys
        .max_pages_per_request
        .checked_mul(keys.page_size)
        .filter(|bound| *bound > 0)
        .ok_or_else(|| refuse(OP, "the page budget this fire's pool row states is zero"))?;
    // A row scores at most one block a `ratio` positions it sees: the
    // scores scratch is a ratio's share of the positions, not all of them
    // (DeepSeek-V4-Flash at ratio 4: 5.4 GB of an 8192-row fire's memory).
    let ratio = count(OP, "the pooling ratio this ranking states", ratio.max(1))?;
    let score_stride = ((max_kv + ratio - 1) / ratio).max(1);
    let scores = ctx.scratch_shared(
        OP,
        "attn.index_topk_scores",
        (selection.rows as usize)
            .saturating_mul(score_stride as usize)
            .saturating_mul(core::mem::size_of::<f32>()),
    )?;

    #[cfg(feature = "cuda")]
    if ctx.tuning().index_tc5
        && !fp4
        && head_dim == 128
        && heads <= 64
        && selection.rows >= TC5_FROM
        && ranked_tc5(
            ctx,
            q,
            weights,
            keys,
            heads,
            top_k,
            ratio,
            turn,
            score_stride,
            scores as usize as u64,
            num_requests,
            compact,
            selection,
        )?
    {
        return Ok(true);
    }

    // The tensor-core scorer where the profile asks for it and the heads fit
    // one mma's rows at its key width.
    // (one to four tiles of 16 heads: DeepSeek-V4's indexer is 64 wide).
    // The ranking's selection by radix (`kernel.sdpa.index_radix`) rather
    // than forty bisection passes.
    let radix = ctx.tuning().index_radix;
    let (entry, smem) = match heads {
        h if ctx.tuning().index_mma && head_dim == 128 && h <= 64 => {
            let tiles: u32 = match h.unsigned_abs().div_ceil(16) {
                1 => 1,
                2 => 2,
                _ => 4,
            };
            (
                format!("::pie::attn::index_topk_paged_mma<::pie::bf16, {tiles}, {radix}, {fp4}>"),
                // The query, 16 rows a tile, and each warp's two chunks of 16
                // keys, at 128 + 8 bf16 a row.
                (16 * tiles + 2 * (K_BLOCK / 32) * 16) * (128 + 8) * 2,
            )
        }
        _ if turn.is_some() => return Ok(false),
        _ => (
            format!("::pie::attn::index_topk_paged<::pie::bf16, {radix}>"),
            0,
        ),
    };
    let mma = smem > 0;
    // The tail turn's frequencies' ramp, where the queries are turned here.
    let (turn_pos, turn_dim, turn_theta, factor, low, high) = match turn {
        Some(t) if t.dim > 0 && t.dim <= 128 && t.dim.is_multiple_of(8) => {
            let dim = stated(OP, t.dim)?;
            let (factor, low, high) = match t.yarn {
                Some(y) => {
                    let (low, high) = crate::elemwise::rope::ramp_bounds(
                        dim,
                        t.theta,
                        y.beta_fast,
                        y.beta_slow,
                        i32::try_from(y.original_max_position).unwrap_or(i32::MAX),
                    );
                    (y.factor, low, high)
                }
                None => (1.0, 0.0, 0.0),
            };
            (t.positions.arg(), dim, t.theta, factor, low, high)
        }
        Some(_) => return Ok(false),
        None => (ArgValue::ABSENT, 0, 0.0, 1.0, 0.0, 0.0),
    };
    let turn_args: Vec<ArgValue> = if mma {
        vec![
            turn_pos,
            turn_dim.arg(),
            turn_theta.arg(),
            factor.arg(),
            low.arg(),
            high.arg(),
        ]
    } else {
        Vec::new()
    };
    ctx.fire(
        OP,
        Fire::at(INDEX, symbol(&entry)).apply(Launch::per_row(selection.rows, K_BLOCK).smem(smem)),
        &[
            q.data.arg(),
            ArgValue::Ptr(weights.map_or(0, |w| w.ptr)),
            keys.keys.arg(),
            q.indptr.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            keys.last_page_lens.arg(),
            ArgValue::Ptr(scores as usize as u64),
            selection.arg(),
            num_requests.arg(),
            heads.arg(),
            head_dim.arg(),
            keys.page_size.arg(),
            score_stride.arg(),
            top_k.arg(),
            ratio.arg(),
            i32::from(compact).arg(),
        ]
        .into_iter()
        .chain(turn_args)
        .chain([ctx.stage()])
        .collect::<Vec<_>>(),
    )?;
    Ok(true)
}

const MAX_BLOCK_MAX_HEADS: i32 = 8;

const MAX_BLOCK_MAX_PER: i32 = 8;

/// `attention.index_topk` scored block by block: each head ranks the blocks
/// a row sees by their best key, keeps the row's own, and writes its run of
/// `top_k` ascending into `selection`, `heads * top_k` wide.
#[allow(clippy::too_many_arguments)]
pub fn topk_block_max(
    ctx: &Ctx,
    q: RaggedTensor,
    keys: &KvPool,
    heads: u32,
    head_dim: u32,
    top_k: u32,
    block: u32,
    selection: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.index_topk";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(selection.dtype, Dtype::I32, "`{OP}` writes i32 block ids");
    let heads = count(OP, "the head count this ranking states", heads)?;
    let head_dim = count(OP, "the key width this ranking states", head_dim)?;
    let top_k = count(OP, "the selection budget this ranking states", top_k)?;
    let block = count(OP, "the block width this ranking scores", block)?;
    if heads <= 0 || heads > MAX_BLOCK_MAX_HEADS {
        return Err(refuse(
            OP,
            format!(
                "a block-max ranking holds at most {MAX_BLOCK_MAX_HEADS} heads' queries, not {heads}"
            ),
        ));
    }
    if head_dim % 32 != 0 || head_dim / 32 > MAX_BLOCK_MAX_PER {
        return Err(refuse(
            OP,
            format!(
                "the key width {head_dim} is not one a block-max ranking lane-splits \
                 (a multiple of 32, at most 256)"
            ),
        ));
    }
    if top_k <= 0 || block <= 0 {
        return Err(refuse(
            OP,
            "a block-max ranking keeps at least one block of at least one key",
        ));
    }

    pool_pitch(OP, keys, head_dim)?;
    let q_width = stated(OP, q.data.width)?;
    if q_width != heads.saturating_mul(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {q_width}-wide index query does not divide by the stated head count \
                 and width"
            ),
        ));
    }
    if stated(OP, selection.width)? != heads.saturating_mul(top_k) {
        return Err(refuse(
            OP,
            "the selection this statement allocated is not one budget per head",
        ));
    }
    let num_requests = kv::lanes_of(OP, q.indptr)?;

    let max_kv = keys
        .max_pages_per_request
        .checked_mul(keys.page_size)
        .filter(|bound| *bound > 0)
        .ok_or_else(|| refuse(OP, "the page budget this fire's pool row states is zero"))?;
    let max_blocks = (max_kv - 1) / block + 1;
    let scores = ctx.scratch(
        OP,
        "attn.index_topk_block_scores",
        (selection.rows as usize)
            .saturating_mul(heads as usize)
            .saturating_mul(max_blocks as usize)
            .saturating_mul(core::mem::size_of::<f32>()),
    )?;

    ctx.fire(
        OP,
        Fire::at(INDEX, "::pie::attn::index_topk_block_max<::pie::bf16>")
            .apply(Launch::per_row(selection.rows, K_BLOCK)),
        &[
            q.data.arg(),
            keys.keys.arg(),
            q.indptr.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            keys.last_page_lens.arg(),
            ArgValue::Ptr(scores as usize as u64),
            selection.arg(),
            num_requests.arg(),
            heads.arg(),
            head_dim.arg(),
            keys.page_size.arg(),
            max_blocks.arg(),
            top_k.arg(),
            block.arg(),
            ctx.stage(),
        ],
    )
}

/// A prefill's rows, from which the pooled index scores on tcgen05 where
/// `kernel.sdpa.index_tc5` says.
#[cfg(feature = "cuda")]
const TC5_FROM: u32 = 512;

/// `pie::attn::index_tc5::Params`, field for field.
#[cfg(feature = "cuda")]
#[repr(C)]
struct Tc5Params {
    q: u64,
    w: u64,
    qo_indptr: u64,
    page_indices: u64,
    page_indptr: u64,
    last_page_lens: u64,
    scores: u64,
    turn_pos: u64,
    win: u64,
    requests: i32,
    heads: i32,
    page_size: i32,
    score_stride: i32,
    top_k: i32,
    ratio: i32,
    per: i32,
    rows: i32,
    turn_dim: i32,
    turn_theta: f32,
    turn_factor: f32,
    turn_low: f32,
    turn_high: f32,
    sub: i32,
}

/// `pie::attn::index_tc5::smem_bytes`: two Q buffers and three key tiles of
/// 32 KB, the warps' sums, the weights, barriers and the alignment.
#[cfg(feature = "cuda")]
const TC5_SMEM: u32 = (2 + 3) * 32768 + 8 * 128 * 4 + 2 * 128 * 4 + 256 + 1024;

/// The pooled index scored on tcgen05 (`index_logits_tc5`) and ranked a
/// block a row (`index_select_rows`): `Ok(false)` where the pool's pages do
/// not land as the kernel reads them (a page's entries whole boxes of 4 to
/// 128 that tile 128 blocks), and the caller scores as before.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn ranked_tc5(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: i32,
    top_k: i32,
    ratio: i32,
    turn: Option<QueryTurn>,
    score_stride: i32,
    scores: u64,
    num_requests: i32,
    compact: bool,
    selection: &mut Tensor,
) -> Result<bool, Error> {
    use crate::linear::tc5::supported;

    const OP: &str = "attention.index_topk";
    let page = keys.page_size;
    let cells = keys.keys.rows;
    let per = if ratio > 0 && page % ratio == 0 {
        page / ratio
    } else {
        0
    };
    if !supported()
        || !matches!(per, 4 | 8 | 16 | 32 | 64 | 128)
        || !cells.is_multiple_of(ratio.unsigned_abs())
        || keys.seq_stride * if compact { i64::from(ratio) } else { 1 } != 128
        || !keys.keys.ptr.is_multiple_of(16)
        || !q.data.ptr.is_multiple_of(16)
    {
        return Ok(false);
    }
    let (turn_pos, turn_dim, turn_theta, turn_factor, turn_low, turn_high) = match turn {
        Some(t) if t.dim > 0 && t.dim <= 128 && t.dim.is_multiple_of(8) => {
            let dim = stated(OP, t.dim)?;
            let (factor, low, high) = match t.yarn {
                Some(y) => {
                    let (low, high) = crate::elemwise::rope::ramp_bounds(
                        dim,
                        t.theta,
                        y.beta_fast,
                        y.beta_slow,
                        i32::try_from(y.original_max_position).unwrap_or(i32::MAX),
                    );
                    (y.factor, low, high)
                }
                None => (1.0, 0.0, 0.0),
            };
            (t.positions.ptr, dim, t.theta, factor, low, high)
        }
        Some(_) => return Ok(false),
        None => (0, 0, 0.0, 1.0, 0.0, 0.0),
    };
    // A page's keys `ratio` rows apart at each group's last row, viewed as
    // [ratio, cells / ratio] rows; a compact pool's a row each.
    let ratio_u = ratio.unsigned_abs();
    let step = if compact { 1 } else { ratio_u };
    let map = crate::attn::bf16_map(
        keys.keys.ptr,
        [128, step, cells / ratio_u],
        [256, 256 * u64::from(step)],
        [64, 1, per.unsigned_abs()],
    )?;
    // Eight consecutive pages' entries a box, where a tile holds eight pages.
    let map8 = crate::attn::bf16_map(
        keys.keys.ptr,
        [128, step, cells / ratio_u],
        [256, 256 * u64::from(step)],
        [64, 1, (8 * per.unsigned_abs()).min(256)],
    )?;
    let win = match ctx.stage() {
        ArgValue::Ptr(at) => at,
        _ => 0,
    };
    let rows = stated(OP, selection.rows)?;
    let params = Tc5Params {
        q: q.data.ptr,
        w: weights.map_or(0, |w| w.ptr),
        qo_indptr: q.indptr.ptr,
        page_indices: keys.page_indices.ptr,
        page_indptr: keys.page_indptr.ptr,
        last_page_lens: keys.last_page_lens.ptr,
        scores,
        turn_pos,
        win,
        requests: num_requests,
        heads,
        page_size: page,
        score_stride,
        top_k,
        ratio,
        per,
        rows,
        turn_dim,
        turn_theta,
        turn_factor,
        turn_low,
        turn_high,
        sub: if compact { 0 } else { ratio - 1 },
    };
    let bytes = ArgValue::of(&params);
    // A pass's 128 lanes hold two rows of 64 heads, or four of 32.
    let h = if heads <= 32 { 32 } else { 64 };
    let groups = selection.rows.div_ceil(128 / h);
    ctx.fire(
        OP,
        Fire::at(
            LOGITS_SM100,
            symbol(&format!("::pie::attn::index_tc5::index_logits_tc5<{h}>")),
        )
        .apply(
            Launch::grid(
                [groups.min(ctx.multiprocessors().unwrap_or(148)), 1, 1],
                [384, 1, 1],
            )
            .smem(TC5_SMEM),
        ),
        &[ArgValue::of(&map), ArgValue::of(&map8), bytes],
    )?;
    let radix = ctx.tuning().index_radix;
    ctx.fire(
        OP,
        Fire::at(
            LOGITS_SM100,
            symbol(&format!(
                "::pie::attn::index_tc5::index_select_rows<{radix}>"
            )),
        )
        .apply(Launch::grid([selection.rows, 1, 1], [K_BLOCK, 1, 1])),
        &[bytes, selection.arg()],
    )?;
    Ok(true)
}
