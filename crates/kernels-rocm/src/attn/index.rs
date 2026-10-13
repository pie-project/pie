use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/index.hip";

const K_BLOCK: u32 = 256;

const POOL_BLOCK: u32 = 256;

/// Keys a scoring block ranks.
const SCORE_KEYS: u32 = 256;

const SELECT_BLOCK: u32 = 1024;

/// Keys a collecting block filters.
const COLLECT_KEYS: u32 = 2048;

/// The most rows a ranking takes through its summaries and candidate
/// lists; past it each row is one block's walk.
const LIST_ROWS: u32 = 512;

/// A scoring block's kept top scores, and a row's candidate list.
const SUMMARY: usize = 16;
const CANDIDATES: usize = 4096;

/// One thread per head, in whole wavefronts.
#[must_use]
fn q_rope_block(n_heads: i32) -> u32 {
    (n_heads.unsigned_abs().div_ceil(64) * 64).clamp(64, 1024)
}

fn pool_pitch(op: &'static str, pool: &KvPool, row: i32) -> Result<(), Error> {
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
        Fire::at(FILE, "::pie::attn::index_knorm_rope<::pie::bf16>")
            .apply(Launch::per_row(k.rows, K_BLOCK)),
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
        Fire::at(FILE, "::pie::attn::index_q_rope<::pie::bf16>")
            .apply(Launch::per_row(q.rows, q_rope_block(n_heads))),
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
        Fire::at(
            "attn/pool.hip",
            "::pie::attn::index_block_mean_paged<::pie::bf16>",
        )
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
    const OP: &str = "attention.index_topk";
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(selection.dtype, Dtype::I32, "`{OP}` writes i32 page ids");
    let heads = count(OP, "the head count this ranking states", heads)?;
    let head_dim = count(OP, "the key width this ranking states", head_dim)?;
    let top_k = count(OP, "the selection budget this ranking states", top_k)?;

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
    // A score row's pitch is whole 16-byte reads for the selection.
    let max_kv = max_kv
        .checked_add(3)
        .map(|v| v & !3)
        .ok_or_else(|| refuse(OP, "the page budget overflows the kernel's int"))?;
    let scores = ctx.scratch(
        OP,
        "attn.index_topk_scores",
        (selection.rows as usize)
            .saturating_mul(max_kv as usize)
            .saturating_mul(core::mem::size_of::<f32>()),
    )?;

    let ratio = count(OP, "the pooling ratio this ranking states", ratio.max(1))?;
    let rows = selection.rows;
    let max_keys = (max_kv / ratio).max(1).unsigned_abs();
    let scores = ArgValue::Ptr(scores as usize as u64);
    let weights = ArgValue::Ptr(weights.map_or(0, |w| w.ptr));
    // The matrix-core scoring holds a row's heads as up to four 16-head
    // tiles of 64- or 128-wide keys; any other shape takes a thread a key.
    let tiles = heads.unsigned_abs().div_ceil(16);
    let mfma = matches!(head_dim, 64 | 128) && tiles <= 4;
    let score_blocks = max_keys.div_ceil(SCORE_KEYS);
    // A decode's few rows go through the summaries and candidate lists; a
    // prefill's many rows fill the device a row a block.
    let lists = if mfma && rows <= LIST_ROWS {
        let rows_n = rows as usize;
        let stride = score_blocks as usize * SUMMARY;
        let take = |name, bytes: usize| ctx.scratch(OP, name, bytes).map(|p| p as usize as u64);
        TopkLists {
            summary: take("attn.index_topk_summary", rows_n * stride * 4)?,
            bound: take("attn.index_topk_bound", rows_n * 4)?,
            count: take("attn.index_topk_count", rows_n * 4)?,
            cand_idx: take("attn.index_topk_cand_idx", rows_n * CANDIDATES * 4)?,
            cand_key: take("attn.index_topk_cand_key", rows_n * CANDIDATES * 4)?,
            summary_stride: stated(OP, u32::try_from(stride).unwrap_or(u32::MAX))?,
        }
    } else {
        TopkLists::default()
    };
    let row_args = |extra: &[ArgValue]| -> Vec<ArgValue> {
        let mut v = vec![
            q.indptr.arg(),
            keys.page_indices.arg(),
            keys.page_indptr.arg(),
            keys.last_page_lens.arg(),
            num_requests.arg(),
            keys.page_size.arg(),
            max_kv.arg(),
        ];
        v.extend_from_slice(extra);
        v
    };
    let launch = Launch::grid([score_blocks, rows, 1], [SCORE_KEYS, 1, 1]);
    if mfma {
        let entry = symbol(&format!(
            "::pie::attn::index_score_paged<::pie::bf16, {head_dim}, {tiles}>"
        ));
        ctx.fire(
            OP,
            Fire::at(FILE, entry).apply(launch),
            &[
                q.data.arg(),
                weights,
                keys.keys.arg(),
                q.indptr.arg(),
                keys.page_indices.arg(),
                keys.page_indptr.arg(),
                keys.last_page_lens.arg(),
                scores,
                num_requests.arg(),
                heads.arg(),
                keys.page_size.arg(),
                max_kv.arg(),
                ratio.arg(),
                ArgValue::Ptr(lists.summary),
                lists.summary_stride.arg(),
                ctx.stage(),
            ],
        )?;
    } else {
        ctx.fire(
            OP,
            Fire::at(FILE, "::pie::attn::index_score_any<::pie::bf16>").apply(launch),
            &[
                q.data.arg(),
                weights,
                keys.keys.arg(),
                q.indptr.arg(),
                keys.page_indices.arg(),
                keys.page_indptr.arg(),
                keys.last_page_lens.arg(),
                scores,
                num_requests.arg(),
                heads.arg(),
                head_dim.arg(),
                keys.page_size.arg(),
                max_kv.arg(),
                ratio.arg(),
                ctx.stage(),
            ],
        )?;
    }
    if lists.count != 0 {
        let mut args = vec![by_value(&lists)];
        args.extend(row_args(&[top_k.arg(), ratio.arg(), ctx.stage()]));
        ctx.fire(
            OP,
            Fire::at(FILE, "::pie::attn::index_topk_bound")
                .apply(Launch::grid([rows, 1, 1], [SELECT_BLOCK, 1, 1])),
            &args,
        )?;
        let mut args = vec![by_value(&lists), scores];
        args.extend(row_args(&[top_k.arg(), ratio.arg(), ctx.stage()]));
        ctx.fire(
            OP,
            Fire::at(FILE, "::pie::attn::index_topk_collect").apply(Launch::grid(
                [max_keys.div_ceil(COLLECT_KEYS), rows, 1],
                [256, 1, 1],
            )),
            &args,
        )?;
    }
    let mut args = vec![scores, selection.arg()];
    args.extend(row_args(&[
        top_k.arg(),
        ratio.arg(),
        by_value(&lists),
        ctx.stage(),
    ]));
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::attn::index_topk_select")
            .apply(Launch::grid([rows, 1, 1], [SELECT_BLOCK, 1, 1])),
        &args,
    )
}

/// `pie::attn::TopkLists`, field for field: a ranking's summaries and
/// candidate lists (all null: every row takes the one-block walk).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct TopkLists {
    summary: u64,
    bound: u64,
    count: u64,
    cand_idx: u64,
    cand_key: u64,
    summary_stride: i32,
}

const _: () = assert!(core::mem::size_of::<TopkLists>() == 48);

fn by_value<P>(params: &P) -> ArgValue {
    ArgValue::Bytes {
        ptr: core::ptr::from_ref(params).cast::<u8>(),
        len: core::mem::size_of::<P>(),
    }
}
