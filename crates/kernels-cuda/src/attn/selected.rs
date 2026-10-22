use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::attn::mla::naive::{NAIVE_BLOCK, NAIVE_MAX_PER, head_group};
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated, symbol};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const SELECTED_SM80: &str = "attn/selected_sm80.cu";

const FILE: &str = "attn/selected_sm80.cu";

const ENTRY: &str = "::pie::attn::selected::sdpa_paged_selected_kernel<::pie::bf16>";

#[must_use]
const fn smem_bytes(head_dim: i32) -> u32 {
    let warps = NAIVE_BLOCK as i64 / 32;
    let per = warps * head_dim as i64 + 2 * warps;
    let bytes = per * 4;
    if bytes < 0 { 0 } else { bytes as u32 }
}

const MMA_THREADS: u32 = 128;

/// The head width the tensor-core walk serves this attention at, if the
/// profile asks for it (`kernel.sdpa.selected_mma`) and a kv head's query
/// heads fill at most one mma's 16 rows; `None` keeps the scalar walk.
fn mma_dim(ctx: &Ctx, head_dim: i32, group: i32) -> Option<i32> {
    (ctx.tuning().selected_mma && matches!(head_dim, 128 | 256) && (1..=16).contains(&group))
        .then_some(head_dim)
}

/// `mma::Shape<D>::kBytes`: Q, four warps' key and value chunks, and their
/// running maxima and sums.
const fn mma_smem_bytes(head_dim: i32) -> u32 {
    let ld = head_dim as u32 + 8;
    16 * ld * 2 + 4 * (2 * 16 * ld * 2) + 4 * 16 * 2 * 4
}

/// `mma::Tile`: eight rows a block, six warps, maps of up to 4096 blocks.
const TILE_ROWS: i32 = 8;
const TILE_THREADS: u32 = 192;
const TILE_GROUP: i32 = 12;
const TILE_MAX_BLOCKS: i64 = 4096;

/// `mma::Tile<D>::kBytes`: 96 query rows, two stages of 16 keys' K and V,
/// eight rows' maps of 4096 blocks and the union's list of them.
const fn tile_smem_bytes(head_dim: i32) -> u32 {
    let ld = head_dim as u32 + 8;
    96 * ld * 2 + 2 * (2 * 16 * ld * 2) + 8 * (4096 / 32) * 4 + 4096 * 4
}

#[allow(clippy::too_many_arguments)]
fn selected(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: Option<u32>,
    sm_scale: f32,
    ratio: u32,
    per_head: bool,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(op, q.data.dtype, { Bf16 => () });
    debug_assert_eq!(
        selection.dtype,
        Dtype::I32,
        "the selection is i32 block ids"
    );
    if window.is_some() {
        return Err(refuse(
            op,
            "a selection and a sliding window are two answers to which keys a row reads",
        ));
    }
    let head_dim = count(op, "the head width this attention states", head_dim)?;
    if head_dim % 32 != 0 || head_dim / 32 > NAIVE_MAX_PER {
        return Err(refuse(
            op,
            format!(
                "the head width {head_dim} is not one this kernel can lane-split \
                 (a multiple of 32, at most 512)"
            ),
        ));
    }
    let q_heads = stated(op, q.data.width)? / head_dim;
    if q_heads <= 0 {
        return Err(refuse(op, "the query row is narrower than one head"));
    }
    let kv_heads = match kv_heads {
        Some(kv_heads) => count(op, "the kv head count this attention states", kv_heads)?,
        None => {
            let row = pool.seq_stride;
            if row <= 0 || row % i64::from(head_dim) != 0 {
                return Err(refuse(
                    op,
                    format!(
                        "the pool's token pitch {row} is no whole number of {head_dim}-wide heads"
                    ),
                ));
            }
            i32::try_from(row / i64::from(head_dim))
                .map_err(|_| refuse(op, "the pool row spells more heads than an i32 holds"))?
        }
    };
    if kv_heads <= 0 || q_heads % kv_heads != 0 {
        return Err(refuse(
            op,
            format!(
                "the {q_heads} query heads this row divides into are not a whole number \
                 of the pool row's {kv_heads} kv heads"
            ),
        ));
    }
    if selection.ptr == 0 {
        return Err(refuse(
            op,
            "the selection this attention attends over is null",
        ));
    }
    if selection.rows != o.rows {
        return Err(refuse(
            op,
            "the selection does not carry one row per query row",
        ));
    }
    let runs = if per_head { kv_heads } else { 1 };
    let width = stated(op, selection.width)?;
    if width <= 0 || width % runs != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide selection is not one run per each of {runs} kv heads"),
        ));
    }
    let top_k = width / runs;
    let ratio = count(op, "the block width this reader expands", ratio)?;
    let total_tokens = stated(op, o.rows)?;
    if total_tokens <= 0 {
        return Err(refuse(op, "the query this attention was handed is empty"));
    }
    let num_requests = kv::lanes_of(op, q.indptr)?;
    // Rows that share their keys, where a fire's requests run eight rows or
    // more and every block a row can see fits the tile's maps.
    let max_blocks = i64::from(pool.max_pages_per_request) * i64::from(pool.page_size)
        / i64::from(ratio.max(1))
        + 1;
    if let Some(dim) = mma_dim(ctx, head_dim, q_heads / kv_heads)
        && q_heads / kv_heads <= TILE_GROUP
        && max_blocks <= TILE_MAX_BLOCKS
        && ctx.smem_fits(tile_smem_bytes(dim))
        && total_tokens >= TILE_ROWS * num_requests.max(1)
    {
        #[allow(clippy::cast_sign_loss)]
        let launch = Launch::grid(
            [
                (total_tokens as u32).div_ceil(TILE_ROWS as u32),
                kv_heads as u32,
                1,
            ],
            [TILE_THREADS, 1, 1],
        )
        .smem(tile_smem_bytes(dim));
        return ctx.fire(
            op,
            Fire::at(
                SELECTED_SM80,
                symbol(&format!(
                    "::pie::attn::selected::mma::sdpa_paged_selected_tile_kernel<{dim}>"
                )),
            )
            .apply(launch),
            &[
                q.data.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                q.indptr.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.last_page_lens.arg(),
                selection.arg(),
                ArgValue::Ptr(o.ptr),
                num_requests.arg(),
                q_heads.arg(),
                kv_heads.arg(),
                pool.page_size.arg(),
                sm_scale.arg(),
                top_k.arg(),
                ratio.arg(),
                i32::from(per_head).arg(),
                total_tokens.arg(),
                ctx.stage(),
            ],
        );
    }
    if let Some(dim) = mma_dim(ctx, head_dim, q_heads / kv_heads) {
        #[allow(clippy::cast_sign_loss)]
        let launch = Launch::grid(
            [total_tokens as u32, kv_heads as u32, 1],
            [MMA_THREADS, 1, 1],
        )
        .smem(mma_smem_bytes(dim));
        return ctx.fire(
            op,
            Fire::at(
                SELECTED_SM80,
                symbol(&format!(
                    "::pie::attn::selected::mma::sdpa_paged_selected_mma_kernel<{dim}>"
                )),
            )
            .apply(launch),
            &[
                q.data.arg(),
                pool.keys.arg(),
                pool.values.arg(),
                q.indptr.arg(),
                pool.page_indices.arg(),
                pool.page_indptr.arg(),
                pool.last_page_lens.arg(),
                selection.arg(),
                ArgValue::Ptr(o.ptr),
                num_requests.arg(),
                q_heads.arg(),
                kv_heads.arg(),
                pool.page_size.arg(),
                sm_scale.arg(),
                top_k.arg(),
                ratio.arg(),
                i32::from(per_head).arg(),
                ctx.stage(),
            ],
        );
    }

    let g = head_group(q_heads, total_tokens);
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::grid(
        [
            total_tokens.max(0) as u32,
            (q_heads / g.max(1)).max(1) as u32,
            1,
        ],
        [NAIVE_BLOCK, 1, 1],
    )
    .smem(smem_bytes(head_dim));
    ctx.fire(
        op,
        Fire::at(FILE, ENTRY).apply(launch),
        &[
            q.data.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            q.indptr.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            selection.arg(),
            ArgValue::Ptr(o.ptr),
            num_requests.arg(),
            q_heads.arg(),
            kv_heads.arg(),
            head_dim.arg(),
            pool.page_size.arg(),
            sm_scale.arg(),
            top_k.arg(),
            ratio.arg(),
            g.arg(),
            i32::from(per_head).arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_decode_selected(
    ctx: &Ctx,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    ratio: u32,
    per_head: bool,
    o: &mut Tensor,
) -> Result<(), Error> {
    selected(
        ctx,
        "attn.decode_selected",
        q,
        selection,
        pool,
        window,
        head_dim,
        None,
        sm_scale,
        ratio,
        per_head,
        o,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_selected(
    ctx: &Ctx,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    ratio: u32,
    per_head: bool,
    o: &mut Tensor,
) -> Result<(), Error> {
    selected(
        ctx,
        "attn.prefill_selected",
        q,
        selection,
        pool,
        window,
        head_dim,
        Some(kv_heads),
        sm_scale,
        ratio,
        per_head,
        o,
    )
}
