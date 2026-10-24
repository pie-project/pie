use crate::error::Error;
use dtype::Dtype;

use crate::attn::kv;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const FILE: &str = "attn/sdpa_selected.hip";

const ENTRY: &str = "::pie::attn::sdpa_selected::sdpa_paged_selected_kernel";

/// The kernel's block: eight 32-lane groups (`kBlock`, `kLanes`).
const BLOCK: u32 = 256;

const GROUPS: i32 = BLOCK as i32 / 32;

/// Elements of a head one lane holds (`kMaxPer`): heads up to 512 wide.
const MAX_PER: i32 = 16;

/// Blocks enough to fill the device before a block shares its groups out
/// over more than one head.
const WAVE_TARGET: i64 = 296;

/// The matrix-core walk's block (`kSelThreads`) and the keys its LDS lists
/// a window at a time (`kSelTable`): a block of keys wider than that takes
/// the vector walk.
const MFMA_THREADS: u32 = 256;
const MFMA_TABLE: i32 = 1024;

/// The most splits a row's entries take (`kMergeSplits`), and the fold's
/// block.
const MAX_SPLITS: u32 = 16;
const MERGE_BLOCK: u32 = 256;

/// The query heads one block answers: as many as divide the heads, down to
/// one when fewer blocks would leave the device idle.
fn head_group(num_heads: i32, total_tokens: i32) -> i32 {
    let mut g = GROUPS;
    while g > 1
        && (num_heads % g != 0 || i64::from(total_tokens) * i64::from(num_heads / g) < WAVE_TARGET)
    {
        g >>= 1;
    }
    g
}

#[must_use]
const fn smem_bytes(head_dim: i32) -> u32 {
    let groups = GROUPS as i64;
    let per = groups * head_dim as i64 + 2 * groups;
    let bytes = per * 4;
    if bytes < 0 { 0 } else { bytes as u32 }
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
    if !crate::attn::kv::native_bf16(pool) {
        return Err(refuse(op, "the selected walk reads bf16 pages"));
    }
    let head_dim = count(op, "the head width this attention states", head_dim)?;
    if head_dim % 32 != 0 || head_dim / 32 > MAX_PER {
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
    // Per head, the row holds one run of blocks for each kv head.
    let runs = if per_head { kv_heads } else { 1 };
    let width = stated(op, selection.width)?;
    if width <= 0 || width % runs != 0 {
        return Err(refuse(
            op,
            format!("the {width}-wide selection is not one run for each of {runs} kv heads"),
        ));
    }
    let top_k = width / runs;
    let ratio = count(op, "the block width this reader expands", ratio)?;
    let total_tokens = stated(op, o.rows)?;
    if total_tokens <= 0 {
        return Err(refuse(op, "the query this attention was handed is empty"));
    }
    let num_requests = kv::lanes_of(op, q.indptr)?;

    let group = q_heads / kv_heads;
    if matches!(head_dim, 64 | 128 | 256) && group <= 16 && ratio <= MFMA_TABLE {
        return selected_mfma(
            ctx,
            op,
            q,
            selection,
            pool,
            [num_requests, q_heads, kv_heads, head_dim, top_k, ratio],
            sm_scale,
            per_head,
            o,
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
        [BLOCK, 1, 1],
    )
    .smem(smem_bytes(head_dim));
    ctx.fire(
        op,
        Fire::at(kernel(FILE, ENTRY)).apply(launch),
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
        "attn.decode",
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
        "attn.prefill",
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

/// The matrix-core walk: a block per (row, kv head, split of the row's
/// entries), the group's heads sharing each key and value read. Rows too few
/// to fill the device split their entries, the splits' sums folded by
/// `sdpa_selected_merge`.
#[allow(clippy::too_many_arguments)]
fn selected_mfma(
    ctx: &Ctx,
    op: &'static str,
    q: RaggedTensor,
    selection: Tensor,
    pool: &KvPool,
    [num_requests, q_heads, kv_heads, head_dim, top_k, ratio]: [i32; 6],
    sm_scale: f32,
    per_head: bool,
    o: &mut Tensor,
) -> Result<(), Error> {
    let group = q_heads / kv_heads;
    let rows = o.rows;
    #[allow(clippy::cast_sign_loss)]
    let blocks = rows.saturating_mul(kv_heads as u32).max(1);
    let splits = (ctx.multiprocessors().unwrap_or(304) / blocks).clamp(1, MAX_SPLITS);
    let part = if splits > 1 {
        let floats = splits as usize * rows as usize * q_heads as usize * (head_dim as usize + 2);
        ctx.scratch(
            op,
            "attn.selected_splits",
            floats * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        0
    };
    let entry = format!("::pie::attn::sdpa_selected::sdpa_selected_mfma<{head_dim}, {group}>");
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch::grid([rows, kv_heads as u32, splits], [MFMA_THREADS, 1, 1]);
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &entry)).apply(launch),
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
            ArgValue::Ptr(part),
            num_requests.arg(),
            q_heads.arg(),
            kv_heads.arg(),
            pool.page_size.arg(),
            (sm_scale * core::f32::consts::LOG2_E).arg(),
            top_k.arg(),
            ratio.arg(),
            i32::from(per_head).arg(),
            ctx.stage(),
        ],
    )?;
    if splits == 1 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            "::pie::attn::sdpa_selected::sdpa_selected_merge",
        ))
        .apply(Launch::grid(
            [rows, q_heads as u32, 1],
            [(head_dim as u32).min(MERGE_BLOCK), 1, 1],
        )),
        &[
            ArgValue::Ptr(part),
            ArgValue::Ptr(o.ptr),
            q_heads.arg(),
            head_dim.arg(),
            stated(op, splits)?.arg(),
            ctx.stage(),
        ],
    )
}
