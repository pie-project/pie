//! The sparse indexer's logits on tcgen05 (`attn/index/logits_sm100.cu`).

use crate::error::Error;

use crate::attn::index::K_BLOCK;
use crate::attn::index::QueryTurn;
use crate::jit::{ArgValue, Ctx, Fire, Launch, kernel, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const LOGITS_SM100: &str = "attn/index/logits_sm100.cu";

/// A prefill's rows, from which the pooled index scores on tcgen05 where
/// `kernel.sdpa.index_tc5` says.
pub(crate) const TC5_FROM: u32 = 512;

/// `pie::attn::index::Params`, field for field.
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

/// `pie::attn::index::smem_bytes`: two Q buffers and three key tiles of
/// 32 KB, the warps' sums, the weights, barriers and the alignment.
const TC5_SMEM: u32 = (2 + 3) * 32768 + 8 * 128 * 4 + 2 * 128 * 4 + 256 + 1024;

/// The pooled index scored on tcgen05 (`index_logits_tc5`) and ranked a
/// block a row (`index_select_rows`): `Ok(false)` where the pool's pages do
/// not land as the kernel reads them (a page's entries whole boxes of 4 to
/// 128 that tile 128 blocks), and the caller scores as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ranked_tc5(
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
    use crate::linear::gemm_sm100::supported;

    const OP: &str = "attn.index_topk";
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
        Fire::at(kernel(
            LOGITS_SM100,
            &format!("::pie::attn::index::index_logits_tc5<{h}>"),
        ))
        .apply(
            Launch::grid(
                [groups.min(ctx.multiprocessors().unwrap_or(148)), 1, 1],
                [384, 1, 1],
            )
            .smem(TC5_SMEM),
        ),
        &[ArgValue::of(&map), ArgValue::of(&map8), bytes],
    )?;
    select(ctx, bytes, selection)?;
    Ok(true)
}

/// The rows a scorer landed in `scores` ranked a block a row
/// (`index_select_rows`), as [`ranked_tc5`] ranks its own.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_rows(
    ctx: &Ctx,
    q: RaggedTensor,
    keys: &KvPool,
    heads: i32,
    top_k: i32,
    ratio: i32,
    score_stride: i32,
    scores: u64,
    num_requests: i32,
    selection: &mut Tensor,
) -> Result<(), Error> {
    let params = Tc5Params {
        q: q.data.ptr,
        w: 0,
        qo_indptr: q.indptr.ptr,
        page_indices: keys.page_indices.ptr,
        page_indptr: keys.page_indptr.ptr,
        last_page_lens: keys.last_page_lens.ptr,
        scores,
        turn_pos: 0,
        win: match ctx.stage() {
            ArgValue::Ptr(at) => at,
            _ => 0,
        },
        requests: num_requests,
        heads,
        page_size: keys.page_size,
        score_stride,
        top_k,
        ratio,
        per: 0,
        rows: stated("attn.index_topk", selection.rows)?,
        turn_dim: 0,
        turn_theta: 0.0,
        turn_factor: 1.0,
        turn_low: 0.0,
        turn_high: 0.0,
        sub: 0,
    };
    select(ctx, ArgValue::of(&params), selection)
}

fn select(ctx: &Ctx, params: ArgValue<'_>, selection: &mut Tensor) -> Result<(), Error> {
    let radix = ctx.tuning().index_radix;
    ctx.fire(
        "attn.index_topk",
        Fire::at(kernel(
            LOGITS_SM100,
            &format!("::pie::attn::index::index_select_rows<{radix}>"),
        ))
        .apply(Launch::grid([selection.rows, 1, 1], [K_BLOCK, 1, 1])),
        &[params, selection.arg()],
    )
}
