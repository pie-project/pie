//! The sparse indexer's logits over FP4 keys on tcgen05's block-scaled MMA
//! (`attn/index/logits_fp4_sm100.cu`).

use crate::error::Error;

use crate::attn::index::FP4_ROW;
use crate::attn::index::logits::select_rows;
use crate::jit::{ArgValue, Ctx, Fire, Launch, kernel, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const LOGITS_FP4_SM100: &str = "attn/index/logits_fp4_sm100.cu";

/// `pie::attn::index::logits_fp4::kThreads`: eight warps of epilogue, one
/// of loads, one of MMAs, four of queries, two of moves.
const THREADS: u32 = 512;

/// `pie::attn::index::Fp4Params`, field for field.
#[repr(C)]
struct Fp4Params {
    q: u64,
    w: u64,
    qo_indptr: u64,
    page_indices: u64,
    page_indptr: u64,
    last_page_lens: u64,
    scores: u64,
    plane: u64,
    win: u64,
    requests: i32,
    heads: i32,
    page_size: i32,
    score_stride: i32,
    top_k: i32,
    ratio: i32,
    per_shift: i32,
    rows: i32,
}

/// `pie::attn::index::logits_fp4::smem_bytes`: two Q tiles and four key
/// tiles of 8 KB, their scale atoms, the weights, six tiles as their pages
/// lay them, barriers and the alignment.
const FP4_SMEM: u32 = (2 + 4) * 8192 + (4 + 2) * 512 + 2 * 128 * 4 + 6 * 128 * 68 + 256 + 1024;

/// The pooled index over FP4 keys scored as they are (`index_logits_fp4`),
/// then ranked a block a row (`index_select_rows`): `Ok(false)` where the
/// pool's pages do not land as the kernel reads them (a key a row, a power
/// of two of at least 4 a page, the plane 16-aligned), and the caller
/// scores as before.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ranked_fp4(
    ctx: &Ctx,
    q: RaggedTensor,
    weights: Option<Tensor>,
    keys: &KvPool,
    heads: i32,
    top_k: i32,
    ratio: i32,
    score_stride: i32,
    scores: u64,
    num_requests: i32,
    compact: bool,
    selection: &mut Tensor,
) -> Result<bool, Error> {
    use crate::linear::gemm_sm100::supported;

    const OP: &str = "attn.index_topk";
    let page = keys.page_size;
    // A compact pool's keys a row each, `page / ratio` a page; at ratio 1
    // every row is a key.
    let per = match (compact, ratio) {
        (true, r) if r > 0 && page % r == 0 => page / r,
        (false, 1) => page,
        _ => 0,
    };
    if !supported()
        || !(1..=64).contains(&heads)
        || per < 4
        || !per.unsigned_abs().is_power_of_two()
        || keys.seq_stride * i64::from(if compact { ratio } else { 1 }) != i64::from(FP4_ROW)
        || !keys.keys.ptr.is_multiple_of(16)
        || !q.data.ptr.is_multiple_of(16)
    {
        return Ok(false);
    }
    let params = Fp4Params {
        q: q.data.ptr,
        w: weights.map_or(0, |w| w.ptr),
        qo_indptr: q.indptr.ptr,
        page_indices: keys.page_indices.ptr,
        page_indptr: keys.page_indptr.ptr,
        last_page_lens: keys.last_page_lens.ptr,
        scores,
        plane: keys.keys.ptr,
        win: match ctx.stage() {
            ArgValue::Ptr(at) => at,
            _ => 0,
        },
        requests: num_requests,
        heads,
        page_size: page,
        score_stride,
        top_k,
        ratio,
        per_shift: per.trailing_zeros() as i32,
        rows: stated(OP, selection.rows)?,
    };
    // A pass's 128 columns hold two rows of 64 heads, or four of 32.
    let h = if heads <= 32 { 32 } else { 64 };
    ctx.fire(
        OP,
        Fire::at(kernel(
            LOGITS_FP4_SM100,
            &format!("::pie::attn::index::index_logits_fp4<{h}>"),
        ))
        .apply(
            Launch::grid(
                [ctx.multiprocessors().unwrap_or(148), 1, 1],
                [THREADS, 1, 1],
            )
            .smem(FP4_SMEM),
        ),
        &[ArgValue::of(&params)],
    )?;
    select_rows(
        ctx,
        q,
        keys,
        heads,
        top_k,
        ratio,
        score_stride,
        scores,
        num_requests,
        selection,
    )?;
    Ok(true)
}
