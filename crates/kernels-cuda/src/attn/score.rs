use dtype::Dtype;

use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor, Tensor};

const SCORE: &str = "attn/score.cu";

const WARPS: u32 = 8;

const BLOCK: u32 = WARPS * 32;

const STAMPS: [u32; 4] = [64, 128, 256, 512];

/// A fire's capturing lanes as a window of it reads them: `lanes` (i32,
/// one a capturing row, in lane order) names the fire lane each capturing
/// row is, and the window's requests are the fire's lanes from
/// `lane_offset`, `requests` of them (a capture counts them off its query
/// boundaries). A capturing row whose lane lies outside the window is left
/// as it is.
#[derive(Clone, Copy, Debug)]
pub struct CaptureRows {
    pub lanes: Tensor,
    pub lane_offset: u32,
    pub requests: u32,
}

impl CaptureRows {
    fn check(&self, op: &'static str, rows: &Tensor, what: &str) -> Result<(), Error> {
        if self.lanes.dtype != Dtype::I32 || self.lanes.rows != rows.rows {
            return Err(refuse(
                op,
                format!(
                    "the capturing lanes' table is {} {:?} entries, and the {what} hold {} \
                     capturing rows",
                    self.lanes.rows, self.lanes.dtype, rows.rows
                ),
            ));
        }
        if rows.dtype != Dtype::F32 {
            return Err(refuse(
                op,
                format!("the {what} are {:?}, and a mass is an f32 row", rows.dtype),
            ));
        }
        Ok(())
    }
}

fn row_heads(op: &'static str, what: &str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide {what} row does not divide by the head width {head_dim}"),
        ));
    }
    Ok(width / head_dim)
}

fn stamp_for(head_dim: u32) -> Option<u32> {
    STAMPS.into_iter().find(|stamp| head_dim <= *stamp)
}

/// Adds to each capturing lane's row of `scores` (`[capturing, context]`)
/// the attention mass its query heads put on each key of its pages over its
/// last `observe` query rows, averaged over the rows observed and over the
/// model's `q_heads` query heads (each head this rank attends adds a
/// `1 / q_heads` share). With `accumulate` false the rows are zeroed first.
#[allow(clippy::too_many_arguments)]
pub fn capture(
    ctx: &Ctx,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    head_dim: u32,
    kv_heads: u32,
    q_heads: u32,
    sm_scale: f32,
    observe: u32,
    rows: CaptureRows,
    accumulate: bool,
    scores: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.score_capture";
    plan.accepts(OP, head_dim, Some(kv_heads), None)?;
    dtype_dispatch!(OP, q.data.dtype, { Bf16 => () });
    if !kv::native_bf16(pool) {
        return Err(refuse(
            OP,
            format!(
                "the pool's key pages are stored as {:?}, and this capture reads keys \
                 straight out of the pages: it dequantizes nothing, so a quantized or \
                 fp8 pool has no scores to give",
                pool.keys.dtype
            ),
        ));
    }
    rows.check(OP, scores, "score rows")?;
    let head_dim = count(OP, "the head width this capture states", head_dim)?;
    let head_width = head_dim.unsigned_abs();
    let Some(stamp) = stamp_for(head_width) else {
        return Err(refuse(
            OP,
            format!(
                "the {head_width}-wide head is wider than the {}-wide dot this kernel is \
                 stamped for",
                STAMPS[STAMPS.len() - 1]
            ),
        ));
    };
    let num_q_heads = row_heads(OP, "query", q.data.width, head_width)?;
    let num_kv_heads = count(OP, "the kv head count this capture states", kv_heads)?.unsigned_abs();
    if num_q_heads % num_kv_heads != 0 {
        return Err(refuse(
            OP,
            format!("{num_q_heads} query heads do not group over {num_kv_heads} kv heads"),
        ));
    }
    if q_heads < num_q_heads {
        return Err(refuse(
            OP,
            format!("the model states {q_heads} query heads, and this rank attends {num_q_heads}"),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    let context = count(OP, "the context a score row spans", scores.width)?;
    let page_size = count(OP, "the pool's page size", pool.page_size.unsigned_abs())?;

    let requests = kv::lanes_of(OP, q.indptr)?.unsigned_abs();
    debug_assert_eq!(
        num_q_heads, plan.shape.num_q_heads,
        "`{OP}` captures over the query heads of the plan it was carved with"
    );
    let capturing = scores.rows;
    if capturing == 0 {
        return Ok(());
    }

    let floats = head_width.saturating_add(2 * WARPS);
    let smem = floats
        .checked_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4))
        .ok_or_else(|| {
            refuse(
                OP,
                format!("a {head_width}-wide head over {WARPS} warps overflows its shared plane"),
            )
        })?;

    let hnd = if pool.layout == 0 {
        "::pie::false_type::value"
    } else {
        "::pie::true_type::value"
    };
    let lane_offset = stated(OP, rows.lane_offset)?;
    let requests = stated(OP, requests)?;

    if !accumulate {
        ctx.fire(
            OP,
            Fire::at(kernel(SCORE, "::pie::attn::score_zero"))
                .apply(Launch::grid([capturing, 1, 1], [BLOCK, 1, 1])),
            &[
                scores.arg(),
                rows.lanes.arg(),
                lane_offset.arg(),
                requests.arg(),
                context.arg(),
            ],
        )?;
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            SCORE,
            &format!("::pie::attn::score_capture<{stamp}, {WARPS}, {hnd}>"),
        ))
        .apply(Launch::grid([capturing, num_q_heads, 1], [BLOCK, 1, 1]).smem(smem)),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            scores.arg(),
            rows.lanes.arg(),
            lane_offset.arg(),
            requests.arg(),
            page_size.arg(),
            kv::page_elems(OP, pool, 0)?.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
            (1.0 / q_heads as f32).arg(),
            stated(OP, observe)?.arg(),
            context.arg(),
        ],
    )
}

/// Each capturing lane's key row of `keys` (`[capturing, context]`) summed
/// a kv page of `page_size` keys, into its row of `pages` (`[capturing,
/// context pages]`).
pub fn pages(
    ctx: &Ctx,
    keys: Tensor,
    rows: CaptureRows,
    page_size: u32,
    pages: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "attn.score_pages";
    rows.check(OP, &keys, "key rows")?;
    rows.check(OP, pages, "page rows")?;
    let page_size = count(OP, "the pool's page size", page_size)?;
    let context = count(OP, "the context a score row spans", keys.width)?;
    let want = keys.width.div_ceil(page_size.unsigned_abs());
    if pages.width != want {
        return Err(refuse(
            OP,
            format!(
                "the page rows are {} wide, and {} keys in pages of {page_size} are {want}",
                pages.width, keys.width
            ),
        ));
    }
    if keys.rows == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(SCORE, "::pie::attn::score_pages"))
            .apply(Launch::grid([keys.rows, 1, 1], [BLOCK, 1, 1])),
        &[
            keys.arg(),
            pages.arg(),
            rows.lanes.arg(),
            stated(OP, rows.lane_offset)?.arg(),
            stated(OP, rows.requests)?.arg(),
            context.arg(),
            page_size.arg(),
            stated(OP, pages.width)?.arg(),
        ],
    )
}
