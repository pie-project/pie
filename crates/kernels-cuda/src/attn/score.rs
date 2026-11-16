use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor};

const SCORE: &str = "attn/score.cu";

const WARPS: u32 = 8;

const BLOCK: u32 = WARPS * 32;

const STAMPS: [u32; 4] = [64, 128, 256, 512];

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

/// The score rows a capture writes: one a capturing lane of `context` f32
/// (its key row, the page row right after it), each where the fire lane's
/// entry of the device table at `table` says when the kernel runs (0: the
/// lane captures none). A recorded launch keeps the table's address and not
/// the rows', so the rows may lie elsewhere at each replay: whoever moves
/// them writes the table.
#[derive(Debug, Clone, Copy)]
pub struct Rows {
    pub table: u64,
    pub context: u32,
}

/// Each capturing lane's attention mass a key of the load's context,
/// averaged over its last `observe` query rows and over the model's
/// `q_heads` (every rank's), added to its row where `accumulate` (a layer
/// before's) or to a zeroed one.
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
    lane_offset: u32,
    rows: Rows,
    accumulate: bool,
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
            format!(
                "the model states {q_heads} query heads and this capture's rows hold \
                 {num_q_heads}: a rank holds no more than the model's"
            ),
        ));
    }
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    let context = count(OP, "the load's context, a row's keys", rows.context)?;
    let page_size = count(OP, "the pool's page size", pool.page_size.unsigned_abs())?;
    let requests = kv::lanes_of(OP, q.indptr)?.unsigned_abs();
    let share = 1.0f32 / q_heads as f32;

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

    ctx.fire(
        OP,
        Fire::at(kernel(
            SCORE,
            &format!("::pie::attn::score_capture<{stamp}, {WARPS}, {hnd}>"),
        ))
        .apply(Launch::grid([requests, 1, 1], [BLOCK, 1, 1]).smem(smem)),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            rows.table.arg(),
            page_size.arg(),
            kv::page_elems(OP, pool, 0)?.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, lane_offset)?.arg(),
            context.arg(),
            share.arg(),
            i32::from(accumulate).arg(),
        ],
    )
}

/// Each of `lanes` lanes' key row summed a kv page of `page_size` keys,
/// into the page row right after it ([`Rows`]).
pub fn pages(
    ctx: &Ctx,
    rows: Rows,
    lanes: u32,
    lane_offset: u32,
    page_size: u32,
) -> Result<(), Error> {
    const OP: &str = "attn.score_pages";
    if lanes == 0 {
        return Ok(());
    }
    let context = count(OP, "the load's context, a row's keys", rows.context)?;
    let page_size = count(OP, "the pool's page size", page_size)?;
    ctx.fire(
        OP,
        Fire::at(kernel(SCORE, "::pie::attn::score_pages"))
            .apply(Launch::grid([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            rows.table.arg(),
            stated(OP, lane_offset)?.arg(),
            context.arg(),
            page_size.arg(),
        ],
    )
}
