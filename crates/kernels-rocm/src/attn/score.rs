use crate::attn::kv;
use crate::attn::plan::PrefillPlan;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, kernel, refuse, stated};
use crate::tensor::{KvPool, RaggedTensor};

const FILE: &str = "attn/score.hip";

/// Wavefronts a block splits the keys over, each dotting one key.
const WAVES: u32 = 4;

const BLOCK: u32 = WAVES * 64;

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

/// Where a fire's capturing lanes keep their rows: two tables of one
/// address a fire lane (`lanes` of them), its key row's (`context` f32s)
/// and its page row's (a page of `page_size` keys an f32), 0 for a lane
/// that captures nothing, which no kernel touches.
#[derive(Debug, Clone, Copy)]
pub struct CaptureRows {
    pub keys: u64,
    pub pages: u64,
    pub lanes: u32,
    pub context: u32,
    pub page_size: u32,
}

impl CaptureRows {
    fn holds(&self, op: &'static str, lane_offset: u32, lanes: u32) -> Result<(), Error> {
        if self.lanes < lane_offset.saturating_add(lanes) {
            return Err(refuse(
                op,
                format!(
                    "the row tables hold {} lanes, and {lanes} lanes from lane {lane_offset} \
                     capture",
                    self.lanes
                ),
            ));
        }
        Ok(())
    }
}

/// Each capturing lane's mass a key of the load's context, averaged over
/// its last `observe` query rows and over the model's `q_heads` (every
/// rank's), added into its key row: zeroed first where `zero` (the first
/// layer's capture), else the layers before's.
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
    rows: CaptureRows,
    zero: bool,
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
    if observe == 0 {
        return Err(refuse(
            OP,
            "the observation window is zero rows wide, which is a capture that observes \
             nothing; the caller states the width it wants, and zero is not one",
        ));
    }
    let context = count(OP, "the load's context", rows.context)?;
    let model_heads = count(OP, "the model's query heads", q_heads)?;
    let page_size = count(OP, "the pool's page size", pool.page_size.unsigned_abs())?;

    let requests = kv::lanes_of(OP, q.indptr)?.unsigned_abs();
    debug_assert_eq!(
        num_q_heads, plan.shape.num_q_heads,
        "`{OP}` captures one row per query head of the plan it was carved with"
    );
    rows.holds(OP, lane_offset, requests)?;

    let floats = head_width.saturating_add(2 * WAVES);
    let smem = floats
        .checked_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4))
        .ok_or_else(|| {
            refuse(
                OP,
                format!(
                    "a {head_width}-wide head over {WAVES} wavefronts overflows its shared plane"
                ),
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
            FILE,
            &format!("::pie::attn::score_capture<{stamp}, {WAVES}, {hnd}>"),
        ))
        .apply(Launch::grid([requests, 1, 1], [BLOCK, 1, 1]).smem(smem)),
        &[
            q.data.arg(),
            q.indptr.arg(),
            pool.keys.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.last_page_lens.arg(),
            ArgValue::Ptr(rows.keys),
            page_size.arg(),
            stated(OP, num_q_heads)?.arg(),
            stated(OP, num_kv_heads)?.arg(),
            head_dim.arg(),
            sm_scale.arg(),
            stated(OP, observe)?.arg(),
            stated(OP, lane_offset)?.arg(),
            context.arg(),
            model_heads.arg(),
            i32::from(zero).arg(),
            kv::page_elems(OP, pool, 0)?.arg(),
        ],
    )
}

/// Each capturing lane's key row summed a page of `page_size` keys into its
/// page row: `lanes` of the fire's lanes from `lane_offset`.
pub fn pages(ctx: &Ctx, rows: CaptureRows, lane_offset: u32, lanes: u32) -> Result<(), Error> {
    const OP: &str = "attn.score_pages";
    rows.holds(OP, lane_offset, lanes)?;
    if lanes == 0 {
        return Ok(());
    }
    let context = count(OP, "the load's context", rows.context)?;
    let page_size = count(OP, "the load's page size", rows.page_size)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::score_pages"))
            .apply(Launch::grid([lanes, 1, 1], [BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(rows.keys),
            ArgValue::Ptr(rows.pages),
            stated(OP, lane_offset)?.arg(),
            context.arg(),
            page_size.arg(),
        ],
    )
}
