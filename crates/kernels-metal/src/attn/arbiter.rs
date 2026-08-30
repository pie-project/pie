//! Which sdpa shape a fire earns — the arbitration between the per-row
//! kernel, the tiled one, and the tiled one issued on the matrix unit.
//!
//! # Why this is not a row count
//!
//! The tiled kernel walks RUNS OF EQUAL REQUEST inside each 32-row tile,
//! staging that run's keys into threadgroup memory and letting only the run's
//! own simdgroups read them. So its whole advantage is rows that share a key
//! span, and its cost is a serial pass per run. A prefill is all one run and
//! wins outright. A FLEET OF DECODES IS THE OPPOSITE SHAPE — thirty-two rows,
//! thirty-two runs, one simdgroup live per pass and the tile staged
//! thirty-two times. Measured on llama-1B: batch 32 is 370 tok/s tiled
//! against 728 per-row, and batch 64 is 480 against 915.
//!
//! Asking only whether the fire fills a tile cannot tell those apart, because
//! the two fires have the SAME row count. The request count is what separates
//! them, and the caller already holds it — the window's qo boundaries are one
//! entry per request plus a terminator.
//!
//! # Why it lives beside `attn` rather than inside it
//!
//! The entries in the parent module are one per IR variant, and the IR names
//! the SHAPE: `attention.prefill` is a statement that a prefill is happening,
//! not an instruction to run the tiled shader. Which kernel serves it is this
//! plane's decision and depends on a fact no operand carries (how many
//! requests the rows belong to), so it is taken here and the parent's entries
//! stay one-to-one with the ops.

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};
use crate::tuning::DeviceTuning;

use super::{DecodePlan, Paged, PrefillPlan, SDPA_TILE, kv_heads_agree, tiled, vector};

/// The one head width `sdpa_paged_mma.metal` is instantiated for.
///
/// The matrix path stages three tiles of `KT x D` halves in 32 KB of
/// threadgroup memory, which is what bounds the list: adding a width means
/// choosing its `KT` there first.
const MMA_HEAD_DIM: u32 = 64;

/// A simdgroup owns eight query rows and multiplies 8x8 fragments, so the
/// threadgroup is 128 threads rather than the scalar kernel's 1024. The tile
/// HEIGHT is unchanged, which is why the grid is otherwise the same.
const MMA_THREADS: u32 = 128;

const MMA_ENTRY: &str = "sdpa_paged_mma_bfloat16_d_64";

const MMA_FILE: &str = "attn/sdpa_paged_mma.metal";

/// Whether a fire's attention should take the tiled kernel rather than the
/// per-row one.
///
/// The threshold is `DeviceTuning::sdpa_tile_min_rows_per_request` and not
/// [`SDPA_TILE`], though the two are the same number on the machine both were
/// measured on: the tile's height is the simdgroup count and cannot move,
/// while this is a crossover and belongs to the machine.
#[must_use]
pub fn should_tile(rows: u32, requests: u32, tuning: &DeviceTuning) -> bool {
    rows / requests.max(1) >= tuning.sdpa_tile_min_rows_per_request
}

/// Whether a tiled fire this shape should be issued on the matrix unit.
///
/// A PREFILL switch layered on top of [`should_tile`], which is unchanged and
/// still keeps a fleet of decodes on the per-row kernel where staging a tile
/// per request would lose. The scalar tiled kernel computes Q·Kᵀ and P·V as
/// hand-walked dot products and runs near 0.5 TFLOP/s while the quantized
/// GEMM one dispatch away reaches ~5.6 on the same silicon; the arithmetic is
/// a matmul and this is issuing it as one.
///
/// The log-sum-exp forms decline because the matrix kernel writes no such
/// plane — a fire that needs one is a partial reading being merged, and there
/// is no `_lse` instantiation to serve it.
#[must_use]
pub fn should_mma(head_dim: u32, lse: bool, tuning: &DeviceTuning) -> bool {
    tuning.sdpa_mma && !lse && head_dim == MMA_HEAD_DIM
}

/// The plan the per-row kernel reads, from the one the tiled kernel would
/// have. The two payloads carry the same tables and stay distinct types
/// because the IR declares distinct struct kinds; `mask` is taken separately
/// because `attention.masked` names its own plane into that seat.
fn as_decode(plan: &PrefillPlan, mask: Tensor) -> DecodePlan {
    DecodePlan {
        positions: plan.positions,
        request_of_token: plan.request_of_token,
        mask,
        mask_enabled: plan.mask_enabled,
        mask_stride: plan.mask_stride,
    }
}

/// The tiled fire, on the matrix unit. Same tables, same tile height, an
/// eighth of the threads.
#[allow(clippy::too_many_arguments)]
fn mma(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
) -> Result<(), Error> {
    let shape = Paged::of(op, q, pool, window, head_dim)?;
    let lanes = shape.q_heads.checked_mul(MMA_THREADS).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the grid will not launch: {} query heads, one {MMA_THREADS}-thread group each",
                shape.q_heads
            ),
        )
    })?;
    ctx.fire(
        Fire::at(MMA_FILE, MMA_ENTRY).apply(Grid::of(
            [lanes, shape.rows.div_ceil(SDPA_TILE).max(1), 1],
            [MMA_THREADS, 1, 1],
        )),
        &[
            q.arg(),
            pool.keys.arg(),
            pool.values.arg(),
            o.arg_mut(),
            stated(op, shape.gqa)?.arg(),
            plan.positions.arg(),
            plan.request_of_token.arg(),
            pool.page_indices.arg(),
            pool.page_indptr.arg(),
            pool.page_size.arg(),
            stated(op, shape.kv_heads)?.arg(),
            sm_scale.arg(),
            mask.arg(),
            plan.mask_stride.arg(),
            plan.mask_enabled.arg(),
            shape.window.arg(),
            ctx.absent()?, // the sink seat; `attention.sink` folds that mass in afterwards
            stated(op, shape.rows)?.arg(),
        ],
    )
}

/// One arbitrated launch: the three shapes, chosen once.
#[allow(clippy::too_many_arguments)]
fn arbitrate(
    ctx: &Ctx<'_>,
    op: &'static str,
    q: Tensor,
    pool: &KvPool,
    plan: &PrefillPlan,
    mask: Tensor,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Option<Tensor>,
    requests: u32,
    tuning: &DeviceTuning,
) -> Result<(), Error> {
    if !should_tile(q.rows, requests, tuning) {
        return vector(
            ctx,
            op,
            q,
            pool,
            &as_decode(plan, mask),
            window,
            head_dim,
            sm_scale,
            o,
            lse,
        );
    }
    if should_mma(head_dim, lse.is_some(), tuning) {
        return mma(ctx, op, q, pool, plan, mask, window, head_dim, sm_scale, o);
    }
    tiled(
        ctx, op, q, pool, plan, mask, window, head_dim, sm_scale, o, lse,
    )
}

/// `attention.prefill`, arbitrated. `requests` is how many lanes the rows
/// belong to — the window's own boundary count.
#[allow(clippy::too_many_arguments)]
pub fn prefill(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    tuning: &DeviceTuning,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    arbitrate(
        ctx, OP, q.data, pool, plan, plan.mask, window, head_dim, sm_scale, o, None, requests,
        tuning,
    )
}

/// `attention.prefill_lse`, arbitrated. The matrix arm declines a fire that
/// wants a log-sum-exp plane; the per-row and tiled arms both write one.
#[allow(clippy::too_many_arguments)]
pub fn prefill_lse(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    kv_heads: u32,
    sm_scale: f32,
    o: Tensor,
    lse: Tensor,
    requests: u32,
    tuning: &DeviceTuning,
) -> Result<(), Error> {
    const OP: &str = "attention.prefill_lse";
    kv_heads_agree(OP, pool, head_dim, kv_heads)?;
    arbitrate(
        ctx,
        OP,
        q.data,
        pool,
        plan,
        plan.mask,
        window,
        head_dim,
        sm_scale,
        o,
        Some(lse),
        requests,
        tuning,
    )
}

/// `attention.masked`, arbitrated — the op-named plane rides the mask seat
/// whichever shape is chosen.
#[allow(clippy::too_many_arguments)]
pub fn masked(
    ctx: &Ctx<'_>,
    q: RaggedTensor,
    plan: &PrefillPlan,
    mask: Tensor,
    pool: &KvPool,
    window: Option<u32>,
    head_dim: u32,
    sm_scale: f32,
    o: Tensor,
    requests: u32,
    tuning: &DeviceTuning,
) -> Result<(), Error> {
    const OP: &str = "attention.masked";
    if mask.dtype != dtype::Dtype::U8 {
        return Err(refuse(
            OP,
            format!(
                "the mask this op states is {:?}, and the shader reads packed u8 mask planes",
                mask.dtype
            ),
        ));
    }
    arbitrate(
        ctx, OP, q.data, pool, plan, mask, window, head_dim, sm_scale, o, None, requests, tuning,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fleet_of_decodes_stays_off_the_tile_it_would_fill() {
        let t = DeviceTuning::default();
        // Thirty-two requests of one row each: the same row count as a
        // 32-row prefill, and 370 tok/s tiled against 728 per-row.
        assert!(!should_tile(32, 32, &t));
        assert!(!should_tile(64, 64, &t));
        // One request contributing the same rows earns the tile.
        assert!(should_tile(32, 1, &t));
        assert!(should_tile(2048, 1, &t));
        // And a fleet of prefills earns it once each member fills a tile.
        assert!(!should_tile(124, 4, &t));
        assert!(should_tile(128, 4, &t));
    }

    #[test]
    fn a_zero_request_count_is_read_as_one_rather_than_dividing_by_it() {
        let t = DeviceTuning::default();
        assert!(should_tile(64, 0, &t));
        assert!(!should_tile(1, 0, &t));
    }

    #[test]
    fn the_matrix_arm_takes_only_the_width_it_is_stamped_at() {
        let t = DeviceTuning::default();
        assert!(should_mma(64, false, &t));
        assert!(!should_mma(128, false, &t));
        // A partial reading that will be merged needs a log-sum-exp plane,
        // and the matrix kernel writes none.
        assert!(!should_mma(64, true, &t));
        assert!(!should_mma(
            64,
            false,
            &DeviceTuning {
                sdpa_mma: false,
                ..DeviceTuning::default()
            }
        ));
    }
}
