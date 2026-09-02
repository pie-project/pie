//! Which sdpa shape a fire earns — the arbitration between the per-row
//! kernel, the tiled one, and the tiled one issued on the matrix unit.
//!
//! Not a row-count decision: the tiled kernel walks runs of equal request
//! inside each 32-row tile, staging that run's keys into threadgroup memory,
//! so its advantage is rows that share a key span and its cost is a serial
//! pass per run. A prefill is one run and wins outright; a fleet of decodes
//! is the opposite shape (measured on llama-1B: batch 32 is 370 tok/s tiled
//! vs 728 per-row). Two fires with the same row count can have opposite
//! shapes, so the request count — which the caller already holds — is what
//! decides.
//!
//! Lives beside `attn` rather than inside it because the parent's entries
//! are one per IR variant and the IR names the shape, not the kernel; which
//! kernel serves it depends on a fact no operand carries (request count).

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::{KvPool, RaggedTensor, Tensor};
use crate::tuning::DeviceTuning;

use super::{DecodePlan, Paged, PrefillPlan, SDPA_TILE, kv_heads_agree, lse_plane, tiled, vector};

/// The one head width `sdpa_paged_mma.metal` is instantiated for.
///
/// The matrix path stages three tiles of `KT x D` halves in 32 KB of
/// threadgroup memory, which is what bounds the list: adding a width means
/// choosing its `KT` there first.
const MMA_HEAD_DIM: u32 = 64;

/// The one head width the `_lse` arm of that file is instantiated for. A
/// second list, not the same number spelled twice: the plain and
/// log-sum-exp arms are separate entry points, so a width can be stamped
/// for one and not the other.
const MMA_LSE_HEAD_DIM: u32 = 64;

/// A simdgroup owns eight query rows and multiplies 8x8 fragments, so the
/// threadgroup is 128 threads rather than the scalar kernel's 1024. The tile
/// HEIGHT is unchanged, which is why the grid is otherwise the same.
const MMA_THREADS: u32 = 128;

const MMA_ENTRY: &str = "sdpa_paged_mma_bfloat16_d_64";

/// The same kernel with buffer 18 seated: the log-sum-exp plane.
const MMA_LSE_ENTRY: &str = "sdpa_paged_mma_lse_bfloat16_d_64";

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

/// Whether a tiled fire this shape should be issued on the matrix unit. A
/// prefill switch layered on top of [`should_tile`]: the scalar tiled kernel
/// computes Q·Kᵀ and P·V as hand-walked dot products near 0.5 TFLOP/s, while
/// the quantized GEMM one dispatch away reaches ~5.6 on the same silicon.
/// The log-sum-exp forms are admitted via the `_lse` entry point (needed by
/// gpt-oss, the one sink-bearing family at head width 64).
#[must_use]
pub fn should_mma(head_dim: u32, lse: bool, tuning: &DeviceTuning) -> bool {
    let stamped = if lse { MMA_LSE_HEAD_DIM } else { MMA_HEAD_DIM };
    tuning.sdpa_mma && head_dim == stamped
}

/// The plan the per-row kernel reads, from the one the tiled kernel would
/// have. The two payloads carry the same tables but stay distinct types
/// since the IR declares distinct struct kinds.
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
/// eighth of the threads. `lse` seats buffer 18 and picks the `_lse` entry
/// point: f32, `[rows x q_heads]`, base 2, `-inf` for a row that kept no
/// key, matching the scalar kernel's plane.
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
    lse: Option<Tensor>,
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
    let mut args = vec![
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
    ];
    let entry = match lse {
        None => MMA_ENTRY,
        Some(lse) => {
            lse_plane(op, lse, &shape);
            args.push(lse.arg_mut());
            MMA_LSE_ENTRY
        }
    };
    ctx.fire(
        Fire::at(MMA_FILE, entry).apply(Grid::of(
            [lanes, shape.rows.div_ceil(SDPA_TILE).max(1), 1],
            [MMA_THREADS, 1, 1],
        )),
        &args,
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
        return mma(
            ctx, op, q, pool, plan, mask, window, head_dim, sm_scale, o, lse,
        );
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

/// `attention.prefill_lse`, arbitrated. All three arms write the plane now —
/// the matrix one through its `_lse` entry point, at the one width that entry
/// is stamped for.
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
    
    
    

    /// **THE ONE PART OF THE MATRIX KERNEL A HOST CAN CHECK, AND IT IS THE
    /// PART THE VERIFY QUEUE WARNS ABOUT.** Session C item 6: "wrong register
    /// layout = wrong answers, not slow ones". The fragment map is a pure
    /// function of `simd_lid` — three lines copied verbatim off
    /// `sdpa_paged_mma.metal` — and the log-sum-exp write rests on three
    /// properties of it that no GPU is needed to check:
    ///
    /// 1. The 32 lanes partition into the fragment's EIGHT rows, four lanes
    ///    each, so a simdgroup owns eight whole query rows and there is no
    ///    cross-simdgroup fold to owe.
    /// 2. Each row's four lanes are closed under `xor 1` and `xor 8` — the
    ///    two shuffles the online softmax already runs — so the `max_score`
    ///    and `sum_exp` those lanes carry are the WHOLE row's, not a quarter
    ///    of it.
    /// 3. Exactly one lane per row holds column zero (`fn == 0`), which is
    ///    the predicate the epilogue publishes under: one f32 per row, not
    ///    four racing writes and not none.
    ///
    /// The four lanes together cover all eight columns, which is the same
    /// statement read from the output side.
    #[test]
    fn the_fragment_map_gives_each_row_four_lanes_and_one_writer() {
        // Verbatim from the shader:
        //   qid = simd_lid / 4
        //   fm  = (qid & 4) + ((simd_lid / 2) % 4)
        //   fn  = (qid & 2) * 2 + (simd_lid % 2) * 2
        let map = |lid: u32| -> (u32, u32) {
            let qid = lid / 4;
            ((qid & 4) + ((lid / 2) % 4), (qid & 2) * 2 + (lid % 2) * 2)
        };
        let mut rows: [Vec<(u32, u32)>; 8] = Default::default();
        for lid in 0..32u32 {
            let (fm, col) = map(lid);
            assert!(fm < 8, "lane {lid} claims fragment row {fm}");
            rows[fm as usize].push((lid, col));
        }
        for (fm, lanes) in rows.iter().enumerate() {
            assert_eq!(lanes.len(), 4, "row {fm} is not four lanes wide");

            // (2) closed under the two shuffles the softmax folds with.
            for &(lid, _) in lanes {
                assert!(lanes.iter().any(|&(o, _)| o == lid ^ 1), "row {fm}: xor 1");
                assert!(lanes.iter().any(|&(o, _)| o == lid ^ 8), "row {fm}: xor 8");
            }

            // (1)/(3) one writer, and the eight columns between them.
            let writers = lanes.iter().filter(|&&(_, col)| col == 0).count();
            assert_eq!(writers, 1, "row {fm} publishes its lse {writers} times");
            let mut cols: Vec<u32> = lanes.iter().flat_map(|&(_, c)| [c, c + 1]).collect();
            cols.sort_unstable();
            assert_eq!(cols, (0..8).collect::<Vec<_>>(), "row {fm} is not covered");
        }
    }

}
