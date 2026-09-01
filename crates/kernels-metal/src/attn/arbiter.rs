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

use super::{DecodePlan, Paged, PrefillPlan, SDPA_TILE, kv_heads_agree, lse_plane, tiled, vector};

/// The one head width `sdpa_paged_mma.metal` is instantiated for.
///
/// The matrix path stages three tiles of `KT x D` halves in 32 KB of
/// threadgroup memory, which is what bounds the list: adding a width means
/// choosing its `KT` there first.
const MMA_HEAD_DIM: u32 = 64;

/// The one head width the `_lse` arm of that file is instantiated for.
///
/// **IT IS A SECOND LIST AND NOT THE SAME NUMBER SPELLED TWICE.** The plain
/// and log-sum-exp arms are separate `host_name`s off separate entry points,
/// so a width can be stamped for one and not the other; today they agree, and
/// the day they do not, [`should_mma`] has to read the one that matches the
/// fire rather than the one that happens to be nearer.
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

/// Whether a tiled fire this shape should be issued on the matrix unit.
///
/// A PREFILL switch layered on top of [`should_tile`], which is unchanged and
/// still keeps a fleet of decodes on the per-row kernel where staging a tile
/// per request would lose. The scalar tiled kernel computes Q·Kᵀ and P·V as
/// hand-walked dot products and runs near 0.5 TFLOP/s while the quantized
/// GEMM one dispatch away reaches ~5.6 on the same silicon; the arithmetic is
/// a matmul and this is issuing it as one.
///
/// **THE LOG-SUM-EXP FORMS ARE ADMITTED, AND THAT IS THE WHOLE POINT OF THIS
/// ARM FOR ONE FAMILY.** They used to decline: the matrix kernel wrote no such
/// plane, so a fire that needed one — a partial reading being merged, or an
/// attention-sink rescale reading the denominator back — fell to the scalar
/// tiled kernel. gpt-oss is the one family in this catalog with sinks, its
/// head width IS 64, and `!lse` was the only clause it failed, so the model
/// whose 2048-token prefill spends ~36% of itself in that scalar kernel was
/// the one model this arm could not reach (`.wiki/macos-bench.md` §17). The
/// `_lse` entry point closes that; the width test does not move, because a
/// stamp is still a stamp.
#[must_use]
pub fn should_mma(head_dim: u32, lse: bool, tuning: &DeviceTuning) -> bool {
    let stamped = if lse { MMA_LSE_HEAD_DIM } else { MMA_HEAD_DIM };
    tuning.sdpa_mma && head_dim == stamped
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
///
/// `lse` seats buffer 18 and picks the `_lse` entry point with it: the plane
/// is the scalar kernel's — f32, `[rows x q_heads]`, base 2, `-inf` for a row
/// that kept no key — so `attention.sink` and `attention.merge_lse` read what
/// this writes without knowing which kernel wrote it.
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
    use super::*;
    use crate::encode::ArgValue;
    use crate::probe::Probe;

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
        assert!(!should_mma(
            64,
            false,
            &DeviceTuning {
                sdpa_mma: false,
                ..DeviceTuning::default()
            }
        ));
    }

    /// The clause gpt-oss failed, and the only one it failed. A fire that
    /// wants a log-sum-exp plane is admitted at the width the `_lse` entry
    /// point is stamped for and declined at every other, exactly as the plain
    /// arm is — and the tuning escape hatch still closes both.
    #[test]
    fn a_log_sum_exp_fire_is_admitted_at_the_width_the_lse_stamp_serves() {
        let t = DeviceTuning::default();
        assert!(should_mma(64, true, &t));
        assert!(!should_mma(128, true, &t));
        assert!(!should_mma(256, true, &t));
        assert!(!should_mma(
            64,
            true,
            &DeviceTuning {
                sdpa_mma: false,
                ..DeviceTuning::default()
            }
        ));
    }

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, dtype::Dtype::Bf16)
    }

    /// One kv head of 64, pages of 16 — gpt-oss's attention shape with the
    /// head count cut down to what a probe needs.
    fn fixture(rows: u32, q_heads: u32) -> (Tensor, KvPool, PrefillPlan) {
        let q = bf16(1, rows, q_heads * 64);
        let pool = KvPool {
            keys: bf16(2, 64, 64),
            values: bf16(3, 64, 64),
            page_indices: Tensor::new(4, 8, 1, dtype::Dtype::U32),
            page_indptr: Tensor::new(5, 2, 1, dtype::Dtype::U32),
            page_size: 16,
            seq_stride: 64,
            head_stride: 64,
        };
        let plan = PrefillPlan {
            positions: Tensor::new(6, rows, 1, dtype::Dtype::I32),
            request_of_token: Tensor::new(7, rows, 1, dtype::Dtype::I32),
            mask: Tensor::new(8, 1, 512, dtype::Dtype::U8),
            mask_enabled: Tensor::new(9, 1, 1, dtype::Dtype::U8),
            mask_stride: 512,
        };
        (q, pool, plan)
    }

    /// The whole handoff in one launch: the `_lse` entry point is named, the
    /// plane rides buffer 18 as a WRITE binding, and nothing else about the
    /// fire moves — same grid, same tile height, same 128 threads.
    #[test]
    fn the_matrix_lse_fire_seats_the_plane_at_buffer_eighteen() {
        let probe = Probe::default();
        let (rows, q_heads) = (2048u32, 4u32);
        let (q, pool, plan) = fixture(rows, q_heads);
        let lse = Tensor::new(10, rows, q_heads, dtype::Dtype::F32);
        prefill_lse(
            &probe,
            RaggedTensor {
                data: q,
                indptr: Tensor::new(11, 2, 1, dtype::Dtype::I32),
            },
            &plan,
            &pool,
            None,
            64,
            1,
            0.125,
            bf16(12, rows, q_heads * 64),
            lse,
            1,
            &DeviceTuning::default(),
        )
        .expect("a 2048-row single-request prefill earns the matrix arm");
        let (fire, args) = probe.only();
        assert_eq!(fire.file, MMA_FILE);
        assert_eq!(fire.entrypoint, MMA_LSE_ENTRY);
        assert_eq!(fire.lanes, [q_heads * MMA_THREADS, rows / SDPA_TILE, 1]);
        assert_eq!(fire.group, [MMA_THREADS, 1, 1]);
        assert_eq!(args.len(), 19, "the lse arm seats one buffer more");
        assert_eq!(args[18], ArgValue::BufferMut(10));
    }

    /// The plain arm is unchanged by the split: eighteen seats, the plain
    /// entry point, and no nineteenth binding invented for it.
    #[test]
    fn the_plain_matrix_fire_still_seats_eighteen() {
        let probe = Probe::default();
        let (rows, q_heads) = (2048u32, 4u32);
        let (q, pool, plan) = fixture(rows, q_heads);
        prefill(
            &probe,
            RaggedTensor {
                data: q,
                indptr: Tensor::new(11, 2, 1, dtype::Dtype::I32),
            },
            &plan,
            &pool,
            None,
            64,
            1,
            0.125,
            bf16(12, rows, q_heads * 64),
            1,
            &DeviceTuning::default(),
        )
        .expect("the same fire without a plane earns the same arm");
        let (fire, args) = probe.only();
        assert_eq!(fire.entrypoint, MMA_ENTRY);
        assert_eq!(args.len(), 18);
    }

    /// The escape hatch is the whole of the way back, and it has to close the
    /// new arm too: `[metal.tuning] sdpa_mma = false` puts a log-sum-exp
    /// prefill back on `sdpa_paged_tiled_lse`.
    #[test]
    fn the_tuning_hatch_returns_the_lse_fire_to_the_scalar_kernel() {
        let probe = Probe::default();
        let (rows, q_heads) = (2048u32, 4u32);
        let (q, pool, plan) = fixture(rows, q_heads);
        prefill_lse(
            &probe,
            RaggedTensor {
                data: q,
                indptr: Tensor::new(11, 2, 1, dtype::Dtype::I32),
            },
            &plan,
            &pool,
            None,
            64,
            1,
            0.125,
            bf16(12, rows, q_heads * 64),
            Tensor::new(10, rows, q_heads, dtype::Dtype::F32),
            1,
            &DeviceTuning {
                sdpa_mma: false,
                ..DeviceTuning::default()
            },
        )
        .expect("the scalar tiled arm serves the same fire");
        let (fire, args) = probe.only();
        assert_eq!(fire.entrypoint, "sdpa_paged_tiled_lse_bfloat16_d_64");
        assert_eq!(args.len(), 19);
        assert_eq!(args[18], ArgValue::BufferMut(10));
    }

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

    /// A width the matrix arm has no `_lse` stamp for stays on the scalar
    /// kernel even though the plain arm would decline it for the same reason.
    #[test]
    fn a_width_off_the_lse_stamp_stays_scalar() {
        let probe = Probe::default();
        let (rows, q_heads) = (256u32, 2u32);
        let q = bf16(1, rows, q_heads * 128);
        let pool = KvPool {
            keys: bf16(2, 64, 128),
            values: bf16(3, 64, 128),
            page_indices: Tensor::new(4, 8, 1, dtype::Dtype::U32),
            page_indptr: Tensor::new(5, 2, 1, dtype::Dtype::U32),
            page_size: 16,
            seq_stride: 128,
            head_stride: 128,
        };
        let plan = PrefillPlan {
            positions: Tensor::new(6, rows, 1, dtype::Dtype::I32),
            request_of_token: Tensor::new(7, rows, 1, dtype::Dtype::I32),
            mask: Tensor::new(8, 1, 512, dtype::Dtype::U8),
            mask_enabled: Tensor::new(9, 1, 1, dtype::Dtype::U8),
            mask_stride: 512,
        };
        prefill_lse(
            &probe,
            RaggedTensor {
                data: q,
                indptr: Tensor::new(11, 2, 1, dtype::Dtype::I32),
            },
            &plan,
            &pool,
            None,
            128,
            1,
            0.125,
            bf16(12, rows, q_heads * 128),
            Tensor::new(10, rows, q_heads, dtype::Dtype::F32),
            1,
            &DeviceTuning::default(),
        )
        .expect("a 128-wide head has a scalar lse point");
        assert_eq!(
            probe.only().0.entrypoint,
            "sdpa_paged_tiled_lse_bfloat16_d_128"
        );
    }
}
