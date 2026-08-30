//! `quant`: the affine qmm/qmv points — entry names, instantiation stamps,
//! their grids, and the two entries a dense projection against a quantized
//! weight fires.
//!
//! # Why these names are composed and the rest of the tree's are literals
//!
//! `quant_qmm_t.metal` does not spell its affine points out. The reference
//! driver did — 216 `instantiate_qmm_t` lines, one per `(group, bits, bm,
//! bn)` — and every one of them is a template the Metal compiler expands
//! whether or not a plan will ever fire it. What ships here instead is a
//! `PIE_STAMP_qmm_t(...)` macro: the driver appends ONE invocation of it to
//! the source it is about to compile, and gets the one point it selected.
//! [`Point`] is that pair — the entry symbol and the stamp that conjures it —
//! and `engine_metal::device::Pipelines` is the specialization path that
//! consumes it, keying its library cache on the stamp because the stamp is
//! part of the source.
//!
//! The qmv side needs no stamp: `quant_qmv.metal` instantiates its six affine
//! points (three group sizes x two bit widths) in source, because there is no
//! tile axis to multiply them by. So [`qmv_name`] composes a name and hands
//! back an empty stamp, and both families read the same way at the call site.
//!
//! Entry names are composed at runtime (group size x bit width x tile), so
//! they are interned to `&'static str` — the currency of [`Fire`] and of the
//! driver's pipeline cache.
//!
//! [`Fire`]: crate::encode::Fire

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::{Bank, Tensor};

/// The two source files the points live in.
const QMM_FILE: &str = "linear/quant_qmm_t.metal";
const QMV_FILE: &str = "linear/quant_qmv.metal";

/// The macro `quant_qmm_t.metal` publishes for the epilogue-free tiled point,
/// and the one this plane selects: the IR's dense projections
/// (`linear.matmul`, `linear.lm_head`) carry no bias and no residual, so the
/// `_bias`/`_residual`/`_routed` stamps beside it stay unfired until an op
/// names one.
const QMM_STAMP: &str = "PIE_STAMP_qmm_t";

/// Group sizes the shaders instantiate.
const GROUPS: [i32; 3] = [32, 64, 128];

/// Bit widths they instantiate.
const WIDTHS: [i32; 2] = [4, 8];

/// Column tiles the qmm point is stamped at, WIDEST FIRST, because the
/// selection takes the first that divides the rectangle. Also the ROW tiles
/// of the routed families, which stop at 16 — see [`ROW_TILES`] for why the
/// dense ones do not.
const TILES: [i32; 3] = [64, 32, 16];

/// Row tiles the DENSE families are stamped at, widest first — [`TILES`] plus
/// the 8 rung.
///
/// A separate list because the two are no longer the same set and the
/// difference is a fact about what is instantiated, not a convenience.
/// `quant_qmm_t.metal` stamps 8 for the pre-cast, split and plain families
/// alone; the routed pair keeps `TILES` because `linear::moe` selects its row
/// block from its own `MOE_TILE_ROWS` and never asks for one this narrow.
/// Checking a routed name against this list would accept a tile the compiler
/// will not mint.
const ROW_TILES: [i32; 4] = [64, 32, 16, 8];

/// The threadgroup a row block of `bm` launches at — `[SIMD_SIZE, WN, WM]`,
/// which is the layout `quant_qmm_t.metal` reads `simd_gid` out of.
///
/// **IT IS NOT ONE CONSTANT ANY MORE, AND THAT IS THE 8 RUNG'S DOING.**
/// `mlx::steel::BlockMMA` gives each simdgroup a warp tile `BM / (8 * WM)`
/// rows tall, so an 8-row block admits `WM = 1` and nothing else — the
/// shader's own `qmm_wm` is where that is written down. Half the row split is
/// half the threadgroup: 64 lanes at the 8 rung, 128 above it.
///
/// **LAUNCHING THE WRONG ONE IS A WRONG ANSWER, NOT A SLOW ONE.** Both block
/// loaders divide their tile by `tgp_size` and read `load_unsafe` with no
/// bound check, so 64 lanes arriving at a point compiled for 128 leave half
/// of every staged tile whatever the threadgroup memory last held.
///
/// The threshold is `kFragSize * 2` and not a rung of [`BM_RUNGS`]: it is the
/// narrowest block two simdgroups can each take a whole fragment row of, so
/// it would still be sixteen if the rungs were renumbered tomorrow.
#[must_use]
pub fn qmm_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm < 2 * FRAG_ROWS { 1 } else { 2 }]
}

/// Rows of one `simdgroup_matrix` fragment — `mlx::steel::BaseMMAFrag`'s
/// `kFragRows`, which `static_assert`s itself to 8 and is the unit every row
/// block above is a multiple of.
const FRAG_ROWS: i32 = 8;

/// Threadgroup of the vector point: two simdgroups, four result rows each.
const QMV_GROUP: [u32; 3] = [32, 2, 1];

/// The contraction step the tiled point walks. `K % BK == 0` is one of the
/// three conditions `quant_qmm_t.metal` states for `load_unsafe`.
const QMM_BK: i32 = 32;

/// The row tiles the GEMM is stamped at, NARROWEST FIRST — the order
/// [`bm_rung`] walks.
///
/// A LIST rather than a narrow/wide pair, because the argument does not stop
/// at 32. The GEMM dequantizes a weight tile once per row block, so a batch
/// that spans several blocks pays for the same dequantize again in each: 14.6
/// ms at M=16, 24.4 at 32, 45.7 at 64, measured standalone across a
/// checkpoint's projections, which is why doubling M nearly doubles the time.
/// A taller block halves that work at the cost of halving the threadgroup
/// count, so it is worth taking only once the batch has blocks to spare — at
/// M=32, BM=32 measures 20.4 ms against BM=16's 24.4, and on a prefill the
/// third rung is worth as much as the second was (llama-1B at 1236 tok/s at
/// BM=16, 1616 at 32, 1936 at 64).
///
/// A FOURTH RUNG IS THE FINDING'S EDGE, not an omission. BM=128 was
/// instantiated and measured: Qwen3.6-27B prefills 103.0 tok/s at 128 rows
/// against 64's 104.5, and 106.5 against 106.7 at 512. A 128-row block is
/// 12.8 KiB of threadgroup memory against 64's 7.7, which takes a core from
/// four resident threadgroups to two — and overlapping one threadgroup's
/// weight read with another's MMA is the only thing hiding either. Halving
/// the dequantizations does not pay for it.
///
/// The other three axes were swept at this rung and none of them moved:
/// BK=64 is slower (1647 against 1817 on llama-1B) and illegal below gs=64,
/// since the block loader asserts `BCOLS <= group_size`; WM=4 is slower
/// (1714), because splitting a 64-row block four ways puts each lane back to
/// sixteen accumulators and that is not what the kernel is short of.
///
/// # The floor is 8, and it is the cheapest rung on the ladder
///
/// The list used to start at 16, so the ladder's bottom step ran sixteen rows
/// of arithmetic for whatever the batch actually brought — a four-lane decode
/// paid four times over. Eight is where the same template stops:
/// `mlx::steel::BlockMMA`'s warp tile is `BM / (8 * WM)` rows, so `BM = 8`
/// needs `WM = 1` (`quant_qmm_t.metal`'s `qmm_wm`, which is where that
/// constraint is written down) and `BM = 4` has nowhere left to go.
///
/// **THE RUNG IS NOT FREE AND IT PAYS ANYWAY.** Halving `WM` halves the
/// threadgroup with it — 64 lanes against 128 — so each threadgroup
/// dequantizes the same `BN x BK` weight tile over half the lanes and only
/// the ARITHMETIC halves. The GEMM arm forced at every width with
/// `qmm_min_batch = 2`, ms/fire over 32 warm decode fires through
/// `throughput_probe`, lower is better:
///
/// ```text
///          qwen36-27b        gemma4-31b      gpt-oss-20b     qwen35-0.8b
///   N     BM16    BM8      BM16    BM8      BM16   BM8      BM16   BM8
///    2   229.4  206.2     250.2  222.1      26.4  24.8      11.7  10.2
///    3   241.8  218.9     251.2  223.0      32.3  30.6      14.3  12.7
///    4   250.9  228.0     251.7  223.6      38.0  36.4      14.3  12.8
///    5   262.3  239.4     252.4  224.3      43.9  42.1      17.6  16.0
///    6   269.1  246.5     253.1  225.0      49.7  47.9      17.6  16.1
///    7   282.0  259.4     254.2  226.1      55.6  53.9      19.6  18.2
///    8   289.3  266.8     255.0  226.8      61.4  59.6      19.5  18.2
///   16   355.0  355.0     260.5  260.6      98.3  98.3      29.8  29.7
/// ```
///
/// 8 to 11% off the two giants at every width the rung reaches, 3 to 6% off
/// the mixture's dense projections and 7 to 13% off the small vehicle. The
/// sixteen-lane row is the CONTROL and it is the reason to believe the rest:
/// sixteen rows select the 16 rung under both lists, so the two columns are
/// the same launch and a difference there would have been drift rather than
/// the rung.
///
/// It is not the 57% a fire-count model predicted, and the model's error is
/// worth keeping: it priced one weight-table read as independent of `BM`,
/// which at these widths it is not — the launch has ONE row tile either way,
/// so halving the lanes per threadgroup halves the dequantize throughput
/// alongside the arithmetic and only the difference between them is left
/// over. What the rung actually moved is
/// [`crate::tuning::DeviceTuning::qmm_min_batch`], which the same sweep took
/// from six to five.
const BM_RUNGS: [i32; 4] = [8, 16, 32, 64];

/// Column tiles, narrowest first — the order [`bn`] walks to take the widest
/// that divides.
const BN_RUNGS: [i32; 3] = [16, 32, 64];

/// Threadgroups the split-K dispatch aims to land near.
///
/// MLX picks its split to reach 512 and sends every transposed non-batched
/// decode down this path; a roofline probe finds the same saturation point
/// independently. An earlier sweep here preferred 256 and was measuring a
/// split path that never dispatched its reduce — timing a kernel that
/// computed the wrong answer, so the curve it drew was not this GEMM's.
/// Re-swept on llama-1B at 32 lanes with the reduce in place: 741 tok/s at
/// 128, 873 at 256, 887 at 512, 886 at 1024, 876 at 2048. Flat from 512 on,
/// so take its near edge.
const SPLIT_TARGET_TGS: i32 = 512;

/// The column tile the split dispatch is stamped at.
const SPLIT_BN: i32 = 32;

/// Past this the partials buffer costs more than the threadgroups buy.
const SPLIT_MAX: i32 = 16;

/// The widest projection that takes the split path. A vocabulary head has
/// enough output tiles of its own to never need one, which is what keeps the
/// partials to a few MB instead of hundreds.
const SPLIT_MAX_OUT: i32 = 8192;

/// Each partition must be a whole number of `BK`-wide tiles AND whole
/// quantization groups, or it reads into the next group's scales. 64 is the
/// widest group the points are stamped for and a multiple of [`QMM_BK`].
const SPLIT_K_ALIGN: i32 = 64;

/// Every point this plane may fire, ready to compile — the driver's warm-up
/// census and the gate that the stamps still name entries the compiler will
/// mint.
///
/// [`Fire`]s and not names, because a qmm point IS its stamp: handed the
/// entry alone the driver compiles the shipped source, finds no such symbol,
/// and refuses a point that is in fact reachable.
#[must_use]
pub fn composed() -> Vec<Fire> {
    let mut out = Vec::new();
    for &gs in &GROUPS {
        for &b in &WIDTHS {
            let point = qmv_point("quant.qmv", "fast", gs, b).expect("an axis point");
            out.push(Fire::at(QMV_FILE, point.entry));
            for &bm in &ROW_TILES {
                for &bn in &TILES {
                    let point = qmm_point("quant.qmm_t", "", QMM_STAMP, gs, b, bm, bn)
                        .expect("an axis point, by construction");
                    out.push(Fire::at(QMM_FILE, point.entry).stamp(point.stamp));
                }
            }
        }
    }
    out
}

/// One selected qmm instantiation: the entry symbol and the jit stamp that
/// conjures it when the shader source does not spell it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Point {
    pub entry: &'static str,

    pub stamp: &'static str,
}

pub fn qmm_point(
    op: &'static str,
    form: &str,
    stamp: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<Point, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    let entry = symbol(&format!(
        "affine_qmm_t{form}_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    ));
    Ok(Point {
        entry,
        stamp: if stamp.is_empty() {
            ""
        } else {
            symbol(&format!(
                "{stamp}(\"{entry}\", {group}, {bits}, {bm}, {QMM_BK}, {bn})"
            ))
        },
    })
}

pub fn qmm_name(
    op: &'static str,
    form: &str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    Ok(qmm_point(op, form, "", group, bits, bm, bn)?.entry)
}

pub fn qmm_precast_name(
    op: &'static str,
    before: &str,
    after: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t{before}_fp16_precast{after}_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

/// One qmv instantiation. The stamp is always empty: `quant_qmv.metal`
/// spells its six affine points out, so there is nothing to conjure.
pub fn qmv_point(op: &'static str, form: &str, group: i32, bits: i32) -> Result<Point, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    Ok(Point {
        entry: symbol(&format!("affine_qmv_{form}_bfloat16_gs_{group}_b_{bits}")),
        stamp: "",
    })
}

pub fn qmv_name(
    op: &'static str,
    form: &str,
    group: i32,
    bits: i32,
) -> Result<&'static str, Error> {
    Ok(qmv_point(op, form, group, bits)?.entry)
}

fn check(
    op: &'static str,
    points: &[i32],
    v: i32,
    what: &'static str,
) -> Result<(), Error> {
    points
        .contains(&v)
        .then_some(())
        .ok_or_else(|| refuse(op, format!("no point is stamped at {what} {v}")))
}

/// Which row rung a batch of `rows` should use: the widest the batch can
/// COVER, not the widest that divides it.
///
/// A block only pays for itself once the batch can fill it, and the caller
/// pads the grid up to the rung it gets back — which is why the row count to
/// pad to must be asked of this function and not of the widest rung. Padding
/// a one-row decode to a 64-row block would launch sixty-four rows of
/// arithmetic to compute one.
#[must_use]
pub fn bm_rung(rows: i32) -> i32 {
    let mut best = BM_RUNGS[0];
    for &rung in &BM_RUNGS[1..] {
        if rows >= rung {
            best = rung;
        }
    }
    best
}

/// Rows the batched fire launches its GEMM over, for a fire of `rows`.
///
/// **THE KERNEL TAKES NO `M`.** It is written for full tiles only, so a
/// driver may select it only when `M % BM == 0`, and the row count reaches it
/// through the grid. Handing it the raw fire width therefore makes the GEMM
/// reachable AT EXACT MULTIPLES OF A RUNG AND NOWHERE ELSE, which for a
/// decode is almost never: measured on Qwen3.6-27B, a device affording 24
/// recurrent slots ran 75.6 tok/s at 16 lanes and 30–32 at 2, 4, 6, 8, 12, 20
/// and 24 — a flat curve with one spike, because 16 was the only width that
/// divided a rung. Six times the lanes bought nothing.
///
/// So pad the fire up to its rung. The padding is free of consequence when
/// the rows land in slots the fire does not read: a GEMM row's output depends
/// only on its own input row, so garbage in the tail cannot reach a real one.
/// `capacity` is what makes that true and is the caller's to state — the rows
/// its activation and result rectangles actually hold, not the rows this fire
/// uses.
///
/// Two guards, both falling back to the unpadded width (and so to the matvec,
/// since it will not divide a rung):
///
///   * `rows < min_batch` — padding must not be able to talk the dispatch
///     past the measured crossover. A 2-row fire padded to 16 would launch
///     eight times the arithmetic it needs.
///   * `padded > capacity` — a wider write would run into the next
///     activation's slot. **THE FLOOR THIS IMPOSES MOVED WITH THE 8 RUNG.**
///     It used to be sixteen: any fire at or above the crossover needed a
///     slot sixteen rows deep or it fell straight back to the vector point.
///     `bm_rung` now answers 8 below sixteen rows, so a 2-15 row fire needs
///     EIGHT and a slot of 8-15 rows — which used to decline every time —
///     reaches the GEMM.
#[must_use]
pub fn mb_rows(rows: i32, capacity: i32, min_batch: i32) -> i32 {
    let rows = rows.max(1);
    if rows < min_batch {
        return rows;
    }
    let bm = bm_rung(rows);
    let padded = ((rows + bm - 1) / bm) * bm;
    if padded <= capacity.max(1) { padded } else { rows }
}

/// The column tile for a GEMM that has split-K behind it: the WIDEST that
/// divides the output, full stop.
///
/// This used to gate on a threadgroup count, and that was right when the GEMM
/// had nothing else supplying parallelism. Split-K changed the premise — the
/// split now supplies the threadgroups, so the only thing BN still decides is
/// how many times each weight tile is dequantized, and wider is strictly
/// fewer. Interleaved A/B on a decode step, widest against the old
/// 192-threadgroup rule: 16 lanes 31.57 ms against 37.02, 32 lanes 141.18
/// against 158.45. The old rule is a pessimization once the split exists.
///
/// BN partitions output columns only — every element's K sum is unchanged —
/// so the choice is bit-exact whichever way it goes.
#[must_use]
pub fn bn(out_width: i32) -> Option<i32> {
    BN_RUNGS
        .iter()
        .rev()
        .copied()
        .find(|rung| out_width % rung == 0)
}

/// [`bn`] for a family whose GEMM has NO split-K behind it.
///
/// The rule above is correct *because* the split supplies threadgroups when
/// the output tiles do not. A family that dispatches no split has no such
/// supply, and taking the widest tile then starves the machine: at M=128 with
/// BM=64 a projection to 1024 columns gets 32 threadgroups, which the curve
/// prices at a third of what the same work does at 200.
///
/// So: the narrow tile until there is enough work to fill the machine, and 32
/// after that. NEVER 64 — that is the finding, not an omission. Sixteen (M,N)
/// pairs swept at BM=64 and BN=64 is the best of none of them; forcing each
/// width through a real llama-3.2-1B prefill agrees, 2565.8 / 2663.7 / 2578.3
/// tok/s at BN 16/32/64 over 448 rows and 2270.8 / 2349.8 / 2297.0 over 1024.
/// The threshold sits in the only gap the sweep leaves — 144 threadgroups
/// still wants 16, 192 already wants 32 — and belongs to the MACHINE, so it
/// is read from [`DeviceTuning`] and not written here.
///
/// [`DeviceTuning`]: crate::tuning::DeviceTuning
#[must_use]
pub fn bn_unsplit(out_width: i32, row_tiles: i32, crossover_tg: i32) -> Option<i32> {
    if out_width % BN_RUNGS[0] != 0 {
        return None;
    }
    if out_width % 32 == 0 && (out_width / 32) * row_tiles.max(1) >= crossover_tg {
        return Some(32);
    }
    Some(BN_RUNGS[0])
}

/// How deep to split the contraction when the output tiles alone leave the
/// machine short, or 1 for no split.
///
/// Counts the tiles the SPLIT dispatch will actually launch — [`SPLIT_BN`]
/// wide and `bm` tall. It used to count rows in units of the NARROWEST block,
/// on the theory that a wide block is twice as parallel and should be split
/// half as deep. That is backwards: a wide block covers twice the rows in ONE
/// threadgroup, so it produces half the tiles and needs MORE split, not less
/// — and the numbers that appeared to support it were measured on a split
/// path that never dispatched its reduce. Counting honestly is worth
/// 741 → 870 tok/s at 32 lanes on llama-1B.
#[must_use]
pub fn split_k(out_width: i32, rows: i32, contraction: i32, bm: i32) -> i32 {
    if out_width % SPLIT_BN != 0 || out_width > SPLIT_MAX_OUT || bm <= 0 {
        return 1;
    }
    let tiles = (out_width / SPLIT_BN) * ((rows + bm - 1) / bm);
    if tiles <= 0 {
        return 1;
    }
    let mut split = (SPLIT_TARGET_TGS / tiles)
        .min(SPLIT_MAX)
        .min(contraction / SPLIT_K_ALIGN);
    while split > 1 && contraction % (split * SPLIT_K_ALIGN) != 0 {
        split -= 1;
    }
    if split < 2 { 1 } else { split }
}

/// The staged-input point: the FP16 pre-cast GEMM, and the largest single win
/// recorded on pre-Apple9 silicon — roughly 40% on the GEMM at every shape
/// measured, 938 → 1298 tok/s of gemma-4 prefill.
///
/// M1 and M2 have no native bfloat16 matrix path and emulate it, so this
/// stages the activation to FP16 once (through [`precast_stage`]) and feeds
/// the instruction the hardware has. Stamped in source at gs=64/b=4 alone,
/// because that is the only format whose weight loader dequantizes straight
/// to `half`.
pub fn precast_point(op: &'static str, form: &str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t{form}_fp16_precast_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

/// The routed tiled point whose weight loader dequantizes straight to
/// `half` — the mixture's half of the FP16 matrix path, stamped in source at
/// gs=64/b=4 alone.
pub fn routed_fp16_point(op: &'static str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    check(op, &TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

/// The routed tiled point for an mxfp4 bank. Only the biased form is stamped,
/// because the one family that ships mxfp4 biases every routed projection.
pub fn mxfp4_routed_point(op: &'static str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    check(op, &TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "mxfp4_qmm_t_routed_bias_bfloat16_bm_{bm}_bn_{bn}"
    )))
}

/// The staging pass the precast points read: one thread per element of the
/// activation, bf16 in and `half` out.
///
/// One dispatch, one barrier and one buffer for the whole layer, against a
/// cast the GEMM would otherwise redo once per weight tile — sixteen times
/// for a hidden projection, 128 for gate/up.
pub const PRECAST_STAGE: &str = "cast_qmm_input_bfloat16_to_float16";

/// The staging pass's geometry over a `rows x contraction` activation.
pub fn precast_stage(op: &'static str, rows: i32, contraction: i32) -> Result<Grid, Error> {
    let count = rows
        .checked_mul(contraction)
        .filter(|n| *n > 0)
        .ok_or_else(|| refuse(op, format!("{rows} x {contraction} will not stage")))?;
    Ok(Grid::of([count.unsigned_abs(), 1, 1], [256, 1, 1]))
}

/// The split-K point, and its partials dtype. `f32` partials are the accurate
/// arm; `bfloat16` is the one that keeps the partials buffer to half the
/// bytes.
pub fn splitk_point(
    op: &'static str,
    partials: &str,
    group: i32,
    bits: i32,
    bm: i32,
) -> Result<&'static str, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &ROW_TILES, bm, "row tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_splitk_{partials}_gs_{group}_b_{bits}_bm_{bm}_bn_{SPLIT_BN}"
    )))
}

/// The fold that turns the split's partials back into one result.
///
/// **DISPATCHING THE SPLIT WITHOUT THIS IS NOT A SLOW ANSWER, IT IS A WRONG
/// ONE**, and the reference spent a sweep discovering that: the curve that
/// preferred a 256-threadgroup target was drawn by timing a kernel whose
/// partials were never summed.
pub fn splitk_reduce_point(partials: &str) -> &'static str {
    match partials {
        "f32_bfloat16" => "qmm_splitk_reduce_f32_bfloat16",
        _ => "qmm_splitk_reduce_bfloat16",
    }
}

/// The split dispatch's grid: `out/BN` column tiles, `ceil(M/BM)` row tiles,
/// and one z-plane per split, at [`qmm_group`]'s threadgroup for the row
/// block — `32x2x2 = 128` threads from the 16 rung up and `32x2x1 = 64` at
/// the 8.
///
/// The z axis carries BOTH the split index and the `WM` lanes, which is why
/// its multiplier is the group's own z extent and not a literal 2: the shader
/// reads `tid.z` as the partition and `simd_gid` as the row split, and a grid
/// that doubled z at a point compiled for one row simdgroup would run every
/// partition twice.
pub fn splitk_grid(op: &'static str, out_width: i32, rows: i32, bm: i32, split: i32) -> Result<Grid, Error> {
    if out_width <= 0 || rows <= 0 || bm <= 0 || split <= 0 {
        return Err(refuse(op, "the split dispatch has a zero extent"));
    }
    let group = qmm_group(bm);
    Ok(Grid::of(
        [
            32 * (out_width.unsigned_abs() / SPLIT_BN.unsigned_abs()),
            group[1] * rows.unsigned_abs().div_ceil(bm.unsigned_abs()),
            group[2] * split.unsigned_abs(),
        ],
        group,
    ))
}

/// The reduce's grid: one thread per element of the result.
pub fn splitk_reduce_grid(op: &'static str, out_width: i32, rows: i32) -> Result<Grid, Error> {
    if out_width <= 0 || rows <= 0 {
        return Err(refuse(op, "the reduce has a zero extent"));
    }
    Ok(Grid::of(
        [out_width.unsigned_abs(), rows.unsigned_abs(), 1],
        [256, 1, 1],
    ))
}

pub fn qmm_grid(
    op: &'static str,
    n: i32,
    bn: i32,
    m: i32,
    bm: i32,
    split_k: i32,
) -> Result<[u32; 3], Error> {
    if n <= 0 {
        return Err(refuse(op, "the column count is zero"));
    }
    if m <= 0 {
        return Err(refuse(op, "the row count is zero"));
    }
    if bn <= 0 || bm <= 0 {
        return Err(refuse(op, "the tile is zero"));
    }
    if split_k <= 0 {
        return Err(refuse(op, "the k split is zero"));
    }
    if m % bm != 0 {
        return Err(refuse(
            op,
            format!(
                "the row count is {m}, not a multiple of {bm}: the tile must \
                 divide it because no entrypoint takes m and the shader reads \
                 it from the grid"
            ),
        ));
    }
    if n % bn != 0 {
        return Err(refuse(
            op,
            format!(
                "the column count is {n}, not a multiple of {bn}: `quant_qmm_t.metal` \
                 states `M % BM == 0, N % BN == 0 and K % BK == 0` as the \
                 condition under which the driver may select it at all, and \
                 `load_unsafe` is the only path its hot loop takes"
            ),
        ));
    }
    let lanes = |groups: u32, local: u32, what: &'static str| -> Result<u32, Error> {
        groups
            .checked_mul(local)
            .ok_or_else(|| refuse(op, format!("{what} will not launch at {groups} groups")))
    };
    // The three multipliers are the THREADGROUP's extents and not constants —
    // see [`qmm_group`], whose z extent is 1 at the 8 rung because a row
    // block that narrow admits one simdgroup down M.
    let group = qmm_group(bm);
    Ok([
        lanes(
            n.unsigned_abs().div_ceil(bn.unsigned_abs()),
            group[0],
            "the column tiles",
        )?,
        lanes(m.unsigned_abs() / bm.unsigned_abs(), group[1], "the row tiles")?,
        lanes(split_k.unsigned_abs(), group[2], "the k splits")?,
    ])
}

pub fn qmv_grid(op: &'static str, vecs: i32, out_vec_size: i32) -> Result<[u32; 3], Error> {
    if vecs <= 0 {
        return Err(refuse(op, "the vectors are zero"));
    }
    if out_vec_size <= 0 {
        return Err(refuse(op, "the output vector is zero"));
    }
    let x = vecs
        .unsigned_abs()
        .checked_mul(32)
        .ok_or_else(|| refuse(op, format!("{vecs} vectors will not launch")))?;
    Ok([x, out_vec_size.unsigned_abs().div_ceil(4), 1])
}

/// Interns a composed name, so a runtime-built entry can live in a
/// `&'static str` field. Names are few (the axis grid above) and re-selected
/// every fire, so the leak is bounded and the map earns its keep.
fn symbol(name: &str) -> &'static str {
    static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = INTERNED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), leaked);
    leaked
}

/// The two working planes [`act_x_wt`]'s fast rungs need, seated as MINTS
/// rather than as rectangles.
///
/// **THE SHAPE IS A FIRE-TIME SELECTION, WHICH IS WHY THIS IS NOT A PAIR OF
/// `Option<Tensor>`.** How many rows the staging covers is [`mb_rows`]'s
/// answer and how deep the partials go is [`split_k`]'s, and both are reached
/// several guards inside the entry. What holds the two planes is ONE
/// load-time reservation in the shell, so the shell is the only thing that
/// can say whether a given rectangle is inside it — and the way it says so is
/// `None`, which every rung here answers by falling to the rung that needs no
/// plane. A rectangle handed over before the selection ran would be the
/// caller guessing at the answer this entry exists to give.
///
/// Erased behind `dyn` for [`Ctx`]'s reason: this crate names no driver type.
pub struct Scratch<'a> {
    /// `rows x contraction` halves — what [`PRECAST_STAGE`] writes and the
    /// [`precast_point`] GEMM reads.
    pub precast: &'a dyn Fn(u32, u32) -> Option<Tensor>,

    /// `split * rows x width` — what the [`splitk_point`] GEMM writes and
    /// [`splitk_reduce_point`] folds back.
    ///
    /// **THE PLANE'S DTYPE NAMES THE POINT**, and is read off it rather than
    /// chosen here: `f32` partials are the accurate arm and `bfloat16` the
    /// one that costs half the bytes, which is a question about the
    /// RESERVATION and is answered by whoever sized it.
    pub partials: &'a dyn Fn(u32, u32, u32) -> Option<Tensor>,
}

/// `y = act x w^T` where `w` is a quantized bank — the quantized twin of
/// [`gemm::matmul`](crate::linear::gemm::matmul), selected by the driver on
/// the weight row's form and not on anything in the op.
pub fn matmul(
    ctx: &Ctx<'_>,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y, scratch, capacity_rows)
}

/// The same product at the vocabulary head.
///
/// **THE SAME LADDER, AND THE HEAD DECLINES TWO OF ITS RUNGS BY ARITHMETIC.**
/// [`split_k`] refuses any output past [`SPLIT_MAX_OUT`], so a vocabulary
/// projection never reaches the split — that refusal is what keeps the
/// partials plane a few MB instead of a function of the vocabulary. The row
/// rung and the pre-cast are NOT declined: a head is the widest GEMM in the
/// stack and stages its activation exactly as the projections do.
pub fn lm_head(
    ctx: &Ctx<'_>,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y, scratch, capacity_rows)
}

/// `y = act x w^T` against a bank, tiled where the rectangle allows and
/// vectorized where it does not.
///
/// **THE TILED POINT IS NOT A FALLBACK-FREE CHOICE, AND THAT IS THE WHOLE
/// SELECTION.** `quant_qmm_t.metal`'s hot loop is `load_unsafe` on both
/// operands — it reads a whole `BM x BK` block with no edge predicate — so
/// the shader's own header states `M % BM == 0, N % BN == 0, K % BK == 0` as
/// the condition under which a driver may select it AT ALL. The dense gemm
/// beside this one tiles with `div_ceil` because its tile kernel guards its
/// own edges; this one cannot, so a rectangle whose rows or columns no
/// stamped tile divides takes the vector point, which guards every one of
/// its own reads. That is a correctness rule wearing a performance rule's
/// clothes, and the ragged case is common: a prefill's row count is however
/// many tokens the composition happened to carry.
///
/// # The ladder, three rungs and a floor
///
/// Each rung is a measured helper above and this entry only walks them in
/// order; the numbers live on the helpers, so nothing is restated here.
///
///   1. **[`mb_rows`]** pads the launch up to the row rung the batch can
///      cover, bounded by `capacity_rows`. It is not an arm — it is what
///      makes the three arms below REACHABLE at a batch that is not already
///      an exact multiple of a rung, which for a decode is almost never.
///   2. **[`precast_point`]**, when the machine emulates bfloat and the bank
///      is the one format the staged loader is stamped for. The largest
///      single win recorded, ~40% on the GEMM at every shape measured.
///   3. **[`splitk_point`] + [`splitk_reduce_point`]**, when the output tiles
///      alone leave the machine short. 741 → 887 tok/s on llama-1B at 32
///      lanes.
///   4. The plain stamped point at [`bn_unsplit`]'s column tile, and below
///      the crossover the vector point.
///
/// **RUNGS 2 AND 3 DO NOT COMPOSE, AND THE ORDER IS WHAT MAKES THAT
/// STRUCTURAL** rather than a rule someone has to remember. The split arm is
/// only reachable when the pre-cast arm declined, because the two planes are
/// ONE reservation: `engine_metal::scratch` colors the staging rectangle and
/// the partials rectangle onto the same bytes, on the stated ground that no
/// dense projection is ever inside both. A chain that staged its activation
/// and then wrote partials would overwrite the very halves the GEMM was
/// about to read — a wrong answer, not a slow one.
#[allow(clippy::too_many_arguments)]
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    // A DENSE PROJECTION AGAINST AN MXFP4 BANK IS NOT STAMPED, and answering
    // it with the affine points would read an e8m0 exponent byte as half a
    // bf16 factor. The mxfp4 codec is instantiated for the ROUTED shapes
    // alone (`linear::moe`), because the one family that ships mxfp4 keeps it
    // in the expert banks and its projections in bf16.
    let Some(biases) = w.biases else {
        return Err(refuse(
            op,
            format!(
                "the weight is a symmetric {}-bit bank in groups of {}, and this plane \
                 stamps no dense point for one: `quant_qmv.metal` instantiates the \
                 mxfp4 codec only at the routed shapes",
                w.bits, w.group
            ),
        ));
    };
    let (rows, columns, contraction) = extent(op, act, y)?;
    if rows == 0 {
        return Ok(());
    }
    if contraction % w.group != 0 {
        return Err(refuse(
            op,
            format!(
                "the contraction is {contraction}, not a whole number of {}-code \
                 groups: every point indexes its scales as `k / group`, so a \
                 partial group reads the next row's factor",
                w.group
            ),
        ));
    }
    let group = stated(op, w.group)?;
    let bits = stated(op, w.bits)?;
    let (m, n, k) = (
        stated(op, rows)?,
        stated(op, columns)?,
        stated(op, contraction)?,
    );
    // **THE CROSSOVER IS THE MACHINE'S AND THE FORMAT'S, NOT A CONSTANT.**
    // Which side of it a batch falls on moved by 40% on this very GEMM when
    // the FP16 matrix path arrived, and it moves again per Apple family —
    // hence [`crate::tuning`] rather than a number here.
    //
    // The DENSE arm, always: whether the checkpoint's FFN is routed is a fact
    // about the MODEL, and no operand of `linear.matmul` carries it. On the
    // M1 Max the two agree at 8 and only Apple8 separates them, so the
    // difference this cannot see is one family's mixtures' attention
    // projections. Naming it needs a fact at this seam that the IR does not
    // state.
    let tuned = crate::tuning::current();
    let fp16 = tuned.fp16_gemm_format(w.bits, w.group);
    let min_batch = i32::try_from(tuned.qmm_min_batch(false, fp16)).unwrap_or(i32::MAX);
    let crossover = i32::try_from(tuned.qmm_bn_crossover_tg).unwrap_or(i32::MAX);
    let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
    // **RUNG 1, AND IT IS NOT AN ARM.** The rung is chosen off the fire's OWN
    // width — [`bm_rung`] answers the widest block the batch can cover, not
    // the widest that divides the padded result — and [`mb_rows`] is what
    // pads up to it, or declines to when the batch is under the crossover or
    // the slot has no rows to spare. A decline leaves `padded == m`, which no
    // rung need divide, so the tile falls back to the widest that DOES and
    // otherwise to the vector point below.
    let rung = bm_rung(m);
    let padded = mb_rows(m, capacity, min_batch);
    let block = if padded % rung == 0 {
        Some(rung)
    } else {
        tile(padded)
    };
    if m >= min_batch
        && k % QMM_BK == 0
        && let Some(bm) = block
    {
        // **RUNG 2: the staged input.** Two dispatches and one plane — the
        // cast writes `padded x k` halves, the GEMM reads them at buffer 12
        // and leaves the bf16 activation seat null. The padding is what makes
        // the staging cover rows the fire does not use, and it is the same
        // guarantee `mb_rows` states: those rows are inside the activation
        // slot's own capacity, so the cast reads garbage rather than somebody
        // else's operand and the GEMM lands the product of it in slots
        // nothing reads.
        if fp16
            && let Some(staged) = (scratch.precast)(padded.unsigned_abs(), k.unsigned_abs())
            && let Some(bn) = bn_unsplit(n, padded / bm, crossover)
        {
            let count = padded
                .checked_mul(k)
                .ok_or_else(|| refuse(op, format!("{padded} x {k} will not stage")))?;
            let mut cast = vec![ctx.absent()?; 3];
            cast.push(act.arg());
            for _ in 4..12 {
                cast.push(ctx.absent()?);
            }
            cast.push(staged.arg_mut());
            cast.push(count.arg());
            ctx.fire(
                Fire::at(QMM_FILE, PRECAST_STAGE).apply(precast_stage(op, padded, k)?),
                &cast,
            )?;
            let mut gemm = vec![
                w.codes.arg(),
                w.scales.arg(),
                biases.arg(),
                // The bf16 activation seat: the staged point does not read
                // it, and an argument binds at its own index, so the gap is
                // stated rather than closed up.
                ctx.absent()?,
                y.arg_mut(),
                k.arg(),
                n.arg(),
            ];
            for _ in 7..12 {
                gemm.push(ctx.absent()?);
            }
            gemm.push(staged.arg());
            return ctx.fire(
                Fire::at(QMM_FILE, precast_point(op, "", bm, bn)?)
                    .apply(Grid::of(qmm_grid(op, n, bn, padded, bm, 1)?, qmm_group(bm))),
                &gemm,
            );
        }
        // **RUNG 3: the split contraction.** Reachable only because the
        // pre-cast arm declined — see the entry's header for why that is the
        // structure and not a convention. The split's own column tile is
        // fixed by what is stamped, so [`bn`] does not appear here; what the
        // grid needs is the partition width and the stride from one split's
        // plane of partials to the next.
        let split = split_k(n, padded, k, bm);
        if split > 1
            && let Some(plane) = (scratch.partials)(
                split.unsigned_abs(),
                padded.unsigned_abs(),
                n.unsigned_abs(),
            )
        {
            let stride = padded
                .checked_mul(n)
                .ok_or_else(|| refuse(op, format!("{padded} x {n} partials will not launch")))?;
            let form = dtype_dispatch!(op, plane.dtype, {
                F32 => "f32_bfloat16",
                Bf16 => "bfloat16",
            });
            ctx.fire(
                Fire::at(QMM_FILE, splitk_point(op, form, group, bits, bm)?)
                    .apply(splitk_grid(op, n, padded, bm, split)?),
                &[
                    w.codes.arg(),
                    w.scales.arg(),
                    biases.arg(),
                    act.arg(),
                    // The result seat: a split lands partials at 8 and
                    // nothing at 4 until the fold below.
                    ctx.absent()?,
                    k.arg(),
                    n.arg(),
                    ctx.absent()?,
                    plane.arg_mut(),
                    (k / split).arg(),
                    stride.arg(),
                ],
            )?;
            // **THE FOLD IS NOT OPTIONAL** — see [`splitk_reduce_point`]. It
            // walks the fire's OWN rows and not the padded ones: the rows the
            // pad added hold a product of garbage, and folding them would
            // write it where a later op could read it.
            return ctx.fire(
                Fire::at(QMM_FILE, splitk_reduce_point(form)).apply(splitk_reduce_grid(op, n, m)?),
                &[
                    ctx.absent()?,
                    ctx.absent()?,
                    ctx.absent()?,
                    ctx.absent()?,
                    y.arg_mut(),
                    ctx.absent()?,
                    n.arg(),
                    ctx.absent()?,
                    plane.arg(),
                    ctx.absent()?,
                    stride.arg(),
                    split.arg(),
                ],
            );
        }
        // **RUNG 4: the plain stamped point**, at the column tile measured
        // for a family with no split behind it ([`bn_unsplit`]) — which is
        // what this arm is, having just declined one.
        if let Some(bn) = bn_unsplit(n, padded / bm, crossover) {
            let point = qmm_point(op, "", QMM_STAMP, group, bits, bm, bn)?;
            return ctx.fire(
                Fire::at(QMM_FILE, point.entry)
                    .stamp(point.stamp)
                    .apply(Grid::of(qmm_grid(op, n, bn, padded, bm, 1)?, qmm_group(bm))),
                &[
                    w.codes.arg(),
                    w.scales.arg(),
                    biases.arg(),
                    act.arg(),
                    y.arg_mut(),
                    k.arg(),
                    n.arg(),
                ],
            );
        }
    }
    let point = qmv_point(op, "fast", group, bits)?;
    ctx.fire(
        Fire::at(QMV_FILE, point.entry).apply(Grid::of(qmv_grid(op, m, n)?, QMV_GROUP)),
        &[
            w.codes.arg(),
            w.scales.arg(),
            biases.arg(),
            act.arg(),
            y.arg_mut(),
            k.arg(),
            n.arg(),
        ],
    )
}

/// The widest stamped ROW tile that divides `extent`, or `None` when none
/// does. [`ROW_TILES`] and not [`TILES`], because the one caller is choosing
/// a row block for a launch `mb_rows` declined to pad — and the 8 rung is
/// what lets an odd multiple of eight take the GEMM at all instead of falling
/// to the vector point.
fn tile(extent: i32) -> Option<i32> {
    ROW_TILES.iter().copied().find(|t| extent % t == 0)
}

/// The three extents, of which exactly one may be zero.
///
/// The dense gemm's rule, restated because it is the same rule and its own
/// statement is private to that entry: the two WIDTHS are the weight's, fixed
/// by the checkpoint, so a zero in either is a malformed row and is refused;
/// the ROWS are the composition's, and a guarded region that composed to none
/// is a fire with nothing to do rather than a fault.
fn extent(op: &'static str, act: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    Ok((y.rows, y.width, act.width))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rung_is_the_widest_a_batch_can_cover() {
        assert_eq!(bm_rung(1), 8);
        assert_eq!(bm_rung(8), 8);
        assert_eq!(bm_rung(15), 8);
        assert_eq!(bm_rung(16), 16);
        assert_eq!(bm_rung(31), 16);
        assert_eq!(bm_rung(32), 32);
        assert_eq!(bm_rung(63), 32);
        assert_eq!(bm_rung(64), 64);
        assert_eq!(bm_rung(2048), 64);
    }

    #[test]
    fn padding_never_talks_a_fire_past_its_crossover() {
        // Below the crossover the fire keeps its own width, which will not
        // divide a rung, which is how it reaches the matvec.
        assert_eq!(mb_rows(2, 4096, 8), 2);
        assert_eq!(mb_rows(7, 4096, 8), 7);
        // At and above it, up to the rung the batch can cover — and the
        // rung a batch of eight to fifteen covers is the 8.
        assert_eq!(mb_rows(8, 4096, 8), 8);
        assert_eq!(mb_rows(15, 4096, 8), 16);
        assert_eq!(mb_rows(20, 4096, 8), 32);
        assert_eq!(mb_rows(65, 4096, 8), 128);
        // And never past the rows the caller says it holds.
        assert_eq!(mb_rows(20, 24, 8), 20);
        // The capacity floor the 8 rung moved: a three-row fire at a
        // crossover of two pads to eight, which a slot of eight holds and a
        // slot of seven does not.
        assert_eq!(mb_rows(3, 8, 2), 8);
        assert_eq!(mb_rows(3, 7, 2), 3);
    }

    #[test]
    fn the_eight_rung_launches_one_simdgroup_down_m() {
        // `BlockMMA`'s warp tile is `BM / (8 * WM)` rows, so the 8 rung is
        // `WM = 1` and 64 lanes; every rung above it is `WM = 2` and 128.
        assert_eq!(qmm_group(8), [32, 2, 1]);
        assert_eq!(qmm_group(16), [32, 2, 2]);
        assert_eq!(qmm_group(64), [32, 2, 2]);
        // And the grid's z multiplier follows it, or every split would run
        // twice.
        let narrow = qmm_grid("t", 128, 32, 8, 8, 4).unwrap();
        let wide = qmm_grid("t", 128, 32, 16, 16, 4).unwrap();
        // Same column tiles and same one row tile; the z axis is where the
        // rungs part — four splits at one `WM` lane against four at two.
        assert_eq!(narrow, [4 * 32, 2, 4]);
        assert_eq!(wide, [4 * 32, 2, 8]);
    }

    #[test]
    fn the_unsplit_column_tile_is_sixteen_until_the_machine_is_full() {
        // 144 threadgroups still wants 16, 192 already wants 32 — the M1
        // Max's 160 sits in that gap.
        assert_eq!(bn_unsplit(1024, 4, 160), Some(16));
        assert_eq!(bn_unsplit(2048, 4, 160), Some(32));
        // Never 64, at any width or any occupancy.
        assert_eq!(bn_unsplit(6144, 64, 160), Some(32));
        // A width no tile divides takes no tiled point at all.
        assert_eq!(bn_unsplit(1000, 8, 160), None);
    }

    #[test]
    fn the_split_arm_takes_the_widest_tile_that_divides() {
        assert_eq!(bn(1024), Some(64));
        assert_eq!(bn(3584), Some(64));
        assert_eq!(bn(48), Some(16));
        assert_eq!(bn(1000), None);
    }

    #[test]
    fn a_split_is_whole_groups_capped_and_off_at_the_head() {
        // A projection to hidden: 32 column tiles at one row tile, so the
        // target divides to 16 and the cap holds it there.
        assert_eq!(split_k(1024, 16, 2048, 16), 16);
        // gate/up: 112 column tiles leave room for 4.
        assert_eq!(split_k(3584, 16, 2048, 16), 4);
        // A vocabulary head has tiles of its own and is over the width cap.
        assert_eq!(split_k(151936, 16, 2048, 16), 1);
        // A wide row block produces HALF the tiles and so needs more split,
        // which is the correction worth 741 -> 870 tok/s.
        assert!(split_k(1024, 64, 2048, 64) >= split_k(1024, 64, 2048, 16));
        // Every partition a whole number of 64-code groups.
        let k = 2048;
        let s = split_k(1024, 16, k, 16);
        assert_eq!(k % (s * 64), 0);
    }

    #[test]
    fn the_precast_points_are_stamped_at_g64_b4_alone() {
        assert_eq!(
            precast_point("t", "", 32, 32).unwrap(),
            "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_32_bn_32"
        );
        assert_eq!(
            precast_point("t", "_bias", 16, 64).unwrap(),
            "affine_qmm_t_bias_fp16_precast_bfloat16_gs_64_b_4_bm_16_bn_64"
        );
        assert!(precast_point("t", "", 128, 32).is_err());
        // The 8 rung, stamped in source for the pre-cast family.
        assert_eq!(
            precast_point("t", "", 8, 16).unwrap(),
            "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_8_bn_16"
        );
        // Eight is a ROW tile and not a column one: no point is stamped 8
        // wide, and `bn`/`bn_unsplit` never answer it.
        assert!(precast_point("t", "", 16, 8).is_err());
    }

    #[test]
    fn the_split_points_name_their_partials() {
        assert_eq!(
            splitk_point("t", "bfloat16", 64, 4, 16).unwrap(),
            "affine_qmm_t_splitk_bfloat16_gs_64_b_4_bm_16_bn_32"
        );
        assert_eq!(
            splitk_point("t", "f32_bfloat16", 64, 4, 32).unwrap(),
            "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_32_bn_32"
        );
        assert_eq!(
            splitk_point("t", "f32_bfloat16", 64, 4, 8).unwrap(),
            "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_8_bn_32"
        );
        assert_eq!(splitk_reduce_point("f32_bfloat16"), "qmm_splitk_reduce_f32_bfloat16");
    }
}
