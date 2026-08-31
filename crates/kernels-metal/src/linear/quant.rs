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

/// The three source files the points live in.
const QMM_FILE: &str = "linear/quant_qmm_t.metal";
const QMV_FILE: &str = "linear/quant_qmv.metal";

/// The MULTI-ROW vector point, which shares one weight read across R rows.
/// Its own file for the reason its header states: the one-row file spells
/// thirty-odd points out in source, and this one is stamped.
const QMV_ROWS_FILE: &str = "linear/quant_qmv_rows.metal";

/// The macro `quant_qmv_rows.metal` publishes.
const QMV_ROWS_STAMP: &str = "PIE_STAMP_qmv_rows";

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

/// **THE ROW GROUPS THE VECTOR POINT MAY BE FOLDED AT, NARROWEST FIRST.**
///
/// `quant_qmv_rows.metal` reads a weight block once and applies it to `R`
/// activation rows, so a fire of `m` rows costs `ceil(m / R)` reads of the
/// bank instead of `m`. The rungs are powers of two because the fold's only
/// cost is registers and the register budget halves that way; they stop at
/// eight because that is where the tiled point's own row block begins and a
/// band both arms can serve wants measuring, not guessing.
///
/// [`qmv_rows_fold`] is the walk, and
/// [`crate::tuning::DeviceTuning::qmv_rows_max`] is the ceiling it stops at.
const QMV_ROW_RUNGS: [i32; 3] = [2, 4, 8];

/// The pack widths the multi-row point is stamped at — one weight pack per
/// thread per k step, or two.
///
/// Two is the one-row point's (`packs_per_thread = 2` in `quant_qmv.metal`)
/// and is what a fold of one or two rows can afford; a wider fold holds more
/// activation slices live and may want one. See
/// [`crate::tuning::DeviceTuning::qmv_rows_packs`].
const QMV_PACK_RUNGS: [i32; 2] = [1, 2];

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
            for &r in &QMV_ROW_RUNGS {
                for &p in &QMV_PACK_RUNGS {
                    let point =
                        qmv_rows_point("quant.qmv_rows", gs, b, r, p).expect("an axis point");
                    out.push(Fire::at(QMV_ROWS_FILE, point.entry).stamp(point.stamp));
                }
            }
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

/// One multi-row qmv instantiation — the entry symbol and the stamp that
/// conjures it.
///
/// Stamped rather than spelled in source for `quant_qmm_t.metal`'s reason:
/// the axis is `(group, bits) x fold x pack`, thirty-six points, of which a
/// checkpoint fires one.
pub fn qmv_rows_point(
    op: &'static str,
    group: i32,
    bits: i32,
    rows: i32,
    packs: i32,
) -> Result<Point, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &QMV_ROW_RUNGS, rows, "row group")?;
    check(op, &QMV_PACK_RUNGS, packs, "pack width")?;
    let entry = symbol(&format!(
        "affine_qmv_rows_bfloat16_gs_{group}_b_{bits}_r_{rows}_p_{packs}"
    ));
    Ok(Point {
        entry,
        stamp: symbol(&format!(
            "{QMV_ROWS_STAMP}(\"{entry}\", {group}, {bits}, {rows}, {packs})"
        )),
    })
}

/// Output rows one threadgroup of either vector point lands — two
/// simdgroups, four results each, which is where `QMV_GROUP`'s `y` extent
/// and this divisor both come from.
const QMV_OUT_PER_GROUP: u32 = 8;

/// **THE FOLD A FIRE OF `rows` TAKES, OR `None` FOR THE ONE-ROW POINT.**
///
/// The widest stamped rung that (a) the machine's table allows, (b) DIVIDES
/// the batch, and (c) leaves the launch enough threadgroups to fill the
/// machine. Each clause is a measurement and none of them is symmetry.
///
/// # (b) It must DIVIDE, because a half-empty group costs a full one
///
/// The fold's saving is per GROUP, and a group is paid for whether or not
/// its rows exist. `throughput_probe` on qwen3.6-27B, vector arm forced at
/// every width (`qmm_min_batch = 999`), ms/fire — and read the ODD columns:
///
/// ```text
///   lanes        1       2       3       4       5       6       7       8
///   one-row  62.12  107.51  159.33  207.70       —       —       —  266.63
///   R = 2    62.12   97.64  176.77  185.69  263.86  270.66  352.69  360.14
/// ```
///
/// The folded row is a staircase on the GROUP count — 97.6 at one group,
/// ~180 at two, ~267 at three, ~356 at four — so three rows cost what four
/// cost and lose to the one-row point by 11%, while two and four rows win by
/// 9% and 11%. A rule that folded whatever it was handed would buy the even
/// widths and sell the odd ones.
///
/// # (c) It must still FILL THE MACHINE
///
/// Folding R rows into one threadgroup divides the launch by R, which is the
/// point when the launch had threadgroups to spare and a defect when it did
/// not — the giant and the vehicle:
///
/// ```text
///   qwen36-27b   a projection is 5120 wide   640 threadgroups at one row
///   qwen35-0.8b  the attention four are      128 threadgroups at one row
///                1024 wide
/// ```
///
/// `crossover_tg` is [`crate::tuning::DeviceTuning::qmm_bn_crossover_tg`],
/// BORROWED — that constant's own sweep is the threadgroup count at which
/// this machine stops rewarding a smaller grid, which is the same physical
/// question asked of a different kernel. The borrow is CHECKED rather than
/// assumed: `throughput_probe` on the vehicle, 64 warm fires, the clause
/// disabled by pinning the constant to zero (which reaches nothing else
/// below the crossover, since the vector arm consults no column tile):
///
/// ```text
///   lanes            1      2      3      4
///   guard on      5.65   6.51  10.63  11.63   ms/fire
///   guard off     5.86   6.80  10.98  11.63
/// ```
///
/// Two lanes is the case the clause exists for and the only one it changes:
/// one group of the 0.8B's 1024-wide projections is 128 threadgroups, under
/// the 160 this machine fills at, so the fold declines and is 4.3% better
/// for declining. Four lanes is the CONTROL — two groups is 256
/// threadgroups, the clause passes either way, and the two columns are the
/// same number.
///
/// The step-down is monotone and so the walk is a `find`: a narrower rung is
/// always MORE threadgroups, never fewer.
///
/// # One row is not a fold
///
/// It is the one-row point exactly — already stamped, already compiled, and
/// identical in every operand.
#[must_use]
pub fn qmv_rows_fold(rows: i32, out_width: i32, max: i32, crossover_tg: i32) -> Option<i32> {
    if rows < 2 {
        return None;
    }
    let tiles = out_width.max(1).unsigned_abs().div_ceil(QMV_OUT_PER_GROUP);
    let fills = |rung: i32| {
        let groups = rows.unsigned_abs() / rung.unsigned_abs();
        groups.saturating_mul(tiles) >= crossover_tg.max(0).unsigned_abs()
    };
    QMV_ROW_RUNGS
        .iter()
        .copied()
        .rev()
        .filter(|rung| *rung <= max && *rung <= rows && rows % *rung == 0)
        .find(|rung| fills(*rung))
}

/// The multi-row vector point's geometry: one threadgroup per row GROUP down
/// x, the one-row point's output split down y.
///
/// The x extent is `div_ceil` even though [`qmv_rows_fold`] only ever hands
/// back a rung that divides: the shader guards its stores on the row count
/// either way, so a caller that selected its own fold cannot fall off the
/// end of the batch here.
pub fn qmv_rows_grid(
    op: &'static str,
    vecs: i32,
    rows_per_group: i32,
    out_vec_size: i32,
) -> Result<[u32; 3], Error> {
    if vecs <= 0 {
        return Err(refuse(op, "the vectors are zero"));
    }
    if rows_per_group <= 0 {
        return Err(refuse(op, "the row group is zero"));
    }
    if out_vec_size <= 0 {
        return Err(refuse(op, "the output vector is zero"));
    }
    let groups = vecs.unsigned_abs().div_ceil(rows_per_group.unsigned_abs());
    let x = groups
        .checked_mul(32)
        .ok_or_else(|| refuse(op, format!("{vecs} vectors will not launch")))?;
    Ok([x, out_vec_size.unsigned_abs().div_ceil(4), 1])
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

/// The row block a fire of `rows` launches its GEMM at, and the row count it
/// launches over — or `None` when the slot cannot hold a padded launch at
/// even the narrowest block.
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
/// So pad the fire up to a rung. The padding is free of consequence when the
/// rows land in slots the fire does not read: a GEMM row's output depends
/// only on its own input row, so garbage in the tail cannot reach a real one.
/// `capacity` is what makes that true and is the caller's to state — the rows
/// its activation and result rectangles actually hold, not the rows this fire
/// uses.
///
/// **IT WALKS DOWN THE RUNGS RATHER THAN DECLINING AT THE FIRST ONE THAT WILL
/// NOT FIT.** It used to ask [`bm_rung`] for one rung and answer the UNPADDED
/// width when the slot could not hold it, which dropped the fire to the
/// vector point: a 20-row fire in a 24-row slot declined while a 16-row fire
/// in the same slot did not, so a batch over the crossover could still land
/// on the arm meant for batches under it. Stepping down means a slot that
/// holds any padded launch at all gets one, and which rung it lands on does
/// not reach the answer, because every rung of this template is bit-identical
/// to every other (`the_fingerprint_matrix`).
///
/// # The corner this cannot reach, stated
///
/// A fire in the last seven rows of its own slot — `rows > capacity - 8` with
/// `rows % 8 != 0` — has nowhere to pad to and takes the vector point even
/// over the crossover, which at those widths is the slower arm. Closing it
/// wants the arena to carve its `Dim::Tokens` slots at a multiple of
/// [`BM_RUNGS`]`[0]` rows, which is `engine_metal::arena`'s to do and not
/// this plane's. Until then: a shell whose budget ceiling is a multiple of
/// eight has no such corner, because the only fire that could reach it is one
/// that fills the budget exactly and is therefore already padded.
#[must_use]
pub fn mb_block(rows: i32, capacity: i32) -> Option<(i32, i32)> {
    let rows = rows.max(1);
    let capacity = capacity.max(1);
    let fits = |rung: i32| {
        // `div_ceil` is stable on the unsigned integers only, and every
        // value here is positive by the two clamps above.
        let padded = i32::try_from(rows.unsigned_abs().div_ceil(rung.unsigned_abs()))
            .ok()?
            .saturating_mul(rung);
        (padded <= capacity).then_some((rung, padded))
    };
    // Widest first, and a rung the batch cannot COVER is not offered: a
    // one-row decode padded to sixty-four would launch sixty-four rows of
    // arithmetic to compute one. Below the narrowest rung the floor is the
    // floor, which is what makes a 1-7 row fire reach the tile at all.
    BM_RUNGS
        .iter()
        .rev()
        .copied()
        .filter(|rung| rows >= *rung)
        .find_map(fits)
        .or_else(|| fits(BM_RUNGS[0]))
}

/// The column tile for a tiled fire, and the ONLY such rule left.
///
/// There used to be a second — the widest tile that divides, for a family
/// with split-K behind it — and it was correct *because* the split supplied
/// threadgroups when the output tiles did not. Nothing supplies threadgroups
/// that way any more (see [`act_x_wt`]), and taking the widest tile with no
/// such supply starves the machine: at M=128 with
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

/// The working plane [`act_x_wt`]'s pre-cast rung needs, seated as a MINT
/// rather than as a rectangle.
///
/// **THE SHAPE IS A FIRE-TIME SELECTION, WHICH IS WHY THIS IS NOT AN
/// `Option<Tensor>`.** How many rows the staging covers is [`mb_block`]'s
/// answer, reached several guards inside the entry. What holds the plane is
/// ONE load-time reservation in the shell, so the shell is the only thing
/// that can say whether a given rectangle is inside it — and the way it says
/// so is `None`, which the rung answers by falling to the one that needs no
/// plane. A rectangle handed over before the selection ran would be the
/// caller guessing at the answer this entry exists to give.
///
/// **IT USED TO CARRY A SECOND MINT AND THE SECOND IS GONE.** The split-K
/// partials plane went out with the split arm: nothing wrote it, and a
/// reservation nothing writes is not free on a checkpoint that reaches no
/// pre-cast, where nothing larger aliases it.
///
/// Erased behind `dyn` for [`Ctx`]'s reason: this crate names no driver type.
pub struct Scratch<'a> {
    /// `rows x contraction` halves — what [`PRECAST_STAGE`] writes and the
    /// [`precast_point`] GEMM reads.
    pub precast: &'a dyn Fn(u32, u32) -> Option<Tensor>,

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
/// **THE SAME LADDER, WITH NOTHING DECLINED.** A head is the widest GEMM in
/// the stack: it takes the row rung, it stages its activation for the
/// pre-cast exactly as the projections do, and below the crossover it takes
/// the vector point like every other projection. This entry stays separate
/// from [`matmul`] because the op is, not because the ladder is.
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

/// `y = act x w^T` against a bank, tiled wherever the rectangle allows it and
/// vectorized where it does not.
///
/// # THE LADDER, KEYED ON THE FIRE'S ROWS
///
/// ```text
///   m < qmm_min_batch (5)          the vector point, folded where it may be
///   a rectangle no tile divides    the vector point, folded where it may be
///   otherwise, fp16 and staged     the pre-cast tile at mb_block's row rung
///   otherwise                      the plain stamped tile at that rung
/// ```
///
/// where "folded where it may be" is [`qmv_rows_fold`]: `affine_qmv_rows` at
/// a row group that divides the batch and still fills the machine, and
/// `affine_qmv_fast` — one row per threadgroup — everywhere else. The two
/// are BIT-IDENTICAL (`affine_floor`'s
/// `the_folded_vector_point_lands_the_one_row_bits`), so the fold is a pure
/// performance selection and not a second numerical policy beside the
/// crossover below.
///
///   1. **[`mb_block`]** picks the row block and pads the launch up to it,
///      bounded by `capacity_rows`. It is not an arm — it is what makes the
///      arms below REACHABLE at a batch that is not already an exact multiple
///      of a rung, which for a decode is almost never. Its walk DOWN the
///      rungs is what keeps a fire in a slot with no room to spare on the
///      tile rather than dropping it to the vector point.
///   2. **[`precast_point`]**, when the machine emulates bfloat and the bank
///      is the one format the staged loader is stamped for. The largest
///      single win recorded on the wide shapes, ~40% on the GEMM at every one
///      measured — and bit-identical to rung 3, which is what makes it legal
///      for the staging plane to decline for a rectangle it cannot hold.
///   3. The plain stamped point at [`bn_unsplit`]'s column tile.
///   4. The vector point, both as the arm below the crossover and as the
///      floor for a rectangle no stamped tile divides.
///
/// # The crossover, and the arm moving with the composition
///
/// [`crate::tuning::DeviceTuning::qmm_min_batch`] is read against `m`, the
/// rows the fire brought, which is the number the sweep behind it measured.
/// A dense projection is ONE matmul over the whole fire — `engine_metal`'s
/// `window` puts embed, norm and every projection in the region that stands
/// over all of a fire's rows — so `m` is the COMPOSITION's and the arm moves
/// with the neighbours. **THAT IS THE RULING AND NOT AN OVERSIGHT:**
///
/// > We do NOT need bit-level identity. If a much faster path has small
/// > numerical drift from nondeterminism, that is obviously acceptable.
///
/// The drift is real and its size is known. The two arms are not two speeds
/// of one kernel: `qdot` computes `scale * Σ code_i x_i + bias * Σ x_i` over
/// a lane's slice of k and folds thirty-two lanes with `simd_sum`, while the
/// tile dequantizes to `bf16(code * scale + bias)` and multiplies. Different
/// arithmetic, not merely a different order. Measured at one or two bf16 ulp
/// a step, which `throughput_probe`'s correctness arm saw as two lanes of
/// eight on the vehicle taking a different token at a step decided by 0.0625
/// — under its `TIE` line, which is where the gate is drawn.
///
/// **WHAT THE CROSSOVER IS WORTH.** ms/fire over 32 warm decode fires (16 for
/// the giant), the vector arm forced with `qmm_min_batch = 999` against the
/// tile arm forced with `= 2`:
///
/// ```text
///   qwen35-d0.8b-mlxu4, 32 warm decode fires
///   lanes        1      2      3      4      5      6      7      8     16     32
///   vector    5.76   7.11  10.96  12.31  16.00  15.66  17.95  18.21  29.75  53.65
///   tile     10.14  10.18  12.68  12.54  15.65  15.75  17.92  17.86  29.78  53.20
///
///   qwen36-27b-mlxu4, 16 warm decode fires
///   lanes        1      2             4                    8     16
///   vector   61.99 107.32        207.50               266.63 355.11
///   tile    197.45 206.16        227.90               266.66 355.10
/// ```
///
/// Five is where the lines cross on all four checkpoints the sweep took, and
/// from eight lanes up the two columns are the same to the third digit —
/// which is the CONTROL that says the rest of the table is the arm and not
/// drift. Below five the vector point is worth 76% of a one-lane decode on
/// the small vehicle and 3.2x on the giant.
///
/// # The tile family is one fingerprint, and that part is not a policy
///
/// Everything the ladder picks among ABOVE the crossover lands the same bits.
/// Measured over a synthetic 4-bit bank at K = 1024, every stamped point
/// fired at the same rows and compared bit for bit (`engine-metal`'s
/// `affine_floor`, `the_fingerprint_matrix`):
///
/// ```text
///   plain tile     bm 8/16/32/64 x bn 16/32/64   one fingerprint, all twelve
///   pre-cast tile  bm 8/16/32/64 x bn 16/32/64   the SAME fingerprint
///   vector point                                 parts from it
/// ```
///
/// The tile's row block, its column tile and whether its operands were staged
/// to `half` first are all invisible in the answer: `BM`, `BN`, `WM` and `WN`
/// decide who holds which element, `BK` is 32 throughout, and every output
/// element accumulates over k in one ascending order into an `f32`
/// accumulator. So [`mb_block`]'s rung walk and the pre-cast rung's decline
/// are free of consequence, and the fingerprint matrix is the regression that
/// keeps them so — an accidental change to one rung's k-order would show up
/// there rather than as a mystery at a checkpoint.
///
/// # The rectangle can still refuse the tile
///
/// `quant_qmm_t.metal`'s hot loop is `load_unsafe` on both operands — it
/// reads a whole `BM x BK` block with no edge predicate — so the shader's own
/// header states `M % BM == 0, N % BN == 0, K % BK == 0` as the condition
/// under which a driver may select it AT ALL. The dense gemm beside this one
/// tiles with `div_ceil` because its tile kernel guards its own edges; this
/// one cannot. `N` and `K` are the WEIGHT's and so a refusal on either is a
/// property of the checkpoint, the same for every fire. `M` is padded up by
/// [`mb_block`], and a rectangle it cannot pad takes the vector point.
///
/// # Two arms that are NOT here
///
/// **THE SPLIT-K ARM IS GONE, AND SO ARE ITS HELPERS.** It partitions k and
/// folds the pieces at a depth that was a function of the fire's rows, and
/// the fold is a third summation order on top of the two above. Nothing sizes
/// a partials plane and nothing binds `affine_qmm_t_splitk`; those
/// instantiations sit in `quant_qmm_t.metal` unminted. What the removal costs
/// is on the formats that do not reach FP16 — 8-bit, and group 32 or 128 —
/// where the reference's own sweep prices it at 741 → 887 tok/s on llama-1B
/// at 32 lanes. It comes back if a measured shape wants it.
///
/// **AND `quant_narrow.metal` IS GONE.** It was a one-thread-per-column point
/// written to land the tile's bits at one and two rows, and it did — but the
/// vector point beats it across its whole range (107.8 against 61.99 ms on
/// the giant at one lane; 142 GB/s against 230), so on this ladder there is
/// no width at which it would be selected. An arm nobody may take is not an
/// arm.
///
/// # The two-to-four band, and the hypothesis that did not survive it
///
/// Between one lane and the crossover, the vector arm's cost is very nearly
/// linear in the rows — 62.19 / 107.51 / 159.33 / 207.70 ms on the giant —
/// which is what a kernel repeating a read it could share looks like, and
/// `quant_qmv_rows.metal` was written to share it. **The read was never the
/// bill.** Fired standalone over a bank past the system cache and then over a
/// slice inside it, the one-row point costs the SAME per row either way: it
/// is arithmetic-bound, and the 365 GB/s it reaches at sixteen rows is a
/// coincidence of this machine's balance. The fold survives on a smaller
/// argument — it removes the load instructions and the scale arithmetic
/// beside them, about a tenth — and that tenth is where the band's numbers
/// moved. Both measurements are on
/// [`crate::tuning::DeviceTuning::qmv_rows_max`].
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
    // Which side of it a SHELL falls on moved by 40% on this very GEMM when
    // the FP16 matrix path arrived, and it moves again per Apple family —
    // hence [`crate::tuning`] rather than a number here.
    //
    // The DENSE arm, always: whether the checkpoint's FFN is routed is a fact
    // about the MODEL, and no operand of `linear.matmul` carries it.
    let tuned = crate::tuning::current();
    let fp16 = tuned.fp16_gemm_format(w.bits, w.group);
    let min_batch = i32::try_from(tuned.qmm_min_batch(false, fp16)).unwrap_or(i32::MAX);
    let crossover = i32::try_from(tuned.qmm_bn_crossover_tg).unwrap_or(i32::MAX);
    let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
    // **THE CROSSOVER IS ASKED OF `m`** — the rows this fire brought, which
    // is the number the sweep behind `qmm_min_batch` measured. See this
    // entry's header for the table and for what moves with the composition.
    //
    // **RUNG 1, AND IT IS NOT AN ARM.** [`mb_block`] answers the widest row
    // block the batch can cover whose padded launch the slot can hold, and
    // the row count to launch over. `None` is a rectangle that cannot be
    // padded at all, which takes the vector point below.
    if m >= min_batch
        && k % QMM_BK == 0
        && let Some((bm, padded)) = mb_block(m, capacity)
    {
        // **RUNG 2: the staged input.** Two dispatches and one plane — the
        // cast writes `padded x k` halves, the GEMM reads them at buffer 12
        // and leaves the bf16 activation seat null. The padding is what makes
        // the staging cover rows the fire does not use, and it is the same
        // guarantee `mb_block` states: those rows are inside the activation
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
        // **RUNG 3: the plain stamped point**, at the column tile measured
        // for a family with no split behind it ([`bn_unsplit`]) — which is
        // what every tiled arm here is, now that the split is gone.
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
    // **RUNG 4: the vector point, FOLDED.** The one-row point walks one
    // threadgroup per row, so a fire of `m` rows reads the bank `m` times;
    // `quant_qmv_rows.metal` reads it once per group of `R`. The fold is the
    // narrowest rung that covers the batch ([`qmv_rows_fold`]), and at one
    // row there is nothing to fold and the one-row point is fired unchanged.
    let rows_max = i32::try_from(tuned.qmv_rows_max).unwrap_or(1);
    let packs = i32::try_from(tuned.qmv_rows_packs).unwrap_or(QMV_PACK_RUNGS[1]);
    if let Some(fold) = qmv_rows_fold(m, n, rows_max, crossover) {
        let point = qmv_rows_point(op, group, bits, fold, packs)?;
        return ctx.fire(
            Fire::at(QMV_ROWS_FILE, point.entry)
                .stamp(point.stamp)
                .apply(Grid::of(qmv_rows_grid(op, m, fold, n)?, QMV_GROUP)),
            &[
                w.codes.arg(),
                w.scales.arg(),
                biases.arg(),
                act.arg(),
                y.arg_mut(),
                k.arg(),
                n.arg(),
                // The batch, which is where a padded group's stores stop.
                m.arg(),
            ],
        );
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
    fn a_launch_pads_to_the_widest_rung_the_slot_can_hold() {
        // The rung the batch can cover, padded up, whenever the slot has the
        // rows to spare — and a 1-7 row fire reaches the floor rung, which is
        // what a decode needs now that the crossover is not asked of `m`.
        assert_eq!(mb_block(1, 4096), Some((8, 8)));
        assert_eq!(mb_block(2, 4096), Some((8, 8)));
        assert_eq!(mb_block(7, 4096), Some((8, 8)));
        assert_eq!(mb_block(8, 4096), Some((8, 8)));
        assert_eq!(mb_block(15, 4096), Some((8, 16)));
        assert_eq!(mb_block(20, 4096), Some((16, 32)));
        assert_eq!(mb_block(65, 4096), Some((64, 128)));
        // **THE STEP DOWN.** A 20-row fire in a 24-row slot cannot pad to the
        // 16 rung's 32, so it takes the 8 rung's 24 rather than declining to
        // the vector point — which is the arm whose bits are not the tile's.
        assert_eq!(mb_block(20, 24), Some((8, 24)));
        // A 3-row fire needs eight rows of slot and gets them at seven
        // nowhere: this is the corner stated on `mb_block`.
        assert_eq!(mb_block(3, 8), Some((8, 8)));
        assert_eq!(mb_block(3, 7), None);
        // The corner at the top of a budget, which is the same corner.
        assert_eq!(mb_block(100, 100), None);
        // 96 rows cover the 64 rung but pad to 128, which a 100-row slot
        // will not hold; the 32 rung's 96 is exact and fits, so it is taken.
        assert_eq!(mb_block(96, 100), Some((32, 96)));
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
    fn a_fold_divides_the_batch_and_still_fills_the_machine() {
        // A projection wide enough that the fill clause never bites: 5120
        // columns is 640 tiles, so even one group is far past 160.
        let wide = |rows| qmv_rows_fold(rows, 5120, 2, 160);
        // One row is not a fold.
        assert_eq!(wide(1), None);
        assert_eq!(wide(2), Some(2));
        // Three rows: no stamped rung divides three, and a half-empty group
        // costs a full one — 176.77 ms against the one-row point's 159.33 on
        // the giant, which is the measurement `qmv_rows_fold` carries.
        assert_eq!(wide(3), None);
        assert_eq!(wide(4), Some(2));
        assert_eq!(wide(5), None);
        assert_eq!(wide(6), Some(2));
        // The ceiling is the machine's and is obeyed: the 4 rung divides
        // four rows and is still not offered at a ceiling of two.
        assert_eq!(qmv_rows_fold(4, 5120, 4, 160), Some(4));
        assert_eq!(qmv_rows_fold(8, 5120, 8, 160), Some(8));
        // Widest first among the rungs that divide.
        assert_eq!(qmv_rows_fold(8, 5120, 4, 160), Some(4));
    }

    #[test]
    fn a_fold_that_would_empty_the_machine_declines() {
        // The vehicle's attention four are 1024 wide — 128 tiles, and one
        // group of them is under the 160 this machine fills at. So a
        // two-row fire folds on the wide projections and not on the narrow
        // ones, which is the whole reason the clause is here.
        assert_eq!(qmv_rows_fold(2, 1024, 2, 160), None);
        assert_eq!(qmv_rows_fold(2, 3584, 2, 160), Some(2));
        // Four rows over the narrow projection: two groups of 128 tiles is
        // 256, which fills, so the fold is back.
        assert_eq!(qmv_rows_fold(4, 1024, 2, 160), Some(2));
        // A machine that fills at nothing folds everything.
        assert_eq!(qmv_rows_fold(2, 1024, 2, 0), Some(2));
    }

    #[test]
    fn the_folded_point_launches_one_threadgroup_per_group() {
        // Two rows folded two ways is ONE group down x, and the output split
        // down y is the one-row point's exactly.
        assert_eq!(qmv_rows_grid("t", 2, 2, 5120).unwrap(), [32, 1280, 1]);
        assert_eq!(qmv_grid("t", 2, 5120).unwrap(), [64, 1280, 1]);
        assert_eq!(qmv_rows_grid("t", 8, 2, 5120).unwrap(), [4 * 32, 1280, 1]);
        assert!(qmv_rows_grid("t", 0, 2, 5120).is_err());
        assert!(qmv_rows_grid("t", 2, 0, 5120).is_err());
    }

    #[test]
    fn the_folded_points_are_stamped_on_five_axes() {
        let point = qmv_rows_point("t", 64, 4, 2, 1).unwrap();
        assert_eq!(point.entry, "affine_qmv_rows_bfloat16_gs_64_b_4_r_2_p_1");
        assert_eq!(
            point.stamp,
            "PIE_STAMP_qmv_rows(\"affine_qmv_rows_bfloat16_gs_64_b_4_r_2_p_1\", 64, 4, 2, 1)"
        );
        // Every axis is checked against what the macro will mint.
        assert!(qmv_rows_point("t", 64, 4, 3, 1).is_err());
        assert!(qmv_rows_point("t", 64, 4, 2, 4).is_err());
        assert!(qmv_rows_point("t", 48, 4, 2, 1).is_err());
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
        // wide, and `bn_unsplit` never answers it.
        assert!(precast_point("t", "", 16, 8).is_err());
    }
}
