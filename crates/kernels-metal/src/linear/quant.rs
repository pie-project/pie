//! Affine qmm/qmv entry points: names, instantiation stamps, grids, and the
//! two entries a dense projection against a quantized weight fires.
//!
//! `quant_qmm_t.metal` points are stamped: a `PIE_STAMP_qmm_t(...)` macro
//! invocation, appended before compiling, selects one `(group, bits, bm,
//! bn)` point; [`Point`] pairs the entry with that stamp. `quant_qmv.metal`
//! instantiates its six points directly, so [`qmv_name`] hands back an
//! empty stamp. Entry names are interned to `&'static str`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::{Bank, Tensor};

/// The three source files the points live in.
const QMM_FILE: &str = "linear/quant_qmm_t.metal";
const QMV_FILE: &str = "linear/quant_qmv.metal";

/// The multi-row vector point's file — stamped, unlike the one-row file.
const QMV_ROWS_FILE: &str = "linear/quant_qmv_rows.metal";

/// The macro `quant_qmv_rows.metal` publishes.
const QMV_ROWS_STAMP: &str = "PIE_STAMP_qmv_rows";

/// The macro `quant_qmm_t.metal` publishes for the epilogue-free tiled
/// point this plane selects: dense projections carry no bias or residual.
const QMM_STAMP: &str = "PIE_STAMP_qmm_t";

/// Group sizes the shaders instantiate.
const GROUPS: [i32; 3] = [32, 64, 128];

/// Bit widths they instantiate. 2-bit gets no dedicated precast or
/// `qmm_min_batch` tuning: it takes the plain tiled kernel and crossover.
const WIDTHS: [i32; 3] = [2, 4, 8];

/// Whether the tiled qmm family is stamped at `bits` (an unstamped width is a decline, not a fault).
pub(crate) fn qmm_stamps_width(bits: u32) -> bool {
    i32::try_from(bits).is_ok_and(|bits| WIDTHS.contains(&bits))
}

/// Column tiles the qmm point is stamped at, widest first; also the routed families' row tiles.
const TILES: [i32; 3] = [64, 32, 16];

/// Row tiles the dense families are stamped at, widest first — [`TILES`] plus the 8 rung.
const ROW_TILES: [i32; 4] = [64, 32, 16, 8];

/// The threadgroup a row block of `bm` launches at — `[SIMD_SIZE, WN, WM]`.
/// Launching the wrong `WM` is a wrong answer, not a slow one: both block
/// loaders divide their tile by `tgp_size` with no bound check.
#[must_use]
pub fn qmm_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm < 2 * FRAG_ROWS { 1 } else { 2 }]
}

/// Rows of one `simdgroup_matrix` fragment (`BaseMMAFrag::kFragRows`).
const FRAG_ROWS: i32 = 8;

const QMV_GROUP: [u32; 3] = [32, 2, 1];

/// The contraction step the tiled point walks; `K % BK == 0` is required for `load_unsafe`.
const QMM_BK: i32 = 32;

/// Row tiles the GEMM is stamped at, narrowest first ([`bm_rung`]'s walk order); floor is 8 since `BM = 4` leaves `WM` nowhere to go.
const BM_RUNGS: [i32; 4] = [8, 16, 32, 64];

/// Column tiles, narrowest first — the order [`bn`] walks.
const BN_RUNGS: [i32; 3] = [16, 32, 64];

/// Row groups the vector point may fold at, narrowest first (one weight-block read serves `R` rows).
const QMV_ROW_RUNGS: [i32; 3] = [2, 4, 8];

/// Pack widths the multi-row point is stamped at; only width 2 matches the one-row point bit for bit.
const QMV_PACK_RUNGS: [i32; 2] = [1, 2];

/// Every point this plane may fire, ready to compile. Not on the boot path — a probe/test helper.
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

/// One selected qmm instantiation: the entry symbol and the jit stamp that conjures it.
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

/// One qmv instantiation; the stamp is always empty (`quant_qmv.metal` spells its points out).
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

/// One multi-row qmv instantiation, stamped: the axis is `(group, bits) x fold x pack`, 36 points.
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

/// Output rows one threadgroup of either vector point lands: two simdgroups, four results each.
const QMV_OUT_PER_GROUP: u32 = 8;

/// The fold a fire of `rows` takes, or `None` for the one-row point: the widest rung dividing the batch that fills the machine.
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

/// The multi-row vector point's geometry: one threadgroup per row group
/// down x, the one-row point's output split down y. `div_ceil` down x so a
/// caller-selected fold can't fall off the end of the batch.
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

/// The widest row rung a batch of `rows` can COVER, not the widest that divides it — callers pad up
/// to what this returns, so a one-row decode must not get back a 64-row block.
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

/// The row block a fire of `rows` launches its GEMM at, and the padded row
/// count — or `None` if no rung fits `capacity` (the kernel takes no `M`).
#[must_use]
pub fn mb_block(rows: i32, capacity: i32) -> Option<(i32, i32)> {
    let rows = rows.max(1);
    let capacity = capacity.max(1);
    let fits = |rung: i32| {
        // Positive by the two clamps above, so `div_ceil`'s unsigned bound holds.
        let padded = i32::try_from(rows.unsigned_abs().div_ceil(rung.unsigned_abs()))
            .ok()?
            .saturating_mul(rung);
        (padded <= capacity).then_some((rung, padded))
    };
    // Widest first, then a rung the batch cannot cover is not offered.
    BM_RUNGS
        .iter()
        .rev()
        .copied()
        .filter(|rung| rows >= *rung)
        .find_map(fits)
        .or_else(|| fits(BM_RUNGS[0]))
}

/// The column tile for a tiled fire: narrow until there's enough work to fill the machine, then 32 — never 64.
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

/// The staged-input point: the FP16 pre-cast GEMM, stamped at gs=64/b=4 alone.
pub fn precast_point(op: &'static str, form: &str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t{form}_fp16_precast_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

/// The routed tiled point whose weight loader dequantizes straight to `half`, stamped at gs=64/b=4 alone.
pub fn routed_fp16_point(op: &'static str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    check(op, &TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

/// The routed tiled point for an mxfp4 bank, in both forms: gpt-oss's `down_bias` needs the unbiased one too.
pub fn mxfp4_routed_point(
    op: &'static str,
    form: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    check(op, &TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "mxfp4_qmm_t_routed{form}_bfloat16_bm_{bm}_bn_{bn}"
    )))
}

/// The staging pass the precast points read: one thread per activation element, bf16 in and `half` out.
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
    // The three multipliers are the threadgroup's extents, not constants;
    // see `qmm_group`, whose z extent is 1 at the 8 rung.
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

/// Interns a composed name into a `&'static str`; names are few, so the leak is bounded.
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

/// The working plane [`act_x_wt`]'s pre-cast rung needs; `None` means the rectangle doesn't fit.
pub struct Scratch<'a> {
    /// `rows x contraction` halves — what [`PRECAST_STAGE`] writes and the
    /// [`precast_point`] GEMM reads.
    pub precast: &'a dyn Fn(u32, u32) -> Option<Tensor>,

}

/// `y = act x w^T` where `w` is a quantized bank — the quantized twin of [`gemm::matmul`](crate::linear::gemm::matmul).
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

/// The same product at the vocabulary head, on the same ladder as [`matmul`], kept separate for the op name.
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

/// `y = act x w^T` against a bank: below `qmm_min_batch`, or a rectangle no
/// tile divides, the vector point (folded via [`qmv_rows_fold`] where a
/// rung divides the batch); otherwise the stamped tile at [`mb_block`]'s
/// row rung ([`precast_point`] when fp16 and staged).
///
/// The vector and tile arms are different arithmetic, with small drift by
/// design across the crossover; the fold matches the one-row point bit for
/// bit only at pack width 2 ([`QMV_PACK_RUNGS`]). `M` is padded by
/// [`mb_block`] to `M % BM == 0`; a rectangle it cannot pad takes the vector
/// point.
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
    // A dense projection against an mxfp4 bank is not stamped: the affine
    // points would read an e8m0 exponent byte as half a bf16 factor.
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
    // The crossover is the machine's and the format's, read from `crate::tuning`.
    let tuned = crate::tuning::current();
    let fp16 = tuned.fp16_gemm_format(w.bits, w.group);
    let min_batch = i32::try_from(tuned.qmm_min_batch(false, fp16)).unwrap_or(i32::MAX);
    let crossover = i32::try_from(tuned.qmm_bn_crossover_tg).unwrap_or(i32::MAX);
    let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
    // Rung 1, not an arm itself: `None` takes the vector point below.
    if m >= min_batch
        && k % QMM_BK == 0
        && let Some((bm, padded)) = mb_block(m, capacity)
    {
        // Rung 2: the staged input. The cast writes `padded x k` halves,
        // the GEMM reads them at buffer 12 and leaves the bf16 seat null.
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
                // The bf16 activation seat: unread here, but binds at its own index.
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
        // Rung 3: the plain stamped point, at the column tile from
        // `bn_unsplit`.
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
    // Rung 4: the vector point, folded where a group of `R` rows can share
    // one weight read; at one row there is nothing to fold.
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

/// The three extents. A zero width is a malformed row; zero rows is a fire with nothing to do.
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
        // Eight is a ROW tile, not a column one: no point is stamped 8 wide.
        assert!(precast_point("t", "", 16, 8).is_err());
    }
}
