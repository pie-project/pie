use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::tensor::{Bank, Tensor};

const QMM_FILE: &str = "linear/quant_qmm_t.metal";
const QMV_FILE: &str = "linear/quant_qmv.metal";

const QMV_ROWS_FILE: &str = "linear/quant_qmv_rows.metal";

const QMV_ROWS_STAMP: &str = "PIE_STAMP_qmv_rows";

const QMM_STAMP: &str = "PIE_STAMP_qmm_t";

const QMM_WIDE_STAMP: &str = "PIE_STAMP_qmm_t_wide";

fn tiled_form() -> (&'static str, &'static str) {
    if crate::tuning::current().qmm_wide_range {
        ("_wide", QMM_WIDE_STAMP)
    } else {
        ("", QMM_STAMP)
    }
}

const GROUPS: [i32; 3] = [32, 64, 128];

const WIDTHS: [i32; 3] = [2, 4, 8];

pub(crate) fn qmm_stamps_width(bits: u32) -> bool {
    i32::try_from(bits).is_ok_and(|bits| WIDTHS.contains(&bits))
}

const TILES: [i32; 3] = [64, 32, 16];

const ROW_TILES: [i32; 4] = [64, 32, 16, 8];

#[must_use]
pub fn qmm_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm < 2 * FRAG_ROWS { 1 } else { 2 }]
}

const FRAG_ROWS: i32 = 8;

const QMV_GROUP: [u32; 3] = [32, 2, 1];

const QMM_BK: i32 = 32;
const PRECAST_BK: i32 = 64;

const PRECAST_MIN_BM: i32 = 16;

const BM_RUNGS: [i32; 4] = [8, 16, 32, 64];

const BN_RUNGS: [i32; 3] = [16, 32, 64];

const SPLITK_FILL_TG: u32 = 512;

const SPLITK_MAX: i32 = 8;

const SPLITK_BN: i32 = 32;

const SPLITK_REDUCE: &str = "qmm_splitk_reduce_f32_bfloat16";

#[must_use]
pub fn splitk(n: i32, bm: i32, padded: i32, k: i32, group: i32, bits: i32) -> i32 {
    if bm != BM_RUNGS[0] || n <= 0 || k <= 0 || n % SPLITK_BN != 0 || !(bits == 4 || bits == 8) {
        return 1;
    }
    let tiles = (n / SPLITK_BN).unsigned_abs() * (padded / bm).max(1).unsigned_abs();
    if tiles == 0 || tiles >= SPLITK_FILL_TG {
        return 1;
    }
    let mut split = i32::try_from((SPLITK_FILL_TG / tiles).next_power_of_two())
        .unwrap_or(1)
        .clamp(1, SPLITK_MAX);
    let unit = group.max(QMM_BK);
    while split > 1 && k % (split * unit) != 0 {
        split /= 2;
    }
    split
}

pub fn splitk_point(
    op: &'static str,
    group: i32,
    bits: i32,
    bm: i32,
) -> Result<&'static str, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &[4, 8], bits, "split-K bit width")?;
    check(op, &ROW_TILES, bm, "row tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_splitk_f32_bfloat16_gs_{group}_b_{bits}_bm_{bm}_bn_{SPLITK_BN}"
    )))
}

const QMV_ROW_RUNGS: [i32; 7] = [2, 3, 4, 5, 6, 7, 8];

const QMV_ROW_RUNGS_PACK2: [i32; 6] = [2, 3, 4, 6, 7, 8];

const QMV_ROW_RUNGS_PACK2_2BIT: [i32; 3] = [2, 4, 8];

const QMV_ROW_RUNGS_2BIT: [i32; 2] = [2, 3];

const QMM_MIN_BATCH_2BIT: i32 = 5;

fn qmv_rungs_at(packs: i32, bits: i32) -> &'static [i32] {
    if packs >= 2 && bits == 2 {
        &QMV_ROW_RUNGS_PACK2_2BIT
    } else if packs >= 2 {
        &QMV_ROW_RUNGS_PACK2
    } else if bits == 2 {
        &QMV_ROW_RUNGS_2BIT
    } else {
        &QMV_ROW_RUNGS
    }
}

const QMV_PACK_RUNGS: [i32; 2] = [1, 2];

#[must_use]
pub fn composed() -> Vec<Fire> {
    let mut out = Vec::new();
    for &gs in &GROUPS {
        for &b in &WIDTHS {
            let point = qmv_point("quant.qmv", "fast", gs, b).expect("an axis point");
            out.push(Fire::at(QMV_FILE, point.entry));
            for &p in &QMV_PACK_RUNGS {
                for &r in qmv_rungs_at(p, b) {
                    let point =
                        qmv_rows_point("quant.qmv_rows", gs, b, r, p).expect("an axis point");
                    out.push(Fire::at(QMV_ROWS_FILE, point.entry).stamp(point.stamp));
                }
            }
            for &bm in &ROW_TILES {
                for &bn in &TILES {
                    let (form, stamp) = tiled_form();
                    let point = qmm_point("quant.qmm_t", form, stamp, gs, b, bm, bn)
                        .expect("an axis point, by construction");
                    out.push(Fire::at(QMM_FILE, point.entry).stamp(point.stamp));
                }
            }
        }
    }
    out
}

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

pub fn qmv_rows_point(
    op: &'static str,
    group: i32,
    bits: i32,
    rows: i32,
    packs: i32,
) -> Result<Point, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &QMV_PACK_RUNGS, packs, "pack width")?;
    check(op, qmv_rungs_at(packs, bits), rows, "row group")?;
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

const QMV_OUT_PER_GROUP: u32 = 8;

#[must_use]
pub fn qmv_rows_fold(
    rows: i32,
    out_width: i32,
    max: i32,
    crossover_tg: i32,
    packs: i32,
    bits: i32,
) -> Option<i32> {
    if rows < 2 {
        return None;
    }
    let tiles = out_width.max(1).unsigned_abs().div_ceil(QMV_OUT_PER_GROUP);
    let fills = |rung: i32| {
        let groups = rows.unsigned_abs() / rung.unsigned_abs();
        groups.saturating_mul(tiles) >= crossover_tg.max(0).unsigned_abs()
    };
    qmv_rungs_at(packs, bits)
        .iter()
        .copied()
        .rev()
        .filter(|rung| *rung <= max && *rung <= rows && rows % *rung == 0)
        .find(|rung| fills(*rung))
}

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

fn check(op: &'static str, points: &[i32], v: i32, what: &'static str) -> Result<(), Error> {
    points
        .contains(&v)
        .then_some(())
        .ok_or_else(|| refuse(op, format!("no point is stamped at {what} {v}")))
}

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

#[must_use]
pub fn mb_block(rows: i32, capacity: i32) -> Option<(i32, i32)> {
    let rows = rows.max(1);
    let capacity = capacity.max(1);
    let fits = |rung: i32| {
        let padded = i32::try_from(rows.unsigned_abs().div_ceil(rung.unsigned_abs()))
            .ok()?
            .saturating_mul(rung);
        (padded <= capacity).then_some((rung, padded))
    };
    BM_RUNGS
        .iter()
        .rev()
        .copied()
        .filter(|rung| rows >= *rung)
        .find_map(fits)
        .or_else(|| fits(BM_RUNGS[0]))
}

const WIDEN_TO: i32 = 32;

#[must_use]
pub fn widen_rung(bm: i32, padded: i32, fills: impl Fn(i32) -> bool) -> i32 {
    BM_RUNGS
        .iter()
        .copied()
        .filter(|rung| *rung > bm && *rung <= WIDEN_TO && *rung == padded)
        .find(|rung| fills(*rung))
        .unwrap_or(bm)
}

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

pub fn precast_point(
    op: &'static str,
    form: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t{form}_fp16_precast_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

pub fn routed_fp16_point(op: &'static str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_routed_fp16_bfloat16_gs_64_b_4_bm_{bm}_bn_{bn}"
    )))
}

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

pub const PRECAST_STAGE: &str = "cast_qmm_input_bfloat16_to_float16";

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
    let group = qmm_group(bm);
    Ok([
        lanes(
            n.unsigned_abs().div_ceil(bn.unsigned_abs()),
            group[0],
            "the column tiles",
        )?,
        lanes(
            m.unsigned_abs() / bm.unsigned_abs(),
            group[1],
            "the row tiles",
        )?,
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

pub struct Scratch<'a> {
    pub precast: &'a dyn Fn(u32, u32) -> Option<Tensor>,

    pub partials: &'a dyn Fn(u32, u32) -> Option<Tensor>,
}

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
    let tuned = crate::tuning::current();
    let fp16 = tuned.fp16_gemm_format(w.bits, w.group);
    let min_batch = if bits == 2 {
        QMM_MIN_BATCH_2BIT
    } else {
        i32::try_from(tuned.qmm_min_batch(false, fp16)).unwrap_or(i32::MAX)
    };
    let crossover = i32::try_from(tuned.qmm_bn_crossover_tg).unwrap_or(i32::MAX);
    let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
    let fold_widest = *QMV_ROW_RUNGS.last().expect("a rung");
    let tile_fills = |padded: i32, bm: i32| {
        let tiles = u64::from(n.unsigned_abs().div_ceil(BN_RUNGS[0].unsigned_abs()))
            * u64::from((padded / bm).unsigned_abs());
        tiles >= u64::from(crossover.unsigned_abs())
    };
    let split_plane = |padded: i32, bm: i32| -> Option<(i32, Tensor)> {
        let split = splitk(n, bm, padded, k, group, bits);
        if split <= 1 {
            return None;
        }
        let rows = padded.checked_mul(split)?.unsigned_abs();
        (scratch.partials)(rows, n.unsigned_abs()).map(|plane| (split, plane))
    };
    if m >= min_batch
        && k % QMM_BK == 0
        && let Some((bm, padded)) = mb_block(m, capacity)
        && (m > fold_widest || tile_fills(padded, bm) || split_plane(padded, bm).is_some())
    {
        let bm = widen_rung(bm, padded, |rung| tile_fills(padded, rung));
        if fp16
            && bm >= PRECAST_MIN_BM
            && k % PRECAST_BK == 0
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
        if let Some((split, partials)) = split_plane(padded, bm) {
            let stride = padded
                .checked_mul(n)
                .ok_or_else(|| refuse(op, format!("{padded} x {n} partials will not stack")))?;
            ctx.fire(
                Fire::at(QMM_FILE, splitk_point(op, group, bits, bm)?).apply(Grid::of(
                    qmm_grid(op, n, SPLITK_BN, padded, bm, split)?,
                    qmm_group(bm),
                )),
                &[
                    w.codes.arg(),
                    w.scales.arg(),
                    biases.arg(),
                    act.arg(),
                    ctx.absent()?,
                    k.arg(),
                    n.arg(),
                    ctx.absent()?,
                    partials.arg_mut(),
                    (k / split).arg(),
                    stride.arg(),
                ],
            )?;
            let mut reduce = vec![ctx.absent()?; 4];
            reduce.push(y.arg_mut());
            reduce.push(ctx.absent()?);
            reduce.push(n.arg());
            reduce.push(ctx.absent()?);
            reduce.push(partials.arg());
            reduce.push(ctx.absent()?);
            reduce.push(stride.arg());
            reduce.push(split.arg());
            return ctx.fire(
                Fire::at(QMM_FILE, SPLITK_REDUCE).apply(Grid::of(
                    [n.unsigned_abs(), m.unsigned_abs(), 1],
                    [256, 1, 1],
                )),
                &reduce,
            );
        }
        if let Some(bn) = bn_unsplit(n, padded / bm, crossover) {
            let (form, stamp) = tiled_form();
            let point = qmm_point(op, form, stamp, group, bits, bm, bn)?;
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
    let rows_max = i32::try_from(tuned.qmv_rows_max).unwrap_or(1);
    let packs = i32::try_from(tuned.qmv_rows_packs).unwrap_or(QMV_PACK_RUNGS[1]);
    if let Some(fold) = qmv_rows_fold(m, n, rows_max, crossover, packs, bits) {
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
    fn quant_every_case() {
        the_folded_points_are_stamped_on_five_axes();
        a_batch_between_rungs_takes_the_wider_tile();
        the_split_follows_the_tile_count();
        the_precast_points_are_stamped_at_g64_b4_alone();
    }

    fn the_folded_points_are_stamped_on_five_axes() {
        let point = qmv_rows_point("t", 64, 4, 2, 1).unwrap();
        assert_eq!(point.entry, "affine_qmv_rows_bfloat16_gs_64_b_4_r_2_p_1");
        assert_eq!(
            point.stamp,
            "PIE_STAMP_qmv_rows(\"affine_qmv_rows_bfloat16_gs_64_b_4_r_2_p_1\", 64, 4, 2, 1)"
        );
        assert!(qmv_rows_point("t", 64, 4, 3, 1).is_ok());
        assert!(qmv_rows_point("t", 64, 4, 9, 1).is_err());
        assert!(qmv_rows_point("t", 64, 4, 3, 2).is_ok());
        assert!(qmv_rows_point("t", 64, 4, 5, 2).is_err());
        assert!(qmv_rows_point("t", 64, 4, 7, 2).is_ok());
        assert!(qmv_rows_point("t", 64, 2, 3, 2).is_err());
        assert!(qmv_rows_point("t", 64, 2, 4, 2).is_ok());
        assert!(qmv_rows_point("t", 64, 2, 3, 1).is_ok());
        assert!(qmv_rows_point("t", 64, 2, 4, 1).is_err());
        assert!(qmv_rows_point("t", 64, 2, 4, 2).is_ok());
        assert!(qmv_rows_point("t", 64, 4, 2, 4).is_err());
        assert!(qmv_rows_point("t", 48, 4, 2, 1).is_err());
    }

    fn a_batch_between_rungs_takes_the_wider_tile() {
        let fills = |_: i32| true;
        assert_eq!(mb_block(12, 64), Some((8, 16)));
        assert_eq!(widen_rung(8, 16, fills), 16);
        assert_eq!(mb_block(24, 64), Some((16, 32)));
        assert_eq!(widen_rung(16, 32, fills), 32);
        assert_eq!(mb_block(48, 64), Some((32, 64)));
        assert_eq!(widen_rung(32, 64, fills), 32);
        assert_eq!(widen_rung(16, 16, fills), 16);
        assert_eq!(widen_rung(8, 8, fills), 8);
        assert_eq!(widen_rung(8, 16, |_| false), 8);
    }

    fn the_split_follows_the_tile_count() {
        assert_eq!(splitk(17408, 8, 8, 5120, 64, 4), 1);
        assert_eq!(splitk(5120, 8, 8, 5120, 64, 4), 4);
        assert_eq!(splitk(5120, 8, 8, 17408, 64, 4), 4);
        assert_eq!(splitk(1024, 8, 8, 5120, 64, 4), 8);
        assert_eq!(splitk(1024, 16, 16, 5120, 64, 4), 1);
        assert_eq!(splitk(1024, 8, 8, 5120, 64, 2), 1);
        assert_eq!(splitk(1024, 8, 8, 5120, 128, 4), 8);
        assert_eq!(splitk(1024, 8, 8, 576, 64, 4), 1);
        assert_eq!(splitk(1000, 8, 8, 5120, 64, 4), 1);
        assert_eq!(
            splitk_point("t", 64, 4, 8).unwrap(),
            "affine_qmm_t_splitk_f32_bfloat16_gs_64_b_4_bm_8_bn_32"
        );
        assert!(splitk_point("t", 64, 2, 8).is_err());
    }

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
        assert_eq!(
            precast_point("t", "", 8, 16).unwrap(),
            "affine_qmm_t_fp16_precast_bfloat16_gs_64_b_4_bm_8_bn_16"
        );
        assert!(precast_point("t", "", 16, 8).is_err());
    }
}
