//! The tiled matmul over a quantized bank (`linear/qmm_t.metal`): row and
//! column tiles, the split-K partials and their reduce, the fp16 precast,
//! and the routed (MoE) points the expert matmuls take.

use crate::Tuning;
use crate::encode::{Arg, Ctx, Fire, Grid, refuse};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

use super::{GROUPS, Point, Scratch, WIDTHS, check, symbol};
use crate::linear::qmv::rows::QMV_ROW_RUNGS;

pub(super) const QMM_FILE: &str = "linear/qmm_t.metal";

const QMM_STAMP: &str = "PIE_STAMP_qmm_t";

const QMM_WIDE_STAMP: &str = "PIE_STAMP_qmm_t_wide";

fn tiled_form(wide_range: bool) -> (&'static str, &'static str) {
    if wide_range {
        ("_wide", QMM_WIDE_STAMP)
    } else {
        ("", QMM_STAMP)
    }
}

pub(crate) fn qmm_stamps_width(bits: u32) -> bool {
    i32::try_from(bits).is_ok_and(|bits| WIDTHS.contains(&bits))
}

const TILES: [i32; 3] = [64, 32, 16];

const ROW_TILES: [i32; 4] = [64, 32, 16, 8];

#[must_use]
pub fn qmm_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm < 2 * FRAG_ROWS { 1 } else { 2 }]
}

/// The threadgroup of the MXFP4 and fp16 routed matmuls: `qmm_t.metal`'s
/// `routed_wm`, two simdgroups for a tile of at most 16 rows.
#[must_use]
pub fn routed_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm <= 2 * FRAG_ROWS { 1 } else { 2 }]
}

const FRAG_ROWS: i32 = 8;

/// The contraction tile of the tiled kernels.
pub(crate) const QMM_BK: i32 = 32;
const PRECAST_BK: i32 = 64;

const BM_RUNGS: [i32; 4] = [8, 16, 32, 64];

const BN_RUNGS: [i32; 3] = [16, 32, 64];

const SPLITK_BN: i32 = 32;

pub(super) const SPLITK_REDUCE: &str = "qmm_splitk_reduce_f32_bfloat16";

/// The slices a skinny matmul's contraction splits into under
/// `kernel.qmm_splitk` (`limit` is its `(fill threadgroups, most slices)`):
/// 1 where it does not split.
#[must_use]
pub fn splitk(
    n: i32,
    bm: i32,
    padded: i32,
    k: i32,
    group: i32,
    bits: i32,
    limit: Option<(u32, u32)>,
) -> i32 {
    let Some((fill_tg, max)) = limit else {
        return 1;
    };
    if bm != BM_RUNGS[0] || n <= 0 || k <= 0 || n % SPLITK_BN != 0 || !(bits == 4 || bits == 8) {
        return 1;
    }
    let tiles = (n / SPLITK_BN).unsigned_abs() * (padded / bm).max(1).unsigned_abs();
    if tiles == 0 || tiles >= fill_tg {
        return 1;
    }
    let mut split = i32::try_from((fill_tg / tiles).next_power_of_two())
        .unwrap_or(1)
        .clamp(1, i32::try_from(max).unwrap_or(1));
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
    // A prompt-sized batch takes the widest tile that pads it by at most a
    // sixteenth: a few rows past a whole number of 64-row tiles would
    // otherwise run a whole tile more.
    let snug = |(rung, padded): (i32, i32)| {
        (rows < PROMPT_ROWS || padded - rows <= rows / 16).then_some((rung, padded))
    };
    BM_RUNGS
        .iter()
        .rev()
        .copied()
        .filter(|rung| rows >= *rung)
        .find_map(|rung| fits(rung).and_then(snug))
        .or_else(|| {
            BM_RUNGS
                .iter()
                .rev()
                .copied()
                .filter(|rung| rows >= *rung)
                .find_map(fits)
        })
        .or_else(|| fits(BM_RUNGS[0]))
}

/// The rows from which [`mb_block`] weighs a tile's padding.
const PROMPT_ROWS: i32 = 256;

/// The widest row tile up to `widen_to` that still fills, for a padded
/// matmul now tiled at `bm`.
#[must_use]
pub fn widen_rung(bm: i32, padded: i32, widen_to: i32, fills: impl Fn(i32) -> bool) -> i32 {
    BM_RUNGS
        .iter()
        .copied()
        .filter(|rung| *rung > bm && *rung <= widen_to && *rung == padded)
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
                "the column count is {n}, not a multiple of {bn}: `qmm_t.metal` \
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

/// Folds the `split` stacked f32 partials of an `m x n` product into `y`, over `cols` columns.
pub(super) fn splitk_reduce(
    ctx: &Ctx<'_>,
    op: &'static str,
    y: Tensor,
    partial: Tensor,
    (m, n, cols): (i32, i32, i32),
    split: i32,
) -> Result<(), Error> {
    let stride = m
        .checked_mul(n)
        .ok_or_else(|| refuse(op, format!("{m} x {n} partials will not stack")))?;
    let mut reduce = vec![ctx.absent()?; 4];
    reduce.extend([
        y.arg_mut(),
        ctx.absent()?,
        n.arg(),
        ctx.absent()?,
        partial.arg(),
        ctx.absent()?,
        stride.arg(),
        split.arg(),
    ]);
    ctx.fire(
        Fire::at(QMM_FILE, SPLITK_REDUCE).apply(Grid::of(
            [cols.unsigned_abs(), m.unsigned_abs(), 1],
            [256, 1, 1],
        )),
        &reduce,
    )
}

/// The `Qmm::Tiled` rung: whether it fired.
pub(super) fn fire(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    op: &'static str,
    (act, w, biases, y): (Tensor, Bank, Tensor, Tensor),
    scratch: &Scratch<'_>,
    (crossover_tg, capacity_rows, fp16): (u32, u32, bool),
    (m, n, k, cols, ld, group, bits): (i32, i32, i32, i32, i32, i32, i32),
) -> Result<bool, Error> {
    let crossover = i32::try_from(crossover_tg).unwrap_or(i32::MAX);
    let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
    let fold_widest = *QMV_ROW_RUNGS.last().expect("a rung");
    let tile_fills = |padded: i32, bm: i32| {
        let tiles = u64::from(cols.unsigned_abs().div_ceil(BN_RUNGS[0].unsigned_abs()))
            * u64::from((padded / bm).unsigned_abs());
        tiles >= u64::from(crossover.unsigned_abs())
    };
    let split_plane = |padded: i32, bm: i32| -> Option<(i32, Tensor)> {
        let split = splitk(cols, bm, padded, k, group, bits, tuning.qmm_splitk());
        if split <= 1 {
            return None;
        }
        let rows = padded.checked_mul(split)?.unsigned_abs();
        (scratch.partials)(rows, n.unsigned_abs()).map(|plane| (split, plane))
    };
    if let Some((bm, padded)) = mb_block(m, capacity)
        && (m > fold_widest || tile_fills(padded, bm) || split_plane(padded, bm).is_some())
    {
        let widen_to = i32::try_from(tuning.qmm_widen()).unwrap_or(32);
        let bm = widen_rung(bm, padded, widen_to, |rung| tile_fills(padded, rung));
        if fp16
            && bm >= i32::try_from(tuning.qmm_precast_min_bm()).unwrap_or(16)
            && k % PRECAST_BK == 0
            && let Some(staged) = (scratch.precast)(padded.unsigned_abs(), k.unsigned_abs())
            && let Some(bn) = bn_unsplit(cols, padded / bm, crossover)
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
            gemm.push(ctx.absent()?);
            gemm.push(ld.arg());
            return ctx
                .fire(
                    Fire::at(QMM_FILE, precast_point(op, "", bm, bn)?).apply(Grid::of(
                        qmm_grid(op, cols, bn, padded, bm, 1)?,
                        qmm_group(bm),
                    )),
                    &gemm,
                )
                .map(|()| true);
        }
        if let Some((split, partials)) = split_plane(padded, bm) {
            let stride = padded
                .checked_mul(n)
                .ok_or_else(|| refuse(op, format!("{padded} x {n} partials will not stack")))?;
            ctx.fire(
                Fire::at(QMM_FILE, splitk_point(op, group, bits, bm)?).apply(Grid::of(
                    qmm_grid(op, cols, SPLITK_BN, padded, bm, split)?,
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
                    ctx.absent()?,
                    ctx.absent()?,
                    ctx.absent()?,
                    ld.arg(),
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
            return ctx
                .fire(
                    Fire::at(QMM_FILE, SPLITK_REDUCE).apply(Grid::of(
                        [cols.unsigned_abs(), m.unsigned_abs(), 1],
                        [256, 1, 1],
                    )),
                    &reduce,
                )
                .map(|()| true);
        }
        if let Some(bn) = bn_unsplit(cols, padded / bm, crossover) {
            let (form, stamp) = tiled_form(tuning.wide_range());
            let point = qmm_point(op, form, stamp, group, bits, bm, bn)?;
            let mut tiled = vec![
                w.codes.arg(),
                w.scales.arg(),
                biases.arg(),
                act.arg(),
                y.arg_mut(),
                k.arg(),
                n.arg(),
            ];
            for _ in 7..14 {
                tiled.push(ctx.absent()?);
            }
            tiled.push(ld.arg());
            return ctx
                .fire(
                    Fire::at(QMM_FILE, point.entry)
                        .stamp(point.stamp)
                        .apply(Grid::of(
                            qmm_grid(op, cols, bn, padded, bm, 1)?,
                            qmm_group(bm),
                        )),
                    &tiled,
                )
                .map(|()| true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_every_case() {
        a_batch_between_rungs_takes_the_wider_tile();
        the_split_follows_the_tile_count();
        the_precast_points_are_stamped_at_g64_b4_alone();
    }

    fn a_batch_between_rungs_takes_the_wider_tile() {
        let fills = |_: i32| true;
        assert_eq!(mb_block(12, 64), Some((8, 16)));
        assert_eq!(widen_rung(8, 16, 32, fills), 16);
        assert_eq!(mb_block(24, 64), Some((16, 32)));
        assert_eq!(widen_rung(16, 32, 32, fills), 32);
        assert_eq!(mb_block(48, 64), Some((32, 64)));
        assert_eq!(widen_rung(32, 64, 32, fills), 32);
        assert_eq!(widen_rung(16, 16, 32, fills), 16);
        assert_eq!(widen_rung(8, 8, 32, fills), 8);
        assert_eq!(widen_rung(8, 16, 32, |_| false), 8);
        assert_eq!(mb_block(512, 1024), Some((64, 512)));
        assert_eq!(mb_block(525, 1024), Some((32, 544)));
        assert_eq!(mb_block(560, 1024), Some((64, 576)));
    }

    fn the_split_follows_the_tile_count() {
        assert_eq!(splitk(17408, 8, 8, 5120, 64, 4, Some((512, 8))), 1);
        assert_eq!(splitk(5120, 8, 8, 5120, 64, 4, Some((512, 8))), 4);
        assert_eq!(splitk(5120, 8, 8, 17408, 64, 4, Some((512, 8))), 4);
        assert_eq!(splitk(1024, 8, 8, 5120, 64, 4, Some((512, 8))), 8);
        assert_eq!(splitk(1024, 16, 16, 5120, 64, 4, Some((512, 8))), 1);
        assert_eq!(splitk(1024, 8, 8, 5120, 64, 2, Some((512, 8))), 1);
        assert_eq!(splitk(1024, 8, 8, 5120, 128, 4, Some((512, 8))), 8);
        assert_eq!(splitk(1024, 8, 8, 576, 64, 4, Some((512, 8))), 1);
        assert_eq!(splitk(1000, 8, 8, 5120, 64, 4, Some((512, 8))), 1);
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
