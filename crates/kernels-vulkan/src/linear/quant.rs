#![allow(clippy::too_many_arguments)]

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};
use crate::tuning::Qmm;

const QMM_FILE: &str = "quant/qmm_t.slang";
const QMM_RT_FILE: &str = "quant/qmm_rt.slang";
const QMV1_FILE: &str = "quant/qmv1.slang";
const QMV_FILE: &str = "quant/qmv.slang";

const GROUPS: [i32; 3] = [32, 64, 128];

const WIDTHS: [i32; 3] = [2, 4, 8];

const TILES: [i32; 2] = [32, 16];

const ROW_TILES: [i32; 4] = [64, 32, 16, 8];

const FRAG_ROWS: i32 = 8;

fn qmm_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm < 2 * FRAG_ROWS { 1 } else { 2 }]
}

const QMV_GROUP: [u32; 3] = [32, 4, 1];

const QMV_CHUNK: i32 = 32;

pub(crate) const QMM_BK: i32 = 32;

const BM_RUNGS: [i32; 4] = [8, 16, 32, 64];

const BN_RUNGS: [i32; 2] = [16, 32];

pub(crate) fn qmm_stamped(group: i64, bits: i64) -> bool {
    i32::try_from(group).is_ok_and(|g| GROUPS.contains(&g))
        && i32::try_from(bits).is_ok_and(|b| WIDTHS.contains(&b))
}

fn qmm_name(
    op: &'static str,
    group: i32,
    bits: i32,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_bf16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    )))
}

/// The register-tiled point's column tile.
const RT_COLS: u32 = 64;

fn qmm_rt_name(group: i32, bits: i32, bm: i32, k16: bool) -> &'static str {
    let block = if k16 { "_k16" } else { "" };
    symbol(&format!(
        "affine_qmm_rt{block}_bf16_gs_{group}_b_{bits}_bm_{bm}"
    ))
}

/// The one-row point loads 128 bits of codes at once, which must stay
/// inside one scale group.
pub(crate) fn qmv1_fits(group: i64, bits: i64) -> bool {
    matches!(bits, 4 | 8) && 128 / bits <= group
}

fn qmv1_name(group: i32, bits: i32, lanes: u32, rows: u32) -> &'static str {
    if lanes == 32 && rows == 1 {
        symbol(&format!("affine_qmv1_bf16_gs_{group}_b_{bits}"))
    } else {
        symbol(&format!(
            "affine_qmv1_bf16_gs_{group}_b_{bits}_l{lanes}_r{rows}"
        ))
    }
}

/// The vector point at `vecs` activation rows a workgroup: 4 carries the
/// plain name, 1, 2 and 8 a `_v` suffix.
fn qmv_name(op: &'static str, group: i32, bits: i32, vecs: u32) -> Result<&'static str, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    if !matches!(vecs, 1 | 2 | 4 | 8) {
        return Err(refuse(
            op,
            format!("no vector point reads {vecs} rows a workgroup"),
        ));
    }
    Ok(symbol(&if vecs == 4 {
        format!("affine_qmv_bf16_gs_{group}_b_{bits}")
    } else {
        format!("affine_qmv_bf16_gs_{group}_b_{bits}_v{vecs}")
    }))
}

fn check(op: &'static str, points: &[i32], v: i32, what: &'static str) -> Result<(), Error> {
    points
        .contains(&v)
        .then_some(())
        .ok_or_else(|| refuse(op, format!("no point is instantiated at {what} {v}")))
}

fn mb_block(rows: i32, capacity: i32) -> Option<(i32, i32)> {
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

fn bn_unsplit(out_width: i32, row_tiles: i32, crossover_tg: i32) -> Option<i32> {
    if out_width % BN_RUNGS[0] != 0 {
        return None;
    }
    if out_width % 32 == 0 && (out_width / 32) * row_tiles.max(1) >= crossover_tg {
        return Some(32);
    }
    Some(BN_RUNGS[0])
}

fn qmm_grid(op: &'static str, n: i32, bn: i32, m: i32, bm: i32) -> Result<[u32; 3], Error> {
    if n <= 0 || m <= 0 || bn <= 0 || bm <= 0 {
        return Err(refuse(op, "a dimension of the tiled grid is zero"));
    }
    if m % bm != 0 {
        return Err(refuse(
            op,
            format!("the row count is {m}, not a multiple of {bm}: the grid is whole tiles"),
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
        lanes(1, group[2], "the k splits")?,
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

use crate::encode::symbol;

const QMM_CM_TILE: u32 = 64;

fn qmm_cm_name(op: &'static str, group: i32, bits: i32) -> Result<&'static str, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    Ok(symbol(&format!("affine_qmm_t_cm_bf16_gs_{group}_b_{bits}")))
}

/// The `kernel.qmm` variant an entry belongs to.
#[must_use]
pub fn ran(entry: &str) -> Option<&'static str> {
    if entry.starts_with("affine_qmm_t_cm_") {
        Some("coopmat")
    } else if entry.starts_with("affine_qmm_rt_") {
        Some("register_tiled")
    } else if entry.starts_with("affine_qmm_t_") {
        Some("tiled")
    } else if entry.starts_with("affine_qmv1_") {
        Some("one_row")
    } else if entry.starts_with("affine_qmv_") {
        Some("qmv")
    } else {
        None
    }
}

pub fn matmul(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
    act: Tensor,
    w: Bank,
    y: Tensor,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt(ctx, tuning, "linear.matmul", act, w, y, capacity_rows)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
    act: Tensor,
    w: Bank,
    y: Tensor,
    capacity_rows: u32,
) -> Result<(), Error> {
    act_x_wt(ctx, tuning, "linear.lm_head", act, w, y, capacity_rows)
}

/// Runs the first `kernel.qmm` rung that fits and can run; a tiled rung
/// declines where the row capacity cannot hold its padding.
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
    capacity_rows: u32,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });

    let Some(biases) = w.biases else {
        return Err(refuse(
            op,
            format!(
                "the weight is a symmetric {}-bit bank in groups of {}, and this plane \
                 instantiates no dense point for one",
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
                 groups: every point indexes its scales as `k / group`",
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
    let args = [
        w.codes.arg(),
        w.scales.arg(),
        biases.arg(),
        act.arg(),
        y.arg_mut(),
        k.arg(),
        n.arg(),
        m.arg(),
    ];
    let at = tuning.qmm_at(rows, columns, contraction, w.bits, w.group);
    for choice in tuning.qmm(&at) {
        match choice {
            Qmm::Coopmat => {
                let name = qmm_cm_name(op, group, bits)?;
                return ctx.fire(
                    Fire::at(QMM_FILE, name).apply(Grid::of(
                        [
                            (n as u32).div_ceil(QMM_CM_TILE) * 32,
                            (m as u32).div_ceil(QMM_CM_TILE) * 4,
                            1,
                        ],
                        [32, 4, 1],
                    )),
                    &args,
                );
            }
            Qmm::RegisterTiled { bm, k16 } => {
                let row_tiles = rows.div_ceil(bm);
                return ctx.fire(
                    Fire::at(QMM_RT_FILE, qmm_rt_name(group, bits, bm as i32, k16)).apply(
                        Grid::of(
                            [
                                columns.div_ceil(RT_COLS) * (RT_COLS / 4),
                                row_tiles * (bm / 4),
                                1,
                            ],
                            [RT_COLS / 4, bm / 4, 1],
                        ),
                    ),
                    &args,
                );
            }
            Qmm::Tiled { crossover_tg } => {
                let crossover = i32::try_from(crossover_tg).unwrap_or(i32::MAX);
                let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
                if let Some((bm, padded)) = mb_block(m, capacity)
                    && let Some(bn) = bn_unsplit(n, padded / bm, crossover)
                {
                    return ctx.fire(
                        Fire::at(QMM_FILE, qmm_name(op, group, bits, bm, bn)?)
                            .apply(Grid::of(qmm_grid(op, n, bn, padded, bm)?, qmm_group(bm))),
                        &args,
                    );
                }
            }
            Qmm::OneRow {
                lanes,
                rows: per_lane,
            } => {
                return ctx.fire(
                    Fire::at(QMV1_FILE, qmv1_name(group, bits, lanes, per_lane)).apply(Grid::of(
                        // 256 invocations land 256 / lanes rows, `per_lane` each.
                        [columns.div_ceil(256 / lanes * per_lane) * 64, 4, 1],
                        [64, 4, 1],
                    )),
                    &args,
                );
            }
            Qmm::Qmv { vecs } => {
                let chunk = if bits == 8 { 16 } else { QMV_CHUNK };
                if k % chunk != 0 {
                    return Err(refuse(
                        op,
                        format!(
                            "the vector point walks the contraction in {chunk}-code chunks; \
                             this batch is {m} x {k}"
                        ),
                    ));
                }
                let vec_blocks = rows.div_ceil(vecs);
                return ctx.fire(
                    Fire::at(QMV_FILE, qmv_name(op, group, bits, vecs)?).apply(Grid::of(
                        [32 * columns.div_ceil(8), 4 * vec_blocks, 1],
                        QMV_GROUP,
                    )),
                    &args,
                );
            }
        }
    }
    unreachable!("`kernel.qmm` ends on the vector kernel, which serves every shape")
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
