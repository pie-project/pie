#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

const QMM_FILE: &str = "quant/qmm_t.slang";
const QMV_FILE: &str = "quant/qmv.slang";

const GROUPS: [i32; 3] = [32, 64, 128];

const WIDTHS: [i32; 3] = [2, 4, 8];

#[allow(dead_code)]
pub(crate) fn qmm_stamps_width(bits: u32) -> bool {
    i32::try_from(bits).is_ok_and(|bits| WIDTHS.contains(&bits))
}

const TILES: [i32; 2] = [32, 16];

const ROW_TILES: [i32; 4] = [64, 32, 16, 8];

const FRAG_ROWS: i32 = 8;

#[must_use]
pub fn qmm_group(bm: i32) -> [u32; 3] {
    [32, 2, if bm < 2 * FRAG_ROWS { 1 } else { 2 }]
}

const QMV_GROUP: [u32; 3] = [32, 4, 1];

const QMV_MAX_VECS: i32 = 4;

const QMV_CHUNK: i32 = 32;

const QMM_BK: i32 = 32;

const BM_RUNGS: [i32; 4] = [8, 16, 32, 64];

const BN_RUNGS: [i32; 2] = [16, 32];

#[must_use]
pub fn composed() -> Vec<Fire> {
    let mut out = Vec::new();
    for &gs in &GROUPS {
        for &b in &WIDTHS {
            out.push(Fire::at(
                QMV_FILE,
                qmv_name("quant.qmv", "", gs, b).expect("an axis point"),
            ));
            for &bm in &ROW_TILES {
                for &bn in &TILES {
                    out.push(Fire::at(
                        QMM_FILE,
                        qmm_name("quant.qmm_t", "", gs, b, bm, bn).expect("an axis point"),
                    ));
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
    let _ = stamp;
    Ok(Point {
        entry: qmm_name(op, form, group, bits, bm, bn)?,
        stamp: "",
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
    if !form.is_empty() {
        return Err(refuse(
            op,
            format!("no tiled point is instantiated in the `{form}` form"),
        ));
    }
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &ROW_TILES, bm, "row tile")?;
    check(op, &TILES, bn, "column tile")?;
    Ok(symbol(&format!(
        "affine_qmm_t_bf16_gs_{group}_b_{bits}_bm_{bm}_bn_{bn}"
    )))
}

pub fn qmm_precast_name(
    op: &'static str,
    before: &str,
    after: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    let _ = (before, after, bm, bn);
    Err(Error::Unsupported { op })
}

pub fn qmv_point(op: &'static str, form: &str, group: i32, bits: i32) -> Result<Point, Error> {
    Ok(Point {
        entry: qmv_name(op, form, group, bits)?,
        stamp: "",
    })
}

pub fn qmv_name(
    op: &'static str,
    form: &str,
    group: i32,
    bits: i32,
) -> Result<&'static str, Error> {
    if !form.is_empty() && form != "fast" {
        return Err(refuse(
            op,
            format!("no vector point is instantiated in the `{form}` form"),
        ));
    }
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    Ok(symbol(&format!("affine_qmv_bf16_gs_{group}_b_{bits}")))
}

pub fn qmv_rows_point(
    op: &'static str,
    group: i32,
    bits: i32,
    rows: i32,
    packs: i32,
) -> Result<Point, Error> {
    let _ = (group, bits, rows, packs);
    Err(Error::Unsupported { op })
}

pub fn precast_point(
    op: &'static str,
    form: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    let _ = (form, bm, bn);
    Err(Error::Unsupported { op })
}

pub fn routed_fp16_point(op: &'static str, bm: i32, bn: i32) -> Result<&'static str, Error> {
    let _ = (bm, bn);
    Err(Error::Unsupported { op })
}

pub fn mxfp4_routed_point(
    op: &'static str,
    form: &str,
    bm: i32,
    bn: i32,
) -> Result<&'static str, Error> {
    let _ = (form, bm, bn);
    Err(Error::Unsupported { op })
}

fn check(op: &'static str, points: &[i32], v: i32, what: &'static str) -> Result<(), Error> {
    points
        .contains(&v)
        .then_some(())
        .ok_or_else(|| refuse(op, format!("no point is instantiated at {what} {v}")))
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
}

pub const QMM_CM_TILE: u32 = 64;

pub fn qmm_cm_name(op: &'static str, group: i32, bits: i32) -> Result<&'static str, Error> {
    Ok(match (group, bits) {
        (32, 2) => "affine_qmm_t_cm_bf16_gs_32_b_2",
        (32, 4) => "affine_qmm_t_cm_bf16_gs_32_b_4",
        (32, 8) => "affine_qmm_t_cm_bf16_gs_32_b_8",
        (64, 2) => "affine_qmm_t_cm_bf16_gs_64_b_2",
        (64, 4) => "affine_qmm_t_cm_bf16_gs_64_b_4",
        (64, 8) => "affine_qmm_t_cm_bf16_gs_64_b_8",
        (128, 2) => "affine_qmm_t_cm_bf16_gs_128_b_2",
        (128, 4) => "affine_qmm_t_cm_bf16_gs_128_b_4",
        (128, 8) => "affine_qmm_t_cm_bf16_gs_128_b_8",
        _ => {
            return Err(refuse(
                op,
                format!("no coopmat tile is instantiated for {bits}-bit groups of {group}"),
            ));
        }
    })
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

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
    scratch: Scratch<'_>,
    capacity_rows: u32,
) -> Result<(), Error> {
    let _ = scratch;
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
    let tuned = crate::tuning::current();
    let min_batch = i32::try_from(tuned.qmm_min_batch(false, false)).unwrap_or(i32::MAX);

    if crate::tuning::device().coopmat && m >= min_batch && k % QMM_BK == 0 {
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
    let crossover = i32::try_from(tuned.qmm_bn_crossover_tg).unwrap_or(i32::MAX);
    let capacity = i32::try_from(capacity_rows).unwrap_or(i32::MAX);
    if m >= min_batch
        && k % QMM_BK == 0
        && let Some((bm, padded)) = mb_block(m, capacity)
        && let Some(bn) = bn_unsplit(n, padded / bm, crossover)
    {
        return ctx.fire(
            Fire::at(QMM_FILE, qmm_name(op, "", group, bits, bm, bn)?)
                .apply(Grid::of(qmm_grid(op, n, bn, padded, bm, 1)?, qmm_group(bm))),
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
    let chunk = if bits == 8 { 16 } else { QMV_CHUNK };
    if k % chunk != 0 {
        return Err(refuse(
            op,
            format!(
                "the vector point walks the contraction in {chunk}-code chunks; this \
                 batch is {m} x {k}"
            ),
        ));
    }

    let vec_blocks = (m as u32).div_ceil(QMV_MAX_VECS as u32);
    ctx.fire(
        Fire::at(QMV_FILE, qmv_name(op, "", group, bits)?).apply(Grid::of(
            [32 * (n as u32).div_ceil(8), 4 * vec_blocks, 1],
            QMV_GROUP,
        )),
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
