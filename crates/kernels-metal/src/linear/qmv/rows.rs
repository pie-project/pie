//! The matrix-vector kernel folded over a few rows a launch
//! (`linear/qmv_rows.metal`), for batches too small for a tile and too many
//! for one vector a launch.

use crate::encode::refuse;
use crate::error::Error;
use crate::linear::qmm::{GROUPS, Point, WIDTHS, check, symbol};

pub(crate) const QMV_ROWS_FILE: &str = "linear/qmv_rows.metal";

const QMV_ROWS_STAMP: &str = "PIE_STAMP_qmv_rows";

pub(crate) const QMV_ROW_RUNGS: [i32; 7] = [2, 3, 4, 5, 6, 7, 8];

const QMV_ROW_RUNGS_PACK2: [i32; 6] = [2, 3, 4, 6, 7, 8];

const QMV_ROW_RUNGS_PACK2_2BIT: [i32; 3] = [2, 4, 8];

const QMV_ROW_RUNGS_2BIT: [i32; 2] = [2, 3];

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

pub(crate) const PACK_RUNGS: [i32; 2] = [1, 2];

pub fn qmv_rows_point(
    op: &'static str,
    group: i32,
    bits: i32,
    rows: i32,
    packs: i32,
) -> Result<Point, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    check(op, &PACK_RUNGS, packs, "pack width")?;
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
}
