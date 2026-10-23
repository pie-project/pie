//! The matrix-vector kernels over a quantized bank (`linear/qmv.metal`): the
//! `kernel.qmm` ladder's last rung, which serves every shape; folded over a
//! few rows a launch in [`rows`]; and the ternary PTQ1_0 bank's in [`ptq1_0`].

pub mod ptq1_0;
pub mod rows;

use crate::encode::{Arg, Ctx, Fire, Grid, refuse};
use crate::error::Error;
use crate::linear::qmm::{GROUPS, Point, WIDTHS, check, symbol};
use crate::tensor::{Bank, Tensor};

const QMV_FILE: &str = "linear/qmv.metal";

pub(crate) const QMV_GROUP: [u32; 3] = [32, 2, 1];

pub fn qmv_point(op: &'static str, form: &str, group: i32, bits: i32) -> Result<Point, Error> {
    check(op, &GROUPS, group, "group size")?;
    check(op, &WIDTHS, bits, "bit width")?;
    Ok(Point {
        entry: symbol(&format!("affine_qmv_{form}_bfloat16_gs_{group}_b_{bits}")),
        stamp: "",
    })
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

/// The `Qmm::Qmv` rung: the rows folded a launch ([`rows`]) where that fills
/// the device, else the plain matrix-vector kernel.
pub(crate) fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    (act, w, biases, y): (Tensor, Bank, Tensor, Tensor),
    (rows_max, packs, crossover_tg): (u32, u32, u32),
    (m, n, k, cols, ld, group, bits): (i32, i32, i32, i32, i32, i32, i32),
) -> Result<(), Error> {
    let crossover = i32::try_from(crossover_tg).unwrap_or(i32::MAX);
    let rows_max = i32::try_from(rows_max).unwrap_or(1);
    let packs = i32::try_from(packs).unwrap_or(rows::PACK_RUNGS[1]);
    if let Some(fold) = rows::qmv_rows_fold(m, cols, rows_max, crossover, packs, bits) {
        let point = rows::qmv_rows_point(op, group, bits, fold, packs)?;
        return ctx.fire(
            Fire::at(rows::QMV_ROWS_FILE, point.entry)
                .stamp(point.stamp)
                .apply(Grid::of(rows::qmv_rows_grid(op, m, fold, cols)?, QMV_GROUP)),
            &[
                w.codes.arg(),
                w.scales.arg(),
                biases.arg(),
                act.arg(),
                y.arg_mut(),
                k.arg(),
                n.arg(),
                m.arg(),
                ld.arg(),
            ],
        );
    }
    let point = qmv_point(op, "fast", group, bits)?;
    let mut fast = vec![
        w.codes.arg(),
        w.scales.arg(),
        biases.arg(),
        act.arg(),
        y.arg_mut(),
        k.arg(),
        n.arg(),
    ];
    for _ in 7..12 {
        fast.push(ctx.absent()?);
    }
    fast.push(ld.arg());
    ctx.fire(
        Fire::at(QMV_FILE, point.entry).apply(Grid::of(qmv_grid(op, m, cols)?, QMV_GROUP)),
        &fast,
    )
}
