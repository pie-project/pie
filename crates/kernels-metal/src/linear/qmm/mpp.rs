//! The Metal 4 tensor-op matmul (`linear/qmm_mpp.metal`): the activation's
//! per-group input sums first, then tiles the profile sizes, split across
//! partitions for few rows and folded by the tiled kernel's reduce.

use crate::Tuning;
use crate::encode::{Arg, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

use super::tiled::{QMM_FILE, SPLITK_REDUCE};
use super::{Scratch, symbol};

/// The `Qmm::Mpp` rung: whether it fired.
#[allow(clippy::too_many_arguments)]
pub(super) fn fire(
    ctx: &Ctx<'_>,
    tuning: &Tuning,
    (act, w, biases, y): (Tensor, Bank, Tensor, Tensor),
    scratch: &Scratch<'_>,
    (rows, columns, stride, contraction): (u32, u32, u32, u32),
    capacity_rows: u32,
    (k, n, ld): (i32, i32, i32),
) -> Result<bool, Error> {
    let (bm, bn) = crate::tuning::mpp_tile_for(rows, columns);
    // A column cut lands part of a row, which the MPP kernel never does.
    if columns == stride {
        let padded = rows.div_ceil(bm) * bm;
        if padded <= capacity_rows
            && let Some(staged) = (scratch.precast)(padded, contraction)
        {
            const FILE: &str = "linear/qmm_mpp.metal";
            let aux = Tensor::new(staged.buf, padded, contraction / 64, dtype::Dtype::F32);
            ctx.fire(
                Fire::at(FILE, "qmm_mpp_input_sums")
                    .apply(Grid::of([32 * (contraction / 64), padded, 1], [32, 4, 1])),
                &[act.arg(), aux.arg_mut(), k.arg(), bm.arg()],
            )?;
            let crate::tuning::QmmMpp {
                split_rows,
                split_cols,
                pipeline_rows,
                pipeline_cols,
                linear_from,
            } = tuning.qmm_mpp_tiles();
            let partitions = if rows > split_rows { 2u32 } else { 4u32 };
            let partial = if (rows <= split_rows
                || (rows <= 2 * split_rows && columns <= split_cols))
                && columns <= 65536
                && contraction % 256 == 0
            {
                (scratch.partials)(padded * partitions, columns)
            } else {
                None
            };
            let split = if partial.is_some() { partitions } else { 1u32 };
            let packed = w.mpp_codes.is_some();
            let local = bm == 8 && packed && split == 4;
            let simdgroups = if bm == 8 { 2 } else { 4 };
            let pipeline = !local && rows <= pipeline_rows && columns <= pipeline_cols;
            let linear_mode = if rows >= linear_from && split == 1 && packed {
                16
            } else {
                -1
            };
            let kind = if packed { "packed" } else { "native" };
            let index = if packed
                && rows >= linear_from
                && split == 1
                && [
                    (padded, contraction),
                    (padded, columns),
                    (contraction, columns),
                ]
                .iter()
                .all(|&(a, b)| a.checked_mul(b).is_some())
            {
                "uint"
            } else {
                "ulong"
            };
            let groups_per_tg = simdgroups * if local { split } else { 1 };
            let entry = symbol(&format!(
                "affine_qmm_mpp_{kind}_m{bm}_n{bn}_s{simdgroups}_k{split}_p{pipeline}_l{local}_g{}_{index}",
                linear_mode + 1
            ));
            let stamp = symbol(&format!(
                "PIE_MPP_POINT(\"{entry}\",{bm},{bn},{simdgroups},{packed},false,{split},{pipeline},{local},{linear_mode},{index})"
            ));
            ctx.fire(
                Fire::at(FILE, entry).stamp(stamp).apply(Grid::of(
                    [
                        32 * groups_per_tg
                            * (columns / bn)
                            * if linear_mode >= 0 { padded / bm } else { 1 },
                        if linear_mode >= 0 { 1 } else { padded / bm },
                        if local { 1 } else { split },
                    ],
                    [32 * groups_per_tg, 1, 1],
                )),
                &[
                    if packed {
                        w.mpp_codes.unwrap().arg()
                    } else {
                        w.codes.arg()
                    },
                    w.scales.arg(),
                    biases.arg(),
                    act.arg(),
                    if local {
                        y.arg_mut()
                    } else {
                        partial.unwrap_or(y).arg_mut()
                    },
                    k.arg(),
                    n.arg(),
                    aux.arg(),
                    (if local { rows } else { padded } as i32).arg(),
                    ld.arg(),
                ],
            )?;
            if local {
                return Ok(true);
            }
            if let Some(partial) = partial {
                let stride = (padded * columns) as i32;
                let mut args = vec![ctx.absent()?; 4];
                args.extend([
                    y.arg_mut(),
                    ctx.absent()?,
                    n.arg(),
                    ctx.absent()?,
                    partial.arg(),
                    ctx.absent()?,
                    stride.arg(),
                    (split as i32).arg(),
                ]);
                return ctx
                    .fire(
                        Fire::at(QMM_FILE, SPLITK_REDUCE)
                            .apply(Grid::of([columns, rows, 1], [256, 1, 1])),
                        &args,
                    )
                    .map(|()| true);
            }
            return Ok(true);
        }
    }
    Ok(false)
}
