//! The lane table a grid rule derives from another (`derive_grid`).

use crate::conv::{GridRule, add_k, boxes, cmp_k, div_k, exclusive, k, lane_pair, mul_k, rem_k};
use crate::error::Error;
use crate::hlo::{Cmp, Elem, Func, Val};
use crate::tensor::{Landing, Operand};

/// Derives the output lane table of `rule` from `grid`: each clip's box
/// mapped, a box the rule cannot map zeroed, offsets packed in lane order.
pub fn derive_grid(f: &mut Func, grid: Operand, rule: GridRule, y: Landing) -> Result<Val, Error> {
    const OP: &str = "conv.grid";
    lane_pair(OP, grid, y)?;
    let g = grid.val;
    let b = boxes(f, g)?;
    let (t, h, w) = (b.t, b.h, b.w);
    let lanes = i64::from(grid.rows);
    let yes = f.const_i(Elem::Pred, 1, &[lanes]);
    let (ot, oh, ow, ok) = match rule {
        GridRule::Conv {
            k: kk,
            stride,
            pad,
            pad_back,
            causal_t,
        } => {
            let back_t = if causal_t { 0 } else { pad_back[0] };
            let mut outs = Vec::new();
            let mut ok = yes;
            for (n, i, back) in [(t, 0, back_t), (h, 1, pad_back[1]), (w, 2, pad_back[2])] {
                let span = add_k(f, n, i64::from(pad[i]) + i64::from(back) - i64::from(kk[i]))?;
                let fine = cmp_k(f, Cmp::Ge, span, 0)?;
                ok = f.and(ok, fine)?;
                let o = div_k(f, span, i64::from(stride[i].max(1)))?;
                outs.push(add_k(f, o, 1)?);
            }
            (outs[0], outs[1], outs[2], ok)
        }
        GridRule::Upsample {
            factor,
            keep_first_frame,
        } => {
            let plain = mul_k(f, t, i64::from(factor[0]))?;
            let ot = if keep_first_frame {
                let tm = add_k(f, t, -1)?;
                let kept = mul_k(f, tm, i64::from(factor[0]))?;
                let kept = add_k(f, kept, 1)?;
                let pos = cmp_k(f, Cmp::Gt, t, 0)?;
                f.select(pos, kept, plain)?
            } else {
                plain
            };
            let oh = mul_k(f, h, i64::from(factor[1]))?;
            let ow = mul_k(f, w, i64::from(factor[2]))?;
            (ot, oh, ow, yes)
        }
        GridRule::Shuffle { r, trim_t } => {
            let ot = mul_k(f, t, i64::from(r[0]))?;
            let ot = add_k(f, ot, -i64::from(trim_t))?;
            let ok = cmp_k(f, Cmp::Gt, ot, 0)?;
            let oh = mul_k(f, h, i64::from(r[1]))?;
            let ow = mul_k(f, w, i64::from(r[2]))?;
            (ot, oh, ow, ok)
        }
        GridRule::Unshuffle { r } | GridRule::AvgDown { factor: r } => {
            let avg = matches!(rule, GridRule::AvgDown { .. });
            if r.contains(&0) {
                let no = f.not(yes);
                (t, h, w, no)
            } else {
                let mut ok = yes;
                let axes: &[(Val, u32)] = if avg {
                    &[(h, r[1]), (w, r[2])]
                } else {
                    &[(t, r[0]), (h, r[1]), (w, r[2])]
                };
                for &(n, fq) in axes {
                    let m = rem_k(f, n, i64::from(fq))?;
                    let z = cmp_k(f, Cmp::Eq, m, 0)?;
                    ok = f.and(ok, z)?;
                }
                let ot = if avg {
                    let up = add_k(f, t, i64::from(r[0]) - 1)?;
                    div_k(f, up, i64::from(r[0]))?
                } else {
                    div_k(f, t, i64::from(r[0]))?
                };
                let oh = div_k(f, h, i64::from(r[1]))?;
                let ow = div_k(f, w, i64::from(r[2]))?;
                (ot, oh, ow, ok)
            }
        }
    };
    let zero = k(f, t, 0);
    let ot = f.select(ok, ot, zero)?;
    let oh = f.select(ok, oh, zero)?;
    let ow = f.select(ok, ow, zero)?;
    let vox = f.mul(ot, oh)?;
    let vox = f.mul(vox, ow)?;
    let off = exclusive(f, vox)?;
    let mut cols = Vec::new();
    for c in [ot, oh, ow, off] {
        cols.push(f.reshape(c, &[lanes, 1])?);
    }
    let out = f.concat(&cols, 1)?;
    Ok(out)
}

/// The first `rows` entries of a one-per-row table, `[rows]`.
fn per_row(f: &mut Func, v: Val, rows: i64) -> Result<Val, Error> {
    let n: i64 = f.dims(v).iter().product();
    let flat = f.reshape(v, &[n])?;
    Ok(f.slice_axis(flat, 0, 0, rows)?)
}

/// Each token row's cell, `[rows, max(3, axis + 1)]` f32
/// (`ops::conv::cells`): a row of a clip of `tgrid` its patch's `(t, h, w)`
/// there, any other row `positions[row]` in column `axis`; zeros
/// elsewhere. kernels-cuda `elemwise::posenc::cells`.
pub fn cells(
    f: &mut Func,
    tgrid: Operand,
    positions: Operand,
    axis: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "conv.cells";
    if y.width != axis.max(2) + 1 || positions.rows < y.rows {
        return Err(crate::error::refuse(
            OP,
            format!(
                "{} cell columns over {} positions for {} rows; a cell is max(3, axis + 1) \
                 columns and each row reads its position",
                y.width, positions.rows, y.rows
            ),
        ));
    }
    let rows = i64::from(y.rows);
    let at = crate::conv::place(f, tgrid.val, rows)?;
    let pos = per_row(f, positions.val, rows)?;
    let zero = k(f, at.v.t, 0);
    let mut cols = Vec::new();
    for c in 0..y.width {
        let inside = match c {
            0 => at.v.t,
            1 => at.v.h,
            2 => at.v.w,
            _ => zero,
        };
        let outside = if c == axis { pos } else { zero };
        let c = f.select(at.valid, inside, outside)?;
        let c = f.convert(c, Elem::F32);
        cols.push(f.reshape(c, &[rows, 1])?);
    }
    Ok(f.concat(&cols, 1)?)
}

/// Each row's count of the rows `csr` packs in its lane's group, `[rows,
/// 1]` f32 (`ops::gather::group_rows`): `csr[g + 1] - csr[g]` for `g =
/// groups[lanes[row]]`. kernels-cuda `elemwise::posenc::group_rows`.
pub fn group_rows(
    f: &mut Func,
    csr: Operand,
    groups: Operand,
    lanes: Operand,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "gather.group_rows";
    if y.width != 1 || lanes.rows < y.rows || csr.rows == 0 {
        return Err(crate::error::refuse(
            OP,
            "one count column, a lane for each row and an indptr",
        ));
    }
    let rows = i64::from(y.rows);
    let lane_count = i64::from(groups.rows);
    let groups_v = per_row(f, groups.val, lane_count)?;
    let csr_v = per_row(f, csr.val, i64::from(csr.rows))?;
    let lane = per_row(f, lanes.val, rows)?;
    let lo = cmp_k(f, Cmp::Ge, lane, 0)?;
    let hi = cmp_k(f, Cmp::Lt, lane, lane_count)?;
    let ok = f.and(lo, hi)?;
    let zero = k(f, lane, 0);
    let lane = f.select(ok, lane, zero)?;
    let g = super::take1(f, groups_v, lane)?;
    let glo = cmp_k(f, Cmp::Ge, g, 0)?;
    let ghi = cmp_k(f, Cmp::Lt, g, i64::from(csr.rows) - 1)?;
    let gok = f.and(glo, ghi)?;
    let ok = f.and(ok, gok)?;
    let gz = k(f, g, 0);
    let g = f.select(ok, g, gz)?;
    let g1 = add_k(f, g, 1)?;
    let a = super::take1(f, csr_v, g)?;
    let b = super::take1(f, csr_v, g1)?;
    let n = f.sub(b, a)?;
    let nz = k(f, n, 0);
    let n = f.select(ok, n, nz)?;
    let n = f.convert(n, Elem::F32);
    Ok(f.reshape(n, &[rows, 1])?)
}
