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
