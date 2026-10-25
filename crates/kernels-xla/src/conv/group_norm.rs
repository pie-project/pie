//! Group norm over each clip's voxels.

use crate::conv::{boxes, k, lanes_of, nonzero, place, spread, zero_invalid};
use crate::error::{Error, refuse};
use crate::hlo::{Combine, Elem, Fold, Func, Val};
use crate::tensor::{Landing, Operand};
use dtype::Dtype;

/// Group norm per clip: each clip's `(group)` moments over all its voxels
/// and the group's channels, in f32, then `(x - mean) * rsqrt(var + eps) *
/// weight + bias` (f32 `[C]` planes), optionally `silu`; a row no lane holds
/// lands zero. kernels-cuda `elemwise::norm::group_norm`.
pub fn group_norm(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    groups: u32,
    weight: Operand,
    bias: Operand,
    eps: f32,
    silu: bool,
    y: Landing,
    base: u32,
) -> Result<Val, Error> {
    let grid = crate::conv::rebased(f, grid, base)?;
    const OP: &str = "conv.group_norm";
    if x.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: x.dtype,
        });
    }
    if y.rows != x.rows || y.width != x.width || y.dtype != x.dtype {
        return Err(refuse(OP, "the landing is one row per input row"));
    }
    let lanes = lanes_of(OP, "input", grid)?;
    nonzero(OP, "the channel count", x.width)?;
    nonzero(OP, "the group count", groups)?;
    if !x.width.is_multiple_of(groups) {
        return Err(refuse(
            OP,
            format!("{} channels do not divide into {groups} groups", x.width),
        ));
    }
    for (what, t) in [("weight", weight), ("bias", bias)] {
        if t.dtype != Dtype::F32 || t.elements() != u64::from(x.width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {}x{} {:?}; expected {} f32",
                    t.rows, t.width, t.dtype, x.width
                ),
            ));
        }
    }
    let (rows, c, g) = (i64::from(x.rows), i64::from(x.width), i64::from(groups));
    let cg = c / g;
    let l = i64::from(lanes);
    let gv = grid.val;
    let p = place(f, gv, rows)?;
    let far = k(f, p.lane, i64::from(i32::MAX));
    let lane = f.select(p.valid, p.lane, far)?;
    let bx = boxes(f, gv)?;
    let vox = bx.voxels(f)?;
    let vox = f.convert(vox, Elem::F32);
    let count = f.scale(vox, cg as f64)?;
    let count = spread(f, count, &[l, g], &[0])?;
    let one = f.like_f(count, 1.0);
    let denom = f.max(count, one)?;

    let xv = x.f32(f);
    let per_group = |f: &mut Func, v: Val| -> Result<Val, Error> {
        let v = f.reshape(v, &[rows, g, cg])?;
        let s = f.reduce(v, &[2], Fold::Sum)?;
        let zero = f.const_f(Elem::F32, 0.0, &[l, g]);
        let s = f.put_rows(zero, lane, s, Combine::Add)?;
        Ok(f.div(s, denom)?)
    };
    let back = |f: &mut Func, m: Val| -> Result<Val, Error> {
        let m = f.take_rows(m, p.lane)?;
        let m = spread(f, m, &[rows, g, cg], &[0, 1])?;
        Ok(f.reshape(m, &[rows, c])?)
    };
    let mean = per_group(f, xv)?;
    let mean_r = back(f, mean)?;
    let d = f.sub(xv, mean_r)?;
    let sq = f.mul(d, d)?;
    let var = per_group(f, sq)?;
    let var = f.offset(var, f64::from(eps))?;
    let rstd = f.rsqrt(var);
    let rstd_r = back(f, rstd)?;
    let n = f.mul(d, rstd_r)?;
    let wv = weight.val;
    let wv = f.reshape(wv, &[c])?;
    let wv = spread(f, wv, &[rows, c], &[1])?;
    let bv = bias.val;
    let bv = f.reshape(bv, &[c])?;
    let bv = spread(f, bv, &[rows, c], &[1])?;
    let v = f.mul(n, wv)?;
    let v = f.add(v, bv)?;
    let v = if silu { f.silu(v)? } else { v };
    let v = zero_invalid(f, v, p.valid)?;
    Ok(v)
}
