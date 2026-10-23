//! Group norm over each clip's voxels.

use crate::conv::{boxes, k, lanes_of, nonzero, place, spread, zero_invalid};
use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Combine, Elem, Fold, Val};
use crate::tensor::Tensor;
use dtype::Dtype;

/// Group norm per clip: each clip's `(group)` moments over all its voxels
/// and the group's channels, in f32, then `(x - mean) * rsqrt(var + eps) *
/// weight + bias` (f32 `[C]` planes), optionally `silu`; a row no lane holds
/// lands zero. kernels-cuda `elemwise::norm::group_norm`.
pub fn group_norm(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    groups: u32,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    silu: bool,
    y: Tensor,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let (rows, c, g) = (i64::from(x.rows), i64::from(x.width), i64::from(groups));
        let cg = c / g;
        let l = i64::from(lanes);
        let gv = cx.read(grid)?;
        let p = place(cx, gv, rows)?;
        let far = k(cx, p.lane, i64::from(i32::MAX));
        let lane = cx.select(p.valid, p.lane, far)?;
        let bx = boxes(cx, gv)?;
        let vox = bx.voxels(cx)?;
        let vox = cx.convert(vox, Elem::F32);
        let count = cx.scale(vox, cg as f64)?;
        let count = spread(cx, count, &[l, g], &[0])?;
        let one = cx.like_f(count, 1.0);
        let denom = cx.max(count, one)?;

        let xv = cx.read_f32(x)?;
        let per_group = |cx: &mut Cx<'_>, v: Val| -> Result<Val, Error> {
            let v = cx.reshape(v, &[rows, g, cg])?;
            let s = cx.reduce(v, &[2], Fold::Sum)?;
            let zero = cx.const_f(Elem::F32, 0.0, &[l, g]);
            let s = cx.put_rows(zero, lane, s, Combine::Add)?;
            Ok(cx.div(s, denom)?)
        };
        let back = |cx: &mut Cx<'_>, m: Val| -> Result<Val, Error> {
            let m = cx.take_rows(m, p.lane)?;
            let m = spread(cx, m, &[rows, g, cg], &[0, 1])?;
            Ok(cx.reshape(m, &[rows, c])?)
        };
        let mean = per_group(cx, xv)?;
        let mean_r = back(cx, mean)?;
        let d = cx.sub(xv, mean_r)?;
        let sq = cx.mul(d, d)?;
        let var = per_group(cx, sq)?;
        let var = cx.offset(var, f64::from(eps))?;
        let rstd = cx.rsqrt(var);
        let rstd_r = back(cx, rstd)?;
        let n = cx.mul(d, rstd_r)?;
        let wv = cx.read(weight)?;
        let wv = cx.reshape(wv, &[c])?;
        let wv = spread(cx, wv, &[rows, c], &[1])?;
        let bv = cx.read(bias)?;
        let bv = cx.reshape(bv, &[c])?;
        let bv = spread(cx, bv, &[rows, c], &[1])?;
        let v = cx.mul(n, wv)?;
        let v = cx.add(v, bv)?;
        let v = if silu { cx.silu(v)? } else { v };
        let v = zero_invalid(cx, v, p.valid)?;
        cx.write(y, v)
    })
}
