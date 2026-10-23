//! Resampling: nearest upsample, pixel (un)shuffle, average down, (un)patchify.

use crate::conv::{
    Boxes, Placed, add_k, boxes, cmp_k, div_k, k, lane_pair, movable, mul_k, place, ravel, rem_k,
    spread, take_elems, take_rows_or_zero, unit_where_invalid, volume,
};
use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Val};
use crate::tensor::Tensor;

/// Nearest upsampling per clip: output voxel `(t, h, w)` reads input
/// `(t / ft, h / fh, w / fw)`, or with `keep_first_frame` time `0 → 0` and
/// `t → (t - 1) / ft + 1`. kernels-cuda `conv::resample::upsample_nearest`.
pub fn upsample_nearest(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    keep_first_frame: bool,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.upsample";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    volume(OP, factor)?;
    if y.width != x.width {
        return Err(refuse(OP, "an upsample keeps the channel width"));
    }
    let [ft, fh, fw] = factor.map(i64::from);
    ctx.emit(&mut |cx| {
        let og = cx.read(y_grid)?;
        let o = place(cx, og, i64::from(y.rows))?;
        let ig = cx.read(grid)?;
        let ib = boxes(cx, ig)?;
        let inb = ib.at(cx, o.lane)?;
        let inb = unit_where_invalid(cx, inb, o.valid)?;
        let ti = if keep_first_frame {
            let tm = add_k(cx, o.v.t, -1)?;
            let tm = div_k(cx, tm, ft)?;
            let tm = add_k(cx, tm, 1)?;
            let first = cmp_k(cx, Cmp::Eq, o.v.t, 0)?;
            let z = k(cx, tm, 0);
            cx.select(first, z, tm)?
        } else {
            div_k(cx, o.v.t, ft)?
        };
        let hi = div_k(cx, o.v.h, fh)?;
        let wi = div_k(cx, o.v.w, fw)?;
        let src = ravel(cx, &inb, ti, hi, wi)?;
        let xv = cx.read(x)?;
        let rows = take_rows_or_zero(cx, xv, src, o.valid)?;
        cx.write(y, rows)
    })
}

/// Per-row source coordinates for the shuffles, `[rows]`.
struct Mapped {
    o: Placed,
    inb: Boxes,
}

fn mapped(cx: &mut Cx<'_>, grid: Tensor, y_grid: Tensor, rows: u32) -> Result<Mapped, Error> {
    let og = cx.read(y_grid)?;
    let o = place(cx, og, i64::from(rows))?;
    let ig = cx.read(grid)?;
    let ib = boxes(cx, ig)?;
    let inb = ib.at(cx, o.lane)?;
    let inb = unit_where_invalid(cx, inb, o.valid)?;
    Ok(Mapped { o, inb })
}

/// Depth-to-space per clip: output channel `c` of voxel `(t', h, w)` (with
/// `t' = t + trim_t`) reads input channel `c * R + ((t' % r1) * r2 + h % r2)
/// * r3 + w % r3` of voxel `(t' / r1, h / r2, w / r3)`, `R = r1 r2 r3`.
/// kernels-cuda `conv::resample::pixel_shuffle`.
pub fn pixel_shuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    trim_t: u32,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.pixel_shuffle";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    let vol = volume(OP, r)?;
    if !x.width.is_multiple_of(vol) || y.width != x.width / vol {
        return Err(refuse(
            OP,
            format!(
                "{} channels do not unpack as {} x {vol} output channels",
                x.width, y.width
            ),
        ));
    }
    let [r1, r2, r3] = r.map(i64::from);
    ctx.emit(&mut |cx| {
        let m = mapped(cx, grid, y_grid, y.rows)?;
        let tt = add_k(cx, m.o.v.t, i64::from(trim_t))?;
        let ti = div_k(cx, tt, r1)?;
        let hi = div_k(cx, m.o.v.h, r2)?;
        let wi = div_k(cx, m.o.v.w, r3)?;
        let src = ravel(cx, &m.inb, ti, hi, wi)?;
        let bt = rem_k(cx, tt, r1)?;
        let bt = mul_k(cx, bt, r2)?;
        let bh = rem_k(cx, m.o.v.h, r2)?;
        let blk = cx.add(bt, bh)?;
        let blk = mul_k(cx, blk, r3)?;
        let bw = rem_k(cx, m.o.v.w, r3)?;
        let blk = cx.add(blk, bw)?;
        let base = mul_k(cx, src, i64::from(x.width))?;
        let base = cx.add(base, blk)?;
        let dims = [i64::from(y.rows), i64::from(y.width)];
        let base = spread(cx, base, &dims, &[0])?;
        let col = cx.iota(Elem::I32, &dims, 1);
        let col = mul_k(cx, col, i64::from(vol))?;
        let idx = cx.add(base, col)?;
        let valid = spread(cx, m.o.valid, &dims, &[0])?;
        let xv = cx.read(x)?;
        let v = take_elems(cx, xv, idx, valid)?;
        cx.write(y, v)
    })
}

/// Space-to-depth per clip, the inverse of [`pixel_shuffle`] without trim:
/// output channel `cin * R + b` (`b = (i1 * r2 + i2) * r3 + i3`) of voxel
/// `(t, h, w)` reads channel `cin` of input voxel `(t r1 + i1, h r2 + i2,
/// w r3 + i3)`. kernels-cuda `conv::resample::pixel_unshuffle`.
pub fn pixel_unshuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.pixel_unshuffle";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    let vol = volume(OP, r)?;
    if u64::from(y.width) != u64::from(x.width) * u64::from(vol) {
        return Err(refuse(
            OP,
            format!(
                "{} channels do not pack as {} = C x {vol} output channels",
                x.width, y.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let m = mapped(cx, grid, y_grid, y.rows)?;
        let dims = [i64::from(y.rows), i64::from(y.width)];
        let (idx, valid) = unshuffle_index(cx, &m, r, i64::from(x.width), &dims, 1, None)?;
        let xv = cx.read(x)?;
        let v = take_elems(cx, xv, idx, valid)?;
        cx.write(y, v)
    })
}

/// The flat source index (and validity) of each element of `dims` for a
/// space-to-depth read: the packed channel `q` sits on axis `q_axis` scaled by
/// `q_scale` (plus the axis `q_plus` when given). `pad_t` shifts time for
/// [`avg_down`]'s front padding; a negative time is invalid.
fn unshuffle_index(
    cx: &mut Cx<'_>,
    m: &Mapped,
    r: [u32; 3],
    c_in: i64,
    dims: &[i64],
    q_scale: i64,
    q_plus: Option<(i64, Val)>,
) -> Result<(Val, Val), Error> {
    let [r1, r2, r3] = r.map(i64::from);
    let vol = r1 * r2 * r3;
    let q = cx.iota(Elem::I32, dims, 1);
    let mut q = mul_k(cx, q, q_scale)?;
    let mut pad_t = None;
    if let Some((axis, pad)) = q_plus {
        let j = cx.iota(Elem::I32, dims, axis);
        q = cx.add(q, j)?;
        pad_t = Some(pad);
    }
    let cin = div_k(cx, q, vol)?;
    let blk = rem_k(cx, q, vol)?;
    let i1 = div_k(cx, blk, r2 * r3)?;
    let i2 = div_k(cx, blk, r3)?;
    let i2 = rem_k(cx, i2, r2)?;
    let i3 = rem_k(cx, blk, r3)?;
    let row = |cx: &mut Cx<'_>, v: Val| spread(cx, v, dims, &[0]);
    let ot = mul_k(cx, m.o.v.t, r1)?;
    let ot = row(cx, ot)?;
    let ti = cx.add(ot, i1)?;
    let ti = match pad_t {
        Some(pad) => {
            let pad = row(cx, pad)?;
            cx.sub(ti, pad)?
        }
        None => ti,
    };
    let oh = mul_k(cx, m.o.v.h, r2)?;
    let oh = row(cx, oh)?;
    let hi = cx.add(oh, i2)?;
    let ow = mul_k(cx, m.o.v.w, r3)?;
    let ow = row(cx, ow)?;
    let wi = cx.add(ow, i3)?;
    let b = Boxes {
        t: row(cx, m.inb.t)?,
        h: row(cx, m.inb.h)?,
        w: row(cx, m.inb.w)?,
        off: row(cx, m.inb.off)?,
    };
    let zero = k(cx, ti, 0);
    let ti_safe = cx.max(ti, zero)?;
    let src = ravel(cx, &b, ti_safe, hi, wi)?;
    let idx = mul_k(cx, src, c_in)?;
    let idx = cx.add(idx, cin)?;
    let valid = row(cx, m.o.valid)?;
    let t_ok = cmp_k(cx, Cmp::Ge, ti, 0)?;
    let valid = cx.and(valid, t_ok)?;
    Ok((idx, valid))
}

/// Space-to-depth then a mean over each run of `group` packed channels:
/// output channel `n` averages packed channels `n * group .. (n + 1) * group`
/// (in f32, divided by `group`), time padded in front to a whole block with
/// frames that count as zero. kernels-cuda `conv::resample::avg_down`.
pub fn avg_down(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    group: u32,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "conv.avg_down";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    let vol = volume(OP, r)?;
    let widened = u64::from(x.width) * u64::from(vol);
    if group == 0
        || !widened.is_multiple_of(u64::from(group))
        || u64::from(y.width) != widened / u64::from(group)
    {
        return Err(refuse(
            OP,
            format!(
                "{} channels widen to {widened} and do not fold into {} groups of {group}",
                x.width, y.width
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let m = mapped(cx, grid, y_grid, y.rows)?;
        let r1 = i64::from(r[0]);
        let tm = rem_k(cx, m.inb.t, r1)?;
        let top = k(cx, tm, r1);
        let tm = cx.sub(top, tm)?;
        let pad = rem_k(cx, tm, r1)?;
        let dims = [i64::from(y.rows), i64::from(y.width), i64::from(group)];
        let (idx, valid) = unshuffle_index(
            cx,
            &m,
            r,
            i64::from(x.width),
            &dims,
            i64::from(group),
            Some((2, pad)),
        )?;
        let xv = cx.read_f32(x)?;
        let v = take_elems(cx, xv, idx, valid)?;
        let s = cx.reduce(v, &[2], Fold::Sum)?;
        let g = cx.like_f(s, f64::from(group));
        let s = cx.div(s, g)?;
        cx.write(y, s)
    })
}

/// [`pixel_unshuffle`] by the patch size.
pub fn patchify(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    pixel_unshuffle(ctx, x, grid, p, y, y_grid)
}

/// [`pixel_shuffle`] by the patch size, untrimmed.
pub fn unpatchify(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    p: [u32; 3],
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    pixel_shuffle(ctx, x, grid, p, 0, y, y_grid)
}
