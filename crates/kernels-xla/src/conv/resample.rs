//! Resampling: nearest upsample, pixel (un)shuffle, average down, (un)patchify.

use crate::conv::{
    Boxes, Placed, add_k, boxes, cmp_k, div_k, k, lane_pair, movable, mul_k, place, ravel, rebased,
    rem_k, spread, take_elems, take_rows_or_zero, unit_where_invalid, volume,
};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, Val};
use crate::tensor::{Landing, Operand};

/// Nearest upsampling per clip: output voxel `(t, h, w)` reads input
/// `(t / ft, h / fh, w / fw)`, or with `keep_first_frame` time `0 → 0` and
/// `t → (t - 1) / ft + 1`. kernels-cuda `conv::resample::upsample_nearest`.
pub fn upsample_nearest(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    factor: [u32; 3],
    keep_first_frame: bool,
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    let grid = rebased(f, grid, bases[0])?;
    let y_grid = rebased(f, y_grid, bases[1])?;
    const OP: &str = "conv.upsample";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    volume(OP, factor)?;
    if y.width != x.width {
        return Err(refuse(OP, "an upsample keeps the channel width"));
    }
    let [ft, fh, fw] = factor.map(i64::from);
    let og = y_grid.val;
    let o = place(f, og, i64::from(y.rows))?;
    let ig = grid.val;
    let ib = boxes(f, ig)?;
    let inb = ib.at(f, o.lane)?;
    let inb = unit_where_invalid(f, inb, o.valid)?;
    let ti = if keep_first_frame {
        let tm = add_k(f, o.v.t, -1)?;
        let tm = div_k(f, tm, ft)?;
        let tm = add_k(f, tm, 1)?;
        let first = cmp_k(f, Cmp::Eq, o.v.t, 0)?;
        let z = k(f, tm, 0);
        f.select(first, z, tm)?
    } else {
        div_k(f, o.v.t, ft)?
    };
    let hi = div_k(f, o.v.h, fh)?;
    let wi = div_k(f, o.v.w, fw)?;
    let src = ravel(f, &inb, ti, hi, wi)?;
    let xv = x.val;
    let rows = take_rows_or_zero(f, xv, src, o.valid)?;
    Ok(rows)
}

/// Per-row source coordinates for the shuffles, `[rows]`.
struct Mapped {
    o: Placed,
    inb: Boxes,
}

fn mapped(f: &mut Func, grid: Operand, y_grid: Operand, rows: u32) -> Result<Mapped, Error> {
    let og = y_grid.val;
    let o = place(f, og, i64::from(rows))?;
    let ig = grid.val;
    let ib = boxes(f, ig)?;
    let inb = ib.at(f, o.lane)?;
    let inb = unit_where_invalid(f, inb, o.valid)?;
    Ok(Mapped { o, inb })
}

/// Depth-to-space per clip: output channel `c` of voxel `(t', h, w)` (with
/// `t' = t + trim_t`) reads input channel `c * R + ((t' % r1) * r2 + h % r2)
/// * r3 + w % r3` of voxel `(t' / r1, h / r2, w / r3)`, `R = r1 r2 r3`.
/// kernels-cuda `conv::resample::pixel_shuffle`.
pub fn pixel_shuffle(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    r: [u32; 3],
    trim_t: u32,
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    let grid = rebased(f, grid, bases[0])?;
    let y_grid = rebased(f, y_grid, bases[1])?;
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
    let m = mapped(f, grid, y_grid, y.rows)?;
    let tt = add_k(f, m.o.v.t, i64::from(trim_t))?;
    let ti = div_k(f, tt, r1)?;
    let hi = div_k(f, m.o.v.h, r2)?;
    let wi = div_k(f, m.o.v.w, r3)?;
    let src = ravel(f, &m.inb, ti, hi, wi)?;
    let bt = rem_k(f, tt, r1)?;
    let bt = mul_k(f, bt, r2)?;
    let bh = rem_k(f, m.o.v.h, r2)?;
    let blk = f.add(bt, bh)?;
    let blk = mul_k(f, blk, r3)?;
    let bw = rem_k(f, m.o.v.w, r3)?;
    let blk = f.add(blk, bw)?;
    let base = mul_k(f, src, i64::from(x.width))?;
    let base = f.add(base, blk)?;
    let dims = [i64::from(y.rows), i64::from(y.width)];
    let base = spread(f, base, &dims, &[0])?;
    let col = f.iota(Elem::I32, &dims, 1);
    let col = mul_k(f, col, i64::from(vol))?;
    let idx = f.add(base, col)?;
    let valid = spread(f, m.o.valid, &dims, &[0])?;
    let xv = x.val;
    let v = take_elems(f, xv, idx, valid)?;
    Ok(v)
}

/// Space-to-depth per clip, the inverse of [`pixel_shuffle`] without trim:
/// output channel `cin * R + b` (`b = (i1 * r2 + i2) * r3 + i3`) of voxel
/// `(t, h, w)` reads channel `cin` of input voxel `(t r1 + i1, h r2 + i2,
/// w r3 + i3)`. kernels-cuda `conv::resample::pixel_unshuffle`.
pub fn pixel_unshuffle(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    r: [u32; 3],
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    let grid = rebased(f, grid, bases[0])?;
    let y_grid = rebased(f, y_grid, bases[1])?;
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
    let m = mapped(f, grid, y_grid, y.rows)?;
    let dims = [i64::from(y.rows), i64::from(y.width)];
    let (idx, valid) = unshuffle_index(f, &m, r, i64::from(x.width), &dims, 1, None)?;
    let xv = x.val;
    let v = take_elems(f, xv, idx, valid)?;
    Ok(v)
}

/// The flat source index (and validity) of each element of `dims` for a
/// space-to-depth read: the packed channel `q` sits on axis `q_axis` scaled by
/// `q_scale` (plus the axis `q_plus` when given). `pad_t` shifts time for
/// [`avg_down`]'s front padding; a negative time is invalid.
fn unshuffle_index(
    f: &mut Func,
    m: &Mapped,
    r: [u32; 3],
    c_in: i64,
    dims: &[i64],
    q_scale: i64,
    q_plus: Option<(i64, Val)>,
) -> Result<(Val, Val), Error> {
    let [r1, r2, r3] = r.map(i64::from);
    let vol = r1 * r2 * r3;
    let q = f.iota(Elem::I32, dims, 1);
    let mut q = mul_k(f, q, q_scale)?;
    let mut pad_t = None;
    if let Some((axis, pad)) = q_plus {
        let j = f.iota(Elem::I32, dims, axis);
        q = f.add(q, j)?;
        pad_t = Some(pad);
    }
    let cin = div_k(f, q, vol)?;
    let blk = rem_k(f, q, vol)?;
    let i1 = div_k(f, blk, r2 * r3)?;
    let i2 = div_k(f, blk, r3)?;
    let i2 = rem_k(f, i2, r2)?;
    let i3 = rem_k(f, blk, r3)?;
    let row = |f: &mut Func, v: Val| spread(f, v, dims, &[0]);
    let ot = mul_k(f, m.o.v.t, r1)?;
    let ot = row(f, ot)?;
    let ti = f.add(ot, i1)?;
    let ti = match pad_t {
        Some(pad) => {
            let pad = row(f, pad)?;
            f.sub(ti, pad)?
        }
        None => ti,
    };
    let oh = mul_k(f, m.o.v.h, r2)?;
    let oh = row(f, oh)?;
    let hi = f.add(oh, i2)?;
    let ow = mul_k(f, m.o.v.w, r3)?;
    let ow = row(f, ow)?;
    let wi = f.add(ow, i3)?;
    let b = Boxes {
        t: row(f, m.inb.t)?,
        h: row(f, m.inb.h)?,
        w: row(f, m.inb.w)?,
        off: row(f, m.inb.off)?,
    };
    let zero = k(f, ti, 0);
    let ti_safe = f.max(ti, zero)?;
    let src = ravel(f, &b, ti_safe, hi, wi)?;
    let idx = mul_k(f, src, c_in)?;
    let idx = f.add(idx, cin)?;
    let valid = row(f, m.o.valid)?;
    let t_ok = cmp_k(f, Cmp::Ge, ti, 0)?;
    let valid = f.and(valid, t_ok)?;
    Ok((idx, valid))
}

/// Space-to-depth then a mean over each run of `group` packed channels:
/// output channel `n` averages packed channels `n * group .. (n + 1) * group`
/// (in f32, divided by `group`), time padded in front to a whole block with
/// frames that count as zero. kernels-cuda `conv::resample::avg_down`.
pub fn avg_down(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    r: [u32; 3],
    group: u32,
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    let grid = rebased(f, grid, bases[0])?;
    let y_grid = rebased(f, y_grid, bases[1])?;
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
    let m = mapped(f, grid, y_grid, y.rows)?;
    let r1 = i64::from(r[0]);
    let tm = rem_k(f, m.inb.t, r1)?;
    let top = k(f, tm, r1);
    let tm = f.sub(top, tm)?;
    let pad = rem_k(f, tm, r1)?;
    let dims = [i64::from(y.rows), i64::from(y.width), i64::from(group)];
    let (idx, valid) = unshuffle_index(
        f,
        &m,
        r,
        i64::from(x.width),
        &dims,
        i64::from(group),
        Some((2, pad)),
    )?;
    let xv = x.f32(f);
    let v = take_elems(f, xv, idx, valid)?;
    let s = f.reduce(v, &[2], Fold::Sum)?;
    let g = f.like_f(s, f64::from(group));
    let s = f.div(s, g)?;
    Ok(s)
}

/// [`pixel_unshuffle`] by the patch size. `bases`: the rows `grid` and
/// `y_grid` address `x`'s and `y`'s row 0 at (`[x, y]`), nonzero in a window
/// of a segmented fire whose grids hold fire-wide rows.
pub fn patchify(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    p: [u32; 3],
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    pixel_unshuffle(f, x, grid, p, y, y_grid, bases)
}

/// [`pixel_shuffle`] by the patch size, untrimmed; `bases` as
/// [`patchify`]'s.
pub fn unpatchify(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    p: [u32; 3],
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    pixel_shuffle(f, x, grid, p, 0, y, y_grid, bases)
}
