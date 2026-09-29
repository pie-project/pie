//! Voxel ops over per-clip lane tables (model_ir `Spatial`). Reference:
//! kernels-cuda `spatial::*` (`kernels/spatial/*.cuh`), whose module surface
//! (`spatial::conv3d`, `spatial::group_norm`, …) these entries mirror.
//!
//! A lane table (`grid`) is an i32 `[lanes, 4]` of `{t, h, w, row_offset}`:
//! clip `l` holds rows `[off, off + t*h*w)` of its plane, raster order
//! `(t, h, w)`. The table is device data, so every entry here is written
//! against it as data: each output row finds its lane by comparing its index
//! against the table (lanes are few), unravels its voxel with integer
//! arithmetic, and reads its sources by gather. Clips of different sizes
//! share one fire, and a row no lane covers lands zero, as on CUDA. The only
//! static extents are the planes' row counts (the fire's bucket).
//!
//! [`conv3d_boxed`] is the fast path for a table the host knows: each clip is
//! cut out as a `[1, t, h, w, C]` volume and run through
//! `stablehlo.convolution`, so the clip boxes become part of the executable's
//! key. [`conv3d`] is the table-as-data form (an implicit GEMM: per tap, a
//! row gather, then one `dot_general`), and answers for any table.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Combine, Elem, Fold, Func, GatherDims, ScatterDims, Ty, Val};
use crate::tensor::Tensor;

// ------------------------------------------------------------------ types

/// How a derived table maps each clip's box (`model_ir::GridRule`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRule {
    Conv {
        k: [u32; 3],
        stride: [u32; 3],
        pad: [u32; 3],
        pad_back: [u32; 3],
        causal_t: bool,
    },
    Upsample {
        factor: [u32; 3],
        keep_first_frame: bool,
    },
    Shuffle {
        r: [u32; 3],
        trim_t: u32,
    },
    Unshuffle {
        r: [u32; 3],
    },
    AvgDown {
        factor: [u32; 3],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimePad {
    Zero,
    Replicate,
}

/// A 3-d convolution's geometry; the weight is `[C_out, kt*kh*kw*C_in]`,
/// tap-major (tap `(it*kh + ih)*kw + iw` outer, input channel inner).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv3d {
    pub k: [u32; 3],
    pub stride: [u32; 3],
    pub pad: [u32; 3],
    pub pad_back: [u32; 3],
    pub causal_t: bool,
    pub time_pad: TimePad,
}

impl Conv3d {
    #[must_use]
    pub const fn taps(&self) -> u32 {
        self.k[0] * self.k[1] * self.k[2]
    }

    /// The time padding after the clip: none under `causal_t`.
    #[must_use]
    pub const fn back_t(&self) -> u32 {
        if self.causal_t { 0 } else { self.pad_back[0] }
    }

    #[must_use]
    pub fn out_extent(&self, [t, h, w]: [u32; 3]) -> Option<[u32; 3]> {
        let axis = |n: u32, k: u32, s: u32, front: u32, back: u32| {
            (n + front + back).checked_sub(k).map(|span| span / s.max(1) + 1)
        };
        Some([
            axis(t, self.k[0], self.stride[0], self.pad[0], self.back_t())?,
            axis(h, self.k[1], self.stride[1], self.pad[1], self.pad_back[1])?,
            axis(w, self.k[2], self.stride[2], self.pad[2], self.pad_back[2])?,
        ])
    }
}

/// Which keys a spatial attention row sees: its whole clip, or its block of
/// `n` frames (`model_ir::VoxelSegment`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Segment {
    #[default]
    Lane,
    Frames(u32),
}

// ---------------------------------------------------------------- checks

fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("{what} is zero")));
    }
    Ok(v)
}

fn lanes_of(op: &'static str, what: &str, grid: Tensor) -> Result<u32, Error> {
    if grid.dtype != Dtype::I32 || grid.width != 4 {
        return Err(refuse(
            op,
            format!(
                "the {what} grid is {}x{} {:?}; a lane table is `[lanes, 4]` i32 `{{t, h, w, row_offset}}`",
                grid.rows, grid.width, grid.dtype
            ),
        ));
    }
    nonzero(op, "the lane count", grid.rows)
}

fn lane_pair(op: &'static str, grid: Tensor, y_grid: Tensor) -> Result<u32, Error> {
    let lanes = lanes_of(op, "input", grid)?;
    let o = lanes_of(op, "output", y_grid)?;
    if lanes != o {
        return Err(refuse(op, format!("{lanes} input lanes against {o} output lanes")));
    }
    Ok(lanes)
}

fn movable(op: &'static str, x: Tensor, y: Tensor) -> Result<(), Error> {
    if !matches!(x.dtype, Dtype::Bf16 | Dtype::F16) {
        return Err(Error::DtypeUnsupported { op, dtype: x.dtype });
    }
    if x.dtype != y.dtype {
        return Err(refuse(op, format!("{:?} into {:?}; the op keeps the element", x.dtype, y.dtype)));
    }
    Ok(())
}

fn volume(op: &'static str, r: [u32; 3]) -> Result<u32, Error> {
    for (i, v) in r.iter().enumerate() {
        nonzero(op, ["r1", "r2", "r3"][i], *v)?;
    }
    r[0].checked_mul(r[1])
        .and_then(|v| v.checked_mul(r[2]))
        .ok_or_else(|| refuse(op, "the block volume overflows"))
}

// ------------------------------------------------------ integer helpers

fn k(f: &mut Func, like: Val, n: i64) -> Val {
    f.like_i(like, n)
}

fn add_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
    if n == 0 {
        return Ok(a);
    }
    let c = k(f, a, n);
    Ok(f.add(a, c)?)
}

fn mul_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
    if n == 1 {
        return Ok(a);
    }
    let c = k(f, a, n);
    Ok(f.mul(a, c)?)
}

fn div_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
    if n == 1 {
        return Ok(a);
    }
    let c = k(f, a, n);
    Ok(f.div(a, c)?)
}

fn rem_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
    if n == 1 {
        return Ok(k(f, a, 0));
    }
    let c = k(f, a, n);
    Ok(f.rem(a, c)?)
}

fn cmp_k(f: &mut Func, dir: Cmp, a: Val, n: i64) -> Result<Val, Error> {
    let c = k(f, a, n);
    Ok(f.compare(dir, a, c)?)
}

fn all(f: &mut Func, ps: &[Val]) -> Result<Val, Error> {
    let mut acc = ps[0];
    for &p in &ps[1..] {
        acc = f.and(acc, p)?;
    }
    Ok(acc)
}

/// `0 <= a < b`, elementwise.
fn within(f: &mut Func, a: Val, b: Val) -> Result<Val, Error> {
    let lo = cmp_k(f, Cmp::Ge, a, 0)?;
    let hi = f.compare(Cmp::Lt, a, b)?;
    Ok(f.and(lo, hi)?)
}

/// `a` broadcast to `dims` along `map`.
fn spread(f: &mut Func, a: Val, dims: &[i64], map: &[i64]) -> Result<Val, Error> {
    Ok(f.broadcast(a, dims, map)?)
}

/// Exclusive prefix sum of an i32 vector.
fn exclusive(f: &mut Func, v: Val) -> Result<Val, Error> {
    let inc = f.scan(v, 0, Fold::Sum)?;
    Ok(f.sub(inc, v)?)
}

// --------------------------------------------------------- lane tables

/// A lane table's columns, `[lanes]` each.
struct Boxes {
    t: Val,
    h: Val,
    w: Val,
    off: Val,
}

fn boxes(f: &mut Func, grid: Val) -> Result<Boxes, Error> {
    let lanes = f.dims(grid)[0];
    let col = |f: &mut Func, j: i64| -> Result<Val, Error> {
        let c = f.slice_axis(grid, 1, j, j + 1)?;
        Ok(f.reshape(c, &[lanes])?)
    };
    Ok(Boxes {
        t: col(f, 0)?,
        h: col(f, 1)?,
        w: col(f, 2)?,
        off: col(f, 3)?,
    })
}

impl Boxes {
    fn plane(&self, f: &mut Func) -> Result<Val, Error> {
        Ok(f.mul(self.h, self.w)?)
    }

    fn voxels(&self, f: &mut Func) -> Result<Val, Error> {
        let p = self.plane(f)?;
        Ok(f.mul(p, self.t)?)
    }

    /// Each field picked per row by `lane` (`[rows]`).
    fn at(&self, f: &mut Func, lane: Val) -> Result<Boxes, Error> {
        Ok(Boxes {
            t: take1(f, self.t, lane)?,
            h: take1(f, self.h, lane)?,
            w: take1(f, self.w, lane)?,
            off: take1(f, self.off, lane)?,
        })
    }
}

/// Where each of `rows` rows falls among lanes holding `[start, start +
/// count)`: the first lane that holds it (`lane`, 0 when none), whether one
/// does, and the row's index inside it.
struct Located {
    valid: Val,
    lane: Val,
    local: Val,
}

fn locate(f: &mut Func, start: Val, count: Val, rows: i64) -> Result<Located, Error> {
    let lanes = f.dims(start)[0];
    let dims = [rows, lanes];
    let m = f.iota(Elem::I32, &dims, 0);
    let s = spread(f, start, &dims, &[1])?;
    let n = spread(f, count, &dims, &[1])?;
    let e = f.add(s, n)?;
    let lo = f.compare(Cmp::Ge, m, s)?;
    let hi = f.compare(Cmp::Lt, m, e)?;
    let hit = f.and(lo, hi)?;
    let l = f.iota(Elem::I32, &dims, 1);
    let none = f.like_i(l, lanes);
    let l = f.select(hit, l, none)?;
    let lane = f.reduce(l, &[1], Fold::Min)?;
    let valid = cmp_k(f, Cmp::Lt, lane, lanes)?;
    let zero = k(f, lane, 0);
    let lane = f.select(valid, lane, zero)?;
    let row = f.iota(Elem::I32, &[rows], 0);
    let s = take1(f, start, lane)?;
    let local = f.sub(row, s)?;
    Ok(Located { valid, lane, local })
}

/// A voxel's coordinates in its clip, `[rows]` each.
struct Voxel {
    t: Val,
    h: Val,
    w: Val,
}

/// The rows of a plane laid out by `grid`: which lane each row is in, that
/// lane's box, and the row's voxel. A row no lane holds is `valid = false`
/// and reads a unit box.
struct Placed {
    valid: Val,
    lane: Val,
    b: Boxes,
    v: Voxel,
}

fn place(f: &mut Func, grid: Val, rows: i64) -> Result<Placed, Error> {
    let bx = boxes(f, grid)?;
    let vox = bx.voxels(f)?;
    let at = locate(f, bx.off, vox, rows)?;
    let b = bx.at(f, at.lane)?;
    let b = unit_where_invalid(f, b, at.valid)?;
    let v = unravel(f, &b, at.local)?;
    Ok(Placed {
        valid: at.valid,
        lane: at.lane,
        b,
        v,
    })
}

fn unit_where_invalid(f: &mut Func, b: Boxes, valid: Val) -> Result<Boxes, Error> {
    let one = k(f, b.t, 1);
    let zero = k(f, b.t, 0);
    Ok(Boxes {
        t: f.select(valid, b.t, one)?,
        h: f.select(valid, b.h, one)?,
        w: f.select(valid, b.w, one)?,
        off: f.select(valid, b.off, zero)?,
    })
}

fn unravel(f: &mut Func, b: &Boxes, local: Val) -> Result<Voxel, Error> {
    let zero = k(f, local, 0);
    let local = f.max(local, zero)?;
    let w = f.rem(local, b.w)?;
    let rest = f.div(local, b.w)?;
    let t = f.div(rest, b.h)?;
    let h = f.rem(rest, b.h)?;
    Ok(Voxel { t, h, w })
}

/// `off + (t*H + h)*W + w` in box `b`.
fn ravel(f: &mut Func, b: &Boxes, t: Val, h: Val, w: Val) -> Result<Val, Error> {
    let r = f.mul(t, b.h)?;
    let r = f.add(r, h)?;
    let r = f.mul(r, b.w)?;
    let r = f.add(r, w)?;
    Ok(f.add(r, b.off)?)
}

// ---------------------------------------------------------------- gathers

/// `v[idx]` for a rank-1 `v`, any-rank `idx` (clamped into range).
fn take1(f: &mut Func, v: Val, idx: Val) -> Result<Val, Error> {
    let rank = f.ty(idx).rank() as i64;
    Ok(f.gather(
        v,
        idx,
        &GatherDims {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: rank,
            ..GatherDims::default()
        },
        &[1],
    )?)
}

/// Elements of a rank-2 `x` at flat index `idx` where `valid`, zero elsewhere.
fn take_elems(f: &mut Func, x: Val, idx: Val, valid: Val) -> Result<Val, Error> {
    let n = f.ty(x).elements();
    let flat = f.reshape(x, &[n])?;
    let zero = f.const_i(f.elem(x), 0, &[1]);
    let flat = f.concat(&[flat, zero], 0)?;
    let far = k(f, idx, n);
    let idx = f.select(valid, idx, far)?;
    take1(f, flat, idx)
}

/// Rows of `x` at `idx` (`[rows]`) where `valid`, zero rows elsewhere.
fn take_rows_or_zero(f: &mut Func, x: Val, idx: Val, valid: Val) -> Result<Val, Error> {
    let d = f.dims(x).to_vec();
    let zero = f.const_i(f.elem(x), 0, &[1, d[1]]);
    let x = f.concat(&[x, zero], 0)?;
    let far = k(f, idx, d[0]);
    let idx = f.select(valid, idx, far)?;
    Ok(f.take_rows(x, idx)?)
}

/// Zeroes the rows of a `[rows, width]` whose `valid[row]` is false.
fn zero_invalid(f: &mut Func, x: Val, valid: Val) -> Result<Val, Error> {
    let d = f.dims(x).to_vec();
    let v = spread(f, valid, &d, &[0])?;
    let z = f.like_f(x, 0.0);
    Ok(f.select(v, x, z)?)
}

// ------------------------------------------------------------------- grid

/// Derives the output lane table of `rule` from `grid`: each clip's box
/// mapped, a box the rule cannot map zeroed, offsets packed in lane order.
pub fn derive_grid(ctx: &Ctx<'_>, grid: Tensor, rule: GridRule, y: Tensor) -> Result<(), Error> {
    const OP: &str = "spatial.grid";
    lane_pair(OP, grid, y)?;
    ctx.emit(&mut |cx| {
        let g = cx.read(grid)?;
        let b = boxes(cx, g)?;
        let (t, h, w) = (b.t, b.h, b.w);
        let lanes = i64::from(grid.rows);
        let yes = cx.const_i(Elem::Pred, 1, &[lanes]);
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
                    let span = add_k(cx, n, i64::from(pad[i]) + i64::from(back) - i64::from(kk[i]))?;
                    let fine = cmp_k(cx, Cmp::Ge, span, 0)?;
                    ok = cx.and(ok, fine)?;
                    let o = div_k(cx, span, i64::from(stride[i].max(1)))?;
                    outs.push(add_k(cx, o, 1)?);
                }
                (outs[0], outs[1], outs[2], ok)
            }
            GridRule::Upsample {
                factor,
                keep_first_frame,
            } => {
                let plain = mul_k(cx, t, i64::from(factor[0]))?;
                let ot = if keep_first_frame {
                    let tm = add_k(cx, t, -1)?;
                    let kept = mul_k(cx, tm, i64::from(factor[0]))?;
                    let kept = add_k(cx, kept, 1)?;
                    let pos = cmp_k(cx, Cmp::Gt, t, 0)?;
                    cx.select(pos, kept, plain)?
                } else {
                    plain
                };
                let oh = mul_k(cx, h, i64::from(factor[1]))?;
                let ow = mul_k(cx, w, i64::from(factor[2]))?;
                (ot, oh, ow, yes)
            }
            GridRule::Shuffle { r, trim_t } => {
                let ot = mul_k(cx, t, i64::from(r[0]))?;
                let ot = add_k(cx, ot, -i64::from(trim_t))?;
                let ok = cmp_k(cx, Cmp::Gt, ot, 0)?;
                let oh = mul_k(cx, h, i64::from(r[1]))?;
                let ow = mul_k(cx, w, i64::from(r[2]))?;
                (ot, oh, ow, ok)
            }
            GridRule::Unshuffle { r } | GridRule::AvgDown { factor: r } => {
                let avg = matches!(rule, GridRule::AvgDown { .. });
                if r.contains(&0) {
                    let no = cx.not(yes);
                    (t, h, w, no)
                } else {
                    let mut ok = yes;
                    let axes: &[(Val, u32)] = if avg {
                        &[(h, r[1]), (w, r[2])]
                    } else {
                        &[(t, r[0]), (h, r[1]), (w, r[2])]
                    };
                    for &(n, f) in axes {
                        let m = rem_k(cx, n, i64::from(f))?;
                        let z = cmp_k(cx, Cmp::Eq, m, 0)?;
                        ok = cx.and(ok, z)?;
                    }
                    let ot = if avg {
                        let up = add_k(cx, t, i64::from(r[0]) - 1)?;
                        div_k(cx, up, i64::from(r[0]))?
                    } else {
                        div_k(cx, t, i64::from(r[0]))?
                    };
                    let oh = div_k(cx, h, i64::from(r[1]))?;
                    let ow = div_k(cx, w, i64::from(r[2]))?;
                    (ot, oh, ow, ok)
                }
            }
        };
        let zero = k(cx, t, 0);
        let ot = cx.select(ok, ot, zero)?;
        let oh = cx.select(ok, oh, zero)?;
        let ow = cx.select(ok, ow, zero)?;
        let vox = cx.mul(ot, oh)?;
        let vox = cx.mul(vox, ow)?;
        let off = exclusive(cx, vox)?;
        let mut cols = Vec::new();
        for c in [ot, oh, ow, off] {
            cols.push(cx.reshape(c, &[lanes, 1])?);
        }
        let out = cx.concat(&cols, 1)?;
        cx.write(y, out)
    })
}

// ------------------------------------------------------------------- conv

fn conv_checks(
    op: &'static str,
    x: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: &Conv3d,
    cache: Option<Tensor>,
    y: Tensor,
) -> Result<(), Error> {
    if x.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op, dtype: x.dtype });
    }
    if w.dtype != Dtype::Bf16 || y.dtype != Dtype::Bf16 {
        return Err(refuse(op, "the weight and the landing are bf16"));
    }
    nonzero(op, "the input channels", x.width)?;
    nonzero(op, "the output channels", y.width)?;
    for (i, a) in ["kt", "kh", "kw"].into_iter().enumerate() {
        nonzero(op, a, conv.k[i])?;
        nonzero(op, ["st", "sh", "sw"][i], conv.stride[i])?;
    }
    let taps = u64::from(conv.taps());
    if w.rows != y.width || u64::from(w.width) != taps * u64::from(x.width) {
        return Err(refuse(
            op,
            format!(
                "the weight is {}x{}; this convolution reads [{}, {}] = [C_out, kt*kh*kw*C_in]",
                w.rows,
                w.width,
                y.width,
                taps * u64::from(x.width)
            ),
        ));
    }
    if let Some(b) = bias
        && (b.dtype != Dtype::F32 || b.elements() != u64::from(y.width))
    {
        return Err(refuse(
            op,
            format!("the bias is {}x{} {:?}; expected {} f32", b.rows, b.width, b.dtype, y.width),
        ));
    }
    if let Some(c) = cache {
        if !conv.causal_t || conv.pad[0] == 0 {
            return Err(refuse(op, "a frame cache is read only under `causal_t` with a front pad"));
        }
        if c.dtype != x.dtype || c.width != x.width {
            return Err(refuse(
                op,
                format!("the cache is {}x{} {:?}; it holds `[frames, C_in]` bf16", c.rows, c.width, c.dtype),
            ));
        }
    }
    Ok(())
}

/// Bias and zeroed dead rows on an f32 accumulator, then the bf16 landing.
fn conv_land(
    cx: &mut Cx<'_>,
    acc: Val,
    bias: Option<Tensor>,
    valid: Option<Val>,
    y: Tensor,
) -> Result<(), Error> {
    let d = cx.dims(acc).to_vec();
    let mut acc = acc;
    if let Some(b) = bias {
        let bv = cx.read_f32(b)?;
        let bv = cx.reshape(bv, &[d[1]])?;
        let bv = spread(cx, bv, &d, &[1])?;
        acc = cx.add(acc, bv)?;
    }
    if let Some(valid) = valid {
        acc = zero_invalid(cx, acc, valid)?;
    }
    cx.write(y, acc)
}

/// Bytes of gathered taps one step of the implicit GEMM may hold.
const TAP_BUDGET: i64 = 256 << 20;

/// A 3-d convolution over each clip of `grid`, landing clip `l` of `y_grid`
/// (kernels-cuda `spatial::conv3d`, `conv3d_direct`). Padding: height and
/// width pad with zeros; time pads front by `pad[0]` and back by
/// `pad_back[0]` (none when `causal_t`) with zeros, or with the edge frame
/// under `TimePad::Replicate` (never past the end when causal); under
/// `causal_t` a `cache` of `pad[0]` frames per clip (clip `l`'s at row
/// `pad[0] * Σ_{j<l} h_j*w_j`) stands in front of frame 0. f32 accumulation,
/// f32 bias, one bf16 rounding; a row no output lane holds lands zero.
///
/// The table is read as data (see the module note); [`conv3d_boxed`] is the
/// `stablehlo.convolution` form for a table the host knows.
pub fn conv3d(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    cache: Option<Tensor>,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.conv3d";
    lane_pair(OP, grid, y_grid)?;
    conv_checks(OP, x, w, bias, &conv, cache, y)?;
    let taps = conv.taps() as usize;
    let c_in = i64::from(x.width);
    let rows = i64::from(y.rows);
    let per = (TAP_BUDGET / (rows.max(1) * c_in * 2)).clamp(1, taps as i64) as usize;
    ctx.emit(&mut |cx| {
        let og = cx.read(y_grid)?;
        let o = place(cx, og, rows)?;
        let ig = cx.read(grid)?;
        let ib = boxes(cx, ig)?;
        let cache_base = {
            let planes = ib.plane(cx)?;
            let before = exclusive(cx, planes)?;
            let base = mul_k(cx, before, i64::from(conv.pad[0]))?;
            take1(cx, base, o.lane)?
        };
        let inb = ib.at(cx, o.lane)?;
        let inb = unit_where_invalid(cx, inb, o.valid)?;

        // One table: the plane, the cache behind it, a zero row last.
        let xv = cx.read(x)?;
        let mut parts = vec![xv];
        let cache_at = i64::from(x.rows);
        if let Some(c) = cache {
            parts.push(cx.read(c)?);
        }
        let dead = cache_at + cache.map_or(0, |c| i64::from(c.rows));
        let zero = cx.const_i(Elem::Bf16, 0, &[1, c_in]);
        parts.push(zero);
        let table = cx.concat(&parts, 0)?;

        let [_, kh, kw] = conv.k.map(i64::from);
        let [st, sh, sw] = conv.stride.map(i64::from);
        let [pt, ph, pw] = conv.pad.map(i64::from);
        let replicate = conv.time_pad == TimePad::Replicate;
        let mut index = Vec::with_capacity(taps);
        for tap in 0..taps as i64 {
            let (it, rest) = (tap / (kh * kw), tap % (kh * kw));
            let (ih, iw) = (rest / kw, rest % kw);
            let hi = mul_k(cx, o.v.h, sh)?;
            let hi = add_k(cx, hi, ih - ph)?;
            let wi = mul_k(cx, o.v.w, sw)?;
            let wi = add_k(cx, wi, iw - pw)?;
            let h_ok = within(cx, hi, inb.h)?;
            let w_ok = within(cx, wi, inb.w)?;
            let ti = mul_k(cx, o.v.t, st)?;
            let ti = add_k(cx, ti, it - pt)?;
            let before = cmp_k(cx, Cmp::Lt, ti, 0)?;
            let after = cx.compare(Cmp::Ge, ti, inb.t)?;
            let mut ok = all(cx, &[o.valid, h_ok, w_ok])?;
            // Frames before the clip.
            let mut from_cache = None;
            let ti = if conv.causal_t && cache.is_some() {
                let front = add_k(cx, ti, pt)?;
                let r = cx.mul(front, inb.h)?;
                let r = cx.add(r, hi)?;
                let r = cx.mul(r, inb.w)?;
                let r = cx.add(r, wi)?;
                let r = cx.add(r, cache_base)?;
                let r = add_k(cx, r, cache_at)?;
                from_cache = Some((before, r));
                ti
            } else if replicate {
                let z = k(cx, ti, 0);
                cx.select(before, z, ti)?
            } else {
                let keep = cx.not(before);
                ok = cx.and(ok, keep)?;
                ti
            };
            // Frames after it.
            let ti = if conv.causal_t || !replicate {
                let keep = cx.not(after);
                ok = cx.and(ok, keep)?;
                ti
            } else {
                let last = add_k(cx, inb.t, -1)?;
                cx.select(after, last, ti)?
            };
            let zero = k(cx, ti, 0);
            let ti_safe = cx.max(ti, zero)?;
            let hi_safe = cx.max(hi, zero)?;
            let wi_safe = cx.max(wi, zero)?;
            let mut src = ravel(cx, &inb, ti_safe, hi_safe, wi_safe)?;
            if let Some((before, r)) = from_cache {
                src = cx.select(before, r, src)?;
            }
            let far = k(cx, src, dead);
            let src = cx.select(ok, src, far)?;
            index.push(cx.reshape(src, &[rows, 1])?);
        }
        let index = cx.concat(&index, 1)?;
        let wv = cx.read(w)?;
        let c_out = i64::from(y.width);
        let mut acc = cx.const_f(Elem::F32, 0.0, &[rows, c_out]);
        let mut t0 = 0usize;
        while t0 < taps {
            let n = per.min(taps - t0) as i64;
            let idx = cx.slice_axis(index, 1, t0 as i64, t0 as i64 + n)?;
            let idx = cx.reshape(idx, &[rows * n])?;
            let a = cx.take_rows(table, idx)?;
            let a = cx.reshape(a, &[rows, n * c_in])?;
            let wt = cx.slice_axis(wv, 1, t0 as i64 * c_in, (t0 as i64 + n) * c_in)?;
            let part = cx.matmul_nt(a, wt, Elem::F32)?;
            acc = cx.add(acc, part)?;
            t0 += n as usize;
        }
        conv_land(cx, acc, bias, Some(o.valid), y)
    })
}

/// [`conv3d`] over clip boxes the host knows: `clips` is the input lane
/// table's contents (`{t, h, w, row_offset}` per lane), `y_clips` the output
/// one. Each clip runs as one `stablehlo.convolution` of a `[1, t, h, w,
/// C_in]` volume (time padding and the cache concatenated in front first),
/// so the boxes are static attributes: an executable built here is keyed by
/// them. Rows outside every output box land zero.
pub fn conv3d_boxed(
    ctx: &Ctx<'_>,
    x: Tensor,
    clips: &[[u32; 4]],
    w: Tensor,
    bias: Option<Tensor>,
    conv: Conv3d,
    cache: Option<Tensor>,
    y: Tensor,
    y_clips: &[[u32; 4]],
) -> Result<(), Error> {
    const OP: &str = "spatial.conv3d";
    conv_checks(OP, x, w, bias, &conv, cache, y)?;
    if clips.len() != y_clips.len() || clips.is_empty() {
        return Err(refuse(
            OP,
            format!("{} input boxes against {} output boxes", clips.len(), y_clips.len()),
        ));
    }
    let pt = u64::from(conv.pad[0]);
    let mut cache_at = 0u64;
    let mut plan = Vec::with_capacity(clips.len());
    for (l, (c, o)) in clips.iter().zip(y_clips).enumerate() {
        let [t, h, wd, off] = *c;
        let plane = u64::from(h) * u64::from(wd);
        let base = cache_at;
        cache_at += pt * plane;
        let want = conv.out_extent([t, h, wd]);
        let vox = plane * u64::from(t);
        if vox == 0 || want.is_none() {
            continue;
        }
        let want = want.unwrap_or_default();
        if [o[0], o[1], o[2]] != want {
            return Err(refuse(
                OP,
                format!("clip {l}'s box {:?} convolves to {want:?}, and its output box is {:?}", &c[..3], &o[..3]),
            ));
        }
        let o_vox = want.iter().map(|&n| u64::from(n)).product::<u64>();
        if u64::from(off) + vox > u64::from(x.rows) || u64::from(o[3]) + o_vox > u64::from(y.rows) {
            return Err(refuse(OP, format!("clip {l}'s rows run past the plane")));
        }
        if let Some(cc) = cache
            && conv.causal_t
            && base + pt * plane > u64::from(cc.rows)
        {
            return Err(refuse(OP, format!("clip {l}'s cached frames run past the cache")));
        }
        plan.push((*c, *o, base));
    }
    ctx.emit(&mut |cx| {
        let c_in = i64::from(x.width);
        let c_out = i64::from(y.width);
        let [kt, kh, kw] = conv.k.map(i64::from);
        let wv = cx.read(w)?;
        let kernel = cx.reshape(wv, &[c_out, kt, kh, kw, c_in])?;
        let xv = cx.read(x)?;
        let cv = match cache {
            Some(c) => Some(cx.read(c)?),
            None => None,
        };
        let mut out = cx.const_f(Elem::F32, 0.0, &[i64::from(y.rows), c_out]);
        for &([t, h, wd, off], [ot, oh, ow, o_off], base) in &plan {
            let (t, h, wd) = (i64::from(t), i64::from(h), i64::from(wd));
            let plane = h * wd;
            let clip = cx.slice_axis(xv, 0, i64::from(off), i64::from(off) + t * plane)?;
            let clip = cx.reshape(clip, &[1, t, h, wd, c_in])?;
            let frame = |cx: &mut Cx<'_>, at: i64| -> Result<Val, Error> {
                Ok(cx.slice_axis(clip, 1, at, at + 1)?)
            };
            let fill = |cx: &mut Cx<'_>, src: Option<Val>, n: i64| -> Result<Val, Error> {
                Ok(match src {
                    Some(f) => cx.broadcast(f, &[1, n, h, wd, c_in], &[0, 1, 2, 3, 4])?,
                    None => cx.const_i(Elem::Bf16, 0, &[1, n, h, wd, c_in]),
                })
            };
            let replicate = conv.time_pad == TimePad::Replicate;
            let pt = i64::from(conv.pad[0]);
            let mut vol5 = Vec::new();
            if pt > 0 {
                vol5.push(match (&cv, conv.causal_t) {
                    (Some(cv), true) => {
                        let rows = cx.slice_axis(*cv, 0, base as i64, base as i64 + pt * plane)?;
                        cx.reshape(rows, &[1, pt, h, wd, c_in])?
                    }
                    _ if replicate => {
                        let f0 = frame(cx, 0)?;
                        fill(cx, Some(f0), pt)?
                    }
                    _ => fill(cx, None, pt)?,
                });
            }
            vol5.push(clip);
            let back = i64::from(conv.back_t());
            if back > 0 {
                vol5.push(if replicate {
                    let fl = frame(cx, t - 1)?;
                    fill(cx, Some(fl), back)?
                } else {
                    fill(cx, None, back)?
                });
            }
            let vol5 = cx.concat(&vol5, 1)?;
            let [_, sh, sw] = conv.stride.map(i64::from);
            let st = i64::from(conv.stride[0]);
            let attrs = format!(
                "window_strides = array<i64: {st}, {sh}, {sw}>, padding = dense<[[0, 0], [{}, {}], [{}, {}]]> : tensor<3x2xi64>, lhs_dilation = array<i64: 1, 1, 1>, rhs_dilation = array<i64: 1, 1, 1>, dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, f]x[o, 0, 1, 2, i]->[b, 0, 1, 2, f]>, feature_group_count = 1 : i64, batch_group_count = 1 : i64",
                conv.pad[1], conv.pad_back[1], conv.pad[2], conv.pad_back[2]
            );
            let (ot, oh, ow) = (i64::from(ot), i64::from(oh), i64::from(ow));
            let y5 = cx.op_n(
                "convolution",
                &[vol5, kernel],
                &attrs,
                vec![Ty::new(Elem::F32, &[1, ot, oh, ow, c_out])],
            )[0];
            let rows = cx.reshape(y5, &[ot * oh * ow, c_out])?;
            let r0 = cx.const_i(Elem::I32, i64::from(o_off), &[]);
            let c0 = cx.const_i(Elem::I32, 0, &[]);
            out = cx.dynamic_update_slice(out, rows, &[r0, c0])?;
        }
        // Rows no box covers stay zero, bias included.
        let valid = if bias.is_some() {
            let row = cx.iota(Elem::I32, &[i64::from(y.rows)], 0);
            let mut hit = cx.const_i(Elem::Pred, 0, &[i64::from(y.rows)]);
            for &(_, [ot, oh, ow, o_off], _) in &plan {
                let n = i64::from(ot) * i64::from(oh) * i64::from(ow);
                let lo = cmp_k(cx, Cmp::Ge, row, i64::from(o_off))?;
                let hi = cmp_k(cx, Cmp::Lt, row, i64::from(o_off) + n)?;
                let inside = cx.and(lo, hi)?;
                hit = cx.or(hit, inside)?;
            }
            Some(hit)
        } else {
            None
        };
        conv_land(cx, out, bias, valid, y)
    })
}

// ----------------------------------------------------------- frame cache

/// Where each cache row sits: its lane (clips hold `frames * h*w` rows each,
/// packed in lane order from 0) and its row inside that lane's block.
fn cache_rows(
    f: &mut Func,
    grid: Val,
    frames: u32,
    rows: i64,
) -> Result<(Located, Boxes, Val), Error> {
    let bx = boxes(f, grid)?;
    let plane = bx.plane(f)?;
    let count = mul_k(f, plane, i64::from(frames))?;
    let start = exclusive(f, count)?;
    let at = locate(f, start, count, rows)?;
    let base = take1(f, start, at.lane)?;
    let b = bx.at(f, at.lane)?;
    let b = unit_where_invalid(f, b, at.valid)?;
    Ok((at, b, base))
}

fn cache_checks(
    op: &'static str,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    cache: Tensor,
    frames: u32,
) -> Result<u32, Error> {
    // The frames are bf16; a slab kept in f32 (a shell that stores every
    // recurrent state wide) holds them exactly.
    if cache.dtype != Dtype::Bf16 || !matches!(slab.dtype, Dtype::Bf16 | Dtype::F32) {
        return Err(refuse(op, "the cache is bf16 and the slab bf16 or f32"));
    }
    let lanes = lanes_of(op, "input", grid)?;
    if slot_ids.dtype != Dtype::I32 || slot_ids.elements() < u64::from(lanes) {
        return Err(refuse(
            op,
            format!(
                "the slot table is {}x{} {:?}; one i32 per lane is needed",
                slot_ids.rows, slot_ids.width, slot_ids.dtype
            ),
        ));
    }
    nonzero(op, "the cached frames", frames)?;
    nonzero(op, "the channel count", cache.width)?;
    nonzero(op, "the slab's slot width", slab.width)?;
    Ok(lanes)
}

/// Each lane's slot index, `[rows]`, from the first `lanes` entries of the
/// slot table.
fn slots_at(cx: &mut Cx<'_>, slot_ids: Tensor, lanes: u32, lane: Val) -> Result<Val, Error> {
    let s = cx.read(slot_ids)?;
    let s = cx.reshape(s, &[slot_ids.elements() as i64])?;
    let s = cx.slice_axis(s, 0, 0, i64::from(lanes))?;
    take1(cx, s, lane)
}

/// Reads each clip's `frames` cached frames out of its slot of the slab
/// (`[slots, frames*h*w*C]` flat per slot) into `cache`, clips packed in lane
/// order; what no clip holds, or what runs past a slot, reads zero.
/// kernels-cuda `spatial::cache_gather`.
pub fn cache_gather(
    ctx: &Ctx<'_>,
    slab: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    cache: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.cache_gather";
    let lanes = cache_checks(OP, slab, slot_ids, grid, cache, frames)?;
    ctx.emit(&mut |cx| {
        let (rows, c) = (i64::from(cache.rows), i64::from(cache.width));
        let g = cx.read(grid)?;
        let (at, _, base) = cache_rows(cx, g, frames, rows)?;
        let slot = slots_at(cx, slot_ids, lanes, at.lane)?;
        let dims = [rows, c];
        let row = cx.iota(Elem::I32, &dims, 0);
        let col = cx.iota(Elem::I32, &dims, 1);
        let base = spread(cx, base, &dims, &[0])?;
        let local = cx.sub(row, base)?;
        let local = mul_k(cx, local, c)?;
        let local = cx.add(local, col)?;
        let stride = i64::from(slab.width);
        let fits = cmp_k(cx, Cmp::Lt, local, stride)?;
        let valid = spread(cx, at.valid, &dims, &[0])?;
        let valid = cx.and(valid, fits)?;
        let slot = spread(cx, slot, &dims, &[0])?;
        let idx = mul_k(cx, slot, stride)?;
        let idx = cx.add(idx, local)?;
        let s = cx.read(slab)?;
        let v = take_elems(cx, s, idx, valid)?;
        cx.write(cache, v)
    })
}

/// Writes each clip's last `frames` frames into its slot of the slab: frame
/// `f` of the block is the clip's frame `t - frames + f`, or, for a clip
/// shorter than the block, frame `f + t` of `cache` (the block read before
/// this fire). kernels-cuda `spatial::cache_store`; `model_ir`'s
/// `CacheStore` passes the input as both `x` and `cache`, as CUDA does.
pub fn cache_store(
    ctx: &Ctx<'_>,
    x: Tensor,
    cache: Tensor,
    slot_ids: Tensor,
    grid: Tensor,
    frames: u32,
    slab: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.cache_store";
    let lanes = cache_checks(OP, slab, slot_ids, grid, cache, frames)?;
    if x.dtype != Dtype::Bf16 || x.width != cache.width {
        return Err(refuse(OP, "the input is not `[rows, C_in]` bf16"));
    }
    ctx.emit(&mut |cx| {
        let (rows, c) = (i64::from(cache.rows), i64::from(cache.width));
        let g = cx.read(grid)?;
        let (at, b, base) = cache_rows(cx, g, frames, rows)?;
        let slot = slots_at(cx, slot_ids, lanes, at.lane)?;
        // Per row: the frame of the block and the voxel in the plane.
        let local_row = cx.iota(Elem::I32, &[rows], 0);
        let local_row = cx.sub(local_row, base)?;
        let plane = b.plane(cx)?;
        let fr = cx.div(local_row, plane)?;
        let hw = cx.rem(local_row, plane)?;
        let src_t = cx.add(b.t, fr)?;
        let src_t = add_k(cx, src_t, -i64::from(frames))?;
        let live = cmp_k(cx, Cmp::Ge, src_t, 0)?;
        let x_row = cx.mul(src_t, plane)?;
        let x_row = cx.add(x_row, hw)?;
        let x_row = cx.add(x_row, b.off)?;
        let c_row = cx.add(fr, b.t)?;
        let c_row = cx.mul(c_row, plane)?;
        let c_row = cx.add(c_row, hw)?;
        let c_row = cx.add(c_row, base)?;
        let dims = [rows, c];
        let col = cx.iota(Elem::I32, &dims, 1);
        let at_x = spread(cx, x_row, &dims, &[0])?;
        let at_x = mul_k(cx, at_x, c)?;
        let at_x = cx.add(at_x, col)?;
        let at_c = spread(cx, c_row, &dims, &[0])?;
        let at_c = mul_k(cx, at_c, c)?;
        let at_c = cx.add(at_c, col)?;
        let yes = cx.const_i(Elem::Pred, 1, &dims);
        let xv = cx.read(x)?;
        let from_x = take_elems(cx, xv, at_x, yes)?;
        let cv = cx.read(cache)?;
        let from_c = take_elems(cx, cv, at_c, yes)?;
        let live = spread(cx, live, &dims, &[0])?;
        let value = cx.select(live, from_x, from_c)?;
        // Where it lands.
        let local = spread(cx, local_row, &dims, &[0])?;
        let local = mul_k(cx, local, c)?;
        let local = cx.add(local, col)?;
        let stride = i64::from(slab.width);
        let fits = cmp_k(cx, Cmp::Lt, local, stride)?;
        let valid = spread(cx, at.valid, &dims, &[0])?;
        let valid = cx.and(valid, fits)?;
        let slot = spread(cx, slot, &dims, &[0])?;
        let idx = mul_k(cx, slot, stride)?;
        let idx = cx.add(idx, local)?;
        let far = k(cx, idx, i64::from(i32::MAX));
        let idx = cx.select(valid, idx, far)?;
        let n = rows * c;
        let idx = cx.reshape(idx, &[n, 1])?;
        let value = cx.reshape(value, &[n])?;
        let s = cx.read(slab)?;
        let slab_elem = cx.elem(s);
        let value = cx.convert(value, slab_elem);
        let total = i64::from(slab.rows) * stride;
        let flat = cx.reshape(s, &[total])?;
        let flat = cx.scatter(
            flat,
            idx,
            value,
            &ScatterDims {
                update_window_dims: vec![],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..ScatterDims::default()
            },
            Combine::Set,
        )?;
        cx.write(slab, flat)
    })
}

// ------------------------------------------------------------- group norm

/// Group norm per clip: each clip's `(group)` moments over all its voxels
/// and the group's channels, in f32, then `(x - mean) * rsqrt(var + eps) *
/// weight + bias` (f32 `[C]` planes), optionally `silu`; a row no lane holds
/// lands zero. kernels-cuda `spatial::group_norm`.
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
    const OP: &str = "spatial.group_norm";
    if x.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op: OP, dtype: x.dtype });
    }
    if y.rows != x.rows || y.width != x.width || y.dtype != x.dtype {
        return Err(refuse(OP, "the landing is one row per input row"));
    }
    let lanes = lanes_of(OP, "input", grid)?;
    nonzero(OP, "the channel count", x.width)?;
    nonzero(OP, "the group count", groups)?;
    if !x.width.is_multiple_of(groups) {
        return Err(refuse(OP, format!("{} channels do not divide into {groups} groups", x.width)));
    }
    for (what, t) in [("weight", weight), ("bias", bias)] {
        if t.dtype != Dtype::F32 || t.elements() != u64::from(x.width) {
            return Err(refuse(
                OP,
                format!("the {what} is {}x{} {:?}; expected {} f32", t.rows, t.width, t.dtype, x.width),
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

// -------------------------------------------------------------- attention

/// Queries a chunk of the dense score matrix holds, bounded by bytes.
const SCORE_BUDGET: i64 = 64 << 20;

/// Single-head attention of each row over the keys of its clip (or its block
/// of frames), `softmax(q·kᵀ · sm_scale) · v` in f32 with one bf16 rounding;
/// a row no lane holds lands zero. kernels-cuda `spatial::attention`.
/// Computed as a dense masked product over all rows of the fire, in query
/// chunks of at most 64 MiB of scores.
pub fn attention(
    ctx: &Ctx<'_>,
    q: Tensor,
    k_: Tensor,
    v: Tensor,
    grid: Tensor,
    segment: Segment,
    sm_scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.attention";
    if q.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op: OP, dtype: q.dtype });
    }
    for (what, t) in [("key", k_), ("value", v), ("output", y)] {
        if t.rows != q.rows || t.width != q.width || t.dtype != q.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {}x{} {:?}; the query is {}x{} {:?}, and all four share one shape",
                    t.rows, t.width, t.dtype, q.rows, q.width, q.dtype
                ),
            ));
        }
    }
    if segment == Segment::Frames(0) {
        return Err(refuse(OP, "a block of zero frames holds no keys; state `Segment::Lane`"));
    }
    lanes_of(OP, "input", grid)?;
    nonzero(OP, "rows", q.rows)?;
    let rows = i64::from(q.rows);
    let chunk = (SCORE_BUDGET / (rows * 4)).clamp(1, rows);
    ctx.emit(&mut |cx| {
        let gv = cx.read(grid)?;
        let p = place(cx, gv, rows)?;
        let (begin, end) = match segment {
            Segment::Lane | Segment::Frames(0) => {
                let vox = cx.mul(p.b.h, p.b.w)?;
                let vox = cx.mul(vox, p.b.t)?;
                let end = cx.add(p.b.off, vox)?;
                (p.b.off, end)
            }
            Segment::Frames(n) => {
                let plane = cx.mul(p.b.h, p.b.w)?;
                let first = div_k(cx, p.v.t, i64::from(n))?;
                let first = mul_k(cx, first, i64::from(n))?;
                let last = add_k(cx, first, i64::from(n))?;
                let last = cx.min(last, p.b.t)?;
                let b = cx.mul(first, plane)?;
                let b = cx.add(b, p.b.off)?;
                let e = cx.mul(last, plane)?;
                let e = cx.add(e, p.b.off)?;
                (b, e)
            }
        };
        let qv = cx.read(q)?;
        let kv = cx.read(k_)?;
        let vv = cx.read_f32(v)?;
        let mut outs = Vec::new();
        let mut q0 = 0;
        while q0 < rows {
            let n = chunk.min(rows - q0);
            let qc = cx.slice_axis(qv, 0, q0, q0 + n)?;
            let s = cx.matmul_nt(qc, kv, Elem::F32)?;
            let s = cx.scale(s, f64::from(sm_scale))?;
            let dims = [n, rows];
            let j = cx.iota(Elem::I32, &dims, 1);
            let b = cx.slice_axis(begin, 0, q0, q0 + n)?;
            let e = cx.slice_axis(end, 0, q0, q0 + n)?;
            let b = spread(cx, b, &dims, &[0])?;
            let e = spread(cx, e, &dims, &[0])?;
            let lo = cx.compare(Cmp::Ge, j, b)?;
            let hi = cx.compare(Cmp::Lt, j, e)?;
            let seen = cx.and(lo, hi)?;
            let ninf = cx.like_f(s, f64::NEG_INFINITY);
            let s = cx.select(seen, s, ninf)?;
            let m = cx.reduce(s, &[1], Fold::Max)?;
            let fin = cx.is_finite(m);
            let zm = cx.like_f(m, 0.0);
            let m = cx.select(fin, m, zm)?;
            let m = spread(cx, m, &dims, &[0])?;
            let z = cx.sub(s, m)?;
            let pr = cx.exp(z);
            let zp = cx.like_f(pr, 0.0);
            let pr = cx.select(seen, pr, zp)?;
            let l = cx.reduce(pr, &[1], Fold::Sum)?;
            let o = cx.dot_general(pr, vv, &[], &[], &[1], &[0], Elem::F32)?;
            let zl = cx.like_f(l, 0.0);
            let pos = cx.compare(Cmp::Gt, l, zl)?;
            let one = cx.like_f(l, 1.0);
            let safe = cx.select(pos, l, one)?;
            let inv = cx.div(one, safe)?;
            let inv = cx.select(pos, inv, zl)?;
            let width = cx.dims(o)[1];
            let inv = spread(cx, inv, &[n, width], &[0])?;
            outs.push(cx.mul(o, inv)?);
            q0 += n;
        }
        let o = cx.concat(&outs, 0)?;
        let o = zero_invalid(cx, o, p.valid)?;
        cx.write(y, o)
    })
}

// -------------------------------------------------------------- resampling

/// Nearest upsampling per clip: output voxel `(t, h, w)` reads input
/// `(t / ft, h / fh, w / fw)`, or with `keep_first_frame` time `0 → 0` and
/// `t → (t - 1) / ft + 1`. kernels-cuda `spatial::upsample_nearest`.
pub fn upsample_nearest(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    factor: [u32; 3],
    keep_first_frame: bool,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.upsample_nearest";
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
/// kernels-cuda `spatial::pixel_shuffle`.
pub fn pixel_shuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    trim_t: u32,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.pixel_shuffle";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    let vol = volume(OP, r)?;
    if !x.width.is_multiple_of(vol) || y.width != x.width / vol {
        return Err(refuse(
            OP,
            format!("{} channels do not unpack as {} x {vol} output channels", x.width, y.width),
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
/// w r3 + i3)`. kernels-cuda `spatial::pixel_unshuffle`.
pub fn pixel_unshuffle(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.pixel_unshuffle";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    let vol = volume(OP, r)?;
    if u64::from(y.width) != u64::from(x.width) * u64::from(vol) {
        return Err(refuse(
            OP,
            format!("{} channels do not pack as {} = C x {vol} output channels", x.width, y.width),
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
/// frames that count as zero. kernels-cuda `spatial::avg_down`.
pub fn avg_down(
    ctx: &Ctx<'_>,
    x: Tensor,
    grid: Tensor,
    r: [u32; 3],
    group: u32,
    y: Tensor,
    y_grid: Tensor,
) -> Result<(), Error> {
    const OP: &str = "spatial.avg_down";
    movable(OP, x, y)?;
    lane_pair(OP, grid, y_grid)?;
    let vol = volume(OP, r)?;
    let widened = u64::from(x.width) * u64::from(vol);
    if group == 0 || !widened.is_multiple_of(u64::from(group)) || u64::from(y.width) != widened / u64::from(group) {
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
        let (idx, valid) =
            unshuffle_index(cx, &m, r, i64::from(x.width), &dims, i64::from(group), Some((2, pad)))?;
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
