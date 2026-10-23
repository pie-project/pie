//! Voxel ops over per-clip lane tables (poem_ir `Conv`): the grid rule,
//! 3-D conv, its frame cache, group norm and resampling, in one file each
//! below; voxel attention is `attn::voxel`. Reference: kernels-cuda `conv`.
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

pub mod cache;
pub mod conv3d;
pub mod group_norm;
pub mod resample;
pub mod rule;

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, GatherDims, Val};
use crate::tensor::Stored;

// ------------------------------------------------------------------ types

/// How a derived table maps each clip's box (`poem_ir::GridRule`).
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
            (n + front + back)
                .checked_sub(k)
                .map(|span| span / s.max(1) + 1)
        };
        Some([
            axis(t, self.k[0], self.stride[0], self.pad[0], self.back_t())?,
            axis(h, self.k[1], self.stride[1], self.pad[1], self.pad_back[1])?,
            axis(w, self.k[2], self.stride[2], self.pad[2], self.pad_back[2])?,
        ])
    }
}

// ---------------------------------------------------------------- checks

pub(crate) fn nonzero(op: &'static str, what: &str, v: u32) -> Result<u32, Error> {
    if v == 0 {
        return Err(refuse(op, format!("{what} is zero")));
    }
    Ok(v)
}

pub(crate) fn lanes_of(op: &'static str, what: &str, grid: impl Stored) -> Result<u32, Error> {
    if grid.dtype() != Dtype::I32 || grid.width() != 4 {
        return Err(refuse(
            op,
            format!(
                "the {what} grid is {}x{} {:?}; a lane table is `[lanes, 4]` i32 `{{t, h, w, row_offset}}`",
                grid.rows(),
                grid.width(),
                grid.dtype()
            ),
        ));
    }
    nonzero(op, "the lane count", grid.rows())
}

fn lane_pair(op: &'static str, grid: impl Stored, y_grid: impl Stored) -> Result<u32, Error> {
    let lanes = lanes_of(op, "input", grid)?;
    let o = lanes_of(op, "output", y_grid)?;
    if lanes != o {
        return Err(refuse(
            op,
            format!("{lanes} input lanes against {o} output lanes"),
        ));
    }
    Ok(lanes)
}

fn movable(op: &'static str, x: impl Stored, y: impl Stored) -> Result<(), Error> {
    if !matches!(x.dtype(), Dtype::Bf16 | Dtype::F16) {
        return Err(Error::DtypeUnsupported {
            op,
            dtype: x.dtype(),
        });
    }
    if x.dtype() != y.dtype() {
        return Err(refuse(
            op,
            format!(
                "{:?} into {:?}; the op keeps the element",
                x.dtype(),
                y.dtype()
            ),
        ));
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

pub(crate) fn add_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
    if n == 0 {
        return Ok(a);
    }
    let c = k(f, a, n);
    Ok(f.add(a, c)?)
}

pub(crate) fn mul_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
    if n == 1 {
        return Ok(a);
    }
    let c = k(f, a, n);
    Ok(f.mul(a, c)?)
}

pub(crate) fn div_k(f: &mut Func, a: Val, n: i64) -> Result<Val, Error> {
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
pub(crate) fn spread(f: &mut Func, a: Val, dims: &[i64], map: &[i64]) -> Result<Val, Error> {
    Ok(f.broadcast(a, dims, map)?)
}

/// Exclusive prefix sum of an i32 vector.
fn exclusive(f: &mut Func, v: Val) -> Result<Val, Error> {
    let inc = f.scan(v, 0, Fold::Sum)?;
    Ok(f.sub(inc, v)?)
}

// --------------------------------------------------------- lane tables

/// A lane table's columns, `[lanes]` each.
pub(crate) struct Boxes {
    pub(crate) t: Val,
    pub(crate) h: Val,
    pub(crate) w: Val,
    pub(crate) off: Val,
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
pub(crate) struct Voxel {
    pub(crate) t: Val,
    pub(crate) h: Val,
    pub(crate) w: Val,
}

/// The rows of a plane laid out by `grid`: which lane each row is in, that
/// lane's box, and the row's voxel. A row no lane holds is `valid = false`
/// and reads a unit box.
pub(crate) struct Placed {
    pub(crate) valid: Val,
    pub(crate) lane: Val,
    pub(crate) b: Boxes,
    pub(crate) v: Voxel,
}

pub(crate) fn place(f: &mut Func, grid: Val, rows: i64) -> Result<Placed, Error> {
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
pub(crate) fn zero_invalid(f: &mut Func, x: Val, valid: Val) -> Result<Val, Error> {
    let d = f.dims(x).to_vec();
    let v = spread(f, valid, &d, &[0])?;
    let z = f.like_f(x, 0.0);
    Ok(f.select(v, x, z)?)
}
