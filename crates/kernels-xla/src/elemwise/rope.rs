//! Rotary embeddings. Every form is one table: for each column of a head
//! (or of a whole row, for a ladder that walks the row), the column it pairs
//! with, the sign its partner's sine carries, and the frequency each position
//! axis drives it at. `turn` applies a table: `out = x·cos θ + sign·x[partner]·sin θ`,
//! which is the GPU kernels' `(a·c − b·s, b·c + a·s)` pair, element for element.
//! Frequencies are computed here, on the host, in f32 (`powf`), as the CUDA
//! kernels compute them; the angle `pos · freq` and its sine and cosine are
//! taken on the device in f32, and each element rounds once, at the write.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::Func;
use crate::hlo::{Cmp, Elem, GatherDims, Val};
use crate::lanes::ci;
use crate::tensor::{Operand, Stored};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
}

const FLOATS: &[Dtype] = &[Dtype::Bf16, Dtype::F16, Dtype::F32];

/// A rotation table over a `unit`-wide run of columns.
pub(crate) struct Turn {
    unit: usize,
    partner: Vec<i64>,
    sign: Vec<f32>,
    /// `freq[axis][column]`: the angle is `Σ_axis pos[axis] · freq[axis][column]`.
    freq: Vec<Vec<f32>>,
    mscale: f32,
}

impl Turn {
    pub(crate) fn new(unit: usize, axes: usize) -> Self {
        Self {
            unit,
            partner: (0..unit as i64).collect(),
            sign: vec![0.0; unit],
            freq: vec![vec![0.0; unit]; axes],
            mscale: 1.0,
        }
    }

    /// Turns `(lo, hi)` by `pos[axis] · freq`.
    pub(crate) fn pair(&mut self, lo: usize, hi: usize, axis: usize, freq: f32) {
        self.partner[lo] = hi as i64;
        self.partner[hi] = lo as i64;
        self.sign[lo] = -1.0;
        self.sign[hi] = 1.0;
        self.freq[axis][lo] = freq;
        self.freq[axis][hi] = freq;
    }

    /// A pair the table names but leaves unturned (a ladder's identity pad).
    pub(crate) fn still(&mut self, lo: usize, hi: usize) {
        self.partner[lo] = hi as i64;
        self.partner[hi] = lo as i64;
        self.sign[lo] = -1.0;
        self.sign[hi] = 1.0;
    }

    pub(crate) fn scaled(mut self, mscale: f32) -> Self {
        self.mscale = mscale;
        self
    }
}

/// `x`'s columns along its last axis, in the order `idx` names (a constant).
pub(crate) fn take_cols(f: &mut Func, x: Val, idx: &[i64]) -> Result<Val, Error> {
    let dims = f.dims(x).to_vec();
    let last = dims.len() - 1;
    // Adjacent pairs swapped (an interleaved rotation's partners): the two
    // halves of each pair crossed, not a gather along the lanes.
    let n = idx.len();
    if n >= 2
        && n.is_multiple_of(2)
        && i64::try_from(n).ok() == dims.last().copied()
        && idx.iter().enumerate().all(|(i, &p)| p == (i ^ 1) as i64)
    {
        let mut pairs = dims.clone();
        pairs[last] = n as i64 / 2;
        pairs.push(2);
        let x2 = f.reshape(x, &pairs)?;
        let lo = f.slice_axis(x2, last + 1, 0, 1)?;
        let hi = f.slice_axis(x2, last + 1, 1, 2)?;
        let swapped = f.concat(&[hi, lo], (last + 1) as i64)?;
        return Ok(f.reshape(swapped, &dims)?);
    }
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for &i in idx {
        match runs.last_mut() {
            Some((start, len)) if *start + *len == i => *len += 1,
            _ => runs.push((i, 1)),
        }
    }
    if runs.len() <= 16 {
        let mut parts = Vec::with_capacity(runs.len());
        for (start, len) in runs {
            parts.push(f.slice_axis(x, last, start, start + len)?);
        }
        return Ok(f.concat(&parts, last as i64)?);
    }
    let n = idx.len() as i64;
    let ids = f.const_ints(Elem::I32, idx, &[n, 1])?;
    let mut sizes = dims.clone();
    sizes[last] = 1;
    Ok(f.gather(
        x,
        ids,
        &GatherDims {
            offset_dims: (0..last as i64).collect(),
            collapsed_slice_dims: vec![last as i64],
            start_index_map: vec![last as i64],
            index_vector_dim: 1,
            ..GatherDims::default()
        },
        &sizes,
    )?)
}

/// The run `[a, b)` of a head's columns `t` turns, as a table of its own,
/// when that run is not the whole head and pairs only within itself.
fn span(t: &Turn) -> Option<(usize, usize, Turn)> {
    let moved = |i: usize| t.sign[i] != 0.0;
    let a = (0..t.unit).find(|&i| moved(i))?;
    let b = (0..t.unit).rfind(|&i| moved(i))? + 1;
    let inside = t.partner[a..b]
        .iter()
        .all(|&p| (a as i64..b as i64).contains(&p));
    ((a > 0 || b < t.unit) && inside).then(|| {
        let sub = Turn {
            unit: b - a,
            partner: t.partner[a..b].iter().map(|&p| p - a as i64).collect(),
            sign: t.sign[a..b].to_vec(),
            freq: t.freq.iter().map(|fq| fq[a..b].to_vec()).collect(),
            mscale: t.mscale,
        };
        (a, b, sub)
    })
}

/// Applies `t` to an f32 `[rows, width]`, `pos` an f32 `[rows, axes]`.
pub(crate) fn turn(f: &mut Func, x: Val, pos: Val, t: &Turn) -> Result<Val, Error> {
    let dims = f.dims(x).to_vec();
    let (rows, width) = (dims[0], dims[1]);
    let unit = t.unit as i64;
    let groups = width / unit;
    let x3 = f.reshape(x, &[rows, groups, unit])?;
    // Only the columns the table turns go through it (a 64-wide rope tail
    // of a 512-wide head: a lane gather and the sines over every column
    // otherwise, 24.6 of a 171 ms DeepSeek-V4 prompt fire on a v5e).
    if let Some((a, b, sub)) = span(t) {
        let mid = f.slice_axis(x3, 2, a as i64, b as i64)?;
        let mid = f.reshape(mid, &[rows, groups * (b - a) as i64])?;
        let mid = turn(f, mid, pos, &sub)?;
        let mid = f.reshape(mid, &[rows, groups, (b - a) as i64])?;
        let mut parts = Vec::with_capacity(3);
        if a > 0 {
            parts.push(f.slice_axis(x3, 2, 0, a as i64)?);
        }
        parts.push(mid);
        if b < t.unit {
            parts.push(f.slice_axis(x3, 2, b as i64, unit)?);
        }
        let out = f.concat(&parts, 2)?;
        return Ok(f.reshape(out, &[rows, width])?);
    }
    // A prompt's heads narrower than a lane tile turn as the row lies where
    // the chip's profile says so (`RopeFlatRows`): `[rows, heads, unit]`
    // relays every head out to a lane tile and back (v6e gpt-oss 4096-row
    // prefill: 10.8 of a 127 ms fire; 123.4 ms against 125.5). A decode's few
    // rows keep the narrow tables (256 lanes: 32.5 ms a step turned flat,
    // 31.8 not), and a v5e every row (1792 rows: 67.6 flat against 65.5).
    let flat = f.rope_flat_rows();
    if unit < 128
        && width % 128 == 0
        && flat > 0
        && rows >= flat
        && let Some(v) = turn_flat(f, x, pos, t)?
    {
        return Ok(v);
    }
    let mut ang: Option<Val> = None;
    for (axis, freq) in t.freq.iter().enumerate() {
        if freq.iter().all(|&f| f == 0.0) {
            continue;
        }
        let p = f.slice_axis(pos, 1, axis as i64, axis as i64 + 1)?;
        let p = f.reshape(p, &[rows])?;
        let p = f.broadcast(p, &[rows, unit], &[0])?;
        let fs: Vec<f64> = freq.iter().map(|&f| f64::from(f)).collect();
        let fq = f.const_floats(Elem::F32, &fs, &[unit])?;
        let fq = f.broadcast(fq, &[rows, unit], &[1])?;
        let a = f.mul(p, fq)?;
        ang = Some(match ang {
            Some(prev) => f.add(prev, a)?,
            None => a,
        });
    }
    let Some(ang) = ang else {
        return Ok(x);
    };
    let mut c = f.cos(ang);
    let mut s = f.sin(ang);
    if t.mscale != 1.0 {
        let m: Vec<f64> = t
            .sign
            .iter()
            .map(|&g| if g == 0.0 { 1.0 } else { f64::from(t.mscale) })
            .collect();
        let m = f.const_floats(Elem::F32, &m, &[unit])?;
        let m = f.broadcast(m, &[rows, unit], &[1])?;
        c = f.mul(c, m)?;
        s = f.mul(s, m)?;
    }
    let sign: Vec<f64> = t.sign.iter().map(|&g| f64::from(g)).collect();
    let sign = f.const_floats(Elem::F32, &sign, &[unit])?;
    let sign = f.broadcast(sign, &[rows, unit], &[1])?;
    let s = f.mul(s, sign)?;
    let c3 = f.broadcast(c, &[rows, groups, unit], &[0, 2])?;
    let s3 = f.broadcast(s, &[rows, groups, unit], &[0, 2])?;
    let xp = take_cols(f, x3, &t.partner)?;
    let a = f.mul(x3, c3)?;
    let b = f.mul(xp, s3)?;
    let turned = f.add(a, b)?;
    // Columns the table leaves alone pass through untouched.
    let out = if t.sign.contains(&0.0) {
        let moved: Vec<i64> = t.sign.iter().map(|&g| i64::from(g != 0.0)).collect();
        let moved = f.const_ints(Elem::I32, &moved, &[unit])?;
        let zero = f.const_i(Elem::I32, 0, &[unit]);
        let moved = f.compare(Cmp::Ne, moved, zero)?;
        let moved = f.broadcast(moved, &[rows, groups, unit], &[2])?;
        f.select(moved, turned, x3)?
    } else {
        turned
    };
    Ok(f.reshape(out, &[rows, width])?)
}

/// [`turn`] over the row as it lies, every table tiled across its heads: a
/// head narrower than a lane tile never lays out `[rows, heads, unit]`.
/// `None` where a partner sits more than two offsets away.
fn turn_flat(f: &mut Func, x: Val, pos: Val, t: &Turn) -> Result<Option<Val>, Error> {
    let dims = f.dims(x).to_vec();
    let (rows, width) = (dims[0], dims[1]);
    let unit = t.unit;
    let w = usize::try_from(width).unwrap_or(0);
    let col = |c: usize| c % unit;
    let mut shifts: Vec<i64> = Vec::new();
    for c in 0..unit {
        let d = t.partner[c] - c as i64;
        if t.sign[c] != 0.0 && !shifts.contains(&d) {
            shifts.push(d);
        }
    }
    if shifts.len() > 2 {
        return Ok(None);
    }
    let mut ang: Option<Val> = None;
    for (axis, freq) in t.freq.iter().enumerate() {
        if freq.iter().all(|&f| f == 0.0) {
            continue;
        }
        let p = f.slice_axis(pos, 1, axis as i64, axis as i64 + 1)?;
        let p = f.reshape(p, &[rows])?;
        let p = f.broadcast(p, &[rows, width], &[0])?;
        let fs: Vec<f64> = (0..w).map(|c| f64::from(freq[col(c)])).collect();
        let fq = f.const_floats(Elem::F32, &fs, &[width])?;
        let fq = f.broadcast(fq, &[rows, width], &[1])?;
        let a = f.mul(p, fq)?;
        ang = Some(match ang {
            Some(prev) => f.add(prev, a)?,
            None => a,
        });
    }
    let Some(ang) = ang else {
        return Ok(Some(x));
    };
    let mut c = f.cos(ang);
    let mut s = f.sin(ang);
    if t.mscale != 1.0 {
        let m: Vec<f64> = (0..w)
            .map(|i| {
                if t.sign[col(i)] == 0.0 {
                    1.0
                } else {
                    f64::from(t.mscale)
                }
            })
            .collect();
        let m = f.const_floats(Elem::F32, &m, &[width])?;
        let m = f.broadcast(m, &[rows, width], &[1])?;
        c = f.mul(c, m)?;
        s = f.mul(s, m)?;
    }
    let sign: Vec<f64> = (0..w).map(|i| f64::from(t.sign[col(i)])).collect();
    let sign = f.const_floats(Elem::F32, &sign, &[width])?;
    let sign = f.broadcast(sign, &[rows, width], &[1])?;
    let s = f.mul(s, sign)?;
    // Each partner `d` lanes on: the row rolled by `d`, picked where it is.
    let mut xp: Option<Val> = None;
    for &d in &shifts {
        let k = d.rem_euclid(width);
        let rolled = if k == 0 {
            x
        } else {
            let hi = f.slice_axis(x, 1, k, width)?;
            let lo = f.slice_axis(x, 1, 0, k)?;
            f.concat(&[hi, lo], 1)?
        };
        xp = Some(match xp {
            None => rolled,
            Some(prev) => {
                let here: Vec<i64> = (0..w)
                    .map(|i| i64::from(t.partner[col(i)] - col(i) as i64 == d))
                    .collect();
                let here = f.const_ints(Elem::I32, &here, &[width])?;
                let zero = f.const_i(Elem::I32, 0, &[width]);
                let here = f.compare(Cmp::Ne, here, zero)?;
                let here = f.broadcast(here, &[rows, width], &[1])?;
                f.select(here, rolled, prev)?
            }
        });
    }
    let Some(xp) = xp else {
        return Ok(Some(x));
    };
    let a = f.mul(x, c)?;
    let b = f.mul(xp, s)?;
    let turned = f.add(a, b)?;
    Ok(Some(if t.sign.contains(&0.0) {
        let moved: Vec<i64> = (0..w).map(|i| i64::from(t.sign[col(i)] != 0.0)).collect();
        let moved = f.const_ints(Elem::I32, &moved, &[width])?;
        let zero = f.const_i(Elem::I32, 0, &[width]);
        let moved = f.compare(Cmp::Ne, moved, zero)?;
        let moved = f.broadcast(moved, &[rows, width], &[1])?;
        f.select(moved, turned, x)?
    } else {
        turned
    }))
}

/// The position stream as f32 `[rows, axes]`: the first `rows · axes` entries
/// of `positions`, row-major, as the GPU kernels index `positions[axes·row + a]`.
pub(crate) fn positions_f32(
    f: &mut Func,
    op: &'static str,
    positions: Operand,
    rows: u32,
    axes: u32,
) -> Result<Val, Error> {
    let have = positions.elements();
    let want = u64::from(rows) * u64::from(axes);
    if have < want {
        return Err(refuse(
            op,
            format!(
                "the position stream is {} x {}, and this rotation reads {axes} per row for {rows} rows",
                positions.rows, positions.width
            ),
        ));
    }
    let v = positions.val;
    let v = f.reshape(v, &[have as i64])?;
    let v = f.slice(v, &[0], &[want as i64], &[1])?;
    let v = f.reshape(v, &[i64::from(rows), i64::from(axes)])?;
    Ok(f.convert(v, Elem::F32))
}

pub(crate) fn int_positions(op: &'static str, positions: impl Stored) -> Result<(), Error> {
    if positions.dtype() != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the position stream is {:?}, and this rotation reads i32",
                positions.dtype()
            ),
        ));
    }
    Ok(())
}

pub(crate) fn heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    if head_dim == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn rotary_within(op: &'static str, rotary: u32, head_dim: u32) -> Result<(), Error> {
    if rotary == 0 || !rotary.is_multiple_of(2) || rotary > head_dim {
        return Err(refuse(
            op,
            format!(
                "the rotated width {rotary} is not a whole number of pairs within the {head_dim}-wide head"
            ),
        ));
    }
    Ok(())
}

/// `theta^(-2i/span)` in f32, as `powf` computes it on the GPUs.
pub(crate) fn inv_freq(theta: f32, i: usize, span: u32) -> f32 {
    theta.powf(-2.0 * i as f32 / span as f32)
}

/// Rotates each of `xs` (same row count) by one table: the new value of
/// each, `None` for a zero-width one, which is left alone.
pub(crate) fn rotate_all(
    f: &mut Func,
    op: &'static str,
    xs: &[Operand],
    positions: Operand,
    axes: u32,
    t: &Turn,
) -> Result<Vec<Option<Val>>, Error> {
    let rows = xs[0].rows;
    for x in xs {
        expect(op, *x, FLOATS)?;
        if x.rows != rows {
            return Err(refuse(op, "q and k ride the same rows"));
        }
        if !x.width.is_multiple_of(t.unit as u32) {
            return Err(refuse(
                op,
                format!(
                    "the {}-wide row is not a whole number of {}-wide heads",
                    x.width, t.unit
                ),
            ));
        }
    }
    let pos = positions_f32(f, op, positions, rows, axes)?;
    let mut out = Vec::with_capacity(xs.len());
    for x in xs {
        if x.width == 0 {
            out.push(None);
            continue;
        }
        // A run of each head's columns turns: only that run widens to f32,
        // and it is written back over the stored row in place, not the
        // whole row widened and joined again (DeepSeek-V4's 64-wide tail of
        // 64 512-wide heads, a 1792-row prompt on a v6e: 5.9 of a 68 ms fire).
        if let Some((a, b, sub)) = span(t) {
            let (r, w) = (f.dims(x.val)[0], f.dims(x.val)[1]);
            let (g, run) = (w / t.unit as i64, (b - a) as i64);
            let elem = f.elem(x.val);
            let x3 = f.reshape(x.val, &[r, g, t.unit as i64])?;
            let mid = f.slice_axis(x3, 2, a as i64, b as i64)?;
            let mid = f.reshape(mid, &[r, g * run])?;
            let mid = f.convert(mid, Elem::F32);
            let mid = turn(f, mid, pos, &sub)?;
            let mid = f.convert(mid, elem);
            let mid = f.reshape(mid, &[r, g, run])?;
            // Written back over whole 128-lane tiles of the head (the run's
            // untouched neighbours with it), so the update lands in place.
            let (lo, hi) = if t.unit.is_multiple_of(128) {
                (a / 128 * 128, b.div_ceil(128) * 128)
            } else {
                (a, b)
            };
            let mut parts = Vec::with_capacity(3);
            if lo < a {
                parts.push(f.slice_axis(x3, 2, lo as i64, a as i64)?);
            }
            parts.push(mid);
            if b < hi {
                parts.push(f.slice_axis(x3, 2, b as i64, hi as i64)?);
            }
            let mid = f.concat(&parts, 2)?;
            let zero = ci(f, 0);
            let at = ci(f, lo as i64);
            let x3 = f.dynamic_update_slice(x3, mid, &[zero, zero, at])?;
            out.push(Some(f.reshape(x3, &[r, w])?));
            continue;
        }
        if let Some(v) = turn_tall(f, x.val, pos, t)? {
            out.push(Some(v));
            continue;
        }
        let v = x.f32(f);
        out.push(Some(turn(f, v, pos, t)?));
    }
    Ok(out)
}

/// Rows a prompt-tall rotation takes a block at most (`turn_tall`).
const TALL_ROWS: i64 = 256;

/// The rows a row-blocked kernel takes a step for `rows` rows: the most of
/// `most`, 128, 64, 32 and 16 that divides them, so no block reaches past
/// the last row (a partial last block's copies read and write past the
/// array: a v5e gemma-4-E2B c256 hung on the shared tail's 272 readout rows
/// turned as two blocks of 256). `None` when no such block divides them.
pub(crate) fn whole_rows(rows: i64, most: i64) -> Option<i64> {
    [most, 128, 64, 32, 16]
        .into_iter()
        .filter(|&r| r <= most)
        .find(|&r| rows % r == 0)
}

/// Whether each lane of `t` and its partner half a head on turn together, or
/// both pass through (a partial rotation's rest: no angle, so the table's
/// cosine is one and its sine zero there).
fn halves(t: &Turn) -> bool {
    let half = t.unit / 2;
    (0..half).all(|i| {
        let paired = t.partner[i] == (i + half) as i64
            && t.partner[i + half] == i as i64
            && t.sign[i] == -1.0
            && t.sign[i + half] == 1.0
            && t.freq.iter().all(|fq| fq[i] == fq[i + half]);
        let still = t.partner[i] == i as i64
            && t.partner[i + half] == (i + half) as i64
            && t.freq.iter().all(|fq| fq[i] == 0.0 && fq[i + half] == 0.0);
        paired || still
    })
}

/// The table's cosines, and its sines signed, `[rows, d]` in f32 (one
/// position axis).
fn tables(f: &mut Func, pos: Val, rows: i64, t: &Turn) -> Result<(Val, Val), Error> {
    let u = t.unit as i64;
    let p = f.reshape(pos, &[rows])?;
    let p = f.broadcast(p, &[rows, u], &[0])?;
    let fs: Vec<f64> = t.freq[0].iter().map(|&x| f64::from(x)).collect();
    let fq = f.const_floats(Elem::F32, &fs, &[u])?;
    let fq = f.broadcast(fq, &[rows, u], &[1])?;
    let ang = f.mul(p, fq)?;
    let c = f.cos(ang);
    let s = f.sin(ang);
    let sign: Vec<f64> = t.sign.iter().map(|&g| f64::from(g)).collect();
    let sign = f.const_floats(Elem::F32, &sign, &[u])?;
    let sign = f.broadcast(sign, &[rows, u], &[1])?;
    let s = f.mul(s, sign)?;
    Ok((c, s))
}

/// A prompt's bf16 rows turned by one Mosaic kernel, a block of rows a step
/// and each head in place, when `t` turns whole heads a lane tile wide or
/// more by their halves (`(i, i + d/2)`, one position axis): the sines and
/// cosines are the table's, `[rows, d]` once a fire, each head's partner its
/// halves crossed (a lane rotation). XLA lays the same rows out as `[rows,
/// heads, d]` in f32, and back. `None` where it does not apply
/// (`PIE_XLA_ROPE_KERNEL=0` turns it off).
fn turn_tall(f: &mut Func, x: Val, pos: Val, t: &Turn) -> Result<Option<Val>, Error> {
    use crate::mosaic::{Body, Grid, Kernel, Operand as Block, Sem, Window};
    let dims = f.dims(x).to_vec();
    let (rows, width) = (dims[0], dims[1]);
    let unit = t.unit;
    if std::env::var("PIE_XLA_ROPE_KERNEL").is_ok_and(|v| v == "0")
        || f.elem(x) != Elem::Bf16
        || rows < TALL_ROWS
        || !unit.is_multiple_of(128)
        || width % unit as i64 != 0
        || t.freq.len() != 1
        || t.mscale != 1.0
        || !halves(t)
    {
        return Ok(None);
    }
    let u = unit as i64;
    let (c, s) = tables(f, pos, rows, t)?;
    let heads = width / u;
    let Some(tr) = whole_rows(rows, TALL_ROWS) else {
        return Ok(None);
    };
    let nb = rows / tr;
    let block = |w: i64, elem: Elem| Block {
        array: vec![rows, w],
        elem,
        block: vec![tr, w],
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
    };
    let k = Kernel {
        name: "rope_tall".into(),
        grid: vec![nb],
        semantics: vec![Sem::Parallel],
        prefetch: vec![],
        ins: vec![
            block(width, Elem::Bf16),
            block(u, Elem::F32),
            block(u, Elem::F32),
        ],
        outs: vec![block(width, Elem::Bf16)],
        // Two buffers of each block, with room: 32 heads of 256 (gemma-4-31B's
        // `q`) hold 17.8 MB, past a v5e's 16 MB default.
        vmem_limit: Some(u64::try_from(8 * tr * width + 16 * tr * u + (4 << 20)).unwrap_or(0)),
        scratch: vec![],
    };
    let c_ = k
        .build(|b, blk| {
            let i0 = b.index(0);
            let cs = b.vload_all(blk.ins[1])?;
            let sn = b.vload_all(blk.ins[2])?;
            let by = b.i32(u / 2);
            for h in 0..heads {
                let col = b.index(h * u);
                let xh = b.vload(blk.ins[0], &[i0, col], &[tr, u])?;
                let xh = b.convert_f(xh, Elem::F32)?;
                let xp = b.rotate(xh, by, 1)?;
                let a = b.mulf(xh, cs)?;
                let c2 = b.mulf(xp, sn)?;
                let y = b.addf(a, c2)?;
                let y = b.convert_f(y, Elem::Bf16)?;
                b.vstore(y, blk.outs[0], &[i0, col])?;
            }
            Ok(())
        })
        .map_err(Error::from)?;
    Ok(Some(c_.call(f, &[x, c, s]).map_err(Error::from)?[0]))
}

/// The first `rotary/2` pairs turn at `theta^(-i/span)`; a halves pair sits
/// `span` apart.
fn neox(head_dim: u32, rotary: u32, theta: f32, span: u32, interleaved: bool) -> Turn {
    let mut t = Turn::new(head_dim as usize, 1);
    for i in 0..(rotary / 2) as usize {
        let (lo, hi) = if interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + span as usize)
        };
        t.pair(lo, hi, 0, inv_freq(theta, i, 2 * span));
    }
    t
}

/// Full-head rotation; `interleaved` pairs `(2i, 2i+1)`, else `(i, i + d/2)`,
/// at `theta^(-2i/d)`. Reference: kernels-cuda `rope_full` (wgpu refuses the
/// interleaved form; CUDA serves it).
pub fn full(
    f: &mut Func,
    q: Operand,
    k: Operand,
    positions: Operand,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<[Option<Val>; 2], Error> {
    const OP: &str = "elemwise.rope";
    int_positions(OP, positions)?;
    rotary_within(OP, head_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    heads(OP, k.width, head_dim)?;
    let t = neox(head_dim, head_dim, theta, head_dim / 2, interleaved);
    let v = rotate_all(f, OP, &[q, k], positions, 1, &t)?;
    Ok([v[0], v[1]])
}

/// The first `rotary_dim/2` pairs of each head turn, pair `i` being
/// `(i, i + span)` at `theta^(-i/span)`; the rest pass through. Reference:
/// kernels-cuda `rope_partial`, kernels-metal `neox_mb`.
#[allow(clippy::too_many_arguments)]
pub fn partial(
    f: &mut Func,
    q: Operand,
    k: Operand,
    positions: Operand,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<[Option<Val>; 2], Error> {
    const OP: &str = "elemwise.rope";
    int_positions(OP, positions)?;
    rotary_within(OP, rotary_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    heads(OP, k.width, head_dim)?;
    let t = neox(head_dim, rotary_dim, theta, span, false);
    let v = rotate_all(f, OP, &[q, k], positions, 1, &t)?;
    Ok([v[0], v[1]])
}

pub fn partial_q(
    f: &mut Func,
    q: Operand,
    positions: Operand,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<Option<Val>, Error> {
    const OP: &str = "elemwise.rope";
    int_positions(OP, positions)?;
    rotary_within(OP, rotary_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    let t = neox(head_dim, rotary_dim, theta, span, false);
    Ok(rotate_all(f, OP, &[q], positions, 1, &t)?[0])
}

/// A per-head rms norm (`norm::rmsnorm_per_head`, or its `_plus_one` form)
/// whose rows a rotation turns next ([`normed`]): the rows it reads, its
/// `[d]` gain and its epsilon.
#[derive(Clone, Copy, Debug)]
pub struct Normed {
    pub x: Operand,
    pub weight: Operand,
    pub plus_one: bool,
    pub eps: f32,
}

/// [`partial`] (and, `rotary_dim` the head and `span` half of it, [`full`]
/// by halves) of the rows each of `xs` norms to, each head normed and
/// turned by one Mosaic kernel (`turn_normed`): the rows read and written
/// once, not landed by the norm's kernel for the rope's to read. `None`,
/// nothing emitted, unless the kernel takes every one of `xs`.
pub fn normed(
    f: &mut Func,
    xs: &[Normed],
    positions: Operand,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: u32,
) -> Result<Option<Vec<Val>>, Error> {
    const OP: &str = "elemwise.rope";
    int_positions(OP, positions)?;
    rotary_within(OP, rotary_dim, head_dim)?;
    let t = neox(head_dim, rotary_dim, theta, span, false);
    let Some(rows) = xs.first().map(|n| n.x.rows) else {
        return Ok(None);
    };
    let mut blocks = Vec::with_capacity(xs.len());
    for n in xs {
        heads(OP, n.x.width, head_dim)?;
        match normed_rows(f, n, &t) {
            Some(r) if n.x.rows == rows => blocks.push(r),
            _ => return Ok(None),
        }
    }
    let pos = positions_f32(f, OP, positions, rows, 1)?;
    let (c, s) = tables(f, pos, i64::from(rows), &t)?;
    let mut out = Vec::with_capacity(xs.len());
    for (n, r) in xs.iter().zip(blocks) {
        out.push(turn_normed(f, n, c, s, r)?);
    }
    Ok(Some(out))
}

/// The narrowest heads the fused kernel takes: Qwen3-0.6B's 128-wide heads
/// lost 1.9% (v5e pf256-512) where Gemma-4's 256 and 512 gain.
const LEAST_HEAD: i64 = 256;

/// The rows a block of `turn_normed` takes of `n`'s under `t`, where the
/// norm's kernel (`norm::heads_kernel`) and the rope's (`turn_tall`) would
/// each take them: the norm's blocks.
fn normed_rows(f: &Func, n: &Normed, t: &Turn) -> Option<i64> {
    use super::norm::{KERNEL_FROM, KERNEL_ROWS, KERNEL_WIDTH};
    let (rows, width, u) = (i64::from(n.x.rows), i64::from(n.x.width), t.unit as i64);
    let off = |var: &str| std::env::var(var).is_ok_and(|v| v == "0");
    if off("PIE_XLA_NORM_KERNEL")
        || off("PIE_XLA_ROPE_KERNEL")
        || f.elem(n.x.val) != Elem::Bf16
        || i64::from(n.weight.rows) * i64::from(n.weight.width) != u
        || rows < KERNEL_FROM
        || u < LEAST_HEAD
        || u % 128 != 0
        || width % u != 0
        || (width < KERNEL_WIDTH && width / u < 8)
        || t.freq.len() != 1
        || t.mscale != 1.0
        || !halves(t)
    {
        return None;
    }
    let most = if width > 8192 {
        KERNEL_ROWS / 2
    } else {
        KERNEL_ROWS
    };
    whole_rows(rows, most)
}

/// `n`'s bf16 rows, `r` a step, each head normed as `norm::heads_kernel`
/// norms it (rounded to bf16 as that kernel lands it) and turned as
/// `turn_tall` turns it, by the tables `c` and `s`.
fn turn_normed(f: &mut Func, n: &Normed, c: Val, s: Val, r: i64) -> Result<Val, Error> {
    use crate::mosaic::{Body, Grid, Kernel, Operand as Block, Sem, Window};
    let (rows, width, u) = (i64::from(n.x.rows), i64::from(n.x.width), f.dims(c)[1]);
    let heads = width / u;
    let g = f.convert(n.weight.val, Elem::F32);
    let g = f.reshape(g, &[u])?;
    let g = if n.plus_one { f.offset(g, 1.0)? } else { g };
    let g = f.reshape(g, &[1, u])?;
    let block = |w: i64, elem: Elem| Block {
        array: vec![rows, w],
        elem,
        block: vec![r, w],
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
    };
    let gain = Block {
        array: vec![1, u],
        elem: Elem::F32,
        block: vec![1, u],
        window: Window::Blocked,
        index: Box::new(|b: &mut Body, _: &Grid| {
            let z = b.i32(0);
            Ok(vec![z, z])
        }),
    };
    let k = Kernel {
        name: "norm_rope".into(),
        grid: vec![rows / r],
        semantics: vec![Sem::Parallel],
        prefetch: vec![],
        ins: vec![
            block(width, Elem::Bf16),
            gain,
            block(u, Elem::F32),
            block(u, Elem::F32),
        ],
        outs: vec![block(width, Elem::Bf16)],
        // Two buffers of the rows in and out and of the tables, and a head's
        // f32 temporaries.
        vmem_limit: Some(u64::try_from(8 * r * width + 32 * r * u + (4 << 20)).unwrap_or(0)),
        scratch: vec![],
    };
    let eps = n.eps;
    let kc = k
        .build(|b, blk| {
            let i0 = b.index(0);
            let g = b.vload_all(blk.ins[1])?;
            let g = b.broadcast(g, &[r, u])?;
            let cs = b.vload_all(blk.ins[2])?;
            let sn = b.vload_all(blk.ins[3])?;
            let inv_n = b.splat_f(&[r, 1], Elem::F32, 1.0 / u as f64)?;
            let eps = b.splat_f(&[r, 1], Elem::F32, f64::from(eps))?;
            let by = b.i32(u / 2);
            for h in 0..heads {
                let col = b.index(h * u);
                let xh = b.vload(blk.ins[0], &[i0, col], &[r, u])?;
                let xh = b.convert_f(xh, Elem::F32)?;
                let sq = b.mulf(xh, xh)?;
                let sum = b.sum_lanes(sq)?;
                let m = b.mulf(sum, inv_n)?;
                let m = b.addf(m, eps)?;
                let inv = b.rsqrt(m)?;
                let inv = b.broadcast(inv, &[r, u])?;
                let y = b.mulf(xh, inv)?;
                let y = b.mulf(y, g)?;
                let y = b.convert_f(y, Elem::Bf16)?;
                let y = b.convert_f(y, Elem::F32)?;
                let yp = b.rotate(y, by, 1)?;
                let a = b.mulf(y, cs)?;
                let c2 = b.mulf(yp, sn)?;
                let o = b.addf(a, c2)?;
                let o = b.convert_f(o, Elem::Bf16)?;
                b.vstore(o, blk.outs[0], &[i0, col])?;
            }
            Ok(())
        })
        .map_err(Error::from)?;
    Ok(kc.call(f, &[n.x.val, g, c, s]).map_err(Error::from)?[0])
}

/// YaRN ramp bounds over a `head_dim`-wide rotation, as kernels-cuda
/// `ramp_bounds` computes them.
#[allow(clippy::cast_precision_loss)]
fn ramp_bounds(
    head_dim: u32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: u32,
) -> (f32, f32) {
    const TWO_PI: f32 = core::f32::consts::TAU;
    let ln_theta = theta.ln();
    let corr_dim = |rot: f32| -> f32 {
        head_dim as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let low_dim = corr_dim(beta_fast).floor().max(0.0);
    let high_dim = corr_dim(beta_slow)
        .ceil()
        .min((head_dim / 2) as f32 - 1.0)
        .max(low_dim);
    (low_dim, high_dim)
}

/// kernels-cuda `yarn_original_freq`.
fn yarn_freq(base: f32, factor: f32, low: f32, high: f32, i: usize) -> f32 {
    let denom = if high == low {
        high + 1e-3 - low
    } else {
        high - low
    };
    let ramp = ((i as f32 - low) / denom).clamp(0.0, 1.0);
    base * ((1.0 - ramp) + ramp / factor)
}

/// The last `rotary_dim` entries of each head turn at `theta^(-2i/r)`,
/// `(i, i + r/2)` or interleaved `(2i, 2i+1)` from the tail's start;
/// `inverse` turns by `-pos`; a YaRN ramp (factor > 1) bends the frequencies.
/// Reference: kernels-cuda `rope_partial_last`.
pub fn partial_last(
    f: &mut Func,
    q: Operand,
    positions: Operand,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
) -> Result<Option<Val>, Error> {
    const OP: &str = "elemwise.rope";
    int_positions(OP, positions)?;
    rotary_within(OP, rotary_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    let ramp = match yarn {
        Some(y) => {
            if y.original_max_position == 0 {
                return Err(refuse(OP, "the YaRN ramp states a zero position span"));
            }
            let (low, high) = ramp_bounds(
                rotary_dim,
                theta,
                y.beta_fast,
                y.beta_slow,
                y.original_max_position,
            );
            Some((y.factor, low, high))
        }
        None => None,
    };
    let offset = (head_dim - rotary_dim) as usize;
    let half = (rotary_dim / 2) as usize;
    let mut t = Turn::new(head_dim as usize, 1);
    for i in 0..half {
        let mut fq = inv_freq(theta, i, rotary_dim);
        if let Some((factor, low, high)) = ramp
            && factor > 1.0
        {
            fq = yarn_freq(fq, factor, low, high, i);
        }
        if inverse {
            fq = -fq;
        }
        let (lo, hi) = if interleaved {
            (offset + 2 * i, offset + 2 * i + 1)
        } else {
            (offset + i, offset + i + half)
        };
        t.pair(lo, hi, 0, fq);
    }
    Ok(rotate_all(f, OP, &[q], positions, 1, &t)?[0])
}

/// Full-head YaRN: `theta^(-2i/d)` bent by the ramp, cos and sin scaled by
/// `attention_factor`. Reference: kernels-cuda `rope_yarn`.
pub fn yarn(
    f: &mut Func,
    q: Operand,
    k: Operand,
    positions: Operand,
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
) -> Result<[Option<Val>; 2], Error> {
    const OP: &str = "elemwise.rope";
    int_positions(OP, positions)?;
    rotary_within(OP, head_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    heads(OP, k.width, head_dim)?;
    if original_max_position == 0 {
        return Err(refuse(OP, "the YaRN block states a zero position span"));
    }
    let (low, high) = ramp_bounds(head_dim, theta, beta_fast, beta_slow, original_max_position);
    let half = (head_dim / 2) as usize;
    let mut t = Turn::new(head_dim as usize, 1);
    for i in 0..half {
        let fq = yarn_freq(inv_freq(theta, i, head_dim), factor, low, high, i);
        let (lo, hi) = if interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + half)
        };
        t.pair(lo, hi, 0, fq);
    }
    let t = t.scaled(attention_factor);
    let v = rotate_all(f, OP, &[q, k], positions, 1, &t)?;
    Ok([v[0], v[1]])
}

#[cfg(test)]
mod row_block_tests {
    #[test]
    fn row_blocks_are_whole() {
        assert_eq!(super::whole_rows(1792, 256), Some(256));
        assert_eq!(super::whole_rows(640, 256), Some(128));
        assert_eq!(super::whole_rows(272, 256), Some(16));
        assert_eq!(super::whole_rows(2176, 256), Some(128));
        assert_eq!(super::whole_rows(600, 256), None);
    }
}
