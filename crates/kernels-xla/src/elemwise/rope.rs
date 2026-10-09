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

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, GatherDims, Val};
use crate::tensor::Tensor;

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
pub(crate) fn take_cols(cx: &mut Cx<'_>, x: Val, idx: &[i64]) -> Result<Val, Error> {
    let dims = cx.dims(x).to_vec();
    let last = dims.len() - 1;
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
            parts.push(cx.slice_axis(x, last, start, start + len)?);
        }
        return Ok(cx.concat(&parts, last as i64)?);
    }
    let n = idx.len() as i64;
    let ids = cx.const_ints(Elem::I32, idx, &[n, 1])?;
    let mut sizes = dims.clone();
    sizes[last] = 1;
    Ok(cx.gather(
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

/// Applies `t` to an f32 `[rows, width]`, `pos` an f32 `[rows, axes]`.
pub(crate) fn turn(cx: &mut Cx<'_>, x: Val, pos: Val, t: &Turn) -> Result<Val, Error> {
    let dims = cx.dims(x).to_vec();
    let (rows, width) = (dims[0], dims[1]);
    let unit = t.unit as i64;
    let groups = width / unit;
    let x3 = cx.reshape(x, &[rows, groups, unit])?;
    let mut ang: Option<Val> = None;
    for (axis, freq) in t.freq.iter().enumerate() {
        if freq.iter().all(|&f| f == 0.0) {
            continue;
        }
        let p = cx.slice_axis(pos, 1, axis as i64, axis as i64 + 1)?;
        let p = cx.reshape(p, &[rows])?;
        let p = cx.broadcast(p, &[rows, unit], &[0])?;
        let fs: Vec<f64> = freq.iter().map(|&f| f64::from(f)).collect();
        let f = cx.const_floats(Elem::F32, &fs, &[unit])?;
        let f = cx.broadcast(f, &[rows, unit], &[1])?;
        let a = cx.mul(p, f)?;
        ang = Some(match ang {
            Some(prev) => cx.add(prev, a)?,
            None => a,
        });
    }
    let Some(ang) = ang else {
        return Ok(x);
    };
    let mut c = cx.cos(ang);
    let mut s = cx.sin(ang);
    if t.mscale != 1.0 {
        let m: Vec<f64> = t
            .sign
            .iter()
            .map(|&g| if g == 0.0 { 1.0 } else { f64::from(t.mscale) })
            .collect();
        let m = cx.const_floats(Elem::F32, &m, &[unit])?;
        let m = cx.broadcast(m, &[rows, unit], &[1])?;
        c = cx.mul(c, m)?;
        s = cx.mul(s, m)?;
    }
    let sign: Vec<f64> = t.sign.iter().map(|&g| f64::from(g)).collect();
    let sign = cx.const_floats(Elem::F32, &sign, &[unit])?;
    let sign = cx.broadcast(sign, &[rows, unit], &[1])?;
    let s = cx.mul(s, sign)?;
    let c3 = cx.broadcast(c, &[rows, groups, unit], &[0, 2])?;
    let s3 = cx.broadcast(s, &[rows, groups, unit], &[0, 2])?;
    let xp = take_cols(cx, x3, &t.partner)?;
    let a = cx.mul(x3, c3)?;
    let b = cx.mul(xp, s3)?;
    let turned = cx.add(a, b)?;
    // Columns the table leaves alone pass through untouched.
    let out = if t.sign.contains(&0.0) {
        let moved: Vec<i64> = t.sign.iter().map(|&g| i64::from(g != 0.0)).collect();
        let moved = cx.const_ints(Elem::I32, &moved, &[unit])?;
        let zero = cx.const_i(Elem::I32, 0, &[unit]);
        let moved = cx.compare(Cmp::Ne, moved, zero)?;
        let moved = cx.broadcast(moved, &[rows, groups, unit], &[2])?;
        cx.select(moved, turned, x3)?
    } else {
        turned
    };
    Ok(cx.reshape(out, &[rows, width])?)
}

/// The position stream as f32 `[rows, axes]`: the first `rows · axes` entries
/// of `positions`, row-major, as the GPU kernels index `positions[axes·row + a]`.
pub(crate) fn positions_f32(
    cx: &mut Cx<'_>,
    op: &'static str,
    positions: Tensor,
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
    let v = cx.read(positions)?;
    let v = cx.reshape(v, &[have as i64])?;
    let v = cx.slice(v, &[0], &[want as i64], &[1])?;
    let v = cx.reshape(v, &[i64::from(rows), i64::from(axes)])?;
    Ok(cx.convert(v, Elem::F32))
}

pub(crate) fn int_positions(op: &'static str, positions: Tensor) -> Result<(), Error> {
    if positions.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the position stream is {:?}, and this rotation reads i32",
                positions.dtype
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

/// Rotates each of `xs` (same row count) in place by one table.
pub(crate) fn rotate_all(
    ctx: &Ctx<'_>,
    op: &'static str,
    xs: &[Tensor],
    positions: Tensor,
    axes: u32,
    t: &Turn,
) -> Result<(), Error> {
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
    ctx.emit(&mut |cx| {
        let pos = positions_f32(cx, op, positions, rows, axes)?;
        for x in xs {
            if x.width == 0 {
                continue;
            }
            let v = cx.read_f32(*x)?;
            let v = turn(cx, v, pos, t)?;
            cx.write(*x, v)?;
        }
        Ok(())
    })
}

fn neox(head_dim: u32, rotary: u32, theta: f32, span: u32, interleaved: bool) -> Turn {
    let mut t = Turn::new(head_dim as usize, 1);
    let half = (rotary / 2) as usize;
    for i in 0..half {
        let (lo, hi) = if interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + half)
        };
        t.pair(lo, hi, 0, inv_freq(theta, i, span));
    }
    t
}

/// Full-head rotation; `interleaved` pairs `(2i, 2i+1)`, else `(i, i + d/2)`,
/// at `theta^(-2i/d)`. Reference: kernels-cuda `rope_full` (wgpu refuses the
/// interleaved form; CUDA serves it).
pub fn full(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_full";
    int_positions(OP, positions)?;
    rotary_within(OP, head_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    heads(OP, k.width, head_dim)?;
    let t = neox(head_dim, head_dim, theta, head_dim, interleaved);
    rotate_all(ctx, OP, &[q, k], positions, 1, &t)
}

/// The first `rotary_dim` entries of each head turn, paired `(i, i + r/2)` at
/// `theta^(-2i/r)` (the HF / mlx-lm / llama.cpp convention); the rest pass
/// through. Reference: kernels-cuda `rope_partial` and kernels-metal
/// `neox_prop_mb` after #603 (kernels-wgpu's `neox_prop` still pairs over
/// the head width and is stale).
pub fn partial(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_partial";
    int_positions(OP, positions)?;
    rotary_within(OP, rotary_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    heads(OP, k.width, head_dim)?;
    let t = neox(head_dim, rotary_dim, theta, rotary_dim, false);
    rotate_all(ctx, OP, &[q, k], positions, 1, &t)
}

pub fn partial_q(
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_partial_q";
    int_positions(OP, positions)?;
    rotary_within(OP, rotary_dim, head_dim)?;
    heads(OP, q.width, head_dim)?;
    let t = neox(head_dim, rotary_dim, theta, rotary_dim, false);
    rotate_all(ctx, OP, &[q], positions, 1, &t)
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
    ctx: &Ctx<'_>,
    q: Tensor,
    positions: Tensor,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_partial_last";
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
        let mut f = inv_freq(theta, i, rotary_dim);
        if let Some((factor, low, high)) = ramp
            && factor > 1.0
        {
            f = yarn_freq(f, factor, low, high, i);
        }
        if inverse {
            f = -f;
        }
        let (lo, hi) = if interleaved {
            (offset + 2 * i, offset + 2 * i + 1)
        } else {
            (offset + i, offset + i + half)
        };
        t.pair(lo, hi, 0, f);
    }
    rotate_all(ctx, OP, &[q], positions, 1, &t)
}

/// Full-head YaRN: `theta^(-2i/d)` bent by the ramp, cos and sin scaled by
/// `attention_factor`. Reference: kernels-cuda `rope_yarn`.
pub fn yarn(
    ctx: &Ctx<'_>,
    q: Tensor,
    k: Tensor,
    positions: Tensor,
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_yarn";
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
        let f = yarn_freq(inv_freq(theta, i, head_dim), factor, low, high, i);
        let (lo, hi) = if interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + half)
        };
        t.pair(lo, hi, 0, f);
    }
    let t = t.scaled(attention_factor);
    rotate_all(ctx, OP, &[q, k], positions, 1, &t)
}
