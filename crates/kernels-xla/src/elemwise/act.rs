//! The elementwise ops engine-wgpu refuses and kernels-metal / kernels-cuda
//! serve: pointwise activations and binaries, DiT modulation, timestep and
//! relative-position tables and multi-axis rotary.
//! Each entry's `///` names the reference it follows.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::Func;
use crate::hlo::{Elem, Val};
use crate::tensor::{Landing, Operand, Stored};

use super::rope::{Turn, positions_f32, turn};

const FLOATS: &[Dtype] = &[Dtype::Bf16, Dtype::F16, Dtype::F32];

fn same_shape(op: &'static str, a: impl Stored, b: impl Stored) -> Result<(), Error> {
    if a.rows() != b.rows() || a.width() != b.width() {
        return Err(refuse(
            op,
            format!(
                "a {}x{} rectangle against a {}x{} one",
                a.rows(),
                a.width(),
                b.rows(),
                b.width()
            ),
        ));
    }
    Ok(())
}

fn unary(
    f: &mut Func,
    op: &'static str,
    x: Operand,
    o: Landing,
    act: fn(&mut Func, Val) -> Result<Val, Error>,
) -> Result<Val, Error> {
    expect(op, o, FLOATS)?;
    same_shape(op, x, o)?;
    let v = x.f32(f);
    act(f, v)
}

/// `o = x·σ(x)`. Reference: kernels-metal `act_silu`.
pub fn silu(f: &mut Func, x: Operand, o: Landing) -> Result<Val, Error> {
    unary(f, "elemwise.act", x, o, |f, v| Ok(f.silu(v)?))
}

/// `o = tanh(x)`. Reference: kernels-metal `act_tanh`.
pub fn tanh(f: &mut Func, x: Operand, o: Landing) -> Result<Val, Error> {
    unary(f, "elemwise.act", x, o, |f, v| Ok(f.tanh(v)))
}

/// `o = √x`. Reference: kernels-metal `act_sqrt`.
pub fn sqrt(f: &mut Func, x: Operand, o: Landing) -> Result<Val, Error> {
    unary(f, "elemwise.act", x, o, |f, v| Ok(f.sqrt(v)))
}

/// `o = ½x(1 + tanh(√(2/π)(x + 0.044715x³)))`. Reference: kernels-metal
/// `act_gelu_tanh`.
pub fn gelu_tanh(f: &mut Func, x: Operand, o: Landing) -> Result<Val, Error> {
    unary(f, "elemwise.act", x, o, |f, v| Ok(f.gelu_tanh(v)?))
}

/// `o = ½x(1 + erf(x/√2))`, erf to 1.5e-7 (A&S 7.1.26), well under bf16.
/// The GPUs refuse `Gelu { tanh: false }`; this serves it.
pub fn gelu_erf(f: &mut Func, x: Operand, o: Landing) -> Result<Val, Error> {
    unary(f, "elemwise.act", x, o, |f, v| Ok(f.gelu_erf(v)?))
}

fn binary(
    f: &mut Func,
    op: &'static str,
    x: Operand,
    y: Operand,
    z: Landing,
    combine: fn(&mut Func, Val, Val) -> Result<Val, Error>,
) -> Result<Val, Error> {
    expect(op, z, FLOATS)?;
    same_shape(op, x, z)?;
    same_shape(op, y, z)?;
    let a = x.f32(f);
    let b = y.f32(f);
    combine(f, a, b)
}

/// `z = x + y`. Reference: kernels-metal `binary_add`.
pub fn add(f: &mut Func, x: Operand, y: Operand, z: Landing) -> Result<Val, Error> {
    binary(f, "elemwise.binary", x, y, z, |f, a, b| Ok(f.add(a, b)?))
}

/// `z = x · y`. Reference: kernels-metal `binary_mul`.
pub fn mul(f: &mut Func, x: Operand, y: Operand, z: Landing) -> Result<Val, Error> {
    binary(f, "elemwise.binary", x, y, z, |f, a, b| Ok(f.mul(a, b)?))
}

/// `z = x / y`. Reference: kernels-metal `binary_div`.
pub fn div(f: &mut Func, x: Operand, y: Operand, z: Landing) -> Result<Val, Error> {
    binary(f, "elemwise.binary", x, y, z, |f, a, b| Ok(f.div(a, b)?))
}

/// `z = x - y·floor(x / y)`: floored, as Python's `%`. Reference:
/// kernels-metal `binary_mod`.
pub fn modulo(f: &mut Func, x: Operand, y: Operand, z: Landing) -> Result<Val, Error> {
    binary(f, "elemwise.binary", x, y, z, |f, a, b| {
        let q = f.div(a, b)?;
        let q = f.floor(q);
        let p = f.mul(b, q)?;
        Ok(f.sub(a, p)?)
    })
}

// ------------------------------------------------------------- modulation

/// How a modulation vector bends a row (poem_ir `ModulateForm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `x·(1 + m[:w]) + m[w:2w]`
    ScaleShift,
    /// `x·(1 + m)`
    Scale,
    /// `tanh(m)·x`
    TanhGate,
}

impl Form {
    const fn vectors(self) -> u32 {
        match self {
            Form::ScaleShift => 2,
            Form::Scale | Form::TanhGate => 1,
        }
    }
}

/// Checks a `vectors · width`-wide modulation plane `m` for `rows` rows.
fn modulation(
    op: &'static str,
    m: impl Stored,
    lane_of_row: Option<impl Stored>,
    x: impl Stored,
    vectors: u32,
) -> Result<(), Error> {
    if m.dtype() != x.dtype() && m.dtype() != Dtype::F32 {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {:?} and the rows it modulates are {:?}; a vector rides \
                 the activation's element or stays f32",
                m.dtype(),
                x.dtype()
            ),
        ));
    }
    if m.width() != vectors * x.width() {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {} wide, and this form reads {vectors} x {}",
                m.width(),
                x.width()
            ),
        ));
    }
    match lane_of_row {
        Some(map) => {
            if map.dtype() != Dtype::I32 || map.elements() != u64::from(x.rows()) {
                return Err(refuse(op, "the lane map is one i32 lane per row"));
            }
        }
        None => {
            if m.rows() < x.rows() {
                return Err(refuse(
                    op,
                    format!("{} modulation vectors for {} rows", m.rows(), x.rows()),
                ));
            }
        }
    }
    Ok(())
}

/// `m`'s vector for each of `rows` rows (row `n` reads `lane_of_row[n]`,
/// else `n`), f32 `[rows, m.width]`.
fn vectors_of(
    f: &mut Func,
    m: Operand,
    lane_of_row: Option<Operand>,
    rows: u32,
) -> Result<Val, Error> {
    let mv = m.f32(f);
    Ok(match lane_of_row {
        Some(map) => {
            let ids = map.val;
            let ids = f.reshape(ids, &[i64::from(rows)])?;
            f.take_rows(mv, ids)?
        }
        None => f.slice_axis(mv, 0, 0, i64::from(rows))?,
    })
}

fn bend(f: &mut Func, x: Val, mv: Val, form: Form) -> Result<Val, Error> {
    let w = f.dims(x)[1];
    let sc = f.slice_axis(mv, 1, 0, w)?;
    Ok(match form {
        Form::ScaleShift => {
            let sh = f.slice_axis(mv, 1, w, 2 * w)?;
            let s1 = f.offset(sc, 1.0)?;
            let v = f.mul(x, s1)?;
            f.add(v, sh)?
        }
        Form::Scale => {
            let s1 = f.offset(sc, 1.0)?;
            f.mul(x, s1)?
        }
        Form::TanhGate => {
            let t = f.tanh(sc);
            f.mul(t, x)?
        }
    })
}

/// `y = form(x, m[lane])`. Reference: kernels-cuda `modulate` (the Metal
/// twin agrees). With `lane_of_row`, `m` is the uncut per-lane plane.
pub fn modulate(
    f: &mut Func,
    form: Form,
    x: Operand,
    m: Operand,
    lane_of_row: Option<Operand>,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.modulate";
    expect(OP, x, FLOATS)?;
    same_shape(OP, x, y)?;
    modulation(OP, m, lane_of_row, x, form.vectors())?;
    let xv = x.f32(f);
    let mv = vectors_of(f, m, lane_of_row, x.rows)?;
    let v = bend(f, xv, mv, form)?;
    Ok(v)
}

/// `r_out = g[lane] · y + r`. Reference: kernels-cuda `gated_residual_add`.
pub fn gated_residual_add(
    f: &mut Func,
    r: Operand,
    g: Operand,
    y: Operand,
    lane_of_row: Option<Operand>,
    r_out: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.residual_add";
    expect(OP, r, FLOATS)?;
    same_shape(OP, r, y)?;
    same_shape(OP, r, r_out)?;
    modulation(OP, g, lane_of_row, r, 1)?;
    let v = gated_sum(f, r, g, y, lane_of_row)?;
    Ok(v)
}

fn gated_sum(
    f: &mut Func,
    r: Operand,
    g: Operand,
    y: Operand,
    lane_of_row: Option<Operand>,
) -> Result<Val, Error> {
    let gv = vectors_of(f, g, lane_of_row, r.rows)?;
    let yv = y.f32(f);
    let rv = r.f32(f);
    let p = f.mul(gv, yv)?;
    Ok(f.add(p, rv)?)
}

// ------------------------------------------------------------------ tables

/// Timestep embedding: `y[n, i] = sin(scale·t[n]·f_i)`, `y[n, i + d/2] =
/// cos(…)` (swapped when `flip_sin_cos`), `f_i = exp(-ln(max_period)·i/(d/2))`;
/// an odd `dim` zeroes the last column. `t` is f32, read flat (`t[n]`).
/// Reference: kernels-metal / kernels-cuda `sinusoid`.
pub fn sinusoid(
    f: &mut Func,
    t: Operand,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.sinusoid";
    if t.dtype != Dtype::F32 || y.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            "the timestep plane and the embedding are both f32",
        ));
    }
    if dim < 2 || y.width != dim {
        return Err(refuse(
            OP,
            format!("a {dim}-wide embedding into a {}-wide plane", y.width),
        ));
    }
    if t.elements() < u64::from(y.rows) {
        return Err(refuse(
            OP,
            format!("{} timesteps for {} rows", t.elements(), y.rows),
        ));
    }
    let half = dim / 2;
    let log_period = max_period.ln();
    let freqs: Vec<f64> = (0..half)
        .map(|i| f64::from((-log_period * i as f32 / half as f32).exp()))
        .collect();
    let (rows, h) = (i64::from(y.rows), i64::from(half));
    let tv = positions_f32(f, OP, t, y.rows, 1)?;
    let tv = f.reshape(tv, &[rows])?;
    let tv = f.broadcast(tv, &[rows, h], &[0])?;
    let fq = f.const_floats(Elem::F32, &freqs, &[h])?;
    let fq = f.broadcast(fq, &[rows, h], &[1])?;
    let a = f.mul(tv, fq)?;
    let a = f.scale(a, f64::from(scale))?;
    let s = f.sin(a);
    let c = f.cos(a);
    let mut parts = if flip_sin_cos { vec![c, s] } else { vec![s, c] };
    if dim % 2 == 1 {
        parts.push(f.const_f(Elem::F32, 0.0, &[rows, 1]));
    }
    let v = f.concat(&parts, 1)?;
    Ok(v)
}

/// T5's bucket for a relative distance `d` (key minus query).
fn relative_bucket(d: i32, bidirectional: bool, num_buckets: i32, log_ratio: f32) -> i32 {
    let mut bucket = 0;
    let mut buckets = num_buckets;
    let n = if bidirectional {
        buckets /= 2;
        if d > 0 {
            bucket += buckets;
        }
        d.abs()
    } else if d < 0 {
        -d
    } else {
        0
    };
    let max_exact = buckets / 2;
    if n < max_exact {
        return bucket + n;
    }
    let x = (n as f32 / max_exact as f32).ln();
    let scaled = x / log_ratio * (buckets - max_exact) as f32;
    let large = (max_exact + scaled as i32).min(buckets - 1);
    bucket + large
}

/// `y[h, c] = embedding[bucket(c - (max_len - 1)), h]`, an f32 `[heads, 2·max_len - 1]`
/// table. Buckets depend only on the stated fields, so they are computed
/// here and the table is one gather. Reference: kernels-metal
/// `relative_bucket_bias`.
pub fn relative_bucket_bias(
    f: &mut Func,
    embedding: Operand,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.relative_bucket_bias";
    expect(OP, embedding, &[Dtype::Bf16, Dtype::F32])?;
    let heads = y.rows;
    if heads == 0 || max_len == 0 {
        return Err(refuse(OP, "no heads or no length"));
    }
    let span = 2 * max_len - 1;
    if y.dtype != Dtype::F32 || y.width != span {
        return Err(refuse(
            OP,
            format!(
                "the table is {} x {} {:?}; this writes one f32 row of {span} per head",
                y.rows, y.width, y.dtype
            ),
        ));
    }
    if embedding.rows < num_buckets || embedding.width < heads {
        return Err(refuse(
            OP,
            format!(
                "the bucket embedding is {} x {}, and the table reads {num_buckets} bucket(s) of {heads} head(s)",
                embedding.rows, embedding.width
            ),
        ));
    }
    let directional = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = directional / 2;
    if max_exact == 0 {
        return Err(refuse(
            OP,
            format!("{num_buckets} bucket(s) leave no exact band"),
        ));
    }
    let ratio = f64::from(max_distance) / f64::from(max_exact);
    if !ratio.is_finite() || ratio <= 1.0 {
        return Err(refuse(
            OP,
            format!("max_distance {max_distance} is at or below max_exact {max_exact}"),
        ));
    }
    #[allow(clippy::cast_possible_truncation)]
    let log_ratio = ratio.ln() as f32;
    let ids: Vec<i64> = (0..span)
        .map(|c| {
            let d = c as i32 - (max_len as i32 - 1);
            i64::from(relative_bucket(
                d,
                bidirectional,
                num_buckets as i32,
                log_ratio,
            ))
        })
        .collect();
    let e = embedding.f32(f);
    let ids = f.const_ints(Elem::I32, &ids, &[i64::from(span)])?;
    let g = f.take_rows(e, ids)?;
    let g = f.slice_axis(g, 1, 0, i64::from(heads))?;
    let g = f.transpose(g, &[1, 0])?;
    Ok(g)
}

// ------------------------------------------------------------ multi-axis rope

/// How a multi-axis rotation lays its pairs (poem_ir `RopeForm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

/// `thetas[axis]^e` as the Metal kernel takes it, `exp(e·ln(base))`.
fn theta_pow(base: f32, e: f32) -> f32 {
    (e * base.ln()).exp()
}

/// Rotary over up to four position axes read from an f32 `[rows, axes]`
/// stream (`positions[axes·row + a]`). Non-ladder forms: the axes own
/// `dims[a]` channels of the `rotary_dim` prefix of each head in order, pair
/// `within` of axis `a` turning at `theta_a^(-2·within/dims[a])` —
/// interleaved `(c + 2j, c + 2j + 1)`, neox `(angle, angle + r/2)`, split
/// `(c + j, c + dims[a]/2 + j)`. `SplitLadder` walks the whole row's
/// `heads · r/2` angles: the first `(heads·r − Σdims)/2` stay put, then the
/// axes take them round-robin at `theta_a^(f/(dims[a]/2 − 1))`. Columns past
/// the rotary prefix are copied. Reference: kernels-metal `rope_axes`.
pub fn rope_axes(
    f: &mut Func,
    x: Operand,
    positions: Operand,
    dims: [u32; 4],
    thetas: [f32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: Landing,
) -> Result<Val, Error> {
    const OP: &str = "elemwise.rope_axes";
    expect(OP, o, FLOATS)?;
    same_shape(OP, x, o)?;
    if positions.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the position stream is {:?}, and this rotation reads f32 coordinates",
                positions.dtype
            ),
        ));
    }
    if head_dim == 0 || !o.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "a {}-wide row is not a whole number of {head_dim}-wide heads",
                o.width
            ),
        ));
    }
    let heads = (o.width / head_dim) as usize;
    let axes = dims.iter().take_while(|d| **d != 0).count();
    if axes == 0 {
        return Err(refuse(OP, "no axis carries a channel"));
    }
    if let Some(odd) = dims[..axes].iter().find(|d| **d % 2 != 0) {
        return Err(refuse(
            OP,
            format!("axis width {odd} is odd, and an angle turns a pair"),
        ));
    }
    let span: u32 = dims[..axes].iter().sum();
    let (hd, r) = (head_dim as usize, rotary_dim as usize);
    let angles = r / 2;
    let table = if form == RopeForm::SplitLadder {
        if rotary_dim == 0 || rotary_dim > head_dim {
            return Err(refuse(
                OP,
                format!("the ladder turns a {rotary_dim}-wide prefix of a {head_dim}-wide head"),
            ));
        }
        let row = heads as u32 * rotary_dim;
        if span == 0 || span > row || !(row - span).is_multiple_of(2) {
            return Err(refuse(
                OP,
                format!("the ladder's axes own {span} channels of a {row}-wide rotated row"),
            ));
        }
        if dims[..axes].iter().any(|d| *d != dims[0]) {
            return Err(refuse(
                OP,
                format!("one ladder hands its axes out round-robin, and {dims:?} is not flat"),
            ));
        }
        let pad = ((row - span) / 2) as usize;
        let mut t = Turn::new(heads * hd, axes);
        for head in 0..heads {
            for angle in 0..angles {
                let idx = head * angles + angle;
                let lo = head * hd + angle;
                let hi = lo + angles;
                if idx < pad {
                    t.still(lo, hi);
                    continue;
                }
                let slot = idx - pad;
                let axis = slot % axes;
                let f = slot / axes;
                let ladder = (dims[axis] / 2) as usize;
                let e = if ladder > 1 {
                    f as f32 / (ladder - 1) as f32
                } else {
                    0.0
                };
                t.pair(lo, hi, axis, theta_pow(thetas[axis], e));
            }
        }
        t
    } else {
        if span != rotary_dim || rotary_dim > head_dim || rotary_dim == 0 {
            return Err(refuse(
                OP,
                format!(
                    "the axes span {span} channels, the rotation states {rotary_dim}, and the head is {head_dim} wide"
                ),
            ));
        }
        let mut t = Turn::new(hd, axes);
        let (mut axis, mut first_angle, mut first_channel) = (0usize, 0usize, 0usize);
        for angle in 0..angles {
            while angle >= first_angle + (dims[axis] / 2) as usize {
                first_angle += (dims[axis] / 2) as usize;
                first_channel += dims[axis] as usize;
                axis += 1;
            }
            let within = angle - first_angle;
            let f = theta_pow(thetas[axis], -2.0 * within as f32 / dims[axis] as f32);
            let (lo, hi) = match form {
                RopeForm::Interleaved => {
                    (first_channel + 2 * within, first_channel + 2 * within + 1)
                }
                RopeForm::Neox => (angle, angle + angles),
                _ => (
                    first_channel + within,
                    first_channel + (dims[axis] / 2) as usize + within,
                ),
            };
            t.pair(lo, hi, axis, f);
        }
        t
    };
    let pos = positions_f32(f, OP, positions, o.rows, axes as u32)?;
    let v = x.f32(f);
    let v = turn(f, v, pos, &table)?;
    Ok(v)
}

// ------------------------------------------------------------ embed + scale
