//! The elementwise ops kernels-wgpu refuses and kernels-metal / kernels-cuda
//! serve: pointwise activations and binaries, DiT modulation, timestep and
//! relative-position tables, multi-axis rotary, and the fused embed/scale/add.
//! Each entry's `///` names the reference it follows.
#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Val};
use crate::tensor::Tensor;

use super::norm::landed;
use super::rope::{Turn, positions_f32, turn};

const FLOATS: &[Dtype] = &[Dtype::Bf16, Dtype::F16, Dtype::F32];

fn same_shape(op: &'static str, a: Tensor, b: Tensor) -> Result<(), Error> {
    if a.rows != b.rows || a.width != b.width {
        return Err(refuse(
            op,
            format!(
                "a {}x{} rectangle against a {}x{} one",
                a.rows, a.width, b.rows, b.width
            ),
        ));
    }
    Ok(())
}

fn unary(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    o: Tensor,
    f: fn(&mut Cx<'_>, Val) -> Result<Val, Error>,
) -> Result<(), Error> {
    expect(op, o, FLOATS)?;
    same_shape(op, x, o)?;
    ctx.emit(&mut |cx| {
        let v = cx.read_f32(x)?;
        let v = f(cx, v)?;
        cx.write(o, v)
    })
}

/// `o = x·σ(x)`. Reference: kernels-metal `act_silu`.
pub fn silu(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    unary(ctx, "elementwise.silu", x, o, |cx, v| Ok(cx.silu(v)?))
}

/// `o = tanh(x)`. Reference: kernels-metal `act_tanh`.
pub fn tanh(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    unary(ctx, "elementwise.tanh", x, o, |cx, v| Ok(cx.tanh(v)))
}

/// `o = ½x(1 + tanh(√(2/π)(x + 0.044715x³)))`. Reference: kernels-metal
/// `act_gelu_tanh`.
pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    unary(ctx, "elementwise.gelu", x, o, |cx, v| Ok(cx.gelu_tanh(v)?))
}

/// `o = ½x(1 + erf(x/√2))`, erf to 1.5e-7 (A&S 7.1.26), well under bf16.
/// The GPUs refuse `Gelu { tanh: false }`; this serves it.
pub fn gelu_erf(ctx: &Ctx<'_>, x: Tensor, o: Tensor) -> Result<(), Error> {
    unary(ctx, "elementwise.gelu", x, o, |cx, v| Ok(cx.gelu_erf(v)?))
}

fn binary(
    ctx: &Ctx<'_>,
    op: &'static str,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    mul: bool,
) -> Result<(), Error> {
    expect(op, z, FLOATS)?;
    same_shape(op, x, z)?;
    same_shape(op, y, z)?;
    ctx.emit(&mut |cx| {
        let a = cx.read_f32(x)?;
        let b = cx.read_f32(y)?;
        let v = if mul { cx.mul(a, b)? } else { cx.add(a, b)? };
        cx.write(z, v)
    })
}

/// `z = x + y`. Reference: kernels-metal `binary_add`.
pub fn add(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    binary(ctx, "elementwise.add", x, y, z, false)
}

/// `z = x · y`. Reference: kernels-metal `binary_mul`.
pub fn mul(ctx: &Ctx<'_>, x: Tensor, y: Tensor, z: Tensor) -> Result<(), Error> {
    binary(ctx, "elementwise.mul", x, y, z, true)
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

/// The scale-free norm a fused modulation runs first (poem_ir `NormKind`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormKind {
    Layernorm { eps: f32 },
    Rmsnorm { head_dim: u32, eps: f32 },
}

/// Checks a `vectors · width`-wide modulation plane `m` for `rows` rows.
fn modulation(
    op: &'static str,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    x: Tensor,
    vectors: u32,
) -> Result<(), Error> {
    if m.dtype != x.dtype && m.dtype != Dtype::F32 {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {:?} and the rows it modulates are {:?}; a vector rides \
                 the activation's element or stays f32",
                m.dtype, x.dtype
            ),
        ));
    }
    if m.width != vectors * x.width {
        return Err(refuse(
            op,
            format!(
                "the modulation plane is {} wide, and this form reads {vectors} x {}",
                m.width, x.width
            ),
        ));
    }
    match lane_of_row {
        Some(map) => {
            if map.dtype != Dtype::I32 || map.elements() != u64::from(x.rows) {
                return Err(refuse(op, "the lane map is one i32 lane per row"));
            }
        }
        None => {
            if m.rows < x.rows {
                return Err(refuse(
                    op,
                    format!("{} modulation vectors for {} rows", m.rows, x.rows),
                ));
            }
        }
    }
    Ok(())
}

/// `m`'s vector for each of `rows` rows (row `n` reads `lane_of_row[n]`,
/// else `n`), f32 `[rows, m.width]`.
fn vectors_of(
    cx: &mut Cx<'_>,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    rows: u32,
) -> Result<Val, Error> {
    let mv = cx.read_f32(m)?;
    Ok(match lane_of_row {
        Some(map) => {
            let ids = cx.read(map)?;
            let ids = cx.reshape(ids, &[i64::from(rows)])?;
            cx.take_rows(mv, ids)?
        }
        None => cx.slice_axis(mv, 0, 0, i64::from(rows))?,
    })
}

fn bend(cx: &mut Cx<'_>, x: Val, mv: Val, form: Form) -> Result<Val, Error> {
    let w = cx.dims(x)[1];
    let sc = cx.slice_axis(mv, 1, 0, w)?;
    Ok(match form {
        Form::ScaleShift => {
            let sh = cx.slice_axis(mv, 1, w, 2 * w)?;
            let s1 = cx.offset(sc, 1.0)?;
            let v = cx.mul(x, s1)?;
            cx.add(v, sh)?
        }
        Form::Scale => {
            let s1 = cx.offset(sc, 1.0)?;
            cx.mul(x, s1)?
        }
        Form::TanhGate => {
            let t = cx.tanh(sc);
            cx.mul(t, x)?
        }
    })
}

/// `y = form(x, m[lane])`. Reference: kernels-cuda `modulate` (the Metal
/// twin agrees). With `lane_of_row`, `m` is the uncut per-lane plane.
pub fn modulate(
    ctx: &Ctx<'_>,
    form: Form,
    x: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.modulate";
    expect(OP, x, FLOATS)?;
    same_shape(OP, x, y)?;
    modulation(OP, m, lane_of_row, x, form.vectors())?;
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let mv = vectors_of(cx, m, lane_of_row, x.rows)?;
        let v = bend(cx, xv, mv, form)?;
        cx.write(y, v)
    })
}

/// `r_out = g[lane] · y + r`. Reference: kernels-cuda `gated_residual_add`.
pub fn gated_residual_add(
    ctx: &Ctx<'_>,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
    r_out: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_add";
    expect(OP, r, FLOATS)?;
    same_shape(OP, r, y)?;
    same_shape(OP, r, r_out)?;
    modulation(OP, g, lane_of_row, r, 1)?;
    ctx.emit(&mut |cx| {
        let v = gated_sum(cx, r, g, y, lane_of_row)?;
        cx.write(r_out, v)
    })
}

fn gated_sum(
    cx: &mut Cx<'_>,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
) -> Result<Val, Error> {
    let gv = vectors_of(cx, g, lane_of_row, r.rows)?;
    let yv = cx.read_f32(y)?;
    let rv = cx.read_f32(r)?;
    let p = cx.mul(gv, yv)?;
    Ok(cx.add(p, rv)?)
}

fn scale_free(cx: &mut Cx<'_>, op: &'static str, x: Val, norm: NormKind) -> Result<Val, Error> {
    match norm {
        NormKind::Layernorm { eps } => super::norm::centered(cx, x, eps),
        NormKind::Rmsnorm { head_dim, eps } => super::norm::inv_rms_rows(cx, op, x, head_dim, eps),
    }
}

/// `normed = norm(x)` (layernorm without affine, or per-head rmsnorm without
/// scale), then `y = form(normed, m[lane])` reading the stored (rounded)
/// `normed`. Reference: kernels-cuda's composed path (`layernorm_no_scale`
/// / `rmsnorm_no_scale`, then `modulate`); the CUDA fused kernel skips the
/// intermediate rounding. Metal and wgpu refuse the op.
pub fn norm_modulate(
    ctx: &Ctx<'_>,
    x: Tensor,
    norm: NormKind,
    normed: Tensor,
    m: Tensor,
    lane_of_row: Option<Tensor>,
    form: Form,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.norm_modulate";
    expect(OP, x, FLOATS)?;
    same_shape(OP, x, normed)?;
    same_shape(OP, x, y)?;
    modulation(OP, m, lane_of_row, x, form.vectors())?;
    ctx.emit(&mut |cx| {
        let xv = cx.read_f32(x)?;
        let n = scale_free(cx, OP, xv, norm)?;
        let n = landed(cx, n, normed.dtype)?;
        cx.write(normed, n)?;
        let mv = vectors_of(cx, m, lane_of_row, x.rows)?;
        let v = bend(cx, n, mv, form)?;
        cx.write(y, v)
    })
}

/// `r_out = g[lane]·y + r`, `normed = norm(r_out)`, `out = form(normed,
/// m[lane])`, each step reading the previous stored value. Reference:
/// kernels-cuda's composed path for `GatedResidualNormModulate`.
pub fn gated_residual_norm_modulate(
    ctx: &Ctx<'_>,
    r: Tensor,
    g: Tensor,
    y: Tensor,
    lane_of_row: Option<Tensor>,
    r_out: Tensor,
    norm: NormKind,
    normed: Tensor,
    m: Tensor,
    form: Form,
    out: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.gated_residual_norm_modulate";
    expect(OP, r, FLOATS)?;
    for t in [y, r_out, normed, out] {
        same_shape(OP, r, t)?;
    }
    modulation(OP, g, lane_of_row, r, 1)?;
    modulation(OP, m, lane_of_row, r, form.vectors())?;
    ctx.emit(&mut |cx| {
        let v = gated_sum(cx, r, g, y, lane_of_row)?;
        let v = landed(cx, v, r_out.dtype)?;
        cx.write(r_out, v)?;
        let n = scale_free(cx, OP, v, norm)?;
        let n = landed(cx, n, normed.dtype)?;
        cx.write(normed, n)?;
        let mv = vectors_of(cx, m, lane_of_row, r.rows)?;
        let o = bend(cx, n, mv, form)?;
        cx.write(out, o)
    })
}

// ------------------------------------------------------------------ tables

/// Timestep embedding: `y[n, i] = sin(scale·t[n]·f_i)`, `y[n, i + d/2] =
/// cos(…)` (swapped when `flip_sin_cos`), `f_i = exp(-ln(max_period)·i/(d/2))`;
/// an odd `dim` zeroes the last column. `t` is f32, read flat (`t[n]`).
/// Reference: kernels-metal / kernels-cuda `sinusoid`.
pub fn sinusoid(
    ctx: &Ctx<'_>,
    t: Tensor,
    dim: u32,
    max_period: f32,
    flip_sin_cos: bool,
    scale: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.sinusoid";
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
    ctx.emit(&mut |cx| {
        let tv = positions_f32(cx, OP, t, y.rows, 1)?;
        let tv = cx.reshape(tv, &[rows])?;
        let tv = cx.broadcast(tv, &[rows, h], &[0])?;
        let f = cx.const_floats(Elem::F32, &freqs, &[h])?;
        let f = cx.broadcast(f, &[rows, h], &[1])?;
        let a = cx.mul(tv, f)?;
        let a = cx.scale(a, f64::from(scale))?;
        let s = cx.sin(a);
        let c = cx.cos(a);
        let mut parts = if flip_sin_cos { vec![c, s] } else { vec![s, c] };
        if dim % 2 == 1 {
            parts.push(cx.const_f(Elem::F32, 0.0, &[rows, 1]));
        }
        let v = cx.concat(&parts, 1)?;
        cx.write(y, v)
    })
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
    ctx: &Ctx<'_>,
    embedding: Tensor,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.relative_bucket_bias";
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
    ctx.emit(&mut |cx| {
        let e = cx.read_f32(embedding)?;
        let ids = cx.const_ints(Elem::I32, &ids, &[i64::from(span)])?;
        let g = cx.take_rows(e, ids)?;
        let g = cx.slice_axis(g, 1, 0, i64::from(heads))?;
        let g = cx.transpose(g, &[1, 0])?;
        cx.write(y, g)
    })
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
    ctx: &Ctx<'_>,
    x: Tensor,
    positions: Tensor,
    dims: [u32; 4],
    thetas: [f32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
    o: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rope_axes";
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
    ctx.emit(&mut |cx| {
        let pos = positions_f32(cx, OP, positions, o.rows, axes as u32)?;
        let v = cx.read_f32(x)?;
        let v = turn(cx, v, pos, &table)?;
        cx.write(o, v)
    })
}

// ------------------------------------------------------------ embed + scale

/// Rows of `table` by token id, ids outside `[0, vocab)` reading row 0.
fn embed_rows(
    cx: &mut Cx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    rows: u32,
) -> Result<Val, Error> {
    let iv = cx.read(ids)?;
    let n = cx.ty(iv).elements();
    let iv = cx.reshape(iv, &[n])?;
    let iv = cx.slice(iv, &[0], &[i64::from(rows)], &[1])?;
    let zero = cx.like_i(iv, 0);
    let top = cx.like_i(iv, i64::from(vocab));
    let ge = cx.compare(Cmp::Ge, iv, zero)?;
    let lt = cx.compare(Cmp::Lt, iv, top)?;
    let ok = cx.and(ge, lt)?;
    let iv = cx.select(ok, iv, zero)?;
    let tv = cx.read(table)?;
    Ok(cx.take_rows(tv, iv)?)
}

/// `e = table[ids]`, `e_scaled = e · bf16(scale)`, both stored; the rounded
/// `e_scaled` is returned in f32.
fn scaled_embed(
    cx: &mut Cx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    e: Tensor,
    scale: f32,
    e_scaled: Tensor,
) -> Result<Val, Error> {
    let ev = embed_rows(cx, ids, table, vocab, e.rows)?;
    cx.write(e, ev)?;
    let ev = cx.convert(ev, Elem::F32);
    let es = cx.scale(ev, bf16_scalar(scale))?;
    let es = landed(cx, es, e_scaled.dtype)?;
    cx.write(e_scaled, es)?;
    Ok(es)
}

fn bf16_scalar(v: f32) -> f64 {
    f64::from(f32::from_bits(u32::from(crate::hlo::bf16_bits(v)) << 16))
}

fn embed_checks(
    op: &'static str,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    e: Tensor,
) -> Result<(), Error> {
    expect(op, table, &[Dtype::Bf16])?;
    if ids.dtype != Dtype::I32 || ids.elements() < u64::from(e.rows) {
        return Err(refuse(op, "one i32 token id per row"));
    }
    if vocab == 0 || table.rows < vocab || table.width != e.width {
        return Err(refuse(
            op,
            format!(
                "a {vocab}-row vocabulary from a {} x {} table into {}-wide rows",
                table.rows, table.width, e.width
            ),
        ));
    }
    Ok(())
}

/// `e = table[ids]`; `e_scaled = e · bf16(embed_scale)`; `y += e_scaled`;
/// `y_scaled = y · bf16(out_scale)`, each stored rounded and read back
/// rounded. Reference: kernels-cuda `layout::embed_scale_add` (the engine
/// reaches it as `elemwise::act::embed_scale_add` here).
pub fn embed_scale_add(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    e: Tensor,
    embed_scale: f32,
    e_scaled: Tensor,
    y: Tensor,
    out_scale: f32,
    y_scaled: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.embed_scale_add";
    embed_checks(OP, ids, table, vocab, e)?;
    for t in [e_scaled, y, y_scaled] {
        same_shape(OP, e, t)?;
    }
    ctx.emit(&mut |cx| {
        let es = scaled_embed(cx, ids, table, vocab, e, embed_scale, e_scaled)?;
        let yv = cx.read_f32(y)?;
        let yv = cx.add(yv, es)?;
        let yv = landed(cx, yv, y.dtype)?;
        cx.write(y, yv)?;
        let ys = cx.scale(yv, bf16_scalar(out_scale))?;
        cx.write(y_scaled, ys)
    })
}

/// As [`embed_scale_add`], the addend being layer `layer`'s `width`-wide
/// column slice of `stacked`, landing in `y_out`. Reference: kernels-cuda
/// `layout::embed_scale_add_select`.
pub fn embed_scale_add_select(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    e: Tensor,
    embed_scale: f32,
    e_scaled: Tensor,
    stacked: Tensor,
    layer: u32,
    width: u32,
    y_out: Tensor,
    out_scale: f32,
    y_scaled: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.embed_scale_add_select";
    embed_checks(OP, ids, table, vocab, e)?;
    for t in [e_scaled, y_out, y_scaled] {
        same_shape(OP, e, t)?;
    }
    let col = u64::from(layer) * u64::from(width);
    if width != y_out.width
        || col + u64::from(width) > u64::from(stacked.width)
        || stacked.rows < y_out.rows
    {
        return Err(refuse(
            OP,
            format!(
                "layer {layer}'s {width}-wide slice does not sit in a {}-wide stacked row landing a {}-wide row",
                stacked.width, y_out.width
            ),
        ));
    }
    let col = col as i64;
    ctx.emit(&mut |cx| {
        let es = scaled_embed(cx, ids, table, vocab, e, embed_scale, e_scaled)?;
        let sv = cx.read_f32(stacked)?;
        let sv = cx.slice(
            sv,
            &[0, col],
            &[i64::from(y_out.rows), col + i64::from(width)],
            &[1, 1],
        )?;
        let yv = cx.add(sv, es)?;
        let yv = landed(cx, yv, y_out.dtype)?;
        cx.write(y_out, yv)?;
        let ys = cx.scale(yv, bf16_scalar(out_scale))?;
        cx.write(y_scaled, ys)
    })
}
