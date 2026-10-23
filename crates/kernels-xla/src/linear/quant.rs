//! Group-quantized dense projections over a split-plane [`Bank`]: codes
//! packed LSB-first, one scale (and, affine, one bias) per `group` codes of a
//! row.
//!
//! - affine (MLX, `biases` present): `w = s · c + b`, `bits ∈ {2, 4, 8}`,
//!   as kernels-slang `linear/qmv.slang` + `prim/affine.slang` decode it.
//! - symmetric (`biases` absent): the mxfp4 bank the checkpoint pairs with
//!   `RawE8M0` scales, `bits == 4`: `w = e2m1(c) · 2^(e − 127)`, as
//!   `common/mxfp4.inc.wgsl` decodes it. (The GPU shells refuse a dense
//!   symmetric bank and serve it only routed; here it is the same decode.)
//!
//! # How the decode reaches the MXU
//!
//! Unpacking codes along a row (`[n, k/2] → [n, k/2, 2] → [n, k]`) puts a
//! 2-wide axis on the TPU's lane dimension: XLA relayouts and materializes
//! every step, ~10× a dense matmul (measured, 8 chained 4096² layers). So the
//! codes are never interleaved back. Instead each byte's `per = 8 / bits`
//! codes become `per` *planes* (bf16 arithmetic on the byte value: `c_i =
//! ⌊v / 2^(bits·i)⌋ mod 2^bits`, exact), concatenated inside each group:
//! the weight `[n, G, group]` holds a group's codes in the order `(i, j)` for
//! the code at `per·j + i`. The activation is permuted the same way (a
//! transpose of the small `[m, G, group/per, per]` view), so the contraction
//! is unchanged. Then, by `m · G`:
//!
//! - small (decode): one batched dot over the groups, `[G, m, n]` partials
//!   scaled by the group factors and summed (`s · Σ x·c + b · Σ x`, the GPU
//!   qmv's own split); only elementwise work feeds the dot, which XLA fuses.
//! - large (prefill): the weight is decoded to bf16 `[n, k]` once
//!   (`s · c + b`, rounded as a GPU qmm stages its tile) and contracted in
//!   one dense dot.
//!
//! Plane layout the engine binds (see `planes`):
//! - `codes`: `rows = n`; the bank's packed dtype (read as the `U8
//!   [n, k · bits / 8]` bytes it lands as, any width) or already that `U8`
//!   view.
//! - `scales`, `biases`: `[n, k / group]`, bf16 (f16 / f32 also read);
//!   mxfp4 scales are `E8m0` or `U8` bytes `[n, k / 32]`.

use dtype::Dtype;

use crate::check::elem_of;
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Func, Val};
use crate::linear::gemm::{dot_split, extent, head_rows, project};
use crate::tensor::{Bank, Landing, Operand, Stored};
use crate::tuning::Tuning;

/// Whether an `m`-row projection over `groups` groups of a packed bank runs
/// group-batched; past it the weight is decoded once and contracted densely.
pub(crate) fn grouped(tuning: Tuning, m: i64, groups: i64) -> bool {
    m * groups <= tuning.grouped.max_cells
}

/// The bank as its planes are read: an mxfp4 bank's e8m0 scales as their
/// bytes. The engine reads these views.
pub fn bank_view<T: Stored>(w: Bank<T>) -> Bank<T> {
    if w.biases.is_none() && w.bits == 4 && w.group == 32 {
        Bank {
            scales: w.scales.with_dtype(Dtype::U8),
            ..w
        }
    } else {
        w
    }
}

/// A packed plane as a `U8 [rows, row_bytes]` handle over the same bytes: a
/// packed dtype is read as the bytes it lands as; a `U8` view must already
/// hold `row_bytes` per row.
pub(crate) fn byte_view<T: Stored>(
    op: &'static str,
    what: &str,
    t: T,
    rows: u32,
    row_bytes: u64,
) -> Result<T, Error> {
    let width = u32::try_from(row_bytes)
        .map_err(|_| refuse(op, format!("a {row_bytes}-byte {what} row")))?;
    let packed = elem_of(op, t.dtype()).is_err();
    if t.rows() != rows || !(packed || (t.dtype() == Dtype::U8 && t.width() == width)) {
        return Err(refuse(
            op,
            format!(
                "the {what} plane is {}x{} {:?}, and this op wants {rows} rows of \
                 {row_bytes} bytes",
                t.rows(),
                t.width(),
                t.dtype()
            ),
        ));
    }
    Ok(t.with_width(width).with_dtype(Dtype::U8))
}

/// The code planes of a byte plane `u8 [n, G · q]`, concatenated inside each
/// group: bf16 `[n, G, per · q]`, position `i · q + j` holding code `i` of
/// the group's byte `j` (the code at `per · j + i`), as its integer value.
pub(crate) fn code_planes(f: &mut Func, bytes: Val, bits: u32, groups: i64) -> Result<Val, Error> {
    let dims = f.dims(bytes).to_vec();
    let (n, w) = (dims[0], dims[1]);
    let q = w / groups;
    let v = f.reshape(bytes, &[n, groups, q])?;
    let v = f.convert(v, Elem::Bf16);
    let per = 8 / bits;
    if per == 1 {
        return Ok(v);
    }
    let base = f64::from(1u32 << bits);
    let mut rest = v;
    let mut planes = Vec::with_capacity(per as usize);
    for _ in 1..per {
        let h = f.scale(rest, 1.0 / base)?;
        let h = f.floor(h);
        let hb = f.scale(h, base)?;
        planes.push(f.sub(rest, hb)?);
        rest = h;
    }
    planes.push(rest);
    Ok(f.concat(&planes, 2)?)
}

/// The activation `[m, G · per · q]` in [`code_planes`] order:
/// `[m, G, per · q]`, position `i · q + j` of group `g` holding column
/// `g · group + per · j + i`.
pub(crate) fn permute_act(f: &mut Func, x: Val, groups: i64, per: i64) -> Result<Val, Error> {
    let dims = f.dims(x).to_vec();
    let (m, k) = (dims[0], dims[1]);
    let group = k / groups;
    if per == 1 {
        return Ok(f.reshape(x, &[m, groups, group])?);
    }
    let v = f.reshape(x, &[m, groups, group / per, per])?;
    let v = f.transpose(v, &[0, 1, 3, 2])?;
    Ok(f.reshape(v, &[m, groups, group])?)
}

/// `Σ_g factor[n, g] · (x_g · w_gᵀ)` for `xp: [m, G, group]`, `w: [n, G,
/// group]` (exact bf16 values), `factor: f32 [n, G]`: f32 `[m, n]`.
pub(crate) fn grouped_dot(f: &mut Func, xp: Val, w: Val, factor: Val) -> Result<Val, Error> {
    // Fewer than 8 rows run as 8: a 1-row `[G, 1, n]` partial wastes the
    // sublanes and is slower than 8 rows (v6e).
    let rows = f.dims(xp)[0];
    let xp = if rows < 8 {
        let el = f.elem(xp);
        let z = f.const_f(el, 0.0, &[]);
        f.pad(xp, z, &[0, 0, 0], &[8 - rows, 0, 0], &[0, 0, 0])?
    } else {
        xp
    };
    let m = f.dims(xp)[0];
    let (n, g) = (f.dims(w)[0], f.dims(w)[1]);
    let p = if f.elem(xp) == Elem::Bf16 {
        f.dot_general(xp, w, &[1], &[1], &[2], &[2], Elem::F32)?
    } else {
        dot_split(f, xp, w, 3, &[1], &[1], &[2], &[2])?
    };
    let fq = f.broadcast(factor, &[g, m, n], &[2, 0])?;
    let p = f.mul(p, fq)?;
    let out = f.reduce(p, &[0], Fold::Sum)?;
    Ok(if rows < m {
        f.slice_axis(out, 0, 0, rows)?
    } else {
        out
    })
}

/// `Σ_g bias[n, g] · Σ_{k ∈ g} x[m, k]`, the affine offset's share: f32
/// `[m, n]` for `x: [m, G · group]`, `bias: f32 [n, G]`.
pub(crate) fn offset_dot(f: &mut Func, x: Val, bias: Val) -> Result<Val, Error> {
    let (m, k) = (f.dims(x)[0], f.dims(x)[1]);
    let g = f.dims(bias)[1];
    let xs = f.convert(x, Elem::F32);
    let xs = f.reshape(xs, &[m, g, k / g])?;
    let xs = f.reduce(xs, &[2], Fold::Sum)?;
    Ok(f.dot_general(xs, bias, &[], &[], &[1], &[1], Elem::F32)?)
}

/// Group planes `w: [n, G, group]` ([`code_planes`] order) decoded to f32
/// `[n, G · group]`: `w · factor (+ bias)` per group.
pub(crate) fn materialize(
    f: &mut Func,
    w: Val,
    factor: Val,
    bias: Option<Val>,
) -> Result<Val, Error> {
    let d = f.dims(w).to_vec();
    let w = f.convert(w, Elem::F32);
    let fq = f.broadcast(factor, &d, &[0, 1])?;
    let mut v = f.mul(w, fq)?;
    if let Some(b) = bias {
        let b = f.broadcast(b, &d, &[0, 1])?;
        v = f.add(v, b)?;
    }
    Ok(f.reshape(v, &[d[0], d[1] * d[2]])?)
}

/// `x · wᵀ` for packed codes `raw: u8 [n, k · bits / 8]` with one `factor`
/// (and `bias`) per `groups`-th of a row, f32 `[m, n]`. `decode` maps code
/// values (bf16) to weight values before the factor; `exact` states that
/// every decoded weight times its factor is a bf16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract_codes(
    f: &mut Func,
    x: Val,
    raw: Val,
    bits: u32,
    groups: i64,
    factor: Val,
    bias: Option<Val>,
    decode: &dyn Fn(&mut Func, Val) -> Result<Val, Error>,
    exact: bool,
    tuning: Tuning,
) -> Result<Val, Error> {
    let m = f.dims(x)[0];
    let per = i64::from(8 / bits);
    let w = code_planes(f, raw, bits, groups)?;
    let w = decode(f, w)?;
    let xp = permute_act(f, x, groups, per)?;
    if grouped(tuning, m, groups) {
        let mut out = grouped_dot(f, xp, w, factor)?;
        if let Some(b) = bias {
            let o = offset_dot(f, x, b)?;
            out = f.add(out, o)?;
        }
        return Ok(out);
    }
    let wv = materialize(f, w, factor, bias)?;
    let k = f.dims(wv)[1];
    let xp = f.reshape(xp, &[m, k])?;
    project(f, xp, wv, exact)
}

/// The e2m1 value of each code value `c ∈ [0, 16)` of a float array, in its
/// own type: magnitude `m = c mod 8` maps to `{0, .5, 1, 1.5, 2, 3, 4, 6}`
/// (`m/2` below 4, `m − 2` below 6, else `2m − 8`), `c ≥ 8` negative. Exact.
pub(crate) fn e2m1(f: &mut Func, c: Val) -> Result<Val, Error> {
    let eight = f.like_f(c, 8.0);
    let neg = f.compare(Cmp::Ge, c, eight)?;
    let lowered = f.sub(c, eight)?;
    let m = f.select(neg, lowered, c)?;
    let half = f.scale(m, 0.5)?;
    let less2 = f.offset(m, -2.0)?;
    let twice = f.scale(m, 2.0)?;
    let twice = f.offset(twice, -8.0)?;
    let four = f.like_f(c, 4.0);
    let six = f.like_f(c, 6.0);
    let below4 = f.compare(Cmp::Lt, m, four)?;
    let below6 = f.compare(Cmp::Lt, m, six)?;
    let hi = f.select(below6, less2, twice)?;
    let mag = f.select(below4, half, hi)?;
    let nmag = f.neg(mag);
    Ok(f.select(neg, nmag, mag)?)
}

/// `2^(e − 127)` of each e8m0 byte of a u32 array; `0xff` is NaN.
pub(crate) fn e8m0(f: &mut Func, e: Val) -> Result<Val, Error> {
    let s23 = f.like_i(e, 23);
    let bits = f.shl(e, s23)?;
    let v = f.bitcast(bits, Elem::F32)?;
    let zero = f.like_i(e, 0);
    let is_zero = f.compare(Cmp::Eq, e, zero)?;
    let tiny = f.like_f(v, 2f64.powi(-127));
    let v = f.select(is_zero, tiny, v)?;
    let ff = f.like_i(e, 0xff);
    let is_nan = f.compare(Cmp::Eq, e, ff)?;
    let nan = f.like_f(v, f64::NAN);
    Ok(f.select(is_nan, nan, v)?)
}

/// The e4m3 (fn) value of each byte of a u32 array, as the nvfp4 shader's
/// `e4m3_to_f32`: subnormals `m · 2^-9`, `0x7f`/`0xff` NaN.
pub(crate) fn e4m3(f: &mut Func, b: Val) -> Result<Val, Error> {
    let three = f.like_i(b, 3);
    let fifteen = f.like_i(b, 15);
    let seven = f.like_i(b, 7);
    let exp = f.shr(b, three)?;
    let exp = f.and(exp, fifteen)?;
    let man = f.and(b, seven)?;
    // normal: (1 + m/8) · 2^(e − 7) on the f32 bits.
    let bias = f.like_i(b, 120);
    let fe = f.add(exp, bias)?;
    let s23 = f.like_i(b, 23);
    let fe = f.shl(fe, s23)?;
    let s20 = f.like_i(b, 20);
    let fm = f.shl(man, s20)?;
    let normal = f.or(fe, fm)?;
    let normal = f.bitcast(normal, Elem::F32)?;
    let sub = f.convert(man, Elem::F32);
    let sub = f.scale(sub, 2f64.powi(-9))?;
    let zero = f.like_i(b, 0);
    let is_sub = f.compare(Cmp::Eq, exp, zero)?;
    let mag = f.select(is_sub, sub, normal)?;
    let is_top = f.compare(Cmp::Eq, exp, fifteen)?;
    let is_seven = f.compare(Cmp::Eq, man, seven)?;
    let is_nan = f.and(is_top, is_seven)?;
    let nan = f.like_f(mag, f64::NAN);
    let mag = f.select(is_nan, nan, mag)?;
    let x80 = f.like_i(b, 0x80);
    let sign = f.and(b, x80)?;
    let neg = f.compare(Cmp::Ne, sign, zero)?;
    let nmag = f.neg(mag);
    Ok(f.select(neg, nmag, mag)?)
}

/// A bank's codes as read, `[rows, K]`, one per code.
#[derive(Clone, Copy)]
pub(crate) enum Codes {
    /// Integer code values (`ui4`/`ui8` as landed, or unpacked in-graph from
    /// a byte/word plane).
    Int(Val),
    /// e2m1 values (`f4E2M1FN` as landed).
    E2m1(Val),
    /// e2m1 codes as integers (unpacked in-graph), decoded through [`e2m1`].
    E2m1Int(Val),
    /// Pre-scaled weights (`f8E5M2`, see `crate::pack`): no factor applies.
    Weights(Val),
}

impl Codes {
    pub(crate) fn val(self) -> Val {
        match self {
            Self::Int(v) | Self::E2m1(v) | Self::E2m1Int(v) | Self::Weights(v) => v,
        }
    }

    /// The same kind of codes over another value.
    pub(crate) fn with(self, v: Val) -> Self {
        match self {
            Self::Int(_) => Self::Int(v),
            Self::E2m1(_) => Self::E2m1(v),
            Self::E2m1Int(_) => Self::E2m1Int(v),
            Self::Weights(_) => Self::Weights(v),
        }
    }
}

/// Checks a codes handle holds `rows` rows of `k` codes of `bits` (native,
/// see `crate::pack`, or a plain integer plane packing them little-end
/// first), answering the codes per row it stores.
pub(crate) fn codes_shape(
    op: &'static str,
    t: impl Stored,
    rows: u32,
    k: u64,
    bits: u32,
) -> Result<(), Error> {
    let ok = t.rows() == rows
        && match t.dtype() {
            Dtype::E5m2 => u64::from(t.width()) == k,
            // The stored width is the read value's to tell (a packed dtype's
            // handle may view one code per element, or its raw bytes).
            d if crate::pack::code_elem(d).is_some() => crate::pack::code_bits(d) == Some(bits),
            d => match elem_of(op, d) {
                Ok(e) if e.is_int() => {
                    u64::from(t.width()) * u64::from(e.bits()) == k * u64::from(bits)
                }
                _ => false,
            },
        };
    if ok {
        Ok(())
    } else {
        Err(refuse(
            op,
            format!(
                "the code plane is {}x{} {:?}, and this op wants {rows} rows of {k} {bits}-bit codes",
                t.rows(),
                t.width(),
                t.dtype()
            ),
        ))
    }
}

/// Reads a codes handle as `[rows, k]` codes. A plane packing several codes
/// per element is unpacked in-graph (a minor-axis interleave: correct, and
/// slow on TPU — the engine lands codes natively, see `crate::pack`).
pub(crate) fn read_codes(
    f: &mut Func,
    t: Operand,
    bits: u32,
    mxfp4: bool,
    rows: i64,
    k: i64,
) -> Result<Codes, Error> {
    let v = t.val;
    // One code per element (`crate::pack`), or several packed per element.
    let native = matches!(f.elem(v), Elem::U4 | Elem::F4E2m1fn | Elem::F8E5m2)
        || f.ty(v).elements() == rows * k;
    let v = if native { f.reshape(v, &[rows, k])? } else { v };
    Ok(match f.elem(v) {
        Elem::F8E5m2 if native => Codes::Weights(v),
        Elem::F4E2m1fn if native => Codes::E2m1(v),
        _ if native => {
            if mxfp4 {
                Codes::E2m1Int(v)
            } else {
                Codes::Int(v)
            }
        }
        elem => {
            let unsigned = match elem {
                Elem::I8 => Elem::U8,
                Elem::I16 => Elem::U16,
                Elem::I32 => Elem::U32,
                Elem::I64 => Elem::U64,
                e => e,
            };
            let v = if unsigned == elem {
                v
            } else {
                f.bitcast(v, unsigned)?
            };
            let per = i64::from(unsigned.bits() / bits);
            let units = k / per;
            let v = f.reshape(v, &[rows, units])?;
            let dims = [rows, units, per];
            let b = f.broadcast(v, &dims, &[0, 1])?;
            let lane = f.iota(unsigned, &dims, 2);
            let step = f.const_i(unsigned, i64::from(bits), &dims);
            let shift = f.mul(lane, step)?;
            let b = f.shr(b, shift)?;
            let b = if bits < unsigned.bits() {
                let mask = f.const_i(unsigned, (1i64 << bits) - 1, &dims);
                f.and(b, mask)?
            } else {
                b
            };
            let q = f.reshape(b, &[rows, k])?;
            if mxfp4 {
                Codes::E2m1Int(q)
            } else {
                Codes::Int(q)
            }
        }
    })
}

/// The codes' values before any factor, in `elem` (exact: small integers,
/// e2m1 values, or e5m2 weights all hold in bf16).
pub(crate) fn code_values(f: &mut Func, c: Codes, elem: Elem) -> Result<Val, Error> {
    Ok(match c {
        Codes::Int(v) | Codes::E2m1(v) | Codes::Weights(v) => f.convert(v, elem),
        Codes::E2m1Int(v) => {
            let fq = f.convert(v, Elem::F32);
            let fq = e2m1(f, fq)?;
            f.convert(fq, elem)
        }
    })
}

/// `x · wᵀ` for codes `[n, k]` with one `factor` (f32 `[n, G]`, and affine
/// `bias`) per group: f32 `[m, n]`.
pub(crate) fn contract_native(
    f: &mut Func,
    x: Val,
    codes: Codes,
    groups: i64,
    factor: Option<Val>,
    bias: Option<Val>,
    tuning: Tuning,
) -> Result<Val, Error> {
    let m = f.dims(x)[0];
    let (n, k) = {
        let v = match codes {
            Codes::Int(v) | Codes::E2m1(v) | Codes::E2m1Int(v) | Codes::Weights(v) => v,
        };
        (f.dims(v)[0], f.dims(v)[1])
    };
    let (Some(factor), false) = (factor, matches!(codes, Codes::Weights(_))) else {
        let w = code_values(f, codes, Elem::Bf16)?;
        return project(f, x, w, true);
    };
    let q = code_values(f, codes, Elem::Bf16)?;
    let q = f.reshape(q, &[n, groups, k / groups])?;
    // Group-batched: the dot contracts each group on its own and the factors
    // scale the `[G, m, n]` partials, so only a free convert feeds the MXU.
    if m <= tuning.grouped.max_rows {
        let xp = f.reshape(x, &[m, groups, k / groups])?;
        let mut out = grouped_dot(f, xp, q, factor)?;
        if let Some(b) = bias {
            let o = offset_dot(f, x, b)?;
            out = f.add(out, o)?;
        }
        return Ok(out);
    }
    let w = materialize(f, q, factor, bias)?;
    // mxfp4 decodes exactly into bf16; an affine weight is `s · c + b` in f32.
    project(f, x, w, bias.is_none())
}

/// What a bank's planes decode as.
#[derive(Clone, Copy)]
pub(crate) enum Form<T = Operand> {
    Affine { biases: T },
    Mxfp4,
}

/// A bank checked against an `[n, k]` projection, its planes as plain views.
#[derive(Clone, Copy)]
pub(crate) struct Planes<T = Operand> {
    pub codes: T,
    pub scales: T,
    pub form: Form<T>,
    pub bits: u32,
    pub groups: u32,
}

pub(crate) fn planes<T: Stored>(
    op: &'static str,
    w: &Bank<T>,
    n: u32,
    k: u32,
) -> Result<Planes<T>, Error> {
    let (group, bits) = (w.group, w.bits);
    if w.codes.dtype() == Dtype::U4g64tiled {
        return Err(refuse(
            op,
            "the codes are U4g64tiled, in m16n8k16 fragment order (kernels-cuda \
             `linear::tiled`); this plane reads row-major codes, the canonical U4g64",
        ));
    }
    if !matches!(bits, 2 | 4 | 8) {
        return Err(refuse(
            op,
            format!(
                "a {bits}-bit code does not pack whole into a byte; this plane reads 2, 4 or 8"
            ),
        ));
    }
    if group == 0 || !k.is_multiple_of(group) {
        return Err(refuse(
            op,
            format!("the contraction is {k}, not a whole number of {group}-code groups"),
        ));
    }
    if !group.is_multiple_of(8 / bits) {
        return Err(refuse(
            op,
            format!("a {group}-code group splits a byte of {bits}-bit codes"),
        ));
    }
    let groups = k / group;
    codes_shape(op, w.codes, n, u64::from(k), bits)?;
    let codes = w.codes;
    let form = match w.biases {
        Some(b) => Form::Affine { biases: b },
        None if bits == 4 && group == 32 => Form::Mxfp4,
        None => {
            return Err(refuse(
                op,
                format!(
                    "the weight is a symmetric {bits}-bit bank in groups of {group}; the only \
                     symmetric bank read is mxfp4 (4 bits, groups of 32, e8m0 scales)"
                ),
            ));
        }
    };
    let factor = |what: &str, t: T, dtypes: &[Dtype]| -> Result<T, Error> {
        if !dtypes.contains(&t.dtype()) {
            return Err(Error::DtypeUnsupported {
                op,
                dtype: t.dtype(),
            });
        }
        if t.rows() != n || t.width() != groups {
            return Err(refuse(
                op,
                format!(
                    "the {what} plane is {}x{}, and a {n}x{k} bank in groups of {group} \
                     has {n}x{groups}",
                    t.rows(),
                    t.width()
                ),
            ));
        }
        Ok(t)
    };
    let (scales, form) = match form {
        Form::Affine { biases } => {
            let floats = &[Dtype::Bf16, Dtype::F16, Dtype::F32];
            let scales = factor("scale", w.scales, floats)?;
            let biases = factor("bias", biases, floats)?;
            (scales, Form::Affine { biases })
        }
        Form::Mxfp4 => {
            let s = factor("scale", w.scales, &[Dtype::E8m0, Dtype::U8])?;
            (s.with_dtype(Dtype::U8), Form::Mxfp4)
        }
    };
    Ok(Planes {
        codes,
        scales,
        form,
        bits,
        groups,
    })
}

/// `x · wᵀ` for the bank, f32 `[m, n]`; `x` is `[m, k]` bf16 or f32.
pub(crate) fn contract(f: &mut Func, x: Val, p: &Planes, tuning: Tuning) -> Result<Val, Error> {
    let g = i64::from(p.groups);
    let n = i64::from(p.codes.rows);
    let k = f.dims(x)[1];
    let raw = p.codes.val;
    if f.elem(raw) == Elem::U8 && f.ty(raw).elements() != n * k {
        // A byte plane packing several codes: the code-plane path.
        let units = f.ty(raw).elements() / n;
        let raw = f.reshape(raw, &[n, units])?;
        return match p.form {
            Form::Affine { biases } => {
                let s = p.scales.f32(f);
                let b = biases.f32(f);
                contract_codes(
                    f,
                    x,
                    raw,
                    p.bits,
                    g,
                    s,
                    Some(b),
                    &|_, w| Ok(w),
                    false,
                    tuning,
                )
            }
            Form::Mxfp4 => {
                let e = p.scales.val;
                let e = f.convert(e, Elem::U32);
                let s = e8m0(f, e)?;
                contract_codes(
                    f,
                    x,
                    raw,
                    p.bits,
                    g,
                    s,
                    None,
                    &|f, w| e2m1(f, w),
                    true,
                    tuning,
                )
            }
        };
    }
    let mxfp4 = matches!(p.form, Form::Mxfp4);
    let codes = read_codes(f, p.codes, p.bits, mxfp4, n, k)?;
    if let Codes::Weights(_) = codes {
        return contract_native(f, x, codes, g, None, None, tuning);
    }
    match p.form {
        Form::Affine { biases } => {
            let s = p.scales.f32(f);
            let b = biases.f32(f);
            contract_native(f, x, codes, g, Some(s), Some(b), tuning)
        }
        Form::Mxfp4 => {
            let e = p.scales.val;
            let e = f.convert(e, Elem::U32);
            let s = e8m0(f, e)?;
            contract_native(f, x, codes, g, Some(s), None, tuning)
        }
    }
}

/// `y = act · wᵀ` for a quantized bank, its planes read as [`bank_view`]
/// says. Nothing when `y` has no rows.
pub fn matmul(
    f: &mut Func,
    act: Operand,
    w: Bank<Operand>,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    act_x_wt(f, "linear.matmul", act, w, y, tuning)
}

pub fn lm_head(
    f: &mut Func,
    act: Operand,
    w: Bank<Operand>,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    act_x_wt(f, "linear.matmul", act, w, y, tuning)
}

pub fn act_x_wt(
    f: &mut Func,
    op: &'static str,
    act: Operand,
    w: Bank<Operand>,
    y: Landing,
    tuning: Tuning,
) -> Result<Option<Val>, Error> {
    let (m, n, k) = extent(op, act, y)?;
    let p = planes(op, &w, n, k)?;
    if m == 0 {
        return Ok(None);
    }
    let x = act.val;
    let x = head_rows(f, x, m)?;
    let x = if f.elem(x) == Elem::F16 {
        f.convert(x, Elem::F32)
    } else {
        x
    };
    let out = contract(f, x, &p, tuning)?;
    Ok(Some(out))
}
