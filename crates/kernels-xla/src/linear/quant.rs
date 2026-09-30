//! Group-quantized dense projections over a split-plane [`Bank`]: codes
//! packed LSB-first, one scale (and, affine, one bias) per `group` codes of a
//! row.
//!
//! - affine (MLX, `biases` present): `w = s · c + b`, `bits ∈ {2, 4, 8}`,
//!   as kernels-wgpu `quant/qmv.wgsl` + `common/affine.inc.wgsl` decode it.
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
//! Plane layout the engine binds (see [`planes`]):
//! - `codes`: `rows = n`; the bank's packed dtype (read as the `U8
//!   [n, k · bits / 8]` bytes it lands as, any width) or already that `U8`
//!   view.
//! - `scales`, `biases`: `[n, k / group]`, bf16 (f16 / f32 also read);
//!   mxfp4 scales are `E8m0` or `U8` bytes `[n, k / 32]`.

use dtype::Dtype;

use crate::cx::{Ctx, Cx, elem_of};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Fold, Val};
use crate::linear::gemm::{dot_split, extent, head_rows, project};
use crate::tensor::{Bank, Tensor};

/// Above this many `rows · groups`, the weight is decoded once and
/// contracted densely instead of group by group (measured on v6e: at 8 rows
/// group-batched wins down to groups of 16; at 64 rows over 64 groups the
/// dense decode wins).
const GROUPED_MAX: i64 = 2048;

/// Whether an `m`-row projection over `groups` groups runs group-batched.
pub(crate) const fn grouped(m: i64, groups: i64) -> bool {
    m * groups <= GROUPED_MAX
}

/// A packed plane as a `U8 [rows, row_bytes]` handle over the same bytes: a
/// packed dtype is read as the bytes it lands as; a `U8` view must already
/// hold `row_bytes` per row.
pub(crate) fn byte_view(
    op: &'static str,
    what: &str,
    t: Tensor,
    rows: u32,
    row_bytes: u64,
) -> Result<Tensor, Error> {
    let width = u32::try_from(row_bytes)
        .map_err(|_| refuse(op, format!("a {row_bytes}-byte {what} row")))?;
    let packed = elem_of(op, t.dtype).is_err();
    if t.rows != rows || !(packed || (t.dtype == Dtype::U8 && t.width == width)) {
        return Err(refuse(
            op,
            format!(
                "the {what} plane is {}x{} {:?}, and this op wants {rows} rows of \
                 {row_bytes} bytes",
                t.rows, t.width, t.dtype
            ),
        ));
    }
    Ok(Tensor::new(t.buf, t.rows, width, Dtype::U8))
}

/// The code planes of a byte plane `u8 [n, G · q]`, concatenated inside each
/// group: bf16 `[n, G, per · q]`, position `i · q + j` holding code `i` of
/// the group's byte `j` (the code at `per · j + i`), as its integer value.
pub(crate) fn code_planes(
    cx: &mut Cx<'_>,
    bytes: Val,
    bits: u32,
    groups: i64,
) -> Result<Val, Error> {
    let dims = cx.dims(bytes).to_vec();
    let (n, w) = (dims[0], dims[1]);
    let q = w / groups;
    let v = cx.reshape(bytes, &[n, groups, q])?;
    let v = cx.convert(v, Elem::Bf16);
    let per = 8 / bits;
    if per == 1 {
        return Ok(v);
    }
    let base = f64::from(1u32 << bits);
    let mut rest = v;
    let mut planes = Vec::with_capacity(per as usize);
    for _ in 1..per {
        let h = cx.scale(rest, 1.0 / base)?;
        let h = cx.floor(h);
        let hb = cx.scale(h, base)?;
        planes.push(cx.sub(rest, hb)?);
        rest = h;
    }
    planes.push(rest);
    Ok(cx.concat(&planes, 2)?)
}

/// The activation `[m, G · per · q]` in [`code_planes`] order:
/// `[m, G, per · q]`, position `i · q + j` of group `g` holding column
/// `g · group + per · j + i`.
pub(crate) fn permute_act(cx: &mut Cx<'_>, x: Val, groups: i64, per: i64) -> Result<Val, Error> {
    let dims = cx.dims(x).to_vec();
    let (m, k) = (dims[0], dims[1]);
    let group = k / groups;
    if per == 1 {
        return Ok(cx.reshape(x, &[m, groups, group])?);
    }
    let v = cx.reshape(x, &[m, groups, group / per, per])?;
    let v = cx.transpose(v, &[0, 1, 3, 2])?;
    Ok(cx.reshape(v, &[m, groups, group])?)
}

/// `Σ_g factor[n, g] · (x_g · w_gᵀ)` for `xp: [m, G, group]`, `w: [n, G,
/// group]` (exact bf16 values), `factor: f32 [n, G]`: f32 `[m, n]`.
pub(crate) fn grouped_dot(cx: &mut Cx<'_>, xp: Val, w: Val, factor: Val) -> Result<Val, Error> {
    // Fewer than 8 rows run as 8: a 1-row `[G, 1, n]` partial wastes the
    // sublanes and is slower than 8 rows (v6e).
    let rows = cx.dims(xp)[0];
    let xp = if rows < 8 {
        let el = cx.elem(xp);
        let z = cx.const_f(el, 0.0, &[]);
        cx.pad(xp, z, &[0, 0, 0], &[8 - rows, 0, 0], &[0, 0, 0])?
    } else {
        xp
    };
    let m = cx.dims(xp)[0];
    let (n, g) = (cx.dims(w)[0], cx.dims(w)[1]);
    let p = if cx.elem(xp) == Elem::Bf16 {
        cx.dot_general(xp, w, &[1], &[1], &[2], &[2], Elem::F32)?
    } else {
        dot_split(cx, xp, w, 3, &[1], &[1], &[2], &[2])?
    };
    let f = cx.broadcast(factor, &[g, m, n], &[2, 0])?;
    let p = cx.mul(p, f)?;
    let out = cx.reduce(p, &[0], Fold::Sum)?;
    Ok(if rows < m {
        cx.slice_axis(out, 0, 0, rows)?
    } else {
        out
    })
}

/// `Σ_g bias[n, g] · Σ_{k ∈ g} x[m, k]`, the affine offset's share: f32
/// `[m, n]` for `x: [m, G · group]`, `bias: f32 [n, G]`.
pub(crate) fn offset_dot(cx: &mut Cx<'_>, x: Val, bias: Val) -> Result<Val, Error> {
    let (m, k) = (cx.dims(x)[0], cx.dims(x)[1]);
    let g = cx.dims(bias)[1];
    let xs = cx.convert(x, Elem::F32);
    let xs = cx.reshape(xs, &[m, g, k / g])?;
    let xs = cx.reduce(xs, &[2], Fold::Sum)?;
    Ok(cx.dot_general(xs, bias, &[], &[], &[1], &[1], Elem::F32)?)
}

/// Group planes `w: [n, G, group]` ([`code_planes`] order) decoded to f32
/// `[n, G · group]`: `w · factor (+ bias)` per group.
pub(crate) fn materialize(
    cx: &mut Cx<'_>,
    w: Val,
    factor: Val,
    bias: Option<Val>,
) -> Result<Val, Error> {
    let d = cx.dims(w).to_vec();
    let w = cx.convert(w, Elem::F32);
    let f = cx.broadcast(factor, &d, &[0, 1])?;
    let mut v = cx.mul(w, f)?;
    if let Some(b) = bias {
        let b = cx.broadcast(b, &d, &[0, 1])?;
        v = cx.add(v, b)?;
    }
    Ok(cx.reshape(v, &[d[0], d[1] * d[2]])?)
}

/// `x · wᵀ` for packed codes `raw: u8 [n, k · bits / 8]` with one `factor`
/// (and `bias`) per `groups`-th of a row, f32 `[m, n]`. `decode` maps code
/// values (bf16) to weight values before the factor; `exact` states that
/// every decoded weight times its factor is a bf16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn contract_codes(
    cx: &mut Cx<'_>,
    x: Val,
    raw: Val,
    bits: u32,
    groups: i64,
    factor: Val,
    bias: Option<Val>,
    decode: &dyn Fn(&mut Cx<'_>, Val) -> Result<Val, Error>,
    exact: bool,
) -> Result<Val, Error> {
    let m = cx.dims(x)[0];
    let per = i64::from(8 / bits);
    let w = code_planes(cx, raw, bits, groups)?;
    let w = decode(cx, w)?;
    let xp = permute_act(cx, x, groups, per)?;
    if grouped(m, groups) {
        let mut out = grouped_dot(cx, xp, w, factor)?;
        if let Some(b) = bias {
            let o = offset_dot(cx, x, b)?;
            out = cx.add(out, o)?;
        }
        return Ok(out);
    }
    let wv = materialize(cx, w, factor, bias)?;
    let k = cx.dims(wv)[1];
    let xp = cx.reshape(xp, &[m, k])?;
    project(cx, xp, wv, exact)
}

/// The e2m1 value of each code value `c ∈ [0, 16)` of a float array, in its
/// own type: magnitude `m = c mod 8` maps to `{0, .5, 1, 1.5, 2, 3, 4, 6}`
/// (`m/2` below 4, `m − 2` below 6, else `2m − 8`), `c ≥ 8` negative. Exact.
pub(crate) fn e2m1(cx: &mut Cx<'_>, c: Val) -> Result<Val, Error> {
    let eight = cx.like_f(c, 8.0);
    let neg = cx.compare(Cmp::Ge, c, eight)?;
    let lowered = cx.sub(c, eight)?;
    let m = cx.select(neg, lowered, c)?;
    let half = cx.scale(m, 0.5)?;
    let less2 = cx.offset(m, -2.0)?;
    let twice = cx.scale(m, 2.0)?;
    let twice = cx.offset(twice, -8.0)?;
    let four = cx.like_f(c, 4.0);
    let six = cx.like_f(c, 6.0);
    let below4 = cx.compare(Cmp::Lt, m, four)?;
    let below6 = cx.compare(Cmp::Lt, m, six)?;
    let hi = cx.select(below6, less2, twice)?;
    let mag = cx.select(below4, half, hi)?;
    let nmag = cx.neg(mag);
    Ok(cx.select(neg, nmag, mag)?)
}

/// `2^(e − 127)` of each e8m0 byte of a u32 array; `0xff` is NaN.
pub(crate) fn e8m0(cx: &mut Cx<'_>, e: Val) -> Result<Val, Error> {
    let s23 = cx.like_i(e, 23);
    let bits = cx.shl(e, s23)?;
    let v = cx.bitcast(bits, Elem::F32)?;
    let zero = cx.like_i(e, 0);
    let is_zero = cx.compare(Cmp::Eq, e, zero)?;
    let tiny = cx.like_f(v, 2f64.powi(-127));
    let v = cx.select(is_zero, tiny, v)?;
    let ff = cx.like_i(e, 0xff);
    let is_nan = cx.compare(Cmp::Eq, e, ff)?;
    let nan = cx.like_f(v, f64::NAN);
    Ok(cx.select(is_nan, nan, v)?)
}

/// The e4m3 (fn) value of each byte of a u32 array, as the nvfp4 shader's
/// `e4m3_to_f32`: subnormals `m · 2^-9`, `0x7f`/`0xff` NaN.
pub(crate) fn e4m3(cx: &mut Cx<'_>, b: Val) -> Result<Val, Error> {
    let three = cx.like_i(b, 3);
    let fifteen = cx.like_i(b, 15);
    let seven = cx.like_i(b, 7);
    let exp = cx.shr(b, three)?;
    let exp = cx.and(exp, fifteen)?;
    let man = cx.and(b, seven)?;
    // normal: (1 + m/8) · 2^(e − 7) on the f32 bits.
    let bias = cx.like_i(b, 120);
    let fe = cx.add(exp, bias)?;
    let s23 = cx.like_i(b, 23);
    let fe = cx.shl(fe, s23)?;
    let s20 = cx.like_i(b, 20);
    let fm = cx.shl(man, s20)?;
    let normal = cx.or(fe, fm)?;
    let normal = cx.bitcast(normal, Elem::F32)?;
    let sub = cx.convert(man, Elem::F32);
    let sub = cx.scale(sub, 2f64.powi(-9))?;
    let zero = cx.like_i(b, 0);
    let is_sub = cx.compare(Cmp::Eq, exp, zero)?;
    let mag = cx.select(is_sub, sub, normal)?;
    let is_top = cx.compare(Cmp::Eq, exp, fifteen)?;
    let is_seven = cx.compare(Cmp::Eq, man, seven)?;
    let is_nan = cx.and(is_top, is_seven)?;
    let nan = cx.like_f(mag, f64::NAN);
    let mag = cx.select(is_nan, nan, mag)?;
    let x80 = cx.like_i(b, 0x80);
    let sign = cx.and(b, x80)?;
    let neg = cx.compare(Cmp::Ne, sign, zero)?;
    let nmag = cx.neg(mag);
    Ok(cx.select(neg, nmag, mag)?)
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
    t: Tensor,
    rows: u32,
    k: u64,
    bits: u32,
) -> Result<(), Error> {
    let ok = t.rows == rows
        && match t.dtype {
            Dtype::E5m2 => u64::from(t.width) == k,
            // The stored width is the read value's to tell (a packed dtype's
            // handle may view one code per element, or its raw bytes).
            d if crate::pack::code_elem(d).is_some() => crate::pack::code_bits(d) == Some(bits),
            d => match elem_of(op, d) {
                Ok(e) if e.is_int() => {
                    u64::from(t.width) * u64::from(e.bits()) == k * u64::from(bits)
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
                t.rows, t.width, t.dtype
            ),
        ))
    }
}

/// Reads a codes handle as `[rows, k]` codes. A plane packing several codes
/// per element is unpacked in-graph (a minor-axis interleave: correct, and
/// slow on TPU — the engine lands codes natively, see `crate::pack`).
pub(crate) fn read_codes(
    cx: &mut Cx<'_>,
    t: Tensor,
    bits: u32,
    mxfp4: bool,
    rows: i64,
    k: i64,
) -> Result<Codes, Error> {
    let v = cx.read(t)?;
    // One code per element (`crate::pack`), or several packed per element.
    let native = matches!(cx.elem(v), Elem::U4 | Elem::F4E2m1fn | Elem::F8E5m2)
        || cx.ty(v).elements() == rows * k;
    let v = if native {
        cx.reshape(v, &[rows, k])?
    } else {
        v
    };
    Ok(match cx.elem(v) {
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
                cx.bitcast(v, unsigned)?
            };
            let per = i64::from(unsigned.bits() / bits);
            let units = k / per;
            let v = cx.reshape(v, &[rows, units])?;
            let dims = [rows, units, per];
            let b = cx.broadcast(v, &dims, &[0, 1])?;
            let lane = cx.iota(unsigned, &dims, 2);
            let step = cx.const_i(unsigned, i64::from(bits), &dims);
            let shift = cx.mul(lane, step)?;
            let b = cx.shr(b, shift)?;
            let b = if bits < unsigned.bits() {
                let mask = cx.const_i(unsigned, (1i64 << bits) - 1, &dims);
                cx.and(b, mask)?
            } else {
                b
            };
            let q = cx.reshape(b, &[rows, k])?;
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
pub(crate) fn code_values(cx: &mut Cx<'_>, c: Codes, elem: Elem) -> Result<Val, Error> {
    Ok(match c {
        Codes::Int(v) | Codes::E2m1(v) | Codes::Weights(v) => cx.convert(v, elem),
        Codes::E2m1Int(v) => {
            let f = cx.convert(v, Elem::F32);
            let f = e2m1(cx, f)?;
            cx.convert(f, elem)
        }
    })
}

/// A projection whose `m` rows at most run group-batched: the dot
/// contracts each group on its own and the factors scale the `[G, m, n]`
/// partials, so only a free convert feeds the MXU (measured on v6e: at bf16
/// speed up to 64 rows; past it decoding the weight once wins).
pub(crate) const GROUPED_ROWS: i64 = 64;

/// `x · wᵀ` for codes `[n, k]` with one `factor` (f32 `[n, G]`, and affine
/// `bias`) per group: f32 `[m, n]`.
pub(crate) fn contract_native(
    cx: &mut Cx<'_>,
    x: Val,
    codes: Codes,
    groups: i64,
    factor: Option<Val>,
    bias: Option<Val>,
) -> Result<Val, Error> {
    let m = cx.dims(x)[0];
    let (n, k) = {
        let v = match codes {
            Codes::Int(v) | Codes::E2m1(v) | Codes::E2m1Int(v) | Codes::Weights(v) => v,
        };
        (cx.dims(v)[0], cx.dims(v)[1])
    };
    let (Some(factor), false) = (factor, matches!(codes, Codes::Weights(_))) else {
        let w = code_values(cx, codes, Elem::Bf16)?;
        return project(cx, x, w, true);
    };
    let q = code_values(cx, codes, Elem::Bf16)?;
    let q = cx.reshape(q, &[n, groups, k / groups])?;
    if m <= GROUPED_ROWS {
        let xp = cx.reshape(x, &[m, groups, k / groups])?;
        let mut out = grouped_dot(cx, xp, q, factor)?;
        if let Some(b) = bias {
            let o = offset_dot(cx, x, b)?;
            out = cx.add(out, o)?;
        }
        return Ok(out);
    }
    let w = materialize(cx, q, factor, bias)?;
    // mxfp4 decodes exactly into bf16; an affine weight is `s · c + b` in f32.
    project(cx, x, w, bias.is_none())
}

/// What a bank's planes decode as.
#[derive(Clone, Copy)]
pub(crate) enum Form {
    Affine { biases: Tensor },
    Mxfp4,
}

/// A bank checked against an `[n, k]` projection, its planes as plain views.
#[derive(Clone, Copy)]
pub(crate) struct Planes {
    pub codes: Tensor,
    pub scales: Tensor,
    pub form: Form,
    pub bits: u32,
    pub groups: u32,
}

pub(crate) fn planes(op: &'static str, w: &Bank, n: u32, k: u32) -> Result<Planes, Error> {
    let (group, bits) = (w.group, w.bits);
    if w.codes.dtype == Dtype::U4g64tiled {
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
    let factor = |what: &str, t: Tensor, dtypes: &[Dtype]| -> Result<Tensor, Error> {
        if !dtypes.contains(&t.dtype) {
            return Err(Error::DtypeUnsupported { op, dtype: t.dtype });
        }
        if t.rows != n || t.width != groups {
            return Err(refuse(
                op,
                format!(
                    "the {what} plane is {}x{}, and a {n}x{k} bank in groups of {group} \
                     has {n}x{groups}",
                    t.rows, t.width
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
            (Tensor::new(s.buf, s.rows, s.width, Dtype::U8), Form::Mxfp4)
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
pub(crate) fn contract(cx: &mut Cx<'_>, x: Val, p: &Planes) -> Result<Val, Error> {
    let g = i64::from(p.groups);
    let n = i64::from(p.codes.rows);
    let k = cx.dims(x)[1];
    let raw = cx.read(p.codes)?;
    if cx.elem(raw) == Elem::U8 && cx.ty(raw).elements() != n * k {
        // A byte plane packing several codes: the code-plane path.
        let units = cx.ty(raw).elements() / n;
        let raw = cx.reshape(raw, &[n, units])?;
        return match p.form {
            Form::Affine { biases } => {
                let s = cx.read_f32(p.scales)?;
                let b = cx.read_f32(biases)?;
                contract_codes(cx, x, raw, p.bits, g, s, Some(b), &|_, w| Ok(w), false)
            }
            Form::Mxfp4 => {
                let e = cx.read(p.scales)?;
                let e = cx.convert(e, Elem::U32);
                let s = e8m0(cx, e)?;
                contract_codes(cx, x, raw, p.bits, g, s, None, &|cx, w| e2m1(cx, w), true)
            }
        };
    }
    let mxfp4 = matches!(p.form, Form::Mxfp4);
    let codes = read_codes(cx, p.codes, p.bits, mxfp4, n, k)?;
    if let Codes::Weights(_) = codes {
        return contract_native(cx, x, codes, g, None, None);
    }
    match p.form {
        Form::Affine { biases } => {
            let s = cx.read_f32(p.scales)?;
            let b = cx.read_f32(biases)?;
            contract_native(cx, x, codes, g, Some(s), Some(b))
        }
        Form::Mxfp4 => {
            let e = cx.read(p.scales)?;
            let e = cx.convert(e, Elem::U32);
            let s = e8m0(cx, e)?;
            contract_native(cx, x, codes, g, Some(s), None)
        }
    }
}

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Bank, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Bank, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Bank,
    y: Tensor,
) -> Result<(), Error> {
    let (m, n, k) = extent(op, act, y)?;
    let p = planes(op, &w, n, k)?;
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let x = cx.read(act)?;
        let x = head_rows(cx, x, m)?;
        let x = if cx.elem(x) == Elem::F16 {
            cx.convert(x, Elem::F32)
        } else {
            x
        };
        let out = contract(cx, x, &p)?;
        cx.write(y, out)
    })
}
