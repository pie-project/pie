//! GGUF K-quant projections (Q2_K … Q6_K, dtypes `U2g16k`, `I3g16k`,
//! `U4g32k`, `U5g32k`, `I6g16k`): a stored weight row is `k / 256`
//! super-blocks of the GGUF block bytes, one row per output column.
//!
//! The byte plane is transposed once to `[block_bytes, blocks, n]`, so every
//! block field is a slice of the leading axis and every sub-block split or
//! broadcast touches only leading axes while `n` stays on the TPU's lanes
//! (with the bytes on the minor axis, each field relayouts: ~15× a dense
//! matmul, measured). The fields decode with shift/and over whole planes,
//! exactly as kernels-wgpu `quant/kquant.wgsl` reads them, to a `[256,
//! blocks, n]` weight rounded to bf16 and contracted in one `dot_general`
//! against the activation viewed `[m, blocks, 256]`.
//!
//! Plane layout the engine binds: `w` is `rows = n`, `width = row_bytes =
//! (k / 256) · block_bytes`, dtype `U8` (or the K-quant dtype itself, read
//! as `U8`), the GGUF bytes of each row back to back.

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::linear::gemm::{extent, head_rows};
use crate::tensor::{Landing, Operand, Stored};

const SUPER: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Scheme {
    const fn name(self) -> &'static str {
        match self {
            Self::Q2K => "q2_k",
            Self::Q3K => "q3_k",
            Self::Q4K => "q4_k",
            Self::Q5K => "q5_k",
            Self::Q6K => "q6_k",
        }
    }

    #[must_use]
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }

    const fn of_dtype(dtype: Dtype) -> Option<Self> {
        match dtype {
            Dtype::U2g16k => Some(Self::Q2K),
            Dtype::I3g16k => Some(Self::Q3K),
            Dtype::U4g32k => Some(Self::Q4K),
            Dtype::U5g32k => Some(Self::Q5K),
            Dtype::I6g16k => Some(Self::Q6K),
            _ => None,
        }
    }
}

const FAMILY: [Scheme; 5] = [
    Scheme::Q2K,
    Scheme::Q3K,
    Scheme::Q4K,
    Scheme::Q5K,
    Scheme::Q6K,
];

/// The scheme whose rows over a `k`-wide contraction are `row_bytes` long.
pub fn scheme(op: &'static str, k: u32, row_bytes: u32) -> Result<Scheme, Error> {
    let blocks = k / SUPER;
    for scheme in FAMILY {
        if row_bytes == blocks * scheme.block_bytes() {
            return Ok(scheme);
        }
    }
    let ladder: Vec<String> = FAMILY
        .iter()
        .map(|s| format!("{} ({})", blocks * s.block_bytes(), s.name()))
        .collect();
    Err(refuse(
        op,
        format!(
            "a {row_bytes}-byte weight row is none of the five K-quant widths over a \
             {k}-wide contraction ({blocks} super-blocks): {}",
            ladder.join(", ")
        ),
    ))
}

/// A K-quant plane as it is read: its bytes. The engine reads this view.
pub fn view<T: Stored>(w: T) -> T {
    w.with_dtype(Dtype::U8)
}

/// `y = act · wᵀ` for a K-quant weight read as [`view`] says (the width names
/// the scheme). Nothing when `y` has no rows.
pub fn matmul(f: &mut Func, act: Operand, w: Operand, y: Landing) -> Result<Option<Val>, Error> {
    act_x_wt(f, "linear.matmul", act, w, y)
}

pub fn lm_head(f: &mut Func, act: Operand, w: Operand, y: Landing) -> Result<Option<Val>, Error> {
    act_x_wt(f, "linear.matmul", act, w, y)
}

fn act_x_wt(
    f: &mut Func,
    op: &'static str,
    act: Operand,
    w: Operand,
    y: Landing,
) -> Result<Option<Val>, Error> {
    let (m, n, k) = extent(op, act, y)?;
    if !k.is_multiple_of(SUPER) {
        return Err(refuse(
            op,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    if w.rows != n {
        return Err(refuse(
            op,
            format!(
                "the weight has {} rows and this projection lands {n} columns; a stored \
                 K-quant plane is one row per column",
                w.rows
            ),
        ));
    }
    let scheme = scheme(op, k, w.width)?;
    match (w.dtype, Scheme::of_dtype(w.dtype)) {
        (Dtype::U8, _) => {}
        (_, Some(s)) if s == scheme => {}
        (other, _) => return Err(Error::DtypeUnsupported { op, dtype: other }),
    }
    let w = w.with_dtype(Dtype::U8);
    if m == 0 {
        return Ok(None);
    }
    let wv = dequant(f, w, scheme, k)?;
    let x = act.val;
    let x = head_rows(f, x, m)?;
    let b = i64::from(k / SUPER);
    let x = f.reshape(x, &[i64::from(m), b, i64::from(SUPER)])?;
    let (x, wv) = if f.elem(x) == Elem::Bf16 {
        (x, f.convert(wv, Elem::Bf16))
    } else {
        (f.convert(x, Elem::F32), wv)
    };
    let out = f.dot_general(x, wv, &[], &[], &[1, 2], &[1, 0], Elem::F32)?;
    Ok(Some(out))
}

/// Block field helpers over a u32 `[block_bytes, blocks, n]` byte plane.
struct Blocks {
    v: Val,
    b: i64,
    n: i64,
}

impl Blocks {
    /// Bytes `[lo, hi)` of every block, `[hi − lo, blocks, n]`, split as
    /// `[dims.., blocks, n]`.
    fn field(&self, f: &mut Func, lo: i64, hi: i64, dims: &[i64]) -> Result<Val, Error> {
        let v = f.slice_axis(self.v, 0, lo, hi)?;
        let mut shape = dims.to_vec();
        shape.extend_from_slice(&[self.b, self.n]);
        Ok(f.reshape(v, &shape)?)
    }

    /// The little-endian f16 at byte `at` of every block, f32 `[blocks, n]`.
    fn f16(&self, f: &mut Func, at: i64) -> Result<Val, Error> {
        let lo = self.field(f, at, at + 1, &[])?;
        let hi = self.field(f, at + 1, at + 2, &[])?;
        let hi = shl(f, hi, 8)?;
        let h = f.or(lo, hi)?;
        let h = f.convert(h, Elem::U16);
        let h = f.bitcast(h, Elem::F16)?;
        Ok(f.convert(h, Elem::F32))
    }
}

fn shl(f: &mut Func, v: Val, s: i64) -> Result<Val, Error> {
    let c = f.like_i(v, s);
    Ok(f.shl(v, c)?)
}

fn shr(f: &mut Func, v: Val, s: i64) -> Result<Val, Error> {
    let c = f.like_i(v, s);
    Ok(f.shr(v, c)?)
}

fn and(f: &mut Func, v: Val, m: i64) -> Result<Val, Error> {
    let c = f.like_i(v, m);
    Ok(f.and(v, c)?)
}

/// `v >> Σ step · iota(axis)` for a u32 `v`, over the `(axis, step)` pairs.
fn shr_by(f: &mut Func, v: Val, shifts: &[(i64, i64)]) -> Result<Val, Error> {
    let dims = f.dims(v).to_vec();
    let mut sh: Option<Val> = None;
    for &(axis, step) in shifts {
        let at = f.iota(Elem::U32, &dims, axis);
        let at = if step == 1 {
            at
        } else {
            let s = f.like_i(at, step);
            f.mul(at, s)?
        };
        sh = Some(match sh {
            Some(prev) => f.add(prev, at)?,
            None => at,
        });
    }
    match sh {
        Some(sh) => Ok(f.shr(v, sh)?),
        None => Ok(v),
    }
}

/// `[s, blocks, n]` per-sub-block factors spread over `[s, t, blocks, n]`.
fn spread(f: &mut Func, v: Val, t: i64) -> Result<Val, Error> {
    let d = f.dims(v).to_vec();
    Ok(f.broadcast(v, &[d[0], t, d[1], d[2]], &[0, 2, 3])?)
}

/// `[blocks, n]` per-block factors spread over `[s, blocks, n]`.
fn per_block(f: &mut Func, v: Val, s: i64) -> Result<Val, Error> {
    let d = f.dims(v).to_vec();
    Ok(f.broadcast(v, &[s, d[0], d[1]], &[1, 2])?)
}

fn f32_of(f: &mut Func, v: Val) -> Val {
    f.convert(v, Elem::F32)
}

/// Q4_K / Q5_K's twelve packed 6-bit scale and min bytes at `at`, as f32
/// `[8, blocks, n]` each (`q4k_scale_min`).
fn scale_min(f: &mut Func, blk: &Blocks, at: i64) -> Result<(Val, Val), Error> {
    let s0 = blk.field(f, at, at + 4, &[4])?;
    let s1 = blk.field(f, at + 4, at + 8, &[4])?;
    let s2 = blk.field(f, at + 8, at + 12, &[4])?;
    let sc_lo = and(f, s0, 63)?;
    let m_lo = and(f, s1, 63)?;
    let a = and(f, s2, 15)?;
    let t = shr(f, s0, 6)?;
    let t = shl(f, t, 4)?;
    let sc_hi = f.or(a, t)?;
    let a = shr(f, s2, 4)?;
    let t = shr(f, s1, 6)?;
    let t = shl(f, t, 4)?;
    let m_hi = f.or(a, t)?;
    let sc = f.concat(&[sc_lo, sc_hi], 0)?;
    let m = f.concat(&[m_lo, m_hi], 0)?;
    Ok((f32_of(f, sc), f32_of(f, m)))
}

/// The low/high nibbles of Q4_K / Q5_K's 128 code bytes at `at`, u32
/// `[8, 32, blocks, n]`: sub-block `2p` is the low nibble of byte run `p`,
/// `2p+1` its high nibble.
fn nibbles(f: &mut Func, blk: &Blocks, at: i64) -> Result<Val, Error> {
    let (b, n) = (blk.b, blk.n);
    let qs = blk.field(f, at, at + 128, &[4, 1, 32])?;
    let lo = and(f, qs, 15)?;
    let hi = shr(f, qs, 4)?;
    let q = f.concat(&[lo, hi], 1)?;
    Ok(f.reshape(q, &[8, 32, b, n])?)
}

/// Q2_K / Q3_K's 64 bytes of 2-bit codes at `at`, u32 `[16, 16, blocks,
/// n]`: sub-block `8h + 2s + j`, element `l` is bits `2s` of byte
/// `32h + 16j + l`.
fn crumbs(f: &mut Func, blk: &Blocks, at: i64) -> Result<Val, Error> {
    let (b, n) = (blk.b, blk.n);
    let qs = blk.field(f, at, at + 64, &[2, 2, 16])?;
    let qs = f.broadcast(qs, &[2, 4, 2, 16, b, n], &[0, 2, 3, 4, 5])?;
    let q = shr_by(f, qs, &[(1, 2)])?;
    let q = and(f, q, 3)?;
    Ok(f.reshape(q, &[16, 16, b, n])?)
}

/// A K-quant plane `U8 [n, blocks · block_bytes]` as affine parts: codes
/// `q` f32 `[S, T, blocks, n]` (sub-block, element; non-negative integers
/// below 64) and per-sub-block `scale`, `bias` f32 `[S, blocks, n]`, with
/// `w = scale · q + bias` exactly the shader's weight:
///
/// - Q2_K `d·(sc & 15) · q − dmin·(sc >> 4)`; Q4_K / Q5_K `d·sc · q −
///   dmin·m`: the bias is the min term.
/// - Q3_K `d·(sc − 32) · (q − borrow)`, Q6_K `d·sc · (q − 32)`: the code is
///   shifted non-negative (`q + 4·bit`, `q`) and the bias is `−offset ·
///   scale`.
fn parts(f: &mut Func, w: Operand, scheme: Scheme, k: u32) -> Result<(Val, Val, Val), Error> {
    let n = i64::from(w.rows);
    let b = i64::from(k / SUPER);
    let bb = i64::from(scheme.block_bytes());
    let raw = w.val;
    let raw = f.reshape(raw, &[n, b, bb])?;
    let raw = f.transpose(raw, &[2, 1, 0])?;
    let v = f.convert(raw, Elem::U32);
    let blk = Blocks { v, b, n };
    Ok(match scheme {
        Scheme::Q2K => {
            let sc = blk.field(f, 0, 16, &[16])?;
            let q = crumbs(f, &blk, 16)?;
            let d = blk.f16(f, 80)?;
            let dmin = blk.f16(f, 82)?;
            let lo = and(f, sc, 15)?;
            let lo = f32_of(f, lo);
            let hi = shr(f, sc, 4)?;
            let hi = f32_of(f, hi);
            let d = per_block(f, d, 16)?;
            let dmin = per_block(f, dmin, 16)?;
            let scale = f.mul(d, lo)?;
            let min = f.mul(dmin, hi)?;
            let bias = f.neg(min);
            (f32_of(f, q), scale, bias)
        }
        Scheme::Q3K => {
            let q = crumbs(f, &blk, 32)?;
            let q = f.reshape(q, &[2, 4, 2, 16, b, n])?;
            // The high-bit mask: bit `4h + s` of byte `16j + l`; a clear bit
            // borrows 4, so `q − borrow + 4 = q + 4 · bit`.
            let hm = blk.field(f, 0, 32, &[2, 16])?;
            let hm = f.broadcast(hm, &[2, 4, 2, 16, b, n], &[2, 3, 4, 5])?;
            let bit = shr_by(f, hm, &[(0, 4), (1, 1)])?;
            let bit = and(f, bit, 1)?;
            let bit = shl(f, bit, 2)?;
            let q = f.add(q, bit)?;
            let q = f.reshape(q, &[16, 16, b, n])?;
            // Scales: group `g` of four takes the low (g < 2) or high nibble
            // of bytes `4 · (g & 1) + j`, its top two bits from byte `8 + j`
            // at bit `2g`.
            let s0 = blk.field(f, 96, 100, &[1, 4])?;
            let s1 = blk.field(f, 100, 104, &[1, 4])?;
            let s2 = blk.field(f, 104, 108, &[4])?;
            let g0 = and(f, s0, 15)?;
            let g1 = and(f, s1, 15)?;
            let g2 = shr(f, s0, 4)?;
            let g3 = shr(f, s1, 4)?;
            let low = f.concat(&[g0, g1, g2, g3], 0)?;
            let top = f.broadcast(s2, &[4, 4, b, n], &[1, 2, 3])?;
            let top = shr_by(f, top, &[(0, 2)])?;
            let top = and(f, top, 3)?;
            let top = shl(f, top, 4)?;
            let sc = f.or(low, top)?;
            let sc = f32_of(f, sc);
            let sc = f.offset(sc, -32.0)?;
            let sc = f.reshape(sc, &[16, b, n])?;
            let d = blk.f16(f, 108)?;
            let d = per_block(f, d, 16)?;
            let scale = f.mul(d, sc)?;
            let bias = f.scale(scale, -4.0)?;
            (f32_of(f, q), scale, bias)
        }
        Scheme::Q4K | Scheme::Q5K => {
            let d = blk.f16(f, 0)?;
            let dmin = blk.f16(f, 2)?;
            let (sc, m) = scale_min(f, &blk, 4)?;
            let q = if scheme == Scheme::Q4K {
                nibbles(f, &blk, 16)?
            } else {
                let low = nibbles(f, &blk, 48)?;
                let qh = blk.field(f, 16, 48, &[32])?;
                let qh = f.broadcast(qh, &[8, 32, b, n], &[1, 2, 3])?;
                let fifth = shr_by(f, qh, &[(0, 1)])?;
                let fifth = and(f, fifth, 1)?;
                let fifth = shl(f, fifth, 4)?;
                f.or(low, fifth)?
            };
            let d = per_block(f, d, 8)?;
            let dmin = per_block(f, dmin, 8)?;
            let scale = f.mul(d, sc)?;
            let min = f.mul(dmin, m)?;
            let bias = f.neg(min);
            (f32_of(f, q), scale, bias)
        }
        Scheme::Q6K => {
            // Element `128h + 32q + i`: the low nibble (q < 2) or high nibble
            // (q ≥ 2) of byte `64h + 32 (q & 1) + i`, its top two bits from
            // byte `128 + 32h + i` at bit `2q`; the weight is `q − 32`.
            let ql = blk.field(f, 0, 128, &[2, 1, 2, 32])?;
            let lo = and(f, ql, 15)?;
            let hi = shr(f, ql, 4)?;
            let low = f.concat(&[lo, hi], 1)?;
            let low = f.reshape(low, &[2, 4, 32, b, n])?;
            let qh = blk.field(f, 128, 192, &[2, 32])?;
            let qh = f.broadcast(qh, &[2, 4, 32, b, n], &[0, 2, 3, 4])?;
            let top = shr_by(f, qh, &[(1, 2)])?;
            let top = and(f, top, 3)?;
            let top = shl(f, top, 4)?;
            let q = f.or(low, top)?;
            let q = f.reshape(q, &[16, 16, b, n])?;
            let sc = blk.field(f, 192, 208, &[16])?;
            let sc = f32_of(f, sc);
            let wrap = f.offset(sc, -256.0)?;
            let limit = f.like_f(sc, 127.0);
            let big = f.compare(crate::hlo::Cmp::Gt, sc, limit)?;
            let sc = f.select(big, wrap, sc)?;
            let d = blk.f16(f, 208)?;
            let d = per_block(f, d, 16)?;
            let scale = f.mul(d, sc)?;
            let bias = f.scale(scale, -32.0)?;
            (f32_of(f, q), scale, bias)
        }
    })
}

/// A K-quant plane decoded to f32 `[256, blocks, n]` (element-in-block,
/// block, column).
fn dequant(f: &mut Func, w: Operand, scheme: Scheme, k: u32) -> Result<Val, Error> {
    let (q, scale, bias) = parts(f, w, scheme, k)?;
    let t = f.dims(q)[1];
    let scale = spread(f, scale, t)?;
    let bias = spread(f, bias, t)?;
    let v = f.mul(q, scale)?;
    let v = f.add(v, bias)?;
    let b = f.dims(q)[2];
    Ok(f.reshape(v, &[i64::from(SUPER), b, i64::from(w.rows)])?)
}

/// The sub-block a K-quant scheme's scales cover: the `group` of its
/// [`to_affine`] bank.
#[must_use]
pub const fn affine_group(scheme: Scheme) -> u32 {
    match scheme {
        Scheme::Q4K | Scheme::Q5K => 32,
        Scheme::Q2K | Scheme::Q3K | Scheme::Q6K => 16,
    }
}

/// Repacks a K-quant weight (`w` as [`matmul`] reads it) into an 8-bit
/// affine bank the [`quant`](super::quant) entries serve exactly: `codes`
/// `U8 [n, k]` (one code per byte, row-major), `scales` and `biases` `F32
/// [n, k / affine_group(scheme)]`, `w = scale · code + bias` per sub-block.
/// A load-time transform: the in-graph K-quant decode costs 4–17× a dense
/// matmul per call on v6e, the repacked bank ~1.6–2.6× (measured).
/// Serve it as `Bank { codes, scales, biases: Some(biases), group:
/// affine_group(scheme), bits: 8 }`. `w` is read as its bytes; returns the
/// scheme and the `(codes, scales, biases)` planes.
pub fn to_affine(
    f: &mut Func,
    w: Operand,
    k: u32,
    codes: Landing,
    scales: Landing,
    biases: Landing,
) -> Result<(Scheme, [Val; 3]), Error> {
    const OP: &str = "linear.kquant_to_affine";
    if k == 0 || !k.is_multiple_of(SUPER) {
        return Err(refuse(
            OP,
            format!("K is {k}, not a whole number of {SUPER}-element K-quant super-blocks"),
        ));
    }
    let scheme = scheme(OP, k, w.width)?;
    match (w.dtype, Scheme::of_dtype(w.dtype)) {
        (Dtype::U8, _) => {}
        (_, Some(s)) if s == scheme => {}
        (other, _) => {
            return Err(Error::DtypeUnsupported {
                op: OP,
                dtype: other,
            });
        }
    }
    let n = w.rows;
    let group = affine_group(scheme);
    crate::check::expect(OP, codes, &[Dtype::U8])?;
    crate::check::expect(OP, scales, &[Dtype::F32])?;
    crate::check::expect(OP, biases, &[Dtype::F32])?;
    crate::check::shaped(OP, "code", codes, n, k)?;
    crate::check::shaped(OP, "scale", scales, n, k / group)?;
    crate::check::shaped(OP, "bias", biases, n, k / group)?;
    let w = w.with_dtype(Dtype::U8);
    let (q, scale, bias) = parts(f, w, scheme, k)?;
    // [S, T, B, n] → [n, B, S, T]; [S, B, n] → [n, B, S].
    let q = f.transpose(q, &[3, 2, 0, 1])?;
    let q = f.convert(q, Elem::U8);
    let scale = f.transpose(scale, &[2, 1, 0])?;
    let bias = f.transpose(bias, &[2, 1, 0])?;
    Ok((scheme, [q, scale, bias]))
}
