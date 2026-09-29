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

use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Val};
use crate::linear::gemm::{extent, head_rows};
use crate::tensor::Tensor;

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

const FAMILY: [Scheme; 5] = [Scheme::Q2K, Scheme::Q3K, Scheme::Q4K, Scheme::Q5K, Scheme::Q6K];

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

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

fn act_x_wt(ctx: &Ctx<'_>, op: &'static str, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
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
    let w = Tensor::new(w.buf, w.rows, w.width, Dtype::U8);
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let wv = dequant(cx, w, scheme, k)?;
        let x = cx.read(act)?;
        let x = head_rows(cx, x, m)?;
        let b = i64::from(k / SUPER);
        let x = cx.reshape(x, &[i64::from(m), b, i64::from(SUPER)])?;
        let (x, wv) = if cx.elem(x) == Elem::Bf16 {
            (x, cx.convert(wv, Elem::Bf16))
        } else {
            (cx.convert(x, Elem::F32), wv)
        };
        let out = cx.dot_general(x, wv, &[], &[], &[1, 2], &[1, 0], Elem::F32)?;
        cx.write(y, out)
    })
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
    fn field(&self, cx: &mut Cx<'_>, lo: i64, hi: i64, dims: &[i64]) -> Result<Val, Error> {
        let f = cx.slice_axis(self.v, 0, lo, hi)?;
        let mut shape = dims.to_vec();
        shape.extend_from_slice(&[self.b, self.n]);
        Ok(cx.reshape(f, &shape)?)
    }

    /// The little-endian f16 at byte `at` of every block, f32 `[blocks, n]`.
    fn f16(&self, cx: &mut Cx<'_>, at: i64) -> Result<Val, Error> {
        let lo = self.field(cx, at, at + 1, &[])?;
        let hi = self.field(cx, at + 1, at + 2, &[])?;
        let hi = shl(cx, hi, 8)?;
        let h = cx.or(lo, hi)?;
        let h = cx.convert(h, Elem::U16);
        let h = cx.bitcast(h, Elem::F16)?;
        Ok(cx.convert(h, Elem::F32))
    }
}

fn shl(cx: &mut Cx<'_>, v: Val, s: i64) -> Result<Val, Error> {
    let c = cx.like_i(v, s);
    Ok(cx.shl(v, c)?)
}

fn shr(cx: &mut Cx<'_>, v: Val, s: i64) -> Result<Val, Error> {
    let c = cx.like_i(v, s);
    Ok(cx.shr(v, c)?)
}

fn and(cx: &mut Cx<'_>, v: Val, m: i64) -> Result<Val, Error> {
    let c = cx.like_i(v, m);
    Ok(cx.and(v, c)?)
}

/// `v >> Σ step · iota(axis)` for a u32 `v`, over the `(axis, step)` pairs.
fn shr_by(cx: &mut Cx<'_>, v: Val, shifts: &[(i64, i64)]) -> Result<Val, Error> {
    let dims = cx.dims(v).to_vec();
    let mut sh: Option<Val> = None;
    for &(axis, step) in shifts {
        let at = cx.iota(Elem::U32, &dims, axis);
        let at = if step == 1 {
            at
        } else {
            let s = cx.like_i(at, step);
            cx.mul(at, s)?
        };
        sh = Some(match sh {
            Some(prev) => cx.add(prev, at)?,
            None => at,
        });
    }
    match sh {
        Some(sh) => Ok(cx.shr(v, sh)?),
        None => Ok(v),
    }
}

/// `[s, blocks, n]` per-sub-block factors spread over `[s, t, blocks, n]`.
fn spread(cx: &mut Cx<'_>, v: Val, t: i64) -> Result<Val, Error> {
    let d = cx.dims(v).to_vec();
    Ok(cx.broadcast(v, &[d[0], t, d[1], d[2]], &[0, 2, 3])?)
}

/// `[blocks, n]` per-block factors spread over `[s, blocks, n]`.
fn per_block(cx: &mut Cx<'_>, v: Val, s: i64) -> Result<Val, Error> {
    let d = cx.dims(v).to_vec();
    Ok(cx.broadcast(v, &[s, d[0], d[1]], &[1, 2])?)
}

fn f32_of(cx: &mut Cx<'_>, v: Val) -> Val {
    cx.convert(v, Elem::F32)
}

/// Q4_K / Q5_K's twelve packed 6-bit scale and min bytes at `at`, as f32
/// `[8, blocks, n]` each (`q4k_scale_min`).
fn scale_min(cx: &mut Cx<'_>, blk: &Blocks, at: i64) -> Result<(Val, Val), Error> {
    let s0 = blk.field(cx, at, at + 4, &[4])?;
    let s1 = blk.field(cx, at + 4, at + 8, &[4])?;
    let s2 = blk.field(cx, at + 8, at + 12, &[4])?;
    let sc_lo = and(cx, s0, 63)?;
    let m_lo = and(cx, s1, 63)?;
    let a = and(cx, s2, 15)?;
    let t = shr(cx, s0, 6)?;
    let t = shl(cx, t, 4)?;
    let sc_hi = cx.or(a, t)?;
    let a = shr(cx, s2, 4)?;
    let t = shr(cx, s1, 6)?;
    let t = shl(cx, t, 4)?;
    let m_hi = cx.or(a, t)?;
    let sc = cx.concat(&[sc_lo, sc_hi], 0)?;
    let m = cx.concat(&[m_lo, m_hi], 0)?;
    Ok((f32_of(cx, sc), f32_of(cx, m)))
}

/// The low/high nibbles of Q4_K / Q5_K's 128 code bytes at `at`, u32
/// `[8, 32, blocks, n]`: sub-block `2p` is the low nibble of byte run `p`,
/// `2p+1` its high nibble.
fn nibbles(cx: &mut Cx<'_>, blk: &Blocks, at: i64) -> Result<Val, Error> {
    let (b, n) = (blk.b, blk.n);
    let qs = blk.field(cx, at, at + 128, &[4, 1, 32])?;
    let lo = and(cx, qs, 15)?;
    let hi = shr(cx, qs, 4)?;
    let q = cx.concat(&[lo, hi], 1)?;
    Ok(cx.reshape(q, &[8, 32, b, n])?)
}

/// Q2_K / Q3_K's 64 bytes of 2-bit codes at `at`, u32 `[16, 16, blocks,
/// n]`: sub-block `8h + 2s + j`, element `l` is bits `2s` of byte
/// `32h + 16j + l`.
fn crumbs(cx: &mut Cx<'_>, blk: &Blocks, at: i64) -> Result<Val, Error> {
    let (b, n) = (blk.b, blk.n);
    let qs = blk.field(cx, at, at + 64, &[2, 2, 16])?;
    let qs = cx.broadcast(qs, &[2, 4, 2, 16, b, n], &[0, 2, 3, 4, 5])?;
    let q = shr_by(cx, qs, &[(1, 2)])?;
    let q = and(cx, q, 3)?;
    Ok(cx.reshape(q, &[16, 16, b, n])?)
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
fn parts(cx: &mut Cx<'_>, w: Tensor, scheme: Scheme, k: u32) -> Result<(Val, Val, Val), Error> {
    let n = i64::from(w.rows);
    let b = i64::from(k / SUPER);
    let bb = i64::from(scheme.block_bytes());
    let raw = cx.read(w)?;
    let raw = cx.reshape(raw, &[n, b, bb])?;
    let raw = cx.transpose(raw, &[2, 1, 0])?;
    let v = cx.convert(raw, Elem::U32);
    let blk = Blocks { v, b, n };
    Ok(match scheme {
        Scheme::Q2K => {
            let sc = blk.field(cx, 0, 16, &[16])?;
            let q = crumbs(cx, &blk, 16)?;
            let d = blk.f16(cx, 80)?;
            let dmin = blk.f16(cx, 82)?;
            let lo = and(cx, sc, 15)?;
            let lo = f32_of(cx, lo);
            let hi = shr(cx, sc, 4)?;
            let hi = f32_of(cx, hi);
            let d = per_block(cx, d, 16)?;
            let dmin = per_block(cx, dmin, 16)?;
            let scale = cx.mul(d, lo)?;
            let min = cx.mul(dmin, hi)?;
            let bias = cx.neg(min);
            (f32_of(cx, q), scale, bias)
        }
        Scheme::Q3K => {
            let q = crumbs(cx, &blk, 32)?;
            let q = cx.reshape(q, &[2, 4, 2, 16, b, n])?;
            // The high-bit mask: bit `4h + s` of byte `16j + l`; a clear bit
            // borrows 4, so `q − borrow + 4 = q + 4 · bit`.
            let hm = blk.field(cx, 0, 32, &[2, 16])?;
            let hm = cx.broadcast(hm, &[2, 4, 2, 16, b, n], &[2, 3, 4, 5])?;
            let bit = shr_by(cx, hm, &[(0, 4), (1, 1)])?;
            let bit = and(cx, bit, 1)?;
            let bit = shl(cx, bit, 2)?;
            let q = cx.add(q, bit)?;
            let q = cx.reshape(q, &[16, 16, b, n])?;
            // Scales: group `g` of four takes the low (g < 2) or high nibble
            // of bytes `4 · (g & 1) + j`, its top two bits from byte `8 + j`
            // at bit `2g`.
            let s0 = blk.field(cx, 96, 100, &[1, 4])?;
            let s1 = blk.field(cx, 100, 104, &[1, 4])?;
            let s2 = blk.field(cx, 104, 108, &[4])?;
            let g0 = and(cx, s0, 15)?;
            let g1 = and(cx, s1, 15)?;
            let g2 = shr(cx, s0, 4)?;
            let g3 = shr(cx, s1, 4)?;
            let low = cx.concat(&[g0, g1, g2, g3], 0)?;
            let top = cx.broadcast(s2, &[4, 4, b, n], &[1, 2, 3])?;
            let top = shr_by(cx, top, &[(0, 2)])?;
            let top = and(cx, top, 3)?;
            let top = shl(cx, top, 4)?;
            let sc = cx.or(low, top)?;
            let sc = f32_of(cx, sc);
            let sc = cx.offset(sc, -32.0)?;
            let sc = cx.reshape(sc, &[16, b, n])?;
            let d = blk.f16(cx, 108)?;
            let d = per_block(cx, d, 16)?;
            let scale = cx.mul(d, sc)?;
            let bias = cx.scale(scale, -4.0)?;
            (f32_of(cx, q), scale, bias)
        }
        Scheme::Q4K | Scheme::Q5K => {
            let d = blk.f16(cx, 0)?;
            let dmin = blk.f16(cx, 2)?;
            let (sc, m) = scale_min(cx, &blk, 4)?;
            let q = if scheme == Scheme::Q4K {
                nibbles(cx, &blk, 16)?
            } else {
                let low = nibbles(cx, &blk, 48)?;
                let qh = blk.field(cx, 16, 48, &[32])?;
                let qh = cx.broadcast(qh, &[8, 32, b, n], &[1, 2, 3])?;
                let fifth = shr_by(cx, qh, &[(0, 1)])?;
                let fifth = and(cx, fifth, 1)?;
                let fifth = shl(cx, fifth, 4)?;
                cx.or(low, fifth)?
            };
            let d = per_block(cx, d, 8)?;
            let dmin = per_block(cx, dmin, 8)?;
            let scale = cx.mul(d, sc)?;
            let min = cx.mul(dmin, m)?;
            let bias = cx.neg(min);
            (f32_of(cx, q), scale, bias)
        }
        Scheme::Q6K => {
            // Element `128h + 32q + i`: the low nibble (q < 2) or high nibble
            // (q ≥ 2) of byte `64h + 32 (q & 1) + i`, its top two bits from
            // byte `128 + 32h + i` at bit `2q`; the weight is `q − 32`.
            let ql = blk.field(cx, 0, 128, &[2, 1, 2, 32])?;
            let lo = and(cx, ql, 15)?;
            let hi = shr(cx, ql, 4)?;
            let low = cx.concat(&[lo, hi], 1)?;
            let low = cx.reshape(low, &[2, 4, 32, b, n])?;
            let qh = blk.field(cx, 128, 192, &[2, 32])?;
            let qh = cx.broadcast(qh, &[2, 4, 32, b, n], &[0, 2, 3, 4])?;
            let top = shr_by(cx, qh, &[(1, 2)])?;
            let top = and(cx, top, 3)?;
            let top = shl(cx, top, 4)?;
            let q = cx.or(low, top)?;
            let q = cx.reshape(q, &[16, 16, b, n])?;
            let sc = blk.field(cx, 192, 208, &[16])?;
            let sc = f32_of(cx, sc);
            let wrap = cx.offset(sc, -256.0)?;
            let limit = cx.like_f(sc, 127.0);
            let big = cx.compare(crate::hlo::Cmp::Gt, sc, limit)?;
            let sc = cx.select(big, wrap, sc)?;
            let d = blk.f16(cx, 208)?;
            let d = per_block(cx, d, 16)?;
            let scale = cx.mul(d, sc)?;
            let bias = cx.scale(scale, -32.0)?;
            (f32_of(cx, q), scale, bias)
        }
    })
}

/// A K-quant plane decoded to f32 `[256, blocks, n]` (element-in-block,
/// block, column).
fn dequant(cx: &mut Cx<'_>, w: Tensor, scheme: Scheme, k: u32) -> Result<Val, Error> {
    let (q, scale, bias) = parts(cx, w, scheme, k)?;
    let t = cx.dims(q)[1];
    let scale = spread(cx, scale, t)?;
    let bias = spread(cx, bias, t)?;
    let v = cx.mul(q, scale)?;
    let v = cx.add(v, bias)?;
    let b = cx.dims(q)[2];
    Ok(cx.reshape(v, &[i64::from(SUPER), b, i64::from(w.rows)])?)
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
/// affine_group(scheme), bits: 8 }`.
pub fn to_affine(
    ctx: &Ctx<'_>,
    w: Tensor,
    k: u32,
    codes: Tensor,
    scales: Tensor,
    biases: Tensor,
) -> Result<Scheme, Error> {
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
        (other, _) => return Err(Error::DtypeUnsupported { op: OP, dtype: other }),
    }
    let n = w.rows;
    let group = affine_group(scheme);
    crate::cx::expect(OP, codes, &[Dtype::U8])?;
    crate::cx::expect(OP, scales, &[Dtype::F32])?;
    crate::cx::expect(OP, biases, &[Dtype::F32])?;
    crate::cx::shaped(OP, "code", codes, n, k)?;
    crate::cx::shaped(OP, "scale", scales, n, k / group)?;
    crate::cx::shaped(OP, "bias", biases, n, k / group)?;
    let w = Tensor::new(w.buf, w.rows, w.width, Dtype::U8);
    ctx.emit(&mut |cx| {
        let (q, scale, bias) = parts(cx, w, scheme, k)?;
        // [S, T, B, n] → [n, B, S, T]; [S, B, n] → [n, B, S].
        let q = cx.transpose(q, &[3, 2, 0, 1])?;
        let q = cx.convert(q, Elem::U8);
        cx.write(codes, q)?;
        let scale = cx.transpose(scale, &[2, 1, 0])?;
        cx.write(scales, scale)?;
        let bias = cx.transpose(bias, &[2, 1, 0])?;
        cx.write(biases, bias)
    })?;
    Ok(scheme)
}
