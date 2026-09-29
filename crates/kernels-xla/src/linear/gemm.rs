//! Dense projections, `y = act · wᵀ` over a row-major `[n, k]` weight: one
//! `dot_general` per call, bf16 operands and an f32 result rounded once where
//! it lands. The CUDA-only fused forms (`matmul_bias`, `matmul_geglu`,
//! `lm_head_softcap`, `rel_bias`) fold their epilogue into the same f32
//! result before the one rounding.
//!
//! [`project`] is the contraction every linear family here shares: the
//! quantized families decode their planes to a dense `[n, k]` value and hand
//! it to the same dot, so XLA can fuse the decode into the matmul.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::cx::{Ctx, Cx, expect};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Val};
use crate::tensor::Tensor;

/// The activation dtypes a projection reads.
pub(crate) const ACTS: &[Dtype] = &[Dtype::Bf16, Dtype::F32, Dtype::F16];

/// `x` rounded to the nearest bf16 (ties to even), kept in f32. Done on the
/// bits, so no convert pair is there for XLA to fold away under excess
/// precision: the split in [`bf16_terms`] depends on the rounding happening.
pub(crate) fn round_bf16_bits(cx: &mut Cx<'_>, x: Val) -> Result<Val, Error> {
    let bits = cx.bitcast(x, Elem::U32)?;
    let sixteen = cx.like_i(bits, 16);
    let lsb = cx.shr(bits, sixteen)?;
    let one = cx.like_i(bits, 1);
    let lsb = cx.and(lsb, one)?;
    let half = cx.like_i(bits, 0x7FFF);
    let r = cx.add(bits, half)?;
    let r = cx.add(r, lsb)?;
    let mask = cx.like_i(bits, 0xFFFF_0000);
    let r = cx.and(r, mask)?;
    Ok(cx.bitcast(r, Elem::F32)?)
}

/// An f32 value as `n` bf16 terms whose f32 sum is the value to `8n`
/// significant bits: a dot against an exactly-bf16 operand then runs `n`
/// single MXU passes instead of HIGHEST's six.
pub(crate) fn bf16_terms(cx: &mut Cx<'_>, x: Val, n: usize) -> Result<Vec<Val>, Error> {
    let x = cx.convert(x, Elem::F32);
    let mut rest = x;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 == n {
            out.push(cx.convert(rest, Elem::Bf16));
            break;
        }
        let head = round_bf16_bits(cx, rest)?;
        out.push(cx.convert(head, Elem::Bf16));
        rest = cx.sub(rest, head)?;
    }
    Ok(out)
}

/// `dot_general` of an f32 `lhs` against an exactly-bf16 `rhs`, as `terms`
/// bf16 passes summed in f32.
pub(crate) fn dot_split(
    cx: &mut Cx<'_>,
    lhs: Val,
    rhs: Val,
    terms: usize,
    lhs_batch: &[i64],
    rhs_batch: &[i64],
    lhs_contract: &[i64],
    rhs_contract: &[i64],
) -> Result<Val, Error> {
    let rhs = cx.convert(rhs, Elem::Bf16);
    let mut acc: Option<Val> = None;
    for t in bf16_terms(cx, lhs, terms)? {
        let d = cx.dot_general(t, rhs, lhs_batch, rhs_batch, lhs_contract, rhs_contract, Elem::F32)?;
        acc = Some(match acc {
            Some(a) => cx.add(a, d)?,
            None => d,
        });
    }
    acc.ok_or_else(|| refuse("linear.dot_split", "no terms"))
}

/// `x · wᵀ` for `x: [m, k]`, `w: [n, k]`, as f32 `[m, n]`.
///
/// bf16 against bf16 (or a decoded weight, any float) is one bf16 MXU pass:
/// the weight is rounded to bf16 first, as a GPU qmm stages its decoded tile.
/// An f32 activation against a bf16 weight is contracted exactly (three bf16
/// passes, the f32 lane-gemm's answer); anything else wider than bf16 on both
/// sides runs at HIGHEST.
pub(crate) fn project(cx: &mut Cx<'_>, x: Val, w: Val, w_exact_bf16: bool) -> Result<Val, Error> {
    let xe = cx.elem(x);
    let we = cx.elem(w);
    match xe {
        Elem::Bf16 => {
            let w = cx.convert(w, Elem::Bf16);
            Ok(cx.matmul_nt(x, w, Elem::F32)?)
        }
        Elem::F32 | Elem::F16 if we == Elem::Bf16 || w_exact_bf16 => {
            dot_split(cx, x, w, 3, &[], &[], &[1], &[1])
        }
        _ => {
            let x = cx.convert(x, Elem::F32);
            let w = cx.convert(w, Elem::F32);
            Ok(cx.matmul_nt(x, w, Elem::F32)?)
        }
    }
}

/// The first `rows` rows of a `[r, w]` value (the whole value when `r == rows`).
pub(crate) fn head_rows(cx: &mut Cx<'_>, v: Val, rows: u32) -> Result<Val, Error> {
    let r = cx.dims(v)[0];
    if r == i64::from(rows) {
        return Ok(v);
    }
    Ok(cx.slice_axis(v, 0, 0, i64::from(rows))?)
}

/// The `(m, n, k)` a projection of `act` into `y` walks; `act` may carry more
/// rows than `y` lands (the extra rows are not read).
pub(crate) fn extent(op: &'static str, act: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    expect(op, act, ACTS)?;
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    if act.rows < y.rows {
        return Err(refuse(
            op,
            format!(
                "the activation has {} rows and the result lands {}",
                act.rows, y.rows
            ),
        ));
    }
    Ok((y.rows, y.width, act.width))
}

fn dense(op: &'static str, act: Tensor, w: Tensor, y: Tensor) -> Result<(u32, u32, u32), Error> {
    let (m, n, k) = extent(op, act, y)?;
    expect(op, w, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
    if w.width != k || w.rows != n {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and the projection is {k} in, {n} out",
                w.rows, w.width
            ),
        ));
    }
    Ok((m, n, k))
}

/// `act · wᵀ` in f32, `[y.rows, w.rows]`.
fn dense_f32(cx: &mut Cx<'_>, act: Tensor, w: Tensor, rows: u32) -> Result<Val, Error> {
    let x = cx.read(act)?;
    let x = head_rows(cx, x, rows)?;
    let wv = cx.read(w)?;
    project(cx, x, wv, false)
}

pub fn matmul(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.matmul", act, w, y)
}

pub fn lm_head(ctx: &Ctx<'_>, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    act_x_wt(ctx, "linear.lm_head", act, w, y)
}

/// `y = act · wᵀ` for a dense weight. Also the f32-activation lane gemm of
/// kernels-cuda / kernels-metal `linear::lane_gemm::act_x_wt` (same
/// signature): an f32 activation is contracted at f32 precision.
pub fn act_x_wt(ctx: &Ctx<'_>, op: &'static str, act: Tensor, w: Tensor, y: Tensor) -> Result<(), Error> {
    let (m, _, _) = dense(op, act, w, y)?;
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let out = dense_f32(cx, act, w, m)?;
        cx.write(y, out)
    })
}

/// `y = act · wᵀ + bias`, the bias one value per column, added to the f32
/// result before its one rounding.
/// Reference: kernels-cuda `linear::gemm::matmul_bias` (engine-cuda
/// `Linear::MatmulBias`; its non-gemv tactics add the bias after the store).
pub fn matmul_bias(ctx: &Ctx<'_>, act: Tensor, w: Tensor, bias: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.matmul_bias";
    let (m, n, _) = dense(OP, act, w, y)?;
    if bias.elements() != u64::from(n) {
        return Err(refuse(OP, "the bias is not one value per column"));
    }
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let out = dense_f32(cx, act, w, m)?;
        let b = cx.read_f32(bias)?;
        let b = cx.reshape(b, &[i64::from(n)])?;
        let b = cx.broadcast(b, &[i64::from(m), i64::from(n)], &[1])?;
        let out = cx.add(out, b)?;
        cx.write(y, out)
    })
}

/// `y = cap · tanh((act · wᵀ) / cap)`, the cap applied to the f32 logits.
/// Reference: kernels-cuda `linear::skinny` `Epilogue::Softcap` (fused on
/// the accumulator) / `attn::logit_softcap` after `lm_head` (engine-cuda
/// `Linear::LmHeadSoftcap`).
pub fn lm_head_softcap(ctx: &Ctx<'_>, act: Tensor, w: Tensor, cap: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.lm_head_softcap";
    if !(cap.is_finite() && cap > 0.0) {
        return Err(refuse(OP, format!("{cap} is not a logit soft cap")));
    }
    let (m, _, _) = dense(OP, act, w, y)?;
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let out = dense_f32(cx, act, w, m)?;
        let out = softcap(cx, out, cap)?;
        cx.write(y, out)
    })
}

/// `cap · tanh(x / cap)`, as `attn::logit_softcap` computes it.
pub(crate) fn softcap(cx: &mut Cx<'_>, x: Val, cap: f32) -> Result<Val, Error> {
    let inv = 1.0f32 / cap;
    let s = cx.scale(x, f64::from(inv))?;
    let t = cx.tanh(s);
    Ok(cx.scale(t, f64::from(cap))?)
}

/// `packed = act · wᵀ` (the `[gate | up]` halves, `2 · intermediate` wide),
/// then `y = gelu_tanh(gate) · up` read back from the bf16 `packed`.
/// Reference: engine-cuda `Linear::MatmulGeglu` (a matmul into `packed`,
/// then `mlp::geglu_tanh_packed`).
pub fn matmul_geglu(
    ctx: &Ctx<'_>,
    act: Tensor,
    w: Tensor,
    intermediate: u32,
    packed: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul_geglu";
    let (m, n, _) = dense(OP, act, w, packed)?;
    if n != intermediate.saturating_mul(2) || y.width != intermediate || y.rows != m {
        return Err(refuse(
            OP,
            format!(
                "the weight lands {n} columns and the activation {}x{}, where the \
                 intermediate is {intermediate}",
                y.rows, y.width
            ),
        ));
    }
    if m == 0 {
        return Ok(());
    }
    ctx.emit(&mut |cx| {
        let out = dense_f32(cx, act, w, m)?;
        cx.write(packed, out)?;
        let p = cx.read_f32(packed)?;
        let i = i64::from(intermediate);
        let g = cx.slice_axis(p, 1, 0, i)?;
        let u = cx.slice_axis(p, 1, i, 2 * i)?;
        let g = cx.gelu_tanh(g)?;
        let out = cx.mul(g, u)?;
        cx.write(y, out)
    })
}

/// The relative-position bias of each head: `y[r, h, d] = Σ_j x[r, h, j] ·
/// w[j, d]`, `x` `[rows, heads · d_rel]`, `w` `[d_rel, extent]`, `y` f32
/// `[rows, heads · extent]`.
/// Reference: kernels-cuda `linear::rel_bias::rel_bias` (engine-cuda
/// `Linear::RelBias`).
pub fn rel_bias(
    ctx: &Ctx<'_>,
    x: Tensor,
    w: Tensor,
    heads: u32,
    d_rel: u32,
    extent: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.rel_bias";
    expect(OP, x, &[Dtype::Bf16, Dtype::F16])?;
    if heads == 0 || d_rel == 0 || extent == 0 {
        return Err(refuse(OP, "the heads, relative width and extent are all nonzero"));
    }
    if x.width != heads * d_rel {
        return Err(refuse(
            OP,
            format!(
                "the relative features are {} wide and the statement names {heads} x {d_rel}",
                x.width
            ),
        ));
    }
    if w.dtype != x.dtype || w.rows != d_rel || w.width != extent {
        return Err(refuse(
            OP,
            format!(
                "the profile bank is [{}, {}] {:?} and the statement names [{d_rel}, {extent}] {:?}",
                w.rows, w.width, w.dtype, x.dtype
            ),
        ));
    }
    if y.rows != x.rows || y.width != heads * extent {
        return Err(refuse(
            OP,
            format!(
                "the bias lands [{}, {}] and the statement names [{}, {heads} x {extent}]",
                y.rows, y.width, x.rows
            ),
        ));
    }
    ctx.emit(&mut |cx| {
        let xv = cx.read(x)?;
        let xv = cx.reshape(xv, &[i64::from(x.rows) * i64::from(heads), i64::from(d_rel)])?;
        let wv = cx.read(w)?;
        let (xv, wv) = if x.dtype == Dtype::Bf16 {
            (xv, wv)
        } else {
            (cx.convert(xv, Elem::F32), cx.convert(wv, Elem::F32))
        };
        let out = cx.dot_general(xv, wv, &[], &[], &[1], &[0], Elem::F32)?;
        cx.write(y, out)
    })
}
