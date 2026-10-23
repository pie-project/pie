//! Dense projections, `y = act · wᵀ` over a row-major `[n, k]` weight: one
//! `dot_general` per call, bf16 operands and an f32 result rounded once where
//! it lands. `rel_bias` folds its epilogue into the same f32 result before
//! the one rounding.
//!
//! `project` is the contraction every linear family here shares: the
//! quantized families decode their planes to a dense `[n, k]` value and hand
//! it to the same dot, so XLA can fuse the decode into the matmul.

#![allow(clippy::too_many_arguments)]

use dtype::Dtype;

use crate::check::expect;
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::tensor::{Landing, Operand, Stored};

/// The activation dtypes a projection reads.
pub(crate) const ACTS: &[Dtype] = &[Dtype::Bf16, Dtype::F32, Dtype::F16];

/// `x` rounded to the nearest bf16 (ties to even), kept in f32. Done on the
/// bits, so no convert pair is there for XLA to fold away under excess
/// precision: the split in [`bf16_terms`] depends on the rounding happening.
pub(crate) fn round_bf16_bits(f: &mut Func, x: Val) -> Result<Val, Error> {
    let bits = f.bitcast(x, Elem::U32)?;
    let sixteen = f.like_i(bits, 16);
    let lsb = f.shr(bits, sixteen)?;
    let one = f.like_i(bits, 1);
    let lsb = f.and(lsb, one)?;
    let half = f.like_i(bits, 0x7FFF);
    let r = f.add(bits, half)?;
    let r = f.add(r, lsb)?;
    let mask = f.like_i(bits, 0xFFFF_0000);
    let r = f.and(r, mask)?;
    Ok(f.bitcast(r, Elem::F32)?)
}

/// An f32 value as `n` bf16 terms whose f32 sum is the value to `8n`
/// significant bits: a dot against an exactly-bf16 operand then runs `n`
/// single MXU passes instead of HIGHEST's six.
pub(crate) fn bf16_terms(f: &mut Func, x: Val, n: usize) -> Result<Vec<Val>, Error> {
    let x = f.convert(x, Elem::F32);
    let mut rest = x;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 == n {
            out.push(f.convert(rest, Elem::Bf16));
            break;
        }
        let head = round_bf16_bits(f, rest)?;
        out.push(f.convert(head, Elem::Bf16));
        rest = f.sub(rest, head)?;
    }
    Ok(out)
}

/// `dot_general` of an f32 `lhs` against an exactly-bf16 `rhs`, as `terms`
/// bf16 passes summed in f32.
pub(crate) fn dot_split(
    f: &mut Func,
    lhs: Val,
    rhs: Val,
    terms: usize,
    lhs_batch: &[i64],
    rhs_batch: &[i64],
    lhs_contract: &[i64],
    rhs_contract: &[i64],
) -> Result<Val, Error> {
    let rhs = f.convert(rhs, Elem::Bf16);
    let mut acc: Option<Val> = None;
    for t in bf16_terms(f, lhs, terms)? {
        let d = f.dot_general(
            t,
            rhs,
            lhs_batch,
            rhs_batch,
            lhs_contract,
            rhs_contract,
            Elem::F32,
        )?;
        acc = Some(match acc {
            Some(a) => f.add(a, d)?,
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
pub(crate) fn project(f: &mut Func, x: Val, w: Val, w_exact_bf16: bool) -> Result<Val, Error> {
    let xe = f.elem(x);
    let we = f.elem(w);
    match xe {
        Elem::Bf16 => {
            let w = f.convert(w, Elem::Bf16);
            Ok(f.matmul_nt(x, w, Elem::F32)?)
        }
        Elem::F32 | Elem::F16 if we == Elem::Bf16 || w_exact_bf16 => {
            dot_split(f, x, w, 3, &[], &[], &[1], &[1])
        }
        _ => {
            let x = f.convert(x, Elem::F32);
            let w = f.convert(w, Elem::F32);
            Ok(f.matmul_nt(x, w, Elem::F32)?)
        }
    }
}

/// The first `rows` rows of a `[r, w]` value (the whole value when `r == rows`).
pub(crate) fn head_rows(f: &mut Func, v: Val, rows: u32) -> Result<Val, Error> {
    let r = f.dims(v)[0];
    if r == i64::from(rows) {
        return Ok(v);
    }
    Ok(f.slice_axis(v, 0, 0, i64::from(rows))?)
}

/// The `(m, n, k)` a projection of `act` into `y` walks; `act` may carry more
/// rows than `y` lands (the extra rows are not read).
pub(crate) fn extent(
    op: &'static str,
    act: impl Stored,
    y: impl Stored,
) -> Result<(u32, u32, u32), Error> {
    expect(op, act, ACTS)?;
    if y.width() == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width() == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    if act.rows() < y.rows() {
        return Err(refuse(
            op,
            format!(
                "the activation has {} rows and the result lands {}",
                act.rows(),
                y.rows()
            ),
        ));
    }
    Ok((y.rows(), y.width(), act.width()))
}

fn dense(
    op: &'static str,
    act: impl Stored,
    w: impl Stored,
    y: impl Stored,
) -> Result<(u32, u32, u32), Error> {
    let (m, n, k) = extent(op, act, y)?;
    expect(op, w, &[Dtype::Bf16, Dtype::F16, Dtype::F32])?;
    if w.width() != k || w.rows() != n {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and the projection is {k} in, {n} out",
                w.rows(),
                w.width()
            ),
        ));
    }
    Ok((m, n, k))
}

/// `act · wᵀ` in f32, `[y.rows, w.rows]`.
fn dense_f32(f: &mut Func, act: Operand, w: Operand, rows: u32) -> Result<Val, Error> {
    let x = act.val;
    let x = head_rows(f, x, rows)?;
    let wv = w.val;
    project(f, x, wv, false)
}

pub fn matmul(f: &mut Func, act: Operand, w: Operand, y: Landing) -> Result<Option<Val>, Error> {
    act_x_wt(f, "linear.matmul", act, w, y)
}

pub fn lm_head(f: &mut Func, act: Operand, w: Operand, y: Landing) -> Result<Option<Val>, Error> {
    act_x_wt(f, "linear.matmul", act, w, y)
}

/// `y = act · wᵀ` for a dense weight. Also the f32-activation lane gemm of
/// kernels-cuda / kernels-metal `linear::lane_gemm::act_x_wt` (same
/// signature): an f32 activation is contracted at f32 precision. Nothing
/// when `y` has no rows.
pub fn act_x_wt(
    f: &mut Func,
    op: &'static str,
    act: Operand,
    w: Operand,
    y: Landing,
) -> Result<Option<Val>, Error> {
    let (m, _, _) = dense(op, act, w, y)?;
    if m == 0 {
        return Ok(None);
    }
    Ok(Some(dense_f32(f, act, w, m)?))
}

/// The relative-position bias of each head: `y[r, h, d] = Σ_j x[r, h, j] ·
/// w[j, d]`, `x` `[rows, heads · d_rel]`, `w` `[d_rel, extent]`, `y` f32
/// `[rows, heads · extent]`.
/// Reference: kernels-cuda `linear::rel_bias::rel_bias` (engine-cuda
/// `Linear::RelBias`).
pub fn rel_bias(
    f: &mut Func,
    x: Operand,
    w: Operand,
    heads: u32,
    d_rel: u32,
    extent: u32,
    y: Landing,
) -> Result<Val, Error> {
    const OP: &str = "linear.rel_bias";
    expect(OP, x, &[Dtype::Bf16, Dtype::F16])?;
    if heads == 0 || d_rel == 0 || extent == 0 {
        return Err(refuse(
            OP,
            "the heads, relative width and extent are all nonzero",
        ));
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
    let xv = x.val;
    let xv = f.reshape(
        xv,
        &[i64::from(x.rows) * i64::from(heads), i64::from(d_rel)],
    )?;
    let wv = w.val;
    let (xv, wv) = if x.dtype == Dtype::Bf16 {
        (xv, wv)
    } else {
        (f.convert(xv, Elem::F32), f.convert(wv, Elem::F32))
    };
    let out = f.dot_general(xv, wv, &[], &[], &[1], &[0], Elem::F32)?;
    Ok(out)
}
