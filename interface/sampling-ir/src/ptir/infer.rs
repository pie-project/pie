//! Per-op shape/dtype inference for PTIR stage bodies — the validator core
//! and the executable-semantics type table ([`super::interp`] indexes it).
//!
//! Every op is value → value over trace-known shapes (overview §5.1, D2):
//! inference is total per op and errors carry the op index. Channel ops type
//! against the container's channel declarations (`ACT` materializes F32,
//! [`super::container::ChanDType::program_dtype`]).

use alloc::vec::Vec;
use core::fmt;

use super::op::Op;
use crate::types::{DType, Predicate, Shape, ValueId, ValueType, MAX_RANK};

/// An inference failure at `op_index`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyError {
    pub op_index: u32,
    pub kind: BodyErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BodyErrorKind {
    /// Operand id undefined at this point (out of range / forward ref).
    ValueIdOutOfRange(ValueId),
    ShapeMismatch,
    DTypeMismatch,
    /// Channel index outside the container's declaration table.
    ChannelOutOfRange(u32),
    /// Name index outside the container's name table.
    NameOutOfRange(u16),
}

impl fmt::Display for BodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            BodyErrorKind::ValueIdOutOfRange(v) => {
                write!(f, "op {}: operand value id {v} is undefined here", self.op_index)
            }
            BodyErrorKind::ShapeMismatch => {
                write!(f, "op {}: incompatible operand shapes", self.op_index)
            }
            BodyErrorKind::DTypeMismatch => {
                write!(f, "op {}: incompatible operand dtypes", self.op_index)
            }
            BodyErrorKind::ChannelOutOfRange(c) => {
                write!(f, "op {}: channel index {c} out of range", self.op_index)
            }
            BodyErrorKind::NameOutOfRange(n) => {
                write!(f, "op {}: name index {n} out of range", self.op_index)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BodyError {}

/// Context a body types against: the channel element types (in declaration
/// order, `ACT` already materialized) and the name-table size.
pub struct BodyCtx<'a> {
    pub channel_types: &'a [ValueType],
    pub n_names: u16,
}

/// The per-value type table (index = value id) of one stage body, inferring
/// in SSA order. Ops defining 0 ids contribute no entries but are still
/// checked.
pub fn body_types(ops: &[Op], ctx: &BodyCtx<'_>) -> Result<Vec<ValueType>, BodyError> {
    let mut types: Vec<ValueType> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        match infer(i as u32, op, &types, ctx)? {
            Results::None => {}
            Results::One(t) => types.push(t),
            Results::Two(a, b) => {
                types.push(a);
                types.push(b);
            }
        }
    }
    Ok(types)
}

fn err(op_index: u32, kind: BodyErrorKind) -> BodyError {
    BodyError { op_index, kind }
}

/// Elementwise broadcast of two operand shapes (equal, or one scalar).
fn broadcast2(a: Shape, b: Shape) -> Option<Shape> {
    if a == b {
        Some(a)
    } else if a.is_scalar() {
        Some(b)
    } else if b.is_scalar() {
        Some(a)
    } else {
        None
    }
}

/// `Broadcast` rule (v4-exact): `src` left-aligned against `target`
/// (trailing axes padded with 1); each axis equals the target or is 1.
fn can_broadcast_to(src: Shape, target: Shape) -> bool {
    if src.rank() > target.rank() {
        return false;
    }
    let sd = src.dims();
    let td = target.dims();
    for i in 0..target.rank() {
        let s = if i < sd.len() { sd[i] } else { 1 };
        if s != td[i] && s != 1 {
            return false;
        }
    }
    true
}

/// `idx.dims ++ src.dims[1..]` — the axis-0 gather result / scatter vals shape.
fn axis0_result(idx: Shape, src: Shape) -> Option<Shape> {
    let mut dims = [0u32; MAX_RANK * 2];
    let n = idx.rank() + src.rank() - 1;
    if n > MAX_RANK {
        return None;
    }
    dims[..idx.rank()].copy_from_slice(idx.dims());
    dims[idx.rank()..n].copy_from_slice(&src.dims()[1..]);
    Shape::new(&dims[..n])
}

enum Results {
    None,
    One(ValueType),
    Two(ValueType, ValueType),
}

fn infer(
    op_index: u32,
    op: &Op,
    types: &[ValueType],
    ctx: &BodyCtx<'_>,
) -> Result<Results, BodyError> {
    let shape_err = || err(op_index, BodyErrorKind::ShapeMismatch);
    let dtype_err = || err(op_index, BodyErrorKind::DTypeMismatch);
    let g = |id: ValueId| -> Result<ValueType, BodyError> {
        types
            .get(id as usize)
            .copied()
            .ok_or(err(op_index, BodyErrorKind::ValueIdOutOfRange(id)))
    };
    let chan = |c: u32| -> Result<ValueType, BodyError> {
        ctx.channel_types
            .get(c as usize)
            .copied()
            .ok_or(err(op_index, BodyErrorKind::ChannelOutOfRange(c)))
    };
    let name_ok = |n: u16| -> Result<(), BodyError> {
        if n < ctx.n_names { Ok(()) } else { Err(err(op_index, BodyErrorKind::NameOutOfRange(n))) }
    };

    let mut out = Results::None;
    let push = |slot: &mut Results, t: ValueType| {
        *slot = match core::mem::replace(slot, Results::None) {
            Results::None => Results::One(t),
            Results::One(a) => Results::Two(a, t),
            Results::Two(..) => unreachable!("op defines at most 2 results"),
        }
    };

    match *op {
        Op::Const(lit) => push(&mut out, ValueType::scalar(lit.dtype())),

        Op::Exp(a) | Op::Log(a) | Op::Recip(a) => {
            let t = g(a)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            push(&mut out, t);
        }
        Op::Neg(a) | Op::Abs(a) | Op::Sign(a) => {
            let t = g(a)?;
            if !t.dtype.is_numeric() {
                return Err(dtype_err());
            }
            push(&mut out, t);
        }
        Op::Cast { value, dtype } => {
            let t = g(value)?;
            push(&mut out, ValueType::new(t.shape, dtype));
        }

        Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::MaxElem(a, b) | Op::MinElem(a, b)
        | Op::Rem(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if !ta.dtype.is_numeric() || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            push(&mut out, ValueType::new(broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?, ta.dtype));
        }
        Op::Div(a, b) => {
            // Unlike PSIR v4 (F32-only), PTIR `div` is defined on every
            // numeric dtype: F32 division, or truncating integer division
            // (0 on divide-by-zero) — §6.2's `parent = div(i, V)` is id math.
            let (ta, tb) = (g(a)?, g(b)?);
            if !ta.dtype.is_numeric() || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            push(&mut out, ValueType::new(
                broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?,
                ta.dtype,
            ));
        }

        Op::Gt(a, b) | Op::Ge(a, b) | Op::Eq(a, b) | Op::Ne(a, b) | Op::Lt(a, b) | Op::Le(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if !ta.dtype.is_numeric() || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            push(&mut out, ValueType::new(
                broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?,
                DType::Bool,
            ));
        }
        Op::And(a, b) | Op::Or(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if ta.dtype != DType::Bool || tb.dtype != DType::Bool {
                return Err(dtype_err());
            }
            push(&mut out, ValueType::new(
                broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?,
                DType::Bool,
            ));
        }
        Op::Not(a) => {
            let t = g(a)?;
            if t.dtype != DType::Bool {
                return Err(dtype_err());
            }
            push(&mut out, t);
        }

        Op::Select { cond, a, b } => {
            let (tc, ta, tb) = (g(cond)?, g(a)?, g(b)?);
            if tc.dtype != DType::Bool || ta.dtype != tb.dtype {
                return Err(dtype_err());
            }
            let ab = broadcast2(ta.shape, tb.shape).ok_or_else(shape_err)?;
            push(&mut out, ValueType::new(broadcast2(ab, tc.shape).ok_or_else(shape_err)?, ta.dtype));
        }

        Op::ReduceSum(v) | Op::ReduceMax(v) | Op::ReduceMin(v) => {
            let t = g(v)?;
            if !t.dtype.is_numeric() {
                return Err(dtype_err());
            }
            push(&mut out, ValueType::new(t.shape.drop_last().ok_or_else(shape_err)?, t.dtype));
        }
        Op::ReduceArgmax(v) => {
            let t = g(v)?;
            if !t.dtype.is_numeric() {
                return Err(dtype_err());
            }
            push(&mut out, ValueType::new(t.shape.drop_last().ok_or_else(shape_err)?, DType::I32));
        }

        Op::Broadcast { value, shape } => {
            let t = g(value)?;
            if !can_broadcast_to(t.shape, shape) {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(shape, t.dtype));
        }
        Op::Reshape { value, shape } => {
            let t = g(value)?;
            if t.shape.numel() != shape.numel() {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(shape, t.dtype));
        }
        Op::Transpose(v) => {
            let t = g(v)?;
            let (m, n) = match *t.shape.dims() {
                [m, n] => (m, n),
                _ => return Err(shape_err()),
            };
            push(&mut out, ValueType::new(Shape::matrix(n, m), t.dtype));
        }

        Op::CumSum(v) | Op::CumProd(v) => {
            let t = g(v)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            if t.shape.rank() == 0 {
                return Err(shape_err());
            }
            push(&mut out, t);
        }

        Op::SortDesc(v) => {
            let t = g(v)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            let n = match *t.shape.dims() {
                [n] => n,
                _ => return Err(shape_err()),
            };
            push(&mut out, ValueType::vector(n, DType::F32));
            push(&mut out, ValueType::vector(n, DType::U32));
        }
        Op::TopK { input, k } => {
            let t = g(input)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            let oshape = match *t.shape.dims() {
                [n] if k >= 1 && k <= n => Shape::vector(k),
                [m, n] if k >= 1 && k <= n => Shape::matrix(m, k),
                _ => return Err(shape_err()),
            };
            push(&mut out, ValueType::new(oshape, DType::F32));
            push(&mut out, ValueType::new(oshape, DType::U32));
        }
        Op::MatMul(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if ta.dtype != DType::F32 || tb.dtype != DType::F32 {
                return Err(dtype_err());
            }
            let (m, ka) = match *ta.shape.dims() {
                [m, k] => (m, k),
                _ => return Err(shape_err()),
            };
            let (kb, n) = match *tb.shape.dims() {
                [k, n] => (k, n),
                _ => return Err(shape_err()),
            };
            if ka != kb {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(Shape::matrix(m, n), DType::F32));
        }
        Op::PivotThreshold { input, predicate } => {
            let t = g(input)?;
            if t.dtype != DType::F32 {
                return Err(dtype_err());
            }
            if t.shape.rank() != 1 && t.shape.rank() != 2 {
                return Err(shape_err());
            }
            match predicate {
                Predicate::RankLe(k_id) => {
                    let kt = g(k_id)?;
                    if !kt.dtype.is_int() {
                        return Err(dtype_err());
                    }
                    let per_row_ok = t.shape.rank() == 2 && *kt.shape.dims() == [t.shape.rows()];
                    if !kt.shape.is_scalar() && !per_row_ok {
                        return Err(shape_err());
                    }
                }
                Predicate::CummassLe(thr) | Predicate::ProbGe(thr) => {
                    let tt = g(thr)?;
                    if tt.dtype != DType::F32 {
                        return Err(dtype_err());
                    }
                    let per_row_ok = t.shape.rank() == 2 && *tt.shape.dims() == [t.shape.rows()];
                    if !tt.shape.is_scalar() && !per_row_ok {
                        return Err(shape_err());
                    }
                }
            }
            push(&mut out, ValueType::new(t.shape, DType::Bool));
        }

        Op::Gather { src, idx } => {
            let (ts, ti) = (g(src)?, g(idx)?);
            if ts.shape.rank() == 0 {
                return Err(shape_err());
            }
            if !ti.dtype.is_int() {
                return Err(dtype_err());
            }
            let oshape = axis0_result(ti.shape, ts.shape).ok_or_else(shape_err)?;
            push(&mut out, ValueType::new(oshape, ts.dtype));
        }
        Op::GatherRow { src, idx } => {
            let (ts, ti) = (g(src)?, g(idx)?);
            let m = match *ts.shape.dims() {
                [m, _n] => m,
                _ => return Err(shape_err()),
            };
            if !ti.dtype.is_int() {
                return Err(dtype_err());
            }
            match *ti.shape.dims() {
                [k] if k == m => {}
                _ => return Err(shape_err()),
            }
            push(&mut out, ValueType::vector(m, ts.dtype));
        }
        Op::ScatterAdd { base, idx, vals } | Op::ScatterSet { base, idx, vals } => {
            let (tb, ti, tv) = (g(base)?, g(idx)?, g(vals)?);
            if tb.shape.rank() == 0 {
                return Err(shape_err());
            }
            if matches!(op, Op::ScatterAdd { .. }) && !tb.dtype.is_numeric() {
                return Err(dtype_err());
            }
            if !ti.dtype.is_int() {
                return Err(dtype_err());
            }
            if tv.dtype != tb.dtype {
                return Err(dtype_err());
            }
            let expect = axis0_result(ti.shape, tb.shape).ok_or_else(shape_err)?;
            // vals: exact `idx.dims ++ base.dims[1..]`, or a scalar broadcast.
            if tv.shape != expect && !tv.shape.is_scalar() {
                return Err(shape_err());
            }
            push(&mut out, tb);
        }
        Op::Iota { len } => {
            if len == 0 {
                return Err(shape_err());
            }
            push(&mut out, ValueType::vector(len, DType::U32));
        }
        Op::MaskApply { logits, mask } => {
            let (tl, tm) = (g(logits)?, g(mask)?);
            let n = tl.shape.last_len().ok_or_else(shape_err)?;
            if tl.dtype != DType::F32 || tm.dtype != DType::U32 {
                return Err(dtype_err());
            }
            match *tm.shape.dims() {
                [w] if w == n.div_ceil(32) => {}
                _ => return Err(shape_err()),
            }
            push(&mut out, tl);
        }

        Op::Rng { shape, .. } => {
            if shape.rank() == 0 {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(shape, DType::F32));
        }
        Op::RngKeyed { state, shape, .. } => {
            let ts = g(state)?;
            if ts.dtype != DType::U32 {
                return Err(dtype_err());
            }
            if *ts.shape.dims() != [2] || shape.rank() == 0 {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(shape, DType::F32));
        }

        Op::ChanTake(c) | Op::ChanRead(c) => push(&mut out, chan(c)?),
        Op::ChanPut { chan: c, value } => {
            let ct = chan(c)?;
            let vt = g(value)?;
            if vt.dtype != ct.dtype {
                return Err(dtype_err());
            }
            if vt.shape != ct.shape {
                return Err(shape_err());
            }
        }

        Op::IntrinsicVal { shape, dtype, .. } => push(&mut out, ValueType::new(shape, dtype)),
        Op::KernelCall { name, ref args, shape, dtype } => {
            name_ok(name)?;
            for &a in args {
                g(a)?;
            }
            push(&mut out, ValueType::new(shape, dtype));
        }
        Op::SinkCall { name, ref args } => {
            name_ok(name)?;
            for &a in args {
                g(a)?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Literal;
    use alloc::vec;

    fn ctx_with(channels: &[ValueType]) -> BodyCtx<'_> {
        BodyCtx { channel_types: channels, n_names: 1 }
    }

    #[test]
    fn chan_take_types_from_decl() {
        let chans = [ValueType::vector(4, DType::I32)];
        let ops = vec![Op::ChanTake(0), Op::Neg(0), Op::ChanPut { chan: 0, value: 1 }];
        let t = body_types(&ops, &ctx_with(&chans)).unwrap();
        assert_eq!(t.len(), 2); // ChanPut defines no id
        assert_eq!(t[0], ValueType::vector(4, DType::I32));
    }

    #[test]
    fn chan_put_type_mismatch_rejected() {
        let chans = [ValueType::vector(4, DType::I32)];
        let ops = vec![Op::Const(Literal::F32(1.0)), Op::ChanPut { chan: 0, value: 0 }];
        assert_eq!(
            body_types(&ops, &ctx_with(&chans)).unwrap_err().kind,
            BodyErrorKind::DTypeMismatch
        );
    }

    #[test]
    fn row_gather_and_scatter_axis0() {
        // pages [3,5] gathered by parent [3] → [3,5]; then scatter one row back.
        let chans = [ValueType::new(Shape::matrix(3, 5), DType::U32), ValueType::vector(3, DType::U32)];
        let ops = vec![
            Op::ChanTake(0),                                 // 0: [3,5] u32
            Op::ChanRead(1),                                 // 1: [3] u32
            Op::Gather { src: 0, idx: 1 },                   // 2: [3,5] u32
            Op::Const(Literal::U32(0)),                      // 3: scalar
            Op::Gather { src: 2, idx: 3 },                   // 4: [5] u32 (scalar idx → row)
            Op::ScatterSet { base: 2, idx: 1, vals: 2 },     // wrong vals shape? [3]++[5]=[3,5] ok
        ];
        let t = body_types(&ops, &ctx_with(&chans)).unwrap();
        assert_eq!(t[2], ValueType::new(Shape::matrix(3, 5), DType::U32));
        assert_eq!(t[4], ValueType::vector(5, DType::U32));
        assert_eq!(t[5], ValueType::new(Shape::matrix(3, 5), DType::U32));
    }

    #[test]
    fn topk_matmul_transpose_shapes() {
        let chans = [ValueType::new(Shape::matrix(2, 8), DType::F32)];
        let ops = vec![
            Op::ChanRead(0),                    // 0: [2,8]
            Op::TopK { input: 0, k: 3 },        // 1: [2,3] f32, 2: [2,3] u32
            Op::Transpose(0),                   // 3: [8,2]
            Op::MatMul(0, 3),                   // 4: [2,2]
        ];
        let t = body_types(&ops, &ctx_with(&chans)).unwrap();
        assert_eq!(t[1], ValueType::new(Shape::matrix(2, 3), DType::F32));
        assert_eq!(t[2], ValueType::new(Shape::matrix(2, 3), DType::U32));
        assert_eq!(t[3], ValueType::new(Shape::matrix(8, 2), DType::F32));
        assert_eq!(t[4], ValueType::new(Shape::matrix(2, 2), DType::F32));
    }

    #[test]
    fn forward_ref_rejected() {
        let ops = vec![Op::Exp(3)];
        assert_eq!(
            body_types(&ops, &ctx_with(&[])).unwrap_err().kind,
            BodyErrorKind::ValueIdOutOfRange(3)
        );
    }

    #[test]
    fn rng_keyed_needs_u32_pair_state() {
        let chans = [ValueType::vector(2, DType::U32), ValueType::vector(3, DType::U32)];
        let ok = vec![
            Op::ChanTake(0),
            Op::RngKeyed { state: 0, shape: Shape::vector(8), kind: crate::types::RngKind::Gumbel },
        ];
        assert!(body_types(&ok, &ctx_with(&chans)).is_ok());
        let bad = vec![
            Op::ChanTake(1), // [3] not [2]
            Op::RngKeyed { state: 0, shape: Shape::vector(8), kind: crate::types::RngKind::Gumbel },
        ];
        assert_eq!(
            body_types(&bad, &ctx_with(&chans)).unwrap_err().kind,
            BodyErrorKind::ShapeMismatch
        );
    }
}
