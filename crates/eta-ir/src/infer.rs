use alloc::vec::Vec;
use core::fmt;

use super::op::Op;
use crate::types::{
    Dtype, MAX_RANK, Predicate, Shape, ValueId, ValueType, float_result, is_float, is_int,
    is_numeric, supports,
};

/// What a reduction over a row of `dtype` returns: its accumulator's dtype,
/// f32 for a `bf16` row (a `bf16` row accumulates in f32 and the result is
/// the accumulator, never rounded; see `types`), the row's own otherwise.
#[must_use]
pub fn reduced_dtype(dtype: Dtype) -> Dtype {
    if dtype == Dtype::Bf16 {
        Dtype::F32
    } else {
        dtype
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyError {
    pub op_index: u32,
    pub kind: BodyErrorKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BodyErrorKind {
    ValueIdOutOfRange(ValueId),
    ShapeMismatch,
    ShapeMismatchBin(Shape, Shape),
    DTypeMismatch,
    UnsupportedDtype(Dtype),
    ChannelOutOfRange(u32),
    NameOutOfRange(u16),
}

impl fmt::Display for BodyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            BodyErrorKind::ValueIdOutOfRange(v) => {
                write!(
                    f,
                    "op {}: operand value id {v} is undefined here",
                    self.op_index
                )
            }
            BodyErrorKind::ShapeMismatch => {
                write!(f, "op {}: incompatible operand shapes", self.op_index)
            }
            BodyErrorKind::UnsupportedDtype(d) => {
                write!(
                    f,
                    "op {}: {d:?} is not a dtype ETA computes in",
                    self.op_index
                )
            }
            BodyErrorKind::ShapeMismatchBin(a, b) => {
                write!(
                    f,
                    "op {}: incompatible operand shapes {a:?} vs {b:?}",
                    self.op_index
                )
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

pub struct BodyCtx<'a> {
    pub channel_types: &'a [ValueType],
    pub n_names: u32,
}

pub fn body_types(ops: &[Op], ctx: &BodyCtx<'_>) -> Result<Vec<ValueType>, BodyError> {
    let mut types: Vec<ValueType> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let index = u32::try_from(i).unwrap_or(u32::MAX);
        let produced = types.len();
        match infer(index, op, &types, ctx)? {
            Results::None => {}
            Results::One(t) => types.push(t),
            Results::Two(a, b) => {
                types.push(a);
                types.push(b);
            }
        }
        for t in &types[produced..] {
            if !supports(t.dtype) {
                return Err(err(index, BodyErrorKind::UnsupportedDtype(t.dtype)));
            }
        }
    }
    Ok(types)
}

fn err(op_index: u32, kind: BodyErrorKind) -> BodyError {
    BodyError { op_index, kind }
}

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

fn append_dim(shape: Shape, dim: u32) -> Option<Shape> {
    let mut dims = [0u32; MAX_RANK];
    if shape.rank() >= MAX_RANK || dim == 0 {
        return None;
    }
    dims[..shape.rank()].copy_from_slice(shape.dims());
    dims[shape.rank()] = dim;
    Shape::new(&dims[..shape.rank() + 1])
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
    let shape_err2 = |a: crate::types::Shape, b: crate::types::Shape| {
        err(op_index, BodyErrorKind::ShapeMismatchBin(a, b))
    };
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
        if u32::from(n) < ctx.n_names {
            Ok(())
        } else {
            Err(err(op_index, BodyErrorKind::NameOutOfRange(n)))
        }
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

        Op::Exp(a)
        | Op::Log(a)
        | Op::Recip(a)
        | Op::Sin(a)
        | Op::Cos(a)
        | Op::Sqrt(a)
        | Op::Rsqrt(a) => {
            let t = g(a)?;
            if !is_float(t.dtype) {
                return Err(dtype_err());
            }
            push(&mut out, t);
        }
        Op::Neg(a) | Op::Abs(a) | Op::Sign(a) => {
            let t = g(a)?;
            if !is_numeric(t.dtype) {
                return Err(dtype_err());
            }
            push(&mut out, t);
        }
        Op::Cast { value, dtype } => {
            let t = g(value)?;
            push(&mut out, ValueType::new(t.shape, dtype));
        }

        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::MaxElem(a, b)
        | Op::MinElem(a, b)
        | Op::Rem(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            let dtype = float_result(ta.dtype, tb.dtype).ok_or_else(dtype_err)?;
            if !is_numeric(dtype) {
                return Err(dtype_err());
            }
            push(
                &mut out,
                ValueType::new(
                    broadcast2(ta.shape, tb.shape).ok_or_else(|| shape_err2(ta.shape, tb.shape))?,
                    dtype,
                ),
            );
        }
        Op::Div(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            let dtype = float_result(ta.dtype, tb.dtype).ok_or_else(dtype_err)?;
            if !is_numeric(dtype) {
                return Err(dtype_err());
            }
            push(
                &mut out,
                ValueType::new(
                    broadcast2(ta.shape, tb.shape).ok_or_else(|| shape_err2(ta.shape, tb.shape))?,
                    dtype,
                ),
            );
        }

        Op::Gt(a, b) | Op::Ge(a, b) | Op::Eq(a, b) | Op::Ne(a, b) | Op::Lt(a, b) | Op::Le(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            let common = float_result(ta.dtype, tb.dtype).ok_or_else(dtype_err)?;
            if !is_numeric(common) {
                return Err(dtype_err());
            }
            push(
                &mut out,
                ValueType::new(
                    broadcast2(ta.shape, tb.shape).ok_or_else(|| shape_err2(ta.shape, tb.shape))?,
                    Dtype::Bool,
                ),
            );
        }
        Op::And(a, b) | Op::Or(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if ta.dtype != Dtype::Bool || tb.dtype != Dtype::Bool {
                return Err(dtype_err());
            }
            push(
                &mut out,
                ValueType::new(
                    broadcast2(ta.shape, tb.shape).ok_or_else(|| shape_err2(ta.shape, tb.shape))?,
                    Dtype::Bool,
                ),
            );
        }
        Op::Not(a) => {
            let t = g(a)?;
            if t.dtype != Dtype::Bool {
                return Err(dtype_err());
            }
            push(&mut out, t);
        }

        Op::Select { cond, a, b } => {
            let (tc, ta, tb) = (g(cond)?, g(a)?, g(b)?);
            if tc.dtype != Dtype::Bool {
                return Err(dtype_err());
            }
            let dtype = float_result(ta.dtype, tb.dtype).ok_or_else(dtype_err)?;
            let ab =
                broadcast2(ta.shape, tb.shape).ok_or_else(|| shape_err2(ta.shape, tb.shape))?;
            push(
                &mut out,
                ValueType::new(broadcast2(ab, tc.shape).ok_or_else(shape_err)?, dtype),
            );
        }

        Op::ReduceSum(v) | Op::ReduceMax(v) | Op::ReduceMin(v) => {
            let t = g(v)?;
            if !is_numeric(t.dtype) {
                return Err(dtype_err());
            }
            push(
                &mut out,
                ValueType::new(
                    t.shape.drop_last().ok_or_else(shape_err)?,
                    reduced_dtype(t.dtype),
                ),
            );
        }
        Op::ReduceArgmax(v) => {
            let t = g(v)?;
            if !is_numeric(t.dtype) {
                return Err(dtype_err());
            }
            push(
                &mut out,
                ValueType::new(t.shape.drop_last().ok_or_else(shape_err)?, Dtype::I32),
            );
        }

        Op::GumbelArgmax {
            logits,
            keep,
            scale,
            state,
        } => {
            let (tl, tk, td, ts) = (g(logits)?, g(keep)?, g(scale)?, g(state)?);
            if !is_float(tl.dtype)
                || tk.dtype != Dtype::Bool
                || !is_float(td.dtype)
                || ts.dtype != Dtype::U32
            {
                return Err(dtype_err());
            }
            if (!tk.shape.is_scalar() && tk.shape != tl.shape)
                || !td.shape.is_scalar()
                || *ts.shape.dims() != [2]
            {
                return Err(shape_err());
            }
            push(
                &mut out,
                ValueType::new(tl.shape.drop_last().ok_or_else(shape_err)?, Dtype::I32),
            );
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
            if !is_numeric(t.dtype) {
                return Err(dtype_err());
            }
            if t.shape.rank() == 0 {
                return Err(shape_err());
            }
            push(&mut out, t);
        }

        Op::SortDesc(v) => {
            let t = g(v)?;
            if !is_float(t.dtype) {
                return Err(dtype_err());
            }
            let n = match *t.shape.dims() {
                [n] => n,
                _ => return Err(shape_err()),
            };
            push(&mut out, ValueType::vector(n, t.dtype));
            push(&mut out, ValueType::vector(n, Dtype::U32));
        }
        Op::TopK { input, k } => {
            let t = g(input)?;
            if !is_float(t.dtype) {
                return Err(dtype_err());
            }
            let oshape = match *t.shape.dims() {
                [n] if k >= 1 && k <= n => Shape::vector(k),
                [m, n] if k >= 1 && k <= n => Shape::matrix(m, k),
                _ => return Err(shape_err()),
            };
            push(&mut out, ValueType::new(oshape, t.dtype));
            push(&mut out, ValueType::new(oshape, Dtype::U32));
        }
        Op::MatMul(a, b) => {
            let (ta, tb) = (g(a)?, g(b)?);
            if !is_float(ta.dtype) || !is_float(tb.dtype) {
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
            push(&mut out, ValueType::new(Shape::matrix(m, n), Dtype::F32));
        }
        Op::PivotThreshold { input, predicate } => {
            let t = g(input)?;
            if !is_float(t.dtype) {
                return Err(dtype_err());
            }
            if t.shape.rank() != 1 && t.shape.rank() != 2 {
                return Err(shape_err());
            }
            match predicate {
                Predicate::RankLe(k_id) => {
                    let kt = g(k_id)?;
                    if !is_int(kt.dtype) {
                        return Err(dtype_err());
                    }
                    let per_row_ok = t.shape.rank() == 2 && kt.shape.dims() == &t.shape.dims()[..1];
                    if !kt.shape.is_scalar() && !per_row_ok {
                        return Err(shape_err());
                    }
                }
                Predicate::CummassLe(thr) | Predicate::ProbGe(thr) => {
                    let tt = g(thr)?;
                    if !is_float(tt.dtype) {
                        return Err(dtype_err());
                    }
                    let per_row_ok = t.shape.rank() == 2 && tt.shape.dims() == &t.shape.dims()[..1];
                    if !tt.shape.is_scalar() && !per_row_ok {
                        return Err(shape_err());
                    }
                }
            }
            push(&mut out, ValueType::new(t.shape, Dtype::Bool));
        }
        Op::Gather { src, idx } => {
            let (ts, ti) = (g(src)?, g(idx)?);
            if ts.shape.rank() == 0 {
                return Err(shape_err());
            }
            if !is_int(ti.dtype) {
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
            if !is_int(ti.dtype) {
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
            if matches!(op, Op::ScatterAdd { .. }) && !is_numeric(tb.dtype) {
                return Err(dtype_err());
            }
            if !is_int(ti.dtype) {
                return Err(dtype_err());
            }
            if float_result(tb.dtype, tv.dtype).is_none() {
                return Err(dtype_err());
            }
            let expect = axis0_result(ti.shape, tb.shape).ok_or_else(shape_err)?;
            if tv.shape != expect && !tv.shape.is_scalar() {
                return Err(shape_err());
            }
            push(&mut out, tb);
        }
        Op::Iota { len } => {
            if len == 0 {
                return Err(shape_err());
            }
            push(&mut out, ValueType::vector(len, Dtype::U32));
        }
        Op::MaskApply { logits, mask } => {
            let (tl, tm) = (g(logits)?, g(mask)?);
            let n = tl.shape.last_len().ok_or_else(shape_err)?;
            if !is_float(tl.dtype) || tm.dtype != Dtype::U32 {
                return Err(dtype_err());
            }
            match *tm.shape.dims() {
                [w] if w == n.div_ceil(32) => {}
                _ => return Err(shape_err()),
            }
            push(&mut out, tl);
        }
        Op::CausalMask { positions, len }
        | Op::SlidingWindowMask { positions, len, .. }
        | Op::SinkWindowMask { positions, len, .. } => {
            let positions = g(positions)?;
            if positions.dtype != Dtype::U32 {
                return Err(dtype_err());
            }
            let shape = append_dim(positions.shape, len).ok_or_else(shape_err)?;
            push(&mut out, ValueType::new(shape, Dtype::Bool));
        }
        Op::Rng { shape, .. } => {
            if shape.rank() == 0 {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(shape, Dtype::F32));
        }
        Op::RngKeyed { state, shape, .. } => {
            let ts = g(state)?;
            if ts.dtype != Dtype::U32 {
                return Err(dtype_err());
            }
            if *ts.shape.dims() != [2] || shape.rank() == 0 {
                return Err(shape_err());
            }
            push(&mut out, ValueType::new(shape, Dtype::F32));
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

        Op::Output { shape, dtype, .. } => push(&mut out, ValueType::new(shape, dtype)),
        Op::KernelCall {
            name,
            ref args,
            shape,
            dtype,
        } => {
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
    use crate::types::{Literal, RngKind};
    use alloc::vec;
    use alloc::vec::Vec;

    impl ValueType {
        fn matrix_of(m: u32, n: u32, dtype: Dtype) -> Self {
            Self::new(Shape::matrix(m, n), dtype)
        }
    }

    fn ctx_with(channels: &[ValueType]) -> BodyCtx<'_> {
        BodyCtx {
            channel_types: channels,
            n_names: 1,
        }
    }

    /// The bf16 rules (`types`): elementwise ops mix f32 and bf16 into
    /// bf16, a reduction returns its f32 accumulator, a scan and an order
    /// op keep bf16, and a cast converts in and out.
    #[test]
    fn bf16_types_by_the_rules() {
        let chans = [
            ValueType::vector(5, Dtype::F32),
            ValueType::matrix_of(2, 5, Dtype::F32),
        ];
        let ops = vec![
            Op::ChanTake(0), // 0 f32 [5]
            Op::Cast {
                value: 0,
                dtype: Dtype::Bf16,
            }, // 1 bf16 [5]
            Op::Const(Literal::F32(0.5)), // 2 f32 []
            Op::Mul(1, 2),   // 3 bf16
            Op::Add(2, 1),   // 4 bf16
            Op::Exp(3),      // 5 bf16
            Op::ReduceSum(1), // 6 f32 []
            Op::ReduceMax(1), // 7 f32 []
            Op::ReduceArgmax(1), // 8 i32 []
            Op::CumSum(1),   // 9 bf16 [5]
            Op::TopK { input: 1, k: 2 }, // 10 bf16, 11 u32
            Op::SortDesc(1), // 12 bf16, 13 u32
            Op::Gt(1, 0),    // 14 bool
            Op::Select {
                cond: 14,
                a: 1,
                b: 0,
            }, // 15 bf16
            Op::Cast {
                value: 1,
                dtype: Dtype::I32,
            }, // 16 i32
            Op::Cast {
                value: 1,
                dtype: Dtype::F32,
            }, // 17 f32
            Op::PivotThreshold {
                input: 1,
                predicate: Predicate::CummassLe(2),
            }, // 18 bool
            Op::ChanTake(1), // 19 f32 [2, 5]
            Op::Cast {
                value: 19,
                dtype: Dtype::Bf16,
            }, // 20 bf16 [2, 5]
            Op::Transpose(20), // 21 bf16 [5, 2]
            Op::MatMul(20, 21), // 22 f32 [2, 2]
            Op::Iota { len: 1 }, // 23 u32 [1]
            Op::Gather { src: 1, idx: 23 }, // 24 bf16 [1]
            Op::ScatterAdd {
                base: 1,
                idx: 23,
                vals: 2,
            }, // 25 bf16 [5]
        ];
        let types = body_types(&ops, &ctx_with(&chans)).expect("types");
        let dtypes: Vec<Dtype> = types.iter().map(|t| t.dtype).collect();
        use Dtype::*;
        assert_eq!(
            dtypes,
            [
                F32, Bf16, F32, Bf16, Bf16, Bf16, F32, F32, I32, Bf16, Bf16, U32, Bf16, U32, Bool,
                Bf16, I32, F32, Bool, F32, Bf16, Bf16, F32, U32, Bf16, Bf16
            ]
        );
        assert_eq!(types[6].shape, Shape::SCALAR);
        assert_eq!(types[22].shape, Shape::matrix(2, 2));
    }

    #[test]
    fn bf16_is_refused_where_a_value_is_an_index_or_a_mask() {
        let chans = [
            ValueType::vector(4, Dtype::Bf16),
            ValueType::vector(1, Dtype::U32),
        ];
        for op in [
            Op::Gather { src: 1, idx: 0 },
            Op::And(0, 0),
            Op::MaskApply { logits: 1, mask: 0 },
            Op::RngKeyed {
                state: 0,
                shape: Shape::vector(2),
                kind: RngKind::Uniform,
            },
        ] {
            let ops = vec![Op::ChanTake(0), Op::ChanTake(1), op];
            assert_eq!(
                body_types(&ops, &ctx_with(&chans)).unwrap_err().kind,
                BodyErrorKind::DTypeMismatch
            );
        }
        // An int never mixes with bf16.
        let ops = vec![Op::ChanTake(0), Op::ChanTake(1), Op::Add(0, 1)];
        assert!(body_types(&ops, &ctx_with(&chans)).is_err());
    }

    /// `gumbel_argmax` takes a float row, a row-shaped or scalar bool, a
    /// float scalar scale and a `[2]` u32 state, and returns each row's
    /// index.
    #[test]
    fn gumbel_argmax_types() {
        let chans = [
            ValueType::new(Shape::matrix(3, 7), Dtype::Bf16),
            ValueType::new(Shape::matrix(3, 7), Dtype::Bool),
            ValueType::vector(2, Dtype::U32),
        ];
        let draw = |keep, scale| Op::GumbelArgmax {
            logits: 0,
            keep,
            scale,
            state: 2,
        };
        let base = || {
            vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::ChanTake(2),
                Op::Const(Literal::F32(0.7)),
                Op::Const(Literal::Bool(true)),
            ]
        };
        for keep in [1, 4] {
            let mut ops = base();
            ops.push(draw(keep, 3));
            let types = body_types(&ops, &ctx_with(&chans)).expect("types");
            assert_eq!(types[5], ValueType::vector(3, Dtype::I32));
        }
        // The scale is a float scalar; keep is a bool.
        for (keep, scale) in [(1, 1), (3, 3), (1, 0)] {
            let mut ops = base();
            ops.push(draw(keep, scale));
            assert!(
                body_types(&ops, &ctx_with(&chans)).is_err(),
                "{keep} {scale}"
            );
        }
    }

    #[test]
    fn chan_put_type_mismatch_rejected() {
        let chans = [ValueType::vector(4, Dtype::I32)];
        let ops = vec![
            Op::Const(Literal::F32(1.0)),
            Op::ChanPut { chan: 0, value: 0 },
        ];
        assert_eq!(
            body_types(&ops, &ctx_with(&chans)).unwrap_err().kind,
            BodyErrorKind::DTypeMismatch
        );
    }
}
