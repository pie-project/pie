//! `Tensor` — an SSA value (overview §1) — plus the free-function op surface that
//! matches the overview §3/§6 examples verbatim. Ops emit echo's canonical
//! [`ptir::op::Op`](pie_ptir::op::Op); composed ops (`gumbel`,
//! `mask_apply`, `softmax`, …) inline echo's [`expand`](pie_ptir::expand)
//! expansions so a backend that fuses the core fuses these for free.

use alloc::vec::Vec;

use pie_ptir::op::{IntrinsicId, Op};
use pie_ptir::types::{DType, Literal, Predicate, RngKind, Shape, ValueId, ValueType};

use crate::context::emit;

/// An SSA value: a node in the current stage, or a deferred trace-known constant.
#[derive(Clone, Debug)]
pub struct Tensor {
    inner: TensorInner,
}

#[derive(Clone, Debug)]
enum TensorInner {
    Node { id: ValueId, ty: ValueType },
    Const(ConstData),
}

impl Tensor {
    pub(crate) fn node(id: ValueId, ty: ValueType) -> Tensor {
        Tensor {
            inner: TensorInner::Node { id, ty },
        }
    }

    /// A trace-known constant value (overview §1). Accepts a scalar
    /// (`Tensor::constant(-1i32)`) or an array (`Tensor::constant([0u32, 1])`).
    /// A body constant materializes to `Const` (scalar), `Broadcast` (uniform
    /// vector), or `Iota`/affine (a sequence) — the closed op set has no general
    /// vector-const op (overview §1: small consts fold to immediates).
    pub fn constant(v: impl IntoConst) -> Tensor {
        Tensor {
            inner: TensorInner::Const(v.into_const()),
        }
    }

    pub fn ty(&self) -> ValueType {
        match &self.inner {
            TensorInner::Node { ty, .. } => *ty,
            TensorInner::Const(c) => ValueType::new(c.shape, c.dtype),
        }
    }
    pub fn dtype(&self) -> DType {
        self.ty().dtype
    }
    pub fn shape(&self) -> Shape {
        self.ty().shape
    }

    pub(crate) fn as_const_data(&self) -> Option<ConstData> {
        match &self.inner {
            TensorInner::Const(c) => Some(c.clone()),
            TensorInner::Node { .. } => None,
        }
    }
}

/// A trace-known constant value: a typed scalar/vector immediate.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstData {
    pub shape: Shape,
    pub dtype: DType,
    /// Raw little-endian element bytes (4 bytes/element; `bool` = one byte).
    pub bytes: Vec<u8>,
}

/// A borrowed operand: a resolved node, or a not-yet-materialized constant.
#[doc(hidden)]
#[derive(Clone)]
pub enum Arg {
    Node { id: ValueId, ty: ValueType },
    Const(ConstData),
}

impl Arg {
    pub(crate) fn ty(&self) -> ValueType {
        match self {
            Arg::Node { ty, .. } => *ty,
            Arg::Const(c) => ValueType::new(c.shape, c.dtype),
        }
    }
    /// Materialize into the current stage as echo ops, yielding an SSA id + type.
    pub(crate) fn materialize(self) -> (ValueId, ValueType) {
        match self {
            Arg::Node { id, ty } => (id, ty),
            Arg::Const(c) => materialize_const(c),
        }
    }
}

/// Anything usable as a tensor operand: a `Tensor`, a channel take/read result,
/// or a scalar literal (`u32` / `f32`; integer literals resolve to `u32`).
pub trait AsTensor {
    #[doc(hidden)]
    fn to_arg(&self) -> Arg;
}
impl AsTensor for Tensor {
    fn to_arg(&self) -> Arg {
        match &self.inner {
            TensorInner::Node { id, ty } => Arg::Node { id: *id, ty: *ty },
            TensorInner::Const(c) => Arg::Const(c.clone()),
        }
    }
}
impl AsTensor for &Tensor {
    fn to_arg(&self) -> Arg {
        (*self).to_arg()
    }
}
impl AsTensor for u32 {
    fn to_arg(&self) -> Arg {
        Arg::Const(ConstData {
            shape: Shape::SCALAR,
            dtype: DType::U32,
            bytes: self.to_le_bytes().to_vec(),
        })
    }
}
impl AsTensor for f32 {
    fn to_arg(&self) -> Arg {
        Arg::Const(ConstData {
            shape: Shape::SCALAR,
            dtype: DType::F32,
            bytes: self.to_le_bytes().to_vec(),
        })
    }
}

// ---------------------------------------------------------------------------
// Constant → echo op materialization
// ---------------------------------------------------------------------------

fn scalar_literal(dtype: DType, bytes: &[u8]) -> Literal {
    let w = |i: usize| bytes.get(i).copied().unwrap_or(0);
    let word = [w(0), w(1), w(2), w(3)];
    match dtype {
        DType::F32 => Literal::F32(f32::from_le_bytes(word)),
        DType::I32 => Literal::I32(i32::from_le_bytes(word)),
        DType::U32 => Literal::U32(u32::from_le_bytes(word)),
        DType::Bool => Literal::Bool(w(0) != 0),
    }
}

fn elem_at(dtype: DType, bytes: &[u8], i: usize) -> f64 {
    match dtype {
        DType::Bool => (bytes.get(i).copied().unwrap_or(0) != 0) as u8 as f64,
        _ => {
            let o = i * 4;
            let word = [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
            match dtype {
                DType::F32 => f32::from_le_bytes(word) as f64,
                DType::I32 => i32::from_le_bytes(word) as f64,
                DType::U32 => u32::from_le_bytes(word) as f64,
                DType::Bool => unreachable!(),
            }
        }
    }
}

fn elem_size(d: DType) -> usize {
    match d {
        DType::Bool => 1,
        _ => 4,
    }
}

/// Lower a trace-known constant to echo ops (see [`Tensor::constant`]).
fn materialize_const(c: ConstData) -> (ValueId, ValueType) {
    let ty = ValueType::new(c.shape, c.dtype);
    if c.shape.is_scalar() {
        let id = emit(
            Op::Const(scalar_literal(c.dtype, &c.bytes)),
            &[ValueType::scalar(c.dtype)],
        );
        return (id, ty);
    }
    let n = c.shape.numel() as usize;
    let vals: Vec<f64> = (0..n).map(|i| elem_at(c.dtype, &c.bytes, i)).collect();

    // uniform ⇒ broadcast(scalar).
    if !vals.is_empty() && vals.iter().all(|&v| v == vals[0]) {
        let s = emit(
            Op::Const(scalar_literal(c.dtype, &c.bytes[..elem_size(c.dtype)])),
            &[ValueType::scalar(c.dtype)],
        );
        let id = emit(
            Op::Broadcast {
                value: s,
                shape: c.shape,
            },
            &[ty],
        );
        return (id, ty);
    }
    // affine `a + b*i` over U32 ⇒ iota (+ optional mul/add).
    if c.dtype == DType::U32 && n >= 2 {
        let a = vals[0];
        let b = vals[1] - vals[0];
        if b >= 0.0 && vals.iter().enumerate().all(|(i, &v)| v == a + b * i as f64) {
            let io = emit(Op::Iota { len: n as u32 }, &[ty]);
            let mut cur = io;
            if b != 1.0 {
                let bc = emit(
                    Op::Const(Literal::U32(b as u32)),
                    &[ValueType::scalar(DType::U32)],
                );
                cur = emit(Op::Mul(cur, bc), &[ty]);
            }
            if a != 0.0 {
                let ac = emit(
                    Op::Const(Literal::U32(a as u32)),
                    &[ValueType::scalar(DType::U32)],
                );
                cur = emit(Op::Add(cur, ac), &[ty]);
            }
            return (cur, ty);
        }
    }
    panic!(
        "general vector constant {vals:?} (dtype {:?}) is not representable in the closed op set; \
         use iota/broadcast, an arithmetic expression, or feed it through a channel",
        c.dtype
    );
}

// ---------------------------------------------------------------------------
// Constant construction (author-facing)
// ---------------------------------------------------------------------------

/// Anything convertible to a trace-known [`ConstData`]: scalars and arrays of
/// `f32` / `i32` / `u32` / `bool`.
pub trait IntoConst {
    fn into_const(self) -> ConstData;
}

fn scalar_bytes_of(v: f64, dt: DType) -> Vec<u8> {
    match dt {
        DType::F32 => (v as f32).to_le_bytes().to_vec(),
        DType::I32 => (v as i32).to_le_bytes().to_vec(),
        DType::U32 => (v as u32).to_le_bytes().to_vec(),
        DType::Bool => alloc::vec![(v != 0.0) as u8],
    }
}

macro_rules! num_const {
    ($t:ty, $dt:expr) => {
        impl IntoConst for $t {
            fn into_const(self) -> ConstData {
                ConstData {
                    shape: Shape::SCALAR,
                    dtype: $dt,
                    bytes: scalar_bytes_of(self as f64, $dt),
                }
            }
        }
        impl<const N: usize> IntoConst for [$t; N] {
            fn into_const(self) -> ConstData {
                let mut bytes = Vec::new();
                for x in self {
                    bytes.extend_from_slice(&scalar_bytes_of(x as f64, $dt));
                }
                ConstData {
                    shape: Shape::vector(N as u32),
                    dtype: $dt,
                    bytes,
                }
            }
        }
        impl IntoConst for Vec<$t> {
            fn into_const(self) -> ConstData {
                let n = self.len() as u32;
                let mut bytes = Vec::new();
                for x in self {
                    bytes.extend_from_slice(&scalar_bytes_of(x as f64, $dt));
                }
                ConstData {
                    shape: Shape::vector(n),
                    dtype: $dt,
                    bytes,
                }
            }
        }
    };
}
num_const!(f32, DType::F32);
num_const!(i32, DType::I32);
num_const!(u32, DType::U32);

impl IntoConst for bool {
    fn into_const(self) -> ConstData {
        ConstData {
            shape: Shape::SCALAR,
            dtype: DType::Bool,
            bytes: alloc::vec![self as u8],
        }
    }
}
impl<const N: usize> IntoConst for [bool; N] {
    fn into_const(self) -> ConstData {
        ConstData {
            shape: Shape::vector(N as u32),
            dtype: DType::Bool,
            bytes: self.iter().map(|&b| b as u8).collect(),
        }
    }
}
impl IntoConst for Vec<bool> {
    fn into_const(self) -> ConstData {
        let n = self.len() as u32;
        ConstData {
            shape: Shape::vector(n),
            dtype: DType::Bool,
            bytes: self.iter().map(|&b| b as u8).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// Anything usable as a shape argument: a `[u32; N]` dim array or a [`Shape`].
pub trait IntoShape {
    fn into_shape(self) -> Shape;
}
impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}
impl<const N: usize> IntoShape for [u32; N] {
    fn into_shape(self) -> Shape {
        Shape::new(&self).expect("shape rank exceeds MAX_RANK")
    }
}

// ---------------------------------------------------------------------------
// Op emission helpers
// ---------------------------------------------------------------------------

fn non_scalar_shape(a: Shape, b: Shape) -> Shape {
    if a.is_scalar() { b } else { a }
}

fn reconcile(a: Arg, b: Arg) -> (Arg, Arg) {
    fn coerce(c: &ConstData, to: DType) -> Option<ConstData> {
        if c.dtype == to || !c.shape.is_scalar() {
            return None;
        }
        let v = elem_at(c.dtype, &c.bytes, 0);
        Some(ConstData {
            shape: Shape::SCALAR,
            dtype: to,
            bytes: scalar_bytes_of(v, to),
        })
    }
    match (&a, &b) {
        (Arg::Const(ca), Arg::Node { ty, .. }) => {
            if let Some(c) = coerce(ca, ty.dtype) {
                return (Arg::Const(c), b);
            }
        }
        (Arg::Node { ty, .. }, Arg::Const(cb)) => {
            if let Some(c) = coerce(cb, ty.dtype) {
                return (a, Arg::Const(c));
            }
        }
        _ => {}
    }
    (a, b)
}

fn emit_unary(
    x: &impl AsTensor,
    mk: impl FnOnce(ValueId) -> Op,
    out: impl FnOnce(ValueType) -> ValueType,
) -> Tensor {
    let (id, ty) = x.to_arg().materialize();
    let rty = out(ty);
    Tensor::node(emit(mk(id), &[rty]), rty)
}

fn emit_binary(
    a: &impl AsTensor,
    b: &impl AsTensor,
    mk: impl FnOnce(ValueId, ValueId) -> Op,
    result_dtype: impl FnOnce(DType) -> DType,
) -> Tensor {
    let (aa, bb) = reconcile(a.to_arg(), b.to_arg());
    let shape = non_scalar_shape(aa.ty().shape, bb.ty().shape);
    let (ia, tya) = aa.materialize();
    let (ib, _) = bb.materialize();
    let rty = ValueType::new(shape, result_dtype(tya.dtype));
    Tensor::node(emit(mk(ia, ib), &[rty]), rty)
}

fn reduce_shape(s: Shape) -> Shape {
    s.drop_last().unwrap_or(Shape::SCALAR)
}

/// Push a raw op (for expansion inlining), returning its first result id.
fn push(op: Op, tys: &[ValueType]) -> ValueId {
    emit(op, tys)
}

// ---------------------------------------------------------------------------
// The free-function op surface (overview appendix; matches §3/§6 verbatim)
// ---------------------------------------------------------------------------

// -- map: unary --
pub fn neg(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Neg, |t| t)
}
pub fn exp(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Exp, |t| t)
}
pub fn log(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Log, |t| t)
}
pub fn cast(x: impl AsTensor, to: DType) -> Tensor {
    emit_unary(
        &x,
        move |id| Op::Cast {
            value: id,
            dtype: to,
        },
        move |t| ValueType::new(t.shape, to),
    )
}

// -- map: binary --
pub fn add(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Add, |d| d)
}
pub fn sub(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Sub, |d| d)
}
pub fn mul(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Mul, |d| d)
}
pub fn div(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Div, |d| d)
}
pub fn rem(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Rem, |d| d)
}
pub fn max_elem(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::MaxElem, |d| d)
}
pub fn min_elem(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::MinElem, |d| d)
}

// -- compare / logic (bool results) --
pub fn eq(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Eq, |_| DType::Bool)
}
pub fn ne(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Ne, |_| DType::Bool)
}
pub fn lt(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Lt, |_| DType::Bool)
}
pub fn le(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Le, |_| DType::Bool)
}
pub fn gt(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Gt, |_| DType::Bool)
}
pub fn ge(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Ge, |_| DType::Bool)
}
pub fn and(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::And, |_| DType::Bool)
}
pub fn or(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Or, |_| DType::Bool)
}
pub fn not(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Not, |t| ValueType::new(t.shape, DType::Bool))
}

// -- choice --
pub fn select(cond: impl AsTensor, a: impl AsTensor, b: impl AsTensor) -> Tensor {
    let (ca, _) = cond.to_arg().materialize();
    let (aa, bb) = reconcile(a.to_arg(), b.to_arg());
    let shape = non_scalar_shape(aa.ty().shape, bb.ty().shape);
    let (ia, tya) = aa.materialize();
    let (ib, _) = bb.materialize();
    let rty = ValueType::new(shape, tya.dtype);
    Tensor::node(
        emit(
            Op::Select {
                cond: ca,
                a: ia,
                b: ib,
            },
            &[rty],
        ),
        rty,
    )
}

// -- shape --
pub fn reshape(x: impl AsTensor, shape: impl IntoShape) -> Tensor {
    let s = shape.into_shape();
    emit_unary(
        &x,
        move |id| Op::Reshape {
            value: id,
            shape: s,
        },
        move |t| ValueType::new(s, t.dtype),
    )
}
pub fn broadcast(x: impl AsTensor, shape: impl IntoShape) -> Tensor {
    let s = shape.into_shape();
    emit_unary(
        &x,
        move |id| Op::Broadcast {
            value: id,
            shape: s,
        },
        move |t| ValueType::new(s, t.dtype),
    )
}
/// `transpose(x)` — rank-2 transpose `[m, n] → [n, m]`.
pub fn transpose(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Transpose, |t| {
        let d = t.shape.dims();
        let s = if d.len() == 2 {
            Shape::matrix(d[1], d[0])
        } else {
            t.shape
        };
        ValueType::new(s, t.dtype)
    })
}

// -- index --
pub fn iota(len: u32) -> Tensor {
    let ty = ValueType::new(Shape::vector(len), DType::U32);
    Tensor::node(emit(Op::Iota { len }, &[ty]), ty)
}
/// Axis-0 generalized gather: `gather(src[n, rest..], idx[S..]) -> [S.., rest..]`.
pub fn gather(src: impl AsTensor, idx: impl AsTensor) -> Tensor {
    let (is, tys) = src.to_arg().materialize();
    let (ii, tyi) = idx.to_arg().materialize();
    let mut dims: Vec<u32> = tyi.shape.dims().to_vec();
    let src_rest = &tys.shape.dims()[tys.shape.rank().min(1)..];
    dims.extend_from_slice(src_rest);
    let rshape = Shape::new(&dims).expect("gather result rank");
    let rty = ValueType::new(rshape, tys.dtype);
    Tensor::node(emit(Op::Gather { src: is, idx: ii }, &[rty]), rty)
}
/// Per-row column pick `out[i] = src[i, idx[i]]` (`src[m, n]`, `idx[m]` → `[m]`).
pub fn gather_row(src: impl AsTensor, idx: impl AsTensor) -> Tensor {
    let (is, tys) = src.to_arg().materialize();
    let (ii, _) = idx.to_arg().materialize();
    let m = tys.shape.dims().first().copied().unwrap_or(0);
    let rty = ValueType::new(Shape::vector(m), tys.dtype);
    Tensor::node(emit(Op::GatherRow { src: is, idx: ii }, &[rty]), rty)
}
pub fn scatter_set(base: impl AsTensor, idx: impl AsTensor, vals: impl AsTensor) -> Tensor {
    let (ib, tyb) = base.to_arg().materialize();
    let (ii, _) = idx.to_arg().materialize();
    let (iv, _) = vals.to_arg().materialize();
    Tensor::node(
        emit(
            Op::ScatterSet {
                base: ib,
                idx: ii,
                vals: iv,
            },
            &[tyb],
        ),
        tyb,
    )
}
pub fn scatter_add(base: impl AsTensor, idx: impl AsTensor, vals: impl AsTensor) -> Tensor {
    let (ib, tyb) = base.to_arg().materialize();
    let (ii, _) = idx.to_arg().materialize();
    let (iv, _) = vals.to_arg().materialize();
    Tensor::node(
        emit(
            Op::ScatterAdd {
                base: ib,
                idx: ii,
                vals: iv,
            },
            &[tyb],
        ),
        tyb,
    )
}

// -- reduce / scan --
pub fn reduce_sum(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceSum, |t| {
        ValueType::new(reduce_shape(t.shape), t.dtype)
    })
}
pub fn reduce_max(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceMax, |t| {
        ValueType::new(reduce_shape(t.shape), t.dtype)
    })
}
pub fn reduce_min(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceMin, |t| {
        ValueType::new(reduce_shape(t.shape), t.dtype)
    })
}
/// Argmax over the last axis → `I32` token id(s).
pub fn reduce_argmax(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceArgmax, |t| {
        ValueType::new(reduce_shape(t.shape), DType::I32)
    })
}
pub fn cumsum(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::CumSum, |t| t)
}
pub fn cumprod(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::CumProd, |t| t)
}

// -- normalize (echo's expand.rs expansions, type-tracked) --
pub fn softmax(x: impl AsTensor) -> Tensor {
    let (xid, ty) = x.to_arg().materialize();
    let (s, red) = (ty.shape, reduce_shape(ty.shape));
    let m = push(Op::ReduceMax(xid), &[ValueType::new(red, DType::F32)]);
    let mb = push(
        Op::Broadcast { value: m, shape: s },
        &[ValueType::new(s, DType::F32)],
    );
    let c = push(Op::Sub(xid, mb), &[ValueType::new(s, DType::F32)]);
    let e = push(Op::Exp(c), &[ValueType::new(s, DType::F32)]);
    let sum = push(Op::ReduceSum(e), &[ValueType::new(red, DType::F32)]);
    let sb = push(
        Op::Broadcast {
            value: sum,
            shape: s,
        },
        &[ValueType::new(s, DType::F32)],
    );
    let out = push(Op::Div(e, sb), &[ValueType::new(s, DType::F32)]);
    Tensor::node(out, ValueType::new(s, DType::F32))
}
pub fn log_softmax(x: impl AsTensor) -> Tensor {
    let (xid, ty) = x.to_arg().materialize();
    let (s, red) = (ty.shape, reduce_shape(ty.shape));
    let m = push(Op::ReduceMax(xid), &[ValueType::new(red, DType::F32)]);
    let mb = push(
        Op::Broadcast { value: m, shape: s },
        &[ValueType::new(s, DType::F32)],
    );
    let c = push(Op::Sub(xid, mb), &[ValueType::new(s, DType::F32)]);
    let e = push(Op::Exp(c), &[ValueType::new(s, DType::F32)]);
    let sum = push(Op::ReduceSum(e), &[ValueType::new(red, DType::F32)]);
    let l = push(Op::Log(sum), &[ValueType::new(red, DType::F32)]);
    let lb = push(
        Op::Broadcast { value: l, shape: s },
        &[ValueType::new(s, DType::F32)],
    );
    let out = push(Op::Sub(c, lb), &[ValueType::new(s, DType::F32)]);
    Tensor::node(out, ValueType::new(s, DType::F32))
}
pub fn l2norm(x: impl AsTensor) -> Tensor {
    let (xid, ty) = x.to_arg().materialize();
    let (s, red) = (ty.shape, reduce_shape(ty.shape));
    let sq = push(Op::Mul(xid, xid), &[ValueType::new(s, DType::F32)]);
    let sum = push(Op::ReduceSum(sq), &[ValueType::new(red, DType::F32)]);
    let lg = push(Op::Log(sum), &[ValueType::new(red, DType::F32)]);
    let half = push(
        Op::Const(Literal::F32(0.5)),
        &[ValueType::scalar(DType::F32)],
    );
    let h = push(Op::Mul(lg, half), &[ValueType::new(red, DType::F32)]);
    let rt = push(Op::Exp(h), &[ValueType::new(red, DType::F32)]);
    let rb = push(
        Op::Broadcast {
            value: rt,
            shape: s,
        },
        &[ValueType::new(s, DType::F32)],
    );
    let out = push(Op::Div(xid, rb), &[ValueType::new(s, DType::F32)]);
    Tensor::node(out, ValueType::new(s, DType::F32))
}

// -- order --
/// `top_k(x, k)` → `(values, indices)`. `k` is a trace-known immediate (§5.1).
/// `values` keep `x`'s dtype; `indices` are `U32`; last axis becomes length `k`.
pub fn top_k(x: impl AsTensor, k: u32) -> (Tensor, Tensor) {
    let (ix, tyx) = x.to_arg().materialize();
    let mut dims: Vec<u32> = tyx.shape.dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = k;
    }
    let out_shape = Shape::new(&dims).unwrap_or_else(|| Shape::vector(k));
    let val_ty = ValueType::new(out_shape, tyx.dtype);
    let idx_ty = ValueType::new(out_shape, DType::U32);
    let base = emit(Op::TopK { input: ix, k }, &[val_ty, idx_ty]);
    (Tensor::node(base, val_ty), Tensor::node(base + 1, idx_ty))
}
/// The top-k rank predicate (`pivot_threshold(x, rank_le(k))`).
pub fn rank_le(k: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::RankLe(k.to_arg()))
}
pub fn cummass_le(p: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::CummassLe(p.to_arg()))
}
pub fn prob_ge(thr: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::ProbGe(thr.to_arg()))
}
/// A pivot predicate carrying its (not-yet-materialized) cut value.
pub struct PredicateArg(PredKind);
enum PredKind {
    RankLe(Arg),
    CummassLe(Arg),
    ProbGe(Arg),
}
/// `pivot_threshold(input, predicate)` → bool keep-mask, same shape.
pub fn pivot_threshold(input: impl AsTensor, predicate: PredicateArg) -> Tensor {
    let (ii, tyi) = input.to_arg().materialize();
    let pred = match predicate.0 {
        PredKind::RankLe(a) => Predicate::RankLe(a.materialize().0),
        PredKind::CummassLe(a) => Predicate::CummassLe(a.materialize().0),
        PredKind::ProbGe(a) => Predicate::ProbGe(a.materialize().0),
    };
    let rty = ValueType::new(tyi.shape, DType::Bool);
    let id = emit(
        Op::PivotThreshold {
            input: ii,
            predicate: pred,
        },
        &[rty],
    );
    Tensor::node(id, rty)
}

// -- linear --
pub fn matmul(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    let (ia, tya) = a.to_arg().materialize();
    let (ib, tyb) = b.to_arg().materialize();
    let m = tya.shape.dims().first().copied().unwrap_or(0);
    let n = tyb.shape.dims().last().copied().unwrap_or(0);
    let rty = ValueType::new(Shape::matrix(m, n), DType::F32);
    Tensor::node(emit(Op::MatMul(ia, ib), &[rty]), rty)
}

// -- sampling (echo's expand: gumbel = RngKeyed; mask_apply = select(mask, x, -inf)) --
/// `gumbel(state, shape)` — Gumbel noise, a pure function of the `[2]` U32 rng
/// `state` (`[key, ctr]`) + element index (overview §3; replay-deterministic T8).
pub fn gumbel(state: impl AsTensor, shape: impl IntoShape) -> Tensor {
    rng_noise(state, shape, RngKind::Gumbel)
}
/// `rng(state, shape)` — state-keyed uniform `[0,1)` noise, same determinism.
pub fn rng(state: impl AsTensor, shape: impl IntoShape) -> Tensor {
    rng_noise(state, shape, RngKind::Uniform)
}
fn rng_noise(state: impl AsTensor, shape: impl IntoShape, kind: RngKind) -> Tensor {
    let s = shape.into_shape();
    let (istate, _) = state.to_arg().materialize();
    let rty = ValueType::new(s, DType::F32);
    Tensor::node(
        emit(
            Op::RngKeyed {
                state: istate,
                shape: s,
                kind,
            },
            &[rty],
        ),
        rty,
    )
}
/// `mask_apply(logits, mask)` — bool-mask over logits (allowed → pass, else −∞),
/// expanded to `select(mask, logits, -inf)` (echo's composed form).
pub fn mask_apply(logits: impl AsTensor, mask: impl AsTensor) -> Tensor {
    let (il, tyl) = logits.to_arg().materialize();
    let (im, _) = mask.to_arg().materialize();
    let ninf = push(
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        &[ValueType::scalar(DType::F32)],
    );
    Tensor::node(
        emit(
            Op::Select {
                cond: im,
                a: il,
                b: ninf,
            },
            &[tyl],
        ),
        tyl,
    )
}

fn append_mask_axis(shape: Shape, len: u32) -> Shape {
    let mut dims = shape.dims().to_vec();
    dims.push(len);
    Shape::new(&dims).expect("structured mask rank")
}

pub fn causal_mask(positions: impl AsTensor, len: u32) -> Tensor {
    let (positions, ty) = positions.to_arg().materialize();
    let result = ValueType::new(append_mask_axis(ty.shape, len), DType::Bool);
    Tensor::node(emit(Op::CausalMask { positions, len }, &[result]), result)
}

pub fn sliding_window_mask(positions: impl AsTensor, len: u32, window: u32) -> Tensor {
    let (positions, ty) = positions.to_arg().materialize();
    let result = ValueType::new(append_mask_axis(ty.shape, len), DType::Bool);
    Tensor::node(
        emit(
            Op::SlidingWindowMask {
                positions,
                len,
                window,
            },
            &[result],
        ),
        result,
    )
}

pub fn sink_window_mask(positions: impl AsTensor, len: u32, sink: u32, window: u32) -> Tensor {
    let (positions, ty) = positions.to_arg().materialize();
    let result = ValueType::new(append_mask_axis(ty.shape, len), DType::Bool);
    Tensor::node(
        emit(
            Op::SinkWindowMask {
                positions,
                len,
                sink,
                window,
            },
            &[result],
        ),
        result,
    )
}

/// For every row and key, report whether the key occurs anywhere in the row.
/// This is ordinary SSA composition; it introduces no wire opcode.
pub fn row_membership(rows: impl AsTensor, keys: impl AsTensor) -> Tensor {
    let rows = rows.to_arg();
    let keys = keys.to_arg();
    let row_type = rows.ty();
    let key_type = keys.ty();
    let [row_count, depth] = *row_type.shape.dims() else {
        panic!("row_membership rows must have shape [R,D]");
    };
    let [key_count] = *key_type.shape.dims() else {
        panic!("row_membership keys must have shape [K]");
    };
    assert_eq!(
        row_type.dtype, key_type.dtype,
        "row_membership rows and keys must have the same dtype"
    );

    let row_stride = key_count
        .checked_mul(depth)
        .expect("row_membership shape overflow");
    let row_flat_len = row_count
        .checked_mul(depth)
        .expect("row_membership shape overflow");
    let flat_len = row_count
        .checked_mul(key_count)
        .and_then(|value| value.checked_mul(depth))
        .expect("row_membership shape overflow");
    let (rows, _) = rows.materialize();
    let (keys, _) = keys.materialize();
    let rows = Tensor::node(rows, row_type);
    let keys = Tensor::node(keys, key_type);
    let linear = iota(flat_len);
    let row_index = div(&linear, row_stride);
    let depth_index = rem(&linear, depth);
    let row_value_index = add(mul(row_index, depth), depth_index);
    let row_values = gather(reshape(rows, [row_flat_len]), row_value_index);
    let key_index = rem(div(&linear, depth), key_count);
    let key_values = gather(keys, key_index);
    let matches = eq(
        reshape(row_values, [row_count, key_count, depth]),
        reshape(key_values, [row_count, key_count, depth]),
    );
    cast(reduce_max(cast(matches, DType::U32)), DType::Bool)
}

/// Masked argmax expressed over ordinary PTIR primitives.
pub fn masked_argmax(logits: impl AsTensor, mask: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (mask, _) = mask.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(logits_type.shape), DType::I32);
    let negative_infinity = push(
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        &[ValueType::scalar(DType::F32)],
    );
    let masked = push(
        Op::Select {
            cond: mask,
            a: logits,
            b: negative_infinity,
        },
        &[logits_type],
    );
    let result = push(Op::ReduceArgmax(masked), &[result_type]);
    Tensor::node(result, result_type)
}

/// Semantic Gumbel-max sampler over the input's complete shape.
pub fn gumbel_max(logits: impl AsTensor, state: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (state, _) = state.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(logits_type.shape), DType::I32);
    let noise = push(
        Op::RngKeyed {
            state,
            shape: logits_type.shape,
            kind: RngKind::Gumbel,
        },
        &[ValueType::new(logits_type.shape, DType::F32)],
    );
    let perturbed = push(Op::Add(logits, noise), &[logits_type]);
    let result = push(Op::ReduceArgmax(perturbed), &[result_type]);
    Tensor::node(result, result_type)
}

/// Shannon entropy `-sum(p * log(p))`.
pub fn entropy(probabilities: impl AsTensor) -> Tensor {
    let (probabilities, probability_type) = probabilities.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(probability_type.shape), DType::F32);
    let log_probabilities = push(Op::Log(probabilities), &[probability_type]);
    let terms = push(
        Op::Mul(probabilities, log_probabilities),
        &[probability_type],
    );
    let sum = push(Op::ReduceSum(terms), &[result_type]);
    let result = push(Op::Neg(sum), &[result_type]);
    Tensor::node(result, result_type)
}

/// Entropy when both probabilities and log-probabilities already exist.
pub fn entropy_from_logprobs(
    probabilities: impl AsTensor,
    log_probabilities: impl AsTensor,
) -> Tensor {
    let (probabilities, probability_type) = probabilities.to_arg().materialize();
    let (log_probabilities, _) = log_probabilities.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(probability_type.shape), DType::F32);
    let terms = push(
        Op::Mul(probabilities, log_probabilities),
        &[probability_type],
    );
    let sum = push(Op::ReduceSum(terms), &[result_type]);
    let result = push(Op::Neg(sum), &[result_type]);
    Tensor::node(result, result_type)
}

/// Compiler-visible gather of the scalar selected by a token/index result.
pub fn scalar_gather(src: impl AsTensor, index: impl AsTensor) -> Tensor {
    let (src, src_type) = src.to_arg().materialize();
    let (index, index_type) = index.to_arg().materialize();
    let (op, result_shape) = if let [rows, _] = src_type.shape.dims() {
        assert_eq!(
            index_type.shape.dims(),
            &[*rows],
            "scalar_gather over a matrix requires one index per row"
        );
        (Op::GatherRow { src, idx: index }, Shape::vector(*rows))
    } else {
        let mut dimensions: Vec<u32> = index_type.shape.dims().to_vec();
        dimensions.extend_from_slice(&src_type.shape.dims()[src_type.shape.rank().min(1)..]);
        (
            Op::Gather { src, idx: index },
            Shape::new(&dimensions).expect("scalar gather result rank"),
        )
    };
    let result_type = ValueType::new(result_shape, src_type.dtype);
    let result = emit(op, &[result_type]);
    Tensor::node(result, result_type)
}

/// Exact nucleus sampler expressed entirely as ordinary composable SSA.
/// Temperature scaling remains an ordinary preceding operation.
pub fn nucleus_sample(logits: impl AsTensor, top_p: impl AsTensor, state: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (top_p, _) = top_p.to_arg().materialize();
    let (state, _) = state.to_arg().materialize();
    let row_type = ValueType::new(reduce_shape(logits_type.shape), DType::F32);
    let token_type = ValueType::new(reduce_shape(logits_type.shape), DType::I32);

    let maximum = push(Op::ReduceMax(logits), &[row_type]);
    let maximum = push(
        Op::Broadcast {
            value: maximum,
            shape: logits_type.shape,
        },
        &[logits_type],
    );
    let centered = push(Op::Sub(logits, maximum), &[logits_type]);
    let exponentials = push(Op::Exp(centered), &[logits_type]);
    let sum = push(Op::ReduceSum(exponentials), &[row_type]);
    let sum = push(
        Op::Broadcast {
            value: sum,
            shape: logits_type.shape,
        },
        &[logits_type],
    );
    let probabilities = push(Op::Div(exponentials, sum), &[logits_type]);
    let keep = push(
        Op::PivotThreshold {
            input: probabilities,
            predicate: Predicate::CummassLe(top_p),
        },
        &[ValueType::new(logits_type.shape, DType::Bool)],
    );
    let negative_infinity = push(
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        &[ValueType::scalar(DType::F32)],
    );
    let masked = push(
        Op::Select {
            cond: keep,
            a: logits,
            b: negative_infinity,
        },
        &[logits_type],
    );
    let noise = push(
        Op::RngKeyed {
            state,
            shape: logits_type.shape,
            kind: RngKind::Gumbel,
        },
        &[logits_type],
    );
    let perturbed = push(Op::Add(masked, noise), &[logits_type]);
    let result = push(Op::ReduceArgmax(perturbed), &[token_type]);
    Tensor::node(result, token_type)
}

// -- intrinsic value leaf (used by `intrinsics`) --
pub(crate) fn intrinsic_val(intr: IntrinsicId, shape: Shape, dtype: DType) -> Tensor {
    let ty = ValueType::new(shape, dtype);
    Tensor::node(emit(Op::IntrinsicVal { intr, shape, dtype }, &[ty]), ty)
}

/// Reshape a value id to `target` if it differs but numel matches (used by
/// `Channel::put` to fit a scalar into a `[1]` channel).
pub(crate) fn reshape_id_to(id: ValueId, from: ValueType, target: Shape) -> ValueId {
    if from.shape == target {
        return id;
    }
    emit(
        Op::Reshape {
            value: id,
            shape: target,
        },
        &[ValueType::new(target, from.dtype)],
    )
}
