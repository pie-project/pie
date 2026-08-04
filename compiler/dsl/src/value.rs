//! `Tensor` — an SSA value — plus the free-function op surface. Ops emit the
//! IR's canonical [`ptir::op::Op`](pie_ir::op::Op); composed ops (`gumbel`,
//! `mask_apply`, `softmax`, …) inline the IR's [`expand`]
//! expansions so a backend that fuses the core fuses these for free.

use alloc::vec::Vec;

use pie_ir::container::const_elem_size;
use pie_ir::expand;
use pie_ir::op::{IntrinsicId, Op};
use pie_ir::types::{DType, Literal, Predicate, RngKind, Shape, ValueId, ValueType};

use crate::context::{self, emit};
use crate::error::Span;

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

    /// A trace-known constant value. Accepts a scalar
    /// (`Tensor::constant(-1i32)`) or an array (`Tensor::constant([0u32, 1])`).
    ///
    /// The op set carries constants as *scalars*: `const` holds one
    /// [`Literal`] and nothing else. So the tensor constants that lower are
    /// exactly the ones a scalar plus one op can spell — a uniform tensor
    /// (`broadcast`), and a `u32` affine ramp `a + b*i` (`iota`, then the
    /// arithmetic). Anything else — a per-token logit bias, a banned-token
    /// set, a sampling schedule — is *bulk data*, and bulk data lives in a
    /// channel:
    ///
    /// ```ignore
    /// let bias = Channel::from(vec![0.0f32, -3.5, 0.0, 12.25]).named("bias");
    /// // ... in a stage body:
    /// let logits = logits + bias.read();
    /// ```
    ///
    /// That is not a workaround for a missing op. The op stream is a stream of
    /// fixed-size records — it is what lets a backend emit a kernel from an op
    /// tag alone — and a buffer of arbitrary bytes is not a record. A channel
    /// is this system's name for a buffer the device reads, so a constant
    /// tensor is a channel that is seeded once and never written again.
    ///
    /// Passing anything else here records a trace error naming the shape and
    /// pointing at the channel.
    pub fn constant(v: impl IntoConst) -> Tensor {
        Tensor {
            inner: TensorInner::Const(v.into_const()),
        }
    }

    /// The value's static [`ValueType`] (shape and dtype).
    pub fn ty(&self) -> ValueType {
        match &self.inner {
            TensorInner::Node { ty, .. } => *ty,
            TensorInner::Const(c) => ValueType::new(c.shape, c.dtype),
        }
    }
    /// The value's element [`DType`].
    pub fn dtype(&self) -> DType {
        self.ty().dtype
    }
    /// The value's [`Shape`].
    pub fn shape(&self) -> Shape {
        self.ty().shape
    }

    /// Ceiling division, the same spelling std gives the host scalars
    /// (`u32::div_ceil`): `n.div_ceil(page_size)` reads the same whether `n`
    /// is a prompt length on the host or a device-resolved KV length.
    ///
    /// This is the page-count arithmetic every decode epilogue does, and
    /// spelling it `(n + (page_size - 1)) / page_size` hid a rounding rule
    /// behind an off-by-one that had to be re-read every time.
    ///
    /// A trace-known scalar divisor has its `- 1` folded here, so the emitted
    /// ops are exactly the ones the hand-written form produced.
    pub fn div_ceil(&self, rhs: impl AsTensor) -> Tensor {
        let d = rhs.to_arg();
        match const_scalar(&d) {
            Some(v) => {
                let one_less = Tensor::from_arg(scalar_arg(v - 1.0, d.ty().dtype));
                (self + one_less) / Tensor::from_arg(d)
            }
            None => {
                let d = Tensor::from_arg(d);
                (self + &d - 1u32) / &d
            }
        }
    }

    fn from_arg(a: Arg) -> Tensor {
        match a {
            Arg::Node { id, ty } => Tensor::node(id, ty),
            Arg::Const(c) => Tensor {
                inner: TensorInner::Const(c),
            },
        }
    }
}

/// The numeric value of a trace-known scalar operand, if it is one.
fn const_scalar(a: &Arg) -> Option<f64> {
    let Arg::Const(c) = a else { return None };
    if !c.shape.is_scalar() {
        return None;
    }
    let b = c.bytes.as_slice();
    Some(match c.dtype {
        DType::Bool => (b.first().copied().unwrap_or(0) != 0) as u8 as f64,
        DType::F32 => f32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
        DType::I32 => i32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
        DType::U32 => u32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
    })
}

/// A scalar constant operand of the given dtype.
fn scalar_arg(v: f64, dtype: DType) -> Arg {
    Arg::Const(ConstData {
        shape: Shape::SCALAR,
        dtype,
        bytes: scalar_bytes_of(v, dtype),
    })
}

/// A trace-known constant value: a typed scalar/vector immediate.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstData {
    /// Logical shape; its `numel` must match the element count in `bytes`.
    pub shape: Shape,
    /// Element dtype the `bytes` are decoded as.
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
    /// Materialize into the current stage as IR ops, yielding an SSA id + type.
    pub(crate) fn materialize(self) -> (ValueId, ValueType) {
        match self {
            Arg::Node { id, ty } => (id, ty),
            Arg::Const(c) => materialize_const(c),
        }
    }
}

/// Anything usable as a tensor operand: a `Tensor`, a channel take/read result,
/// or a scalar literal (`u32`, `i32`, `f32` or `bool`). A scalar operand takes
/// on the dtype of the tensor it is combined with, so the literal's own suffix
/// only matters when both operands are scalars.
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
macro_rules! as_tensor_scalar {
    ($($t:ty),*) => {$(
        impl AsTensor for $t {
            fn to_arg(&self) -> Arg {
                Arg::Const((*self).into_const())
            }
        }
    )*};
}
as_tensor_scalar!(u32, i32, f32, bool);

// ---------------------------------------------------------------------------
// Constant → IR op materialization
// ---------------------------------------------------------------------------

/// Record an authoring mistake and return a well-typed stand-in value.
///
/// The recorder keeps going on a poison rather than unwinding, so one
/// `build()` reports every authoring mistake in the pass instead of the first.
/// The stand-in is a real `Const`/`Broadcast` pair of the requested type, so
/// ops downstream of the mistake still type and still get recorded — their
/// diagnostics are about them, not about the recovery. `build()` refuses the
/// trace before any of it reaches the wire.
///
/// Record an authoring mistake and stand in for the value it was supposed to
/// produce, so the recorder keeps going and one `build()` reports every
/// mistake instead of the first.
///
/// The stand-in is fully typed, which is what lets the rest of the trace keep
/// type-checking against it rather than cascading.
///
/// `#[track_caller]` here and on every op that can reach it is load-bearing:
/// the span is the whole value of the report, and without an unbroken chain of
/// it back to the author, the error names a line inside this file.
#[track_caller]
pub(crate) fn poison(detail: alloc::string::String, ty: ValueType) -> Tensor {
    Tensor::node(poison_id(detail, ty), ty)
}

/// [`poison`] as a raw value id, for the recorders that deal in ids.
#[track_caller]
fn poison_id(detail: alloc::string::String, ty: ValueType) -> ValueId {
    context::record_error(detail, Span::here());
    let scalar = emit(
        Op::Const(scalar_literal(ty.dtype, &[0; 8])),
        &[ValueType::scalar(ty.dtype)],
    );
    if ty.shape.is_scalar() {
        return scalar;
    }
    emit(
        Op::Broadcast {
            value: scalar,
            shape: ty.shape,
        },
        &[ty],
    )
}

/// [`poison`] for a value produced outside any traced stage: no op can be
/// emitted there, so the stand-in is a zeroed constant of the same type. The
/// error still lands (via [`PENDING`](crate::context)) in the trace the channel
/// belongs to.
#[track_caller]
pub(crate) fn poison_const(detail: alloc::string::String, ty: ValueType) -> Tensor {
    context::record_error(detail, Span::here());
    let width = if ty.dtype == DType::Bool { 1 } else { 4 };
    Tensor {
        inner: TensorInner::Const(ConstData {
            shape: ty.shape,
            dtype: ty.dtype,
            bytes: alloc::vec![0u8; ty.shape.numel() as usize * width],
        }),
    }
}

/// Record an authoring mistake whose recovery is a shape rather than a value.
#[track_caller]
fn poison_shape(detail: alloc::string::String, fallback: Shape) -> Shape {
    context::record_error(detail, Span::here());
    fallback
}

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
    let word = |o: usize| [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
    match dtype {
        DType::Bool => (bytes.get(i).copied().unwrap_or(0) != 0) as u8 as f64,
        DType::F32 => f32::from_le_bytes(word(i * 4)) as f64,
        DType::I32 => i32::from_le_bytes(word(i * 4)) as f64,
        DType::U32 => u32::from_le_bytes(word(i * 4)) as f64,
    }
}

/// Lower a trace-known constant to IR ops (see [`Tensor::constant`]).
#[track_caller]
fn materialize_const(c: ConstData) -> (ValueId, ValueType) {
    let ty = ValueType::new(c.shape, c.dtype);
    if c.shape.is_scalar() {
        let id = emit(
            Op::Const(scalar_literal(c.dtype, &c.bytes)),
            &[ValueType::scalar(c.dtype)],
        );
        return (id, ty);
    }
    // `usize` is 32 bits in the wasm guest this crate traces in, so a
    // declared shape wider than that would wrap to a short element count and
    // materialize a constant that quietly disagrees with its own type.
    let Ok(n) = usize::try_from(c.shape.numel()) else {
        return (
            poison_id(
                alloc::format!(
                    "constant of shape {:?} has {} elements, more than this target can address",
                    c.shape,
                    c.shape.numel()
                ),
                ty,
            ),
            ty,
        );
    };
    let vals: Vec<f64> = (0..n).map(|i| elem_at(c.dtype, &c.bytes, i)).collect();

    // uniform ⇒ broadcast(scalar).
    if !vals.is_empty() && vals.iter().all(|&v| v == vals[0]) {
        let s = emit(
            Op::Const(scalar_literal(
                c.dtype,
                &c.bytes[..const_elem_size(c.dtype)],
            )),
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
    let poisoned = poison_id(
        alloc::format!(
            "a {:?} constant of shape {:?} is bulk data, and the op set carries constants as \
             scalars: `const` holds one literal, so only a uniform tensor (broadcast) and a u32 \
             affine ramp a+b*i (iota) are reachable from it. Seed a channel with the values and \
             read it in the body — `Channel::from(values)` — or build the tensor from an \
             arithmetic expression",
            c.dtype,
            c.shape
        ),
        ty,
    );
    (poisoned, ty)
}

// ---------------------------------------------------------------------------
// Constant construction (author-facing)
// ---------------------------------------------------------------------------

/// Anything convertible to a trace-known [`ConstData`]: scalars, slices,
/// arrays and vectors of `f32` / `i32` / `u32` / `bool`.
pub trait IntoConst {
    /// Encodes `self` into a typed [`ConstData`].
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
        impl IntoConst for &[$t] {
            fn into_const(self) -> ConstData {
                let mut bytes = Vec::new();
                for &x in self {
                    bytes.extend_from_slice(&scalar_bytes_of(x as f64, $dt));
                }
                ConstData {
                    shape: Shape::vector(self.len() as u32),
                    dtype: $dt,
                    bytes,
                }
            }
        }
        impl<const N: usize> IntoConst for [$t; N] {
            fn into_const(self) -> ConstData {
                self.as_slice().into_const()
            }
        }
        impl IntoConst for Vec<$t> {
            fn into_const(self) -> ConstData {
                self.as_slice().into_const()
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
impl IntoConst for &[bool] {
    fn into_const(self) -> ConstData {
        ConstData {
            shape: Shape::vector(self.len() as u32),
            dtype: DType::Bool,
            bytes: self.iter().map(|&b| b as u8).collect(),
        }
    }
}
impl<const N: usize> IntoConst for [bool; N] {
    fn into_const(self) -> ConstData {
        self.as_slice().into_const()
    }
}
impl IntoConst for Vec<bool> {
    fn into_const(self) -> ConstData {
        self.as_slice().into_const()
    }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// Anything usable as a shape argument: a `[u32; N]` dim array or a [`Shape`].
pub trait IntoShape {
    /// Resolves `self` into a [`Shape`]; an unexpressible rank or a zero
    /// dimension records an authoring error and falls back to a scalar.
    fn into_shape(self) -> Shape;
}
impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        self
    }
}
impl<const N: usize> IntoShape for [u32; N] {
    #[track_caller]
    fn into_shape(self) -> Shape {
        match Shape::new(&self) {
            Some(shape) => shape,
            None => poison_shape(
                alloc::format!(
                    "shape {self:?} is not expressible: rank must be at most {} and no dimension \
                     may be zero",
                    pie_ir::types::MAX_RANK
                ),
                Shape::SCALAR,
            ),
        }
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
// The free-function op surface
// ---------------------------------------------------------------------------

// -- map: unary --
/// `-x` elementwise; numeric dtypes only (not `bool`), shape and dtype preserved.
pub fn neg(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Neg, |t| t)
}
/// `|x|` elementwise.
pub fn abs(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Abs, |t| t)
}
/// `signum(x)` elementwise: `-1`, `0` or `+1`, same dtype as `x`.
pub fn sign(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Sign, |t| t)
}
/// `1 / x` elementwise, F32 only.
pub fn recip(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Recip, |t| t)
}
/// `e^x` elementwise; `F32` only, shape preserved.
pub fn exp(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Exp, |t| t)
}
/// Natural logarithm elementwise; `F32` only, shape preserved.
pub fn log(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Log, |t| t)
}
/// `x` converted elementwise to dtype `to`, shape preserved.
///
/// A cast to the dtype `x` already has is the identity, and returns `x`
/// unchanged rather than emitting an op. Worth doing here rather than asking
/// authors not to write it: a cast is often applied to a value whose dtype
/// depends on which branch produced it, and "cast it and let the trace decide"
/// should not cost a device op when the answer is "nothing to do".
pub fn cast(x: impl AsTensor, to: DType) -> Tensor {
    let x = Tensor::from_arg(x.to_arg());
    if x.dtype() == to {
        return x;
    }
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
/// `a + b` elementwise; operands share one numeric dtype and equal shapes (a
/// scalar broadcasts against a vector), and the result takes that shape and dtype.
pub fn add(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Add, |d| d)
}
/// `a - b` elementwise; same dtype/shape rule as [`add`].
pub fn sub(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Sub, |d| d)
}
/// `a * b` elementwise; same dtype/shape rule as [`add`].
pub fn mul(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Mul, |d| d)
}
/// `a / b` elementwise (`F32` divide, or truncating integer divide that yields
/// `0` on divide-by-zero); same dtype/shape rule as [`add`].
pub fn div(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Div, |d| d)
}
/// `a % b` elementwise; same dtype/shape rule as [`add`].
pub fn rem(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Rem, |d| d)
}
/// Elementwise maximum of `a` and `b`; same dtype/shape rule as [`add`].
pub fn max_elem(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::MaxElem, |d| d)
}
/// Elementwise minimum of `a` and `b`; same dtype/shape rule as [`add`].
pub fn min_elem(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::MinElem, |d| d)
}

// -- map: binary, spelled as Rust operators --
//
// The five arithmetic intrinsics above are also reachable as `+ - * / %` so
// that page arithmetic and score math read as arithmetic. Comparisons are not:
// `PartialOrd` must yield `bool`, so [`lt`] and friends stay functions.

macro_rules! tensor_binop {
    ($($trait:ident, $method:ident, $intrinsic:ident;)*) => {$(
        impl<T: AsTensor> core::ops::$trait<T> for Tensor {
            type Output = Tensor;
            fn $method(self, rhs: T) -> Tensor {
                $intrinsic(self, rhs)
            }
        }
        impl<T: AsTensor> core::ops::$trait<T> for &Tensor {
            type Output = Tensor;
            fn $method(self, rhs: T) -> Tensor {
                $intrinsic(self, rhs)
            }
        }
        tensor_binop!(@scalar $trait, $method, $intrinsic, u32, i32, f32);
    )*};
    (@scalar $trait:ident, $method:ident, $intrinsic:ident, $($t:ty),*) => {$(
        impl core::ops::$trait<Tensor> for $t {
            type Output = Tensor;
            fn $method(self, rhs: Tensor) -> Tensor {
                $intrinsic(self, rhs)
            }
        }
        impl core::ops::$trait<&Tensor> for $t {
            type Output = Tensor;
            fn $method(self, rhs: &Tensor) -> Tensor {
                $intrinsic(self, rhs)
            }
        }
    )*};
}

tensor_binop! {
    Add, add, add;
    Sub, sub, sub;
    Mul, mul, mul;
    Div, div, div;
    Rem, rem, rem;
}

macro_rules! tensor_binop_assign {
    ($($trait:ident, $method:ident, $intrinsic:ident;)*) => {$(
        impl<T: AsTensor> core::ops::$trait<T> for Tensor {
            fn $method(&mut self, rhs: T) {
                *self = $intrinsic(&*self, rhs);
            }
        }
    )*};
}

tensor_binop_assign! {
    AddAssign, add_assign, add;
    SubAssign, sub_assign, sub;
    MulAssign, mul_assign, mul;
    DivAssign, div_assign, div;
    RemAssign, rem_assign, rem;
}

impl core::ops::Neg for Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        neg(self)
    }
}
impl core::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        neg(self)
    }
}

// -- compare / logic (bool results) --
/// `a == b` elementwise → `Bool`; numeric operands sharing a dtype, equal
/// shapes (a scalar broadcasts against a vector).
pub fn eq(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Eq, |_| DType::Bool)
}
/// `a != b` elementwise → `Bool`; same operand rule as [`eq`].
pub fn ne(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Ne, |_| DType::Bool)
}
/// `a < b` elementwise → `Bool`; same operand rule as [`eq`].
pub fn lt(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Lt, |_| DType::Bool)
}
/// `a <= b` elementwise → `Bool`; same operand rule as [`eq`].
pub fn le(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Le, |_| DType::Bool)
}
/// `a > b` elementwise → `Bool`; same operand rule as [`eq`].
pub fn gt(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Gt, |_| DType::Bool)
}
/// `a >= b` elementwise → `Bool`; same operand rule as [`eq`].
pub fn ge(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Ge, |_| DType::Bool)
}
/// `a && b` elementwise; both operands `Bool`, equal shapes (a scalar
/// broadcasts), result `Bool`.
pub fn and(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::And, |_| DType::Bool)
}
/// `a || b` elementwise; same operand rule as [`and`].
pub fn or(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Or, |_| DType::Bool)
}
/// `!x` elementwise; `x` must be `Bool`, shape preserved.
pub fn not(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Not, |t| ValueType::new(t.shape, DType::Bool))
}

// -- choice --
/// `cond ? a : b` elementwise; `cond` is `Bool`, `a` and `b` share a dtype,
/// and the three shapes broadcast together (a scalar broadcasts). Result takes
/// that dtype.
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
/// `x` reinterpreted with `shape`; the element count must be unchanged, dtype
/// preserved.
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
/// `x` broadcast up to `shape`; `x` left-aligns against `shape` (trailing axes
/// padded with `1`) and each of its axes must equal the target or be `1`. Dtype
/// preserved.
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
/// The `U32` vector `[0, 1, …, len-1]`; `len` must be nonzero.
pub fn iota(len: u32) -> Tensor {
    let ty = ValueType::new(Shape::vector(len), DType::U32);
    Tensor::node(emit(Op::Iota { len }, &[ty]), ty)
}
/// The CSR row-offset vector for `rows` runs of equal length `run_len`:
/// `[0, run_len, 2*run_len, …, rows*run_len]`, one entry more than there are
/// rows.
///
/// This is what a descriptor's `*_indptr` port wants, and the shape every
/// decode epilogue rebuilds each fire as its page count grows. Spelled out it
/// was `iota(2) * broadcast(&page_count, [2])` — arithmetic that happens to
/// evaluate to `[0, page_count]` without ever saying so.
pub fn indptr(rows: u32, run_len: impl AsTensor) -> Tensor {
    let n = rows + 1;
    iota(n) * broadcast(run_len, [n])
}

/// Axis-0 generalized gather: `gather(src[n, rest..], idx[S..]) -> [S.., rest..]`.#[track_caller]
pub fn gather(src: impl AsTensor, idx: impl AsTensor) -> Tensor {
    let (is, tys) = src.to_arg().materialize();
    let (ii, tyi) = idx.to_arg().materialize();
    let mut dims: Vec<u32> = tyi.shape.dims().to_vec();
    let src_rest = &tys.shape.dims()[tys.shape.rank().min(1)..];
    dims.extend_from_slice(src_rest);
    let Some(rshape) = Shape::new(&dims) else {
        return poison(
            alloc::format!(
                "gather of {:?} by {:?} has result shape {dims:?}, whose rank exceeds {}",
                tys.shape,
                tyi.shape,
                pie_ir::types::MAX_RANK
            ),
            ValueType::new(tyi.shape, tys.dtype),
        );
    };
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
/// `base` with axis-0 rows `idx` overwritten by `vals`; result keeps `base`'s
/// shape and dtype. `idx` is an integer index and `vals` has shape
/// `idx ++ base[1..]` (or is a scalar) sharing `base`'s dtype.
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
/// Like [`scatter_set`], but adds `vals` into the `idx` rows instead of
/// overwriting; `base` must be numeric.
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
/// Sum over the last axis: `[.., n] → [..]` (a `[n]` vector reduces to a
/// scalar); numeric, dtype preserved.
pub fn reduce_sum(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceSum, |t| {
        ValueType::new(reduce_shape(t.shape), t.dtype)
    })
}
/// Maximum over the last axis: `[.., n] → [..]`; numeric, dtype preserved.
pub fn reduce_max(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceMax, |t| {
        ValueType::new(reduce_shape(t.shape), t.dtype)
    })
}
/// Minimum over the last axis: `[.., n] → [..]`; numeric, dtype preserved.
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
/// Inclusive prefix sum along the last axis; numeric, shape and dtype
/// preserved. Integer lanes scan in their own dtype and wrap on overflow, so
/// a `u32` offset scan stays exact past 2^24.
pub fn cumsum(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::CumSum, |t| t)
}
/// Inclusive prefix product along the last axis; numeric, shape and dtype
/// preserved. Integer lanes wrap on overflow.
pub fn cumprod(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::CumProd, |t| t)
}

// -- normalize (pie_ir::expand sequences, type-tracked) --

/// Records [`pie_ir::expand`] steps into the trace, attaching the result type
/// each step lands in.
///
/// The op order lives in `pie_ir::expand`; this only knows how to name the
/// three shapes an expansion step can have. Adding a step there needs no edit
/// here, which is the point — the two are deliberately separate so a new
/// expansion step is a single-crate change that cannot be half-done.
struct Traced {
    row: ValueType,
    reduced: ValueType,
}

impl Traced {
    fn over(row: ValueType) -> Self {
        Self {
            row,
            reduced: ValueType::new(reduce_shape(row.shape), row.dtype),
        }
    }
}

impl expand::Sink for Traced {
    fn push(&mut self, op: Op, shape: expand::StepShape) -> ValueId {
        let ty = match shape {
            expand::StepShape::Row => self.row,
            expand::StepShape::Reduced => self.reduced,
            // The expansions' only scalars are `Literal::F32` constants, so
            // the declared type follows the literal, not the row.
            expand::StepShape::Scalar => ValueType::scalar(DType::F32),
            expand::StepShape::RowMask => ValueType::new(self.row.shape, DType::Bool),
            expand::StepShape::ReducedIndex => ValueType::new(self.reduced.shape, DType::I32),
        };
        emit(op, &[ty])
    }
}

/// Runs one `pie_ir::expand` sequence over `x` and returns its typed result.
fn expanded(x: impl AsTensor, seq: impl FnOnce(&mut Traced, ValueId, Shape) -> ValueId) -> Tensor {
    let (xid, ty) = x.to_arg().materialize();
    let row = ValueType::new(ty.shape, DType::F32);
    let mut sink = Traced::over(row);
    Tensor::node(seq(&mut sink, xid, ty.shape), row)
}

/// Row-wise softmax over the last axis; result matches `x`'s shape with dtype
/// `F32`.
pub fn softmax(x: impl AsTensor) -> Tensor {
    expanded(x, expand::softmax)
}
/// Row-wise log-softmax over the last axis; result matches `x`'s shape with
/// dtype `F32`.
pub fn log_softmax(x: impl AsTensor) -> Tensor {
    expanded(x, expand::log_softmax)
}
/// Row-wise L2 normalization (`x / sqrt(sum(x²))`) over the last axis; result
/// matches `x`'s shape with dtype `F32`.
pub fn l2norm(x: impl AsTensor) -> Tensor {
    expanded(x, expand::l2norm)
}

// -- order --
/// `top_k(x, k)` → `(values, indices)`. `k` is a trace-known immediate.
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
/// Descending sort of a 1-D row: `(values, original_indices)`.
///
/// Like [`top_k`], this is a library region in the compiler and therefore a
/// hard fusion barrier — prefer a threshold or reduction when one suffices.
pub fn sort_desc(x: impl AsTensor) -> (Tensor, Tensor) {
    let (ix, tyx) = x.to_arg().materialize();
    let n = tyx.shape.dims().last().copied().unwrap_or(0);
    let val_ty = ValueType::vector(n, DType::F32);
    let idx_ty = ValueType::vector(n, DType::U32);
    let base = emit(Op::SortDesc(ix), &[val_ty, idx_ty]);
    (Tensor::node(base, val_ty), Tensor::node(base + 1, idx_ty))
}
/// The top-k rank predicate (`pivot_threshold(x, rank_le(k))`).
pub fn rank_le(k: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::RankLe(k.to_arg()))
}
/// The top-p (nucleus) predicate (`pivot_threshold(probs, cummass_le(p))`):
/// keeps the inclusive nucleus whose descending cumulative probability mass
/// reaches `p`.
pub fn cummass_le(p: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::CummassLe(p.to_arg()))
}
/// The min-p predicate (`pivot_threshold(probs, prob_ge(thr))`): keeps entries
/// whose probability is `>= thr`.
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
/// Matrix product `[m, k] · [k, n] → [m, n]`; both operands `F32`, and the
/// inner dimensions must agree.
pub fn matmul(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    let (ia, tya) = a.to_arg().materialize();
    let (ib, tyb) = b.to_arg().materialize();
    let m = tya.shape.dims().first().copied().unwrap_or(0);
    let n = tyb.shape.dims().last().copied().unwrap_or(0);
    let rty = ValueType::new(Shape::matrix(m, n), DType::F32);
    Tensor::node(emit(Op::MatMul(ia, ib), &[rty]), rty)
}

// -- sampling (the IR's expand: gumbel = RngKeyed; mask_apply = select(mask, x, -inf)) --
/// `gumbel(state, shape)` — Gumbel noise, a pure function of the `[2]` U32 rng
/// `state` (`[key, ctr]`) + element index (replay-deterministic T8).
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
/// expanded to `select(mask, logits, -inf)` (the IR's composed form).
pub fn mask_apply(logits: impl AsTensor, mask: impl AsTensor) -> Tensor {
    let (il, tyl) = logits.to_arg().materialize();
    let (im, _) = mask.to_arg().materialize();
    let mut sink = Traced::over(tyl);
    Tensor::node(expand::mask_apply(&mut sink, il, im), tyl)
}

#[track_caller]
fn append_mask_axis(shape: Shape, len: u32) -> Shape {
    let mut dims = shape.dims().to_vec();
    dims.push(len);
    match Shape::new(&dims) {
        Some(shape) => shape,
        None => poison_shape(
            alloc::format!(
                "a structured mask over {shape:?} with length {len} has shape {dims:?}, whose \
                 rank exceeds {}",
                pie_ir::types::MAX_RANK
            ),
            shape,
        ),
    }
}

/// Boolean causal attention mask over query `positions` (`U32`): appends a
/// length-`len` key axis (`[..] → [.., len]`), `true` where `key <= position`.
pub fn causal_mask(positions: impl AsTensor, len: u32) -> Tensor {
    let (positions, ty) = positions.to_arg().materialize();
    let result = ValueType::new(append_mask_axis(ty.shape, len), DType::Bool);
    Tensor::node(emit(Op::CausalMask { positions, len }, &[result]), result)
}

/// Boolean causal sliding-window mask; like [`causal_mask`], but a key is
/// visible only within `window` positions back (`key <= position && key +
/// window > position`).
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

/// Boolean causal mask keeping a recent `window` plus the first `sink` keys
/// (attention sinks); like [`sliding_window_mask`], but leading `sink`
/// positions stay visible regardless of distance.
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
#[track_caller]
pub fn row_membership(rows: impl AsTensor, keys: impl AsTensor) -> Tensor {
    let rows = rows.to_arg();
    let keys = keys.to_arg();
    let row_type = rows.ty();
    let key_type = keys.ty();
    // The result is [R, K] whatever goes wrong, so every recovery below is a
    // poison of that shape once `row_count`/`key_count` are known, and of the
    // key shape while they are not.
    let [row_count, depth] = *row_type.shape.dims() else {
        return poison(
            alloc::format!(
                "row_membership rows must have shape [R, D], got {:?}",
                row_type.shape
            ),
            key_type,
        );
    };
    let [key_count] = *key_type.shape.dims() else {
        return poison(
            alloc::format!(
                "row_membership keys must have shape [K], got {:?}",
                key_type.shape
            ),
            ValueType::new(Shape::vector(row_count), DType::Bool),
        );
    };
    let result_type = match Shape::new(&[row_count, key_count]) {
        Some(shape) => ValueType::new(shape, DType::Bool),
        None => ValueType::new(Shape::SCALAR, DType::Bool),
    };
    if row_type.dtype != key_type.dtype {
        return poison(
            alloc::format!(
                "row_membership rows and keys must have the same dtype, got {:?} and {:?}",
                row_type.dtype,
                key_type.dtype
            ),
            result_type,
        );
    }

    // The walk below indexes a [R, K, D] cross product flattened to one axis,
    // so the extents multiply. `u32` is the wire's dim type: a product that
    // leaves it would index a tensor the container cannot describe.
    let extents = key_count
        .checked_mul(depth)
        .zip(row_count.checked_mul(depth))
        .zip(
            row_count
                .checked_mul(key_count)
                .and_then(|value| value.checked_mul(depth)),
        );
    let Some(((row_stride, row_flat_len), flat_len)) = extents else {
        return poison(
            alloc::format!(
                "row_membership over {row_count} rows x {key_count} keys x depth {depth} needs a \
                 {}-element intermediate, which overflows the wire's u32 extents",
                u64::from(row_count) * u64::from(key_count) * u64::from(depth)
            ),
            result_type,
        );
    };
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
#[track_caller]
pub fn scalar_gather(src: impl AsTensor, index: impl AsTensor) -> Tensor {
    let (src, src_type) = src.to_arg().materialize();
    let (index, index_type) = index.to_arg().materialize();
    let (op, result_shape) = if let [rows, _] = src_type.shape.dims() {
        if index_type.shape.dims() != [*rows] {
            return poison(
                alloc::format!(
                    "scalar_gather over a {:?} matrix requires one index per row ([{rows}]), got \
                     {:?}",
                    src_type.shape,
                    index_type.shape
                ),
                ValueType::new(Shape::vector(*rows), src_type.dtype),
            );
        }
        (Op::GatherRow { src, idx: index }, Shape::vector(*rows))
    } else {
        let mut dimensions: Vec<u32> = index_type.shape.dims().to_vec();
        dimensions.extend_from_slice(&src_type.shape.dims()[src_type.shape.rank().min(1)..]);
        let Some(shape) = Shape::new(&dimensions) else {
            return poison(
                alloc::format!(
                    "scalar_gather of {:?} by {:?} has result shape {dimensions:?}, whose rank \
                     exceeds {}",
                    src_type.shape,
                    index_type.shape,
                    pie_ir::types::MAX_RANK
                ),
                ValueType::new(index_type.shape, src_type.dtype),
            );
        };
        (Op::Gather { src, idx: index }, shape)
    };
    let result_type = ValueType::new(result_shape, src_type.dtype);
    let result = emit(op, &[result_type]);
    Tensor::node(result, result_type)
}

/// Exact nucleus sampler expressed entirely as ordinary composable SSA.
/// Temperature scaling remains an ordinary preceding operation.
///
/// The op sequence is [`expand::nucleus_sample`], shared with the region
/// matcher that has to recognize it again on the way to a fused kernel.
pub fn nucleus_sample(logits: impl AsTensor, top_p: impl AsTensor, state: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (top_p, _) = top_p.to_arg().materialize();
    let (state, _) = state.to_arg().materialize();
    let mut sink = Traced::over(logits_type);
    let token_type = ValueType::new(reduce_shape(logits_type.shape), DType::I32);
    let result = expand::nucleus_sample(&mut sink, logits, top_p, state, logits_type.shape);
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
