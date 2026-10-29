use alloc::vec::Vec;

use eta_ir::container::const_elem_size;
use eta_ir::expand;
use eta_ir::op::Op;
use eta_ir::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId, ValueType};

use crate::context::{self, emit};
use crate::error::Span;

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
    pub fn dtype(&self) -> Dtype {
        self.ty().dtype
    }
    pub fn shape(&self) -> Shape {
        self.ty().shape
    }

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

fn const_scalar(a: &Arg) -> Option<f64> {
    let Arg::Const(c) = a else { return None };
    if !c.shape.is_scalar() {
        return None;
    }
    let b = c.bytes.as_slice();
    Some(match c.dtype {
        Dtype::Bool => (b.first().copied().unwrap_or(0) != 0) as u8 as f64,
        Dtype::F32 => f32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
        Dtype::I32 => i32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
        Dtype::U32 => u32::from_le_bytes(b.get(..4)?.try_into().ok()?) as f64,
        _ => return None,
    })
}

fn scalar_arg(v: f64, dtype: Dtype) -> Arg {
    Arg::Const(ConstData {
        shape: Shape::SCALAR,
        dtype,
        bytes: scalar_bytes_of(v, dtype),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConstData {
    pub shape: Shape,
    pub dtype: Dtype,
    pub bytes: Vec<u8>,
}

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
    pub(crate) fn materialize(self) -> (ValueId, ValueType) {
        match self {
            Arg::Node { id, ty } => (id, ty),
            Arg::Const(c) => materialize_const(c),
        }
    }
}

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

#[track_caller]
pub(crate) fn poison(detail: alloc::string::String, ty: ValueType) -> Tensor {
    Tensor::node(poison_id(detail, ty), ty)
}

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

#[track_caller]
pub(crate) fn poison_const(detail: alloc::string::String, ty: ValueType) -> Tensor {
    context::record_error(detail, Span::here());
    let width = if ty.dtype == Dtype::Bool { 1 } else { 4 };
    Tensor {
        inner: TensorInner::Const(ConstData {
            shape: ty.shape,
            dtype: ty.dtype,
            bytes: alloc::vec![0u8; ty.shape.numel() as usize * width],
        }),
    }
}

#[track_caller]
fn poison_shape(detail: alloc::string::String, fallback: Shape) -> Shape {
    context::record_error(detail, Span::here());
    fallback
}

#[track_caller]
fn poison_dtype(dtype: Dtype) {
    context::record_error(
        alloc::format!("{dtype:?} is not a dtype ETA computes in"),
        Span::here(),
    );
}

fn scalar_literal(dtype: Dtype, bytes: &[u8]) -> Literal {
    let w = |i: usize| bytes.get(i).copied().unwrap_or(0);
    let word = [w(0), w(1), w(2), w(3)];
    match dtype {
        Dtype::F32 => Literal::F32(f32::from_le_bytes(word)),
        Dtype::I32 => Literal::I32(i32::from_le_bytes(word)),
        Dtype::U32 => Literal::U32(u32::from_le_bytes(word)),
        Dtype::Bool => Literal::Bool(w(0) != 0),
        other => {
            poison_dtype(other);
            Literal::F32(f32::from_le_bytes(word))
        }
    }
}

fn elem_at(dtype: Dtype, bytes: &[u8], i: usize) -> f64 {
    let word = |o: usize| [bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]];
    match dtype {
        Dtype::Bool => (bytes.get(i).copied().unwrap_or(0) != 0) as u8 as f64,
        Dtype::F32 => f32::from_le_bytes(word(i * 4)) as f64,
        Dtype::I32 => i32::from_le_bytes(word(i * 4)) as f64,
        Dtype::U32 => u32::from_le_bytes(word(i * 4)) as f64,
        other => {
            poison_dtype(other);
            f32::from_le_bytes(word(i * 4)) as f64
        }
    }
}

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
    if c.dtype == Dtype::U32 && n >= 2 {
        let a = vals[0];
        let b = vals[1] - vals[0];
        if b >= 0.0 && vals.iter().enumerate().all(|(i, &v)| v == a + b * i as f64) {
            let io = emit(Op::Iota { len: n as u32 }, &[ty]);
            let mut cur = io;
            if b != 1.0 {
                let bc = emit(
                    Op::Const(Literal::U32(b as u32)),
                    &[ValueType::scalar(Dtype::U32)],
                );
                cur = emit(Op::Mul(cur, bc), &[ty]);
            }
            if a != 0.0 {
                let ac = emit(
                    Op::Const(Literal::U32(a as u32)),
                    &[ValueType::scalar(Dtype::U32)],
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

pub trait IntoConst {
    fn into_const(self) -> ConstData;
}

fn scalar_bytes_of(v: f64, dt: Dtype) -> Vec<u8> {
    match dt {
        Dtype::F32 => (v as f32).to_le_bytes().to_vec(),
        Dtype::I32 => (v as i32).to_le_bytes().to_vec(),
        Dtype::U32 => (v as u32).to_le_bytes().to_vec(),
        Dtype::Bool => alloc::vec![(v != 0.0) as u8],
        other => {
            poison_dtype(other);
            (v as f32).to_le_bytes().to_vec()
        }
    }
}

macro_rules! num_const {
    ($t:ty, $dt:expr) => {
        impl IntoConst for $t {
            fn into_const(self) -> ConstData {
                ConstData {
                    shape: Shape::SCALAR,
                    dtype: $dt,
                    bytes: self.to_le_bytes().to_vec(),
                }
            }
        }
        impl IntoConst for &[$t] {
            fn into_const(self) -> ConstData {
                let mut bytes = Vec::with_capacity(self.len() * core::mem::size_of::<$t>());
                for &x in self {
                    bytes.extend_from_slice(&x.to_le_bytes());
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
num_const!(f32, Dtype::F32);
num_const!(i32, Dtype::I32);
num_const!(u32, Dtype::U32);

impl IntoConst for bool {
    fn into_const(self) -> ConstData {
        ConstData {
            shape: Shape::SCALAR,
            dtype: Dtype::Bool,
            bytes: alloc::vec![self as u8],
        }
    }
}
impl IntoConst for &[bool] {
    fn into_const(self) -> ConstData {
        ConstData {
            shape: Shape::vector(self.len() as u32),
            dtype: Dtype::Bool,
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

pub trait IntoShape {
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
                    eta_ir::types::MAX_RANK
                ),
                Shape::SCALAR,
            ),
        }
    }
}

fn non_scalar_shape(a: Shape, b: Shape) -> Shape {
    if a.is_scalar() { b } else { a }
}

fn reconcile(a: Arg, b: Arg) -> (Arg, Arg) {
    fn coerce(c: &ConstData, to: Dtype) -> Option<ConstData> {
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
    result_dtype: impl FnOnce(Dtype) -> Dtype,
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

fn push(op: Op, tys: &[ValueType]) -> ValueId {
    emit(op, tys)
}

pub fn neg(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Neg, |t| t)
}
pub fn abs(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Abs, |t| t)
}
pub fn sign(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Sign, |t| t)
}
pub fn recip(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Recip, |t| t)
}
pub fn exp(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Exp, |t| t)
}
pub fn log(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Log, |t| t)
}
pub fn sin(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Sin, |t| t)
}
pub fn cos(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Cos, |t| t)
}
pub fn sqrt(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Sqrt, |t| t)
}
pub fn rsqrt(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Rsqrt, |t| t)
}
pub fn cast(x: impl AsTensor, to: Dtype) -> Tensor {
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

pub fn eq(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Eq, |_| Dtype::Bool)
}
pub fn ne(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Ne, |_| Dtype::Bool)
}
pub fn lt(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Lt, |_| Dtype::Bool)
}
pub fn le(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Le, |_| Dtype::Bool)
}
pub fn gt(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Gt, |_| Dtype::Bool)
}
pub fn ge(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Ge, |_| Dtype::Bool)
}
pub fn and(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::And, |_| Dtype::Bool)
}
pub fn or(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    emit_binary(&a, &b, Op::Or, |_| Dtype::Bool)
}
pub fn not(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::Not, |t| ValueType::new(t.shape, Dtype::Bool))
}

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

pub fn iota(len: u32) -> Tensor {
    let ty = ValueType::new(Shape::vector(len), Dtype::U32);
    Tensor::node(emit(Op::Iota { len }, &[ty]), ty)
}
pub fn indptr(rows: u32, run_len: impl AsTensor) -> Tensor {
    let n = rows + 1;
    iota(n) * broadcast(run_len, [n])
}

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
                eta_ir::types::MAX_RANK
            ),
            ValueType::new(tyi.shape, tys.dtype),
        );
    };
    let rty = ValueType::new(rshape, tys.dtype);
    Tensor::node(emit(Op::Gather { src: is, idx: ii }, &[rty]), rty)
}
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
pub fn reduce_argmax(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::ReduceArgmax, |t| {
        ValueType::new(reduce_shape(t.shape), Dtype::I32)
    })
}
pub fn cumsum(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::CumSum, |t| t)
}
pub fn cumprod(x: impl AsTensor) -> Tensor {
    emit_unary(&x, Op::CumProd, |t| t)
}

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
            expand::StepShape::Scalar => ValueType::scalar(Dtype::F32),
            expand::StepShape::RowMask => ValueType::new(self.row.shape, Dtype::Bool),
            expand::StepShape::ReducedIndex => ValueType::new(self.reduced.shape, Dtype::I32),
        };
        emit(op, &[ty])
    }
}

fn expanded(x: impl AsTensor, seq: impl FnOnce(&mut Traced, ValueId, Shape) -> ValueId) -> Tensor {
    let (xid, ty) = x.to_arg().materialize();
    let row = ValueType::new(ty.shape, Dtype::F32);
    let mut sink = Traced::over(row);
    Tensor::node(seq(&mut sink, xid, ty.shape), row)
}

pub fn softmax(x: impl AsTensor) -> Tensor {
    expanded(x, expand::softmax)
}
pub fn log_softmax(x: impl AsTensor) -> Tensor {
    expanded(x, expand::log_softmax)
}
pub fn l2norm(x: impl AsTensor) -> Tensor {
    expanded(x, expand::l2norm)
}

pub fn top_k(x: impl AsTensor, k: u32) -> (Tensor, Tensor) {
    let (ix, tyx) = x.to_arg().materialize();
    let mut dims: Vec<u32> = tyx.shape.dims().to_vec();
    if let Some(last) = dims.last_mut() {
        *last = k;
    }
    let out_shape = Shape::new(&dims).unwrap_or_else(|| Shape::vector(k));
    let val_ty = ValueType::new(out_shape, tyx.dtype);
    let idx_ty = ValueType::new(out_shape, Dtype::U32);
    let base = emit(Op::TopK { input: ix, k }, &[val_ty, idx_ty]);
    (Tensor::node(base, val_ty), Tensor::node(base + 1, idx_ty))
}
pub fn sort_desc(x: impl AsTensor) -> (Tensor, Tensor) {
    let (ix, tyx) = x.to_arg().materialize();
    let n = tyx.shape.dims().last().copied().unwrap_or(0);
    let val_ty = ValueType::vector(n, Dtype::F32);
    let idx_ty = ValueType::vector(n, Dtype::U32);
    let base = emit(Op::SortDesc(ix), &[val_ty, idx_ty]);
    (Tensor::node(base, val_ty), Tensor::node(base + 1, idx_ty))
}
pub fn rank_le(k: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::RankLe(k.to_arg()))
}
pub fn cummass_le(p: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::CummassLe(p.to_arg()))
}
pub fn prob_ge(thr: impl AsTensor) -> PredicateArg {
    PredicateArg(PredKind::ProbGe(thr.to_arg()))
}
pub struct PredicateArg(PredKind);
enum PredKind {
    RankLe(Arg),
    CummassLe(Arg),
    ProbGe(Arg),
}
pub fn pivot_threshold(input: impl AsTensor, predicate: PredicateArg) -> Tensor {
    let (ii, tyi) = input.to_arg().materialize();
    let pred = match predicate.0 {
        PredKind::RankLe(a) => Predicate::RankLe(a.materialize().0),
        PredKind::CummassLe(a) => Predicate::CummassLe(a.materialize().0),
        PredKind::ProbGe(a) => Predicate::ProbGe(a.materialize().0),
    };
    let rty = ValueType::new(tyi.shape, Dtype::Bool);
    let id = emit(
        Op::PivotThreshold {
            input: ii,
            predicate: pred,
        },
        &[rty],
    );
    Tensor::node(id, rty)
}

pub fn matmul(a: impl AsTensor, b: impl AsTensor) -> Tensor {
    let (ia, tya) = a.to_arg().materialize();
    let (ib, tyb) = b.to_arg().materialize();
    let m = tya.shape.dims().first().copied().unwrap_or(0);
    let n = tyb.shape.dims().last().copied().unwrap_or(0);
    let rty = ValueType::new(Shape::matrix(m, n), Dtype::F32);
    Tensor::node(emit(Op::MatMul(ia, ib), &[rty]), rty)
}

pub fn gumbel(state: impl AsTensor, shape: impl IntoShape) -> Tensor {
    rng_noise(state, shape, RngKind::Gumbel)
}
pub fn rng(state: impl AsTensor, shape: impl IntoShape) -> Tensor {
    rng_noise(state, shape, RngKind::Uniform)
}
pub fn normal(state: impl AsTensor, shape: impl IntoShape) -> Tensor {
    rng_noise(state, shape, RngKind::Normal)
}
fn rng_noise(state: impl AsTensor, shape: impl IntoShape, kind: RngKind) -> Tensor {
    let s = shape.into_shape();
    let (istate, _) = state.to_arg().materialize();
    let rty = ValueType::new(s, Dtype::F32);
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
                eta_ir::types::MAX_RANK
            ),
            shape,
        ),
    }
}

pub fn causal_mask(positions: impl AsTensor, len: u32) -> Tensor {
    let (positions, ty) = positions.to_arg().materialize();
    let result = ValueType::new(append_mask_axis(ty.shape, len), Dtype::Bool);
    Tensor::node(emit(Op::CausalMask { positions, len }, &[result]), result)
}

pub fn sliding_window_mask(positions: impl AsTensor, len: u32, window: u32) -> Tensor {
    let (positions, ty) = positions.to_arg().materialize();
    let result = ValueType::new(append_mask_axis(ty.shape, len), Dtype::Bool);
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
    let result = ValueType::new(append_mask_axis(ty.shape, len), Dtype::Bool);
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

#[track_caller]
pub fn row_membership(rows: impl AsTensor, keys: impl AsTensor) -> Tensor {
    let rows = rows.to_arg();
    let keys = keys.to_arg();
    let row_type = rows.ty();
    let key_type = keys.ty();
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
            ValueType::new(Shape::vector(row_count), Dtype::Bool),
        );
    };
    let result_type = match Shape::new(&[row_count, key_count]) {
        Some(shape) => ValueType::new(shape, Dtype::Bool),
        None => ValueType::new(Shape::SCALAR, Dtype::Bool),
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
    cast(reduce_max(cast(matches, Dtype::U32)), Dtype::Bool)
}

pub fn masked_argmax(logits: impl AsTensor, mask: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (mask, _) = mask.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(logits_type.shape), Dtype::I32);
    let negative_infinity = push(
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        &[ValueType::scalar(Dtype::F32)],
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

pub fn gumbel_max(logits: impl AsTensor, state: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (state, _) = state.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(logits_type.shape), Dtype::I32);
    let noise = push(
        Op::RngKeyed {
            state,
            shape: logits_type.shape,
            kind: RngKind::Gumbel,
        },
        &[ValueType::new(logits_type.shape, Dtype::F32)],
    );
    let perturbed = push(Op::Add(logits, noise), &[logits_type]);
    let result = push(Op::ReduceArgmax(perturbed), &[result_type]);
    Tensor::node(result, result_type)
}

pub fn entropy(probabilities: impl AsTensor) -> Tensor {
    let (probabilities, probability_type) = probabilities.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(probability_type.shape), Dtype::F32);
    let floored = max_elem(
        Tensor::node(probabilities, probability_type),
        f32::MIN_POSITIVE,
    );
    let (floored, _) = floored.to_arg().materialize();
    let log_probabilities = push(Op::Log(floored), &[probability_type]);
    let terms = push(
        Op::Mul(probabilities, log_probabilities),
        &[probability_type],
    );
    let sum = push(Op::ReduceSum(terms), &[result_type]);
    let result = push(Op::Neg(sum), &[result_type]);
    Tensor::node(result, result_type)
}

pub fn entropy_from_logprobs(
    probabilities: impl AsTensor,
    log_probabilities: impl AsTensor,
) -> Tensor {
    let (probabilities, probability_type) = probabilities.to_arg().materialize();
    let (log_probabilities, _) = log_probabilities.to_arg().materialize();
    let result_type = ValueType::new(reduce_shape(probability_type.shape), Dtype::F32);
    let terms = push(
        Op::Mul(probabilities, log_probabilities),
        &[probability_type],
    );
    let sum = push(Op::ReduceSum(terms), &[result_type]);
    let result = push(Op::Neg(sum), &[result_type]);
    Tensor::node(result, result_type)
}

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
                    eta_ir::types::MAX_RANK
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

pub fn nucleus_sample(logits: impl AsTensor, top_p: impl AsTensor, state: impl AsTensor) -> Tensor {
    let (logits, logits_type) = logits.to_arg().materialize();
    let (top_p, _) = top_p.to_arg().materialize();
    let (state, _) = state.to_arg().materialize();
    let mut sink = Traced::over(logits_type);
    let token_type = ValueType::new(reduce_shape(logits_type.shape), Dtype::I32);
    let result = expand::nucleus_sample(&mut sink, logits, top_p, state, logits_type.shape);
    Tensor::node(result, token_type)
}

pub(crate) fn output_val(index: u16, lane: u32, shape: Shape, dtype: Dtype) -> Tensor {
    let ty = ValueType::new(shape, dtype);
    Tensor::node(
        emit(
            Op::Output {
                index,
                lane,
                shape,
                dtype,
            },
            &[ty],
        ),
        ty,
    )
}

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
