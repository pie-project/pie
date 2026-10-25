//! The region form: what a fused chain computes, a function of one row.
//!
//! Values `0..params.len()` are the parameters, each op's result the next
//! value. Every value is the row's (one element a column, `width` of them)
//! or one a row (a scalar: a mean, an `rsqrt`, a layer's scale). An op over a
//! row and a scalar reads the scalar at every column. Values compute in f32
//! (or are bools); `rounds` names the values rounded where they are defined,
//! each to the narrower dtype the unfused ops store it in, and read on as
//! rounded, so a fusion lands what the ops it replaces land. How a parameter
//! is stored (a row of its own, one every row reads, one element) is the
//! backend's, not the region's.

use std::fmt;

use dtype::Dtype;

/// A value of a region: a parameter, or an op's result.
pub type ValueId = u32;

/// What one value holds: the row's elements, or one value a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    Row,
    Scalar,
}

/// What a value's elements are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Elem {
    F32,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Literal {
    F32(f32),
    Bool(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unary {
    Exp,
    Log,
    Recip,
    Sin,
    Cos,
    Sqrt,
    Rsqrt,
    Neg,
    Abs,
    Sign,
    Not,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Rem,
    Gt,
    Ge,
    Eq,
    Ne,
    Lt,
    Le,
    And,
    Or,
}

impl Binary {
    /// Whether it compares two numbers into a bool.
    #[must_use]
    pub const fn compares(self) -> bool {
        matches!(
            self,
            Binary::Gt | Binary::Ge | Binary::Eq | Binary::Ne | Binary::Lt | Binary::Le
        )
    }

    /// Whether it joins two bools.
    #[must_use]
    pub const fn logic(self) -> bool {
        matches!(self, Binary::And | Binary::Or)
    }
}

/// A reduction over the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reduce {
    Sum,
    Max,
    Min,
}

/// One primitive op of a region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Const(Literal),
    Unary(Unary, ValueId),
    Binary(Binary, ValueId, ValueId),
    /// `cond ? a : b`.
    Select(ValueId, ValueId, ValueId),
    /// The value as `Elem` (a bool as 0 or 1, a number as whether it is
    /// nonzero).
    Cast(ValueId, Elem),
    /// A scalar across the row.
    Broadcast(ValueId),
    /// The row combined into one value.
    Reduce(Reduce, ValueId),
}

impl Op {
    /// The values it reads, in order.
    #[must_use]
    pub fn args(&self) -> Vec<ValueId> {
        match *self {
            Op::Const(_) => Vec::new(),
            Op::Unary(_, a) | Op::Cast(a, _) | Op::Broadcast(a) | Op::Reduce(_, a) => vec![a],
            Op::Binary(_, a, b) => vec![a, b],
            Op::Select(c, a, b) => vec![c, a, b],
        }
    }
}

/// A function of one row: parameters in, primitive ops over them, results
/// out.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// The row's elements.
    pub width: u32,
    /// Values `0..n`: each parameter a row or one value a row, f32.
    pub params: Vec<Shape>,
    /// Their results numbered from `n`, in order.
    pub ops: Vec<Op>,
    /// What leaves the kernel; `results[k]` is written through output `k`.
    pub results: Vec<ValueId>,
    /// The values rounded where they are defined, each to a dtype narrower
    /// than the f32 it is computed in: the rounding points of the program
    /// the region came from. A result rounds by its store, its plane of the
    /// dtype it rounds to.
    pub rounds: Vec<(ValueId, Dtype)>,
}

/// Why a region is not a function of one row.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RegionError {
    /// An op reads a value no earlier op defines.
    Undefined { op: u32, value: ValueId },
    /// An op over values it does not take (a number's op over bools, a
    /// reduction of a scalar, a broadcast of a row).
    Mistyped { op: u32 },
    /// A result that names no value.
    NoSuchResult { result: u32 },
    /// A rounding of no f32 value, or to a dtype that is not a narrower
    /// float.
    Rounding { value: ValueId },
    /// A row of no width.
    Width,
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionError::Undefined { op, value } => {
                write!(
                    f,
                    "op {op} reads value {value}, which nothing before it defines"
                )
            }
            RegionError::Mistyped { op } => write!(f, "op {op} is over values it does not take"),
            RegionError::NoSuchResult { result } => {
                write!(f, "result {result} names a value nothing defines")
            }
            RegionError::Rounding { value } => write!(
                f,
                "value {value} is rounded, and it is no f32 value or rounded to no narrower float"
            ),
            RegionError::Width => f.write_str("the region's row is no element wide"),
        }
    }
}

impl std::error::Error for RegionError {}

impl Region {
    /// The shape and elements of every value, the parameters first, or why
    /// the region is not a function of one row.
    pub fn types(&self) -> Result<Vec<(Shape, Elem)>, RegionError> {
        if self.width == 0 {
            return Err(RegionError::Width);
        }
        let mut types: Vec<(Shape, Elem)> = self
            .params
            .iter()
            .map(|&shape| (shape, Elem::F32))
            .collect();
        for (op, at) in self.ops.iter().zip(0u32..) {
            let get = |v: ValueId| {
                types
                    .get(v as usize)
                    .copied()
                    .ok_or(RegionError::Undefined { op: at, value: v })
            };
            let mistyped = RegionError::Mistyped { op: at };
            let joined = |a: Shape, b: Shape| {
                if a == Shape::Row || b == Shape::Row {
                    Shape::Row
                } else {
                    Shape::Scalar
                }
            };
            let ty = match *op {
                Op::Const(Literal::F32(_)) => (Shape::Scalar, Elem::F32),
                Op::Const(Literal::Bool(_)) => (Shape::Scalar, Elem::Bool),
                Op::Unary(unary, a) => {
                    let (shape, elem) = get(a)?;
                    let wants = if unary == Unary::Not {
                        Elem::Bool
                    } else {
                        Elem::F32
                    };
                    if elem != wants {
                        return Err(mistyped);
                    }
                    (shape, elem)
                }
                Op::Binary(binary, a, b) => {
                    let ((sa, ea), (sb, eb)) = (get(a)?, get(b)?);
                    let wants = if binary.logic() {
                        Elem::Bool
                    } else {
                        Elem::F32
                    };
                    if ea != wants || eb != wants {
                        return Err(mistyped);
                    }
                    let elem = if binary.compares() || binary.logic() {
                        Elem::Bool
                    } else {
                        Elem::F32
                    };
                    (joined(sa, sb), elem)
                }
                Op::Select(c, a, b) => {
                    let ((sc, ec), (sa, ea), (sb, eb)) = (get(c)?, get(a)?, get(b)?);
                    if ec != Elem::Bool || ea != eb {
                        return Err(mistyped);
                    }
                    (joined(sc, joined(sa, sb)), ea)
                }
                Op::Cast(a, elem) => (get(a)?.0, elem),
                Op::Broadcast(a) => {
                    let (shape, elem) = get(a)?;
                    if shape != Shape::Scalar {
                        return Err(mistyped);
                    }
                    (Shape::Row, elem)
                }
                Op::Reduce(_, a) => {
                    let (shape, elem) = get(a)?;
                    if shape != Shape::Row || elem != Elem::F32 {
                        return Err(mistyped);
                    }
                    (Shape::Scalar, Elem::F32)
                }
            };
            types.push(ty);
        }
        for (result, &value) in (0u32..).zip(&self.results) {
            if value as usize >= types.len() {
                return Err(RegionError::NoSuchResult { result });
            }
        }
        for &(value, dtype) in &self.rounds {
            let f32_value = types
                .get(value as usize)
                .is_some_and(|ty| ty.1 == Elem::F32);
            if !f32_value || !matches!(dtype, Dtype::Bf16 | Dtype::F16) {
                return Err(RegionError::Rounding { value });
            }
        }
        Ok(types)
    }

    /// The region's key: a hash of what it is (its width, parameters, ops,
    /// results and roundings), not of any text a backend writes for it.
    /// Two regions of one key are the same function.
    #[must_use]
    pub fn key(&self) -> u64 {
        crate::fnv1a64(format!("{self:?}").as_bytes())
    }

    /// The dtype `value` rounds to where it is defined, if it rounds.
    #[must_use]
    pub fn rounding(&self, value: ValueId) -> Option<Dtype> {
        self.rounds
            .iter()
            .find(|(v, _)| *v == value)
            .map(|(_, dtype)| *dtype)
    }
}
