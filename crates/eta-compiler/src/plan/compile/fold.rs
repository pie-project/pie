//! Constant folding, algebraic simplification and CSE keying. Conservative:
//! every function returns `None`/`false` when it cannot prove a rewrite is
//! sound.
//!
//! `Dtype::F32` is excluded from the algebra: IEEE-754 floats are not the
//! ring the identities are written for (`x + 0.0` does not preserve `-0.0`;
//! `MaxElem`/`MinElem` propagate operand order under `NaN`, so swapping
//! operands is observable). A rewrite that moves a sign of zero or flips
//! which `NaN` survives would change bits a sampler reads, so float cases
//! are simply not rewritten; integer and boolean cases are.

use alloc::vec::Vec;

use eta_ir::container::{encode_op, put_u32};
use eta_ir::op::Op;
use eta_ir::types::{Dtype, Literal};

use super::canonical::canonical_symbolic_type;
use super::symbolic::{Dimension, SymbolicType};

pub(crate) fn simplify_alias(
    op: &Op,
    result_type: &SymbolicType,
    literals: &[Option<Literal>],
) -> Option<u32> {
    if result_type
        .dims
        .iter()
        .all(|dim| *dim == Dimension::Static(1))
    {
        match *op {
            Op::CumSum(value) | Op::CumProd(value) => return Some(value),
            _ => {}
        }
    }
    // not an identity over IEEE-754 (see module docs).
    if result_type.dtype == Dtype::F32 {
        return None;
    }
    let literal = |value: u32| literals.get(value as usize).copied().flatten();
    match *op {
        Op::Add(a, b) if is_zero(literal(b)) => Some(a),
        Op::Add(a, b) if is_zero(literal(a)) => Some(b),
        Op::Sub(a, b) if is_zero(literal(b)) => Some(a),
        Op::Mul(a, b) if is_one(literal(b)) => Some(a),
        Op::Mul(a, b) if is_one(literal(a)) => Some(b),
        Op::Div(a, b) if is_one(literal(b)) => Some(a),
        Op::And(a, b) if literal(b) == Some(Literal::Bool(true)) => Some(a),
        Op::And(a, b) if literal(a) == Some(Literal::Bool(true)) => Some(b),
        Op::Or(a, b) if literal(b) == Some(Literal::Bool(false)) => Some(a),
        Op::Or(a, b) if literal(a) == Some(Literal::Bool(false)) => Some(b),
        _ => None,
    }
}

pub(crate) fn is_zero(literal: Option<Literal>) -> bool {
    matches!(literal, Some(Literal::I32(0) | Literal::U32(0)))
}

pub(crate) fn is_one(literal: Option<Literal>) -> bool {
    matches!(literal, Some(Literal::I32(1) | Literal::U32(1)))
}

pub(crate) fn fold_scalar(op: &Op, literals: &[Option<Literal>]) -> Option<Literal> {
    let get = |value: u32| literals.get(value as usize).copied().flatten();
    match *op {
        Op::Neg(value) => match get(value)? {
            Literal::I32(value) => Some(Literal::I32(value.wrapping_neg())),
            Literal::U32(value) => Some(Literal::U32(value.wrapping_neg())),
            _ => None,
        },
        Op::Abs(value) => match get(value)? {
            Literal::I32(value) => Some(Literal::I32(value.wrapping_abs())),
            Literal::U32(value) => Some(Literal::U32(value)),
            _ => None,
        },
        Op::Sign(value) => match get(value)? {
            Literal::I32(value) => Some(Literal::I32(value.signum())),
            Literal::U32(value) => Some(Literal::U32(u32::from(value != 0))),
            _ => None,
        },
        Op::Not(value) => match get(value)? {
            Literal::Bool(value) => Some(Literal::Bool(!value)),
            _ => None,
        },
        Op::Add(a, b) => fold_int_binary(get(a)?, get(b)?, i32::wrapping_add, u32::wrapping_add),
        Op::Sub(a, b) => fold_int_binary(get(a)?, get(b)?, i32::wrapping_sub, u32::wrapping_sub),
        Op::Mul(a, b) => fold_int_binary(get(a)?, get(b)?, i32::wrapping_mul, u32::wrapping_mul),
        Op::Div(a, b) => match (get(a)?, get(b)?) {
            (Literal::I32(a), Literal::I32(b)) => {
                Some(Literal::I32(if b == 0 { 0 } else { a.wrapping_div(b) }))
            }
            (Literal::U32(a), Literal::U32(b)) => Some(Literal::U32(a.checked_div(b).unwrap_or(0))),
            _ => None,
        },
        Op::Rem(a, b) => match (get(a)?, get(b)?) {
            (Literal::I32(a), Literal::I32(b)) => {
                Some(Literal::I32(if b == 0 { 0 } else { a.wrapping_rem(b) }))
            }
            (Literal::U32(a), Literal::U32(b)) => {
                Some(Literal::U32(if b == 0 { 0 } else { a % b }))
            }
            _ => None,
        },
        Op::MaxElem(a, b) => fold_ordered(get(a)?, get(b)?, true),
        Op::MinElem(a, b) => fold_ordered(get(a)?, get(b)?, false),
        Op::Eq(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering == 0),
        Op::Ne(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering != 0),
        Op::Lt(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering < 0),
        Op::Le(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering <= 0),
        Op::Gt(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering > 0),
        Op::Ge(a, b) => fold_compare(get(a)?, get(b)?, |ordering| ordering >= 0),
        Op::And(a, b) => match (get(a)?, get(b)?) {
            (Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(a && b)),
            _ => None,
        },
        Op::Or(a, b) => match (get(a)?, get(b)?) {
            (Literal::Bool(a), Literal::Bool(b)) => Some(Literal::Bool(a || b)),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn fold_int_binary(
    a: Literal,
    b: Literal,
    signed: fn(i32, i32) -> i32,
    unsigned: fn(u32, u32) -> u32,
) -> Option<Literal> {
    match (a, b) {
        (Literal::I32(a), Literal::I32(b)) => Some(Literal::I32(signed(a, b))),
        (Literal::U32(a), Literal::U32(b)) => Some(Literal::U32(unsigned(a, b))),
        _ => None,
    }
}

pub(crate) fn fold_ordered(a: Literal, b: Literal, maximum: bool) -> Option<Literal> {
    match (a, b) {
        (Literal::I32(a), Literal::I32(b)) => {
            Some(Literal::I32(if maximum { a.max(b) } else { a.min(b) }))
        }
        (Literal::U32(a), Literal::U32(b)) => {
            Some(Literal::U32(if maximum { a.max(b) } else { a.min(b) }))
        }
        _ => None,
    }
}

pub(crate) fn fold_compare(
    a: Literal,
    b: Literal,
    predicate: impl FnOnce(i8) -> bool,
) -> Option<Literal> {
    let ordering = match (a, b) {
        (Literal::I32(a), Literal::I32(b)) => a.cmp(&b),
        (Literal::U32(a), Literal::U32(b)) => a.cmp(&b),
        (Literal::Bool(a), Literal::Bool(b)) => a.cmp(&b),
        _ => return None,
    };
    let ordering = match ordering {
        core::cmp::Ordering::Less => -1,
        core::cmp::Ordering::Equal => 0,
        core::cmp::Ordering::Greater => 1,
    };
    Some(Literal::Bool(predicate(ordering)))
}

/// Put commutative operands in a canonical order so that `a + b` and `b + a`
/// share a CSE key and a stage signature.
///
/// Skipped for F32: the swap is observable under `NaN` (see module docs).
pub(crate) fn canonicalize_commutative(op: &mut Op, result_type: Option<&SymbolicType>) {
    if result_type.is_some_and(|result_type| result_type.dtype == Dtype::F32) {
        return;
    }
    match op {
        Op::Add(a, b)
        | Op::Mul(a, b)
        | Op::MaxElem(a, b)
        | Op::MinElem(a, b)
        | Op::Eq(a, b)
        | Op::Ne(a, b)
        | Op::And(a, b)
        | Op::Or(a, b)
            if *a > *b =>
        {
            core::mem::swap(a, b);
        }
        _ => {}
    }
}

/// CSE may merge two identical ops only when neither has an effect, since
/// merging removes one of them. [`Op::is_effectful`] owns that list.
pub(crate) fn cse_candidate(op: &Op) -> bool {
    !op.is_effectful()
}

pub(crate) fn cse_key(op: &Op, result_types: &[SymbolicType]) -> Vec<u8> {
    let mut bytes = Vec::new();
    encode_op(&mut bytes, op);
    put_u32(&mut bytes, result_types.len() as u32);
    for value_type in result_types {
        canonical_symbolic_type(&mut bytes, value_type);
    }
    bytes
}
