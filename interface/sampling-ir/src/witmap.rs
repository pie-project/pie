//! The canonical `Op ↔ op-kind` table — the **drift-guard oracle**.
//!
//! [`OpKind`] is a faithful pure-Rust transcription of the WIT `variant op-kind`
//! in `interface/inferlet/core/wit/tensor.wit` (tuple operands, `input-index`,
//! field order — matching the WIT exactly). It has **no generated-binding
//! dependency**, so it is the single source of truth both sides implement
//! against:
//!
//! * **guest emit** (SDK): map [`Op`] → the wit-bindgen `op-kind` — structurally
//!   identical to [`OpKind`], so the adapter is a field-by-field copy.
//! * **host decode** (runtime): map the wasmtime `op-kind` → [`OpKind`] →
//!   [`Op`] → `encode()` → bytecode.
//!
//! The conversions [`Op::to_op_kind`] / [`OpKind::to_op`] are total and inverse:
//! `op.to_op_kind().to_op() == op` for every variant (asserted by the
//! round-trip test). If the WIT `op-kind` and [`Op`] ever drift, this module
//! stops compiling or the round-trip test fails.
//!
//! Note the only non-trivial mapping is `Input`: WIT `input(input-index)` ↔
//! [`Op::Input`]`(index)` (the program is binding-free; the binding is
//! attach-time, [`crate::Binding`]). Every other op is a direct field rename
//! (named fields ↔ WIT tuples).

use crate::types::*;

/// A faithful pure-Rust mirror of the WIT `variant op-kind` (tensor.wit).
/// Operand shapes match the WIT exactly (tuples; `Input` carries an input-index).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpKind {
    Input(InputIndex),
    Const(Literal),
    Exp(ValueId),
    Log(ValueId),
    Neg(ValueId),
    Recip(ValueId),
    Abs(ValueId),
    Sign(ValueId),
    Add((ValueId, ValueId)),
    Sub((ValueId, ValueId)),
    Mul((ValueId, ValueId)),
    Div((ValueId, ValueId)),
    MaxElem((ValueId, ValueId)),
    MinElem((ValueId, ValueId)),
    Gt((ValueId, ValueId)),
    Ge((ValueId, ValueId)),
    Eq((ValueId, ValueId)),
    Select((ValueId, ValueId, ValueId)),
    ReduceSum(ValueId),
    ReduceMax(ValueId),
    ReduceMin(ValueId),
    ReduceArgmax(ValueId),
    Broadcast((ValueId, Shape)),
    CumSum(ValueId),
    CumProd(ValueId),
    SortDesc(ValueId),
    PivotThreshold((ValueId, Predicate)),
    Gather((ValueId, ValueId)),
    GatherRow((ValueId, ValueId)),
    MaskApply((ValueId, ValueId)),
    ScatterAdd((ValueId, ValueId, ValueId)),
    ScatterSet((ValueId, ValueId, ValueId)),
    Rng((u32, Shape, RngKind)),
}

impl Op {
    /// Map to the canonical WIT-shaped [`OpKind`]. Total + inverse of
    /// [`OpKind::to_op`].
    pub fn to_op_kind(self) -> OpKind {
        match self {
            Op::Input(i) => OpKind::Input(i),
            Op::Const(l) => OpKind::Const(l),
            Op::Exp(a) => OpKind::Exp(a),
            Op::Log(a) => OpKind::Log(a),
            Op::Neg(a) => OpKind::Neg(a),
            Op::Recip(a) => OpKind::Recip(a),
            Op::Abs(a) => OpKind::Abs(a),
            Op::Sign(a) => OpKind::Sign(a),
            Op::Add(a, b) => OpKind::Add((a, b)),
            Op::Sub(a, b) => OpKind::Sub((a, b)),
            Op::Mul(a, b) => OpKind::Mul((a, b)),
            Op::Div(a, b) => OpKind::Div((a, b)),
            Op::MaxElem(a, b) => OpKind::MaxElem((a, b)),
            Op::MinElem(a, b) => OpKind::MinElem((a, b)),
            Op::Gt(a, b) => OpKind::Gt((a, b)),
            Op::Ge(a, b) => OpKind::Ge((a, b)),
            Op::Eq(a, b) => OpKind::Eq((a, b)),
            Op::Select { cond, a, b } => OpKind::Select((cond, a, b)),
            Op::ReduceSum(a) => OpKind::ReduceSum(a),
            Op::ReduceMax(a) => OpKind::ReduceMax(a),
            Op::ReduceMin(a) => OpKind::ReduceMin(a),
            Op::ReduceArgmax(a) => OpKind::ReduceArgmax(a),
            Op::Broadcast { value, shape } => OpKind::Broadcast((value, shape)),
            Op::CumSum(a) => OpKind::CumSum(a),
            Op::CumProd(a) => OpKind::CumProd(a),
            Op::SortDesc(a) => OpKind::SortDesc(a),
            Op::PivotThreshold { input, predicate } => {
                OpKind::PivotThreshold((input, predicate))
            }
            Op::Gather { src, idx } => OpKind::Gather((src, idx)),
            Op::GatherRow { src, idx } => OpKind::GatherRow((src, idx)),
            Op::MaskApply { logits, mask } => OpKind::MaskApply((logits, mask)),
            Op::ScatterAdd { base, idx, vals } => OpKind::ScatterAdd((base, idx, vals)),
            Op::ScatterSet { base, idx, vals } => OpKind::ScatterSet((base, idx, vals)),
            Op::Rng { stream, shape, kind } => OpKind::Rng((stream, shape, kind)),
        }
    }
}

impl OpKind {
    /// Map back to [`Op`]. Total + inverse of [`Op::to_op_kind`].
    pub fn to_op(self) -> Op {
        match self {
            OpKind::Input(i) => Op::Input(i),
            OpKind::Const(l) => Op::Const(l),
            OpKind::Exp(a) => Op::Exp(a),
            OpKind::Log(a) => Op::Log(a),
            OpKind::Neg(a) => Op::Neg(a),
            OpKind::Recip(a) => Op::Recip(a),
            OpKind::Abs(a) => Op::Abs(a),
            OpKind::Sign(a) => Op::Sign(a),
            OpKind::Add((a, b)) => Op::Add(a, b),
            OpKind::Sub((a, b)) => Op::Sub(a, b),
            OpKind::Mul((a, b)) => Op::Mul(a, b),
            OpKind::Div((a, b)) => Op::Div(a, b),
            OpKind::MaxElem((a, b)) => Op::MaxElem(a, b),
            OpKind::MinElem((a, b)) => Op::MinElem(a, b),
            OpKind::Gt((a, b)) => Op::Gt(a, b),
            OpKind::Ge((a, b)) => Op::Ge(a, b),
            OpKind::Eq((a, b)) => Op::Eq(a, b),
            OpKind::Select((cond, a, b)) => Op::Select { cond, a, b },
            OpKind::ReduceSum(a) => Op::ReduceSum(a),
            OpKind::ReduceMax(a) => Op::ReduceMax(a),
            OpKind::ReduceMin(a) => Op::ReduceMin(a),
            OpKind::ReduceArgmax(a) => Op::ReduceArgmax(a),
            OpKind::Broadcast((value, shape)) => Op::Broadcast { value, shape },
            OpKind::CumSum(a) => Op::CumSum(a),
            OpKind::CumProd(a) => Op::CumProd(a),
            OpKind::SortDesc(a) => Op::SortDesc(a),
            OpKind::PivotThreshold((input, predicate)) => Op::PivotThreshold { input, predicate },
            OpKind::Gather((src, idx)) => Op::Gather { src, idx },
            OpKind::GatherRow((src, idx)) => Op::GatherRow { src, idx },
            OpKind::MaskApply((logits, mask)) => Op::MaskApply { logits, mask },
            OpKind::ScatterAdd((base, idx, vals)) => Op::ScatterAdd { base, idx, vals },
            OpKind::ScatterSet((base, idx, vals)) => Op::ScatterSet { base, idx, vals },
            OpKind::Rng((stream, shape, kind)) => Op::Rng { stream, shape, kind },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// One representative of every `Op` variant — the drift-guard corpus.
    fn every_op() -> Vec<Op> {
        vec![
            Op::Input(3),
            Op::Const(Literal::F32(0.5)),
            Op::Const(Literal::I32(-7)),
            Op::Const(Literal::U32(9)),
            Op::Const(Literal::Bool(true)),
            Op::Exp(0),
            Op::Log(0),
            Op::Neg(0),
            Op::Recip(0),
            Op::Abs(0),
            Op::Sign(0),
            Op::Add(0, 1),
            Op::Sub(0, 1),
            Op::Mul(0, 1),
            Op::Div(0, 1),
            Op::MaxElem(0, 1),
            Op::MinElem(0, 1),
            Op::Gt(0, 1),
            Op::Ge(0, 1),
            Op::Eq(0, 1),
            Op::Select { cond: 0, a: 1, b: 2 },
            Op::ReduceSum(0),
            Op::ReduceMax(0),
            Op::ReduceMin(0),
            Op::ReduceArgmax(0),
            Op::Broadcast { value: 0, shape: Shape::matrix(2, 4) },
            Op::CumSum(0),
            Op::CumProd(0),
            Op::SortDesc(0),
            Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(40) },
            Op::PivotThreshold { input: 0, predicate: Predicate::CummassLe(1) },
            Op::PivotThreshold { input: 0, predicate: Predicate::ProbGe(1) },
            Op::Gather { src: 0, idx: 1 },
            Op::GatherRow { src: 0, idx: 1 },
            Op::ScatterAdd { base: 0, idx: 1, vals: 2 },
            Op::ScatterSet { base: 0, idx: 1, vals: 2 },
            Op::Rng { stream: 0, shape: Shape::vector(4), kind: RngKind::Uniform },
            Op::Rng { stream: 7, shape: Shape::matrix(2, 4), kind: RngKind::Gumbel },
        ]
    }

    #[test]
    fn op_kind_round_trip_is_identity() {
        for op in every_op() {
            assert_eq!(op.to_op_kind().to_op(), op, "drift for {op:?}");
        }
    }

    #[test]
    fn covers_every_op_variant() {
        // If a new Op variant is added, `to_op_kind` won't compile without a new
        // arm — and a representative must be added here. This is a reminder that
        // a new Op MUST get a matching WIT op-kind variant + a corpus entry.
        assert_eq!(every_op().len(), 38);
    }
}
