use alloc::vec::Vec;

use super::op::Op;
use crate::types::{Dtype, Literal, Predicate, RngKind, Shape, ValueId};

pub fn next_id(ops: &[Op]) -> ValueId {
    ops.iter().map(|o| o.result_count()).sum()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepShape {
    /// The row's shape and dtype.
    Row,
    /// The row's shape less its last axis, at the dtype a reduction of the
    /// row returns (`infer::reduced_dtype`: f32 for a bf16 row).
    Reduced,
    Scalar,
    RowMask,
    ReducedIndex,
    /// [`StepShape::Reduced`]'s shape at the row's own dtype: a per-row
    /// statistic a bf16 row holds exactly (its max), cast to it before it
    /// is broadcast.
    ReducedAsRow,
    /// The row's shape at [`StepShape::Reduced`]'s dtype: a per-row f32
    /// statistic broadcast over a bf16 row unrounded.
    RowStat,
}

pub trait Sink {
    fn push(&mut self, op: Op, shape: StepShape) -> ValueId;
}

impl Sink for Vec<Op> {
    fn push(&mut self, op: Op, _shape: StepShape) -> ValueId {
        let id = next_id(self);
        Vec::push(self, op);
        id
    }
}

fn push(sink: &mut impl Sink, op: Op, shape: StepShape) -> ValueId {
    sink.push(op, shape)
}

pub fn gumbel(sink: &mut impl Sink, state: ValueId, shape: Shape) -> ValueId {
    push(
        sink,
        Op::RngKeyed {
            state,
            shape,
            kind: RngKind::Gumbel,
        },
        StepShape::Row,
    )
}

pub fn mask_apply(sink: &mut impl Sink, logits: ValueId, mask: ValueId) -> ValueId {
    let ninf = push(
        sink,
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        StepShape::Scalar,
    );
    push(
        sink,
        Op::Select {
            cond: mask,
            a: logits,
            b: ninf,
        },
        StepShape::Row,
    )
}

pub fn softmax(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = push(sink, Op::Broadcast { value: m, shape }, StepShape::Row);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let sb = push(sink, Op::Broadcast { value: s, shape }, StepShape::Row);
    push(sink, Op::Div(e, sb), StepShape::Row)
}

pub fn log_softmax(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = push(sink, Op::Broadcast { value: m, shape }, StepShape::Row);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let l = push(sink, Op::Log(s), StepShape::Reduced);
    let lb = push(sink, Op::Broadcast { value: l, shape }, StepShape::Row);
    push(sink, Op::Sub(c, lb), StepShape::Row)
}

/// A bf16 row's per-row statistic `stat` (f32, [`StepShape::Reduced`]) as
/// an operand of an op over the row. A rank-1 row's statistic is a scalar,
/// which an elementwise op mixes in as f32 with no broadcast. A wider row's
/// is broadcast over it: as bf16 where the row holds it `exact`ly (its max,
/// one of its elements), so no row-sized f32 is laid; as f32 otherwise (a
/// sum, a log-sum-exp), so a statistic is never rounded before the one
/// rounding of the op's result.
fn row_stat(sink: &mut impl Sink, stat: ValueId, shape: Shape, exact: bool) -> ValueId {
    if shape.rank() <= 1 {
        return stat;
    }
    if !exact {
        return push(
            sink,
            Op::Broadcast { value: stat, shape },
            StepShape::RowStat,
        );
    }
    let narrow = push(
        sink,
        Op::Cast {
            value: stat,
            dtype: Dtype::Bf16,
        },
        StepShape::ReducedAsRow,
    );
    push(
        sink,
        Op::Broadcast {
            value: narrow,
            shape,
        },
        StepShape::Row,
    )
}

/// [`softmax`] over a bf16 row: the max and the sum reduce in f32, the
/// row-sized values (the shifted row, its exponentials, the result) are
/// bf16; see [`row_stat`] for how a statistic meets the row.
pub fn softmax_bf16(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = row_stat(sink, m, shape, true);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let sb = row_stat(sink, s, shape, false);
    push(sink, Op::Div(e, sb), StepShape::Row)
}

/// [`log_softmax`] over a bf16 row: `x - (max + log(sum(exp(x - max))))`,
/// the log-sum-exp in f32 and the result rounded once.
pub fn log_softmax_bf16(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = row_stat(sink, m, shape, true);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let l = push(sink, Op::Log(s), StepShape::Reduced);
    let lse = push(sink, Op::Add(m, l), StepShape::Reduced);
    let lb = row_stat(sink, lse, shape, false);
    push(sink, Op::Sub(x, lb), StepShape::Row)
}

/// [`l2norm`] over a bf16 row: the sum of squares in f32.
pub fn l2norm_bf16(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let sq = push(sink, Op::Mul(x, x), StepShape::Row);
    let s = push(sink, Op::ReduceSum(sq), StepShape::Reduced);
    let lg = push(sink, Op::Log(s), StepShape::Reduced);
    let half = push(sink, Op::Const(Literal::F32(0.5)), StepShape::Scalar);
    let h = push(sink, Op::Mul(lg, half), StepShape::Reduced);
    let rt = push(sink, Op::Exp(h), StepShape::Reduced);
    let rb = row_stat(sink, rt, shape, false);
    push(sink, Op::Div(x, rb), StepShape::Row)
}

pub fn l2norm(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let sq = push(sink, Op::Mul(x, x), StepShape::Row);
    let s = push(sink, Op::ReduceSum(sq), StepShape::Reduced);
    let lg = push(sink, Op::Log(s), StepShape::Reduced);
    let half = push(sink, Op::Const(Literal::F32(0.5)), StepShape::Scalar);
    let h = push(sink, Op::Mul(lg, half), StepShape::Reduced);
    let rt = push(sink, Op::Exp(h), StepShape::Reduced);
    let rb = push(sink, Op::Broadcast { value: rt, shape }, StepShape::Row);
    push(sink, Op::Div(x, rb), StepShape::Row)
}

/// Top-p sampling over `logits` of `dtype` (f32 or bf16, which picks
/// [`softmax`] or [`softmax_bf16`]).
pub fn nucleus_sample(
    sink: &mut impl Sink,
    logits: ValueId,
    top_p: ValueId,
    state: ValueId,
    shape: Shape,
    dtype: Dtype,
) -> ValueId {
    if dtype == Dtype::Bf16 {
        // The draw reduces the perturbed lanes in f32 (`types`: a draw is
        // never a bf16 row).
        let probabilities = softmax_bf16(sink, logits, shape);
        let keep = push(
            sink,
            Op::PivotThreshold {
                input: probabilities,
                predicate: Predicate::CummassLe(top_p),
            },
            StepShape::RowMask,
        );
        let one = push(sink, Op::Const(Literal::F32(1.0)), StepShape::Scalar);
        return push(
            sink,
            Op::GumbelArgmax {
                logits,
                keep,
                scale: one,
                state,
            },
            StepShape::ReducedIndex,
        );
    }
    let probabilities = softmax(sink, logits, shape);
    let keep = push(
        sink,
        Op::PivotThreshold {
            input: probabilities,
            predicate: Predicate::CummassLe(top_p),
        },
        StepShape::RowMask,
    );
    let masked = mask_apply(sink, logits, keep);
    let noise = gumbel(sink, state, shape);
    let perturbed = push(sink, Op::Add(masked, noise), StepShape::Row);
    push(sink, Op::ReduceArgmax(perturbed), StepShape::ReducedIndex)
}
