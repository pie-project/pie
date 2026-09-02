//! Composed ops (`gumbel`, `mask_apply`, `softmax`, `log_softmax`, `l2norm`) as expansions over the core: each helper appends ops to a list and returns the result's value id.
//! `next_id(ops)` must equal the SSA id the next op would define; the helpers keep that invariant internally.

use alloc::vec::Vec;

use super::op::Op;
use crate::types::{Literal, Predicate, RngKind, Shape, ValueId};

/// The SSA id the next appended op's first result would take.
pub fn next_id(ops: &[Op]) -> ValueId {
    ops.iter().map(|o| o.result_count()).sum()
}

/// The shape of one expansion step's result, relative to the expansion's input row.
/// The expansion only tags which shape a step lands in; the [`Sink`] turns that into whatever type it needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepShape {
    /// Same shape as the expansion's input.
    Row,
    /// The input with its last axis reduced away.
    Reduced,
    /// A scalar constant.
    Scalar,
    /// The input's shape, but a boolean element — a per-element keep mask.
    RowMask,
    /// The input with its last axis reduced away, holding an index rather
    /// than a value.
    ReducedIndex,
}

/// Where an expansion appends its ops. Written once here so the two recorders can't drift apart.
pub trait Sink {
    /// Append `op` and return the id of its first result.
    fn push(&mut self, op: Op, shape: StepShape) -> ValueId;
}

/// The untyped recorder: `eta-ir` and its callers just want the ops.
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

/// `gumbel(state, shape)` — state-keyed Gumbel noise, which is exactly
/// [`Op::RngKeyed`] with [`RngKind::Gumbel`] (the fused form, not `-log(-log(u))`).
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

/// `mask_apply(logits, mask)` = `select(mask, logits, -inf)` — the composed
/// bool-mask form.
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

/// Numerically-stable row softmax: `exp(x - max) / sum(exp(x - max))`.
/// `shape` is `x`'s (trace-known) shape, needed to lift the row reductions.
pub fn softmax(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = push(sink, Op::Broadcast { value: m, shape }, StepShape::Row);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let sb = push(sink, Op::Broadcast { value: s, shape }, StepShape::Row);
    push(sink, Op::Div(e, sb), StepShape::Row)
}

/// Stable row log-softmax: `(x - max) - log(sum(exp(x - max)))`.
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

/// Row L2 normalization: `x / sqrt(sum(x^2))`, with `sqrt(y) = exp(0.5·log(y))`
/// over the core map set (there is no dedicated sqrt op; backends fuse it).
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

/// Exact nucleus (top-p) sampling:
/// `argmax(mask_apply(logits, cummass_le(softmax(logits), top_p)) + gumbel(state))`.
///
/// Temperature scaling is not part of it — it stays an ordinary preceding `Mul`.
pub fn nucleus_sample(
    sink: &mut impl Sink,
    logits: ValueId,
    top_p: ValueId,
    state: ValueId,
    shape: Shape,
) -> ValueId {
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

