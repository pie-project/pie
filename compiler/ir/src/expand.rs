//! Composed ops (overview appendix: `gumbel`, `mask_apply`, `softmax`,
//! `log_softmax`, `l2norm`) as **expansions over the core** — sugar the SDK
//! tracer inlines, first-party by construction (D5). Each helper appends the
//! expansion to an op list and returns the result's value id, so builders and
//! tests share one definition and every backend that fuses the core fuses
//! these for free.
//!
//! `next_id(ops)` must equal the SSA id the next op would define; the helpers
//! keep that invariant internally.

use alloc::vec::Vec;

use super::op::Op;
use crate::types::{Literal, RngKind, Shape, ValueId};

/// The SSA id the next appended op's first result would take.
pub fn next_id(ops: &[Op]) -> ValueId {
    ops.iter().map(|o| o.result_count()).sum()
}

/// The shape of one expansion step's result, relative to the expansion's
/// input row.
///
/// The expansions themselves do no shape inference — they only say which of
/// three shapes each step lands in, and the [`Sink`] turns that into whatever
/// it needs. That is what lets the sequences be shared: `pie-ir` needs
/// nothing, `pie-dsl` needs a full [`crate::types::ValueType`] per op, and
/// neither has to restate the op order to get it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepShape {
    /// Same shape as the expansion's input.
    Row,
    /// The input with its last axis reduced away.
    Reduced,
    /// A scalar constant.
    Scalar,
}

/// Where an expansion appends its ops.
///
/// There are two recorders and they must produce the same op sequence, which
/// is why the sequence is written once here rather than once per recorder.
/// `ptir_issues.md` C3 called the two copies drifted; they were not, but only
/// because someone kept them in step by hand.
pub trait Sink {
    /// Append `op` and return the id of its first result.
    fn push(&mut self, op: Op, shape: StepShape) -> ValueId;
}

/// The untyped recorder: `pie-ir` and its callers just want the ops.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{BodyCtx, body_types};
    use crate::types::{DType, ValueType};
    use alloc::vec;
    use alloc::vec::Vec;

    /// A [`Sink`] that also remembers what shape each step claimed to be.
    struct Recording {
        ops: Vec<Op>,
        claims: Vec<(ValueId, StepShape)>,
    }

    impl Sink for Recording {
        fn push(&mut self, op: Op, shape: StepShape) -> ValueId {
            let id = Sink::push(&mut self.ops, op, shape);
            self.claims.push((id, shape));
            id
        }
    }

    #[test]
    fn expansions_type_check() {
        let chans = [
            ValueType::new(Shape::matrix(2, 8), DType::F32),
            ValueType::vector(2, DType::U32),
            ValueType::new(Shape::matrix(2, 8), DType::Bool),
        ];
        let mut sink = Recording {
            ops: vec![Op::ChanRead(0), Op::ChanTake(1), Op::ChanRead(2)],
            claims: Vec::new(),
        };
        let x = 0;
        let state = 1;
        let mask = 2;
        let shape = Shape::matrix(2, 8);
        let g = gumbel(&mut sink, state, shape);
        let ma = mask_apply(&mut sink, x, mask);
        let sm = softmax(&mut sink, x, shape);
        let lsm = log_softmax(&mut sink, x, shape);
        let l2 = l2norm(&mut sink, x, shape);
        let t = body_types(
            &sink.ops,
            &BodyCtx {
                channel_types: &chans,
                n_names: 0,
            },
        )
        .unwrap();
        for id in [g, ma, sm, lsm, l2] {
            assert_eq!(t[id as usize], ValueType::new(shape, DType::F32), "id {id}");
        }

        // Every step's `StepShape` must be the type inference actually gives
        // it. `pie-dsl` builds its recorded `ValueType`s out of nothing but
        // this tag, so a step tagged wrong would be recorded with the wrong
        // type there and only there — which is exactly how two hand-kept
        // copies of these sequences drift.
        let row = ValueType::new(shape, DType::F32);
        let reduced = ValueType::new(shape.drop_last().unwrap(), DType::F32);
        for (id, claim) in sink.claims {
            let want = match claim {
                StepShape::Row => row,
                StepShape::Reduced => reduced,
                StepShape::Scalar => ValueType::scalar(DType::F32),
            };
            assert_eq!(t[id as usize], want, "id {id} claimed {claim:?}");
        }
    }
}
