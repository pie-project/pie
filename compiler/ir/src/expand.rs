//! Composed ops (`gumbel`, `mask_apply`, `softmax`,
//! `log_softmax`, `l2norm`) as **expansions over the core** — sugar the SDK
//! tracer inlines, first-party by construction. Each helper appends the
//! expansion to an op list and returns the result's value id, so builders and
//! tests share one definition and every backend that fuses the core fuses
//! these for free.
//!
//! `next_id(ops)` must equal the SSA id the next op would define; the helpers
//! keep that invariant internally.

use alloc::vec::Vec;

use super::op::Op;
use crate::types::{Literal, Predicate, RngKind, Shape, ValueId};

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
    /// The input's shape, but a boolean element — a per-element keep mask.
    RowMask,
    /// The input with its last axis reduced away, holding an index rather
    /// than a value.
    ReducedIndex,
}

/// Where an expansion appends its ops.
///
/// There are two recorders and they must produce the same op sequence, which
/// is why the sequence is written once here rather than once per recorder.
/// Two expansions kept in step by hand can stay identical indefinitely and
/// still be a latent bug: nothing reports the day one of them is edited
/// alone, so the cost of the duplication is paid all at once, later.
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

/// Exact nucleus (top-p) sampling:
/// `argmax(mask_apply(logits, cummass_le(softmax(logits), top_p)) + gumbel(state))`.
///
/// This is the one expansion `pie-plan` also has to *recognize* — the whole
/// nucleus region template exists to fuse it back into a single kernel. That
/// recognizer, `compile::region::match_nucleus`, cannot see `pie-dsl`, so
/// until this lived here the SDK spelled the chain out flat in `value.rs` and
/// the matcher was written against a copy of it. Both are now checked against
/// this one sequence: the SDK builds it, and the matcher's fixtures are
/// generated from it and then mutated one op at a time.
///
/// Temperature scaling is deliberately not part of it — it stays an ordinary
/// preceding `Mul`, which is what lets the region absorb it or not.
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
            ValueType::vector(2, DType::F32),
        ];
        let mut sink = Recording {
            ops: vec![
                Op::ChanRead(0),
                Op::ChanTake(1),
                Op::ChanRead(2),
                Op::ChanRead(3),
            ],
            claims: Vec::new(),
        };
        let x = 0;
        let state = 1;
        let mask = 2;
        let top_p = 3;
        let shape = Shape::matrix(2, 8);
        let g = gumbel(&mut sink, state, shape);
        let ma = mask_apply(&mut sink, x, mask);
        let sm = softmax(&mut sink, x, shape);
        let lsm = log_softmax(&mut sink, x, shape);
        let l2 = l2norm(&mut sink, x, shape);
        let ns = nucleus_sample(&mut sink, x, top_p, state, shape);
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
        assert_eq!(
            t[ns as usize],
            ValueType::new(shape.drop_last().unwrap(), DType::I32),
            "nucleus_sample yields one token id per row"
        );

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
                StepShape::RowMask => ValueType::new(shape, DType::Bool),
                StepShape::ReducedIndex => ValueType::new(shape.drop_last().unwrap(), DType::I32),
            };
            assert_eq!(t[id as usize], want, "id {id} claimed {claim:?}");
        }
    }
}
