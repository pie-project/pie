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

fn push(ops: &mut Vec<Op>, op: Op) -> ValueId {
    let id = next_id(ops);
    ops.push(op);
    id
}

/// `gumbel(state, shape)` = `-log(-log(u))` over state-keyed uniform noise —
/// exactly [`Op::RngKeyed`] with [`RngKind::Gumbel`] (the fused form).
pub fn gumbel(ops: &mut Vec<Op>, state: ValueId, shape: Shape) -> ValueId {
    push(
        ops,
        Op::RngKeyed {
            state,
            shape,
            kind: RngKind::Gumbel,
        },
    )
}

/// `mask_apply(logits, mask)` = `select(mask, logits, -inf)` — the composed
/// bool-mask form (the packed-word special case is core [`Op::MaskApply`]).
pub fn mask_apply(ops: &mut Vec<Op>, logits: ValueId, mask: ValueId) -> ValueId {
    let ninf = push(ops, Op::Const(Literal::F32(f32::NEG_INFINITY)));
    push(
        ops,
        Op::Select {
            cond: mask,
            a: logits,
            b: ninf,
        },
    )
}

/// Numerically-stable row softmax: `exp(x - max) / sum(exp(x - max))`.
/// `shape` is `x`'s (trace-known) shape, needed to lift the row reductions.
pub fn softmax(ops: &mut Vec<Op>, x: ValueId, shape: Shape) -> ValueId {
    let m = push(ops, Op::ReduceMax(x));
    let mb = push(ops, Op::Broadcast { value: m, shape });
    let c = push(ops, Op::Sub(x, mb));
    let e = push(ops, Op::Exp(c));
    let s = push(ops, Op::ReduceSum(e));
    let sb = push(ops, Op::Broadcast { value: s, shape });
    push(ops, Op::Div(e, sb))
}

/// Stable row log-softmax: `(x - max) - log(sum(exp(x - max)))`.
pub fn log_softmax(ops: &mut Vec<Op>, x: ValueId, shape: Shape) -> ValueId {
    let m = push(ops, Op::ReduceMax(x));
    let mb = push(ops, Op::Broadcast { value: m, shape });
    let c = push(ops, Op::Sub(x, mb));
    let e = push(ops, Op::Exp(c));
    let s = push(ops, Op::ReduceSum(e));
    let l = push(ops, Op::Log(s));
    let lb = push(ops, Op::Broadcast { value: l, shape });
    push(ops, Op::Sub(c, lb))
}

/// Row L2 normalization: `x / sqrt(sum(x^2))`, with `sqrt(y) = exp(0.5·log(y))`
/// over the core map set (no dedicated sqrt op; backends fuse).
pub fn l2norm(ops: &mut Vec<Op>, x: ValueId, shape: Shape) -> ValueId {
    let sq = push(ops, Op::Mul(x, x));
    let s = push(ops, Op::ReduceSum(sq));
    let lg = push(ops, Op::Log(s));
    let half = push(ops, Op::Const(Literal::F32(0.5)));
    let h = push(ops, Op::Mul(lg, half));
    let rt = push(ops, Op::Exp(h));
    let rb = push(ops, Op::Broadcast { value: rt, shape });
    push(ops, Op::Div(x, rb))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{BodyCtx, body_types};
    use crate::types::{DType, ValueType};
    use alloc::vec;

    #[test]
    fn expansions_type_check() {
        let chans = [
            ValueType::new(Shape::matrix(2, 8), DType::F32),
            ValueType::vector(2, DType::U32),
            ValueType::new(Shape::matrix(2, 8), DType::Bool),
        ];
        let mut ops = vec![Op::ChanRead(0), Op::ChanTake(1), Op::ChanRead(2)];
        let x = 0;
        let state = 1;
        let mask = 2;
        let shape = Shape::matrix(2, 8);
        let g = gumbel(&mut ops, state, shape);
        let ma = mask_apply(&mut ops, x, mask);
        let sm = softmax(&mut ops, x, shape);
        let lsm = log_softmax(&mut ops, x, shape);
        let l2 = l2norm(&mut ops, x, shape);
        let t = body_types(
            &ops,
            &BodyCtx {
                channel_types: &chans,
                n_names: 0,
            },
        )
        .unwrap();
        for id in [g, ma, sm, lsm, l2] {
            assert_eq!(t[id as usize], ValueType::new(shape, DType::F32), "id {id}");
        }
    }
}
