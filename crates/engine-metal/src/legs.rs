//! The matmuls an op is made of, for the load-time scans that size scratch,
//! repack banks and classify weights by what multiplies them. A fused op
//! hides its legs inside one node; this is where they come back out.

use poem_ir::{Fused, Linear, Operation, ValueId};

/// `(act, w, y)` for every matmul `op` performs: a dense or lm-head
/// matmul is one leg, a fused dense MLP is its gate-up and down legs.
pub(crate) fn matmuls(op: &Operation) -> impl Iterator<Item = (ValueId, ValueId, ValueId)> {
    let legs: [Option<(ValueId, ValueId, ValueId)>; 2] = match op {
        Operation::Linear(Linear::Matmul { act, w, y } | Linear::LmHead { act, w, y }) => {
            [Some((*act, *w, *y)), None]
        }
        Operation::Fused(Fused::MlpSwiglu {
            act,
            gate_up,
            down,
            packed,
            h,
            y,
            ..
        }) => [Some((*act, *gate_up, *packed)), Some((*h, *down, *y))],
        _ => [None, None],
    };
    legs.into_iter().flatten()
}
