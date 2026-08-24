//! Sweep a supergraph plan's fact words and dedup the surviving behaviors
//! into lanes.
//!
//! # `resolve` STOOD HERE, and it was the second derivation of one truth
//!
//! `Resolution`, `Lowered` and `lower` went with it. `resolve` walked each
//! distinct kernel a plan states and asked three questions in order — is this
//! a plane-gated tier-2 statement, does the plane's `#[claims]` table carry
//! the point, does a `canon` row spell a symbol for it — and
//! `program::call_of` asked exactly the same three, in the same order, with
//! its own header admitting it: *"mirroring `sweep::resolve`"*.
//!
//! Two derivations of one truth is bad enough; these two were not equally
//! expressive, and the LIVE one was the poorer. `Resolution` separated
//! `violations` (a `cuda::` statement on a plane that does not declare it —
//! a refused plan) from `unresolved` (the honest backlog), while `call_of`
//! answered an `Option` and collapsed both into `Why::Unclaimed`. Every
//! driver and the width walk call `call_of`; `resolve` and everything built
//! on it existed for `bin/lanes.rs` and nothing else. So the driver acted on
//! the poorer answer and a report binary read the richer one.
//!
//! `program::call_for` is the three-way answer now, `program::Why` grew
//! `WrongPlane`, and `bin/lanes.rs` formats the report it used to be handed.

use model_ir::plan::Plan;

/// One behavior class: the fact words it serves and the ops that survive.
pub struct Lane {
    pub words: Vec<u64>,
    pub ops: Vec<u32>,
}

#[must_use]
pub fn lanes(plan: &Plan) -> Vec<Lane> {
    let facts = plan.facts.len();
    assert!(facts <= 20, "a plan over {facts} facts");
    let mut lanes: Vec<Lane> = Vec::new();
    for word in 0..1u64 << facts {
        let ops: Vec<u32> = plan
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.cond.holds(word))
            .map(|(i, _)| i as u32)
            .collect();
        match lanes.iter_mut().find(|lane| lane.ops == ops) {
            Some(lane) => lane.words.push(word),
            None => lanes.push(Lane {
                words: vec![word],
                ops,
            }),
        }
    }
    lanes
}
