//! The prepare/capture split (design §6, decision #16).
//!
//! Struct-producing ops are host work — a CUDA graph cannot capture them, a
//! Metal command buffer cannot encode them. Everything else is device work,
//! capturable and replayable. The boundary is decidable from the IR alone:
//! whether an op's output is `Ty::Struct` is written in `Plan::values`, so the
//! split is a mechanical read, not a judgement call. That legibility is the
//! point of explicit plan ops — the old driver conjured plan objects
//! internally, and the boundary lived in its head.

use model_ir::{Operands, Plan, Ty};

/// A plan's nodes, partitioned into the two phases of a fire. Both lists keep
/// program order, so a walk over either preserves the ordering the trace
/// stated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phases {
    /// Host work: nodes defining a `Struct` value. Run eagerly every fire.
    pub prepare: Vec<u32>,
    /// Device work: everything else. Captured once and replayed where the
    /// backend supports it — arena offsets are stable, only contents change.
    pub capture: Vec<u32>,
}

/// Split a plan by output type: a node any of whose outputs is declared
/// `Ty::Struct` belongs to the prepare phase; every other node belongs to the
/// capture phase.
///
/// Computed once per plan, not per fire — the split depends on nothing a fire
/// binds.
#[must_use]
pub fn phases(plan: &Plan) -> Phases {
    let mut prepare = Vec::new();
    let mut capture = Vec::new();
    let mut outs = Vec::new();
    for (i, node) in plan.nodes.iter().enumerate() {
        outs.clear();
        node.op.outputs(&mut outs);
        let structural = outs
            .iter()
            .any(|id| matches!(plan.values[id.0 as usize].ty, Ty::Struct(_)));
        let phase = if structural { &mut prepare } else { &mut capture };
        phase.push(i as u32);
    }
    Phases { prepare, capture }
}
