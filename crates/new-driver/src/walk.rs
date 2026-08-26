//! The walk itself: program order, guard, dispatch. That is the whole loop.

use new_kernels::{Dispatch, KernelError};
use new_model_ir::Plan;

use crate::phase::Phases;

/// Walk the given node indices in order, skipping each node whose `cond` does
/// not hold for this fire's fact word, and `D::exec`-ing the rest.
///
/// Standing rules, inherited rather than enforced here:
///
/// - `exec` is enqueue/encode only, never sync (#15) — so a completed walk
///   means the work is *queued*, and this same loop runs identically inside a
///   graph capture.
/// - Merge values need no execution: a φ is data, not an op (`Def::Merge`
///   never appears in `Plan::nodes`) — the compiler aliased its branches onto
///   one arena slot, so there is nothing to walk past.
/// - Ordering between cache writers and readers is program order, which this
///   walk preserves by construction: nodes fire in the sequence given, on one
///   queue.
pub fn walk<D: Dispatch>(
    d: &mut D,
    plan: &Plan,
    nodes: &[u32],
    facts: u64,
) -> Result<(), KernelError> {
    for &i in nodes {
        let node = &plan.nodes[i as usize];
        if node.cond.holds(facts) {
            d.exec(node)?;
        }
    }
    Ok(())
}

/// One whole fire for callers that do not capture: the prepare phase, then the
/// capture phase, both walked eagerly. A capturing driver instead calls
/// [`walk`] twice itself — prepare outside the capture, capture inside it.
pub fn fire<D: Dispatch>(
    d: &mut D,
    plan: &Plan,
    phases: &Phases,
    facts: u64,
) -> Result<(), KernelError> {
    walk(d, plan, &phases.prepare, facts)?;
    walk(d, plan, &phases.capture, facts)
}
