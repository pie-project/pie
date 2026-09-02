//! Tag each node's phase, coalesce adjacent nodes that agree about phase and
//! about which classes run them, then [`hoist`] the prepare regions in front
//! of the capture ones.
//!
//! A region is the unit the descriptor carries a row count for and the unit
//! a stream forks around, so the run should be as long as it can be. Nodes
//! that share a mask share a window (one `desc.count[region]`); nodes that
//! share a phase are all host work or all graph body.
//!
//! Only adjacent nodes coalesce, not "same mask anywhere in the plan":
//! program order is kept, since reordering is a scheduling question this
//! pass has no dependence graph to decide.
//!
//! [`hoist`] is the one reordering here, but it is not scheduling: it moves
//! nothing past a value it depends on, because the phase boundary is itself
//! a dependence boundary — the prepare half is upstream of the capture half
//! in its entirety, so hoisting restores an order rather than choosing one.
//! [`hoist`] proves that of the plan in front of it instead of assuming it.

use model_ir::{Def, Operands, Operation, RowAxis, Trace, Ty, ValueId};

use crate::compiled::{Lowering, Phase, Region};
use crate::error::Error;
use crate::unit::node_axis;
use model_ir::ClassTable;

/// The regions of a plan, in program order, covering every node exactly once.
pub(crate) fn coalesce(trace: &Trace, classes: &ClassTable) -> Result<Vec<Region>, Error> {
    let mut regions: Vec<Region> = Vec::new();
    let mut outs: Vec<ValueId> = Vec::new();
    // The axis of the run currently open, alongside the mask and the phase.
    // `None` is "nothing in this run has said yet", which a `Const`-shaped
    // node leaves untouched.
    let mut open_axis: Option<RowAxis> = None;

    for (j, node) in trace.nodes.iter().enumerate() {
        let j = j as u32;
        let phase = phase_of(trace, node, &mut outs);
        let axis = node_axis(trace, j, node, &mut outs)?;
        let collective = matches!(node.op, Operation::Collective(_));
        let mask = classes.node_mask[j as usize].clone();

        // The third reason a run breaks: an axis splits two row spaces,
        // whose counts come out of different window tables entirely. A node
        // that names no axis constrains nothing and joins whatever is open.
        let joins = axis.is_none() || open_axis.is_none() || open_axis == axis;
        match regions.last_mut() {
            Some(open) if open.phase == phase && open.mask == mask && joins => {
                open.nodes.end = j + 1;
                open.collective |= collective;
                open_axis = open_axis.or(axis);
                open.axis = open_axis;
            }
            _ => {
                regions.push(Region {
                    nodes: j..j + 1,
                    mask,
                    phase,
                    axis,
                    // Every region, without exception, for now: zero-row
                    // always-launch is the correctness mechanism, and
                    // conditionals (`Switch`/`If`) are a later optimization.
                    lowering: Lowering::AlwaysLaunch,
                    // One stream, until `crate::stream` builds the dep DAG
                    // over these regions and stamps the three fields below.
                    stream: 0,
                    wait: Vec::new(),
                    open: None,
                    close: None,
                    collective,
                });
                // A fresh run opens on this node's axis, `None` included.
                open_axis = axis;
            }
        }
    }

    Ok(regions)
}

/// Move the prepare regions in front of the capture regions, keeping each
/// half in program order.
///
/// Done in the compiler rather than the engine because `engine::fire::walk`
/// refuses a template with a prepare region standing after a capture one
/// deliberately — the order is the compiler's output, so the repair belongs
/// here rather than being silently patched at fire time.
///
/// `coalesce` above keeps program order, which is wrong for exactly one
/// shape: a plan build stated at the end of the forward pass (e.g. an
/// appended multi-token-prediction head), which would otherwise land its
/// `Phase::Prepare` region after hundreds of capture regions.
///
/// A stable partition rather than a sort: prepare regions may feed each
/// other and nothing here has a dependence graph to re-derive their order
/// from, but program order is already a topological order.
///
/// # Errors
///
/// [`Error::HoistBlocked`] when a prepare node reads a value a capture node
/// computes — see that variant for why the answer is a refusal and not a
/// partial hoist. The check runs over the whole plan before a single region
/// moves, so a refused plan leaves the table untouched.
pub(crate) fn hoist(trace: &Trace, regions: &mut Vec<Region>) -> Result<(), Error> {
    hoistable(trace)?;
    // The common case: a plan that already states its host work first is
    // left alone, byte for byte, and pays this pass one scan.
    if regions
        .iter()
        .skip_while(|region| region.phase == Phase::Prepare)
        .all(|region| region.phase == Phase::Capture)
    {
        return Ok(());
    }
    let mut hoisted: Vec<Region> = Vec::with_capacity(regions.len());
    hoisted.extend(
        regions
            .iter()
            .filter(|region| region.phase == Phase::Prepare)
            .cloned(),
    );
    hoisted.extend(
        regions
            .iter()
            .filter(|region| region.phase == Phase::Capture)
            .cloned(),
    );
    *regions = hoisted;
    Ok(())
}

/// The hoist's precondition: does any prepare node read a capture node's
/// output? Asked over nodes rather than regions, and transitively through
/// `Def::Merge`, because a phi is data and not a dispatch — the arms are
/// what actually wrote the bytes.
fn hoistable(trace: &Trace) -> Result<(), Error> {
    let prepare: Vec<bool> = {
        let mut outs = Vec::new();
        trace.nodes
            .iter()
            .map(|node| phase_of(trace, node, &mut outs) == Phase::Prepare)
            .collect()
    };

    let mut inputs = Vec::new();
    let mut stack: Vec<ValueId> = Vec::new();
    let mut seen: Vec<bool> = vec![false; trace.values.len()];
    for (at, node) in trace.nodes.iter().enumerate() {
        if !prepare[at] {
            continue;
        }
        inputs.clear();
        node.op.inputs(&mut inputs);
        seen.iter_mut().for_each(|slot| *slot = false);
        stack.extend_from_slice(&inputs);
        while let Some(value) = stack.pop() {
            match seen.get_mut(value.0 as usize) {
                Some(slot) if !*slot => *slot = true,
                _ => continue,
            }
            match trace.values.get(value.0 as usize).map(|decl| &decl.def) {
                Some(Def::Op(by)) => {
                    if !prepare.get(*by as usize).copied().unwrap_or(false) {
                        return Err(Error::HoistBlocked {
                            node: at as u32,
                            value,
                            produced_by: *by,
                        });
                    }
                }
                // A phi is not a writer. Its arms are, so the question is
                // asked of them.
                Some(Def::Merge(arms)) => stack.extend(arms.iter().map(|(arm, _)| *arm)),
                // A weight, a cache slab, a runtime binding: all of them stand
                // before the fire, which is what makes the hoist free.
                _ => {}
            }
        }
    }
    Ok(())
}

/// Which half of the fire this node runs in. The rule is the type, not a
/// list of names: a node is `Prepare` iff it defines a `Ty::Struct` — a
/// host-owned plan object, backend-opaque, outside the arena.
fn phase_of(trace: &Trace, node: &model_ir::Node, outs: &mut Vec<ValueId>) -> Phase {
    outs.clear();
    node.op.outputs(outs);
    let host = outs.iter().any(|v| {
        trace.values
            .get(v.0 as usize)
            .is_some_and(|decl| matches!(decl.ty, Ty::Struct(_)))
    });
    if host { Phase::Prepare } else { Phase::Capture }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use model_ir::{Guard, resolve_classes};

    fn regions_of(b: &Build) -> Vec<Region> {
        let classes = resolve_classes(&b.trace).expect("the fixture plans resolve");
        coalesce(&b.trace, &classes).expect("the fixture coalesces")
    }

    #[test]
    fn a_plan_build_over_an_activation_is_refused_rather_than_hoisted() {
        // The precondition, stated as the failure it guards. There is no
        // instant that is both after node 0 computed the indptr and before the
        // graph node 2 reads the schedule in, so the honest answer at the door
        // is a refusal naming all three.
        let mut b = Build::new();
        let x = b.input(4);
        let computed = b.op(x, 4, Guard::Always); // node 0: capture
        let plan = b.prepare_over(computed, Guard::Always); // node 1
        let o = b.decode(computed, plan, Guard::Always); // node 2
        b.out(o);

        let mut regions = regions_of(&b);
        let stood = regions.clone();
        let refusal = hoist(&b.trace, &mut regions).expect_err("an activation blocks the hoist");
        assert_eq!(
            refusal,
            Error::HoistBlocked {
                node: 1,
                value: computed,
                produced_by: 0,
            },
        );
        assert_eq!(regions, stood, "a refused plan leaves the table untouched");
        assert!(refusal.to_string().contains("host work"));
    }

    #[test]
    fn a_plan_build_reading_a_merge_of_activations_is_refused_through_the_phi() {
        // A phi is data and never a dispatch, so the question has to reach the
        // ARMS — a check that stopped at `Def::Merge` would hoist this.
        let mut b = Build::new();
        let x = b.input(4);
        let d = b.op(x, 4, fact(0));
        let p = b.op(x, 4, Guard::not(fact(0)));
        let m = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let plan = b.prepare_over(m, Guard::Always);
        let o = b.decode(m, plan, Guard::Always);
        b.out(o);

        let mut regions = regions_of(&b);
        let refusal = hoist(&b.trace, &mut regions).expect_err("the arms are activations too");
        assert!(matches!(refusal, Error::HoistBlocked { node: 2, .. }));
    }

}
