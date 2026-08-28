//! P5 then P2 then P5 again: tag each node's phase, coalesce adjacent nodes
//! that agree about phase AND about which classes run them, then [`hoist`] the
//! prepare regions in front of the capture ones.
//!
//! WHY COALESCE AT ALL. A region is the unit the descriptor carries a row
//! count for, the unit P6 forks a stream around, and the unit P3 may one day
//! wrap in a conditional — and all three want the run to be as long as it can
//! be. Nodes that share a mask share a window, so their launches all read one
//! `desc.count[region]`; nodes that share a phase are all host work or all
//! graph body, and a boundary between the two is a boundary the fire path has
//! anyway.
//!
//! WHY ADJACENT, AND NOT "same mask anywhere in the plan". Because program
//! order is kept. Reordering nodes to bring same-masked ones together is a
//! scheduling pass — correctness-neutral, and deferred (design's open items) —
//! and doing it here would mean this pass deciding a question about data
//! dependence that it has no dependence graph to decide with. What the
//! restriction costs is one extra region on each side of every windowed op,
//! and an extra region boundary is a launch, not a recapture.
//!
//! AND WHY THE ONE REORDERING THERE IS, IS NOT THAT PASS. [`hoist`] moves
//! whole regions and so looks like the scheduling the paragraph above defers.
//! It is not: it moves nothing PAST a value it depends on, because the phase
//! boundary is a dependence boundary — a prepare node reads what the driver
//! bound before the fire and writes a descriptor slot the graph reads, so the
//! prepare half is upstream of the capture half in its entirety, and putting
//! it there is restoring an order rather than choosing one. [`hoist`] proves
//! that of the plan in front of it instead of assuming it.

use model_ir::{Def, Operands, Operation, Trace, Ty, ValueId};

use crate::compiled::{Lowering, Phase, Region};
use crate::error::Error;
use model_ir::ClassTable;

/// The regions of a plan, in program order, covering every node exactly once.
pub(crate) fn coalesce(trace: &Trace, classes: &ClassTable) -> Vec<Region> {
    let mut regions: Vec<Region> = Vec::new();
    let mut outs: Vec<ValueId> = Vec::new();

    for (j, node) in trace.nodes.iter().enumerate() {
        let j = j as u32;
        let phase = phase_of(trace, node, &mut outs);
        let collective = matches!(node.op, Operation::Collective(_));
        // `node_mask` is parallel to `trace.nodes`, so the index is the same
        // one — but this crate is a front door, and a plan whose sweep and
        // node list disagree gets the conservative reading rather than a
        // panic: an absent mask is the empty one, which runs in no class.
        let mask = classes
            .node_mask
            .get(j as usize)
            .cloned()
            .unwrap_or_default();

        match regions.last_mut() {
            Some(open) if open.phase == phase && open.mask == mask => {
                open.nodes.end = j + 1;
                open.collective |= collective;
            }
            _ => regions.push(Region {
                nodes: j..j + 1,
                mask,
                phase,
                // EVERY REGION, WITHOUT EXCEPTION, IN v1. Zero-row
                // always-launch is the correctness mechanism and conditionals
                // are the optimization (design §4); P3 is where the profile
                // gets consulted and some of these become `Switch` or `If`.
                lowering: Lowering::AlwaysLaunch,
                // ONE STREAM, until P6 says otherwise. `crate::stream` builds
                // the dep DAG over these regions and stamps the three fields
                // below in place; a plan it finds nothing in keeps exactly
                // what P2 wrote here, which is what "pays nothing" means.
                stream: 0,
                wait: Vec::new(),
                open: None,
                close: None,
                // P6's other half, and deferred past it: SM partition is
                // capture-baked, so a variant multiplies bodies (decision #14).
                sm_hint: None,
                collective,
            }),
        }
    }

    regions
}

/// P5's second half: move the prepare regions in front of the capture regions,
/// keeping each half in program order.
///
/// WHY THE COMPILER AND NOT THE DRIVER. Design §5 lists "plan structs
/// (flashinfer …)" as absorbed by a "prepare-phase hoist → descriptor slots",
/// and `driver::fire::walk`'s rule 3 refuses a template with a prepare region
/// standing after a capture one — deliberately, because the order is the
/// compiler's output and a walk that quietly repaired it would hide this pass
/// behind a fire that mostly works. So the repair belongs here, and rule 3
/// becomes the assertion on the artifact rather than a fault on the fire path.
///
/// WHAT MADE IT NECESSARY. `coalesce` above keeps program order, which is the
/// right default and is wrong for exactly one shape: a plan build stated at
/// the END of the forward pass. qwen3.6's multi-token-prediction head is
/// appended after the trunk, so its `Attention::PlanPrefill` — a `Ty::Struct`
/// definer, and therefore `Phase::Prepare` — landed after three hundred and
/// thirty-nine capture regions and made every composition of that SKU
/// unfireable. Nobody wrote anything wrong in the model text; a supergraph has
/// no obligation to state its host work first.
///
/// WHY IT IS A STABLE PARTITION AND NOT A SORT. Prepare regions may in
/// principle feed each other — one plan build reading another's struct — and
/// nothing here has a dependence graph to re-derive their order from. Program
/// order already is a topological order, so keeping it is both the cheapest
/// answer and the only one that needs no argument. The capture half keeps its
/// order for the stronger reason that it IS the dataflow.
///
/// # Errors
///
/// [`Error::HoistBlocked`] when a prepare node reads a value a capture node
/// computes — see that variant for why the answer is a refusal and not a
/// partial hoist. The check runs over the whole plan before a single region
/// moves, so a refused plan leaves the table untouched.
pub(crate) fn hoist(trace: &Trace, regions: &mut Vec<Region>) -> Result<(), Error> {
    hoistable(trace)?;
    // The common case, and the one every other catalog text is in: a plan
    // that already states its host work first is left alone, byte for byte,
    // and pays this pass one scan.
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
/// output?
///
/// Asked over NODES rather than over regions, and transitively through
/// `Def::Merge`, because a phi is data and not a dispatch — the arms are what
/// actually wrote the bytes, and an arm is a node with a phase.
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

/// Which half of the fire this node runs in.
///
/// THE RULE IS THE TYPE AND NOT A LIST OF NAMES. A node is `Prepare` iff it
/// defines a `Ty::Struct` — a host-owned plan object, backend-opaque, outside
/// the arena. The rewrite kept a hand-written list of prepare kernels beside
/// the ops that were on it, and a new plan builder that nobody remembered to
/// add ran inside the capture, where its host allocation could not go.
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
        coalesce(&b.trace, &classes)
    }

    /// P2 and P5 together, which is what `compile` calls and therefore what
    /// every claim about the TEMPLATE has to be asked of.
    fn hoisted(b: &Build) -> Vec<Region> {
        let mut regions = regions_of(b);
        hoist(&b.trace, &mut regions).expect("the fixture plans hoist");
        regions
    }

    #[test]
    fn a_plan_no_guard_can_tell_apart_is_one_region() {
        // Three unconditional ops in a chain: one mask, one phase, one run.
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Guard::Always);
        let c = b.op(a, 8, Guard::Always);
        let d = b.op(c, 8, Guard::Always);
        b.out(d);

        let regions = regions_of(&b);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].nodes, 0..3);
        assert_eq!(regions[0].phase, Phase::Capture);
        assert_eq!(regions[0].lowering, Lowering::AlwaysLaunch);
        assert!(!regions[0].collective);
    }

    #[test]
    fn a_windowed_op_splits_its_neighborhood() {
        // The decode/prefill shape, in miniature: an unconditional producer, a
        // guarded pair over its output, an unconditional consumer of the
        // merge. The guarded nodes run in one class each, so neither can join
        // the runs on either side of them — three neighbors, four regions.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always); // node 0: everywhere
        let d = b.op(q, 8, fact(0)); // node 1: qo_one only
        let p = b.op(q, 8, Guard::not(fact(0))); // node 2: the other class
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 8);
        let y = b.op(o, 8, Guard::Always); // node 3: everywhere again
        b.out(y);

        let regions = regions_of(&b);
        assert_eq!(regions.len(), 4);
        assert_eq!(regions[0].nodes, 0..1);
        assert_eq!(regions[1].nodes, 1..2);
        assert_eq!(regions[2].nodes, 2..3);
        assert_eq!(regions[3].nodes, 3..4);
        // The two windows are one class each, and they are not the same class.
        assert_eq!(regions[1].mask.len(), 1);
        assert_eq!(regions[2].mask.len(), 1);
        assert_ne!(regions[1].mask, regions[2].mask);
        // The unconditional neighbors run in both.
        assert_eq!(regions[0].mask.len(), 2);
        assert_eq!(regions[3].mask.len(), 2);
    }

    #[test]
    fn a_prepare_node_never_shares_a_region_with_a_capture_node() {
        // Same mask on all three — every node is unconditional — so the ONLY
        // thing that can split this run is the phase, which is the point.
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Guard::Always); // node 0: capture
        let plan = b.prepare(Guard::Always); // node 1: defines a Struct
        let o = b.decode(q, plan, Guard::Always); // node 2: capture
        b.out(o);

        let regions = regions_of(&b);
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].phase, Phase::Capture);
        assert_eq!(regions[1].phase, Phase::Prepare);
        assert_eq!(regions[2].phase, Phase::Capture);
        assert_eq!(regions[1].nodes, 1..2);
        // The mask is what did NOT split them.
        assert_eq!(regions[0].mask, regions[1].mask);
        assert_eq!(regions[1].mask, regions[2].mask);
    }

    #[test]
    fn two_prepare_nodes_in_a_row_are_one_region() {
        let mut b = Build::new();
        let x = b.input(4);
        let one = b.prepare(Guard::Always);
        let two = b.prepare(Guard::Always);
        let a = b.decode(x, one, Guard::Always);
        let c = b.decode(a, two, Guard::Always);
        b.out(c);

        let regions = regions_of(&b);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].nodes, 0..2);
        assert_eq!(regions[0].phase, Phase::Prepare);
        assert_eq!(regions[1].nodes, 2..4);
        assert_eq!(regions[1].phase, Phase::Capture);
    }

    #[test]
    fn a_prepare_node_stated_last_is_hoisted_in_front_of_the_capture_it_follows() {
        // qwen3.6's shape, in miniature: the trunk, then a head appended after
        // it whose attention needs a plan build. Program order puts the build
        // three hundred regions deep; `driver::fire::walk` refuses that, and
        // this is the pass that makes it not happen.
        let mut b = Build::new();
        let x = b.input(4);
        let trunk = b.op(x, 4, Guard::Always); // node 0: capture
        let more = b.op(trunk, 4, Guard::Always); // node 1: capture
        let plan = b.prepare(fact(0)); // node 2: the late plan build
        let head = b.decode(more, plan, fact(0)); // node 3: capture
        b.out(head);

        let before = regions_of(&b);
        assert_eq!(before[0].phase, Phase::Capture);
        assert_eq!(before[1].phase, Phase::Prepare, "P2 keeps program order");

        let regions = hoisted(&b);
        assert_eq!(
            regions.len(),
            before.len(),
            "a hoist moves, it never merges"
        );
        assert_eq!(regions[0].phase, Phase::Prepare);
        assert_eq!(regions[0].nodes, 2..3);
        assert!(
            regions[1..].iter().all(|r| r.phase == Phase::Capture),
            "the prepare half runs first, whole",
        );
        // And the capture half is still the dataflow, untouched.
        assert_eq!(regions[1].nodes, 0..2);
        assert_eq!(regions[2].nodes, 3..4);
    }

    #[test]
    fn two_prepare_regions_keep_their_order_across_the_hoist() {
        // Nothing here has a dependence graph to re-derive a prepare order
        // from, so program order is the one that ships — and it is already a
        // topological order, which is why keeping it needs no argument.
        let mut b = Build::new();
        let x = b.input(4);
        let one = b.prepare(Guard::Always); // node 0
        let a = b.decode(x, one, Guard::Always); // node 1
        let two = b.prepare(fact(0)); // node 2 — a different mask, so a
        let c = b.decode(a, two, fact(0)); // node 3   different region
        b.out(c);

        let regions = hoisted(&b);
        let prepare: Vec<_> = regions
            .iter()
            .filter(|r| r.phase == Phase::Prepare)
            .map(|r| r.nodes.clone())
            .collect();
        assert_eq!(prepare, vec![0..1, 2..3]);
        assert_eq!(regions[0].nodes, 0..1);
        assert_eq!(regions[1].nodes, 2..3);
    }

    #[test]
    fn a_plan_whose_prepare_already_leads_is_left_exactly_as_it_stood() {
        let mut b = Build::new();
        let x = b.input(4);
        let one = b.prepare(Guard::Always);
        let a = b.decode(x, one, Guard::Always);
        b.out(a);

        assert_eq!(hoisted(&b), regions_of(&b));
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

    #[test]
    fn a_collective_marks_the_region_that_carries_it() {
        // A collective takes any guard it likes — this one is guarded — and the
        // flag is what stops P3 putting the region it lands in inside an
        // elidable body (decision #5).
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Guard::Always);
        let g = b.all_gather(a, 8, fact(0));
        let o = b.merge(&[(g, fact(0)), (a, Guard::not(fact(0)))], 8);
        b.out(o);

        let regions = regions_of(&b);
        let carrying: Vec<&Region> = regions.iter().filter(|r| r.collective).collect();
        assert_eq!(carrying.len(), 1);
        assert_eq!(carrying[0].nodes, 1..2);
        assert_eq!(carrying[0].lowering, Lowering::AlwaysLaunch);
    }

    #[test]
    fn every_node_lands_in_exactly_one_region() {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Guard::Always);
        let d = b.op(q, 8, fact(0));
        let m = b.op(q, 8, Guard::and(Guard::not(fact(0)), fact(1)));
        let p = b.op(q, 8, Guard::and(Guard::not(fact(0)), Guard::not(fact(1))));
        let o = b.merge(
            &[
                (d, fact(0)),
                (m, Guard::and(Guard::not(fact(0)), fact(1))),
                (p, Guard::and(Guard::not(fact(0)), Guard::not(fact(1)))),
            ],
            8,
        );
        let y = b.op(o, 8, Guard::Always);
        b.append(y, Guard::Always);
        b.out(y);

        let regions = regions_of(&b);
        let mut covered = 0u32;
        for region in &regions {
            assert_eq!(region.nodes.start, covered, "regions tile the node list");
            covered = region.nodes.end;
        }
        assert_eq!(covered as usize, b.trace.nodes.len());
    }
}
