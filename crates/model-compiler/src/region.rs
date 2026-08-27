//! P5 then P2: tag each node's phase, then coalesce adjacent nodes that agree
//! about phase AND about which classes run them.
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

use model_ir::{Operands, Operation, Plan, Ty, ValueId};

use crate::baked::{Lowering, Phase, Region};
use model_ir::Classes;

/// The regions of a plan, in program order, covering every node exactly once.
pub(crate) fn coalesce(plan: &Plan, classes: &Classes) -> Vec<Region> {
    let mut regions: Vec<Region> = Vec::new();
    let mut outs: Vec<ValueId> = Vec::new();

    for (j, node) in plan.nodes.iter().enumerate() {
        let j = j as u32;
        let phase = phase_of(plan, node, &mut outs);
        let collective = matches!(node.op, Operation::Collective(_));
        // `node_mask` is parallel to `plan.nodes`, so the index is the same
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

/// Which half of the fire this node runs in.
///
/// THE RULE IS THE TYPE AND NOT A LIST OF NAMES. A node is `Prepare` iff it
/// defines a `Ty::Struct` — a host-owned plan object, backend-opaque, outside
/// the arena. The rewrite kept a hand-written list of prepare kernels beside
/// the ops that were on it, and a new plan builder that nobody remembered to
/// add ran inside the capture, where its host allocation could not go.
fn phase_of(plan: &Plan, node: &model_ir::Node, outs: &mut Vec<ValueId>) -> Phase {
    outs.clear();
    node.op.outputs(outs);
    let host = outs.iter().any(|v| {
        plan.values
            .get(v.0 as usize)
            .is_some_and(|decl| matches!(decl.ty, Ty::Struct(_)))
    });
    if host { Phase::Prepare } else { Phase::Capture }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use model_ir::{Cond, resolve_classes};

    fn regions_of(b: &Build) -> Vec<Region> {
        let classes = resolve_classes(&b.plan).expect("the fixture plans resolve");
        coalesce(&b.plan, &classes)
    }

    #[test]
    fn a_plan_no_guard_can_tell_apart_is_one_region() {
        // Three unconditional ops in a chain: one mask, one phase, one run.
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let c = b.op(a, 8, Cond::Always);
        let d = b.op(c, 8, Cond::Always);
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
        let q = b.op(x, 8, Cond::Always); // node 0: everywhere
        let d = b.op(q, 8, fact(0)); // node 1: qo_one only
        let p = b.op(q, 8, Cond::not(fact(0))); // node 2: the other class
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always); // node 3: everywhere again
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
        let q = b.op(x, 4, Cond::Always); // node 0: capture
        let plan = b.prepare(Cond::Always); // node 1: defines a Struct
        let o = b.decode(q, plan, Cond::Always); // node 2: capture
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
        let one = b.prepare(Cond::Always);
        let two = b.prepare(Cond::Always);
        let a = b.decode(x, one, Cond::Always);
        let c = b.decode(a, two, Cond::Always);
        b.out(c);

        let regions = regions_of(&b);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].nodes, 0..2);
        assert_eq!(regions[0].phase, Phase::Prepare);
        assert_eq!(regions[1].nodes, 2..4);
        assert_eq!(regions[1].phase, Phase::Capture);
    }

    #[test]
    fn a_collective_marks_the_region_that_carries_it() {
        // A collective takes any cond it likes — this one is guarded — and the
        // flag is what stops P3 putting the region it lands in inside an
        // elidable body (decision #5).
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let g = b.all_gather(a, 8, fact(0));
        let o = b.merge(&[(g, fact(0)), (a, Cond::not(fact(0)))], 8);
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
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0));
        let m = b.op(q, 8, Cond::and(Cond::not(fact(0)), fact(1)));
        let p = b.op(q, 8, Cond::and(Cond::not(fact(0)), Cond::not(fact(1))));
        let o = b.merge(
            &[
                (d, fact(0)),
                (m, Cond::and(Cond::not(fact(0)), fact(1))),
                (p, Cond::and(Cond::not(fact(0)), Cond::not(fact(1)))),
            ],
            8,
        );
        let y = b.op(o, 8, Cond::Always);
        b.append(y, Cond::Always);
        b.out(y);

        let regions = regions_of(&b);
        let mut covered = 0u32;
        for region in &regions {
            assert_eq!(region.nodes.start, covered, "regions tile the node list");
            covered = region.nodes.end;
        }
        assert_eq!(covered as usize, b.plan.nodes.len());
    }
}
