use model_ir::{Def, Linear, Operands, Operation, RowAxis, Trace, Ty, ValueId};

use crate::compiled::{Lowering, Phase, Region};
use crate::error::Error;
use crate::unit::node_axis;
use model_ir::ClassTable;

pub(crate) fn coalesce(trace: &Trace, classes: &ClassTable) -> Result<Vec<Region>, Error> {
    let mut regions: Vec<Region> = Vec::new();
    let mut outs: Vec<ValueId> = Vec::new();
    let mut open_axis: Option<RowAxis> = None;
    let break_after = routed_breaks(trace);

    for (j, node) in trace.nodes.iter().enumerate() {
        let j = j as u32;
        let phase = phase_of(trace, node, &mut outs);
        let axis = node_axis(trace, j, node, &mut outs)?;
        let collective = matches!(node.op, Operation::Collective(_));
        let mask = classes.node_mask[j as usize].clone();

        let joins = axis.is_none() || open_axis.is_none() || open_axis == axis;
        let joins = joins && !(j > 0 && break_after.contains(&(j - 1)));
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
                    lowering: Lowering::AlwaysLaunch,
                    stream: 0,
                    wait: Vec::new(),
                    open: None,
                    close: None,
                    collective,
                });
                open_axis = axis;
            }
        }
    }

    Ok(regions)
}

fn routed_breaks(trace: &Trace) -> std::collections::BTreeSet<u32> {
    let mut breaks = std::collections::BTreeSet::new();
    let mut outs: Vec<ValueId> = Vec::new();
    let mut ins: Vec<ValueId> = Vec::new();
    for (r, node) in trace.nodes.iter().enumerate() {
        if !is_router(node) {
            continue;
        }
        breaks.insert(r as u32);
        outs.clear();
        node.op.outputs(&mut outs);
        let mut last: Option<u32> = None;
        for (k, later) in trace.nodes.iter().enumerate().skip(r + 1) {
            ins.clear();
            later.op.inputs(&mut ins);
            if ins.iter().any(|id| outs.contains(id)) {
                last = Some(k as u32);
            }
        }
        if let Some(last) = last {
            breaks.insert(last);
        }
    }
    breaks
}

pub(crate) fn is_router(node: &model_ir::Node) -> bool {
    matches!(
        node.op,
        Operation::Linear(
            Linear::MoeTopkSoftmax { .. }
                | Linear::MoeTopkSoftmaxScaled { .. }
                | Linear::MoeTopkSigmoid { .. }
                | Linear::MoeTopkSigmoidSink { .. }
                | Linear::MoeTopkSqrtSoftplus { .. }
                | Linear::MoeHashRoute { .. }
        )
    )
}

pub(crate) fn hoist(trace: &Trace, regions: &mut Vec<Region>) -> Result<(), Error> {
    hoistable(trace)?;
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

fn hoistable(trace: &Trace) -> Result<(), Error> {
    let prepare: Vec<bool> = {
        let mut outs = Vec::new();
        trace
            .nodes
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
                Some(Def::Merge(arms)) => stack.extend(arms.iter().map(|(arm, _)| *arm)),
                _ => {}
            }
        }
    }
    Ok(())
}

fn phase_of(trace: &Trace, node: &model_ir::Node, outs: &mut Vec<ValueId>) -> Phase {
    outs.clear();
    node.op.outputs(outs);
    let host = outs.iter().any(|v| {
        trace
            .values
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
    fn region_every_case() {
        a_plan_build_over_an_activation_is_refused_rather_than_hoisted();
        a_plan_build_reading_a_merge_of_activations_is_refused_through_the_phi();
    }

    fn a_plan_build_over_an_activation_is_refused_rather_than_hoisted() {
        let mut b = Build::new();
        let x = b.input(4);
        let computed = b.op(x, 4, Guard::Always);
        let plan = b.prepare_over(computed, Guard::Always);
        let o = b.decode(computed, plan, Guard::Always);
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

    fn a_plan_build_reading_a_merge_of_activations_is_refused_through_the_phi() {
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
