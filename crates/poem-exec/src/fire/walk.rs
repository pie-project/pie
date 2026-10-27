use crate::dispatch::Dispatch;
use poem_compiler::{CompiledModel, Lowering, Phase};
use poem_ir::{Operands, Operation, Trace};

use crate::Result;
use crate::fire::Fault;
use crate::fire::compose::MaskSpan;
use crate::fire::descriptor::FireDescriptor;
use crate::fire::fallback::{Serve, grouped as grouped_fallback};
use crate::fire::sink::Sink;

pub fn walk<D: Dispatch + Serve, S: Sink>(
    trace: &Trace,
    compiled: &CompiledModel,
    descriptor: &FireDescriptor,
    dispatch: &mut D,
    sink: &mut S,
    filter: Filter,
) -> Result<()> {
    let Filter {
        phases,
        units,
        regions,
    } = filter;
    let classes = compiled.classes.classes.len();
    if descriptor.classes.len() != classes {
        return Err(Fault::ClassTable {
            descriptor: descriptor.classes.len(),
            compiled: classes,
        }
        .into());
    }

    let mut captured = false;
    let mut runs: Vec<MaskSpan> = Vec::new();
    let mut outs: Vec<poem_ir::ValueId> = Vec::new();
    let mut cut: Vec<(poem_ir::RowAxis, &poem_ir::ClassSet, Vec<MaskSpan>)> = Vec::new();
    for (index, region) in compiled.template().iter().enumerate() {
        match region.phase {
            Phase::Prepare if captured => {
                return Err(Fault::PrepareAfterCapture {
                    region: index as u32,
                }
                .into());
            }
            Phase::Prepare => {}
            Phase::Capture => captured = true,
        }

        let unit = compiled.unit_of(index);
        let axis = compiled.axis_of(index);
        match cut
            .iter()
            .find(|(a, mask, _)| *a == axis && *mask == &region.mask)
        {
            Some((_, _, spans)) => {
                runs.clear();
                runs.extend_from_slice(spans);
            }
            None => {
                descriptor.table(axis).spans_into(&region.mask, &mut runs);
                cut.push((axis, &region.mask, runs.clone()));
            }
        }

        let dispatches =
            phases.admits(region.phase) && units.admits(unit) && regions.admits(index as u32);

        if runs.len() > 1 {
            let bound = super::fallback::bound(compiled, axis, &region.mask);
            let promised = super::fallback::promised(compiled, axis, region);
            if promised || runs.len() > bound as usize {
                return Err(Fault::Fragmented {
                    region: index as u32,
                    runs: runs.len() as u32,
                    bound,
                    promised,
                }
                .into());
            }
        }
        let grouped = runs.len() > 1 && grouped_fallback(compiled, axis, region.nodes.clone());

        let (open, arm, close) = match region.lowering {
            Lowering::AlwaysLaunch => (false, None, false),
            Lowering::If => (true, None, true),
            Lowering::Switch { arm, arms, .. } => (arm == 0, Some(arm), arm + 1 == arms),
        };

        sink.region_begin(region);
        if dispatches {
            for &event in &region.wait {
                sink.join(event);
            }
            if let Some(event) = region.open {
                sink.fork(event);
            }
            if open {
                sink.cond_begin(&region.lowering);
            }
            if let Some(arm) = arm {
                sink.cond_arm(arm);
            }
        }

        let copy = dispatches && !grouped && runs.len() > 1 && dispatch.copies(region);
        if copy {
            dispatch.gather(region)?;
        }

        let once = grouped || copy;
        let mut passes = 1;
        if !once && let Some(&cap) = descriptor.run_caps.get(index) {
            let max_passes = descriptor.run_passes.get(index).copied().unwrap_or(0);
            if max_passes > 1 {
                passes = super::compose::pass_spans(&mut runs, cap, max_passes);
            } else {
                super::compose::chunk_spans(&mut runs, cap);
            }
        }
        let tail_start = if passes > 1 {
            region
                .nodes
                .clone()
                .rfind(|&node| {
                    trace.nodes.get(node as usize).is_some_and(|node| {
                        matches!(
                            node.op,
                            Operation::Linear(
                                poem_ir::Linear::MoeSelect {
                                    bias: None,
                                    quantized: false,
                                    quant: None,
                                    ..
                                } | poem_ir::Linear::MoeSelect {
                                    bias: None,
                                    quantized: true,
                                    quant: None,
                                    ..
                                } | poem_ir::Linear::MoeSelect {
                                    bias: None,
                                    quantized: true,
                                    quant: Some(_),
                                    ..
                                }
                            )
                        )
                    })
                })
                .map_or(region.nodes.end, |last| last + 1)
        } else {
            region.nodes.end
        };
        let launches = if once { 1 } else { runs.len().max(1) };
        for launch in 0..launches {
            sink.run(launch as u32, launches as u32);
            let rows = if once {
                runs.iter().map(|span| span.rows).sum()
            } else {
                runs.get(launch).map_or(0, |span| span.rows)
            };

            for node_at in region.nodes.clone() {
                let node = node_at;
                let Some(node) = trace.nodes.get(node as usize) else {
                    return Err(Fault::NoSuchNode {
                        node,
                        nodes: trace.nodes.len(),
                    }
                    .into());
                };
                if !dispatches {
                    continue;
                }
                if passes > 1 {
                    sink.tail(node_at >= tail_start);
                }
                if rows == 0 && !runs_bare(node) && !received(compiled, node, &mut outs) {
                    continue;
                }
                dispatch.exec(node)?;
            }
        }

        if copy {
            dispatch.scatter(region)?;
        }

        if dispatches {
            if close {
                sink.cond_end();
            }
            if let Some(event) = region.close {
                sink.fork(event);
            }
        }
        sink.region_end(region);
    }

    Ok(())
}

/// Whether `node` runs over no rows of its own: a collective, which meets
/// the other ranks of its group whatever its own rows (an expert's dispatch
/// to a rank with none of its own). A tensor-parallel one under a guard (a
/// layer's captured scores gathered whole, only for lanes whose program
/// reads them) is the exception: every rank of its group fires the same
/// lanes, so where none holds its guard every rank skips it together.
fn runs_bare(node: &poem_ir::Node) -> bool {
    match &node.op {
        Operation::Collective(
            poem_ir::Collective::AllReduce { .. }
            | poem_ir::Collective::AllGather { .. }
            | poem_ir::Collective::ReduceScatter { .. },
        ) => node.guard == poem_ir::Guard::Always,
        Operation::Collective(_) => true,
        _ => false,
    }
}

/// Whether `node` makes an expert region's value: it works on the rows this
/// rank received from every rank, so it runs whatever this rank's own rows
/// (an expert-parallel rank with none of its own still serves the others').
fn received(
    compiled: &CompiledModel,
    node: &poem_ir::Node,
    outs: &mut Vec<poem_ir::ValueId>,
) -> bool {
    if compiled.expert.is_empty() {
        return false;
    }
    outs.clear();
    node.op.outputs(outs);
    outs.iter()
        .any(|v| compiled.expert.get(v.0 as usize).copied().unwrap_or(false))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Filter {
    pub phases: Phases,
    pub units: Units,
    pub regions: Regions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phases {
    #[default]
    All,
    Prepare,
    Capture,
}

impl Phases {
    #[must_use]
    pub fn admits(self, phase: Phase) -> bool {
        match self {
            Phases::All => true,
            Phases::Prepare => phase == Phase::Prepare,
            Phases::Capture => phase == Phase::Capture,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Units {
    #[default]
    All,
    One(u32),
}

impl Units {
    #[must_use]
    pub fn admits(self, unit: u32) -> bool {
        match self {
            Units::All => true,
            Units::One(only) => unit == only,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Regions {
    #[default]
    All,
    Span {
        from: u32,
        upto: u32,
    },
}

impl Regions {
    #[must_use]
    pub fn admits(self, index: u32) -> bool {
        match self {
            Regions::All => true,
            Regions::Span { from, upto } => from <= index && index < upto,
        }
    }
}

pub fn walk_phases<D: Dispatch + Serve, S: Sink>(
    trace: &Trace,
    compiled: &CompiledModel,
    descriptor: &FireDescriptor,
    dispatch: &mut D,
    sink: &mut S,
    phases: Phases,
) -> Result<()> {
    let filter = Filter {
        phases,
        ..Filter::default()
    };
    walk(trace, compiled, descriptor, dispatch, sink, filter)
}

#[allow(clippy::too_many_arguments)]
pub fn walk_regions<D: Dispatch + Serve, S: Sink>(
    trace: &Trace,
    compiled: &CompiledModel,
    descriptor: &FireDescriptor,
    dispatch: &mut D,
    sink: &mut S,
    phases: Phases,
    units: Units,
    regions: Regions,
) -> Result<()> {
    let filter = Filter {
        phases,
        units,
        regions,
    };
    walk(trace, compiled, descriptor, dispatch, sink, filter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::compose::{Lane, compose};
    use crate::fire::fixture::{Build, MockDispatch, Recorder, fact};
    use crate::fire::sink::EagerSink;

    use poem_compiler::{Budget, DeviceProfile, compile};
    use poem_ir::Guard;

    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    fn diagram() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let plan = b.prepare(Guard::Always);
        let q = b.op(x, 4, Guard::Always);
        let d = b.decode(q, plan, fact(0));
        let p = b.op(q, 4, !fact(0));
        let o = b.merge(&[(d, fact(0)), (p, !fact(0))], 4);
        let y = b.op(o, 4, Guard::Always);
        b.out(y);
        b
    }

    fn fire(compiled: &CompiledModel, lanes: &[Lane]) -> FireDescriptor {
        FireDescriptor::of(&compose(compiled, &budget(), lanes).expect("composes"))
    }

    /// A layer's scores captured and gathered whole under the fact a lane
    /// holds only where its program reads them, beside the logits every lane
    /// lands: a fire of no such lane skips the gather with the capture, as
    /// every rank of the group does; one of such a lane runs both.
    #[test]
    fn a_guarded_gather_runs_only_where_a_lane_holds_its_guard() {
        let mut b = Build::new();
        let x = b.input(8);
        let y = b.op(x, 4, Guard::Always);
        let captured = b.op(x, 4, fact(0));
        let whole = b.gather(captured, 8, fact(0));
        let everyone = b.gather(y, 8, Guard::Always);
        b.out(everyone).out(whole);
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let ran = |word: u64| {
            let descriptor = fire(&compiled, &[Lane::new(word, 3)]);
            let mut dispatch = MockDispatch::new(&b.trace);
            walk(
                &b.trace,
                &compiled,
                &descriptor,
                &mut dispatch,
                &mut EagerSink,
                Filter::default(),
            )
            .expect("walks");
            dispatch.nodes()
        };
        assert_eq!(ran(0), vec![0, 3], "the logits and their gather alone");
        assert_eq!(ran(1), vec![0, 1, 2, 3], "the capture and its gather too");
    }

    #[test]
    fn the_phase_filter_splits_one_walk_into_two_instants_and_loses_no_region() {
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 7), Lane::new(1, 1)]);

        let mut whole = MockDispatch::new(&b.trace);
        walk(
            &b.trace,
            &compiled,
            &descriptor,
            &mut whole,
            &mut EagerSink,
            Filter::default(),
        )
        .expect("walks");

        let mut split = MockDispatch::new(&b.trace);
        let mut structure = (Recorder::default(), Recorder::default());
        walk_phases(
            &b.trace,
            &compiled,
            &descriptor,
            &mut split,
            &mut structure.0,
            Phases::Prepare,
        )
        .expect("the prepare pass walks");
        assert_eq!(split.nodes(), vec![0], "the plan build, and nothing else");
        walk_phases(
            &b.trace,
            &compiled,
            &descriptor,
            &mut split,
            &mut structure.1,
            Phases::Capture,
        )
        .expect("the capture pass walks");

        assert_eq!(split.nodes(), whole.nodes());
        assert_eq!(split.names(), whole.names());
        assert_eq!(structure.0.events, structure.1.events);
        assert_eq!(
            structure.0.events.len(),
            compiled.template().len() * 2,
            "every region is opened and closed under a filter that dispatches none of it"
        );
    }
}
