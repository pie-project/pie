//! [`walk()`]: walks the regions the compiler baked, in order, handing each
//! region's structure to a [`Sink`] and its nodes to a [`Dispatch`].

use crate::dispatch::Dispatch;
use model_compiler::{CompiledModel, Lowering, Phase};
use model_ir::{Operation, Trace};

use crate::Result;
use crate::fire::Fault;
use crate::fire::compose::MaskSpan;
use crate::fire::descriptor::FireDescriptor;
use crate::fire::fallback::{Serve, grouped as grouped_fallback};
use crate::fire::sink::Sink;

/// Walk one fire. Every region runs in `CompiledModel::template` order and is
/// announced to the sink whether or not this fire has rows for it — the
/// structure is composition-independent, which is what lets one recorded
/// graph serve every composition.
///
/// `filter` says which regions this pass DISPATCHES — phase, unit and span
/// in one argument; [`Filter::default`] is the whole fire.
///
/// # Errors
///
/// [`Fault::ClassTable`] or [`Fault::NoSuchNode`] for a descriptor or a plan
/// that does not belong to this artifact, [`Fault::PrepareAfterCapture`] for a
/// template whose phases are out of order, and
/// [`Error::Kernel`](crate::Error::Kernel) for whatever the backend answered —
/// which is always about the backend and never about the plan.
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
    // A mask indexes the window table by position, so a table of the wrong
    // width does not fail to find a class — it finds another class's rows and
    // runs the fire over them.
    let classes = compiled.classes.classes.len();
    if descriptor.classes.len() != classes {
        return Err(Fault::ClassTable {
            descriptor: descriptor.classes.len(),
            compiled: classes,
        }
        .into());
    }

    let mut captured = false;
    // One buffer for the whole walk, refilled per region, to avoid an
    // allocation per region.
    let mut runs: Vec<MaskSpan> = Vec::new();
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

        // The region's window, cut into the intervals it actually covers —
        // usually one; several when the compiler could not seat it onto a
        // single consecutive row order. An empty window is no intervals, and
        // the loop below still turns once at zero rows.
        let unit = compiled.unit_of(index);
        let axis = compiled.axis_of(index);
        descriptor.table(axis).spans_into(&region.mask, &mut runs);

        // Whether this pass dispatches this region at all — phase, unit and
        // span filters as one question, asked once.
        let dispatches =
            phases.admits(region.phase) && units.admits(unit) && regions.admits(index as u32);

        // A region the compiler answered `Fallback::Grouped` for is one
        // launch over all the intervals, asked only when the window is
        // actually in pieces.
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

        // Conditionals are an optimization, not semantics: the same
        // zero-row rule below decides an eager walk. A switch group's arm
        // carries which arm it is and how many the group has.
        let (open, arm, close) = match region.lowering {
            Lowering::AlwaysLaunch => (false, None, false),
            Lowering::If => (true, None, true),
            Lowering::Switch { arm, arms, .. } => (arm == 0, Some(arm), arm + 1 == arms),
        };

        sink.region_begin(region);
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

        // A region this fire found in pieces that the shell says it copies
        // runs once, over a rectangle the shell gathered the pieces into.
        let copy = dispatches && !grouped && runs.len() > 1 && dispatch.copies(region);
        if copy {
            dispatch.gather(region)?;
        }

        // `max(1)` is the empty window: it turns once, at zero rows, so the
        // collective rule below still sees every node.
        let once = grouped || copy;
        let launches = if once { 1 } else { runs.len().max(1) };
        for launch in 0..launches {
            sink.run(launch as u32, launches as u32);
            // One launch stands over all the window's rows for a copy or
            // grouped region, so the zero-row skip below reads their sum.
            let rows = if once {
                runs.iter().map(|span| span.rows).sum()
            } else {
                runs.get(launch).map_or(0, |span| span.rows)
            };

            for node in region.nodes.clone() {
                // Resolved before the filter, so a template naming a node the
                // plan lacks is the same refusal in every pass.
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
                let collective = matches!(node.op, Operation::Collective(_));
                if rows == 0 && !collective {
                    continue;
                }
                dispatch.exec(node)?;
            }
        }

        // The copy's other half: the gathered rectangle's rows go back to
        // the fire rows they were read from.
        if copy {
            dispatch.scatter(region)?;
        }

        if close {
            sink.cond_end();
        }
        if let Some(event) = region.close {
            sink.fork(event);
        }
        sink.region_end(region);
    }

    Ok(())
}

/// Which of a template's regions a walk dispatches: phase, unit and span as
/// one argument. [`Default`] is the whole fire, which is what an eager shell
/// passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Filter {
    /// Which phases' nodes are dispatched.
    pub phases: Phases,
    /// Which capture unit's nodes are dispatched.
    pub units: Units,
    /// Which stretch of the template is dispatched.
    pub regions: Regions,
}

/// Which phases' nodes a walk dispatches. The structure is not filtered, only
/// the dispatch: every region is announced to the sink under every setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phases {
    /// Both, which is what [`Filter`]'s default means: one pass, the whole fire.
    #[default]
    All,
    /// `Phase::Prepare` only — the host work: plan builders and the staging
    /// they hand the device.
    Prepare,
    /// `Phase::Capture` only — the enqueue-only half, which is the half a
    /// shell records.
    Capture,
}

impl Phases {
    /// Does this setting dispatch a region of `phase`?
    #[must_use]
    pub fn admits(self, phase: Phase) -> bool {
        match self {
            Phases::All => true,
            Phases::Prepare => phase == Phase::Prepare,
            Phases::Capture => phase == Phase::Capture,
        }
    }
}

/// Which capture unit's nodes a walk dispatches. A fire launches one exec per
/// capture unit, each recorded inside its own `cudaStreamBeginCapture`; like
/// [`Phases`], the structure is not filtered, only the dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Units {
    /// Every unit, which is what [`Filter`]'s default means: the whole script,
    /// dispatched in one pass. What an EAGER shell walks — it has no capture
    /// to bracket, so it has no reason to cut the script into execs.
    #[default]
    All,
    /// One unit's regions only — what a RECORDING shell walks, once per
    /// entry of `CompiledModel::units`.
    One(u32),
}

impl Units {
    /// Does this setting dispatch a region recorded into `unit`?
    #[must_use]
    pub fn admits(self, unit: u32) -> bool {
        match self {
            Units::All => true,
            Units::One(only) => unit == only,
        }
    }
}

/// Which contiguous stretch of the template a walk dispatches — the filter
/// segmented capture is cut with. A region that cannot be captured splits the
/// template around it, re-issued eagerly between execs. Every region is
/// still announced to the sink regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Regions {
    /// Every region of the template — what an eager shell walks, and what a
    /// shell whose whole composition is capturable records.
    #[default]
    All,
    /// The half-open stretch `[from, upto)` of `CompiledModel::template`, in
    /// template order — one segment's regions, or one island's.
    Span {
        /// The first region this pass dispatches.
        from: u32,
        /// One past the last.
        upto: u32,
    },
}

impl Regions {
    /// Does this setting dispatch the region at `index`?
    #[must_use]
    pub fn admits(self, index: u32) -> bool {
        match self {
            Regions::All => true,
            Regions::Span { from, upto } => from <= index && index < upto,
        }
    }
}

/// [`walk()`] at one phase — a shell that captures calls it twice, since
/// prepare regions must run on an open stream and capture regions inside
/// `cudaStreamBeginCapture`.
///
/// # Errors
///
/// As [`walk()`].
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

/// [`walk()`] at the whole filter spelled out — the form a segmented capture
/// calls: `exec₁ → island → exec₂ → …` on one stream, each stretch a call
/// over the same template.
///
/// # Errors
///
/// As [`walk()`].
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
    
    
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_ir::Guard;

    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    /// Plan build, shared producer, the split attention pair, shared
    /// consumer.
    fn diagram() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let plan = b.prepare(Guard::Always); // node 0 — prepare
        let q = b.op(x, 4, Guard::Always); // node 1
        let d = b.decode(q, plan, fact(0)); // node 2 — decode window
        let p = b.op(q, 4, Guard::not(fact(0))); // node 3 — prefill window
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always); // node 4
        b.out(y);
        b
    }

    fn fire(compiled: &CompiledModel, lanes: &[Lane]) -> FireDescriptor {
        FireDescriptor::of(&compose(compiled, &budget(), lanes).expect("composes"))
    }

    #[test]
    fn the_phase_filter_splits_one_walk_into_two_instants_and_loses_no_region() {
        // The two passes together must dispatch exactly what one whole walk
        // does, in the same order, and announce the same regions in both.
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 7), Lane::new(1, 1)]);

        let mut whole = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut whole, &mut EagerSink, Filter::default()).expect("walks");

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
        // Same regions, same order, in both passes: the count IS the index.
        assert_eq!(structure.0.events, structure.1.events);
        assert_eq!(
            structure.0.events.len(),
            compiled.template().len() * 2,
            "every region is opened and closed under a filter that dispatches none of it"
        );
    }

}
