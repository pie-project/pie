//! [`walk()`]: one interpreter, two modes (design §6).
//!
//! The whole fire, as a loop: take the regions the compiler baked, in the
//! order it baked them, and for each one hand its structure to a [`Sink`] and
//! its nodes to a [`Dispatch`]. There is no scheduling here, no lowering
//! decision and no allocation — all three happened once, at load, in
//! `model_compiler::compile`. What is left is the part that has to be right
//! 5000 times a second.
//!
//! # The four rules it enforces, and none of them is an optimization
//!
//! **1. Zero rows means the node does not run** (decision #3). A region's
//! window is its mask's rows in this fire's descriptor; when that is zero, the
//! composition simply does not include the behavior — an all-decode fire has
//! no prefill-attention rows — and eager mode skips the dispatch entirely.
//! Recorded mode makes the same decision on the device instead, where the
//! kernel reads the count and returns in about a microsecond. Same walk, same
//! answer, different instant.
//!
//! **2. A collective runs anyway** (decision #5). NCCL matches calls by ORDER,
//! not by name: a rank that skips an `all_reduce` because its window is empty
//! either deadlocks the ranks that did not skip, or — worse — pairs its next
//! collective with their current one and returns numbers that are wrong in a
//! way nothing checks. So the zero-row skip is guarded by the op family, and a
//! zero-count collective joins the rendezvous. The engine's half of the
//! contract is that the descriptor is replicated identically across ranks.
//!
//! **3. Prepare comes before capture** (design §5). Prepare ops are host work
//! that writes descriptor slots the graph then reads — the flashinfer-style
//! plan builds P5 hoists out of the capture — so one standing after a capture
//! region is a slot written after it was read. The walk REFUSES such a
//! template rather than reordering it: the order is the compiler's output, and
//! quietly repairing it here would hide a P2/P5 bug behind a fire that mostly
//! works.
//!
//! **4. A window that is not one interval is not one launch — unless the
//! shell or a kernel says otherwise** (design §3). P4 chooses one row order
//! for the whole plan and makes as many windowed consumers consecutive under
//! it as a PQ-tree can; the ones it cannot seat get a `Fallback` row instead,
//! and the menu it writes is bucket-keyed because the cost model is
//! (`model_compiler::layout`'s `CROSSOVER_ROWS`).
//!
//! `Split` is this loop turning `r` times — dispatch the region's nodes once
//! per maximal interval, each over its own pointer and extent — and the
//! consecutive case is that same loop turning once. THE UNION OF THE RUNS IS
//! THE WINDOW — the intervals partition the mask's rows — which is what makes
//! a split a slow path and not a different answer.
//!
//! **THIS RULE USED TO SAY "SO THERE IS NO BRANCH ON THE FALLBACK ANYWHERE
//! HERE", AND THAT IS NO LONGER TRUE.** The claim was that the launch count is
//! `WindowTable::spans(mask).len()` and nothing else — `1` for a window P4
//! seated, `0` for a window this composition has no rows for, `r` for the one
//! it withdrew — so every other entry on the menu was a difference in COST the
//! walk could stay ignorant of. That held only while every entry a shell
//! served was served by re-running the same dispatch over a smaller rectangle,
//! and two entries are not:
//!
//! - **`Copy`** gathers the window's scattered rows into one rectangle, runs
//!   the SAME nodes over it once, and scatters the answers back. The bytes it
//!   computes are the bytes a split computes, and the gate for it is
//!   bit-identity against one. Asked of the SHELL
//!   ([`fallback::Serve::copies`](crate::fire::fallback::Serve::copies)),
//!   because paying a copy needs a scratch rectangle the arena does not carve
//!   and a row-movement kernel `kernels` publishes no entry for; a backend
//!   with neither answers `false` and gets the split it always got.
//! - **`Grouped`** is ONE launch handed the whole interval list, which the
//!   kernel walks itself (the SGMV shape — rows contiguous within a segment,
//!   the weight side already runtime-indexed). Asked of the TABLE
//!   ([`fallback::grouped`](crate::fire::fallback::grouped)), because whether
//!   a kernel takes a segment list is what `DeviceProfile::grouped` carries
//!   into the bake.
//!
//! Both collapse the trip count to 1 where the span count is `r`, and no
//! reading of the spans recovers that — hence the branch. `Grouped` wins the
//! tie: it moves no bytes, so a region that could be either is never gathered.
//! Both are asked ONLY of a region this fire already found in pieces, so the
//! scan stays on the path P4 exists to make rare, and a fire of an artifact
//! that seated everything never touches the table at all.
//!
//! It is worth being plain about what the branch costs, because the old
//! sentence was a design property and not an accident. A walk that reads the
//! fallback table is a walk whose trip count depends on something other than
//! the descriptor, so "captured ≡ eager" (decision #11) now rests on the table
//! being the same in both passes rather than on there being nothing to
//! disagree about. It is — `CompiledModel` is immutable and both passes are handed the
//! same one — and a shell's own window table reads the same answer from the
//! same place (`driver_cuda::window::Windows::of`), so a disagreement is a
//! panic there and not a wrong window here. The alternative was to let the
//! COMPILER emit a one-span answer, which would have meant `CompiledModel` knowing
//! what a segment list is; the branch is the smaller of the two.
//!
//! # The phase filter, and why it is not a second walk
//!
//! [`walk()`] runs every region of the template. A shell that CAPTURES has to
//! run the two phases at two different instants — prepare is host work
//! (`std::vector`s, work estimation, a pageable upload) and host work inside
//! `cudaStreamBeginCapture` is either refused or, worse, recorded as nothing —
//! so [`walk_phases()`] takes which phases' NODES to dispatch and the shell
//! calls it twice: prepare on the open stream, capture inside the capture.
//!
//! It filters the DISPATCH and never the structure: every region is still
//! announced to the sink, in template order, in both passes. That is not a
//! detail — a sink counts regions to know which window it is in
//! (`driver_cuda::window::Cursor`), and a filter that skipped the announcement
//! would renumber every region after the first prepare one and hand the whole
//! capture somebody else's rows.
//!
//! # The conditional bracket, and why an eager sink may ignore it
//!
//! A region P3 put behind a conditional node is announced with
//! [`Sink::cond_begin`] before its first node and [`Sink::cond_end`] after its
//! last, and a SWITCH group's arms are announced with [`Sink::cond_arm`] in
//! between. **The zero-row rule below is unchanged by all of it.** A
//! conditional decides exactly what the row count already decides — whether
//! this composition includes the behavior — so an eager sink that no-ops all
//! three runs the same nodes over the same rows and gets the same numbers.
//! What a RECORDING sink must do with them is not optional: a graph outlives
//! the fire that recorded it, so the decision has to be a node in the graph
//! rather than a branch the recorder took.
//!
//! # The event points, and what they do NOT do
//!
//! P6 stamps three fields on a region — the stream it belongs to, the events
//! its stream waits on, the events its stream records — and this loop hands
//! all three to the [`Sink`] in place. **It does not reorder anything.** The
//! regions still run front to back, every node still runs once, and the
//! zero-row and collective rules above are untouched: a fork says where the
//! next launch lands, not when. That is what makes eager mode's no-op sink the
//! SERIALIZATION of P6's DAG rather than a different program (see
//! [`EagerSink`](crate::fire::EagerSink)).
//!
//! # Why the plan is an argument
//!
//! `Dispatch::exec` takes a `&Node`, and `CompiledModel` does not carry the nodes — it
//! carries RANGES of them (`Region::nodes` indexes `Trace::nodes`). Design §2
//! lists the static tables P8 will copy out of the plan, and nodes are not
//! among them, deliberately: they are the plan. So the caller holds the `Trace`
//! beside the `CompiledModel`, which is the arrangement `model_compiler::compiled`'s own
//! docs describe, and [`Fault::NoSuchNode`] is what says the two were not
//! baked from each other.

use kernels::Dispatch;
use model_compiler::{CompiledModel, Lowering, Phase};
use model_ir::{Operation, Trace};

use crate::Result;
use crate::fire::Fault;
use crate::fire::compose::MaskSpan;
use crate::fire::descriptor::FireDescriptor;
use crate::fire::fallback::Serve;
use crate::fire::sink::Sink;

/// Walk one fire.
///
/// The regions run in `CompiledModel::template` order and every one of them is
/// announced to the sink, whether or not this fire has rows for it — the
/// structure is composition-independent, which is exactly the property that
/// lets one recorded graph serve every composition (design §5).
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
) -> Result<()> {
    walk_phases(trace, compiled, descriptor, dispatch, sink, Phases::All)
}

/// Which phases' nodes a walk dispatches.
///
/// **THE STRUCTURE IS NOT FILTERED, ONLY THE DISPATCH.** Every region is
/// announced to the sink under every setting, so a region's number means the
/// same thing in a prepare pass, a capture pass and a whole walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phases {
    /// Both, which is what [`walk()`] means: one pass, the whole fire.
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

/// Walk one fire, dispatching only the nodes of `phases`.
///
/// [`walk()`] is this at [`Phases::All`], and a shell that captures is the
/// reason the parameter exists: the prepare regions must run on an open
/// stream and the capture regions inside `cudaStreamBeginCapture`, which is
/// two instants and therefore two calls. The phase-order check below is
/// unchanged by the filter — it is a statement about the TEMPLATE, and a
/// template that would read a slot before writing it is refused in either
/// pass.
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
    // One buffer for the whole walk, refilled per region. `spans_into` is what
    // keeps the split from costing an allocation per region in front of a
    // launch that costs tens of microseconds.
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

        // The region's window, cut into the intervals it actually covers.
        //
        // ONE ENTRY IS THE CASE P4 EXISTS TO PRODUCE, and it is the case
        // everything below reads as if it were the only one: every node in a
        // region has the same mask — that equality is what defines the run
        // (P2) — so they share a window and read one count. What P4 could not
        // seat covers SEVERAL intervals, and `Fallback::Split { r }` is the
        // compiler's answer for it: the nodes run once per interval, each over
        // its own pointer and extent, and the intervals' rows add up to
        // `rows_of(mask)` by construction (`WindowTable::spans`). An empty
        // window is no intervals at all, and the loop below still turns once —
        // a zero-row pass, which is what dispatches a collective in a
        // composition that has no rows for it.
        descriptor.spans_into(&region.mask, &mut runs);

        // Rule 4's branch, and the whole of it. A region P4 answered
        // `Fallback::Grouped` for is ONE launch over the intervals rather than
        // one launch per interval — the kernel takes the segment list and
        // walks it — so the trip count is 1 where the span count is `r`, and
        // the shell resolves that launch at the UNION of the runs (its own
        // window table cuts the same one window, off the same table, so the
        // two cannot disagree about how many there are).
        //
        // ASKED ONLY WHEN THE WINDOW IS ACTUALLY IN PIECES. One span and one
        // launch are the same launch whatever the table says, and an empty
        // window is the zero-row pass below either way; so the scan happens on
        // the path P4 exists to make rare, and a fire of an artifact that
        // seated everything never touches the table at all.
        let grouped = runs.len() > 1 && crate::fire::fallback::grouped(compiled, region.nodes.clone());

        // P3's answer, and the two instants a recording sink needs it at.
        //
        // **SEMANTICALLY THIS CHANGES NOTHING**, which is what "conditionals
        // are an optimization, not the semantics" means: an eager walk of a
        // conditional region decides by the same zero-row rule below, so
        // `EagerSink`'s no-op `cond_begin` is the correct implementation and
        // not a stub. A RECORDING sink is the one that must bracket, because
        // the graph it writes outlives the fire that wrote it.
        //
        // A SWITCH GROUP IS A RUN OF REGIONS AND THE TABLE IS FLAT. Each arm
        // carries which arm it is and how many the group has
        // (`Lowering::Switch`), so the group opens at arm 0 and closes at the
        // last one and the walk needs no second table and no state between
        // regions — which is the same reason `Region::open` is a field rather
        // than a group object (build log 24 (c)).
        let (open, arm, close) = match region.lowering {
            Lowering::AlwaysLaunch => (false, None, false),
            Lowering::If => (true, None, true),
            Lowering::Switch { arm, arms, .. } => (arm == 0, Some(arm), arm + 1 == arms),
        };

        // P6's event points, in the one order that means what they say
        // (`model_compiler::stream`): the region's stream waits for whatever
        // it was told to wait for, THEN opens the fork the arms behind it will
        // wait on, THEN runs. A plan P6 found nothing in has all three of
        // these empty and the loop is the straight line it always was.
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

        // RULE 4'S ONE BRANCH (see the header). A region this fire found in
        // pieces that the shell says it copies runs ONCE, over a rectangle
        // the shell gathered the pieces into; every other region runs the
        // loop rule 4 always ran. The question is asked after
        // `region_begin`, because that is what tells a shell's cursor which
        // region is being asked about — and only of a region actually in
        // pieces, so a plan P4 seated whole never reaches a backend's answer
        // at all.
        //
        // NOT IN A PREPARE PASS. A gather is a launch and a prepare pass is
        // host work on an open stream; the same filter that holds the nodes
        // back holds these back, for the same reason. The prepare region's
        // own copy is not a launch — a plan builder over a gathered window
        // simply builds ONE schedule over the union — so it needs nothing
        // here beyond `launches == 1`, which the line below gives it.
        let copy = !grouped && runs.len() > 1 && dispatch.copies(region);
        if copy && phases.admits(region.phase) {
            dispatch.gather(region)?;
        }

        // The launches. `max(1)` is the empty window: it turns once, at zero
        // rows, so the collective rule below still sees every node of every
        // region exactly as it did when a window was one span or none.
        //
        // A COLLECTIVE IN A SPLIT REGION JOINS THE RENDEZVOUS ONCE PER RUN,
        // and that is the rule holding rather than bending. Decision #5 is
        // about call ORDER matching across ranks, and the run count is a
        // function of the class table — which the engine replicates
        // identically across ranks — so every rank issues the same number of
        // them in the same order. What would break the rendezvous is a rank
        // deciding the count from something local, and nothing here does. A
        // COPIED or GROUPED region issues ONE, and that is the same argument:
        // both read the artifact's table and this fire's class table, both of
        // them replicated, and never anything about this rank.
        let once = grouped || copy;
        let launches = if once { 1 } else { runs.len().max(1) };
        for launch in 0..launches {
            sink.run(launch as u32, launches as u32);
            // One launch stands over ALL the window's rows — the gather made
            // that true for a copy, the segment list for a grouped region —
            // so the zero-row skip below reads their sum and not the first
            // interval's count.
            let rows = if once {
                runs.iter().map(|span| span.rows).sum()
            } else {
                runs.get(launch).map_or(0, |span| span.rows)
            };

            for node in region.nodes.clone() {
                // Resolved before the filter, so that a template naming a node
                // the plan lacks is the same refusal in a prepare pass, a
                // capture pass and a whole walk. A filter that changed which
                // templates are legal would be a second walk wearing this
                // one's name.
                let Some(node) = trace.nodes.get(node as usize) else {
                    return Err(Fault::NoSuchNode {
                        node,
                        nodes: trace.nodes.len(),
                    }
                    .into());
                };
                if !phases.admits(region.phase) {
                    continue;
                }
                let collective = matches!(node.op, Operation::Collective(_));
                if rows == 0 && !collective {
                    continue;
                }
                dispatch.exec(node)?;
            }
        }

        // The copy's other half, behind the last node and inside the same
        // brackets: the gathered rectangle's rows go back to the fire rows
        // they were read from. Rows the window does not cover are untouched,
        // which is what keeps a copy one consumer's slow path.
        if copy && phases.admits(region.phase) {
            dispatch.scatter(region)?;
        }

        if close {
            sink.cond_end();
        }
        // The join half: the arm records its exit on its own stream, after its
        // last launch, and the region after the group waits on it above.
        if let Some(event) = region.close {
            sink.fork(event);
        }
        sink.region_end(region);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::compose::{ClassWindow, Lane, WindowTable, compose};
    use crate::fire::fixture::{Build, Event, MockDispatch, Recorder, fact};
    use crate::fire::sink::EagerSink;
    use crate::{Error, fire::Fault};
    use kernels::KernelError;
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_ir::Guard;

    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    /// Design §0's diagram, with the prepare node the real catalog plans put
    /// at the front: plan build, shared producer, the split attention pair,
    /// shared consumer.
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

    /// A plan whose collective is GUARDED — legal, and the case decision #5 is
    /// about: node 2 runs in one class and must be dispatched even in a fire
    /// that has no rows for it.
    fn with_a_collective() -> Build {
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Guard::Always); // node 0
        let g = b.all_gather(q, 4, fact(0)); // node 1 — collective, decode only
        let p = b.op(q, 4, Guard::not(fact(0))); // node 2 — prefill window
        let o = b.merge(&[(g, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always); // node 3
        b.out(y);
        b
    }

    fn fire(compiled: &CompiledModel, lanes: &[Lane]) -> FireDescriptor {
        FireDescriptor::of(&compose(compiled, &budget(), lanes).expect("composes"))
    }

    #[test]
    fn a_mixed_fire_runs_every_node_once_in_program_order() {
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 7), Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("a mixed fire walks");

        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 3, 4]);
        assert_eq!(
            dispatch.names(),
            vec![
                "attention.plan_decode",
                "elementwise.rmsnorm_no_scale",
                "attention.decode",
                "elementwise.rmsnorm_no_scale",
                "elementwise.rmsnorm_no_scale",
            ],
        );
    }

    #[test]
    fn an_all_decode_fire_skips_the_prefill_window_and_nothing_else() {
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(1, 1), Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("an all-decode fire walks");
        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 4], "node 3 has no rows");

        // And the mirror: all prefill skips the decode attention — AND the
        // prepare node that builds its plan. Node 0's guard is `Always`, but
        // the only thing that reads its struct is node 2, so the backward
        // demand walk narrows its mask to the decode class alone; a fire with
        // no decode rows has no plan to build, and skipping the empty prepare
        // window is design §5's step 4 verbatim.
        let descriptor = fire(&compiled, &[Lane::new(0, 6)]);
        let mut dispatch = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("an all-prefill fire walks");
        assert_eq!(
            dispatch.nodes(),
            vec![1, 3, 4],
            "nodes 0 and 2 have no rows"
        );
    }

    #[test]
    fn a_collective_runs_even_when_its_window_is_empty() {
        // NCCL matches by call order: a rank that elides one deadlocks the
        // ranks that did not, or pairs its next collective with their current
        // one. So node 1 runs in a fire with no decode rows at all.
        let b = with_a_collective();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 4)]);

        let mut dispatch = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("a prefill-only fire walks");
        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 3]);
        assert_eq!(dispatch.seen[1], (1, "collective.all_gather"));

        // Even a fire with no lanes at all: an empty batch still joins the
        // rendezvous, with a count of zero.
        let descriptor = fire(&compiled, &[]);
        let mut dispatch = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("an empty fire walks");
        assert_eq!(dispatch.nodes(), vec![1]);
    }

    #[test]
    fn the_sink_sees_every_region_including_the_ones_with_no_rows() {
        // The structure is composition-independent — that is the property
        // that lets ONE recorded graph serve every composition — so a region
        // whose window is empty is still opened and closed.
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.trace);
        let mut sink = Recorder::default();
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut sink).expect("walks");

        let expected: Vec<Event> = compiled
            .template()
            .iter()
            .flat_map(|r| [Event::Begin(r.nodes.start), Event::End(r.nodes.start)])
            .collect();
        assert_eq!(sink.events, expected);
        // v1 lowers everything always-launch, so no conditional event fires.
        assert!(
            !sink
                .events
                .iter()
                .any(|e| matches!(e, Event::CondBegin | Event::CondArm(_) | Event::CondEnd))
        );
        // And one stream means no event nodes.
        assert!(
            !sink
                .events
                .iter()
                .any(|e| matches!(e, Event::Fork(_) | Event::Join(_)))
        );
    }

    #[test]
    fn prepare_stands_before_capture_and_a_template_that_does_not_is_refused() {
        let b = diagram();
        let mut compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 2), Lane::new(1, 1)]);

        // The compiler's own output already satisfies it: the plan build is
        // region 0.
        assert_eq!(compiled.template()[0].phase, Phase::Prepare);
        assert!(
            compiled.template()[1..]
                .iter()
                .all(|r| r.phase == Phase::Capture)
        );

        // Swap the first two regions and the walk refuses rather than
        // repairing: a prepare op writes a descriptor slot the launch reads,
        // so this order reads it before it is written.
        compiled.regions.swap(0, 1);
        let mut dispatch = MockDispatch::new(&b.trace);
        assert_eq!(
            walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Fire(Fault::PrepareAfterCapture { region: 1 })),
        );
        // It refused where it found it, having already run what came before —
        // eager mode has no transaction, and the caller's answer is a poisoned
        // fire, not a rollback.
        assert_eq!(dispatch.nodes(), vec![1]);
    }

    #[test]
    fn the_phase_filter_splits_one_walk_into_two_instants_and_loses_no_region() {
        // THE CAPTURE SPLIT, on the mock plane. A shell that records runs the
        // prepare regions on an open stream and the capture regions inside
        // `cudaStreamBeginCapture`; the two passes together must dispatch
        // exactly what one whole walk does, in the same order, and — because a
        // sink counts regions to know its window — must announce the same
        // regions in both.
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 7), Lane::new(1, 1)]);

        let mut whole = MockDispatch::new(&b.trace);
        walk(&b.trace, &compiled, &descriptor, &mut whole, &mut EagerSink).expect("walks");

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

    #[test]
    fn a_forked_template_brackets_its_arms_with_the_events_the_compiler_baked() {
        // THE ONE THING THE WALK ADDED FOR P6, and the order is the whole of
        // it: a region waits, then opens, then runs, then closes. The
        // `diagram` fixture's two windows are a fork group whose arms are too
        // cheap for the default gate, so the profile here lowers the floor
        // rather than pretending a norm costs what an attention does.
        let b = diagram();
        let profile = DeviceProfile {
            fork_floor_us: 1.0,
            ..DeviceProfile::default()
        };
        let compiled = compile(&b.trace, &budget(), &profile).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(0, 7), Lane::new(1, 1)]);

        // Region 2 is the decode window and region 3 the prefill one: the
        // group's main arm is 2 and 3 is what left the main stream.
        assert_eq!(compiled.template()[2].stream, 0);
        assert_eq!(compiled.template()[3].stream, 1);
        let enter = compiled.template()[2].open.expect("the main arm opens it");
        let exit = compiled.template()[3].close.expect("the arm closes it");

        let mut dispatch = MockDispatch::new(&b.trace);
        let mut sink = Recorder::default();
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut sink).expect("walks");

        assert_eq!(
            sink.events,
            vec![
                Event::Begin(0),
                Event::End(0),
                Event::Begin(1),
                Event::End(1),
                Event::Begin(2),
                Event::Fork(enter.0),
                Event::End(2),
                Event::Begin(3),
                Event::Join(enter.0),
                Event::Fork(exit.0),
                Event::End(3),
                Event::Begin(4),
                Event::Join(exit.0),
                Event::End(4),
            ],
        );
        // And the nodes are the same nodes, in the same order: a fork says
        // where the next launch lands, never when.
        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 3, 4]);
    }

    /// A profile that takes any windowed region: what it is for is exercising
    /// the bracket, since P3 at the default profile chooses one region in the
    /// whole catalog and none at all in a five-node fixture.
    fn conditionalizing() -> DeviceProfile {
        DeviceProfile {
            fat_region_us: 0.0,
            cond_fixed_us: 0.5,
            cond_per_arm_us: 0.0,
            side_streams: 0,
            ..DeviceProfile::default()
        }
    }

    #[test]
    fn a_conditional_region_is_bracketed_and_runs_the_same_nodes_it_always_did() {
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &conditionalizing()).expect("bakes");
        // The two windows took an `If` each; the shared trunk did not, because
        // its mask holds every class and a guard around it is never false.
        assert_eq!(compiled.template()[2].lowering, Lowering::If);
        assert_eq!(compiled.template()[3].lowering, Lowering::If);
        assert_eq!(compiled.template()[4].lowering, Lowering::AlwaysLaunch);

        let descriptor = fire(&compiled, &[Lane::new(0, 7), Lane::new(1, 1)]);
        let mut dispatch = MockDispatch::new(&b.trace);
        let mut sink = Recorder::default();
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut sink).expect("walks");

        assert_eq!(
            sink.events,
            vec![
                Event::Begin(0),
                Event::End(0),
                Event::Begin(1),
                Event::End(1),
                Event::Begin(2),
                Event::CondBegin,
                Event::CondEnd,
                Event::End(2),
                Event::Begin(3),
                Event::CondBegin,
                Event::CondEnd,
                Event::End(3),
                Event::Begin(4),
                Event::End(4),
            ],
        );
        // **AND THE DISPATCH IS UNMOVED**, which is design §4's whole claim:
        // the same five nodes in the same order as the always-launch walk
        // above, because the zero-row rule already decided what the bracket
        // announces.
        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn a_switch_group_opens_once_names_every_arm_and_closes_once() {
        // A one-lane deployment: the only shape in which a merge's arms are
        // exclusive at FIRE granularity, which is what a switch node needs
        // (`model_compiler::lowering`).
        let b = diagram();
        let budget = Budget::new(1, 64);
        let compiled = compile(&b.trace, &budget, &conditionalizing()).expect("bakes");
        assert!(matches!(
            compiled.template()[2].lowering,
            Lowering::Switch { arm: 0, arms: 2, .. }
        ));
        assert!(matches!(
            compiled.template()[3].lowering,
            Lowering::Switch { arm: 1, arms: 2, .. }
        ));

        let descriptor =
            FireDescriptor::of(&compose(&compiled, &budget, &[Lane::new(0, 7)]).expect("composes"));
        let mut dispatch = MockDispatch::new(&b.trace);
        let mut sink = Recorder::default();
        walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut sink).expect("walks");

        assert_eq!(
            sink.events,
            vec![
                Event::Begin(0),
                Event::End(0),
                Event::Begin(1),
                Event::End(1),
                // One bracket around the RUN, not one per arm.
                Event::Begin(2),
                Event::CondBegin,
                Event::CondArm(0),
                Event::End(2),
                Event::Begin(3),
                Event::CondArm(1),
                Event::CondEnd,
                Event::End(3),
                Event::Begin(4),
                Event::End(4),
            ],
        );
        // This fire is prefill-only, so the decode arm has no rows and the
        // eager walk skips its node — the same answer the switch would give.
        // Node 0 goes with it: the decode plan build is narrowed by demand to
        // the classes that read its struct (build log 7).
        assert_eq!(dispatch.nodes(), vec![1, 3, 4]);
    }

    #[test]
    fn eager_is_the_serialization_of_the_dag_and_dispatches_what_one_stream_would() {
        // The claim `EagerSink`'s doc makes, asserted: turning the streams on
        // changes no node, no order and no skip — only which stream a shell
        // that reads `Region::stream` would put each launch on.
        let b = diagram();
        let off = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let on = compile(
            &b.trace,
            &budget(),
            &DeviceProfile {
                fork_floor_us: 1.0,
                ..DeviceProfile::default()
            },
        )
        .expect("bakes");
        assert!(on.streams.events > 0, "the on arm actually forked");

        for lanes in [
            vec![Lane::new(0, 7), Lane::new(1, 1)],
            vec![Lane::new(1, 1)],
            vec![Lane::new(0, 4)],
            Vec::new(),
        ] {
            let mut a = MockDispatch::new(&b.trace);
            walk(
                &b.trace,
                &off,
                &fire(&off, &lanes),
                &mut a,
                &mut EagerSink,
            )
            .expect("walks");
            let mut c = MockDispatch::new(&b.trace);
            walk(&b.trace, &on, &fire(&on, &lanes), &mut c, &mut EagerSink).expect("walks");
            assert_eq!(a.nodes(), c.nodes(), "lanes {lanes:?}");
            assert_eq!(a.names(), c.names(), "lanes {lanes:?}");
        }
    }

    #[test]
    fn a_descriptor_that_is_not_this_artifact_s_is_refused() {
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let mut descriptor = fire(&compiled, &[Lane::new(1, 1)]);
        descriptor.classes = WindowTable::new(vec![ClassWindow::default(); 3]);

        let mut dispatch = MockDispatch::new(&b.trace);
        assert_eq!(
            walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Fire(Fault::ClassTable {
                descriptor: 3,
                compiled: 2,
            })),
        );
        assert!(dispatch.seen.is_empty(), "it refused before it walked");
    }

    #[test]
    fn a_template_that_names_a_node_the_plan_lacks_is_refused() {
        let b = diagram();
        let mut compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(1, 1)]);
        let last = compiled.regions.len() - 1;
        compiled.regions[last].nodes.end += 2;

        let mut dispatch = MockDispatch::new(&b.trace);
        assert_eq!(
            walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Fire(Fault::NoSuchNode { node: 5, nodes: 5 })),
        );
    }

    #[test]
    fn a_backend_refusal_stays_a_backend_refusal() {
        // The two error kinds do not mix: `KernelError` is about the device
        // and arrives whole, rather than being restated as a fire fault.
        let b = diagram();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&compiled, &[Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.trace);
        dispatch.refuse = Some("attention.decode");
        assert_eq!(
            walk(&b.trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Kernel(KernelError::Unsupported {
                op: "attention.decode",
            })),
        );
        assert_eq!(dispatch.nodes(), vec![0, 1], "it stopped where it failed");
    }
}
