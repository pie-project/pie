//! [`walk()`]: one interpreter, two modes (design §6).
//!
//! The whole fire, as a loop: take the regions the compiler baked, in the
//! order it baked them, and for each one hand its structure to a [`Sink`] and
//! its nodes to a [`Dispatch`]. There is no scheduling here, no lowering
//! decision and no allocation — all three happened once, at load, in
//! `model_compiler::compile`. What is left is the part that has to be right
//! 5000 times a second.
//!
//! # The three rules it enforces, and none of them is an optimization
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
//! `Dispatch::exec` takes a `&Node`, and `Baked` does not carry the nodes — it
//! carries RANGES of them (`Region::nodes` indexes `Plan::nodes`). Design §2
//! lists the static tables P8 will copy out of the plan, and nodes are not
//! among them, deliberately: they are the plan. So the caller holds the `Plan`
//! beside the `Baked`, which is the arrangement `model_compiler::baked`'s own
//! docs describe, and [`Fault::NoSuchNode`] is what says the two were not
//! baked from each other.

use kernels::Dispatch;
use model_compiler::{Baked, Lowering, Phase};
use model_ir::{Operation, Plan};

use crate::Result;
use crate::fire::Fault;
use crate::fire::descriptor::FireDescriptor;
use crate::fire::sink::Sink;

/// Walk one fire.
///
/// The regions run in `Baked::template` order and every one of them is
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
pub fn walk<D: Dispatch, S: Sink>(
    plan: &Plan,
    baked: &Baked,
    descriptor: &FireDescriptor,
    dispatch: &mut D,
    sink: &mut S,
) -> Result<()> {
    walk_phases(plan, baked, descriptor, dispatch, sink, Phases::All)
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
pub fn walk_phases<D: Dispatch, S: Sink>(
    plan: &Plan,
    baked: &Baked,
    descriptor: &FireDescriptor,
    dispatch: &mut D,
    sink: &mut S,
    phases: Phases,
) -> Result<()> {
    // A mask indexes the window table by position, so a table of the wrong
    // width does not fail to find a class — it finds another class's rows and
    // runs the fire over them.
    let classes = baked.classes.classes.len();
    if descriptor.classes.len() != classes {
        return Err(Fault::ClassTable {
            descriptor: descriptor.classes.len(),
            baked: classes,
        }
        .into());
    }

    let mut captured = false;
    for (index, region) in baked.template().iter().enumerate() {
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

        // One number for the whole region: every node in it has the same mask
        // — that equality is what defines the run (P2) — so they share a
        // window and read one count.
        let rows = descriptor.rows_of(&region.mask);

        // v1 never takes this branch: P3 constructs `AlwaysLaunch` for every
        // region. It is written now because the day P3 starts choosing, a
        // conditional region has to be BRACKETED for a recording sink or the
        // body lands outside the conditional node — and the arms themselves
        // (`Sink::cond_arm`) arrive with the template structure that names
        // them, which is the same pass. Semantically this changes nothing:
        // an eager walk of a conditional region still decides by the same
        // zero-row rule, which is what "conditionals are an optimization, not
        // the semantics" means.
        let conditional = !matches!(region.lowering, Lowering::AlwaysLaunch);

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
        if conditional {
            sink.cond_begin(&region.lowering);
        }

        for node in region.nodes.clone() {
            // Resolved before the filter, so that a template naming a node
            // the plan lacks is the same refusal in a prepare pass, a capture
            // pass and a whole walk. A filter that changed which templates
            // are legal would be a second walk wearing this one's name.
            let Some(node) = plan.nodes.get(node as usize) else {
                return Err(Fault::NoSuchNode {
                    node,
                    nodes: plan.nodes.len(),
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

        if conditional {
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
    use model_compiler::{Budgets, DeviceProfile, compile};
    use model_ir::Cond;

    fn budgets() -> Budgets {
        Budgets::new(8, 64)
    }

    /// Design §0's diagram, with the prepare node the real catalog plans put
    /// at the front: plan build, shared producer, the split attention pair,
    /// shared consumer.
    fn diagram() -> Build {
        let mut b = Build::new();
        let x = b.input(8);
        let plan = b.prepare(Cond::Always); // node 0 — prepare
        let q = b.op(x, 4, Cond::Always); // node 1
        let d = b.decode(q, plan, fact(0)); // node 2 — decode window
        let p = b.op(q, 4, Cond::not(fact(0))); // node 3 — prefill window
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 4);
        let y = b.op(o, 4, Cond::Always); // node 4
        b.out(y);
        b
    }

    /// A plan whose collective is GUARDED — legal, and the case decision #5 is
    /// about: node 2 runs in one class and must be dispatched even in a fire
    /// that has no rows for it.
    fn with_a_collective() -> Build {
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Cond::Always); // node 0
        let g = b.all_gather(q, 4, fact(0)); // node 1 — collective, decode only
        let p = b.op(q, 4, Cond::not(fact(0))); // node 2 — prefill window
        let o = b.merge(&[(g, fact(0)), (p, Cond::not(fact(0)))], 4);
        let y = b.op(o, 4, Cond::Always); // node 3
        b.out(y);
        b
    }

    fn fire(baked: &Baked, lanes: &[Lane]) -> FireDescriptor {
        FireDescriptor::of(&compose(baked, &budgets(), lanes).expect("composes"))
    }

    #[test]
    fn a_mixed_fire_runs_every_node_once_in_program_order() {
        let b = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(0, 7), Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.plan);
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink)
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
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(1, 1), Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.plan);
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("an all-decode fire walks");
        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 4], "node 3 has no rows");

        // And the mirror: all prefill skips the decode attention — AND the
        // prepare node that builds its plan. Node 0's guard is `Always`, but
        // the only thing that reads its struct is node 2, so the backward
        // demand walk narrows its mask to the decode class alone; a fire with
        // no decode rows has no plan to build, and skipping the empty prepare
        // window is design §5's step 4 verbatim.
        let descriptor = fire(&baked, &[Lane::new(0, 6)]);
        let mut dispatch = MockDispatch::new(&b.plan);
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink)
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
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(0, 4)]);

        let mut dispatch = MockDispatch::new(&b.plan);
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("a prefill-only fire walks");
        assert_eq!(dispatch.nodes(), vec![0, 1, 2, 3]);
        assert_eq!(dispatch.seen[1], (1, "collective.all_gather"));

        // Even a fire with no lanes at all: an empty batch still joins the
        // rendezvous, with a count of zero.
        let descriptor = fire(&baked, &[]);
        let mut dispatch = MockDispatch::new(&b.plan);
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink)
            .expect("an empty fire walks");
        assert_eq!(dispatch.nodes(), vec![1]);
    }

    #[test]
    fn the_sink_sees_every_region_including_the_ones_with_no_rows() {
        // The structure is composition-independent — that is the property
        // that lets ONE recorded graph serve every composition — so a region
        // whose window is empty is still opened and closed.
        let b = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.plan);
        let mut sink = Recorder::default();
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut sink).expect("walks");

        let expected: Vec<Event> = baked
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
        let mut baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(0, 2), Lane::new(1, 1)]);

        // The compiler's own output already satisfies it: the plan build is
        // region 0.
        assert_eq!(baked.template()[0].phase, Phase::Prepare);
        assert!(
            baked.template()[1..]
                .iter()
                .all(|r| r.phase == Phase::Capture)
        );

        // Swap the first two regions and the walk refuses rather than
        // repairing: a prepare op writes a descriptor slot the launch reads,
        // so this order reads it before it is written.
        baked.regions.swap(0, 1);
        let mut dispatch = MockDispatch::new(&b.plan);
        assert_eq!(
            walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink),
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
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(0, 7), Lane::new(1, 1)]);

        let mut whole = MockDispatch::new(&b.plan);
        walk(&b.plan, &baked, &descriptor, &mut whole, &mut EagerSink).expect("walks");

        let mut split = MockDispatch::new(&b.plan);
        let mut structure = (Recorder::default(), Recorder::default());
        walk_phases(
            &b.plan,
            &baked,
            &descriptor,
            &mut split,
            &mut structure.0,
            Phases::Prepare,
        )
        .expect("the prepare pass walks");
        assert_eq!(split.nodes(), vec![0], "the plan build, and nothing else");
        walk_phases(
            &b.plan,
            &baked,
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
            baked.template().len() * 2,
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
        let baked = compile(&b.plan, &budgets(), &profile).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(0, 7), Lane::new(1, 1)]);

        // Region 2 is the decode window and region 3 the prefill one: the
        // group's main arm is 2 and 3 is what left the main stream.
        assert_eq!(baked.template()[2].stream, 0);
        assert_eq!(baked.template()[3].stream, 1);
        let enter = baked.template()[2].open.expect("the main arm opens it");
        let exit = baked.template()[3].close.expect("the arm closes it");

        let mut dispatch = MockDispatch::new(&b.plan);
        let mut sink = Recorder::default();
        walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut sink).expect("walks");

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

    #[test]
    fn eager_is_the_serialization_of_the_dag_and_dispatches_what_one_stream_would() {
        // The claim `EagerSink`'s doc makes, asserted: turning the streams on
        // changes no node, no order and no skip — only which stream a shell
        // that reads `Region::stream` would put each launch on.
        let b = diagram();
        let off = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let on = compile(
            &b.plan,
            &budgets(),
            &DeviceProfile {
                fork_floor_us: 1.0,
                ..DeviceProfile::default()
            },
        )
        .expect("bakes");
        assert!(on.forks.events > 0, "the on arm actually forked");

        for lanes in [
            vec![Lane::new(0, 7), Lane::new(1, 1)],
            vec![Lane::new(1, 1)],
            vec![Lane::new(0, 4)],
            Vec::new(),
        ] {
            let mut a = MockDispatch::new(&b.plan);
            walk(
                &b.plan,
                &off,
                &fire(&off, &lanes),
                &mut a,
                &mut EagerSink,
            )
            .expect("walks");
            let mut c = MockDispatch::new(&b.plan);
            walk(&b.plan, &on, &fire(&on, &lanes), &mut c, &mut EagerSink).expect("walks");
            assert_eq!(a.nodes(), c.nodes(), "lanes {lanes:?}");
            assert_eq!(a.names(), c.names(), "lanes {lanes:?}");
        }
    }

    #[test]
    fn a_descriptor_that_is_not_this_artifact_s_is_refused() {
        let b = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let mut descriptor = fire(&baked, &[Lane::new(1, 1)]);
        descriptor.classes = WindowTable::new(vec![ClassWindow::default(); 3]);

        let mut dispatch = MockDispatch::new(&b.plan);
        assert_eq!(
            walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Fire(Fault::ClassTable {
                descriptor: 3,
                baked: 2,
            })),
        );
        assert!(dispatch.seen.is_empty(), "it refused before it walked");
    }

    #[test]
    fn a_template_that_names_a_node_the_plan_lacks_is_refused() {
        let b = diagram();
        let mut baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(1, 1)]);
        let last = baked.regions.len() - 1;
        baked.regions[last].nodes.end += 2;

        let mut dispatch = MockDispatch::new(&b.plan);
        assert_eq!(
            walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Fire(Fault::NoSuchNode { node: 5, nodes: 5 })),
        );
    }

    #[test]
    fn a_backend_refusal_stays_a_backend_refusal() {
        // The two error kinds do not mix: `KernelError` is about the device
        // and arrives whole, rather than being restated as a fire fault.
        let b = diagram();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let descriptor = fire(&baked, &[Lane::new(1, 1)]);

        let mut dispatch = MockDispatch::new(&b.plan);
        dispatch.refuse = Some("attention.decode");
        assert_eq!(
            walk(&b.plan, &baked, &descriptor, &mut dispatch, &mut EagerSink),
            Err(Error::Kernel(KernelError::Unsupported {
                op: "attention.decode",
            })),
        );
        assert_eq!(dispatch.nodes(), vec![0, 1], "it stopped where it failed");
    }
}
