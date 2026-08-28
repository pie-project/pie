//! `Fallback::Grouped`, in the walk: the launch count drops from `r` to one,
//! and nothing else about the fire moves.
//!
//! # What this file is for
//!
//! Design §3's fallback menu has four entries and `driver::fire::walk` served
//! one of them. `Fallback::Split { r }` is the loop turning `r` times, which
//! needs no branch and no cooperation from a kernel; `Fallback::Copy` is
//! served as a split and owes a copy; `Fallback::View` is unbuilt. The fourth,
//! `Fallback::Grouped`, is the one tart expects to dominate all three, and it
//! is the one that cannot be served by turning the same loop: it is ONE launch
//! handed the whole interval list, which the kernel walks itself. Its trip
//! count is 1 where the span count is `r`, so the walk has to ASK.
//!
//! So rule 4 grew a branch — the first branch on a fallback kind anywhere in
//! that loop — and this is what says the branch does what it claims:
//!
//! - **the two bakes are the same row order.** That is the premise of the
//!   whole comparison, and it is why `DeviceProfile::grouped` (what a shell
//!   can serve) and `DeviceProfile::grouped` (which is also what makes that consumer the cheap one to lose) are
//!   two lists rather than one. If they moved together, the split arm and the
//!   grouped arm would be two different artifacts and the A/B would be worth
//!   nothing;
//! - **the split arm dispatches each correction node `r` times** and announces
//!   `r` runs, which is the shell's cue to resolve `r` windows;
//! - **the grouped arm dispatches each of them ONCE** and announces one run;
//! - **every other node in the plan is dispatched identically** in the two
//!   arms. A grouped answer for one consumer that quietly changed another
//!   consumer's launch count would be a different fire wearing the same name.
//!
//! # The scaffold, and why the file has to use one
//!
//! On this catalog the LoRA correction wins the C1P competition: it is seated,
//! and a seated consumer has no fallback row to answer. Which of two
//! same-sized masks loses is decided by `BTreeMap` lexicographic order inside
//! `layout::insertion_order` — `{4,5,6,7}` loses to `{2,3,6,7}` because
//! `4 > 2` — and that tie-break is an acknowledged latent bug with a
//! cost-weighted replacement intended and unwritten. `DeviceProfile::
//! should lose instead. It is empty by default, it is not the cost model, and
//! the assertion below that the DEFAULT bake still seats the correction is
//! what keeps this file from quietly becoming a claim that it is.
//!
//! NO GPU. `EagerSink`'s sibling and a backend that runs nothing — the same
//! walk the CUDA shell records, which is decision #11's whole point.

use std::collections::HashMap;

use driver::fire::{EventId, FireDescriptor, Lane, Sink, compose, fallback, walk};
use kernels::error::KernelError;
use kernels::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear,
};
use model_compiler::{
    Baked, Budgets, DeviceProfile, Fallback, FamilyCosts, Lowering, Region, compile,
};
use model_dsl::Platform;
use model_ir::{
    Attention, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation, Plan,
};

/// The SKU the recon measured, and the one whose adapter window fragments into
/// four intervals once the scaffold withdraws it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The op the correction is, by the name both profile lists are keyed on.
const CORRECTION: &str = "linear.lora_correct";

/// A deployment's ceilings, at an adapter capacity the catalog can seat.
///
/// **`max_adapters: 8` IS LOAD-BEARING.** The catalog sweeps that ask for 32
/// compile nothing — no text seats more than eight — and every loop body in
/// them is skipped. This file would have the same disease, so it names a
/// number the text can seat and asserts non-vacuity besides.
fn budgets() -> Budgets {
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 8,
    }
}

fn plan() -> Plan {
    let (_, _, trace, _) = model::catalog()
        .into_iter()
        .find(|(sku, ..)| *sku == SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"));
    trace(Platform::Cuda)
}

fn bake(plan: &Plan, profile: &DeviceProfile) -> Baked {
    compile(plan, &budgets(), profile).unwrap_or_else(|why| panic!("`{SKU}` bakes: {why:?}"))
}

/// The three profiles this file compares: the artifact as it ships, and the
/// two arms of the measurement.
fn shipped() -> DeviceProfile {
    DeviceProfile::default()
}

/// The split arm: the SAME withdrawal, served the old way.
///
/// **THE TWO ARMS MUST SHARE A ROW ORDER OR THE COMPARISON IS WORTHLESS**, and
/// with the withdrawal chosen by cost (`model_compiler::layout::choose`) there
/// is only one honest way to move it without also naming the op groupable:
/// tell the cost model the correction is cheap. Twenty-four linear nodes at
/// 1 us lose to the score window's six attention nodes at 60, exactly as
/// twenty-four at 40 us lose once `grouped` discounts them — same mask
/// withdrawn, same frontier, and the only thing left differing is the ANSWER,
/// which is what this file is about.
fn split_arm() -> DeviceProfile {
    let base = DeviceProfile::default();
    DeviceProfile {
        family_us: FamilyCosts {
            linear: 1.0,
            ..base.family_us
        },
        ..base
    }
}

fn grouped_arm() -> DeviceProfile {
    DeviceProfile {
        grouped: vec![CORRECTION.to_string()],
        ..DeviceProfile::default()
    }
}

/// Which nodes of the plan are corrections.
fn corrections(plan: &Plan) -> Vec<u32> {
    plan.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op.name() == CORRECTION)
        .map(|(at, _)| at as u32)
        .collect()
}

/// The class order the artifact ships, over every class.
fn frontier(baked: &Baked) -> Vec<u8> {
    let mut every = model_ir::ClassSet::default();
    for class in 0..baked.classes.classes.len() {
        every.insert(class);
    }
    baked.order.class_order(&every, None)
}

/// One lane per class of the artifact — the composition that presents every
/// behaviour at once, and therefore the one that fragments the most.
///
/// A prefill lane carries three rows and a decode lane one, off the class's
/// own `qo_one` bit, because a class whose lanes are single-row IS the decode
/// class and a fire that gave it three would be composing a lane the word does
/// not describe.
fn one_lane_per_class(baked: &Baked) -> Vec<Lane> {
    baked
        .classes
        .classes
        .iter()
        .map(|class| {
            let decode = class.word() & 1 == 1;
            Lane::new(class.word(), if decode { 1 } else { 3 })
        })
        .collect()
}

fn fire(baked: &Baked, lanes: &[Lane]) -> FireDescriptor {
    FireDescriptor::of(&compose(baked, &budgets(), lanes).expect("the lanes compose"))
}

// ── the mock backend ─────────────────────────────────────────────────────

/// A backend that runs nothing and remembers which NODE it was handed, by the
/// address of the op payload inside the plan's node vector: a `Dispatch*`
/// method is given the op and not the node, and a count of op names cannot
/// tell a node that ran twice from two nodes that ran once.
struct MockDispatch {
    at: HashMap<usize, u32>,
    seen: Vec<u32>,
}

impl MockDispatch {
    fn new(plan: &Plan) -> MockDispatch {
        MockDispatch {
            at: plan
                .nodes
                .iter()
                .enumerate()
                .map(|(j, node)| (payload(&node.op), j as u32))
                .collect(),
            seen: Vec::new(),
        }
    }

    fn counts(&self) -> HashMap<u32, usize> {
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for &node in &self.seen {
            *counts.entry(node).or_default() += 1;
        }
        counts
    }

    fn note<T: Operands>(&mut self, op: &T) -> Result<(), KernelError> {
        let node = *self
            .at
            .get(&address(op))
            .expect("every dispatched op is a node of the plan the mock was built from");
        self.seen.push(node);
        Ok(())
    }
}

fn address<T>(value: &T) -> usize {
    std::ptr::from_ref(value).cast::<()>() as usize
}

fn payload(op: &Operation) -> usize {
    match op {
        Operation::Attention(op) => address(op),
        Operation::Linear(op) => address(op),
        Operation::Elementwise(op) => address(op),
        Operation::Layout(op) => address(op),
        Operation::Collective(op) => address(op),
        Operation::CustomCuda(op) => address(op),
    }
}

/// The split/grouped choice is the walk's own, so this mock takes `Serve`'s
/// defaults: it never copies, and a `Grouped` region is served by the launch
/// count alone.
impl driver::fire::fallback::Serve for MockDispatch {}

impl DispatchAttention for MockDispatch {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.note(op)
    }
}
impl DispatchLinear for MockDispatch {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.note(op)
    }
}
impl DispatchElementwise for MockDispatch {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.note(op)
    }
}
impl DispatchLayout for MockDispatch {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.note(op)
    }
}
impl DispatchCollective for MockDispatch {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.note(op)
    }
}
impl DispatchCustomCuda for MockDispatch {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        self.note(op)
    }
}

/// How many runs each region announced — the structure event the shell's
/// cursor turns into a window lookup, and the number a grouped region has to
/// answer `1` for.
#[derive(Default)]
struct Runs {
    per_region: Vec<u32>,
}

impl Sink for Runs {
    fn region_begin(&mut self, _region: &Region) {
        self.per_region.push(0);
    }
    fn region_end(&mut self, _region: &Region) {}
    fn run(&mut self, run: u32, runs: u32) {
        let held = self
            .per_region
            .last_mut()
            .expect("a run is always announced inside a region");
        assert_eq!(*held, run, "runs arrive in order, from zero");
        *held = run + 1;
        assert!(run < runs, "run {run} of {runs}");
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {}
    fn cond_arm(&mut self, _arm: u8) {}
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {}
    fn join(&mut self, _event: EventId) {}
}

/// One walk of one bake over one composition: what each node cost, and what
/// each region announced.
fn walked(plan: &Plan, baked: &Baked, lanes: &[Lane]) -> (HashMap<u32, usize>, Vec<u32>) {
    let descriptor = fire(baked, lanes);
    let mut dispatch = MockDispatch::new(plan);
    let mut runs = Runs::default();
    walk(plan, baked, &descriptor, &mut dispatch, &mut runs).expect("a fragmented fire walks");
    (dispatch.counts(), runs.per_region)
}

// ── the gates ────────────────────────────────────────────────────────────

/// **THE PREMISE, ASSERTED SO IT CANNOT ROT.** The default bake seats the
/// correction — that is why the scaffold exists — and the two arms of the
/// measurement withdraw it onto the SAME row order, which is why the arms can
/// be compared at all.
#[test]
fn the_default_bake_seats_the_correction_and_the_two_arms_withdraw_it_together() {
    let plan = plan();
    let corrections = corrections(&plan);
    assert!(
        !corrections.is_empty(),
        "`{SKU}` states no correction, and then every gate in this file is vacuous",
    );

    let shipped = bake(&plan, &shipped());
    for &node in &corrections {
        assert!(
            fallback::answers(&shipped, node..node + 1).is_empty(),
            "the shipped bake owes node {node} an answer, so the scaffold below is \
             no longer describing a change",
        );
    }
    assert_eq!(
        frontier(&shipped),
        [4, 0, 2, 6, 7, 3, 1, 5],
        "the order the recon measured; a text or a tie-break that moved it moves \
         every number in this file's doc",
    );

    // The two arms, and the one thing they must agree on.
    let split = bake(&plan, &split_arm());
    let grouped = bake(&plan, &grouped_arm());
    assert_eq!(
        frontier(&split),
        [0, 2, 4, 6, 5, 7, 1, 3],
        "withdrawing the correction instead seats everything else",
    );
    assert_eq!(
        frontier(&split),
        frontier(&grouped),
        "the arms differ in the ANSWER and not in the order; if they differ in the \
         order they are two artifacts and the comparison is worthless",
    );

    // And they differ in exactly the way the names say.
    for &node in &corrections {
        let said_split = fallback::answers(&split, node..node + 1);
        let said_grouped = fallback::answers(&grouped, node..node + 1);
        assert!(
            said_split
                .iter()
                .all(|a| matches!(a, Fallback::Copy | Fallback::Split { .. })),
            "node {node} in the split arm answers {said_split:?}",
        );
        assert_eq!(said_grouped, vec![Fallback::Grouped], "node {node}");
        assert!(!fallback::grouped(&split, node..node + 1), "node {node}");
        assert!(fallback::grouped(&grouped, node..node + 1), "node {node}");
    }

    // Nobody else changed hands. The score-capture window the shipped bake
    // withdrew is seated in both arms, which is the other half of "the
    // scaffold moved one constraint".
    let owed = |baked: &Baked| -> Vec<u32> {
        baked
            .fallback
            .rows
            .iter()
            .map(|row| row.node)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect()
    };
    assert_eq!(owed(&split), corrections);
    assert_eq!(owed(&grouped), corrections);
}

/// **THE GATE.** On one composition, the grouped arm dispatches every
/// correction node ONCE where the split arm dispatches it `r` times, and every
/// other node in the plan is dispatched exactly as often in both.
#[test]
fn the_grouped_arm_pays_one_launch_where_the_split_arm_pays_r() {
    let plan = plan();
    let corrections = corrections(&plan);
    assert!(!corrections.is_empty(), "the SKU states corrections");

    let split = bake(&plan, &split_arm());
    let grouped = bake(&plan, &grouped_arm());
    let lanes = one_lane_per_class(&split);
    assert_eq!(lanes.len(), 8, "`{SKU}` resolves eight classes");

    // How many intervals the adapter window actually breaks into in this fire
    // — read off the composition rather than off the table, and asserted to be
    // more than one, because a `r == 1` fire would make both arms one launch
    // and this test green for the wrong reason.
    let descriptor = fire(&split, &lanes);
    let mask = split
        .template()
        .iter()
        .find(|region| region.nodes.clone().any(|node| corrections.contains(&node)))
        .map(|region| region.mask.clone())
        .expect("some region holds a correction");
    let r = descriptor.spans(&mask).len();
    assert!(
        r > 1,
        "this composition leaves the adapter window in one piece, so there is \
         nothing for `Grouped` to do and nothing for this test to compare",
    );

    let (split_counts, split_runs) = walked(&plan, &split, &lanes);
    let (grouped_counts, grouped_runs) = walked(&plan, &grouped, &lanes);

    // The correction nodes: `r` launches against one.
    for &node in &corrections {
        assert_eq!(
            split_counts.get(&node).copied().unwrap_or(0),
            r,
            "the split arm runs node {node} once per interval",
        );
        assert_eq!(
            grouped_counts.get(&node).copied().unwrap_or(0),
            1,
            "the grouped arm runs node {node} ONCE — a grouped answer that was \
             secretly still splitting would read {r} here",
        );
    }

    // And the regions announced what they cost, which is what a shell's
    // cursor reads to know how many windows to resolve.
    let region_of = |baked: &Baked| {
        baked
            .template()
            .iter()
            .position(|region| region.nodes.clone().any(|node| corrections.contains(&node)))
            .expect("some region holds a correction")
    };
    assert_eq!(split_runs[region_of(&split)], r as u32);
    assert_eq!(grouped_runs[region_of(&grouped)], 1);

    // Nothing else moved. Every non-correction node cost the same in both
    // arms — the two bakes are one row order, so the only difference the walk
    // is entitled to make is the one under test.
    let mut compared = 0usize;
    for (node, count) in &split_counts {
        if corrections.contains(node) {
            continue;
        }
        compared += 1;
        assert_eq!(
            grouped_counts.get(node).copied().unwrap_or(0),
            *count,
            "node {node} costs a different number of launches under the grouped bake",
        );
    }
    assert_eq!(split_runs.len(), grouped_runs.len(), "one template, two bakes");
    // SILENT ON PURPOSE: the numbers ride in this message rather than in a
    // print macro, which `driver` denies in its tests.
    assert!(
        compared > 0,
        "the plan has nodes besides its corrections — `{SKU}`, eight classes in one fire: \
         adapter window = {r} intervals, correction launches {r} -> 1 per node over {} nodes",
        corrections.len(),
    );
}

/// **A COMPOSITION THAT LEAVES THE WINDOW WHOLE COSTS THE SAME IN BOTH ARMS.**
/// The branch is on the fallback row, but the TRIP COUNT is still the span
/// count when there is only one span — so a fire that does not fragment is the
/// fire it always was, grouped answer or not.
#[test]
fn a_fire_that_fragments_nothing_is_unchanged_by_the_grouped_answer() {
    let plan = plan();
    let corrections = corrections(&plan);
    let split = bake(&plan, &split_arm());
    let grouped = bake(&plan, &grouped_arm());

    // Two adapted lanes and nothing else: the adapter classes are the only
    // ones present, so their window is one interval however the order runs.
    let adapted: Vec<Lane> = split
        .classes
        .classes
        .iter()
        .filter(|class| class.word() & 0b10 != 0 && class.word() & 0b1000 == 0)
        .map(|class| Lane::new(class.word(), if class.word() & 1 == 1 { 1 } else { 3 }))
        .collect();
    assert_eq!(adapted.len(), 2, "an adapted prefill lane and an adapted decode one");

    let descriptor = fire(&split, &adapted);
    let mask = split
        .template()
        .iter()
        .find(|region| region.nodes.clone().any(|node| corrections.contains(&node)))
        .map(|region| region.mask.clone())
        .expect("some region holds a correction");
    assert_eq!(
        descriptor.spans(&mask).len(),
        1,
        "the premise: this composition does not fragment the adapter window",
    );

    let (split_counts, split_runs) = walked(&plan, &split, &adapted);
    let (grouped_counts, grouped_runs) = walked(&plan, &grouped, &adapted);
    assert_eq!(split_counts, grouped_counts);
    assert_eq!(split_runs, grouped_runs);
    for &node in &corrections {
        assert_eq!(split_counts.get(&node).copied().unwrap_or(0), 1);
    }
    assert!(!corrections.is_empty(), "the SKU states corrections");
}
