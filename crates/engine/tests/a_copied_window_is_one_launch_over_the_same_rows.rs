//! `Fallback::Copy`, on the golden path: what the WALK does about a window P4
//! could not seat when the shell says it can gather one.
//!
//! # The bug this pins
//!
//! P4's menu writes two rows per withdrawn node because its cost model is
//! bucket-keyed (`model_compiler::layout`'s `CROSSOVER_ROWS`: a two-way split
//! of a 64-row GEMM measured 1.82x the ideal against a copy's 1.07x, and they
//! converge by 2048). On the catalog's fourteen-point lattice that is
//! **`Fallback::Copy` at ten buckets and `Fallback::Split { r: 3 }` at four**
//! — the copy covers every bucket a decode fire lands in. `engine::fire::walk`
//! served all fourteen as splits and said so in its own rule 4: "there is no
//! branch on the fallback anywhere here". So the ten small buckets paid
//! roughly 1.7x what the table asked for, every fire, silently.
//!
//! # What is asserted, and what is deliberately NOT
//!
//! This file is the STRUCTURE half, with no device in the room:
//!
//! - **one launch, not `r`.** The copied region's nodes are dispatched once,
//!   over the union of the runs, where a split dispatches them once per
//!   interval. That is the whole performance claim, counted;
//! - **the bracket is exactly one gather and one scatter**, in that order,
//!   around the nodes and inside the region — a second pair would be two
//!   gathers of the same rows and a missing one would be answers that never
//!   went back;
//! - **only the copied regions change.** Every region P4 seated runs its
//!   nodes exactly once with copies on and with copies off, and the two walks
//!   dispatch the same node MULTISET — a copy moves rows, it does not move
//!   work;
//! - **the default is still the split.** A backend that says nothing about
//!   `Serve` gets the launch counts this repo has always had, which is what
//!   makes the trait not a breaking change;
//! - **the prepare region is copied too.** qwen3.5's `attention.plan_prefill`
//!   states the same `captures_scores` mask its six `prefill_lse` readers do
//!   and P4 owes it no row of its own — but a consumer standing over the
//!   union must read a schedule carved over the union, so
//!   `fallback::copies` asks the question of the MASK. A build where the
//!   builder split while its readers copied would compute wrong logits and
//!   fault nothing, so it is asserted here rather than hoped for.
//!
//! What is NOT asserted is that the numbers are right — a mock backend
//! computes none. That is `engine-cuda`'s
//! `a_copied_window_and_a_split_one_are_the_same_bytes.rs`, which diffs a
//! copy against a split on real weights.

use std::collections::HashMap;

use engine::fire::{EventId, FireDescriptor, Lane, Serve, Sink, compose, fallback, walk};
use kernels::error::KernelError;
use kernels::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear,
};
use model_compiler::{CompiledModel, Budget, DeviceProfile, Fallback, Lowering, Region, compile};
use model_dsl::Platform;
use model_ir::{
    Attention, ClassSet, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation,
    Trace,
};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// A deployment's ceilings, at an adapter capacity the catalog can seat.
///
/// **`max_adapters: 8` IS LOAD-BEARING** for the same reason it is in
/// `a_fragmented_window_is_a_slow_path_not_a_fault.rs`: at 32 no catalog text
/// compiles and every loop body in this crate's sweeps skips. The fourteen
/// buckets are load-bearing too, and differently — they are what make the
/// menu write a `Copy` row and a `Split` row rather than one entry covering
/// everything, which is the thing this file is about.
fn budget() -> Budget {
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 8,
    }
}

/// A backend that runs nothing, remembers which node it was handed by the
/// address of the op payload inside the plan's node vector, and can be told
/// to claim a row gather it does not have.
///
/// **CLAIMING IS ENOUGH FOR THIS FILE.** `Serve::copies` is what the walk
/// branches on and `gather`/`scatter` are what it brackets with; a mock that
/// answers the first and records the other two exercises every line of the
/// branch. Moving actual bytes is a shell's job and a shell's gate.
struct MockDispatch {
    at: HashMap<usize, u32>,
    seen: Vec<u32>,
    copies: bool,
    /// `(region's first node, "gather" | "scatter")`, in call order.
    moved: Vec<(u32, &'static str)>,
}

impl MockDispatch {
    fn new(trace: &Trace, copies: bool) -> MockDispatch {
        MockDispatch {
            at: trace
                .nodes
                .iter()
                .enumerate()
                .map(|(j, node)| (payload(&node.op), j as u32))
                .collect(),
            seen: Vec::new(),
            copies,
            moved: Vec::new(),
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

impl Serve for MockDispatch {
    /// The whole toggle. A real shell asks P4's table at this fire's bucket
    /// and asks whether its own resolution can re-point the region's
    /// operands; a mock that computes nothing needs neither question.
    fn copies(&self, _region: &Region) -> bool {
        self.copies
    }

    fn gather(&mut self, region: &Region) -> Result<(), KernelError> {
        self.moved.push((region.nodes.start, "gather"));
        Ok(())
    }

    fn scatter(&mut self, region: &Region) -> Result<(), KernelError> {
        self.moved.push((region.nodes.start, "scatter"));
        Ok(())
    }
}

/// A sink that writes down how many runs each region was cut into.
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

fn sku() -> (Trace, CompiledModel) {
    let trace = model::trace_of(SKU).unwrap_or_else(|| panic!("`{SKU}` is in the catalog"));
    let trace = trace(Platform::Cuda);
    let compiled = compile(&trace, &budget(), &DeviceProfile::default())
        .unwrap_or_else(|refusal| panic!("`{SKU}` bakes: {refusal:?}"));
    (trace, compiled)
}

/// The smallest composition that leaves a window in pieces: a plain prefill
/// lane, a capturing prefill lane and a capturing decode lane, with the plain
/// one's rows standing between the other two.
fn fragmenting(compiled: &CompiledModel) -> Vec<Lane> {
    [0usize, 4, 5]
        .iter()
        .map(|&class| Lane::new(compiled.classes.classes[class].word(), 1))
        .collect()
}

/// Walk one fire and hand back `(the sink's run counts, the dispatch)`.
fn fire(trace: &Trace, compiled: &CompiledModel, lanes: &[Lane], copies: bool) -> (Runs, MockDispatch) {
    let composition = compose(compiled, &budget(), lanes).expect("the fire composes");
    let descriptor = FireDescriptor::of(&composition);
    let mut dispatch = MockDispatch::new(trace, copies);
    let mut runs = Runs::default();
    walk(trace, compiled, &descriptor, &mut dispatch, &mut runs).expect("the fire walks");
    (runs, dispatch)
}

#[test]
fn a_copied_window_costs_one_launch_where_a_split_one_costs_its_runs() {
    let (trace, compiled) = sku();
    let lanes = fragmenting(&compiled);

    // NOT VACUOUS, AND CHECKED AGAINST THE ARTIFACT. The whole file is about
    // a table entry that used to be ignored, so the entry has to be there:
    // the bucket a three-row fire lands in must be one the menu wrote a
    // `Copy` row for, and some region must actually come back in pieces.
    let composition = compose(&compiled, &budget(), &lanes).expect("three lanes compose");
    assert_eq!(
        composition.present(),
        [4, 0, 5],
        "class 0 stands between 4 and 5"
    );
    let bucket = budget()
        .buckets
        .iter()
        .position(|&rows| rows == composition.bucket())
        .expect("the fire lands in the lattice") as u32;
    let descriptor = FireDescriptor::of(&composition);

    let fragmented: Vec<usize> = compiled
        .template()
        .iter()
        .enumerate()
        .filter(|(_, region)| descriptor.spans(&region.mask).len() > 1)
        .map(|(at, _)| at)
        .collect();
    assert!(
        !fragmented.is_empty(),
        "no window of this composition is in pieces, so this gate fires an ordinary fire",
    );
    let copied: Vec<usize> = fragmented
        .iter()
        .copied()
        .filter(|&at| fallback::copies(&compiled, &compiled.template()[at].mask, bucket))
        .collect();
    assert_eq!(
        copied, fragmented,
        "at bucket {bucket} the table asks for a copy on some fragmented windows and \
         not others, and this file's premise is that it asks for one on all of them",
    );

    let (split, split_dispatch) = fire(&trace, &compiled, &lanes, false);
    let (copy, copy_dispatch) = fire(&trace, &compiled, &lanes, true);

    // THE PERFORMANCE CLAIM, COUNTED. Every fragmented region falls from its
    // run count to one; every other region is untouched.
    for (at, region) in compiled.template().iter().enumerate() {
        let runs = descriptor.spans(&region.mask).len().max(1) as u32;
        assert_eq!(
            split.per_region[at], runs,
            "region {at} ({:?}) split into the wrong number of launches",
            region.nodes,
        );
        let want = if fragmented.contains(&at) { 1 } else { runs };
        assert_eq!(
            copy.per_region[at], want,
            "region {at} ({:?}) costs {} launches under a copy",
            region.nodes, copy.per_region[at],
        );
    }
    // SILENT ON PURPOSE, like the catalog gates: the numbers ride in the
    // assert message, so a green run says nothing and a red one says
    // everything. (`engine` denies both print macros in its tests.)
    let (split_launches, copy_launches) = (
        split.per_region.iter().sum::<u32>(),
        copy.per_region.iter().sum::<u32>(),
    );
    let saved: u32 = split_launches - copy_launches;
    assert!(
        saved > 0,
        "the copy saved no launch at all: {} fragmented regions, {split_launches} launches \
         split against {copy_launches} copied",
        fragmented.len(),
    );

    // THE BRACKET. Exactly one gather and one scatter per copied region, in
    // that order, and nothing at all for a region P4 seated.
    let mut want: Vec<(u32, &str)> = Vec::new();
    for &at in &fragmented {
        let node = compiled.template()[at].nodes.start;
        want.push((node, "gather"));
        want.push((node, "scatter"));
    }
    assert_eq!(copy_dispatch.moved, want);
    assert!(
        split_dispatch.moved.is_empty(),
        "a split moved rows, and the whole point of a split is that it does not",
    );

    // AND THE WORK IS THE SAME WORK. Every node the split ran, the copy ran —
    // once per interval there, once over the union here.
    let split_counts = split_dispatch.counts();
    let copy_counts = copy_dispatch.counts();
    assert_eq!(
        split_counts.len(),
        copy_counts.len(),
        "the two walks dispatched different node SETS",
    );
    for (node, ran) in &copy_counts {
        let region = compiled
            .template()
            .iter()
            .position(|region| region.nodes.contains(node))
            .expect("every dispatched node stands in a region");
        let want = if fragmented.contains(&region) {
            1
        } else {
            split_counts[node]
        };
        assert_eq!(*ran, want, "node {node} of region {region}");
    }
}

#[test]
fn a_backend_that_says_nothing_about_serve_still_gets_every_split_it_had() {
    let (trace, compiled) = sku();
    let lanes = fragmenting(&compiled);
    let composition = compose(&compiled, &budget(), &lanes).expect("three lanes compose");
    let descriptor = FireDescriptor::of(&composition);

    // `MockDispatch::new(trace, false)` IS the default `Serve` impl's answer,
    // and the assertion is that the walk's launch counts are then exactly the
    // ones `WindowTable::spans` states — which is what rule 4 said before the
    // branch existed and what every other gate in this repo is written to.
    let (runs, dispatch) = fire(&trace, &compiled, &lanes, false);
    let mut fragmented = 0usize;
    for (at, region) in compiled.template().iter().enumerate() {
        let spans = descriptor.spans(&region.mask).len();
        fragmented += usize::from(spans > 1);
        assert_eq!(runs.per_region[at], spans.max(1) as u32, "region {at}");
    }
    assert!(fragmented > 0, "this composition fragments nothing");
    assert!(dispatch.moved.is_empty());
}

#[test]
fn the_schedule_builder_takes_the_same_answer_as_the_consumers_that_read_it() {
    let (_, compiled) = sku();
    let lanes = fragmenting(&compiled);
    let composition = compose(&compiled, &budget(), &lanes).expect("three lanes compose");
    let bucket = budget()
        .buckets
        .iter()
        .position(|&rows| rows == composition.bucket())
        .expect("the fire lands in the lattice") as u32;
    let descriptor = FireDescriptor::of(&composition);

    // The prepare region P4 owes nothing and the capture regions it owes rows
    // for state THE SAME MASK — qwen3.5's `attention.plan_prefill` and its six
    // `attention.prefill_lse` readers — and both come back in pieces in this
    // fire. That is the shape the copy has to get right: a builder that split
    // while its readers copied would carve one schedule per interval and the
    // single gathered launch would read the first one.
    let mut prepare = 0usize;
    let mut capture = 0usize;
    for region in compiled.template() {
        if descriptor.spans(&region.mask).len() < 2 {
            continue;
        }
        assert!(
            fallback::copies(&compiled, &region.mask, bucket),
            "region {:?} is in pieces and takes a different answer from its own mask",
            region.nodes,
        );
        match region.phase {
            model_compiler::Phase::Prepare => prepare += 1,
            model_compiler::Phase::Capture => capture += 1,
        }
    }
    assert!(
        prepare > 0 && capture > 0,
        "this fire fragments {prepare} prepare and {capture} capture regions, and the \
         claim is about a builder and its readers being both",
    );

    // And the prepare region is owed no row of its OWN — which is exactly why
    // `fallback::copies` asks the mask rather than the nodes.
    for region in compiled.template() {
        if region.phase != model_compiler::Phase::Prepare {
            continue;
        }
        if descriptor.spans(&region.mask).len() < 2 {
            continue;
        }
        assert!(
            fallback::answers(&compiled, region.nodes.clone()).is_empty(),
            "P4 wrote a row for a prepare region, and this test's reason to exist \
             was that it does not",
        );
    }
}

#[test]
fn the_menu_asks_for_a_copy_below_the_crossover_and_a_split_above_it() {
    let (_, compiled) = sku();
    let lattice = budget().buckets;

    // The table itself, read at every bucket — the claim the walk's branch is
    // built on, and the one that says the ten small buckets were the ones
    // paying for a split they did not ask for.
    let withdrawn: Vec<&Region> = compiled
        .template()
        .iter()
        .filter(|region| !fallback::answers(&compiled, region.nodes.clone()).is_empty())
        .collect();
    assert!(
        !withdrawn.is_empty(),
        "no region of `{SKU}` is owed a fallback"
    );

    let mut copies = 0usize;
    let mut splits = 0usize;
    for bucket in 0..lattice.len() as u32 {
        let mask = &withdrawn[0].mask;
        if fallback::copies(&compiled, mask, bucket) {
            copies += 1;
        } else {
            splits += 1;
        }
    }
    assert_eq!(
        (copies, splits),
        (10, 4),
        "the {}-point lattice's copy/split cut moved",
        lattice.len(),
    );

    // And the split rows say how many launches they would have cost, which is
    // the number the copy replaces with one.
    let r = compiled
        .fallback
        .rows
        .iter()
        .find_map(|row| match row.fallback {
            Fallback::Split { r } => Some(r),
            Fallback::Copy | Fallback::Grouped | Fallback::View => None,
        })
        .expect("the lattice reaches past the crossover");
    assert_eq!(r, 3, "the withdrawn mask breaks into three runs");
    assert_eq!(
        fallback::bound(&compiled, &withdrawn[0].mask),
        3,
        "and the bound derived from the shipped order agrees",
    );

    // The withdrawn mask itself, named so a seriation change is a failure
    // here rather than a silent change of subject.
    assert_eq!(
        withdrawn[0].mask.iter().collect::<Vec<_>>(),
        [4, 5, 6, 7],
        "the withdrawn window is `captures_scores`",
    );
    let order = compiled
        .order
        .class_order(&ClassSet::of(0..compiled.classes.classes.len()), None);
    assert_eq!(
        order,
        [4, 0, 2, 6, 7, 3, 1, 5],
        "the baked class order moved"
    );
}
