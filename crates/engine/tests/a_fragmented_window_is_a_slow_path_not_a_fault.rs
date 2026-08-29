//! The catalog's fragmented windows, fired. What P4 could not seat runs as
//! several launches and computes the right rows — it does not abort the fire.
//!
//! # The bug this pins
//!
//! `model_compiler`'s P4 solves one C1P instance over the whole plan and, for
//! every windowed consumer it cannot make an interval of the row order,
//! WITHDRAWS the constraint and writes the consumer's nodes into
//! `CompiledModel::fallback` with an answer per bucket range (design §3). Nothing read
//! that table. A fire whose class order left such a window in pieces was a
//! `engine_cuda::Fault::Fragmented` — a hard refusal, justified in prose by
//! "the catalog bakes an empty `FallbackTable` today".
//!
//! It does not. Compiled for CUDA at a budget the catalog can actually seat
//! (`max_adapters: 8`; at 32 every SKU refuses on adapter capacity and every
//! catalog test that uses that number skips its body), the four qwen3.5 texts
//! bake 12, 12, 20 and 20 fallback rows — two per `attention.prefill_lse`
//! node, `Fallback::Copy` below the copy/split crossover and
//! `Fallback::Split { r: 3 }` above it — and qwen3.6-27b bakes 84 over 42
//! nodes. All of them are the `captures_scores` window, C4's axis, which
//! CROSSES `qo_one` where every earlier axis nested inside it. On
//! qwen35-d0.8b, 132 of the 255 non-empty compositions leave some region's
//! window in pieces, and the smallest is a fire carrying classes 0, 4 and 5 at
//! once: three lanes.
//!
//! # What is asserted
//!
//! - **the fire is not refused.** The walk returns `Ok` for a composition that
//!   fragments a window, which is the whole point;
//! - **the split partitions the window.** The runs' rows are exactly the rows
//!   of the lanes the mask admits — every one of them, once — checked against
//!   the composition's own lane placement rather than against the span
//!   arithmetic that produced them. Too few rows is a batch silently computing
//!   only its first interval; too many is a kernel reading rows that belong to
//!   another class;
//! - **the same for lanes**, which is a different number in a mixed fire and
//!   is what the geometry vectors and the qo boundaries are indexed by;
//! - **each node runs once per run**, and every unfragmented region still runs
//!   its nodes exactly once — the split is paid where it is owed and nowhere
//!   else;
//! - **the old seam still says what it always said.** `WindowTable::span` —
//!   "can this consumer be ONE launch" — still answers `Err(r)` for these
//!   masks. The fire is served anyway, by asking the other question.
//!
//! NO GPU. This is the golden path: `EagerSink` and a backend that runs
//! nothing, which is the same walk the CUDA shell records.

use std::collections::HashMap;

use engine::fire::{
    EagerSink, EventId, FireDescriptor, Lane, MaskSpan, Sink, compose, fallback, walk,
};
use kernels::error::KernelError;
use kernels::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear,
};
use model_compiler::{CompiledModel, Budget, DeviceProfile, Lowering, Region, compile};
use model_dsl::Platform;
use model_ir::{
    Attention, ClassSet, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation,
    Trace,
};

/// A deployment's ceilings, at an adapter capacity the catalog can seat.
///
/// **`max_adapters: 8` IS LOAD-BEARING AND IS THE WHOLE REASON THIS FILE
/// EXISTS.** `model-compiler`'s and `engine`'s own catalog sweeps ask for 32,
/// no catalog text seats more than 8, and `compile` therefore refuses all 68
/// (SKU × platform) pairs — so every one of those tests walks its loop and
/// asserts nothing. At 8 the five qwen texts bake, and they are the texts that
/// owe fallback rows.
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

/// A backend that runs nothing and remembers which node it was handed, by the
/// address of the op payload inside the plan's node vector — the same identity
/// trick `every_sku_walks_its_classes.rs` uses, and for the same reason: a
/// `Dispatch*` method is given the OP and not the node, and a count of op
/// NAMES cannot tell a node that ran twice from two nodes that ran once.
struct MockDispatch {
    at: HashMap<usize, u32>,
    seen: Vec<u32>,
}

impl MockDispatch {
    fn new(trace: &Trace) -> MockDispatch {
        MockDispatch {
            at: trace
                .nodes
                .iter()
                .enumerate()
                .map(|(j, node)| (payload(&node.op), j as u32))
                .collect(),
            seen: Vec::new(),
        }
    }

    /// How many times each node was dispatched.
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
/// The default: no row gather, so every fragmented window this file fires is
/// served as `Fallback::Split` — which is this file's whole subject. The copy
/// the same table asks for below the crossover is gated next door, in
/// `a_copied_window_is_one_launch_over_the_same_rows.rs`.
impl engine::fire::Serve for MockDispatch {}

/// A sink that writes down how many runs each region was cut into — the
/// structure event the split adds, and the one a shell's cursor turns into a
/// window lookup.
#[derive(Default)]
struct Runs {
    /// One entry per region, in template order: how many runs it announced.
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

/// The rows a mask admits, read off the composition's LANE PLACEMENT rather
/// than off the span arithmetic — the independent answer the spans are checked
/// against.
fn rows_of_lanes(descriptor: &FireDescriptor, mask: &ClassSet) -> Vec<u32> {
    let mut rows: Vec<u32> = descriptor
        .lanes
        .iter()
        .filter(|lane| mask.contains(lane.class as usize))
        .flat_map(|lane| lane.row_offset..lane.row_offset + lane.rows)
        .collect();
    rows.sort_unstable();
    rows
}

/// The lanes a mask admits, in fire order, by their position in the seriated
/// lane list.
fn lanes_of_lanes(descriptor: &FireDescriptor, mask: &ClassSet) -> Vec<u32> {
    descriptor
        .lanes
        .iter()
        .enumerate()
        .filter(|(_, lane)| mask.contains(lane.class as usize))
        .map(|(at, _)| at as u32)
        .collect()
}

/// The rows and lanes a list of runs covers, flattened.
fn covered(spans: &[MaskSpan]) -> (Vec<u32>, Vec<u32>) {
    let rows = spans
        .iter()
        .flat_map(|span| span.row_offset..span.row_offset + span.rows)
        .collect();
    let lanes = spans
        .iter()
        .flat_map(|span| span.lane_offset..span.lane_offset + span.lanes)
        .collect();
    (rows, lanes)
}

/// Fire one composition and check every claim in the module doc. Returns how
/// many regions this fire found fragmented, so the caller can refuse to pass
/// vacuously.
fn fire_and_check(
    what: &str,
    trace: &Trace,
    compiled: &CompiledModel,
    lanes: &[Lane],
    wrong: &mut Vec<String>,
) -> usize {
    let fire = match compose(compiled, &budget(), lanes) {
        Ok(fire) => fire,
        Err(refusal) => {
            wrong.push(format!("{what}: the fire does not compose — {refusal}"));
            return 0;
        }
    };
    let descriptor = FireDescriptor::of(&fire);

    let mut dispatch = MockDispatch::new(trace);
    let mut runs = Runs::default();
    // THE ASSERTION THE WHOLE FILE IS FOR: this used to be a refusal.
    if let Err(refusal) = walk(trace, compiled, &descriptor, &mut dispatch, &mut runs) {
        wrong.push(format!("{what}: the fire is refused — {refusal}"));
        return 0;
    }

    let counts = dispatch.counts();
    let mut fragmented = 0usize;
    for (at, region) in compiled.template().iter().enumerate() {
        let spans = descriptor.spans(&region.mask);
        if spans.len() > 1 {
            fragmented += 1;

            // The old seam still refuses it — a shell that can only make one
            // launch is still told so, and told how many it would take.
            if descriptor.span(&region.mask) != Err(spans.len()) {
                wrong.push(format!(
                    "{what}: region {at} covers {} runs and `span` did not say so",
                    spans.len(),
                ));
            }
            // And P4 did not PROMISE it consecutive: a fragmented window it
            // promised is the bake-integrity failure `Fault::Fragmented`
            // keeps, and a fire that produced one would mean the class order
            // did not come from P4's tree.
            if fallback::promised(compiled, region) {
                wrong.push(format!(
                    "{what}: region {at} is in {} pieces and P4 promised it whole",
                    spans.len(),
                ));
            }
            let bound = fallback::bound(compiled, &region.mask);
            if spans.len() > bound as usize {
                wrong.push(format!(
                    "{what}: region {at} covers {} runs where its baked order breaks \
                     the mask into {bound}",
                    spans.len(),
                ));
            }
        }

        // THE ROWS. The runs partition exactly the rows of the lanes this
        // mask admits: sorted equality catches a missing interval, a doubled
        // one, and an interval that reached into a class the mask excludes.
        let (rows, seats) = covered(&spans);
        if rows != rows_of_lanes(&descriptor, &region.mask) {
            wrong.push(format!(
                "{what}: region {at}'s {} runs cover {} rows, and its classes hold {}",
                spans.len(),
                rows.len(),
                rows_of_lanes(&descriptor, &region.mask).len(),
            ));
        }
        if seats != lanes_of_lanes(&descriptor, &region.mask) {
            wrong.push(format!(
                "{what}: region {at}'s runs cover lanes {seats:?}, and its classes hold {:?}",
                lanes_of_lanes(&descriptor, &region.mask),
            ));
        }
        // The runs are ascending and disjoint, and none of them is empty —
        // an empty run is a launch that costs a kernel and computes nothing,
        // and two runs that touch are one run the merge failed to close.
        for pair in spans.windows(2) {
            if pair[0].row_offset + pair[0].rows >= pair[1].row_offset {
                wrong.push(format!(
                    "{what}: region {at}'s runs {pair:?} are not disjoint and ascending",
                ));
            }
        }
        if spans.iter().any(|span| span.rows == 0) {
            wrong.push(format!("{what}: region {at} cut an empty run"));
        }

        // THE DISPATCH COUNT. Once per run for a region with rows, and once
        // for a collective in a region with none (decision #5).
        if runs.per_region.get(at) != Some(&(spans.len().max(1) as u32)) {
            wrong.push(format!(
                "{what}: region {at} announced {:?} runs and covers {}",
                runs.per_region.get(at),
                spans.len().max(1),
            ));
        }
        for node in region.nodes.clone() {
            let collective = matches!(trace.nodes[node as usize].op, Operation::Collective(_));
            let want = if spans.is_empty() {
                usize::from(collective)
            } else {
                spans.len()
            };
            let ran = counts.get(&node).copied().unwrap_or(0);
            if ran != want {
                wrong.push(format!(
                    "{what}: node {node} of region {at} ran {ran} times, and its window \
                     is {} run(s)",
                    spans.len(),
                ));
            }
        }
    }
    fragmented
}

/// One lane per class, so every window of the artifact is non-empty at once —
/// which is the composition that fragments every mask P4 withdrew.
fn every_class(compiled: &CompiledModel) -> Vec<Lane> {
    compiled
        .classes
        .classes
        .iter()
        .enumerate()
        .map(|(at, class)| Lane::new(class.word(), 1 + at as u32))
        .collect()
}

#[test]
fn every_catalog_window_p4_could_not_seat_fires_as_several_launches() {
    let mut wrong: Vec<String> = Vec::new();
    let mut owed = 0usize;
    let mut fragmented = 0usize;
    let mut unwalkable: Vec<&str> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let Ok(compiled) = compile(&trace, &budget(), &DeviceProfile::default()) else {
            continue; // an adapter capacity this text cannot seat; not this test's subject.
        };
        if compiled.fallback.rows.is_empty() {
            continue;
        }
        owed += 1;
        // THE DEBT THIS BLOCK WAS OPENED FOR IS PAID, AND THE GUARD STAYS.
        // qwen3.6-27b used to land here: its multi-token-prediction head is
        // appended after the trunk, so the head's plan build — a `Ty::Struct`
        // definer, and therefore `Phase::Prepare` — stood after three hundred
        // and thirty-nine capture regions, and `engine::fire::walk` refused
        // every composition of it by name, split or no split. P5 now hoists
        // the prepare half in front of the capture half
        // (`model_compiler::region::hoist`), so the list below is empty.
        //
        // The check is kept rather than deleted because an EMPTY list is the
        // claim: this file's subject is fragmented windows, and a text the
        // walk refuses outright never reaches `fire_and_check`, so a
        // regression in the phase order would quietly shrink what this test
        // covers instead of failing it.
        if walk(
            &trace,
            &compiled,
            &FireDescriptor::of(
                &compose(&compiled, &budget(), &every_class(&compiled)).expect("composes"),
            ),
            &mut MockDispatch::new(&trace),
            &mut EagerSink,
        )
        .is_err()
        {
            unwalkable.push(sku);
            continue;
        }
        fragmented += fire_and_check(
            &format!("`{sku}` every-class"),
            &trace,
            &compiled,
            &every_class(&compiled),
            &mut wrong,
        );
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
    // NOT VACUOUS, SAID OUT LOUD. The premise of the whole file is that the
    // catalog owes fallback rows and that a real fire cashes them; a budget
    // change or a model text that made either false would otherwise turn this
    // into a green test of nothing — which is exactly how the hard fault
    // survived.
    assert!(owed > 0, "no catalog text bakes a fallback row");
    assert!(
        fragmented > 0,
        "{owed} texts owe fallback rows and no fire found a window in pieces",
    );
    assert_eq!(
        unwalkable,
        [] as [&str; 0],
        "a catalog text `walk` refuses outright never reaches the split path below, \
         so its fragmented windows go unchecked",
    );
}

#[test]
fn the_smallest_qwen_fire_that_fragments_a_window_is_three_lanes() {
    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    let (_, _, trace, _) = model::catalog()
        .into_iter()
        .find(|(sku, ..)| *sku == SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"));
    let trace = trace(Platform::Cuda);
    let compiled = compile(&trace, &budget(), &DeviceProfile::default()).expect("`{SKU}` bakes");

    // Eight classes over four fact bits, and the score-capture window is
    // `{4,5,6,7}` — the classes whose word sets bit 3. P4 seats it against
    // the frontier it found; a fire carrying class 0 BETWEEN classes 4 and 5
    // is the smallest one that does not.
    let words: Vec<u64> = [0usize, 4, 5]
        .iter()
        .map(|&class| compiled.classes.classes[class].word())
        .collect();
    let lanes: Vec<Lane> = words.iter().map(|&word| Lane::new(word, 1)).collect();

    let fire = compose(&compiled, &budget(), &lanes).expect("three lanes compose");
    assert_eq!(fire.present(), [4, 0, 5], "class 0 stands between 4 and 5");

    let mut wrong: Vec<String> = Vec::new();
    let fragmented = fire_and_check(
        &format!("`{SKU}` {{0,4,5}}"),
        &trace,
        &compiled,
        &lanes,
        &mut wrong,
    );
    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
    assert!(
        fragmented > 0,
        "classes 0, 4 and 5 at once leave the score-capture window in pieces",
    );

    // And the split is two launches over the three rows: one for class 4's
    // row, one for class 5's, with class 0's row standing between them.
    let scores = compiled
        .template()
        .iter()
        .find(|region| !fallback::answers(&compiled, region.nodes.clone()).is_empty())
        .expect("some region is owed a fallback");
    let descriptor = FireDescriptor::of(&fire);
    let spans = descriptor.spans(&scores.mask);
    assert_eq!(spans.len(), 2, "{spans:?}");
    assert_eq!(
        spans.iter().map(|span| span.rows).sum::<u32>(),
        descriptor.rows_of(&scores.mask),
    );
    assert_eq!(
        spans.iter().map(|span| span.lanes).sum::<u32>(),
        descriptor.classes.lanes_of(&scores.mask),
    );
}
