//! Pins `Fallback::Grouped` in the walk: the launch count drops from `r`
//! to one, and nothing else about the fire moves. Unlike `Split { r }`
//! (a loop turning `r` times), `Grouped` is one launch handed the whole
//! interval list, so the walk needs a branch — this checks: the two bakes
//! share a row order, the split arm dispatches each correction node `r`
//! times and announces `r` runs, the grouped arm dispatches once and
//! announces one run, and every other node is dispatched identically in
//! both arms.
//!
//! Uses a scaffold: on this catalog the LoRA correction normally wins
//! seating outright (no fallback row to compare), so `split_arm`/
//! `grouped_arm` bias the cost model to force the withdrawal in a
//! controlled, comparable way. No GPU: a mock dispatch records the same
//! walk the CUDA shell records.

use std::collections::HashMap;

use model_exec::KernelError;
use model_exec::dispatch::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear,
};
use model_compiler::{
    CompiledModel, Budget, DeviceProfile, FamilyCosts, Lowering, Region, compile,
};
use model_dsl::Platform;
use model_exec::fire::{EventId, Filter, FireDescriptor, Lane, Sink, compose, walk};
use model_ir::{
    Attention, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation, Trace,
};

/// The SKU whose adapter window fragments into six intervals once the
/// scaffold withdraws it — one per class of the window, the worst case.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The op the correction is, by the name both profile lists are keyed on.
const CORRECTION: &str = "linear.lora_correct";

/// A deployment's ceilings, at an adapter capacity the catalog can seat.
/// `max_adapters: 8` is load-bearing: a sweep asking for 32 compiles
/// nothing (no text seats more than eight), so the loop bodies below would
/// silently skip.
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

fn trace() -> Trace {
    let trace = models::sku(SKU).unwrap_or_else(|| panic!("`{SKU}` is in the catalog")).trace;
    trace(Platform::Cuda)
}

fn bake(trace: &Trace, profile: &DeviceProfile) -> CompiledModel {
    compile(trace, &budget(), profile).unwrap_or_else(|why| panic!("`{SKU}` bakes: {why:?}"))
}

/// The split arm: the same withdrawal, served the old way. The two arms
/// must share a row order or the comparison is worthless, so this tells
/// the cost model the correction is cheap rather than naming the op
/// groupable — same mask withdrawn, same frontier, only the answer differs.
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
fn corrections(trace: &Trace) -> Vec<u32> {
    trace.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op.name() == CORRECTION)
        .map(|(at, _)| at as u32)
        .collect()
}

/// One lane per class of the artifact — the composition that presents every
/// behaviour at once, and therefore the one that fragments the most. A
/// prefill lane carries three rows and a decode lane one, off the class's
/// own `qo_one` bit.
fn one_lane_per_class(compiled: &CompiledModel) -> Vec<Lane> {
    compiled
        .classes
        .classes
        .iter()
        .map(|class| {
            let decode = class.word() & 1 == 1;
            Lane::new(class.word(), if decode { 1 } else { 3 })
        })
        .collect()
}

fn fire(compiled: &CompiledModel, lanes: &[Lane]) -> FireDescriptor {
    FireDescriptor::of(&compose(compiled, &budget(), lanes).expect("the lanes compose"))
}

// ── the mock backend ─────────────────────────────────────────────────────

/// A backend that runs nothing and remembers which node it was handed, by
/// the address of the op payload inside the plan's node vector: a
/// `Dispatch*` method is given the op, not the node.
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
impl model_exec::fire::fallback::Serve for MockDispatch {}

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
fn walked(trace: &Trace, compiled: &CompiledModel, lanes: &[Lane]) -> (HashMap<u32, usize>, Vec<u32>) {
    let descriptor = fire(compiled, lanes);
    let mut dispatch = MockDispatch::new(trace);
    let mut runs = Runs::default();
    walk(
        trace,
        compiled,
        &descriptor,
        &mut dispatch,
        &mut runs,
        Filter::default(),
    )
    .expect("a fragmented fire walks");
    (dispatch.counts(), runs.per_region)
}

// ── the gates ────────────────────────────────────────────────────────────

/// The gate: on one composition, the grouped arm dispatches every
/// correction node once where the split arm dispatches it `r` times, and
/// every other node in the plan is dispatched exactly as often in both.
#[test]
fn the_grouped_arm_pays_one_launch_where_the_split_arm_pays_r() {
    let trace = trace();
    let corrections = corrections(&trace);
    assert!(!corrections.is_empty(), "the SKU states corrections");

    let split = bake(&trace, &split_arm());
    let grouped = bake(&trace, &grouped_arm());
    let lanes = one_lane_per_class(&split);
    assert_eq!(lanes.len(), 12, "`{SKU}` resolves twelve classes");

    // How many intervals the adapter window breaks into in this fire; must
    // be more than one, or an `r == 1` fire would make both arms one
    // launch and this test green for the wrong reason.
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

    let (split_counts, split_runs) = walked(&trace, &split, &lanes);
    let (grouped_counts, grouped_runs) = walked(&trace, &grouped, &lanes);

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
    let region_of = |compiled: &CompiledModel| {
        compiled
            .template()
            .iter()
            .position(|region| region.nodes.clone().any(|node| corrections.contains(&node)))
            .expect("some region holds a correction")
    };
    assert_eq!(split_runs[region_of(&split)], r as u32);
    assert_eq!(grouped_runs[region_of(&grouped)], 1);

    // Nothing else moved: every non-correction node cost the same in both arms.
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
    // Silent on purpose: numbers ride in this message, not a print macro.
    assert!(
        compared > 0,
        "the plan has nodes besides its corrections — `{SKU}`, twelve classes in one fire: \
         adapter window = {r} intervals, correction launches {r} -> 1 per node over {} nodes",
        corrections.len(),
    );
}

impl model_exec::DispatchProbe for MockDispatch {}
