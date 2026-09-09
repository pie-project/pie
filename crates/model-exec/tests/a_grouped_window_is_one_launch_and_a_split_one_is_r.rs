use std::collections::HashMap;

use model_compiler::{
    Budget, CompiledModel, DeviceProfile, FamilyCosts, Lowering, Region, compile,
};
use model_dsl::Platform;
use model_exec::KernelError;
use model_exec::dispatch::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear, DispatchSpatial,
};
use model_exec::fire::{EventId, Filter, FireDescriptor, Lane, Sink, compose, walk};
use model_ir::{
    Attention, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation, Spatial,
    Trace,
};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const CORRECTION: &str = "linear.lora_correct";

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
    let trace = models::sku(SKU)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"))
        .trace;
    trace(Platform::Cuda)
}

fn bake(trace: &Trace, profile: &DeviceProfile) -> CompiledModel {
    compile(trace, &budget(), profile).unwrap_or_else(|why| panic!("`{SKU}` bakes: {why:?}"))
}

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

fn corrections(trace: &Trace) -> Vec<u32> {
    trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.op.name() == CORRECTION)
        .map(|(at, _)| at as u32)
        .collect()
}

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
        Operation::Spatial(op) => address(op),
    }
}

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

impl DispatchSpatial for MockDispatch {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
        self.note(op)
    }
}

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

fn walked(
    trace: &Trace,
    compiled: &CompiledModel,
    lanes: &[Lane],
) -> (HashMap<u32, usize>, Vec<u32>) {
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

#[test]
fn the_grouped_arm_pays_one_launch_where_the_split_arm_pays_r() {
    let trace = trace();
    let corrections = corrections(&trace);
    assert!(!corrections.is_empty(), "the SKU states corrections");

    let split = bake(&trace, &split_arm());
    let grouped = bake(&trace, &grouped_arm());
    let lanes = one_lane_per_class(&split);
    assert_eq!(lanes.len(), 12, "`{SKU}` resolves twelve classes");

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

    let region_of = |compiled: &CompiledModel| {
        compiled
            .template()
            .iter()
            .position(|region| region.nodes.clone().any(|node| corrections.contains(&node)))
            .expect("some region holds a correction")
    };
    assert_eq!(split_runs[region_of(&split)], r as u32);
    assert_eq!(grouped_runs[region_of(&grouped)], 1);

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
    assert_eq!(
        split_runs.len(),
        grouped_runs.len(),
        "one template, two bakes"
    );
    assert!(
        compared > 0,
        "the plan has nodes besides its corrections — `{SKU}`, twelve classes in one fire: \
         adapter window = {r} intervals, correction launches {r} -> 1 per node over {} nodes",
        corrections.len(),
    );
}

impl model_exec::DispatchProbe for MockDispatch {}
