use std::collections::HashMap;

use model_compiler::{Budget, CompiledModel, DeviceProfile, Lowering, Region, compile};
use model_dsl::Platform;
use model_exec::KernelError;
use model_exec::dispatch::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear, DispatchSpatial,
};
use model_exec::fire::{
    EventId, Filter, FireDescriptor, Lane, Serve, Sink, compose, fallback, walk,
};
use model_ir::{
    Attention, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation, Spatial,
    Trace,
};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

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

struct MockDispatch {
    at: HashMap<usize, u32>,
    seen: Vec<u32>,
    copies: bool,
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
        Operation::Spatial(op) => address(op),
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

impl DispatchSpatial for MockDispatch {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl Serve for MockDispatch {
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
    let trace = models::sku(SKU)
        .map(|row| row.trace)
        .unwrap_or_else(|| panic!("`{SKU}` is in the catalog"));
    let trace = trace(Platform::Cuda);
    let compiled = compile(&trace, &budget(), &DeviceProfile::default())
        .unwrap_or_else(|refusal| panic!("`{SKU}` bakes: {refusal:?}"));
    (trace, compiled)
}

fn fragmenting(compiled: &CompiledModel) -> Vec<Lane> {
    [0usize, 4, 5]
        .iter()
        .map(|&class| Lane::new(compiled.classes.classes[class].word(), 1))
        .collect()
}

fn fire(
    trace: &Trace,
    compiled: &CompiledModel,
    lanes: &[Lane],
    copies: bool,
) -> (Runs, MockDispatch) {
    let composition = compose(compiled, &budget(), lanes).expect("the fire composes");
    let descriptor = FireDescriptor::of(&composition);
    let mut dispatch = MockDispatch::new(trace, copies);
    let mut runs = Runs::default();
    walk(
        trace,
        compiled,
        &descriptor,
        &mut dispatch,
        &mut runs,
        Filter::default(),
    )
    .expect("the fire walks");
    (runs, dispatch)
}

#[test]
fn a_copied_window_is_one_launch_over_the_same_rows_every_case() {
    a_copied_window_costs_one_launch_where_a_split_one_costs_its_runs();
    the_schedule_builder_takes_the_same_answer_as_the_consumers_that_read_it();
}

fn a_copied_window_costs_one_launch_where_a_split_one_costs_its_runs() {
    let (trace, compiled) = sku();
    let lanes = fragmenting(&compiled);

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
        .filter(|&at| {
            fallback::copies(
                &compiled,
                model_ir::RowAxis::Tokens,
                &compiled.template()[at].mask,
                bucket,
            )
        })
        .collect();
    assert_eq!(
        copied, fragmented,
        "at bucket {bucket} the table asks for a copy on some fragmented windows and \
         not others, and this file's premise is that it asks for one on all of them",
    );

    let (split, split_dispatch) = fire(&trace, &compiled, &lanes, false);
    let (copy, copy_dispatch) = fire(&trace, &compiled, &lanes, true);

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

    let mut prepare = 0usize;
    let mut capture = 0usize;
    for region in compiled.template() {
        if descriptor.spans(&region.mask).len() < 2 {
            continue;
        }
        assert!(
            fallback::copies(&compiled, model_ir::RowAxis::Tokens, &region.mask, bucket),
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

    for region in compiled.template() {
        if region.phase != model_compiler::Phase::Prepare {
            continue;
        }
        if descriptor.spans(&region.mask).len() < 2 {
            continue;
        }
        assert!(
            fallback::answers(&compiled, model_ir::RowAxis::Tokens, region.nodes.clone())
                .is_empty(),
            "P4 wrote a row for a prepare region, and this test's reason to exist \
             was that it does not",
        );
    }
}

impl model_exec::DispatchProbe for MockDispatch {}
