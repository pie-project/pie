//! The catalog, fired. Six model texts, four platforms, every lane mix that the
//! design's vocabulary can name — walked with a backend that runs nothing and
//! remembers everything.
//!
//! WHY THE REAL CATALOG AND NOT A FIXTURE. Every unit test in `src/fire/`
//! builds its plan by hand, which checks this crate against this crate's own
//! idea of what a fire looks like. What the catalog adds is the only thing a
//! fixture cannot: plans somebody else wrote, in the authoring surface, at the
//! size and shape the deployment ships — sixty layers of residual chains, MoE
//! fan-outs, a nested split on `masked` over `qo_one`, per-layer seams. A walk
//! that is right on a five-node chain and wrong on those is a walk that
//! computes.
//!
//! WHAT IT ASSERTS, and each is a bug that does not fault:
//!
//! - **the nodes that ran are exactly the nodes the classes demand** — the
//!   union, over the classes this fire has lanes in, of what those classes
//!   run. Too few is a window silently missing from a mixed batch, which
//!   presents as garbage tokens for the requests in the class that was
//!   dropped; too many is a kernel reading rows that belong to somebody else;
//! - **every node ran at most once, in program order** — a driver that
//!   dispatched a node twice would write its output rectangle twice, and the
//!   second write is a race against the reader in between;
//! - **collectives ran regardless** — decision #5. The one node family whose
//!   presence is not a function of the composition;
//! - **the eager and the recording walk see the same structure** — every
//!   region announced, in template order, whether or not it had rows.
//!
//! NO GPU, and that is the point of porting-order step 3: this is the golden
//! path, and the shell's recorded graph is later diffed against it.

use std::collections::{BTreeSet, HashMap};

use driver::fire::{EagerSink, EventId, FireDescriptor, Lane, Sink, compose, walk};
use kernels::error::KernelError;
use kernels::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear,
};
use model_compiler::{Budgets, DeviceProfile, Lowering, Region, compile};
use model_dsl::Platform;
use model_ir::{
    Attention, Classes, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation,
    Plan,
};

/// Every platform a plan can be traced at. A model text may emit a different op
/// per platform, so the split-and-merge structure is not the same graph on each,
/// and one platform passing says nothing about the others.
const PLATFORMS: [Platform; 4] = [
    Platform::Cuda,
    Platform::Metal,
    Platform::Wgpu,
    Platform::Vulkan,
];

/// A deployment's ceilings: 256 concurrent requests, 8192 token rows, the
/// bucket lattice a decode-heavy serve rounds up to.
fn budgets() -> Budgets {
    Budgets {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 32,
    }
}

/// A backend that runs nothing and remembers everything: `(node index, op
/// name)`, in the order the walk called it.
///
/// **HOW IT KNOWS THE NODE INDEX**, since the contract does not tell it. A
/// `Dispatch*` method is handed the OP and not the node — deliberately, since
/// "`cond` and `layer` are the driver walk's business" — so the mock builds
/// one map at construction from each node's op-payload ADDRESS to its index
/// and looks the incoming reference up in it. The payload lives inside the
/// `Plan`'s node vector, which outlives the walk and is never moved during
/// one, so the address is a stable identity. No `unsafe`: a reference cast to
/// `usize` is a comparison of two things the borrow checker already proved are
/// alive.
///
/// The alternative — recording only op names — cannot say whether a node ran
/// twice or whether two same-named nodes swapped places, and in a plan with
/// sixty identical layers that is nearly everything this file is about.
struct MockDispatch {
    at: HashMap<usize, u32>,
    seen: Vec<(u32, &'static str)>,
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

    fn nodes(&self) -> Vec<u32> {
        self.seen.iter().map(|(node, _)| *node).collect()
    }

    fn note<T: Operands>(&mut self, op: &T) -> Result<(), KernelError> {
        let node = *self
            .at
            .get(&address(op))
            .expect("every dispatched op is a node of the plan the mock was built from");
        self.seen.push((node, op.name()));
        Ok(())
    }
}

fn address<T>(value: &T) -> usize {
    std::ptr::from_ref(value).cast::<()>() as usize
}

/// The address of the op INSIDE the variant — the very reference a `Dispatch*`
/// method receives, rather than the enum's own address, which an unspecified
/// layout may place elsewhere.
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

/// A sink that counts what the structure was, so the walk's claim — every
/// region, in template order, rows or no rows — can be checked against the
/// artifact rather than against itself.
#[derive(Default)]
struct Structure {
    regions: Vec<u32>,
    /// How many launches the walk cut the regions into, summed — one per
    /// region for a plan every window of which P4 seated, more where a
    /// fragmented window pays `Fallback::Split`.
    runs: usize,
    conds: usize,
    events: usize,
}

impl Sink for Structure {
    fn region_begin(&mut self, region: &Region) {
        self.regions.push(region.nodes.start);
    }
    fn region_end(&mut self, _region: &Region) {}
    fn run(&mut self, _run: u32, _runs: u32) {
        self.runs += 1;
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {
        self.conds += 1;
    }
    fn cond_arm(&mut self, _arm: u8) {
        self.conds += 1;
    }
    fn cond_end(&mut self) {}
    fn fork(&mut self, _event: EventId) {
        self.events += 1;
    }
    fn join(&mut self, _event: EventId) {
        self.events += 1;
    }
}

/// The decode bit. Bit 0 in every catalog text — each family's `Classify`
/// packs `qo_one` first, which is the one convention the plans share and the
/// only one this file needs: a plan carries no fact NAMES, so the bit is
/// reached by the position the models all agree on.
const DECODE: u64 = 1;

/// The lane mixes worth firing at a plan, named.
///
/// THE FIRST THREE ARE THE DESIGN'S OWN EXAMPLE and the rest are what the
/// plan's own guards ask for: a bit the sweep ran over is an axis a batch can
/// be mixed along, so every such bit gets its window exercised with and
/// without the decode bit, and one mix carries a lane of EVERY class at once.
/// On gemma — the only catalog text whose guards reach a second bit — that is
/// where the nested split over two facts gets its three classes fired
/// together.
fn mixes(classes: &Classes) -> Vec<(String, Vec<Lane>)> {
    let decode = DECODE & classes.mask;
    let mut mixes = vec![
        (
            "all-decode".to_string(),
            (0..6).map(|_| Lane::new(decode, 1)).collect(),
        ),
        (
            "all-prefill".to_string(),
            [7, 3, 11].map(|rows| Lane::new(0, rows)).to_vec(),
        ),
        (
            // design §0's diagram, verbatim: two prefill lanes of 7 and 3
            // rows, three decode lanes of 1.
            "mixed decode+prefill".to_string(),
            vec![
                Lane::new(0, 7),
                Lane::new(0, 3),
                Lane::new(decode, 1),
                Lane::new(decode, 1),
                Lane::new(decode, 1),
            ],
        ),
    ];

    for at in 1..u64::BITS as u64 {
        let set = 1u64 << at;
        if classes.mask & set == 0 {
            continue;
        }
        mixes.push((
            format!("all-fact{at} decode"),
            (0..4).map(|_| Lane::new(set | decode, 1)).collect(),
        ));
        mixes.push((
            format!("all-fact{at} prefill"),
            vec![Lane::new(set, 9), Lane::new(set, 2)],
        ));
        mixes.push((
            format!("fact{at} and not, decode and prefill"),
            vec![
                Lane::new(set | decode, 1),
                Lane::new(decode, 1),
                Lane::new(set, 5),
                Lane::new(0, 4),
            ],
        ));
    }

    // One lane per class, whatever the classes turned out to be — the
    // composition with every window non-empty at once.
    mixes.push((
        "every class".to_string(),
        classes
            .classes
            .iter()
            .enumerate()
            .map(|(at, class)| Lane::new(class.word(), 1 + at as u32))
            .collect(),
    ));
    mixes
}

#[test]
fn every_sku_walks_exactly_the_nodes_its_composition_demands() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue; // `model-compiler`'s own catalog test is what says so.
            };

            for (name, lanes) in mixes(&baked.classes) {
                let fire = match compose(&baked, &budgets(), &lanes) {
                    Ok(fire) => fire,
                    Err(refusal) => {
                        wrong.push(format!("`{sku}` as {platform:?} [{name}]: {refusal}"));
                        continue;
                    }
                };
                let present: BTreeSet<usize> = fire.present().iter().map(|&c| c as usize).collect();
                let descriptor = FireDescriptor::of(&fire);

                let mut dispatch = MockDispatch::new(&plan);
                if let Err(refusal) =
                    walk(&plan, &baked, &descriptor, &mut dispatch, &mut EagerSink)
                {
                    wrong.push(format!("`{sku}` as {platform:?} [{name}]: {refusal}"));
                    continue;
                }

                let ran = dispatch.nodes();
                if !ran.windows(2).all(|pair| pair[0] < pair[1]) {
                    wrong.push(format!(
                        "`{sku}` as {platform:?} [{name}]: the nodes did not run once each in \
                         program order",
                    ));
                }
                let ran: BTreeSet<u32> = ran.into_iter().collect();

                // What the artifact says should run: every node some present
                // class demands, plus every collective, always (decision #5).
                let mut demanded: BTreeSet<u32> = (0..plan.nodes.len() as u32)
                    .filter(|&node| {
                        present
                            .iter()
                            .any(|&class| baked.classes.node_mask[node as usize].contains(class))
                    })
                    .collect();
                let collectives: BTreeSet<u32> = plan
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, node)| matches!(node.op, Operation::Collective(_)))
                    .map(|(j, _)| j as u32)
                    .collect();
                demanded.extend(&collectives);

                if ran != demanded {
                    let missing: Vec<u32> = demanded.difference(&ran).copied().take(8).collect();
                    let extra: Vec<u32> = ran.difference(&demanded).copied().take(8).collect();
                    wrong.push(format!(
                        "`{sku}` as {platform:?} [{name}]: {} nodes ran, {} were demanded — \
                         missing {missing:?}, extra {extra:?}",
                        ran.len(),
                        demanded.len(),
                    ));
                }

                // THE SAME SET SAID THE DESIGN'S WAY, AND WHERE THE TWO PART
                // COMPANY. `Class::live` is what a class's fact word ADMITS —
                // every node whose `cond` holds — and the walk runs what the
                // classes DEMAND, which is `live` narrowed by the backward
                // demand walk (`Classes::node_mask`). The two differ on real
                // plans and gemma is where it shows: an attention plan build
                // is guarded `Always`, so it is live in every class, but the
                // only node that reads its struct is the decode attention —
                // so an all-prefill fire is right to skip it, and design §5's
                // step 4 says so in as many words ("run prepare ops, skip
                // empty windows").
                //
                // So the containment is the assertion, not the equality: what
                // ran is always something the present classes admit, never
                // anything else.
                let mut live: BTreeSet<u32> = present
                    .iter()
                    .flat_map(|&class| baked.classes.classes[class].live.iter().copied())
                    .collect();
                live.extend(&collectives);
                let unadmitted: Vec<u32> = ran.difference(&live).copied().take(8).collect();
                if !unadmitted.is_empty() {
                    wrong.push(format!(
                        "`{sku}` as {platform:?} [{name}]: nodes {unadmitted:?} ran in a fire \
                         whose classes do not admit them",
                    ));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn every_sku_shows_the_sink_its_whole_template_every_fire() {
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        for platform in PLATFORMS {
            let plan = trace(platform);
            let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
                continue;
            };
            let template: Vec<u32> = baked.template().iter().map(|r| r.nodes.start).collect();

            for (name, lanes) in mixes(&baked.classes) {
                let Ok(fire) = compose(&baked, &budgets(), &lanes) else {
                    continue; // the test above is the one that says so.
                };
                let descriptor = FireDescriptor::of(&fire);
                let mut dispatch = MockDispatch::new(&plan);
                let mut structure = Structure::default();
                if walk(&plan, &baked, &descriptor, &mut dispatch, &mut structure).is_err() {
                    continue;
                }

                // Composition-independent structure: the same regions, in the
                // same order, for every mix. That is what makes ONE recorded
                // graph serve all of them.
                if structure.regions != template {
                    wrong.push(format!(
                        "`{sku}` as {platform:?} [{name}]: the sink saw {} regions of {}",
                        structure.regions.len(),
                        template.len(),
                    ));
                }
                // v1: every region always-launch, one stream.
                if structure.conds != 0 || structure.events != 0 {
                    wrong.push(format!(
                        "`{sku}` as {platform:?} [{name}]: {} conditional and {} stream events \
                         from an artifact that lowers everything always-launch",
                        structure.conds, structure.events,
                    ));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}

#[test]
fn a_composition_is_the_only_thing_that_changes_between_fires() {
    // The claim the whole design rests on: the artifact is baked once and the
    // fire path only writes a descriptor. So walking the same `Baked` with
    // wildly different batches must never need a second `compile` — and the
    // windows those batches produce must tile their rows exactly, every time.
    let mut wrong: Vec<String> = Vec::new();

    for (sku, _, trace, _) in model::catalog() {
        let plan = trace(Platform::Cuda);
        let Ok(baked) = compile(&plan, &budgets(), &DeviceProfile::default()) else {
            continue;
        };

        for (name, lanes) in mixes(&baked.classes) {
            let Ok(fire) = compose(&baked, &budgets(), &lanes) else {
                continue;
            };
            let rows: u32 = lanes.iter().map(|lane| lane.rows).sum();
            if fire.rows() != rows {
                wrong.push(format!("`{sku}` [{name}]: {} rows for {rows}", fire.rows()));
            }
            if fire.bucket() < fire.rows() {
                wrong.push(format!(
                    "`{sku}` [{name}]: bucket {} is under {} rows",
                    fire.bucket(),
                    fire.rows(),
                ));
            }

            // The lanes tile the rows, and every class's window is the runs of
            // the lanes in it.
            let mut at = 0;
            for lane in fire.lanes() {
                if lane.row_offset != at {
                    wrong.push(format!(
                        "`{sku}` [{name}]: lane {} starts at {} with {at} rows placed",
                        lane.source, lane.row_offset,
                    ));
                    break;
                }
                at += lane.rows;
            }
            if at != fire.rows() {
                wrong.push(format!(
                    "`{sku}` [{name}]: the lanes cover {at} of {} rows",
                    fire.rows(),
                ));
            }

            // And the descriptor is a faithful carrier of all of it.
            let descriptor = FireDescriptor::of(&fire);
            match FireDescriptor::unpack(&descriptor.pack()) {
                Ok(back) if back == descriptor => {}
                Ok(_) => wrong.push(format!("`{sku}` [{name}]: the round trip changed it")),
                Err(refusal) => wrong.push(format!("`{sku}` [{name}]: {refusal}")),
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
}
