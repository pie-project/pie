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
//! - **every dispatch followed the template's region order** — an engine that
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

use model_exec::KernelError;
use model_exec::dispatch::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear,
};
use model_compiler::{Budget, DeviceProfile, Lowering, Phase, Region, compile};
use model_dsl::Platform;
use model_exec::fire::{EagerSink, EventId, FireDescriptor, Lane, Sink, compose, walk};
use model_ir::{
    Attention, ClassTable, Collective, CustomCuda, Elementwise, Layout, Linear, Operands, Operation,
    Trace,
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
/// bucket lattice a decode-heavy serve rounds up to, and NO adapters.
///
/// **`max_adapters: 0` IS WHAT MAKES THIS FILE A TEST** (palo C2, design §8).
/// It sat at a flat 32 from before the IR had a bank seat, and `compile` now
/// refuses a load whose ask is bigger than the model text's own capacity — so
/// 32 refused all sixty-eight SKU x platform pairs, every `let Ok(baked) =
/// ... else { continue }` below took the `continue`, and three tests that
/// assert on an empty `wrong` passed by never running a body. The bug this
/// file exists to catch — a `Phase::Prepare` region standing after a
/// `Phase::Capture` one, which is `Fault::PrepareAfterCapture` on every
/// composition of qwen3.6 — sat behind that skip for as long as it was there.
///
/// Zero is the number that seats the WHOLE catalog rather than the subset with
/// a bank: five families declare none, qwen declares eight, and a plan with no
/// bank is exempt at zero and refused above it. The adapter axis has its own
/// files (`engine-cuda/tests/adapter_banks.rs`), which ask each plan for what
/// it seats; what this one is about is the walk, on every text there is.
fn budget() -> Budget {
    Budget {
        max_lanes: 256,
        max_tokens: 8192,
        buckets: vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ],
        max_adapters: 0,
    }
}

/// **THE NON-VACUITY ASSERT, AND WHY EVERY TEST HERE ENDS WITH ONE.** Each
/// loop below skips a pair `compile` refuses, deliberately — which platform a
/// text bakes on is `model-compiler`'s own catalog test to answer, not this
/// one's. The cost of that reading is that a bake nothing can seat is
/// indistinguishable from a walk with nothing wrong in it: `wrong` is empty
/// either way. So the count of pairs that actually got walked is asserted
/// against the catalog's own size, and the day a SKU stops baking this file
/// says so instead of going quiet.
fn no_pair_was_skipped(walked: usize, of: usize) {
    assert_eq!(
        walked, of,
        "{walked} of {of} SKU x platform pairs were walked — the rest were \
         refused at `compile` and every assertion in this test skipped them",
    );
}

/// A backend that runs nothing and remembers everything: `(node index, op
/// name)`, in the order the walk called it.
///
/// **HOW IT KNOWS THE NODE INDEX**, since the contract does not tell it. A
/// `Dispatch*` method is handed the OP and not the node — deliberately, since
/// "`guard` and `layer` are the engine walk's business" — so the mock builds
/// one map at construction from each node's op-payload ADDRESS to its index
/// and looks the incoming reference up in it. The payload lives inside the
/// `Trace`'s node vector, which outlives the walk and is never moved during
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
/// The default: this backend has no row gather, so every fragmented window it
/// meets is a `Fallback::Split` — which is what this file's launch counts
/// have always been about.
impl model_exec::fire::Serve for MockDispatch {}

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
fn mixes(classes: &ClassTable) -> Vec<(String, Vec<Lane>)> {
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
    let catalog = model::catalog();
    let pairs = catalog.len() * PLATFORMS.len();
    let mut walked = 0usize;

    for (sku, _, trace, _) in catalog {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budget(), &DeviceProfile::default()) else {
                continue; // `model-compiler`'s own catalog test is what says so.
            };
            walked += 1;

            for (name, lanes) in mixes(&compiled.classes) {
                let fire = match compose(&compiled, &budget(), &lanes) {
                    Ok(fire) => fire,
                    Err(refusal) => {
                        wrong.push(format!("`{sku}` as {platform:?} [{name}]: {refusal}"));
                        continue;
                    }
                };
                let present: BTreeSet<usize> = fire.present().iter().map(|&c| c as usize).collect();
                let descriptor = FireDescriptor::of(&fire);

                let mut dispatch = MockDispatch::new(&trace);
                if let Err(refusal) =
                    walk(&trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
                {
                    wrong.push(format!("`{sku}` as {platform:?} [{name}]: {refusal}"));
                    continue;
                }

                // **TEMPLATE ORDER, NOT PROGRAM ORDER, AND ONCE PER LAUNCH.**
                // This used to assert the node indices strictly ascend — each
                // node once, in the order the trace stated them — and two
                // landed passes have made that the wrong claim about the
                // artifact rather than a claim the walk broke. P5's hoist puts
                // the prepare regions in front of the graph body, so the
                // template is deliberately no longer program order; and rule
                // 4 dispatches a region's nodes once per interval of a window
                // P4 could not seat, so a node legitimately runs `r` times.
                // What survives, and is what the assertion was always really
                // about, is that the walk followed the TEMPLATE: the region a
                // dispatch belongs to never goes backwards.
                let ran = dispatch.nodes();
                let region_of = |node: u32| {
                    compiled
                        .template()
                        .iter()
                        .position(|region| region.nodes.contains(&node))
                };
                let visited: Vec<Option<usize>> = ran.iter().map(|&n| region_of(n)).collect();
                if !visited.windows(2).all(|pair| pair[0] <= pair[1]) {
                    wrong.push(format!(
                        "`{sku}` as {platform:?} [{name}]: the dispatches did not follow the \
                         template's region order",
                    ));
                }
                let ran: BTreeSet<u32> = ran.into_iter().collect();

                // What the artifact says should run: every node some present
                // class demands, plus every collective, always (decision #5).
                let mut demanded: BTreeSet<u32> = (0..trace.nodes.len() as u32)
                    .filter(|&node| {
                        present
                            .iter()
                            .any(|&class| compiled.classes.node_mask[node as usize].contains(class))
                    })
                    .collect();
                let collectives: BTreeSet<u32> = trace
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
                // every node whose `guard` holds — and the walk runs what the
                // classes DEMAND, which is `live` narrowed by the backward
                // demand walk (`ClassTable::node_mask`). The two differ on real
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
                    .flat_map(|&class| compiled.classes.classes[class].live.iter().copied())
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
    no_pair_was_skipped(walked, pairs);
}

#[test]
fn every_sku_shows_the_sink_its_whole_template_every_fire() {
    let mut wrong: Vec<String> = Vec::new();
    let catalog = model::catalog();
    let pairs = catalog.len() * PLATFORMS.len();
    let mut walked = 0usize;

    for (sku, _, trace, _) in catalog {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budget(), &DeviceProfile::default()) else {
                continue;
            };
            walked += 1;
            let template: Vec<u32> = compiled.template().iter().map(|r| r.nodes.start).collect();
            // Set by the first mix, compared by the rest.
            let mut shape: Option<(usize, usize)> = None;

            for (name, lanes) in mixes(&compiled.classes) {
                let Ok(fire) = compose(&compiled, &budget(), &lanes) else {
                    continue; // the test above is the one that says so.
                };
                let descriptor = FireDescriptor::of(&fire);
                let mut dispatch = MockDispatch::new(&trace);
                let mut structure = Structure::default();
                if walk(&trace, &compiled, &descriptor, &mut dispatch, &mut structure).is_err() {
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
                // **COMPOSITION-INDEPENDENT, WHICH IS THE CLAIM THAT MATTERS.**
                // This used to assert zero conditionals and zero stream events
                // — true of a v1 that lowered everything always-launch on one
                // stream, and false since P6 (fork/join) and P3 (one region of
                // one SKU) landed. Zero was never the property the recorded
                // graph needs; SAMENESS is. One graph serves every composition
                // only if the brackets it was recorded with do not depend on
                // which windows a fire happens to fill, so every mix of one
                // artifact must show the sink the same counts.
                match shape {
                    None => shape = Some((structure.conds, structure.events)),
                    Some(first) if first != (structure.conds, structure.events) => {
                        wrong.push(format!(
                            "`{sku}` as {platform:?} [{name}]: {} conditional and {} stream \
                             events, where another mix of the same artifact showed {first:?}",
                            structure.conds, structure.events,
                        ));
                    }
                    Some(_) => {}
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
    no_pair_was_skipped(walked, pairs);
}

/// **THE BUG THIS FILE WAS SUPPOSED TO CATCH, ASKED DIRECTLY** (design §5).
///
/// Prepare ops are host work that writes descriptor slots the graph then
/// reads, so a `Phase::Prepare` region standing after a `Phase::Capture` one
/// is a slot written after it was read — and [`walk`]'s rule 3 REFUSES such a
/// template rather than reordering it, because the order is the compiler's
/// output. `model_compiler`'s P5 is what puts the prepare half in front.
///
/// It went unseen because a supergraph has no obligation to state its host
/// work first: qwen3.6 appends the multi-token-prediction head after the
/// trunk, so the head's flashinfer plan build — the one `Ty::Struct` definer
/// in that text that is not at the top — landed three hundred and thirty-nine
/// regions deep, and `Fault::PrepareAfterCapture` was the answer to EVERY
/// composition of that SKU. Every other text in the catalog states its plan
/// builds before its first launch and is unaffected, which is exactly the
/// shape of bug one SKU hides from a suite that skips it.
///
/// Asked twice, and the pair is the point: the ARTIFACT's phases are ordered,
/// which is P5's claim, and the WALK accepts every composition of it, which is
/// the claim a fire depends on. The first without the second would pass on a
/// template the walk rejects for some other reason.
#[test]
fn no_sku_bakes_a_prepare_region_behind_the_graph_that_reads_its_slots() {
    let mut wrong: Vec<String> = Vec::new();
    let catalog = model::catalog();
    let pairs = catalog.len() * PLATFORMS.len();
    let mut walked = 0usize;

    for (sku, _, trace, _) in catalog {
        for platform in PLATFORMS {
            let trace = trace(platform);
            let Ok(compiled) = compile(&trace, &budget(), &DeviceProfile::default()) else {
                continue;
            };
            walked += 1;

            let mut captured = false;
            for (at, region) in compiled.template().iter().enumerate() {
                match region.phase {
                    Phase::Capture => captured = true,
                    Phase::Prepare if captured => wrong.push(format!(
                        "`{sku}` as {platform:?}: region {at} (nodes {}..{}) is host prepare \
                         work standing after the graph body",
                        region.nodes.start, region.nodes.end,
                    )),
                    Phase::Prepare => {}
                }
            }

            for (name, lanes) in mixes(&compiled.classes) {
                let Ok(fire) = compose(&compiled, &budget(), &lanes) else {
                    continue;
                };
                let descriptor = FireDescriptor::of(&fire);
                let mut dispatch = MockDispatch::new(&trace);
                if let Err(refusal) =
                    walk(&trace, &compiled, &descriptor, &mut dispatch, &mut EagerSink)
                {
                    wrong.push(format!("`{sku}` as {platform:?} [{name}]: {refusal}"));
                }
            }
        }
    }

    assert!(wrong.is_empty(), "\n{}\n", wrong.join("\n"));
    no_pair_was_skipped(walked, pairs);
}

#[test]
fn a_composition_is_the_only_thing_that_changes_between_fires() {
    // The claim the whole design rests on: the artifact is baked once and the
    // fire path only writes a descriptor. So walking the same `CompiledModel` with
    // wildly different batches must never need a second `compile` — and the
    // windows those batches produce must tile their rows exactly, every time.
    let mut wrong: Vec<String> = Vec::new();
    let catalog = model::catalog();
    let pairs = catalog.len();
    let mut walked = 0usize;

    for (sku, _, trace, _) in catalog {
        let trace = trace(Platform::Cuda);
        let Ok(compiled) = compile(&trace, &budget(), &DeviceProfile::default()) else {
            continue;
        };
        walked += 1;

        for (name, lanes) in mixes(&compiled.classes) {
            let Ok(fire) = compose(&compiled, &budget(), &lanes) else {
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
    no_pair_was_skipped(walked, pairs);
}
