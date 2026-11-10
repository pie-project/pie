#![cfg(feature = "device")]

//! The golden-logits probe: a fixed set of catalog miniatures, one of each
//! cache shape (dense attention, sliding windows, GDN and mamba state,
//! DeepSeek V4's compressed pools, MLA-like and routed rows), each served
//! from a checkpoint seeded by plane name over a fixed script of prefills
//! and greedy decodes, and each step's logits hashed bit for bit. A change
//! that must not move a logit (a memory layout, an address) is proven by
//! the same hashes before and after it.
//!
//! cuBLAS picks a dense product's algorithm by timing it once a process, so
//! two processes may sum a product in two orders. The probe pins those picks
//! in a GEMM algorithm cache shared across runs (`PIE_GOLDEN_CACHE`, by
//! default `pie-golden-cache` under the system's temp dir): the first run
//! times and records each pick, every later one reads it back. Each model is
//! served twice in its process, the first serving only warming the picks,
//! so a product first met inside a graph capture (left untimed there) has
//! been timed before the hashed serving, in every run and every order.
//!
//! Run by hand (both are ignored in the suite): first `golden_fixtures`,
//! which seeds each model's checkpoint into the cache's `fixtures` without a
//! device (host work, kept out of the GPU's lock), then `golden_logits`:
//!
//! ```text
//! cargo test -p engine-cuda --features cuda --test engine_cuda \
//!     golden_fixtures -- --ignored --nocapture
//! flock /tmp/pie-gpu.lock cargo test -p engine-cuda --features cuda \
//!     --test engine_cuda golden_logits -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It prints one `golden <model> <step> <hash> <argmax>` line a step and one
//! `golden <model> all <hash>` line a model. `PIE_GOLDEN_MODELS` (a comma
//! list) picks and orders the models; `PIE_GOLDEN_OUT` writes the lines to a
//! file; `PIE_GOLDEN_CHECK` compares them with such a file and fails on the
//! first model whose hash moved.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Lane, Seated, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const PAGE: u32 = 16;

/// Rows a fire carries at most.
const TOKENS: u32 = 256;

/// Rows a slot holds: past every miniature's sliding window (Gemma 4's
/// 512, MiMo's 128), so each window list trims.
const CONTEXT: u32 = 1024;

/// The models probed, one per cache shape.
const MODELS: &[&str] = &[
    // Dense full attention (Llama 3.3's miniature keeps the whole's width,
    // some 10 GB of planes: too heavy for a probe run on a shared device).
    "qwen3-micro-text",
    // Sliding windows beside global layers, a shared kv tail.
    "gemma4-e4b-mini-l6",
    // Gated delta net state beside full attention.
    "qwen35-tiny",
    // Mamba state beside attention, routed experts.
    "nemotron3-nano-mini",
    // Compressed pools and hash-routed layers.
    "dsv4-flash-mini",
    // Dots3's normed k.
    "dots3-note-mini",
    // Lightning attention beside full, routed experts.
    "minimax-m3-mini",
    // Windowed and full GQA, routed experts.
    "mimo26-flash-mini",
];

/// One step of the script: a prefill of so many rows, or a greedy decode.
#[derive(Clone, Copy)]
enum Step {
    Prefill(u32),
    Decode,
}

/// A short prefill and decodes, then two long prefills appended onto the
/// cache, decodes after each: 37 + 3 + 250 + 3 + 250 + 3 = 546 rows.
const SCRIPT: &[Step] = &[
    Step::Prefill(37),
    Step::Decode,
    Step::Decode,
    Step::Decode,
    Step::Prefill(250),
    Step::Decode,
    Step::Decode,
    Step::Decode,
    Step::Prefill(250),
    Step::Decode,
    Step::Decode,
    Step::Decode,
];

fn fnv_bytes(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

const FNV: u64 = 0xcbf2_9ce4_8422_2325;

fn fnv(name: &str) -> u64 {
    fnv_bytes(FNV, name.as_bytes())
}

/// The hash of a step's logits, bit for bit.
fn hash_logits(logits: &[f32]) -> u64 {
    logits
        .iter()
        .fold(FNV, |hash, v| fnv_bytes(hash, &v.to_bits().to_le_bytes()))
}

/// A checkpoint of `trace`'s own planes seeded by name: float planes norms
/// near one and the rest small, integer planes small indices.
/// The span an index plane's entries keep below: no miniature probed routes
/// to fewer experts.
const INDEX_SPAN: u64 = 8;

type Seeded = (String, Vec<u64>, ztensor::Leaf, Vec<u8>);

/// `trace`'s checkpoint planes seeded by name, in name order.
fn seeded_planes(trace: &Trace) -> Vec<Seeded> {
    let mut out = Vec::new();
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    // A packed plane's scales, biases and tensor scales are made with it
    // when the load quantizes it: the container holds none of their own.
    let made: std::collections::BTreeSet<String> = planes
        .iter()
        .filter(|param| {
            matches!(
                param.dtype,
                Dtype::Mxfp4
                    | Dtype::Nvfp4
                    | Dtype::E4m3tensor
                    | Dtype::U4g64
                    | Dtype::U8g64
                    | Dtype::U4g32
                    | Dtype::U4g64tiled
                    | Dtype::U2g32
                    | Dtype::U2g64
                    | Dtype::U2g128
            )
        })
        .flat_map(|param| {
            [
                poem::scales_name(&param.name),
                poem::biases_name(&param.name),
                poem::global_scales_name(&param.name),
                poem::input_scales_name(&param.name),
            ]
        })
        .collect();
    planes.retain(|param| !made.contains(&param.name));
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        // The shape the weight has before the load packs it: a packed
        // plane's codes are stored narrower than the values they hold.
        let shape = poem::Weight::of_plane(param).shape;
        let count: usize = shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut seed = fnv(&param.name);
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            seed >> 33
        };
        let unit = |bits: u64| (bits as f32 / (1u64 << 31) as f32) - 0.5;
        let value = |bits: u64| {
            if norm {
                1.0 + 0.05 * unit(bits)
            } else {
                0.08 * unit(bits)
            }
        };
        let (leaf, bytes): (ztensor::Leaf, Vec<u8>) = match param.dtype {
            Dtype::Bf16 => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            Dtype::F32 => (
                ztensor::Leaf::F32,
                (0..count)
                    .flat_map(|_| value(next()).to_le_bytes())
                    .collect(),
            ),
            // An index plane (a hash router's token-to-expert table): each
            // row's entries distinct and below [`INDEX_SPAN`].
            Dtype::I64 | Dtype::I32 => {
                let cols = param.shape.last().copied().unwrap_or(1).max(1);
                let wide = param.dtype == Dtype::I64;
                let ids = (0..count as u64).map(|at| (at / cols + at % cols) % INDEX_SPAN);
                if wide {
                    (
                        ztensor::Leaf::I64,
                        ids.flat_map(|id| id.to_le_bytes()).collect(),
                    )
                } else {
                    (
                        ztensor::Leaf::I32,
                        ids.flat_map(|id| (id as u32).to_le_bytes()).collect(),
                    )
                }
            }
            // A plain fp8 plane is stored as served (no host cast makes
            // one): e4m3 codes of a small magnitude, each finite.
            Dtype::E4m3 => (
                ztensor::Leaf::E4M3,
                (0..count)
                    .map(|_| {
                        let bits = next();
                        ((bits & 0x80) | (0x18 + (bits >> 8) % 0x28)) as u8
                    })
                    .collect(),
            ),
            // So are a block scale's exponents (near one) and a byte plane.
            Dtype::E8m0 => (
                ztensor::Leaf::E8M0,
                (0..count).map(|_| (118 + next() % 6) as u8).collect(),
            ),
            Dtype::U8 => (
                ztensor::Leaf::U8,
                (0..count).map(|_| next() as u8).collect(),
            ),
            // A packed plane is stored as the bf16 it is quantized from:
            // the import encodes it, as it does a float checkpoint's.
            _ => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
        };
        out.push((param.name.clone(), shape, leaf, bytes));
    }
    out
}

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    for (name, shape, leaf, bytes) in seeded_planes(trace) {
        writer
            .add(name.as_str(), shape, leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{name}`: {why}"));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// The same planes as a serving artifact: the file a deployed model loads
/// from, whose planes lie in it as the trace reads them.
fn write_serving(path: &Path, trace: &Trace, model: &str) {
    use checkpoint::file::emit::{self, Object};
    let planes = seeded_planes(trace);
    let objects: Vec<Object<'_>> = planes
        .iter()
        .map(|(name, shape, leaf, bytes)| Object::leaf(name, shape.clone(), *leaf, bytes))
        .collect();
    emit::write(
        path,
        &checkpoint::serving::Stamp::of("cuda", model),
        &std::collections::BTreeMap::new(),
        4096,
        &objects,
        |object, plane, _| panic!("{object}/{plane} is not streamed here"),
    )
    .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// `model` traced for CUDA, every plane and its kv cache bf16: a model whose
/// layout states a dtype per component (its routed experts apart) gets
/// bf16 for each.
fn trace_of(model: &str) -> Trace {
    // A catalog deployment by its full name, as it is deployed (its weights'
    // and kv's formats, its drafter); else a model's bf16 trace.
    if let Some(deployment) = poem_compiler::catalog::deployment(model) {
        return deployment.trace(Platform::Cuda);
    }
    let package = poem_compiler::catalog::package_of(model)
        .unwrap_or_else(|| panic!("no package holds `{model}`"));
    let mut deploy = poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16);
    // `PIE_GOLDEN_DRAFTER`: the model with a drafter head of that name (a
    // drafter's rows may be read through a window of their own).
    if let Ok(drafter) = std::env::var("PIE_GOLDEN_DRAFTER") {
        deploy.drafter = Some(drafter);
    }
    let mut first = None;
    for _ in 0..3 {
        match package.trace(model, &deploy, model, Platform::Cuda) {
            Ok(trace) => return trace,
            Err(why) => first.get_or_insert(format!("{why:#}")),
        };
        deploy.weights.push(Dtype::Bf16.into());
    }
    panic!("`{model}` does not trace: {}", first.unwrap_or_default())
}

struct Fixture {
    model: String,
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
}

/// `model`'s trace and its seeded checkpoint, written once into `cache`'s
/// `fixtures` and read back by every later run.
fn fixture(model: &str, cache: &Path) -> Fixture {
    fixture_as(model, cache, false)
}

/// [`fixture`], the checkpoint a serving artifact where `serving`: the same
/// bytes, laid as a deployed model's file lays them.
fn fixture_as(model: &str, cache: &Path, serving: bool) -> Fixture {
    let trace = trace_of(model);
    let dir = cache.join("fixtures");
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    // A deployment with a plane the load serves in another form than bf16
    // (packed codes, fp8) loads from a serving artifact alone: a load
    // converts nothing. Its artifact is imported from the seeded floats.
    let encoded = trace.params.iter().any(|param| {
        param.source == ParamSource::Checkpoint
            && !matches!(
                param.dtype,
                Dtype::Bf16 | Dtype::F32 | Dtype::I64 | Dtype::I32
            )
    });
    let serving = serving || encoded;
    let suffix = if serving { "serving.zt" } else { "zt" };
    let container = dir.join(format!("{model}.{suffix}"));
    if !container.exists() {
        let partial = dir.join(format!("{model}.{suffix}.{}", std::process::id()));
        if encoded {
            let floats = dir.join(format!("{model}.floats.zt"));
            if !floats.exists() {
                let seeded = dir.join(format!("{model}.floats.zt.{}", std::process::id()));
                write_checkpoint(&seeded, &trace);
                std::fs::rename(&seeded, &floats)
                    .unwrap_or_else(|why| panic!("{}: {why}", floats.display()));
            }
            let source = ztensor::Source::open(&floats).expect("the seeded floats open");
            let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
                .expect("the seeded floats are read by the plan's own names");
            drop(source);
            let tiled: Vec<&str> = trace
                .params
                .iter()
                .filter(|param| param.dtype == Dtype::U4g64tiled)
                .map(|param| param.name.as_str())
                .collect();
            checkpoint::testkit::artifact::import(
                &floats,
                &contract,
                &checkpoint::serving::Stamp::of("cuda", model),
                &tiled,
                &partial,
            )
            .unwrap_or_else(|why| panic!("{model} imports: {why}"));
        } else if serving {
            write_serving(&partial, &trace, model);
        } else {
            write_checkpoint(&partial, &trace);
        }
        std::fs::rename(&partial, &container)
            .unwrap_or_else(|why| panic!("{}: {why}", container.display()));
    }
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        model: model.to_string(),
        trace,
        contract,
        container,
    }
}

fn cache_dir() -> PathBuf {
    std::env::var_os("PIE_GOLDEN_CACHE").map_or_else(
        || std::env::temp_dir().join("pie-golden-cache"),
        PathBuf::from,
    )
}

fn load(fixture: &Fixture, cache: &Path) -> Shell {
    load_with(fixture, cache, knobs())
}

fn load_with(fixture: &Fixture, cache: &Path, knobs: engine_cuda::Knobs) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        // `PIE_GOLDEN_TOKENS`: the most rows a fire lands, which the
        // lattice a load arms grows with.
        budget: Budget::new(
            1,
            std::env::var("PIE_GOLDEN_TOKENS")
                .ok()
                .and_then(|tokens| tokens.parse().ok())
                .unwrap_or(TOKENS),
        ),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs,
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model))
}

/// The load's knobs: `PIE_GOLDEN_CHUNK` (bytes) sets the chunk size the
/// state pool aims near, so a miniature's records are cut into pieces as a
/// full model's are. The logits are the same either way.
fn knobs() -> engine_cuda::Knobs {
    let mut knobs = engine_cuda::Knobs::default();
    if let Ok(bytes) = std::env::var("PIE_GOLDEN_CHUNK") {
        knobs.diagnostics.chunk_bytes = Some(bytes.parse().expect("PIE_GOLDEN_CHUNK is bytes"));
    }
    // `PIE_GOLDEN_EAGER`: no graph captured and no body recorded; every
    // fire walks its plan. The logits are a recorded body's.
    if std::env::var_os("PIE_GOLDEN_EAGER").is_some() {
        knobs.chip = Some(engine_cuda::chip::pinned(
            engine_cuda::Graphs::Off,
            Some(engine_cuda::Recording::Off),
        ));
    }
    // `PIE_GOLDEN_ONE_WINDOW`: every windowed row in one window list.
    knobs.diagnostics.one_window = std::env::var_os("PIE_GOLDEN_ONE_WINDOW").is_some();
    // `PIE_GOLDEN_TIER_INLINE`: tier copies ride the compute stream.
    knobs.diagnostics.tier_inline = std::env::var_os("PIE_GOLDEN_TIER_INLINE").is_some();
    // No weight shape stated: the tables and routed banks dense on the
    // device, as the base engine seats them. The weight cache's own probes
    // load with its shapes ([`load_cached`]).
    knobs.diagnostics.dense_weights = true;
    knobs
}

fn argmax(logits: &[f32]) -> u32 {
    let (at, _) = logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(at, best), (i, &v)| {
            if v > best { (i, v) } else { (at, best) }
        });
    u32::try_from(at).expect("a vocab index")
}

/// `tokens` fired as slot 0's next step: its logits.
fn fire(shell: &mut Shell, trace: &Trace, tokens: &[u32]) -> Vec<f32> {
    let word = trace.facts.word(&Request::new(
        u32::try_from(tokens.len()).expect("a count"),
        false,
    ));
    let mut rows = shell
        .fire(&[Lane {
            slot: 0,
            word,
            tokens,
        }])
        .expect("the step fires");
    rows.swap_remove(0)
}

/// The model served over [`SCRIPT`]: each step's logits' hash and top token.
fn served(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    let mut shell = load(fixture, cache);
    shell.open(0).expect("slot 0 opens");
    let mut fed = 0u32;
    let mut last = None;
    let mut steps = Vec::with_capacity(SCRIPT.len());
    for step in SCRIPT {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        fed += u32::try_from(tokens.len()).expect("a count");
        let logits = fire(&mut shell, &fixture.trace, &tokens);
        let top = argmax(&logits);
        last = Some(top);
        steps.push((hash_logits(&logits), top));
    }
    steps
}

/// Slot refs handed out as the runtime's pool would: a free physical chunk
/// opened, under the logical id the engine's table places on it, for a
/// shape whose open chunk is full.
struct Refs {
    facts: engine::chunk::ChunkFacts,
    /// The engine's table, kept as its evacuations move it.
    table: engine::chunk::ChunkTable,
    /// The logical chunks opened so far.
    opened: Vec<u32>,
    /// The next physical chunk to open.
    next: u32,
    /// Each shape's open chunk (logical) and the next index in it.
    open: Vec<Option<(u32, u32)>>,
}

impl Refs {
    fn of(facts: engine::chunk::ChunkFacts, first: u32) -> Refs {
        let open = vec![None; facts.shapes.len()];
        let table = engine::chunk::ChunkTable::identity(facts.total_chunks);
        Refs {
            facts,
            table,
            opened: Vec::new(),
            next: first,
            open,
        }
    }

    fn shape(&self, kind: fn(&engine::chunk::SlotKind) -> bool) -> Option<usize> {
        self.facts.shapes.iter().position(|shape| kind(&shape.kind))
    }

    fn take(&mut self, shape: usize) -> u32 {
        let per_chunk = self.facts.shapes[shape].slots_per_chunk;
        let (chunk, index) = match self.open[shape] {
            Some((chunk, index)) if index < per_chunk => (chunk, index),
            _ => {
                let logical = self.table.logical(self.next).expect("a chunk of the pool");
                self.next += 1;
                self.opened.push(logical);
                (logical, 0)
            }
        };
        self.open[shape] = Some((chunk, index + 1));
        chunk * per_chunk + index
    }

    /// Moves every opened chunk onto the physical chunks from `first`, in
    /// the order they were opened.
    fn evacuate(&mut self, shell: &mut Shell, first: u32) {
        let moves: Vec<(u32, u32)> = self
            .opened
            .iter()
            .zip(first..)
            .map(|(&logical, physical)| (logical, physical))
            .collect();
        let evacuation = engine::chunk::Evacuation { moves };
        shell.evacuate(&evacuation).expect("the chunks move");
        self.table
            .apply(&evacuation)
            .expect("the table moves as the engine's");
    }
}

/// The model served over [`SCRIPT`] as the runtime's shared pool serves it:
/// the lane names slot refs of the engine's stated shapes, in chunks opened
/// in the pool's second segment, committed ahead. After the third step every
/// chunk is evacuated into the first segment and the second is released;
/// after every third step from there the chunks move again, between two
/// places of the first segment, the refs standing throughout. The logits
/// are the plain serving's, bit for bit.
fn served_by_refs(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    use engine::chunk::SlotKind;
    let mut shell = load(fixture, cache);
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    facts.validate().expect("the engine's chunk facts hold");
    eprintln!(
        "{}: chunks of {} bytes, shapes {:?}",
        fixture.model,
        facts.chunk_bytes,
        facts
            .shapes
            .iter()
            .map(|shape| (shape.kind, shape.slot_bytes, shape.slots_per_chunk))
            .collect::<Vec<_>>()
    );
    let segment = facts.segment_chunks;
    // The places in the first segment the chunks move between.
    let spread = segment / 2;
    let mut refs = Refs::of(facts.clone(), segment);
    let list = refs.shape(|kind| matches!(kind, SlotKind::List { window: None, .. }));
    // Each window list, narrowest first: a table of its own.
    let window_lists: Vec<usize> = facts
        .shapes
        .iter()
        .enumerate()
        .filter(|(_, shape)| shape.kind.window().is_some())
        .map(|(at, _)| at)
        .collect();
    let fixed = refs.shape(|kind| matches!(kind, SlotKind::Fixed { .. }));
    shell.take_refs();
    shell.commit_segments(2).expect("two segments commit");
    // The lane's recurrent record: a slot a piece.
    let record = fixed.map(|shape| {
        let SlotKind::Fixed { pieces } = facts.shapes[shape].kind else {
            unreachable!("the fixed shape")
        };
        let pieces: Vec<u32> = (0..pieces).map(|_| refs.take(shape)).collect();
        shell.name_record(&pieces).expect("the record is named")
    });
    let mut pages: Vec<u32> = Vec::new();
    let mut windows: Vec<Vec<u32>> = vec![Vec::new(); window_lists.len()];
    let mut low = false;
    let mut fed = 0u32;
    // `PIE_GOLDEN_DECODES=<n>`: n one-token steps and no prefill, for a
    // model whose kv rows this card's kernels read at a decode alone.
    let decodes = std::env::var("PIE_GOLDEN_DECODES")
        .ok()
        .and_then(|steps| steps.parse::<usize>().ok());
    let script: Vec<Step> = match decodes {
        Some(steps) => vec![Step::Decode; steps],
        None => SCRIPT.to_vec(),
    };
    let mut last = decodes.map(|_| 11u32);
    let mut steps = Vec::with_capacity(script.len());
    for (at, step) in script.iter().enumerate() {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        let after = fed + u32::try_from(tokens.len()).expect("a count");
        while (pages.len() as u32) < after.div_ceil(PAGE) {
            if let Some(shape) = list {
                pages.push(refs.take(shape));
            }
            for (table, &shape) in windows.iter_mut().zip(&window_lists) {
                table.push(refs.take(shape));
            }
        }
        assert!(
            (refs.opened.len() as u32) < spread,
            "the lane's chunks lie under the places they move between"
        );
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
        let more: Vec<&[u32]> = windows.iter().skip(1).map(Vec::as_slice).collect();
        let seated = Seated {
            pages: &pages,
            held: Some(fed),
            window: windows.first().map_or(&[], Vec::as_slice),
            more_windows: &more,
            rs_slot: record,
            ..Seated::of(Lane {
                slot: 0,
                word,
                tokens: &tokens,
            })
        };
        let mut rows = shell
            .fire_media(&[seated], &[], &[])
            .expect("the step fires");
        let logits = rows.swap_remove(0);
        fed = after;
        // What the runtime trims: a window list's slot whose last row lies
        // behind the window of the rows held is the null ref from here on.
        // Each list is trimmed by its own window.
        for (table, &shape) in windows.iter_mut().zip(&window_lists) {
            let retained = facts.shapes[shape].kind.window().unwrap_or(u32::MAX);
            let behind = (fed.saturating_sub(retained) / PAGE) as usize;
            for slot in table.iter_mut().take(behind) {
                *slot = 0;
            }
        }
        let top = argmax(&logits);
        last = Some(top);
        steps.push((hash_logits(&logits), top));
        if at == 2 {
            // Out of the second segment, which then goes back to the device:
            // the chunks keep their logical ids, there.
            refs.evacuate(&mut shell, spread);
            refs.next = spread + refs.opened.len() as u32;
            shell
                .release_segments(1)
                .expect("the second segment goes back");
        } else if at % 3 == 2 {
            let opened = refs.opened.len() as u32;
            refs.evacuate(&mut shell, if low { spread } else { 1 });
            refs.next = if low { spread + opened } else { 1 + opened };
            low = !low;
        }
    }
    steps
}

fn models() -> Vec<String> {
    std::env::var("PIE_GOLDEN_MODELS").map_or_else(
        |_| MODELS.iter().map(|m| (*m).to_string()).collect(),
        |list| {
            list.split(',')
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .map(str::to_string)
                .collect()
        },
    )
}

/// The probe's checkpoints written into the cache without a device: run
/// first, outside the GPU's lock, since seeding them is all host work.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_fixtures() {
    let cache = cache_dir();
    for model in models() {
        let fixture = fixture(&model, &cache);
        let bytes = std::fs::metadata(&fixture.container).map_or(0, |meta| meta.len());
        println!("fixture {model} {} MiB", bytes >> 20);
    }
}

#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_logits() {
    probe(served);
}

/// The probe with each lane naming slot refs and its chunks evacuated as it
/// decodes ([`served_by_refs`]): `PIE_GOLDEN_CHECK` against the plain
/// probe's file proves the two serve the same logits.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_logits_by_refs() {
    probe(served_by_refs);
}

/// Host refs handed out as the runtime's host tier would (CONTRACT-BC 22):
/// a fresh host chunk for a shape whose open one is full, host chunk ids
/// their own places.
struct HostRefs {
    next: u32,
    open: Vec<Option<(u32, u32)>>,
}

impl HostRefs {
    fn take(&mut self, facts: &engine::chunk::ChunkFacts, shape: usize) -> u32 {
        let per_chunk = facts.shapes[shape].slots_per_chunk;
        let (chunk, index) = match self.open[shape] {
            Some((chunk, index)) if index < per_chunk => (chunk, index),
            _ => {
                assert!(
                    self.next < facts.host_chunks,
                    "the host tier holds the lane"
                );
                self.next += 1;
                (self.next - 1, 0)
            }
        };
        self.open[shape] = Some((chunk, index + 1));
        chunk * per_chunk + index
    }
}

/// Every slot of `refs` (shape `shape`; ref 0, a trimmed window slot,
/// stays) out to `tier` and back into fresh device slots, as a suspend and
/// a restore move them: the refs the lane names from then on.
fn park(
    shell: &mut Shell,
    refs: &mut Refs,
    host: &mut HostRefs,
    tier: engine_cuda::store::Tier,
    shape: usize,
    slots: &[u32],
) -> Vec<u32> {
    use engine_cuda::store::Tier;
    let id = refs.facts.shapes[shape].id;
    let facts = refs.facts.clone();
    let mut disk = 0u32;
    let out: Vec<(u32, u32)> = slots
        .iter()
        .filter(|&&slot| slot != 0)
        .map(|&slot| {
            let parked = match tier {
                Tier::Disk => {
                    disk += 1;
                    disk - 1
                }
                _ => host.take(&facts, shape),
            };
            (slot, parked)
        })
        .collect();
    shell
        .copy_tier_slots(id, Tier::Device, tier, &out)
        .expect("the slots park");
    let mut back = Vec::with_capacity(out.len());
    let fresh: Vec<u32> = slots
        .iter()
        .map(|&slot| if slot == 0 { 0 } else { refs.take(shape) })
        .collect();
    for (&(_, parked), &slot) in out.iter().zip(fresh.iter().filter(|&&slot| slot != 0)) {
        back.push((parked, slot));
    }
    shell
        .copy_tier_slots(id, tier, Tier::Device, &back)
        .expect("the slots come back");
    fresh
}

/// The model served over [`SCRIPT`] by slot refs as [`served_by_refs`]
/// serves it but for evacuations; where `offload`, after steps 2 and 6
/// every slot the lane holds (kv list pages, window list pages, the record's
/// pieces) parks below the device and comes back to fresh slots in fresh
/// chunks: the host tier the first time, the kv list pages through the slot
/// file the second. The logits are the never-offloaded serving's, bit for
/// bit.
fn served_offloaded(fixture: &Fixture, cache: &Path, offload: bool) -> Vec<(u64, u32)> {
    use engine::chunk::SlotKind;
    use engine_cuda::store::Tier;
    let mut shell = load(fixture, cache);
    // `PIE_GOLDEN_HOST_MIB`: the host tier's budget (1 GiB otherwise), and
    // the process's resident and locked memory printed before it is seated
    // and two seconds after, when what the tier locks ahead has landed.
    let stated = std::env::var("PIE_GOLDEN_HOST_MIB")
        .ok()
        .and_then(|mib| mib.parse::<u64>().ok());
    let resident = |when: &str| {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mib = |key: &str| {
            status
                .lines()
                .find_map(|line| line.strip_prefix(key))
                .and_then(|rest| rest.split_whitespace().next()?.parse::<u64>().ok())
                .map_or(0, |kib| kib >> 10)
        };
        println!(
            "host tier {} {when}: resident {} MiB, locked {} MiB",
            fixture.model,
            mib("VmRSS:"),
            mib("VmLck:")
        );
    };
    if stated.is_some() {
        resident("before it is seated");
    }
    shell
        .seat_host_kv(Some(stated.map_or(1 << 30, |mib| mib << 20)), 1, CONTEXT, 1)
        .expect("the host tier seats");
    if stated.is_some() {
        std::thread::sleep(std::time::Duration::from_secs(2));
        resident("two seconds after");
    }
    let dir = tempfile::tempdir().expect("a disk tier's dir");
    shell
        .seat_disk_kv(
            Some(256 << 20),
            Some(dir.path()),
            false,
            engine_cuda::World::default(),
        )
        .expect("the disk tier seats");
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    facts.validate().expect("the engine's chunk facts hold");
    assert!(facts.host_chunks > 0, "a host tier is stated");
    assert!(facts.disk_chunks > 0, "a disk tier is stated");
    let mut refs = Refs::of(facts.clone(), 1);
    let mut host = HostRefs {
        next: 0,
        open: vec![None; facts.shapes.len()],
    };
    let list = refs.shape(|kind| matches!(kind, SlotKind::List { window: None, .. }));
    let window_list = refs.shape(|kind| kind.window().is_some());
    let fixed = refs.shape(|kind| matches!(kind, SlotKind::Fixed { .. }));
    shell.take_refs();
    let segments = facts.segments.max.min(4);
    shell
        .commit_segments(segments)
        .expect("the segments commit");
    let mut pieces: Vec<u32> = fixed.map_or_else(Vec::new, |shape| {
        let SlotKind::Fixed { pieces } = facts.shapes[shape].kind else {
            unreachable!("the fixed shape")
        };
        (0..pieces).map(|_| refs.take(shape)).collect()
    });
    let mut record =
        (!pieces.is_empty()).then(|| shell.name_record(&pieces).expect("the record is named"));
    let (mut pages, mut window): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut fed = 0u32;
    let mut last = None;
    let mut steps = Vec::with_capacity(SCRIPT.len());
    for (at, step) in SCRIPT.iter().enumerate() {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        let after = fed + u32::try_from(tokens.len()).expect("a count");
        while (pages.len() as u32) < after.div_ceil(PAGE) {
            if let Some(shape) = list {
                pages.push(refs.take(shape));
            }
            if let Some(shape) = window_list {
                window.push(refs.take(shape));
            }
        }
        assert!(
            u64::from(refs.next) < u64::from(segments) * u64::from(facts.segment_chunks),
            "the lane's chunks lie in the committed segments"
        );
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
        // `PIE_GOLDEN_BULK=<slots>`: before the sixth step, that many kv
        // slots this lane does not name go out to the host tier, and the
        // step is timed: beside the move where it rides the swap stream,
        // behind it where it rides the compute stream.
        let bulk = std::env::var("PIE_GOLDEN_BULK")
            .ok()
            .and_then(|slots| slots.parse::<u32>().ok())
            .filter(|_| at == 5);
        if let (Some(slots), Some(shape)) = (bulk, list) {
            let out: Vec<(u32, u32)> = (0..slots)
                .map(|_| (refs.take(shape), host.take(&facts, shape)))
                .collect();
            let began = std::time::Instant::now();
            shell
                .copy_tier_slots(facts.shapes[shape].id, Tier::Device, Tier::Host, &out)
                .expect("the slots go out");
            println!(
                "bulk {}: {slots} slots of {} bytes enqueued in {:.1} ms",
                fixture.model,
                facts.shapes[shape].slot_bytes,
                began.elapsed().as_secs_f64() * 1e3
            );
        }
        let fired = std::time::Instant::now();
        let seated = Seated {
            pages: &pages,
            held: Some(fed),
            window: &window,
            rs_slot: record,
            ..Seated::of(Lane {
                slot: 0,
                word,
                tokens: &tokens,
            })
        };
        let mut rows = shell
            .fire_media(&[seated], &[], &[])
            .expect("the step fires");
        if bulk.is_some() {
            println!(
                "bulk {}: the step beside it took {:.1} ms",
                fixture.model,
                fired.elapsed().as_secs_f64() * 1e3
            );
        }
        let logits = rows.swap_remove(0);
        fed = after;
        if let Some(retained) = window_list.and_then(|shape| facts.shapes[shape].kind.window()) {
            let behind = (fed.saturating_sub(retained) / PAGE) as usize;
            for slot in window.iter_mut().take(behind) {
                *slot = 0;
            }
        }
        let top = argmax(&logits);
        last = Some(top);
        steps.push((hash_logits(&logits), top));
        if offload && (at == 2 || at == 6) {
            let kv_tier = if at == 2 { Tier::Host } else { Tier::Disk };
            if let Some(shape) = list {
                pages = park(&mut shell, &mut refs, &mut host, kv_tier, shape, &pages);
            }
            if let Some(shape) = window_list {
                window = park(&mut shell, &mut refs, &mut host, Tier::Host, shape, &window);
            }
            if let Some(shape) = fixed {
                pieces = park(&mut shell, &mut refs, &mut host, Tier::Host, shape, &pieces);
                record = Some(shell.name_record(&pieces).expect("the record is named"));
            }
        }
    }
    steps
}

/// CONTRACT-BC 22: a lane offloaded mid-decode (every slot it holds to the
/// host tier and back, then its kv pages through the slot file) serves the
/// never-offloaded lane's logits bit for bit, in one process.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_offload() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let mut moved = Vec::new();
    for model in models() {
        let fixture = fixture(&model, &cache);
        // The warming serving: every product's algorithm picked and kept.
        let _ = served_offloaded(&fixture, &cache, false);
        let kept = served_offloaded(&fixture, &cache, false);
        let offloaded = served_offloaded(&fixture, &cache, true);
        for (at, (kept, offloaded)) in kept.iter().zip(&offloaded).enumerate() {
            let same = if kept == offloaded { "same" } else { "MOVED" };
            println!(
                "offload {model} {at} {:016x} {:016x} {same}",
                kept.0, offloaded.0
            );
        }
        if kept != offloaded {
            moved.push(model);
        }
    }
    assert!(moved.is_empty(), "offloaded logits moved: {moved:?}");
}

fn probe(served: fn(&Fixture, &Path) -> Vec<(u64, u32)>) {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let mut lines = Vec::new();
    for model in models() {
        let fixture = fixture(&model, &cache);
        // The warming serving: every product's algorithm picked and kept.
        let _ = served(&fixture, &cache);
        let steps = served(&fixture, &cache);
        let mut all = FNV;
        for (at, (hash, top)) in steps.iter().enumerate() {
            all = fnv_bytes(all, &hash.to_le_bytes());
            let line = format!("golden {model} {at} {hash:016x} {top}");
            println!("{line}");
            lines.push(line);
        }
        let line = format!("golden {model} all {all:016x}");
        println!("{line}");
        lines.push(line);
    }
    if let Some(out) = std::env::var_os("PIE_GOLDEN_OUT") {
        std::fs::write(&out, lines.join("\n") + "\n")
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&out).display()));
    }
    if let Some(want) = std::env::var_os("PIE_GOLDEN_CHECK") {
        let text = std::fs::read_to_string(&want)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&want).display()));
        let want: std::collections::BTreeMap<(String, String), String> = text
            .lines()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                (words.next()? == "golden").then_some(())?;
                let (model, step, hash) = (words.next()?, words.next()?, words.next()?);
                Some(((model.to_string(), step.to_string()), hash.to_string()))
            })
            .collect();
        let mut moved = Vec::new();
        for line in &lines {
            let words: Vec<&str> = line.split_whitespace().collect();
            let key = (words[1].to_string(), words[2].to_string());
            match want.get(&key) {
                Some(hash) if hash == words[3] => {}
                Some(hash) => {
                    moved.push(format!("{} step {}: {} was {hash}", key.0, key.1, words[3]))
                }
                None => moved.push(format!("{} step {}: not in the golden file", key.0, key.1)),
            }
        }
        assert!(moved.is_empty(), "logits moved:\n  {}", moved.join("\n  "));
    }
}

/// The rows a page of the load's rs buffer holds: the rs list's tokens.
fn rs_page(shell: &mut Shell) -> u32 {
    shell
        .chunk_facts()
        .expect("the pool states its chunks")
        .shapes
        .iter()
        .find(|shape| shape.role == engine::chunk::SlotRole::RsBuffer)
        .map_or(PAGE, |shape| match shape.kind {
            engine::chunk::SlotKind::List { tokens, .. } => tokens,
            _ => PAGE,
        })
}

/// The model the buffered probe serves: gated delta net state with a
/// chunked recurrence, whose rows a lane may buffer unfolded.
const BUFFERED: &str = "qwen35-tiny";

/// A prefill folded as it lands, then decodes that buffer their rows past
/// the state and fold them late (none, two, or all of them a step, never
/// more than sixteen unfolded), then a prefill and a decode folded again.
/// Each step's logits' hash and top token.
fn served_buffered(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    use engine::fire::{FoldLen, RsVerb};
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 4,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: knobs(),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    // As a worker boots it: every chunk the fit left the pool backed at
    // load, and the facts stated from it.
    shell.back_whole().expect("the fitted pool is backed whole");
    shell
        .chunk_facts()
        .expect("the pool states its chunks")
        .validate()
        .expect("the engine's chunk facts hold");
    let page = rs_page(&mut shell);
    shell.open(0).expect("slot 0 opens");
    let mut steps = Vec::new();
    let mut fed = 0u32;
    // The slots the state has folded.
    let mut folded = 0u32;
    let mut step = |shell: &mut Shell, tokens: &[u32], rs: RsVerb| {
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
        let seated = engine_cuda::Seated {
            rs,
            ..engine_cuda::Seated::of(Lane {
                slot: 0,
                word,
                tokens,
            })
        };
        let mut rows = shell
            .fire_media(&[seated], &[], &[])
            .expect("the step fires");
        let logits = rows.swap_remove(0);
        let top = argmax(&logits);
        steps.push((hash_logits(&logits), top));
        top
    };
    let prompt = |from: u32, rows: u32| -> Vec<u32> {
        (from..from + rows).map(|t| (t * 7 + 11) % 1000).collect()
    };
    let mut last = step(&mut shell, &prompt(fed, 37), RsVerb::Fold);
    fed += 37;
    folded += 37;
    for at in 0..40u32 {
        // The buffer's pages from the page the state's last slot lies in.
        let base = folded / page * page;
        let end = fed + 1;
        // Halfway, the buffer's pages are copied onto slots of their own
        // and the lane names those from there on, as a fork's first write
        // into a shared tail page does.
        if at == 20 {
            let pairs: Vec<[u32; 2]> = (base / page..fed.div_ceil(page))
                .map(|at| [at % 64, at % 64 + 64])
                .collect();
            shell
                .copy_whole_slots(engine_cuda::store::layout::RS, &pairs)
                .expect("the buffer's pages copy");
        }
        let shift = if at >= 20 { 64 } else { 0 };
        let pages: Vec<u32> = (base / page..end.div_ceil(page))
            .map(|at| at % 64 + shift)
            .collect();
        let unfolded = end - folded;
        let fold = match at % 7 {
            _ if at == 39 || unfolded >= 14 => unfolded,
            3 => 2.min(unfolded),
            6 => unfolded,
            _ => 0,
        };
        last = step(
            &mut shell,
            &[last],
            RsVerb::Buffer {
                pages,
                base,
                folded,
                fold: FoldLen::Host(fold),
            },
        );
        fed = end;
        folded += fold;
    }
    assert_eq!(
        folded, fed,
        "every row is folded before the lane folds as it lands"
    );
    last = step(&mut shell, &prompt(fed, 50), RsVerb::Fold);
    step(&mut shell, &[last], RsVerb::Fold);
    steps
}

/// The buffered probe ([`served_buffered`]): `golden <model>-buffered ...`
/// lines, written to `PIE_GOLDEN_OUT` and compared with `PIE_GOLDEN_CHECK`
/// as the plain probe's are.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_buffered() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let fixture = fixture(BUFFERED, &cache);
    let _ = served_buffered(&fixture, &cache);
    let steps = served_buffered(&fixture, &cache);
    let mut lines = Vec::new();
    let mut all = FNV;
    for (at, (hash, top)) in steps.iter().enumerate() {
        all = fnv_bytes(all, &hash.to_le_bytes());
        lines.push(format!("golden {BUFFERED}-buffered {at} {hash:016x} {top}"));
    }
    lines.push(format!("golden {BUFFERED}-buffered all {all:016x}"));
    for line in &lines {
        println!("{line}");
    }
    let text = lines.join("\n") + "\n";
    if let Some(out) = std::env::var_os("PIE_GOLDEN_OUT") {
        std::fs::write(&out, &text)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&out).display()));
    }
    if let Some(want) = std::env::var_os("PIE_GOLDEN_CHECK") {
        let want = std::fs::read_to_string(&want)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&want).display()));
        let held: Vec<&str> = want
            .lines()
            .filter(|line| line.contains("-buffered "))
            .collect();
        assert_eq!(
            held,
            lines.iter().map(String::as_str).collect::<Vec<_>>(),
            "the buffered logits moved"
        );
    }
}

/// Two lanes of two seats served in one fire a step (a 37-row prefill each,
/// of prompts of their own, then decodes): each step's two rows of logits
/// hashed together. Recurrent kernels that batch lanes are probed by it.
fn served_pair(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(2, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 2,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let mut steps = Vec::new();
    let mut tokens: [Vec<u32>; 2] = [
        (0..37).map(|t| (t * 7 + 11) % 1000).collect(),
        (0..37).map(|t| (t * 13 + 5) % 1000).collect(),
    ];
    for _ in 0..8 {
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens[0].len()).expect("a count"),
            false,
        ));
        // `PIE_GOLDEN_SWAP`: the second seat's lane first in the fire, the
        // rows hashed in the seats' order all the same.
        let swap = std::env::var_os("PIE_GOLDEN_SWAP").is_some();
        let order = if swap { [1usize, 0] } else { [0, 1] };
        let lanes: Vec<Lane<'_>> = order
            .iter()
            .map(|&seat| Lane {
                slot: seat as u32,
                word,
                tokens: &tokens[seat],
            })
            .collect();
        let mut rows = shell.fire(&lanes).expect("the step fires");
        if swap {
            rows.swap(0, 1);
        }
        let tops = [argmax(&rows[0]), argmax(&rows[1])];
        let hash = fnv_bytes(hash_logits(&rows[0]), &hash_logits(&rows[1]).to_le_bytes());
        steps.push((hash, tops[0]));
        tokens = [vec![tops[0]], vec![tops[1]]];
    }
    steps
}

/// The pair probe ([`served_pair`]) over `PIE_GOLDEN_MODELS`: `golden
/// <model>-pair ...` lines, to `PIE_GOLDEN_OUT`.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_pair() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let mut lines = Vec::new();
    for model in models() {
        let fixture = fixture(&model, &cache);
        let _ = served_pair(&fixture, &cache);
        let steps = served_pair(&fixture, &cache);
        for (at, (hash, top)) in steps.iter().enumerate() {
            lines.push(format!("golden {model}-pair {at} {hash:016x} {top}"));
        }
    }
    for line in &lines {
        println!("{line}");
    }
    if let Some(out) = std::env::var_os("PIE_GOLDEN_OUT") {
        std::fs::write(&out, lines.join("\n") + "\n")
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&out).display()));
    }
}

/// Two lanes of two seats that buffer their rows past the state in one fire
/// a step: a 12-row prefill each folded as it lands, then decodes folded
/// late (none, two, or all a step), the lanes out of step with each other.
fn served_pair_buffered(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    use engine::fire::{FoldLen, RsVerb};
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(2, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 4,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    let page = rs_page(&mut shell);
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let mut steps = Vec::new();
    let mut tokens: [Vec<u32>; 2] = [
        (0..12).map(|t| (t * 7 + 11) % 1000).collect(),
        (0..12).map(|t| (t * 13 + 5) % 1000).collect(),
    ];
    let mut fed = [0u32; 2];
    let mut folded = [0u32; 2];
    for at in 0..24u32 {
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens[0].len()).expect("a count"),
            false,
        ));
        let verbs: Vec<(RsVerb, u32)> = (0..2usize)
            .map(|lane| {
                let end = fed[lane] + tokens[lane].len() as u32;
                let unfolded = end - folded[lane];
                let fold = match (at + 3 * lane as u32) % 7 {
                    _ if at == 0 || at == 23 || unfolded >= 14 => unfolded,
                    3 => 2.min(unfolded),
                    6 => unfolded,
                    _ => 0,
                };
                let base = folded[lane] / page * page;
                let pages = (base / page..end.div_ceil(page))
                    .map(|at| (at % 64) * 2 + lane as u32)
                    .collect();
                (
                    RsVerb::Buffer {
                        pages,
                        base,
                        folded: folded[lane],
                        fold: FoldLen::Host(fold),
                    },
                    fold,
                )
            })
            .collect();
        let seated: Vec<engine_cuda::Seated<'_>> = (0..2usize)
            .map(|lane| engine_cuda::Seated {
                rs: verbs[lane].0.clone(),
                ..engine_cuda::Seated::of(Lane {
                    slot: lane as u32,
                    word,
                    tokens: &tokens[lane],
                })
            })
            .collect();
        let rows = shell.fire_media(&seated, &[], &[]).expect("the step fires");
        let tops = [argmax(&rows[0]), argmax(&rows[1])];
        let hash = fnv_bytes(hash_logits(&rows[0]), &hash_logits(&rows[1]).to_le_bytes());
        steps.push((hash, tops[0]));
        for lane in 0..2 {
            fed[lane] += tokens[lane].len() as u32;
            folded[lane] += verbs[lane].1;
        }
        tokens = [vec![tops[0]], vec![tops[1]]];
    }
    steps
}

/// The buffered pair probe ([`served_pair_buffered`]) on the buffered
/// probe's model: `golden <model>-pair-buffered ...` lines, to
/// `PIE_GOLDEN_OUT`.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_pair_buffered() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let fixture = fixture(BUFFERED, &cache);
    let _ = served_pair_buffered(&fixture, &cache);
    let steps = served_pair_buffered(&fixture, &cache);
    let lines: Vec<String> = steps
        .iter()
        .enumerate()
        .map(|(at, (hash, top))| format!("golden {BUFFERED}-pair-buffered {at} {hash:016x} {top}"))
        .collect();
    for line in &lines {
        println!("{line}");
    }
    if let Some(out) = std::env::var_os("PIE_GOLDEN_OUT") {
        std::fs::write(&out, lines.join("\n") + "\n")
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&out).display()));
    }
}

/// [`served_pair`] with each lane naming slot refs, the second lane's opened
/// first and its record far from the first's: the records and pages the two
/// lanes read are scattered and out of the lanes' order, and the logits are
/// those of two seats side by side.
fn served_pair_by_refs(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    use engine::chunk::SlotKind;
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(2, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 2,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: knobs(),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    let mut refs = Refs::of(facts.clone(), 1);
    let list = refs.shape(|kind| matches!(kind, SlotKind::List { window: None, .. }));
    let window_list = refs.shape(|kind| kind.window().is_some());
    let fixed = refs.shape(|kind| matches!(kind, SlotKind::Fixed { .. }));
    shell.take_refs();
    shell.commit_segments(1).expect("the first segment commits");
    // The second lane's record first, then chunks of slots nobody reads,
    // then the first lane's.
    let mut records = [None, None];
    for lane in [1usize, 0] {
        if let Some(shape) = fixed {
            let SlotKind::Fixed { pieces } = facts.shapes[shape].kind else {
                unreachable!("the fixed shape")
            };
            for _ in 0..3 {
                refs.take(shape);
            }
            let pieces: Vec<u32> = (0..pieces).map(|_| refs.take(shape)).collect();
            records[lane] = Some(shell.name_record(&pieces).expect("the record is named"));
        }
    }
    let mut pages: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    let mut window: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    let mut fed = 0u32;
    let mut steps = Vec::new();
    let mut tokens: [Vec<u32>; 2] = [
        (0..37).map(|t| (t * 7 + 11) % 1000).collect(),
        (0..37).map(|t| (t * 13 + 5) % 1000).collect(),
    ];
    for _ in 0..8 {
        let after = fed + tokens[0].len() as u32;
        for lane in [1usize, 0] {
            while (pages[lane].len() as u32) < after.div_ceil(PAGE) {
                if let Some(shape) = list {
                    pages[lane].push(refs.take(shape));
                }
                if let Some(shape) = window_list {
                    window[lane].push(refs.take(shape));
                }
            }
        }
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens[0].len()).expect("a count"),
            false,
        ));
        let seated: Vec<Seated<'_>> = (0..2usize)
            .map(|lane| Seated {
                pages: &pages[lane],
                held: Some(fed),
                window: &window[lane],
                rs_slot: records[lane],
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word,
                    tokens: &tokens[lane],
                })
            })
            .collect();
        let rows = shell.fire_media(&seated, &[], &[]).expect("the step fires");
        fed = after;
        let tops = [argmax(&rows[0]), argmax(&rows[1])];
        let hash = fnv_bytes(hash_logits(&rows[0]), &hash_logits(&rows[1]).to_le_bytes());
        steps.push((hash, tops[0]));
        tokens = [vec![tops[0]], vec![tops[1]]];
    }
    steps
}

/// The pair probe by slot refs ([`served_pair_by_refs`]): the lines of
/// [`golden_pair`], to compare with its file.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_pair_by_refs() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let mut lines = Vec::new();
    for model in models() {
        let fixture = fixture(&model, &cache);
        let _ = served_pair_by_refs(&fixture, &cache);
        let steps = served_pair_by_refs(&fixture, &cache);
        for (at, (hash, top)) in steps.iter().enumerate() {
            lines.push(format!("golden {model}-pair {at} {hash:016x} {top}"));
        }
    }
    for line in &lines {
        println!("{line}");
    }
    if let Some(out) = std::env::var_os("PIE_GOLDEN_OUT") {
        std::fs::write(&out, lines.join("\n") + "\n")
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&out).display()));
    }
}

/// Whether a lane's logits depend on what shares its fire: two seats'
/// prefills of `PIE_GOLDEN_ROWS` rows (48) fired as one fire of two lanes,
/// and as two fires of one lane each, on `PIE_GOLDEN_MODELS`. Prints, a
/// lane, whether the two servings agree bit for bit and the most they
/// differ by; asserts nothing (a batched product may sum in another order).
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_batching() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let rows: u32 = std::env::var("PIE_GOLDEN_ROWS")
        .ok()
        .and_then(|rows| rows.parse().ok())
        .unwrap_or(48);
    for model in models() {
        let fixture = fixture(&model, &cache);
        let prompts: [Vec<u32>; 2] = [
            (0..rows).map(|t| (t * 7 + 11) % 1000).collect(),
            (0..rows).map(|t| (t * 13 + 5) % 1000).collect(),
        ];
        let word = fixture.trace.facts.word(&Request::new(rows, false));
        let load = || {
            let mut shell = Shell::load(Boot {
                trace: fixture.trace.clone(),
                contract: &fixture.contract,
                checkpoint: &fixture.container,
                budget: Budget::new(2, TOKENS),
                patches: None,
                voxels: None,
                page_size: PAGE,
                context: CONTEXT,
                slots: 2,
                kv_budget: None,
                memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
                ordinal: 0,
                knobs: knobs(),
                cache_dir: Some(&cache),
                runahead: engine::runahead::Runahead::F1,
                residency: engine_cuda::experts::Plan::default(),
                deferred_tier: false,
                world: engine_cuda::World::default(),
                expert: engine_cuda::World::default(),
                comm: core::ptr::null_mut(),
            })
            .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
            shell.open(0).expect("slot 0 opens");
            shell.open(1).expect("slot 1 opens");
            shell
        };
        let lane = |seat: usize| Lane {
            slot: seat as u32,
            word,
            tokens: &prompts[seat],
        };
        // Each serving twice: the first warms the products' algorithms.
        let together = || {
            let mut shell = load();
            shell.fire(&[lane(0), lane(1)]).expect("the pair fires")
        };
        let apart = || {
            let mut shell = load();
            let mut first = shell.fire(&[lane(0)]).expect("the first lane fires");
            let mut second = shell.fire(&[lane(1)]).expect("the second lane fires");
            vec![first.swap_remove(0), second.swap_remove(0)]
        };
        let _ = (together(), apart());
        let (together, apart) = (together(), apart());
        for seat in 0..2 {
            let (a, b) = (&together[seat], &apart[seat]);
            let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            let most = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            println!(
                "batching {model} lane {seat}: {} (most {most:e}, argmax {} / {})",
                if same { "identical" } else { "differs" },
                argmax(a),
                argmax(b)
            );
        }
    }
}

fn contract_of(trace: &Trace, path: &Path) -> std::result::Result<ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .map_err(|why| why.to_string())
}

/// The buffered script ([`served_buffered`]) driven as the runtime drives an
/// engine: through `Engine::load`, `commit_segments` and `submit`, each lane
/// naming slot refs — its kv pages in `Lane::tables`, its record in
/// `Lane::fixed` (written in place), its unfolded rows in rs-list slots
/// through `RsVerb::Buffer::pages`, and, halfway, those pages copied onto
/// slots of their own by `Lane::slot_copies`.
fn submitted_buffered(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    use engine::chunk::{SlotKind, SlotRole};
    use engine::fire::{FoldLen, FrameSubmission, KvDelta, RsVerb, Step};
    use engine::{Engine, load};
    let boot = engine_cuda::DeviceBoot {
        cache_dir: Some(cache.to_path_buf()),
        ..engine_cuda::DeviceBoot::default()
    };
    let mut engine = engine_cuda::open(boot, contract_of).expect("the engine opens");
    let loaded = engine
        .load(load::LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: load::Checkpoint::Path(fixture.container.clone()),
            budgets: load::Budgets {
                max_lanes: 1,
                max_tokens: TOKENS,
                buckets: Vec::new(),
                max_adapters: 0,
                page_size: PAGE,
                max_context: CONTEXT,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: load::Residency::default(),
            ordinal: 0,
            frames_in_flight: 1,
        })
        .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    let facts = loaded
        .caps
        .chunks
        .clone()
        .expect("the engine states chunks");
    facts.validate().expect("the engine's chunk facts hold");
    let shape = |want: fn(&engine::chunk::SlotShape) -> bool| {
        facts
            .shapes
            .iter()
            .position(want)
            .expect("the shape is stated")
    };
    let list = shape(|s| matches!(s.kind, SlotKind::List { .. }) && s.role == SlotRole::State);
    let rs = shape(|s| s.role == SlotRole::RsBuffer);
    let fixed = shape(|s| matches!(s.kind, SlotKind::Fixed { .. }));
    let ids = [list, rs, fixed].map(|at| facts.shapes[at].id);
    let SlotKind::List { tokens: page, .. } = facts.shapes[rs].kind else {
        unreachable!("the rs list")
    };
    let SlotKind::Fixed { pieces } = facts.shapes[fixed].kind else {
        unreachable!("the fixed shape")
    };
    engine
        .commit_segments(facts.segments.committed.max(1))
        .expect("the first segment commits");
    let mut refs = Refs::of(facts.clone(), 1);
    let record: Vec<[u32; 2]> = (0..pieces)
        .map(|_| {
            let slot = refs.take(fixed);
            [slot, slot]
        })
        .collect();
    let mut pages: Vec<u32> = Vec::new();
    // The rs buffer's pages by their place in the sequence.
    let mut buffered: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    let mut steps = Vec::new();
    let mut fed = 0u32;
    let mut folded = 0u32;
    let mut fire = |engine: &mut engine_cuda::Cuda,
                    refs: &mut Refs,
                    pages: &mut Vec<u32>,
                    fed: u32,
                    tokens: Vec<u32>,
                    rs: RsVerb,
                    copies: Vec<[u32; 2]>| {
        let after = fed + tokens.len() as u32;
        while (pages.len() as u32) < after.div_ceil(PAGE) {
            pages.push(refs.take(list));
        }
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
        let lane = engine::fire::Lane {
            slot: 0,
            word,
            tokens,
            kv: KvDelta {
                held: fed,
                pages: pages.clone(),
                ..KvDelta::default()
            },
            rs,
            tables: vec![(ids[0], pages.clone())],
            fixed: vec![(ids[2], record.clone())],
            slot_copies: if copies.is_empty() {
                Vec::new()
            } else {
                vec![(ids[1], copies)]
            },
            ..engine::fire::Lane::default()
        };
        lane.validate_chunks(&facts)
            .expect("the lane's refs are the pool's");
        let mut ticket = engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![lane],
                ..Step::default()
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let logits = &ticket.steps[0].readouts[0].values;
        let top = argmax(logits);
        steps.push((hash_logits(logits), top));
        top
    };
    let prompt = |from: u32, rows: u32| -> Vec<u32> {
        (from..from + rows).map(|t| (t * 7 + 11) % 1000).collect()
    };
    let mut last = fire(
        &mut engine,
        &mut refs,
        &mut pages,
        fed,
        prompt(fed, 37),
        RsVerb::Fold,
        Vec::new(),
    );
    fed += 37;
    folded += 37;
    for at in 0..40u32 {
        let base = folded / page * page;
        let end = fed + 1;
        buffered.retain(|&held, _| held >= base / page);
        let mut copies = Vec::new();
        if at == 20 {
            for slot in buffered.values_mut() {
                let fresh = refs.take(rs);
                copies.push([*slot, fresh]);
                *slot = fresh;
            }
        }
        let named: Vec<u32> = (base / page..end.div_ceil(page))
            .map(|held| *buffered.entry(held).or_insert_with(|| refs.take(rs)))
            .collect();
        let unfolded = end - folded;
        let fold = match at % 7 {
            _ if at == 39 || unfolded >= 14 => unfolded,
            3 => 2.min(unfolded),
            6 => unfolded,
            _ => 0,
        };
        last = fire(
            &mut engine,
            &mut refs,
            &mut pages,
            fed,
            vec![last],
            RsVerb::Buffer {
                pages: named,
                base,
                folded,
                fold: FoldLen::Host(fold),
            },
            copies,
        );
        fed = end;
        folded += fold;
    }
    last = fire(
        &mut engine,
        &mut refs,
        &mut pages,
        fed,
        prompt(fed, 50),
        RsVerb::Fold,
        Vec::new(),
    );
    fed += 50;
    fire(
        &mut engine,
        &mut refs,
        &mut pages,
        fed,
        vec![last],
        RsVerb::Fold,
        Vec::new(),
    );
    steps
}

/// The buffered probe through `Engine::submit` ([`submitted_buffered`]):
/// the lines of [`golden_buffered`], compared with `PIE_GOLDEN_CHECK` as its
/// are.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_submit_buffered() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let fixture = fixture(BUFFERED, &cache);
    let _ = submitted_buffered(&fixture, &cache);
    let steps = submitted_buffered(&fixture, &cache);
    let mut lines = Vec::new();
    let mut all = FNV;
    for (at, (hash, top)) in steps.iter().enumerate() {
        all = fnv_bytes(all, &hash.to_le_bytes());
        lines.push(format!("golden {BUFFERED}-buffered {at} {hash:016x} {top}"));
    }
    lines.push(format!("golden {BUFFERED}-buffered all {all:016x}"));
    for line in &lines {
        println!("{line}");
    }
    if let Some(want) = std::env::var_os("PIE_GOLDEN_CHECK") {
        let want = std::fs::read_to_string(&want)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&want).display()));
        let held: Vec<&str> = want
            .lines()
            .filter(|line| line.contains("-buffered ") && !line.contains("-pair-"))
            .collect();
        assert_eq!(
            held,
            lines.iter().map(String::as_str).collect::<Vec<_>>(),
            "the buffered logits moved"
        );
    }
}

/// The split probe: each of `PIE_GOLDEN_MODELS`' last-row logits after one
/// prefill of `n` tokens against a prefill of `n - 1` and a decode of the
/// last. The two take different kernel arms over the same tokens, so what
/// they differ by is the model's own amplification of a reduction's order:
/// printed, never asserted.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_split() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    for model in models() {
        let fixture = fixture(&model, &cache);
        for n in [37u32, 48, 250] {
            let tokens: Vec<u32> = (0..n).map(|t| (t * 7 + 11) % 1000).collect();
            let mut whole = load(&fixture, &cache);
            whole.open(0).expect("slot 0 opens");
            let one = fire(&mut whole, &fixture.trace, &tokens);
            drop(whole);
            let mut split = load(&fixture, &cache);
            split.open(0).expect("slot 0 opens");
            let _ = fire(&mut split, &fixture.trace, &tokens[..tokens.len() - 1]);
            let two = fire(&mut split, &fixture.trace, &tokens[tokens.len() - 1..]);
            let most = one
                .iter()
                .zip(&two)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let span = one.iter().fold(0f32, |m, v| m.max(v.abs()));
            println!(
                "split {model} n={n}: most |diff| {most:.4} of logits up to {span:.2}, top {} vs {}",
                argmax(&one),
                argmax(&two)
            );
        }
    }
}

/// Where the row-streamed table's blocks lie while [`served_rows`] serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rows {
    /// Every block read from the mapped host copy: no slot named.
    Host,
    /// Every block cached in a pool slot.
    Cached,
    /// Every other block cached, the rest read from the host copy.
    Mixed,
}

/// [`served`] with the model's row-streamed table (a token-id-gathered
/// table the weight cache streams) laid as `rows` says: the logits are the
/// dense table's whichever way. Also returns the blocks the fires read, as
/// the engine's use record counts them.
fn served_rows(fixture: &Fixture, cache: &Path, rows: Rows) -> (Vec<(u64, u32)>, usize) {
    use engine::chunk::{ResidentWeight, WeightAtom, WeightResidency};
    // The table's rows alone: the model's routed experts dense, as the
    // base engine seats them.
    let mut knobs = knobs();
    knobs.diagnostics.dense_weights = false;
    knobs.diagnostics.dense_experts = true;
    let mut shell = load_with(fixture, cache, knobs);
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    facts.validate().expect("the engine's chunk facts hold");
    let table = facts
        .weights
        .first()
        .unwrap_or_else(|| panic!("{} streams no table by its rows", fixture.model))
        .clone();
    // What the load's own fires read is not the script's.
    let _ = shell.weight_uses().expect("the use record reads");
    if rows != Rows::Host {
        // The table's slots from the pool's first chunk on, as a runtime's
        // allocator lays them: where this load's own fires seated pages.
        let cached: Vec<u32> = (0..table.count)
            .filter(|atom| rows == Rows::Cached || atom % 2 == 0)
            .collect();
        let first = 1;
        let chunks = (cached.len() as u32).div_ceil(table.slots_per_chunk);
        shell
            .commit_segments((first + chunks).div_ceil(facts.segment_chunks))
            .expect("the table's segments commit");
        let resident = cached
            .iter()
            .enumerate()
            .map(|(at, &atom)| ResidentWeight {
                atom: WeightAtom {
                    shape: table.id,
                    atom,
                },
                slot: first * table.slots_per_chunk + at as u32,
            })
            .collect();
        shell
            .weight_residency(&WeightResidency {
                staging: Vec::new(),
                resident,
                held: false,
            })
            .expect("the blocks are cached");
    }
    shell.open(0).expect("slot 0 opens");
    let mut fed = 0u32;
    let mut last = None;
    let mut steps = Vec::with_capacity(SCRIPT.len());
    for step in SCRIPT {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        fed += u32::try_from(tokens.len()).expect("a count");
        let logits = fire(&mut shell, &fixture.trace, &tokens);
        let top = argmax(&logits);
        last = Some(top);
        steps.push((hash_logits(&logits), top));
    }
    let uses = shell.weight_uses().expect("the use record reads");
    // The table's reads and its blocks, apart from any other weight shape's
    // (a model's routed experts).
    let of_table: Vec<_> = uses
        .uses
        .iter()
        .filter(|(atom, _)| atom.shape == table.id)
        .collect();
    let read: u64 = of_table.iter().map(|(_, reads)| u64::from(*reads)).sum();
    assert_eq!(
        read,
        u64::from(fed),
        "a read of the table a token the script fed"
    );
    // The record counts the table's blocks by where they lie.
    let cached = match rows {
        Rows::Host => 0,
        Rows::Cached => table.count,
        Rows::Mixed => table.count.div_ceil(2),
    };
    assert_eq!(
        uses.held.iter().find(|held| held.shape == table.id),
        Some(&engine::chunk::WeightHeld {
            shape: table.id,
            resident: cached,
            host: table.count - cached,
        })
    );
    (steps, of_table.len())
}

/// The row-cache probe: `PIE_GOLDEN_MODELS` (the models with a row-streamed
/// table; DeepSeek V4's miniature by default) served with the table's rows
/// read from the host copy, cached in the pool, and half and half. Each is
/// compared with `PIE_GOLDEN_CHECK` (the plain probe's file) as
/// [`golden_logits`] is.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_rows() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["dsv4-flash-mini".to_string()], |_| models());
    let want = std::env::var_os("PIE_GOLDEN_CHECK").map(|path| {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&path).display()))
    });
    for model in models {
        let fixture = fixture(&model, &cache);
        // The warming serving: every product's algorithm picked and kept.
        let _ = served_rows(&fixture, &cache, Rows::Host);
        for rows in [Rows::Host, Rows::Cached, Rows::Mixed] {
            let (steps, blocks) = served_rows(&fixture, &cache, rows);
            let mut all = FNV;
            for (hash, _) in &steps {
                all = fnv_bytes(all, &hash.to_le_bytes());
            }
            let line = format!("golden {model} all {all:016x}");
            println!("{line} ({rows:?}, {blocks} blocks read)");
            if let Some(want) = &want {
                assert!(
                    want.lines().any(|held| held == line),
                    "{model} with its table's rows {rows:?}: the logits moved ({line})"
                );
            }
        }
    }
}

/// A prefill of 37 rows, then 300 decodes of fixed tokens on `shell`: the
/// median and the mean decode fire, in microseconds.
fn decode_times(shell: &mut Shell, trace: &Trace) -> (u128, u128) {
    let prompt: Vec<u32> = (0..37u32).map(|t| (t * 7 + 11) % 1000).collect();
    let _ = fire(shell, trace, &prompt);
    let mut took: Vec<u128> = (0..300u32)
        .map(|at| {
            let start = std::time::Instant::now();
            let _ = fire(shell, trace, &[(at * 13 + 5) % 1000]);
            start.elapsed().as_micros()
        })
        .collect();
    took.sort_unstable();
    (
        took[took.len() / 2],
        took.iter().sum::<u128>() / took.len() as u128,
    )
}

/// The row cache as a deployment that does not stream its weights holds it:
/// every block of every row-streamed table named a slot before the first
/// fire (the runtime's pinned set), so none is read from the host copy; and
/// what a decode fire takes so, against one reading every row from the
/// host. Compare with the dense table's on the base engine by hand.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_rows_resident() {
    use engine::chunk::{ResidentWeight, WeightAtom, WeightResidency};
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["dsv4-flash-mini".to_string()], |_| models());
    for model in models {
        let fixture = fixture(&model, &cache);
        for resident in [false, true, false, true] {
            let mut knobs = knobs();
            knobs.diagnostics.dense_weights = false;
            knobs.diagnostics.dense_experts = true;
            let mut shell = load_with(&fixture, &cache, knobs);
            let facts = shell.chunk_facts().expect("the pool states its chunks");
            let (_, blocks) = shell.row_blocks();
            assert!(blocks > 0, "{model} streams no table by its rows");
            if resident {
                // Every shape's blocks, each a slot, from the pool's first
                // chunk on, as a runtime's allocator lays them.
                let mut first = 1;
                let mut named = Vec::new();
                for table in &facts.weights {
                    named.extend((0..table.count).map(|atom| ResidentWeight {
                        atom: WeightAtom {
                            shape: table.id,
                            atom,
                        },
                        slot: first * table.slots_per_chunk + atom,
                    }));
                    first += table.count.div_ceil(table.slots_per_chunk);
                }
                shell
                    .commit_segments(first.div_ceil(facts.segment_chunks))
                    .expect("the tables' segments commit");
                shell
                    .weight_residency(&WeightResidency {
                        staging: Vec::new(),
                        resident: named,
                        held: false,
                    })
                    .expect("the blocks are cached");
                assert_eq!(
                    shell.row_blocks(),
                    (blocks, blocks),
                    "a block of a table held whole is read from the host copy"
                );
            } else {
                assert_eq!(shell.row_blocks().0, 0);
            }
            shell.open(0).expect("slot 0 opens");
            let (median, mean) = decode_times(&mut shell, &fixture.trace);
            println!(
                "decode-time {model} {}: median {median} us, mean {mean} us",
                if resident { "resident" } else { "host" }
            );
        }
    }
}

/// [`served`] with the model's routed experts laid as `rows` says (every
/// expert read from the mapped host copy, every one cached in a pool slot,
/// or every other one): each step's logits. Checks the use record's counts
/// of where the experts lie.
fn served_experts(fixture: &Fixture, cache: &Path, rows: Rows) -> Vec<Vec<f32>> {
    served_experts_as(fixture, cache, rows, false)
}

/// Whether a told load's held set names other slots than the load seated
/// its weights in ([`served_experts_as`]).
static MOVED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`served_experts`], the load told at the start that its weights are held
/// (`told`: a deployment that streams nothing says so in its load request),
/// so it keeps no host copy and arms at the residency.
fn served_experts_as(fixture: &Fixture, cache: &Path, rows: Rows, told: bool) -> Vec<Vec<f32>> {
    let began = std::time::Instant::now();
    use engine::chunk::{ResidentWeight, WeightAtom, WeightKind, WeightResidency};
    let mut knobs = knobs();
    knobs.diagnostics.dense_weights = false;
    knobs.weights_held = told;
    let mut shell = load_with(fixture, cache, knobs);
    let loaded = began.elapsed();
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    facts.validate().expect("the engine's chunk facts hold");
    let shapes: Vec<_> = facts
        .weights
        .iter()
        .filter(|shape| shape.kind == WeightKind::Expert)
        .cloned()
        .collect();
    assert!(
        !shapes.is_empty(),
        "{} streams no routed expert",
        fixture.model
    );
    let _ = shell.weight_uses().expect("the use record reads");
    let mut want = Vec::new();
    let mut resident = Vec::new();
    // Each shape's slots in chunks of its own, from the pool's first chunk
    // on, as a runtime's allocator lays them: where this load's own fires
    // seated their pages. A load told its weights are held seated them
    // itself and says where (`WeightShape::seated`): a held set adopts
    // that, or ([`MOVED`]) starts a chunk on, so every atom moves into a
    // slot another may lie in.
    let moved = MOVED.load(std::sync::atomic::Ordering::Relaxed);
    let mut first = if told && moved { 2 } else { 1 };
    // A deployment that streams nothing holds every weight shape whole, the
    // row-streamed tables' too; the other two lay the experts alone.
    let named: Vec<_> = facts
        .weights
        .iter()
        .filter(|shape| rows == Rows::Cached || shape.kind == WeightKind::Expert)
        .cloned()
        .collect();
    for shape in &named {
        let cached: Vec<u32> = (0..shape.count)
            .filter(|atom| match rows {
                Rows::Host => false,
                Rows::Cached => true,
                Rows::Mixed => atom % 2 == 0,
            })
            .collect();
        let adopted = shape
            .seated
            .filter(|_| told && rows == Rows::Cached && !moved);
        if told {
            assert!(
                shape.seated.is_some(),
                "a load told its weights are held states where it seated them"
            );
        }
        resident.extend(cached.iter().enumerate().map(|(at, &atom)| ResidentWeight {
            atom: WeightAtom {
                shape: shape.id,
                atom,
            },
            slot: match adopted {
                Some(seated) => seated + atom,
                None => first * shape.slots_per_chunk + at as u32,
            },
        }));
        first += (cached.len() as u32).div_ceil(shape.slots_per_chunk);
        if shape.kind == WeightKind::Expert {
            want.push(engine::chunk::WeightHeld {
                shape: shape.id,
                resident: cached.len() as u32,
                host: shape.count - cached.len() as u32,
            });
        }
    }
    if !resident.is_empty() {
        shell
            .commit_segments(first.div_ceil(facts.segment_chunks))
            .expect("the experts' segments commit");
        shell
            .weight_residency(&WeightResidency {
                staging: Vec::new(),
                resident,
                // A set that holds every expert is a deployment's that
                // streams nothing.
                held: rows == Rows::Cached,
            })
            .expect("the experts are cached");
    }
    println!(
        "boot {} {rows:?}{}: load {:.1} s, to the residency's end {:.1} s",
        fixture.model,
        if told { " told held" } else { "" },
        loaded.as_secs_f64(),
        began.elapsed().as_secs_f64()
    );
    shell.open(0).expect("slot 0 opens");
    let mut fed = 0u32;
    let mut last = None;
    let mut steps = Vec::with_capacity(SCRIPT.len());
    for step in SCRIPT {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        fed += u32::try_from(tokens.len()).expect("a count");
        let logits = fire(&mut shell, &fixture.trace, &tokens);
        last = Some(argmax(&logits));
        steps.push(logits);
    }
    let uses = shell.weight_uses().expect("the use record reads");
    let held: Vec<_> = uses
        .held
        .iter()
        .filter(|held| shapes.iter().any(|shape| shape.id == held.shape))
        .copied()
        .collect();
    assert_eq!(
        held, want,
        "the record counts the experts by where they lie"
    );
    // A packed bank's selects count no read themselves: the engine counts a
    // fire's routes only while the set may change, so a held set of packed
    // banks alone records none.
    let packed = fixture
        .trace
        .params
        .iter()
        .any(|param| param.dtype == poem_ir::Dtype::Mxfp4);
    assert!(
        (packed && rows == Rows::Cached)
            || uses
                .uses
                .iter()
                .any(|(atom, reads)| shapes.iter().any(|s| s.id == atom.shape) && *reads > 0),
        "the fires read no expert the record counted"
    );
    steps
}

/// The expert-cache probe: `PIE_GOLDEN_MODELS` (the MoE miniatures whose
/// banks stream; Nemotron's by default) served with every routed expert
/// cached in the pool, read from the host copy, and half and half. The
/// cached serving is compared with `PIE_GOLDEN_CHECK` as [`golden_logits`]
/// is: a load that streams nothing reads the experts as the dense bank's
/// arm does. The other two are printed against it (a bank with an expert on
/// the host copy runs its select's plain-load arm), and asserted only where
/// `PIE_GOLDEN_STRICT` is set.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_experts() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["nemotron3-nano-mini".to_string()], |_| models());
    let want = std::env::var_os("PIE_GOLDEN_CHECK").map(|path| {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&path).display()))
    });
    let strict = std::env::var_os("PIE_GOLDEN_STRICT").is_some();
    for model in models {
        let fixture = fixture(&model, &cache);
        // The warming serving: every product's algorithm picked and kept.
        let _ = served_experts(&fixture, &cache, Rows::Cached);
        let cached = served_experts(&fixture, &cache, Rows::Cached);
        let hash = |steps: &[Vec<f32>]| {
            steps.iter().fold(FNV, |all, logits| {
                fnv_bytes(all, &hash_logits(logits).to_le_bytes())
            })
        };
        let line = format!("golden {model} all {:016x}", hash(&cached));
        println!("{line} (Cached)");
        if let Some(want) = &want {
            assert!(
                want.lines().any(|held| held == line),
                "{model} with every expert cached: the logits moved ({line})"
            );
        }
        // The same set on a load told from the start that it is held: no
        // host copy, the lattice armed once.
        let serving = fixture_as(&model, &cache, true);
        let told = served_experts_as(&serving, &cache, Rows::Cached, true);
        println!(
            "golden {model} all {:016x} (Cached, told held)",
            hash(&told)
        );
        assert_eq!(
            hash(&told),
            hash(&cached),
            "{model} told its weights are held"
        );
        // And with the runtime's set a chunk off the load's seating: every
        // atom is filled again where it is named.
        MOVED.store(true, std::sync::atomic::Ordering::Relaxed);
        let moved = served_experts_as(&serving, &cache, Rows::Cached, true);
        MOVED.store(false, std::sync::atomic::Ordering::Relaxed);
        println!(
            "golden {model} all {:016x} (Cached, told held, moved)",
            hash(&moved)
        );
        assert_eq!(
            hash(&moved),
            hash(&cached),
            "{model} told its weights are held, its atoms named elsewhere"
        );
        for rows in [Rows::Host, Rows::Mixed] {
            let steps = served_experts(&fixture, &cache, rows);
            let most = steps
                .iter()
                .zip(&cached)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
                .fold(0f32, f32::max);
            let tops = steps
                .iter()
                .zip(&cached)
                .filter(|(a, b)| argmax(a) != argmax(b))
                .count();
            println!(
                "golden {model} all {:016x} ({rows:?}: most |diff| {most:e} from cached, \
                 {tops} of {} steps' top tokens differ)",
                hash(&steps),
                steps.len()
            );
            if strict {
                assert_eq!(
                    hash(&steps),
                    hash(&cached),
                    "{model} with its experts {rows:?}"
                );
            }
            // A load told its weights are held, which the runtime then
            // streams (its pool does not seat them all): the engine makes
            // its miss copy at the residency, arms again, and serves the
            // same.
            if rows == Rows::Mixed {
                let serving = fixture_as(&model, &cache, true);
                let late = served_experts_as(&serving, &cache, rows, true);
                println!("golden {model} all {:016x} (Mixed, told held)", hash(&late));
                assert_eq!(
                    hash(&late),
                    hash(&steps),
                    "{model} told its weights are held and streamed after all"
                );
            }
        }
    }
}

/// The eager probe: each of `PIE_GOLDEN_MODELS` served over [`SCRIPT`] with
/// its bodies recorded (the default) and with every fire walking its plan
/// (no graph, no body). An eager walk may take other kernels than a capture
/// does, so the two are printed against each other, never asserted: a
/// recorded body that replays wrongly shows as a difference far past a
/// kernel's rounding.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_eager() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    for model in models() {
        let fixture = fixture(&model, &cache);
        let serve = |eager: bool| -> Vec<Vec<f32>> {
            let mut knobs = knobs();
            if eager {
                knobs.chip = Some(engine_cuda::chip::pinned(
                    engine_cuda::Graphs::Off,
                    Some(engine_cuda::Recording::Off),
                ));
            }
            let mut shell = load_with(&fixture, &cache, knobs);
            shell.open(0).expect("slot 0 opens");
            let mut fed = 0u32;
            SCRIPT
                .iter()
                .map(|step| {
                    // Fixed tokens, so the two servings read the same rows.
                    let rows = match *step {
                        Step::Prefill(rows) => rows,
                        Step::Decode => 1,
                    };
                    let tokens: Vec<u32> = (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect();
                    fed += rows;
                    fire(&mut shell, &fixture.trace, &tokens)
                })
                .collect()
        };
        let _ = serve(false);
        let (recorded, eager) = (serve(false), serve(true));
        let most = recorded
            .iter()
            .zip(&eager)
            .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
            .fold(0f32, f32::max);
        let span = recorded.iter().flatten().fold(0f32, |m, v| m.max(v.abs()));
        let tops = recorded
            .iter()
            .zip(&eager)
            .filter(|(a, b)| argmax(a) != argmax(b))
            .count();
        println!(
            "eager {model}: most |diff| {most:e} of logits up to {span:.2}, {tops} of {} \
             steps' top tokens differ",
            recorded.len()
        );
    }
}

/// The boot probe: each of `PIE_GOLDEN_MODELS` (Nemotron's miniature by
/// default) loaded with every routed expert held, timed to the end of the
/// first weight residency: on a load not told its weights are held (it pins
/// their host copy, arms on it, and arms again at the residency) and on one
/// told at load (the weights left in the serving file, armed once). The
/// logits are the same; the times are printed.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_boot() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["nemotron3-nano-mini".to_string()], |_| models());
    for model in models {
        let plain = fixture(&model, &cache);
        let serving = fixture_as(&model, &cache, true);
        let hash = |steps: &[Vec<f32>]| {
            steps.iter().fold(FNV, |all, logits| {
                fnv_bytes(all, &hash_logits(logits).to_le_bytes())
            })
        };
        let untold = served_experts_as(&plain, &cache, Rows::Cached, false);
        let told = served_experts_as(&serving, &cache, Rows::Cached, true);
        assert_eq!(
            hash(&told),
            hash(&untold),
            "{model} told its weights are held"
        );
    }
}

/// [`SCRIPT`] served through the engine's own API (`Engine::load`, the
/// weight residency a runtime that holds the weights starts with, `submit`
/// with slot refs) under `residency`: each step's logits. For a model whose
/// cache rows are kv lists alone.
fn submitted(fixture: &Fixture, cache: &Path, residency: engine::load::Residency) -> Vec<Vec<f32>> {
    use engine::chunk::{ResidentWeight, SlotKind, SlotRole, WeightAtom, WeightResidency};
    use engine::fire::{FrameSubmission, KvDelta, Step as Fired};
    use engine::{Engine, load};
    let boot = engine_cuda::DeviceBoot {
        cache_dir: Some(cache.to_path_buf()),
        ..engine_cuda::DeviceBoot::default()
    };
    let mut engine = engine_cuda::open(boot, contract_of).expect("the engine opens");
    let loaded = engine
        .load(load::LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: load::Checkpoint::Path(fixture.container.clone()),
            budgets: load::Budgets {
                max_lanes: 1,
                max_tokens: TOKENS,
                buckets: Vec::new(),
                max_adapters: 0,
                page_size: PAGE,
                max_context: CONTEXT,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency,
            ordinal: 0,
            frames_in_flight: 1,
        })
        .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    let facts = loaded
        .caps
        .chunks
        .clone()
        .expect("the engine states chunks");
    facts.validate().expect("the engine's chunk facts hold");
    // The weights where the load seated them, held: what a runtime that
    // adopts the engine's seating starts with.
    let mut resident = Vec::new();
    let mut next = 1u32;
    for shape in &facts.weights {
        let first = shape
            .seated
            .unwrap_or_else(|| panic!("{} seated no weight at load", fixture.model));
        resident.extend((0..shape.count).map(|atom| ResidentWeight {
            atom: WeightAtom {
                shape: shape.id,
                atom,
            },
            slot: first + atom,
        }));
        next = next.max((first + shape.count - 1) / shape.slots_per_chunk + 1);
    }
    engine
        .weight_residency(&WeightResidency {
            staging: Vec::new(),
            resident,
            held: true,
        })
        .expect("the weights are held where the load seated them");
    let list = facts
        .shapes
        .iter()
        .position(|s| {
            matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State
        })
        .expect("the kv list is stated");
    assert_eq!(
        facts.shapes.len(),
        1,
        "{}'s cache rows are kv lists alone",
        fixture.model
    );
    // The lane's pages in chunks past the weights'.
    let mut refs = Refs::of(facts.clone(), next);
    engine
        .commit_segments(
            (next + 2)
                .div_ceil(facts.segment_chunks)
                .max(facts.segments.committed),
        )
        .expect("the lane's segment commits");
    let mut pages: Vec<u32> = Vec::new();
    let mut fed = 0u32;
    let mut last = None;
    let mut steps = Vec::with_capacity(SCRIPT.len());
    for step in SCRIPT {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        let after = fed + tokens.len() as u32;
        while (pages.len() as u32) < after.div_ceil(PAGE) {
            pages.push(refs.take(list));
        }
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
        let lane = engine::fire::Lane {
            slot: 0,
            word,
            tokens,
            kv: KvDelta {
                held: fed,
                pages: pages.clone(),
                ..KvDelta::default()
            },
            tables: vec![(facts.shapes[list].id, pages.clone())],
            ..engine::fire::Lane::default()
        };
        lane.validate_chunks(&facts)
            .expect("the lane's refs are the pool's");
        let mut ticket = engine
            .submit(&FrameSubmission::of(Fired {
                lanes: vec![lane],
                ..Fired::default()
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let logits = ticket.steps[0].readouts[0].values.clone();
        fed = after;
        last = Some(argmax(&logits));
        steps.push(logits);
    }
    steps
}

/// The budget probe (through the engine's API): a MoE miniature whose
/// weights the runtime holds, loaded with no `device_weight_budget` and with
/// ones under its dense planes' bytes (`PIE_GOLDEN_BUDGETS`, MiB, comma
/// separated; 512 and 128 by default). Under a budget the dense planes that
/// do not fit go to the host tier and the routed experts stay in the pool,
/// charged to no weight budget: the load serves, and its logits are printed
/// against the unbudgeted load's.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_budget() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["minimax-m3-mini".to_string()], |_| models());
    let budgets: Vec<u64> = std::env::var("PIE_GOLDEN_BUDGETS")
        .unwrap_or_else(|_| "512,128".to_string())
        .split(',')
        .filter_map(|mib| mib.trim().parse().ok())
        .collect();
    let held = engine::load::Residency {
        weights_held: true,
        ..engine::load::Residency::default()
    };
    for model in models {
        let fixture = fixture_as(&model, &cache, true);
        let hash = |steps: &[Vec<f32>]| {
            steps.iter().fold(FNV, |all, logits| {
                fnv_bytes(all, &hash_logits(logits).to_le_bytes())
            })
        };
        let _ = submitted(&fixture, &cache, held.clone());
        let whole = submitted(&fixture, &cache, held.clone());
        println!("golden {model} all {:016x} (no budget)", hash(&whole));
        for &mib in &budgets {
            let steps = submitted(
                &fixture,
                &cache,
                engine::load::Residency {
                    device_weight_budget: Some(mib << 20),
                    ..held.clone()
                },
            );
            let most = steps
                .iter()
                .zip(&whole)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(a, b)| (a - b).abs()))
                .fold(0f32, f32::max);
            let tops = steps
                .iter()
                .zip(&whole)
                .filter(|(a, b)| argmax(a) != argmax(b))
                .count();
            println!(
                "golden {model} all {:016x} (device_weight_budget {mib} MiB: most |diff| \
                 {most:e} from no budget, {tops} of {} steps' top tokens differ)",
                hash(&steps),
                steps.len()
            );
        }
    }
}
