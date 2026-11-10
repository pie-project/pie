#![cfg(feature = "cuda")]

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

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count: usize = param.shape.iter().product::<u64>() as usize;
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
            other => panic!("`{}` is {other:?}, not a dtype the probe seeds", param.name),
        };
        writer
            .add(param.name.as_str(), param.shape.clone(), leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// `model` traced for CUDA, every plane and its kv cache bf16: a model whose
/// layout states a dtype per component (its routed experts apart) gets
/// bf16 for each.
fn trace_of(model: &str) -> Trace {
    let package = poem_compiler::catalog::package_of(model)
        .unwrap_or_else(|| panic!("no package holds `{model}`"));
    let mut deploy = poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16);
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
    let trace = trace_of(model);
    let dir = cache.join("fixtures");
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    let container = dir.join(format!("{model}.zt"));
    if !container.exists() {
        let partial = dir.join(format!("{model}.zt.{}", std::process::id()));
        write_checkpoint(&partial, &trace);
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
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 1,
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
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model))
}

/// The load's knobs: `PIE_GOLDEN_CHUNK` (bytes) sets the chunk size the
/// state pool aims near, so a miniature's records are cut into pieces as a
/// full model's are; the logits are the same either way.
fn knobs() -> engine_cuda::Knobs {
    let mut knobs = engine_cuda::Knobs::default();
    if let Ok(bytes) = std::env::var("PIE_GOLDEN_CHUNK") {
        knobs.diagnostics.chunk_bytes = Some(bytes.parse().expect("PIE_GOLDEN_CHUNK is bytes"));
    }
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
    let window_list = refs.shape(|kind| kind.window().is_some());
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
    let (mut pages, mut window): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut low = false;
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
            (refs.opened.len() as u32) < spread,
            "the lane's chunks lie under the places they move between"
        );
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
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
        let logits = rows.swap_remove(0);
        fed = after;
        // What the runtime trims: a window list's slot whose last row lies
        // behind the window of the rows held is the null ref from here on.
        if let Some(retained) = window_list.and_then(|shape| facts.shapes[shape].kind.window()) {
            let behind = (fed.saturating_sub(retained) / PAGE) as usize;
            for slot in window.iter_mut().take(behind) {
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
    let mut shell = load(fixture, cache);
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
        // The table's slots from the second segment on, clear of the chunks
        // the seats' pages open in the first.
        let cached: Vec<u32> = (0..table.count)
            .filter(|atom| rows == Rows::Cached || atom % 2 == 0)
            .collect();
        let first = facts.segment_chunks;
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
    let read: u64 = uses.uses.iter().map(|(_, reads)| u64::from(*reads)).sum();
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
        uses.held,
        vec![engine::chunk::WeightHeld {
            shape: table.id,
            resident: cached,
            host: table.count - cached,
        }]
    );
    (steps, uses.uses.len())
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
            let mut shell = load(&fixture, &cache);
            let facts = shell.chunk_facts().expect("the pool states its chunks");
            let (_, blocks) = shell.row_blocks();
            assert!(blocks > 0, "{model} streams no table by its rows");
            if resident {
                // Every shape's blocks, each a slot, from the second segment.
                let mut first = facts.segment_chunks;
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
