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
/// 512, MiMo's 128), so each ring wraps.
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
        knobs: engine_cuda::Knobs::default(),
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

/// Slot refs handed out as the runtime's pool would: the lowest free chunk
/// opened for a shape whose open chunk is full.
struct Refs {
    facts: engine::chunk::ChunkFacts,
    next_chunk: u32,
    /// Each shape's open chunk and the next index in it.
    open: Vec<Option<(u32, u32)>>,
}

impl Refs {
    fn of(facts: engine::chunk::ChunkFacts) -> Refs {
        let open = vec![None; facts.shapes.len()];
        Refs {
            facts,
            next_chunk: 1,
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
                self.next_chunk += 1;
                (self.next_chunk - 1, 0)
            }
        };
        self.open[shape] = Some((chunk, index + 1));
        chunk * per_chunk + index
    }
}

/// The model served over [`SCRIPT`] as the runtime's shared pool serves it:
/// the lane names slot refs of the engine's stated shapes, every segment its
/// chunks lie in is committed ahead, and after every third step each chunk
/// opened so far is moved onto another physical chunk (an evacuation), the
/// refs standing. The logits are the plain serving's, bit for bit.
fn served_by_refs(fixture: &Fixture, cache: &Path) -> Vec<(u64, u32)> {
    use engine::chunk::SlotKind;
    let mut shell = load(fixture, cache);
    let facts = shell.chunk_facts();
    facts.validate().expect("the engine's chunk facts hold");
    let mut refs = Refs::of(facts.clone());
    let list = refs.shape(|kind| matches!(kind, SlotKind::List { .. }));
    let ring = refs.shape(|kind| matches!(kind, SlotKind::Ring { .. }));
    let fixed = refs.shape(|kind| matches!(kind, SlotKind::Fixed { .. }));
    shell.take_refs();
    // Chunks 1.. hold the lane's slots, and the chunks they move onto lie
    // half a segment up: all in the first segment.
    let spread = facts.segment_chunks.max(2) / 2;
    shell.commit_segments(1).expect("the first segment commits");
    let record = fixed.map(|shape| refs.take(shape));
    let (mut pages, mut window): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    let mut moved = false;
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
            if let Some(shape) = ring {
                window.push(refs.take(shape));
            }
        }
        assert!(
            refs.next_chunk <= spread,
            "the lane's chunks lie under the moves"
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
        let top = argmax(&logits);
        last = Some(top);
        steps.push((hash_logits(&logits), top));
        // Every chunk opened so far changes places with one past the
        // spread, and back the next time.
        if at % 3 == 2 {
            let chunks = 1..refs.next_chunk;
            let moves = chunks
                .map(|logical| (logical, if moved { logical } else { logical + spread }))
                .collect();
            shell
                .evacuate(&engine::chunk::Evacuation { moves })
                .expect("the chunks move");
            moved = !moved;
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
    let mut steps = Vec::new();
    let mut fed = 0u32;
    // The slots the state has folded.
    let mut folded = 0u32;
    let mut last = 0u32;
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
    last = step(&mut shell, &prompt(fed, 37), RsVerb::Fold).max(last * 0);
    fed += 37;
    folded += 37;
    for at in 0..40u32 {
        // The buffer's pages from the page the state's last slot lies in.
        let base = folded / PAGE * PAGE;
        let end = fed + 1;
        let pages: Vec<u32> = (base / PAGE..end.div_ceil(PAGE))
            .map(|page| page % 4)
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
