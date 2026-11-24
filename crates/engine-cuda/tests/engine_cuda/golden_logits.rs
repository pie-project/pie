#![cfg(feature = "device")]

//! The golden-logits probe: a fixed set of catalog miniatures, one of each
//! cache shape (dense attention, sliding windows, GDN and mamba state,
//! DeepSeek V4's compressed pools, MLA-like and routed rows), each served
//! from a checkpoint seeded by plane name over a fixed script of prefills
//! and greedy decodes, and each step's logits hashed bit for bit. A change
//! that must not move a logit (a memory layout, an address) is proven by
//! the same hashes before and after it. Every model is served once first,
//! so the cache's profile holds every product shape the set meets before
//! any logit is hashed; the hashes are then taken under that profile held
//! fixed, and the output names its md5 (`golden profile md5 <md5>`): a run
//! under another profile may pick other kernels, which a check says.
//!
//! A dense product's kernel is chosen by its shape alone (a streaming
//! skeleton for few rows, else cuBLASLt's first heuristic); nothing is timed
//! at load, so every run sums each product in the same order. The cache is
//! shared across runs (`PIE_GOLDEN_CACHE`, which must be set: a run without
//! it is refused rather than rebuilding every fixture), each fixture named by
//! its model and a hash of the planes it holds. Each model is served twice in its process,
//! and the second serving is the one hashed.
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

/// The hash of a step's logits, bit for bit. `PIE_GOLDEN_LOGITS=<dir>`
/// also writes each row hashed, in order, as `<n>.f32` (little-endian): two
/// runs' rows at a step whose top parts, to weigh the parting against the
/// row's drift.
fn hash_logits(logits: &[f32]) -> u64 {
    if let Some(dir) = std::env::var_os("PIE_GOLDEN_LOGITS") {
        static AT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let at = AT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("the logits directory");
        std::fs::write(dir.join(format!("{at}.f32")), bytes).expect("a logits row is written");
    }
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
    // (Its input scales are its activations' calibration, seeded below.)
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
        // An NVFP4 bank's input scales: each expert's activation amax over
        // the largest NVFP4 value, positive.
        let calibration = param.name.ends_with(&poem::input_scales_name(""));
        let mut seed = fnv(&param.name);
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            seed >> 33
        };
        let unit = |bits: u64| (bits as f32 / (1u64 << 31) as f32) - 0.5;
        let value = |bits: u64| {
            if norm {
                1.0 + 0.05 * unit(bits)
            } else if calibration {
                0.002 + 0.001 * unit(bits)
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

pub(super) struct Fixture {
    pub(super) model: String,
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
}

/// `model`'s trace and its seeded checkpoint, written once into `cache`'s
/// `fixtures` and read back by every later run.
pub(super) fn fixture(model: &str, cache: &Path) -> Fixture {
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
    // Named by the planes it holds as well as the model: a fixture seeded
    // for another trace (a plane renamed or reshaped, another drafter, a
    // seeding changed) is another file, never served for this one.
    let key = fixture_key(&trace, serving);
    let container = dir.join(format!("{model}-{key:016x}.{suffix}"));
    if !container.exists() {
        let partial = dir.join(format!("{model}.{suffix}.{}", std::process::id()));
        if encoded {
            let floats = dir.join(format!("{model}-{key:016x}.floats.zt"));
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
        if encoded {
            // The seeded floats are the import's input alone: the artifact
            // is what every later run reads.
            let _ = std::fs::remove_file(dir.join(format!("{model}-{key:016x}.floats.zt")));
        }
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

/// The version of the bytes a fixture is seeded with: bumped whenever
/// [`seeded_planes`], [`write_checkpoint`], [`write_serving`] or the
/// artifact import write other bytes for the same planes, so no fixture of
/// the old seeding is read again.
const SEEDING: u32 = 1;

/// The name a fixture is cached under past its model's: the seeding's
/// version, the form (serving or not) and each checkpoint plane's name,
/// shape, dtype and layout, hashed.
fn fixture_key(trace: &Trace, serving: bool) -> u64 {
    let mut key = fnv_bytes(FNV, &SEEDING.to_le_bytes());
    key = fnv_bytes(key, &[u8::from(serving)]);
    for param in trace
        .params
        .iter()
        .filter(|p| p.source == ParamSource::Checkpoint)
    {
        key = fnv_bytes(key, param.name.as_bytes());
        key = fnv_bytes(key, &[0]);
        for &extent in &param.shape {
            key = fnv_bytes(key, &extent.to_le_bytes());
        }
        key = fnv_bytes(
            key,
            format!("{:?} {:?};", param.dtype, param.layout).as_bytes(),
        );
    }
    key
}

/// The fixture cache, `PIE_GOLDEN_CACHE`: refused unset, since a default
/// in a run's own temp dir rebuilds every fixture (tens of GB) each run.
pub(super) fn cache_dir() -> PathBuf {
    std::env::var_os("PIE_GOLDEN_CACHE").map_or_else(
        || panic!("set PIE_GOLDEN_CACHE to the fixture cache; unset would rebuild every fixture"),
        PathBuf::from,
    )
}

fn load(fixture: &Fixture, cache: &Path) -> Shell {
    load_with(fixture, cache, knobs())
}

/// The cache rows' bytes a load is given, in MiB; 0 for none
/// ([`golden_held_small`]).
static SMALL_POOL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn load_with(fixture: &Fixture, cache: &Path, knobs: engine_cuda::Knobs) -> Shell {
    let shell = load_under(fixture, cache, knobs, engine::Allowance::WHOLE)
        .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    let (_, outside) = shell.held_and_outside();
    assert_eq!(
        outside,
        shell.ledger().bytes(engine::memory::Class::Outside),
        "{}: the load states outside its span what its ledger enters there",
        fixture.model
    );
    shell
}

/// [`load_with`] under `allowance`, its refusal returned.
pub(super) fn load_under(
    fixture: &Fixture,
    cache: &Path,
    knobs: engine_cuda::Knobs,
    allowance: engine::Allowance,
) -> Result<Shell, String> {
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
        kv_budget: match SMALL_POOL.load(std::sync::atomic::Ordering::Relaxed) {
            0 => std::env::var("PIE_GOLDEN_POOL_MIB")
                .ok()
                .and_then(|mib| mib.parse::<u64>().ok())
                .map(|mib| mib << 20),
            mib => Some(mib << 20),
        }
        .map(engine_cuda::store::KvSize::Bytes),
        allowance,
        context_declared: false,
        ordinal: 0,
        knobs,
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .map_err(|why| why.to_string())
}

/// A load of `fixture` under `allowance` at the model's declared context
/// `declared`: the load serves it as long as its room holds.
fn load_declared(
    fixture: &Fixture,
    cache: &Path,
    allowance: engine::Allowance,
    declared: u32,
) -> Result<Shell, String> {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: declared,
        slots: 1,
        kv_budget: None,
        allowance,
        context_declared: true,
        ordinal: 0,
        knobs: knobs(),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .map_err(|why| why.to_string())
}

/// A declared context is cut to the room the load's own room line leaves,
/// past this process's context and modules: under budgets from where the
/// modules leave no sequence a room to where they leave a long one, a load
/// serves the declared context cut to what it seats, or refuses the card by
/// name, never faulting on residency.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn a_declared_context_is_cut_to_the_room_past_the_modules() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    const DECLARED: u32 = 1 << 24;
    let cache = cache_dir();
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let mut served = 0;
    for mib in (512u64..=2048).step_by(128) {
        let allowance = engine::Allowance {
            budget: Some(mib << 20),
        };
        match load_declared(&fixture, &cache, allowance, DECLARED) {
            Ok(shell) => {
                let context = shell.paging().context();
                println!("budget {mib} MiB: serves {context} of {DECLARED} tokens");
                assert!(context <= DECLARED);
                served += 1;
            }
            Err(why) => {
                println!("budget {mib} MiB: refused: {why}");
                assert!(
                    !why.starts_with("weight residency"),
                    "under {mib} MiB the declared context faulted on residency: {why}"
                );
            }
        }
    }
    assert!(
        served > 0,
        "no budget served the declared context cut to its room"
    );
}

/// The load's knobs: `PIE_GOLDEN_CHUNK` (bytes) sets the chunk size the
/// state pool aims near, so a miniature's records are cut into pieces as a
/// full model's are. The logits are the same either way.
pub(super) fn knobs() -> engine_cuda::Knobs {
    let mut knobs = engine_cuda::Knobs::default();
    knobs.profiles = Some(cache_dir().join("profiles"));
    if let Ok(bytes) = std::env::var("PIE_GOLDEN_CHUNK") {
        knobs.diagnostics.chunk_bytes = Some(bytes.parse().expect("PIE_GOLDEN_CHUNK is bytes"));
    }
    // `PIE_GOLDEN_EAGER`: no graph captured and no body recorded; every
    // fire walks its plan. The logits are a recorded body's.
    if std::env::var_os("PIE_GOLDEN_EAGER").is_some() {
        knobs.pin = Some((engine_cuda::Graphs::Off, Some(engine_cuda::Recording::Off)));
    }
    // `PIE_GOLDEN_ONE_WINDOW`: every windowed row in one window list.
    knobs.diagnostics.one_window = std::env::var_os("PIE_GOLDEN_ONE_WINDOW").is_some();
    knobs.diagnostics.windowed_state = std::env::var_os("PIE_GOLDEN_WINDOWED_STATE").is_some();
    knobs.diagnostics.arm_trace = std::env::var_os("PIE_GOLDEN_ARM_TRACE").is_some();
    // `PIE_GOLDEN_TIER_INLINE`: tier copies ride the compute stream.
    knobs.diagnostics.tier_inline = std::env::var_os("PIE_GOLDEN_TIER_INLINE").is_some();
    // No weight shape stated: the tables and routed banks dense on the
    // device, as the base engine seats them. The weight cache's own probes
    // load with its shapes ([`load_cached`]).
    knobs.dense_weights = true;
    knobs
}

/// The buffered probes' knobs: a load that buffers recurrent rows arms
/// every class's compositions for lanes that buffer or replay them, so these
/// fires replay bodies. `PIE_GOLDEN_RS_EAGER` runs them with no body at all
/// (as `PIE_GOLDEN_EAGER`): the eager walk, the reference a body's logits
/// match bit for bit.
fn buffering(mut knobs: engine_cuda::Knobs) -> engine_cuda::Knobs {
    if std::env::var_os("PIE_GOLDEN_RS_EAGER").is_some() {
        knobs.pin = Some((engine_cuda::Graphs::Off, Some(engine_cuda::Recording::Off)));
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
    if std::env::var_os("PIE_GOLDEN_FACTS").is_some() {
        let facts = shell.chunk_facts().expect("the pool states its chunks");
        eprintln!(
            "golden-facts {} chunk {} shapes {:?}",
            fixture.model,
            facts.chunk_bytes,
            facts
                .shapes
                .iter()
                .map(|s| (s.id.0, s.role, s.slot_bytes, s.slots_per_chunk, s.kind.clone()))
                .collect::<Vec<_>>()
        );
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
        self.facts.shapes.iter().position(|shape| {
            !matches!(
                shape.role,
                engine::chunk::SlotRole::Instance | engine::chunk::SlotRole::Channel
            ) && kind(&shape.kind)
        })
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
        let window = windows.first().map_or(&[][..], Vec::as_slice);
        fn seat<'a>(
            (pages, window, more, record): (&'a [u32], &'a [u32], &'a [&'a [u32]], Option<u32>),
            word: u64,
            tokens: &'a [u32],
            held: u32,
        ) -> Seated<'a> {
            Seated {
                pages,
                held: Some(held),
                window,
                more_windows: more,
                rs_slot: record,
                ..Seated::of(Lane {
                    slot: 0,
                    word,
                    tokens,
                })
            }
        }
        let tables = (&pages[..], window, &more[..], record);
        let seated = |word, tokens, held| seat(tables, word, tokens, held);
        // A long prompt of a load that replays (`engine_cuda::replay`) fires
        // as the runtime's step and `Shell::fire` split it: its leading rows
        // encoder-only, then its last window.
        let replay = engine_cuda::replay::Replay::of(&shell)
            .filter(|replay| tokens.len() > replay.window as usize);
        let tail = match replay {
            Some(replay) => {
                let skipped = tokens.len() - replay.window as usize;
                shell
                    .fire_media(
                        &[seated(word | replay.bit, &tokens[..skipped], fed)],
                        &[],
                        &[],
                    )
                    .expect("the encoder-only rows fire");
                (skipped, fed + u32::try_from(skipped).expect("a count"))
            }
            None => (0, fed),
        };
        let mut rows = shell
            .fire_media(&[seated(word, &tokens[tail.0..], tail.1)], &[], &[])
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

/// Tier refs handed out as the runtime's host and disk tiers would
/// (CONTRACT-BC 22): a fresh chunk of the tier's `chunks` for a shape whose
/// open one is full, chunk ids their own places.
struct TierRefs {
    next: u32,
    chunks: u32,
    open: Vec<Option<(u32, u32)>>,
}

impl TierRefs {
    fn of(facts: &engine::chunk::ChunkFacts, chunks: u32) -> TierRefs {
        TierRefs {
            next: 0,
            chunks,
            open: vec![None; facts.shapes.len()],
        }
    }

    fn take(&mut self, facts: &engine::chunk::ChunkFacts, shape: usize) -> u32 {
        let per_chunk = facts.shapes[shape].slots_per_chunk;
        let (chunk, index) = match self.open[shape] {
            Some((chunk, index)) if index < per_chunk => (chunk, index),
            _ => {
                assert!(self.next < self.chunks, "the tier holds the lane");
                self.next += 1;
                (self.next - 1, 0)
            }
        };
        self.open[shape] = Some((chunk, index + 1));
        chunk * per_chunk + index
    }
}

/// Every slot of `refs` (shape `shape`; ref 0, a trimmed window slot,
/// stays) out to `tier` (its refs handed out by `parked`) and back into
/// fresh device slots, as a suspend and a restore move them: the refs the
/// lane names from then on.
fn park(
    shell: &mut Shell,
    refs: &mut Refs,
    parked: &mut TierRefs,
    tier: engine_cuda::store::Tier,
    shape: usize,
    slots: &[u32],
) -> Vec<u32> {
    use engine_cuda::store::Tier;
    let id = refs.facts.shapes[shape].id;
    let facts = refs.facts.clone();
    let out: Vec<(u32, u32)> = slots
        .iter()
        .filter(|&&slot| slot != 0)
        .map(|&slot| (slot, parked.take(&facts, shape)))
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
/// chunks: the host tier the first time, the disk tier the second. The logits are the never-offloaded serving's, bit for
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
    let host = Some(stated.map_or(1 << 30, |mib| mib << 20));
    let ring = engine_cuda::store::staging_ring_for(
        host,
        1,
        engine::runahead::Runahead::DEFAULT_FRAMES_IN_FLIGHT,
    )
    .and_then(|bytes| bytes.map(engine_cuda::store::Ring::pin).transpose())
    .expect("the staging ring pins");
    shell
        .seat_host_kv(host, 1, CONTEXT, 1, ring)
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
    let mut host = TierRefs::of(&facts, facts.host_chunks);
    let mut disk = TierRefs::of(&facts, facts.disk_chunks);
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
            let (tier, parked) = if at == 2 {
                (Tier::Host, &mut host)
            } else {
                (Tier::Disk, &mut disk)
            };
            if let Some(shape) = list {
                pages = park(&mut shell, &mut refs, parked, tier, shape, &pages);
            }
            if let Some(shape) = window_list {
                window = park(&mut shell, &mut refs, parked, tier, shape, &window);
            }
            if let Some(shape) = fixed {
                pieces = park(&mut shell, &mut refs, parked, tier, shape, &pieces);
                record = Some(shell.name_record(&pieces).expect("the record is named"));
            }
        }
    }
    steps
}

/// CONTRACT-BC 22: a lane offloaded mid-decode (every slot it holds to the
/// host tier and back, then to the disk tier and back) serves the
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

/// A slot's words, every one's low half set: no bf16 (nor a narrowing of
/// f32 to one) keeps them. Some are NaNs with payloads, some denormals.
fn tier_pattern(seed: u32, bytes: u64) -> Vec<u8> {
    (0..bytes / 4)
        .flat_map(|at| {
            let x = (at as u32 ^ seed.wrapping_mul(0x9e37_79b9)).wrapping_mul(0x85eb_ca6b);
            let word = match at % 7 {
                0 => 0x7fc0_0000 | (x & 0x003f_ffff) | 1,
                1 => x & 0x007f_ffff | 1,
                _ => x ^ (x >> 15) | 0x0001_0001,
            };
            word.to_le_bytes()
        })
        .collect()
}

fn tier_check(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: byte {at} of {} differs ({:#04x} for {:#04x}); {} bytes differ",
            want.len(),
            got[at],
            want[at],
            got.iter().zip(want).filter(|(a, b)| a != b).count()
        );
    }
}

/// Every parked shape of `model`'s load through the tiers below the device
/// ([`a_tier_round_trip_is_bit_exact_for_every_parked_shape`]): each shape
/// it moved, named by what it holds, and its slot's bytes.
fn tier_round_trip(model: &str, cache: &Path) -> Vec<(String, u64)> {
    use engine_cuda::store::Tier;
    let fixture = fixture(model, cache);
    let mut shell = load(&fixture, cache);
    let host = Some(256 << 20);
    let ring = engine_cuda::store::staging_ring_for(
        host,
        1,
        engine::runahead::Runahead::DEFAULT_FRAMES_IN_FLIGHT,
    )
    .and_then(|bytes| bytes.map(engine_cuda::store::Ring::pin).transpose())
    .expect("the staging ring pins");
    shell
        .seat_host_kv(host, 1, CONTEXT, 1, ring)
        .expect("the host tier seats");
    // Three disk chunks a parked shape, and a part chunk the tier leaves.
    let stated = shell.chunk_facts().expect("the pool states its chunks");
    let disk_chunk = stated.chunk_bytes.next_multiple_of(4096);
    let budget = 3 * stated.shapes.len() as u64 * disk_chunk + disk_chunk / 2;
    let dir = tempfile::tempdir().expect("a disk tier's dir");
    let chunks = shell
        .seat_disk_kv(
            Some(budget),
            Some(dir.path()),
            false,
            engine_cuda::World::default(),
        )
        .expect("the disk tier seats");
    shell.take_refs();
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    facts.validate().expect("the engine's chunk facts hold");
    // The budget divides into disk chunks of a pool chunk each, on the
    // direct I/O boundary, laid out as the device's.
    assert_eq!(
        facts.disk_chunks, chunks,
        "{model}: the facts state the disk chunks"
    );
    assert_eq!(
        u64::from(chunks),
        budget / disk_chunk,
        "{model}: disk chunks of the budget"
    );
    let parked: Vec<engine::chunk::SlotShape> = facts
        .shapes
        .iter()
        .filter(|shape| shape.role != engine::chunk::SlotRole::Adapter)
        .cloned()
        .collect();
    // The null chunk and four device chunks a shape.
    let segments = (1 + 4 * parked.len() as u32)
        .div_ceil(facts.segment_chunks.max(1))
        .min(facts.segments.max);
    shell
        .commit_segments(segments)
        .expect("the segments commit");
    let committed = segments * facts.segment_chunks;
    assert!(
        1 + 4 * parked.len() as u32 <= committed,
        "{model}: four device chunks a shape in the {committed} committed"
    );
    assert!(
        3 * parked.len() as u32 <= chunks && 2 * parked.len() as u32 <= facts.host_chunks,
        "{model}: three disk chunks and two host chunks a shape"
    );
    let stream = shell.stream();
    let write = |shell: &mut Shell, id, slot: u32, bytes: &[u8]| {
        let at = shell.slot_span(id, slot).expect("a slot's place");
        engine_cuda::device::alloc::stage_raw(stream, at, bytes).expect("the slot is written");
    };
    let read = |shell: &mut Shell, id, slot: u32, len: u64| -> Vec<u8> {
        shell.settle_moves().expect("every move lands");
        let at = shell.slot_span(id, slot).expect("a slot's place");
        let mut out = vec![0u8; len as usize];
        engine_cuda::device::alloc::copy_d2h_on(stream, at, &mut out).expect("the slot reads");
        out
    };
    let zip = |a: &[u32], b: &[u32]| -> Vec<(u32, u32)> {
        a.iter().copied().zip(b.iter().copied()).collect()
    };
    let mut moved = Vec::new();
    for (k, shape) in parked.iter().enumerate() {
        let k = k as u32;
        let (id, per, bytes) = (shape.id, shape.slots_per_chunk, shape.slot_bytes);
        assert!(
            u64::from(per) * bytes <= facts.chunk_bytes,
            "{model}: shape {id} packs a chunk as on the device"
        );
        // Device chunks `1 + 4k ..`, disk chunks `3k ..`, host chunks `2k ..`:
        // each shape's own. Slots `n` adjacent ones of a chunk, from `first`.
        let n = per.min(3);
        let refs = |chunk: u32, first: u32| -> Vec<u32> {
            (0..n).map(|i| chunk * per + (first + i) % per).collect()
        };
        let what = |how: &str| format!("{model} shape {id} ({bytes} bytes a slot) {how}");

        // Device -> disk -> device: adjacent disk slots, out of order, in
        // one copy; the device slots overwritten; restored into other
        // slots in two copies back to back.
        let (source, into) = (refs(1 + 4 * k, 0), refs(2 + 4 * k, 0));
        let held: Vec<Vec<u8>> = source
            .iter()
            .enumerate()
            .map(|(i, &slot)| {
                let pattern = tier_pattern(100 * (k + 1) + i as u32, bytes);
                write(&mut shell, id, slot, &pattern);
                pattern
            })
            .collect();
        let disk: Vec<u32> = refs(3 * k, 1).into_iter().rev().collect();
        shell
            .copy_tier_slots(id, Tier::Device, Tier::Disk, &zip(&source, &disk))
            .expect("the slots park on disk");
        for &slot in &source {
            write(&mut shell, id, slot, &vec![0xa5; bytes as usize]);
        }
        let (disk_a, disk_b) = disk.split_at(1);
        let (into_a, into_b) = into.split_at(1);
        shell
            .copy_tier_slots(id, Tier::Disk, Tier::Device, &zip(disk_a, into_a))
            .expect("the first slot restores");
        shell
            .copy_tier_slots(id, Tier::Disk, Tier::Device, &zip(disk_b, into_b))
            .expect("the rest restore");
        for (i, &slot) in into.iter().enumerate() {
            tier_check(
                &what(&format!("slot {i} device -> disk -> device")),
                &read(&mut shell, id, slot, bytes),
                &held[i],
            );
        }
        // The slot beside the parked ones in their disk chunk, written
        // after them in a copy of its own (sharing a block with them where
        // a slot is off the boundary), leaves them as they were.
        if per > n + 1 {
            let beside = 3 * k * per + n + 1;
            let other = refs(3 + 4 * k, 0)[0];
            write(&mut shell, id, other, &tier_pattern(7, bytes));
            shell
                .copy_tier_slots(id, Tier::Device, Tier::Disk, &[(other, beside)])
                .expect("the slot beside parks");
            shell
                .copy_tier_slots(id, Tier::Disk, Tier::Device, &zip(&disk, &into))
                .expect("the slots restore again");
            for (i, &slot) in into.iter().enumerate() {
                tier_check(
                    &what(&format!("slot {i} beside a later neighbour")),
                    &read(&mut shell, id, slot, bytes),
                    &held[i],
                );
            }
        }
        // Negative control: a restore from a disk slot this shape never
        // parked to does not read as the parked bytes.
        let never = refs(3 * k + 2, 0)[0];
        let blank = refs(3 + 4 * k, 1)[0];
        write(&mut shell, id, blank, &vec![0x3c; bytes as usize]);
        shell
            .copy_tier_slots(id, Tier::Disk, Tier::Device, &[(never, blank)])
            .expect("an unparked disk slot restores");
        let got = read(&mut shell, id, blank, bytes);
        assert!(
            held.iter().all(|held| *held != got),
            "{}",
            what("restored from an unparked disk slot reads as a parked one")
        );

        // Host -> disk -> host, into other host slots; and the disk slots
        // up to the device too.
        let host_in = refs(2 * k, 0);
        let host_back = refs(2 * k + 1, 0);
        let seeded: Vec<u8> = (0..n)
            .flat_map(|i| tier_pattern(100 * (k + 1) + 50 + i, bytes))
            .collect();
        shell
            .write_tier_slots(&[(id, host_in.clone())], &seeded)
            .expect("the host slots take the pattern");
        let down: Vec<u32> = refs(3 * k + 1, 1).into_iter().rev().collect();
        shell
            .copy_tier_slots(id, Tier::Host, Tier::Disk, &zip(&host_in, &down))
            .expect("the host slots go down to disk");
        shell
            .write_tier_slots(&[(id, host_in.clone())], &vec![0x5a; seeded.len()])
            .expect("the host slots are written over");
        shell
            .copy_tier_slots(id, Tier::Disk, Tier::Host, &zip(&down, &host_back))
            .expect("the disk slots come up to the host");
        let got = shell
            .read_tier_slots(&[(id, host_back)])
            .expect("the host slots read");
        tier_check(&what("host -> disk -> host"), &got, &seeded);
        let up = refs(4 + 4 * k, 0);
        shell
            .copy_tier_slots(id, Tier::Disk, Tier::Device, &zip(&down, &up))
            .expect("the disk slots come up to the device");
        let row = bytes as usize;
        for (i, &slot) in up.iter().enumerate() {
            tier_check(
                &what(&format!("slot {i} host -> disk -> device")),
                &read(&mut shell, id, slot, bytes),
                &seeded[i * row..(i + 1) * row],
            );
        }
        println!("{}: every round trip bit-exact", what("parks"));
        let held = match (shape.role, &shape.kind) {
            (
                _,
                engine::chunk::SlotKind::List {
                    window: Some(_), ..
                },
            ) => "window".to_string(),
            (engine::chunk::SlotRole::State, engine::chunk::SlotKind::List { .. }) => {
                "kv".to_string()
            }
            (engine::chunk::SlotRole::State, engine::chunk::SlotKind::Fixed { .. }) => {
                "record".to_string()
            }
            (role, _) => format!("{role:?}"),
        };
        moved.push((held, bytes));
    }
    moved
}

/// Every shape a parked set carries comes back bit for bit through the
/// tiers below the device (CONTRACT-BC 22): the kv list, the window lists
/// (Gemma's miniature), the hybrid's rs-list rows and records (Qwen3.5's),
/// and the guests' instance and channel slots, device -> disk -> device
/// and host -> disk -> host (and on up to the device), into other slots,
/// the disk budget divided into disk chunks laid out as the device's.
/// Words no bf16 holds; disk slots adjacent and out of order, restored in
/// two copies back to back; a slot off the 4 KiB boundary written beside
/// its parked neighbours without disturbing them; and a negative control,
/// a restore from a disk slot never parked to, that reads as none of them.
#[test]
fn a_tier_round_trip_is_bit_exact_for_every_parked_shape() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let mut roles = std::collections::BTreeSet::new();
    let mut unaligned = false;
    for model in ["qwen35-tiny", "gemma4-e4b-mini-l6"] {
        for (held, bytes) in tier_round_trip(model, &cache) {
            roles.insert(held);
            unaligned |= bytes % 4096 != 0;
        }
    }
    println!("parked shapes: {roles:?}");
    for want in ["kv", "window", "record", "RsBuffer", "Instance", "Channel"] {
        assert!(roles.contains(want), "a {want} shape parked: {roles:?}");
    }
    assert!(unaligned, "a slot off the 4 KiB boundary parked");
}

fn probe(served: fn(&Fixture, &Path) -> Vec<(u64, u32)>) {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    // The measuring pass: every model served once, so each product shape any
    // of them meets is measured and kept in the cache's profile before a
    // logit is hashed. A model hashed while a later model still measured
    // shapes it shares would hash picks the next run no longer makes.
    for model in models() {
        let fixture = fixture(&model, &cache);
        let _ = served(&fixture, &cache);
    }
    // The profile the hashes are taken under, held fixed from here: its
    // md5 names the picks, so a change of them shows as a new profile, not
    // as a moved logit.
    let profile = profile_md5(&cache);
    let mut lines = vec![format!("golden profile md5 {profile}")];
    println!("{}", lines[0]);
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
        let wanted_profile = text
            .lines()
            .find_map(|line| line.strip_prefix("golden profile md5 "))
            .unwrap_or("not stated");
        if wanted_profile != profile {
            eprintln!(
                "golden: the profile the hashes were taken under is {profile}, the file's \
                 {wanted_profile}: the picks may differ, and with them the logits"
            );
        }
        for line in lines.iter().filter(|line| !line.starts_with("golden profile ")) {
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

/// The md5 of every stored input a load selects its kernels and bodies by:
/// the profile its measurements are kept in (`knobs`: the cache's
/// `profiles`), the bodies records and the host covers, every file in path
/// order, by `md5sum`; "none" where there is no file. A cold cache reads as
/// another configuration.
fn profile_md5(cache: &Path) -> String {
    let mut files = Vec::new();
    let mut dirs = vec![cache.join("profiles"), cache.join("bodies"), cache.join("cover")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return "none".to_string();
    }
    let mut bytes = Vec::new();
    for file in &files {
        bytes.extend(std::fs::read(file).unwrap_or_default());
    }
    let mut child = std::process::Command::new("md5sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("md5sum runs");
    use std::io::Write;
    child
        .stdin
        .take()
        .expect("md5sum's stdin")
        .write_all(&bytes)
        .expect("the profile reaches md5sum");
    let out = child.wait_with_output().expect("md5sum answers");
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("none")
        .to_string()
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

/// The model the buffered probes serve: a chunked recurrence whose rows a
/// lane may buffer unfolded, gated delta net state by default, another
/// (`nemotron3-nano-mini`'s Mamba-2) by `PIE_GOLDEN_BUFFERED`.
fn buffered_model() -> String {
    std::env::var("PIE_GOLDEN_BUFFERED").unwrap_or_else(|_| "qwen35-tiny".to_string())
}

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
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
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
                lineage: engine::fire::Lineage::Set,
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
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let _ = served_buffered(&fixture, &cache);
    let steps = served_buffered(&fixture, &cache);
    let mut lines = Vec::new();
    let mut all = FNV;
    for (at, (hash, top)) in steps.iter().enumerate() {
        all = fnv_bytes(all, &hash.to_le_bytes());
        lines.push(format!("golden {model}-buffered {at} {hash:016x} {top}"));
    }
    lines.push(format!("golden {model}-buffered all {all:016x}"));
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
            .filter(|line| line.contains(&format!(" {model}-buffered ")))
            .collect();
        assert_eq!(
            held,
            lines.iter().map(String::as_str).collect::<Vec<_>>(),
            "the buffered logits moved"
        );
    }
}

/// The rows the carrying probe's lane leaves unfolded before one decode
/// replays them all: more than a page of the rs buffer, and than the
/// decode's own bucket.
const CARRIED: u32 = 40;

/// A prefill folded as it lands, then `CARRIED + 1` decodes of a fixed
/// script: with `buffer` each buffers its row past the state and folds
/// nothing, until the last replays the `CARRIED` rows before it and folds
/// them all; without, each folds as it lands. Each step's logits, and the
/// bodies the last decode replayed and the fires it walked with none.
fn served_carrying(fixture: &Fixture, cache: &Path, buffer: bool) -> (Vec<Vec<f32>>, u64, u64) {
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
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    shell.back_whole().expect("the fitted pool is backed whole");
    let page = rs_page(&mut shell);
    shell.open(0).expect("slot 0 opens");
    let mut steps = Vec::new();
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
        steps.push(rows.swap_remove(0));
    };
    let prompt: Vec<u32> = (0..37).map(|t| (t * 7 + 11) % 1000).collect();
    let folded = prompt.len() as u32;
    step(&mut shell, &prompt, RsVerb::Fold);
    let base = folded / page * page;
    let mut before = shell.body_stats().tally;
    for at in 0..=CARRIED {
        let end = folded + at + 1;
        before = shell.body_stats().tally;
        let rs = if buffer {
            RsVerb::Buffer {
                pages: (base / page..end.div_ceil(page)).collect(),
                base,
                folded,
                fold: FoldLen::Host(if at == CARRIED { CARRIED + 1 } else { 0 }),
                lineage: engine::fire::Lineage::Set,
            }
        } else {
            RsVerb::Fold
        };
        step(&mut shell, &[((folded + at) * 7 + 11) % 1000], rs);
    }
    let after = shell.body_stats().tally;
    (
        steps,
        after.hits - before.hits,
        after.eager_unbodied - before.eager_unbodied,
    )
}

/// One lane replaying `CARRIED` buffered rows ahead of its one row is
/// served by a body, and every step answers what the same rows folded as
/// they land answer, to the tolerance of the two kernels that run them.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_buffered_carrying() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let (folded, _, _) = served_carrying(&fixture, &cache, false);
    let (buffered, hits, unbodied) = served_carrying(&fixture, &cache, true);
    assert_eq!(buffered.len(), folded.len());
    let mut worst = 0f32;
    for (buffered, folded) in buffered.iter().zip(&folded) {
        let scale = folded.iter().fold(0f32, |most, v| most.max(v.abs()));
        let apart = buffered
            .iter()
            .zip(folded)
            .fold(0f32, |most, (a, b)| most.max((a - b).abs()));
        worst = worst.max(apart / scale.max(f32::MIN_POSITIVE));
    }
    println!(
        "golden {model}-buffered-carrying: {} steps, logits apart by {worst:e} of their scale, \
         the last decode replayed {hits} body(ies) and walked {unbodied}",
        buffered.len()
    );
    // A replay's extended tensors state the whole run's rows
    // (`store::rs::Buffers::rows_ext`), past `ssm_gdn_scan_until`, so its
    // gated-delta recurrence is the chunked kernel; a fold of one row takes
    // the scan. The two sum in other orders: on this miniature the logits
    // lie 0.013 to 0.035 of their scale apart at every carried count from 1
    // to 40 (0 where nothing is replayed). Past 0.05 is another cause.
    assert!(
        worst < 0.05,
        "the buffered lane's logits lie {worst:e} of their scale from the folded lane's"
    );
    assert!(
        hits > 0 && unbodied == 0,
        "the decode replaying {CARRIED} buffered rows replayed {hits} body(ies) and walked \
         {unbodied} fire(s) with none"
    );
}

/// Decode steps a lagged lane takes: a fold behind its rows by the lag.
const LAGGED_STEPS: u32 = 30;

/// A prefill folded as it lands, then [`LAGGED_STEPS`] decodes of a fixed
/// script, each buffering its row and folding so that `lag` rows stay
/// unfolded behind it (a run-ahead decode whose rows are folded once they
/// are accepted): from the lag on, each decode carries `lag` rows into its
/// step and commits the oldest of them, every `max(7, lag + 3)`-th step
/// committing them all (so a lag of 20 is reached) and every fifth one
/// none. `lag` 0 folds each row as it lands. Each step's logits, and the
/// fires of the decodes walked with no body.
fn served_lagged(fixture: &Fixture, cache: &Path, lag: u32) -> (Vec<Vec<f32>>, u64) {
    served_lagged_over(fixture, cache, lag, TOKENS, LAGGED_STEPS)
}

/// [`served_lagged`] on a load whose fires take `tokens` rows, over
/// `decodes` decodes.
fn served_lagged_over(
    fixture: &Fixture,
    cache: &Path,
    lag: u32,
    tokens: u32,
    decodes: u32,
) -> (Vec<Vec<f32>>, u64) {
    use engine::fire::{FoldLen, RsVerb};
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, tokens),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: 4,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    shell.back_whole().expect("the fitted pool is backed whole");
    let page = rs_page(&mut shell);
    shell.open(0).expect("slot 0 opens");
    let mut steps = Vec::new();
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
        steps.push(rows.swap_remove(0));
    };
    let prompt: Vec<u32> = (0..37).map(|t| (t * 7 + 11) % 1000).collect();
    let mut fed = prompt.len() as u32;
    // A lagged lane's prompt leaves its last rows unfolded too, so its
    // chunked fire lands what a decode reads them by.
    let mut folded = fed - lag.min(4);
    let rs = if lag == 0 {
        RsVerb::Fold
    } else {
        RsVerb::Buffer {
            pages: (0..fed.div_ceil(page)).collect(),
            base: 0,
            folded: 0,
            fold: FoldLen::Host(folded),
            lineage: engine::fire::Lineage::Set,
        }
    };
    step(&mut shell, &prompt, rs);
    let before = shell.body_stats().tally;
    for at in 0..decodes {
        let token = (fed * 7 + 11) % 1000;
        if lag == 0 {
            step(&mut shell, &[token], RsVerb::Fold);
            fed += 1;
            folded += 1;
            continue;
        }
        let end = fed + 1;
        let base = folded / page * page;
        let whole = 7.max(lag + 3);
        let fold = match at {
            _ if at % whole == whole - 1 => end - folded,
            _ if at % 5 == 4 => 0,
            _ => (end - folded).saturating_sub(lag),
        };
        step(
            &mut shell,
            &[token],
            RsVerb::Buffer {
                pages: (base / page..end.div_ceil(page)).collect(),
                base,
                folded,
                fold: FoldLen::Host(fold),
                lineage: engine::fire::Lineage::Set,
            },
        );
        fed = end;
        folded += fold;
    }
    let after = shell.body_stats().tally;
    (steps, after.eager_unbodied - before.eager_unbodied)
}

/// A lane folding `lag` rows behind its decodes answers, step for step,
/// what the lane folding each row as it lands answers, to the tolerance of
/// the two forms of the recurrence: its decode step reads the buffered rows
/// as a linear attention over their stored keys and corrected values from
/// the committed state (rounded to bf16 only as a fold stores it), where
/// the folding lane re-steps and stores the state a row. Its greedy tokens
/// are the folding lane's but where the two lie within that tolerance.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_lagged() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let (folded, _) = served_lagged(&fixture, &cache, 0);
    let lags: Vec<u32> = std::env::var("PIE_GOLDEN_LAGS").map_or_else(
        |_| vec![1, 2, 3, 9],
        |lags| lags.split(',').map(|lag| lag.parse().expect("a lag")).collect(),
    );
    for lag in lags {
        let (lagged, unbodied) = served_lagged(&fixture, &cache, lag);
        assert_eq!(lagged.len(), folded.len());
        let mut worst = 0f32;
        let mut moved = 0usize;
        let mut flips = 0usize;
        for (lagged, folded) in lagged.iter().zip(&folded) {
            let scale = folded.iter().fold(0f32, |most, v| most.max(v.abs()));
            let apart = lagged
                .iter()
                .zip(folded)
                .fold(0f32, |most, (a, b)| most.max((a - b).abs()));
            worst = worst.max(apart / scale.max(f32::MIN_POSITIVE));
            moved += usize::from(lagged != folded);
            let (a, b) = (argmax(lagged), argmax(folded));
            if a != b {
                let mut sorted = folded.clone();
                sorted.sort_by(|x, y| y.total_cmp(x));
                let gap = (sorted[0] - sorted[1]) / scale.max(f32::MIN_POSITIVE);
                if gap > 0.02 {
                    flips += 1;
                }
            }
            if std::env::var_os("PIE_GOLDEN_LAG_STEPS").is_some() {
                println!("lag {lag} step: apart {:e}", apart / scale.max(f32::MIN_POSITIVE));
            }
        }
        println!(
            "golden {model}-lagged lag {lag}: {} steps, {moved} differ, logits apart by \
             {worst:e} of their scale, {flips} greedy token(s) flipped off a near tie, \
             {unbodied} decode(s) walked with no body",
            lagged.len()
        );
        // The two forms round apart: the folding lane's step rounds the
        // decayed state to bf16 before its update and stores it a row, the
        // lagged lane's sums in f32 and rounds as a fold stores. On this
        // miniature they lie 0.012 to 0.023 of their scale apart, the own
        // row's step alone as much as a lag of eleven (the chunked replay's
        // band beside the scan's is 0.013 to 0.035, `golden_buffered_carrying`).
        assert!(
            worst < 0.05 && flips == 0,
            "at a lag of {lag} the lagged lane's logits lie {worst:e} of their scale from the \
             folded lane's, {flips} greedy token(s) flipped"
        );
        assert_eq!(unbodied, 0, "a lagged decode walked with no body");
    }
}

/// A lane alone on its set carrying more rows unfolded than nine of its
/// fires' widths (the most a per-fire page table held) answers what the
/// folding lane answers, to the tolerance of [`golden_lagged`]: its rows
/// resolve through its own pages on the device, whatever it carries.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_lagged_past_the_fire() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    const FIRE: u32 = 64;
    const STEPS: u32 = 9 * FIRE + 40;
    let cache = cache_dir();
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let (folded, _) = served_lagged_over(&fixture, &cache, 0, FIRE, STEPS);
    let (lagged, unbodied) = served_lagged_over(&fixture, &cache, STEPS + 8, FIRE, STEPS);
    assert_eq!(lagged.len(), folded.len());
    let mut worst = 0f32;
    let mut flips = 0usize;
    for (lagged, folded) in lagged.iter().zip(&folded) {
        let scale = folded.iter().fold(0f32, |most, v| most.max(v.abs()));
        let apart = lagged
            .iter()
            .zip(folded)
            .fold(0f32, |most, (a, b)| most.max((a - b).abs()));
        worst = worst.max(apart / scale.max(f32::MIN_POSITIVE));
        if argmax(lagged) != argmax(folded) {
            let mut sorted = folded.clone();
            sorted.sort_by(|x, y| y.total_cmp(x));
            if (sorted[0] - sorted[1]) / scale.max(f32::MIN_POSITIVE) > 0.02 {
                flips += 1;
            }
        }
    }
    println!(
        "golden {model}-lagged past the fire: {} steps carrying up to {} rows on {FIRE}-row \
         fires, logits apart by {worst:e} of their scale, {flips} flipped, {unbodied} unbodied",
        lagged.len(),
        STEPS + 4
    );
    assert!(
        worst < 0.05 && flips == 0,
        "carrying past the fire the lane's logits lie {worst:e} of their scale from the folded \
         lane's, {flips} greedy token(s) flipped"
    );
}

/// Decode steps the pace probe times a schedule.
const PACE_STEPS: u32 = 240;

/// The median wall time of a one-lane decode step on `fixture` under each
/// schedule of `PIE_GOLDEN_PACE` (a comma list; by default `0,1,11,w10`):
/// a number keeps that many rows unfolded behind each decode, `w<n>` folds
/// the accepted rows whole once they fill a window of `n` (the runtime's
/// run-ahead lag), 0 folds each row as it lands. Each schedule is served by
/// its own load after a warm pass.
fn served_pace(fixture: &Fixture, cache: &Path, schedule: &str) -> f64 {
    use engine::fire::{FoldLen, RsVerb};
    let (lag, window) = match schedule.strip_prefix('w') {
        Some(w) => (0, w.parse::<u32>().expect("a window")),
        None => (schedule.parse::<u32>().expect("a lag"), 0),
    };
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
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    shell.back_whole().expect("the fitted pool is backed whole");
    let page = rs_page(&mut shell);
    let mut times = Vec::new();
    for pass in 0..2 {
        shell.open(pass).expect("the pass's slot opens");
        let prompt: Vec<u32> = (0..37).map(|t| (t * 7 + 11) % 1000).collect();
        let fire = |shell: &mut Shell, tokens: &[u32], rs: RsVerb| {
            let word = fixture.trace.facts.word(&Request::new(
                u32::try_from(tokens.len()).expect("a count"),
                false,
            ));
            let seated = engine_cuda::Seated {
                rs,
                ..engine_cuda::Seated::of(Lane {
                    slot: pass,
                    word,
                    tokens,
                })
            };
            let at = std::time::Instant::now();
            let rows = shell.fire_media(&[seated], &[], &[]).expect("the step fires");
            (argmax(&rows[0]), at.elapsed().as_secs_f64())
        };
        let (mut last, _) = fire(&mut shell, &prompt, RsVerb::Fold);
        let mut fed = prompt.len() as u32;
        let mut folded = fed;
        for _ in 0..PACE_STEPS {
            let end = fed + 1;
            let (top, secs) = if lag == 0 && window == 0 {
                fire(&mut shell, &[last], RsVerb::Fold)
            } else {
                let unfolded = end - folded;
                let fold = if window > 0 {
                    if unfolded > window { unfolded - 1 } else { 0 }
                } else {
                    unfolded.saturating_sub(lag)
                };
                let base = folded / page * page;
                let rs = RsVerb::Buffer {
                    pages: (base / page..end.div_ceil(page)).collect(),
                    base,
                    folded,
                    fold: FoldLen::Host(fold),
                    lineage: engine::fire::Lineage::Set,
                };
                let out = fire(&mut shell, &[last], rs);
                folded += fold;
                out
            };
            last = top;
            fed = end;
            if pass == 1 {
                times.push(secs);
            }
        }
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// The pace probe ([`served_pace`]): `pace <model> <schedule> <ms>` lines.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_lagged_pace() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let schedules = std::env::var("PIE_GOLDEN_PACE").unwrap_or_else(|_| "0,1,11,w10".to_string());
    for schedule in schedules.split(',') {
        let secs = served_pace(&fixture, &cache, schedule);
        println!("pace {model} {schedule} {:.4} ms", secs * 1e3);
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
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: buffering(engine_cuda::Knobs::default()),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
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
                        lineage: engine::fire::Lineage::Set,
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
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let _ = served_pair_buffered(&fixture, &cache);
    let steps = served_pair_buffered(&fixture, &cache);
    let lines: Vec<String> = steps
        .iter()
        .enumerate()
        .map(|(at, (hash, top))| format!("golden {model}-pair-buffered {at} {hash:016x} {top}"))
        .collect();
    for line in &lines {
        println!("{line}");
    }
    if let Some(out) = std::env::var_os("PIE_GOLDEN_OUT") {
        std::fs::write(&out, lines.join("\n") + "\n")
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&out).display()));
    }
}

/// Every lane of a fire replaying its own buffered rows answers as it does
/// replaying alone, the fire's last lanes too: each lane's whole staged
/// `RsLane` (where its replay lies and its place in the extended run) is
/// the one the fire stages, never an earlier fire's left in the table.
#[test]
fn every_lane_of_a_wide_replaying_fire_replays_its_own_rows() {
    use engine::fire::{FoldLen, RsVerb};
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    const FLOOR: f32 = 0.05;
    const LANES: u32 = 4;
    let cache = cache_dir();
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        // As many slots as lanes: a fire's lane table spans the slots, so
        // the last lanes' tails lie where a short copy stops.
        slots: LANES,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        // Walked eagerly, as a fire no body carries is (a masked class's,
        // a beam's): its lane table is staged at its own lanes, not a
        // body's carve.
        knobs: engine_cuda::Knobs {
            pin: Some((engine_cuda::Graphs::Off, Some(engine_cuda::Recording::Off))),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: Some(&cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{model} loads: {why}"));
    let page = rs_page(&mut shell);
    // Lane `lane`'s buffered rows: a length and tokens of its own, so each
    // lane replays apart from every other and lies at its own place.
    let history = |lane: u32| -> Vec<u32> {
        (0..3 + 2 * lane)
            .map(|at| (17 + lane * 53 + at * 37) % 256)
            .collect()
    };
    let buffer = |slot: u32, rows: u32| RsVerb::Buffer {
        pages: (0..rows.div_ceil(page)).map(|at| slot * 16 + at).collect(),
        base: 0,
        folded: 0,
        fold: FoldLen::Host(0),
        lineage: engine::fire::Lineage::Set,
    };
    let fire = |shell: &mut Shell, lanes: &[(u32, Vec<u32>, RsVerb)]| -> Vec<Vec<f32>> {
        let seated: Vec<engine_cuda::Seated<'_>> = lanes
            .iter()
            .map(|(slot, tokens, rs)| engine_cuda::Seated {
                rs: rs.clone(),
                ..engine_cuda::Seated::of(Lane {
                    slot: *slot,
                    word: fixture.trace.facts.word(&Request::new(
                        u32::try_from(tokens.len()).expect("a count"),
                        false,
                    )),
                    tokens,
                })
            })
            .collect();
        shell.fire_media(&seated, &[], &[]).expect("the fire runs")
    };
    // Each lane buffers its history on slot `base + lane`, then replays it
    // under one more row, alone or (its slot opened anew and its history
    // buffered again) in one fire with every other lane, after the lone
    // fires left their lane tables behind.
    let replay_row = |lane: u32| vec![(200 + lane * 11) % 256];
    let stage = |shell: &mut Shell, base: u32| {
        for lane in 0..LANES {
            let slot = base + lane;
            shell.open(slot).expect("the slot opens");
            let rows = history(lane);
            let k = rows.len() as u32;
            fire(shell, &[(slot, rows, buffer(slot, k))]);
        }
    };
    stage(&mut shell, 0);
    let alone: Vec<Vec<f32>> = (0..LANES)
        .map(|lane| {
            let k = history(lane).len() as u32;
            fire(&mut shell, &[(lane, replay_row(lane), buffer(lane, k + 1))]).swap_remove(0)
        })
        .collect();
    stage(&mut shell, 0);
    let together = fire(
        &mut shell,
        &(0..LANES)
            .map(|lane| {
                let slot = lane;
                let k = history(lane).len() as u32;
                (slot, replay_row(lane), buffer(slot, k + 1))
            })
            .collect::<Vec<_>>(),
    );
    let spread = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let apart: Vec<String> = alone
        .iter()
        .zip(&together)
        .enumerate()
        .filter_map(|(lane, (a, b))| {
            let d = spread(a, b);
            eprintln!("{model}: replaying lane {lane} of {LANES} beside the rest vs alone: {d:.4}");
            (d > FLOOR).then(|| format!("lane {lane}: {d}"))
        })
        .collect();
    assert!(
        apart.is_empty(),
        "{model}: lanes of a wide replaying fire answer apart from alone: {apart:?}"
    );
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
        knobs: buffering(engine_cuda::Knobs::default()),
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
            steps: Default::default(),
        })
        .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    let facts = loaded.caps.chunks.clone();
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
    let fixed = shape(|s| matches!(s.kind, SlotKind::Fixed { .. }) && s.role == SlotRole::State);
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
                ..KvDelta::default()
            },
            rs,
            tables: vec![(ids[0], pages.clone())],
            record: engine::fire::LaneRecord::Pieces(vec![(ids[2], record.clone())]),
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
                lineage: engine::fire::Lineage::Set,
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
    let model = buffered_model();
    let fixture = fixture(&model, &cache);
    let _ = submitted_buffered(&fixture, &cache);
    let steps = submitted_buffered(&fixture, &cache);
    let mut lines = Vec::new();
    let mut all = FNV;
    for (at, (hash, top)) in steps.iter().enumerate() {
        all = fnv_bytes(all, &hash.to_le_bytes());
        lines.push(format!("golden {model}-buffered {at} {hash:016x} {top}"));
    }
    lines.push(format!("golden {model}-buffered all {all:016x}"));
    for line in &lines {
        println!("{line}");
    }
    if let Some(want) = std::env::var_os("PIE_GOLDEN_CHECK") {
        let want = std::fs::read_to_string(&want)
            .unwrap_or_else(|why| panic!("{}: {why}", Path::new(&want).display()));
        let held: Vec<&str> = want
            .lines()
            .filter(|line| line.contains(&format!(" {model}-buffered ")))
            .collect();
        assert_eq!(
            held,
            lines.iter().map(String::as_str).collect::<Vec<_>>(),
            "the buffered logits moved"
        );
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

/// The reads since the use record was last asked, as the engine answers
/// them a call behind: the call that snapshots them, the device drained,
/// and the call that answers them.
fn uses_since(shell: &mut Shell) -> engine::chunk::WeightUses {
    let _ = shell.weight_uses().expect("the use record reads");
    shell.drain().expect("the device drains");
    shell.weight_uses().expect("the use record reads")
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
    knobs.dense_weights = false;
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
    let uses = uses_since(&mut shell);
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
    knobs.dense_weights = false;
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
        if told && SMALL_POOL.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            assert!(
                shape.seated.is_none(),
                "a load whose pool cannot hold its weights seats none of them"
            );
        } else if told {
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
    let uses = uses_since(&mut shell);
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
    let packed = fixture.trace.params.iter().any(|param| {
        use poem_ir::Dtype::{
            E4m3, E4m3g128, E4m3tile128, Mxfp4, Nvfp4, U2g32, U2g64, U2g128, U4g32, U4g64, U8g64,
        };
        matches!(
            param.dtype,
            Mxfp4
                | Nvfp4
                | U2g32
                | U2g64
                | U2g128
                | U4g32
                | U4g64
                | U8g64
                | E4m3
                | E4m3tile128
                | E4m3g128
        )
    });
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

/// A load told its weights are held whose pool (`kv_budget`,
/// `PIE_GOLDEN_KV_MIB`, 256 by default) is smaller than them: it seats none
/// in the pool, keeps a copy its fires read in place, and serves what a
/// load that streams every expert from its host copy serves. (A runtime
/// refuses such a deployment; the engine must leave it that choice.)
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_held_small() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["nemotron3-nano-mini".to_string()], |_| models());
    let pool = std::env::var("PIE_GOLDEN_KV_MIB")
        .ok()
        .and_then(|mib| mib.parse::<u64>().ok())
        .unwrap_or(256);
    let hash = |steps: &[Vec<f32>]| {
        steps.iter().fold(FNV, |all, logits| {
            fnv_bytes(all, &hash_logits(logits).to_le_bytes())
        })
    };
    for model in models {
        let fixture = fixture(&model, &cache);
        let _ = served_experts(&fixture, &cache, Rows::Host);
        let host = served_experts(&fixture, &cache, Rows::Host);
        let serving = fixture_as(&model, &cache, true);
        SMALL_POOL.store(pool, std::sync::atomic::Ordering::Relaxed);
        let small = served_experts_as(&serving, &cache, Rows::Host, true);
        SMALL_POOL.store(0, std::sync::atomic::Ordering::Relaxed);
        println!(
            "golden {model} all {:016x} (Host) {:016x} (told held, a {pool} MiB pool)",
            hash(&host),
            hash(&small)
        );
        assert_eq!(
            hash(&small),
            hash(&host),
            "{model} told its weights are held in a pool too small for them"
        );
    }
}

/// A measuring load ends with the pool a recorded one has: `PIE_GOLDEN_MODELS`
/// (MoE miniatures the pool seats whole), each loaded told its weights are
/// held (the load seats them in the pool ahead of the runtime) twice
/// against a cache of its own, first with no record of what its bodies take
/// (it arms them over the least range, measures them, and fits again: on a
/// flat range, one allocation, it reserves the range anew where the card
/// backs more), then from the record that load saved. The first's pool
/// holds every chunk the second's does, less the chunks its first arming's
/// bodies took (the runtime's graph memory may hold them still).
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_measuring_refit() {
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .expect("PIE_GOLDEN_MODELS names the MoE miniatures to load");
    for model in models.split(',').map(str::trim).filter(|m| !m.is_empty()) {
        let fixture = fixture(model, &cache);
        // A cache of its own: no record of the bodies; the profiles shared.
        let own = cache.join(format!("measuring-{}-{model}", std::process::id()));
        let _ = std::fs::remove_dir_all(&own);
        std::fs::create_dir_all(&own).unwrap_or_else(|why| panic!("{}: {why}", own.display()));
        let told = || {
            let mut knobs = knobs();
            knobs.dense_weights = false;
            knobs.weights_held = true;
            knobs
        };
        let (measured, took, chunk) = {
            let mut shell = load_with(&fixture, &own, told());
            let facts = shell.chunk_facts().expect("the pool states its chunks");
            (facts.total_chunks, shell.bodies_took(), facts.chunk_bytes)
        };
        let recorded = {
            let mut shell = load_with(&fixture, &own, told());
            shell
                .chunk_facts()
                .expect("the pool states its chunks")
                .total_chunks
        };
        let _ = std::fs::remove_dir_all(&own);
        let slack = u32::try_from(took.div_ceil(chunk.max(1))).unwrap_or(u32::MAX);
        println!(
            "measuring {model}: {measured} chunks measuring, {recorded} from the record \
             (its bodies took {} MiB, {slack} chunk(s))",
            took >> 20
        );
        assert!(
            measured.saturating_add(slack) >= recorded,
            "{model}: a measuring load's pool holds {measured} chunks and a recorded one's \
             {recorded}, more apart than the {slack} chunk(s) its first arming's bodies took"
        );
    }
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

/// A load of `lanes` seats and fires of up to `lanes` lanes, its pool
/// `kv_budget` bytes where stated.
fn load_lanes(
    fixture: &Fixture,
    cache: &Path,
    knobs: engine_cuda::Knobs,
    lanes: u32,
    kv_budget: Option<u64>,
) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(lanes, TOKENS),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: lanes,
        kv_budget: kv_budget.map(engine_cuda::store::KvSize::Bytes),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs,
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model))
}

/// A stated kv budget is the pool's size, not a reason to arm nothing: a
/// load of many seats whose pool holds one working set and a few chunks
/// more, a recurrent record a chunk, arms every composition it fires.
fn small_pool_arms(fixture: &Fixture, cache: &Path) {
    use engine::chunk::SlotKind;
    const LANES: u32 = 16;
    // A chunk a record: the records the lattice's widest fire names do not
    // fit the pool.
    let record = {
        let mut shell = load_lanes(fixture, cache, knobs(), 1, None);
        let facts = shell.chunk_facts().expect("the pool states its chunks");
        facts
            .shapes
            .iter()
            .find(|shape| {
                matches!(shape.kind, SlotKind::Fixed { .. })
                    && !matches!(
                        shape.role,
                        engine::chunk::SlotRole::Instance | engine::chunk::SlotRole::Channel
                    )
            })
            .expect("a hybrid's recurrent record")
            .slot_bytes
    };
    let mut knobs = knobs();
    knobs.diagnostics.chunk_bytes = Some(record);
    let least = {
        let mut shell = load_lanes(fixture, cache, knobs.clone(), LANES, None);
        let facts = shell.chunk_facts().expect("the pool states its chunks");
        let chunks = |kind: fn(&SlotKind) -> bool, slots: u64| {
            facts
                .shapes
                .iter()
                .find(|shape| {
                    !matches!(
                        shape.role,
                        engine::chunk::SlotRole::Instance | engine::chunk::SlotRole::Channel
                    ) && kind(&shape.kind)
                })
                .map_or(0, |shape| {
                    slots.div_ceil(u64::from(shape.slots_per_chunk.max(1)))
                })
        };
        let pieces = facts
            .shapes
            .iter()
            .filter(|shape| {
                !matches!(
                    shape.role,
                    engine::chunk::SlotRole::Instance | engine::chunk::SlotRole::Channel
                )
            })
            .find_map(|shape| match shape.kind {
                SlotKind::Fixed { pieces } => Some(u64::from(pieces)),
                _ => None,
            })
            .unwrap_or(1);
        let one =
            1 + chunks(
                |kind| matches!(kind, SlotKind::List { window: None, .. }),
                u64::from(CONTEXT / PAGE),
            ) + chunks(|kind| matches!(kind, SlotKind::Fixed { .. }), pieces);
        (one + 4) * facts.chunk_bytes
    };
    let unit = 2u64 << 20;
    let budget = least.div_ceil(unit) * unit;
    let shell = load_lanes(fixture, cache, knobs, LANES, Some(budget));
    let armed = shell.armed().expect("the load armed its lattice");
    println!(
        "small pool {}: {} of {} compositions armed in a {} MiB pool",
        fixture.model,
        armed.armed,
        armed.wanted,
        budget >> 20
    );
    assert!(
        armed.wanted > 0 && armed.armed == armed.wanted,
        "{}: {} of {} compositions armed in a {} MiB pool ({:?})",
        fixture.model,
        armed.armed,
        armed.wanted,
        budget >> 20,
        armed.last_refusal
    );
}

/// How [`served_two_sets`] serves its fires.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Serving {
    /// The lattice's bodies recorded at load, each fire replaying one.
    Recorded,
    /// No graph and no body: every fire walks its plan.
    Eager,
}

/// Two working sets of one hybrid model served in turn by slot refs, as the
/// runtime's shared pool serves them: the first a prompt and greedy decodes,
/// then, the first still holding its record and pages, the second the same
/// prompt from a record and pages of its own. Each set's steps' hashes and
/// top tokens.
fn served_two_sets(
    fixture: &Fixture,
    cache: &Path,
    serving: Serving,
) -> (Vec<(u64, u32)>, Vec<(u64, u32)>) {
    use engine::chunk::SlotKind;
    use engine::fire::RsReset;
    let mut knobs = knobs();
    knobs.pin = match serving {
        Serving::Recorded => knobs.pin,
        Serving::Eager => Some((engine_cuda::Graphs::Off, Some(engine_cuda::Recording::Off))),
    };
    let mut shell = load_with(fixture, cache, knobs);
    // As a worker boots it: the fitted pool backed whole, its facts stated.
    shell.back_whole().expect("the fitted pool is backed whole");
    let facts = shell.chunk_facts().expect("the pool states its chunks");
    facts.validate().expect("the engine's chunk facts hold");
    shell.take_refs();
    let mut refs = Refs::of(facts.clone(), 1);
    let list = refs
        .shape(|kind| matches!(kind, SlotKind::List { window: None, .. }))
        .expect("a kv list");
    let fixed = refs
        .shape(|kind| matches!(kind, SlotKind::Fixed { .. }))
        .expect("a hybrid's recurrent record");
    let SlotKind::Fixed { pieces } = facts.shapes[fixed].kind else {
        unreachable!("the fixed shape")
    };
    let prompt: Vec<u32> = (0..60u32).map(|t| (t * 7 + 11) % 1000).collect();
    let serve = |shell: &mut Shell, refs: &mut Refs| -> Vec<(u64, u32)> {
        let pieces: Vec<u32> = (0..pieces).map(|_| refs.take(fixed)).collect();
        let record = shell.name_record(&pieces).expect("the record is named");
        let mut pages: Vec<u32> = Vec::new();
        let mut fed = 0u32;
        let mut tokens = prompt.clone();
        let mut steps = Vec::new();
        for _ in 0..12 {
            let after = fed + u32::try_from(tokens.len()).expect("a count");
            while (pages.len() as u32) < after.div_ceil(PAGE) {
                pages.push(refs.take(list));
            }
            let word = fixture.trace.facts.word(&Request::new(
                u32::try_from(tokens.len()).expect("a count"),
                false,
            ));
            let seated = Seated {
                pages: &pages,
                held: Some(fed),
                rs_slot: Some(record),
                rs_reset: if fed == 0 {
                    RsReset::Fresh
                } else {
                    RsReset::Held
                },
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
            steps.push((hash_logits(&logits), top));
            tokens = vec![top];
        }
        steps
    };
    let first = serve(&mut shell, &mut refs);
    let second = serve(&mut shell, &mut refs);
    (first, second)
}

/// The resume probe (`PIE_GOLDEN_MODELS`, the gated delta net's miniature
/// by default): a second working set served beside a first by slot refs
/// ([`served_two_sets`]) draws the first's logits bit for bit, with the
/// lattice's bodies recorded at load and with every fire walking its plan;
/// a walk may take other kernels than a body, so the walked fires draw the
/// recorded ones' top tokens. Then a load of many seats in a pool that
/// holds one working set and a few chunks more arms every composition
/// ([`small_pool_arms`]).
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_eager_resume() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS")
        .map_or_else(|_| vec!["qwen35-tiny".to_string()], |_| models());
    let tops = |steps: &[(u64, u32)]| steps.iter().map(|step| step.1).collect::<Vec<u32>>();
    for model in models {
        let fixture = fixture(&model, &cache);
        // The warming serving: every product's algorithm picked and kept.
        let _ = served_two_sets(&fixture, &cache, Serving::Recorded);
        let (recorded, recorded_second) = served_two_sets(&fixture, &cache, Serving::Recorded);
        let (eager, eager_second) = served_two_sets(&fixture, &cache, Serving::Eager);
        for at in 0..recorded.len() {
            println!(
                "resume {model} {at} recorded {:016x}/{:016x} eager {:016x}/{:016x} \
                 tops {} {} {} {}",
                recorded[at].0,
                recorded_second[at].0,
                eager[at].0,
                eager_second[at].0,
                recorded[at].1,
                recorded_second[at].1,
                eager[at].1,
                eager_second[at].1,
            );
        }
        assert_eq!(
            recorded_second, recorded,
            "{model}: recorded, the second set"
        );
        assert_eq!(eager_second, eager, "{model}: eager, the second set");
        assert_eq!(
            tops(&eager),
            tops(&recorded),
            "{model}: eager against recorded"
        );
        small_pool_arms(&fixture, &cache);
    }
}

/// One lane's state after a fire of `tokens` that buffers them all past a
/// fresh state, then a fire of `next` that replays the first `keep` of them
/// and folds those (`RsVerb::Buffer` with `fold`), against slots that fold
/// `tokens` whole and its first `keep` alone. Returns the three states.
fn served_partial_fold(
    fixture: &Fixture,
    cache: &Path,
    tokens: &[u32],
    next: &[u32],
    keep: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model));
    shell.back_whole().expect("the fitted pool is backed whole");
    let page = rs_page(&mut shell);
    let fire = |shell: &mut Shell, slot: u32, tokens: &[u32], rs: RsVerb| {
        let word = fixture.trace.facts.word(&Request::new(
            u32::try_from(tokens.len()).expect("a count"),
            false,
        ));
        shell
            .fire_media(
                &[Seated {
                    rs,
                    ..Seated::of(Lane { slot, word, tokens })
                }],
                &[],
                &[],
            )
            .expect("the fire runs");
    };
    let held = u32::try_from(tokens.len()).expect("a count");
    let end = held + u32::try_from(next.len()).expect("a count");
    let pages: Vec<u32> = (0..end.div_ceil(page)).collect();
    for slot in 0..3 {
        shell.open(slot).expect("the slot opens");
    }
    fire(&mut shell, 0, tokens, RsVerb::Fold);
    let whole = shell.state_bytes(0).expect("slot 0 reads back");
    fire(&mut shell, 1, &tokens[..keep as usize], RsVerb::Fold);
    let kept = shell.state_bytes(1).expect("slot 1 reads back");
    let buffer = |fold: u32| RsVerb::Buffer {
        pages: pages.clone(),
        base: 0,
        folded: 0,
        fold: FoldLen::Host(fold),
        lineage: engine::fire::Lineage::Set,
    };
    fire(&mut shell, 2, tokens, buffer(0));
    fire(&mut shell, 2, next, buffer(keep));
    let partial = shell.state_bytes(2).expect("slot 2 reads back");
    (whole, kept, partial)
}

/// How far apart `a` and `b` lie, read as bf16 words (the state pool's
/// element): the root of the summed squared differences of the finite ones.
fn bf16_apart(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "two slots of one pool");
    let value = |w: &[u8]| {
        f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes([w[0], w[1]])) << 16,
        ))
    };
    a.chunks_exact(2)
        .zip(b.chunks_exact(2))
        .map(|(x, y)| (value(x), value(y)))
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// A fire that buffers a window's rows and a later fire that folds only the
/// first `keep` of them (a speculative verify keeping fewer than its drafts)
/// leave the state a fold of those `keep` rows leaves, to rounding — not one
/// that also took the rows it did not keep. On a model whose PLE conv
/// (Qwen4, Qwen3.8-Flash-Next) commits as it runs, the partial state holds
/// that conv's row as the whole window left it, and so stands nearer the
/// whole fold than the kept one. `PIE_GOLDEN_PARTIAL` names the model
/// (`qwen38-flash-next-micro` by default; `qwen35-tiny` for GDN alone).
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_partial_fold_keeps_its_rows() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let model = std::env::var("PIE_GOLDEN_PARTIAL")
        .unwrap_or_else(|_| "qwen38-flash-next-micro".to_string());
    let fixture = fixture(&model, &cache);
    let tokens: Vec<u32> = (0..6).map(|t| (t * 7 + 11) % 1000).collect();
    let next: Vec<u32> = (6..9).map(|t| (t * 7 + 11) % 1000).collect();
    let keep = 3;
    let (whole, kept, partial) = served_partial_fold(&fixture, &cache, &tokens, &next, keep);
    let to_kept = bf16_apart(&partial, &kept);
    let to_whole = bf16_apart(&partial, &whole);
    let kept_whole = bf16_apart(&kept, &whole);
    println!(
        "golden {model}-partial-fold: the partial state lies {to_kept:.4} from the kept fold and \
         {to_whole:.4} from the whole one (the two folds {kept_whole:.4} apart)"
    );
    assert!(
        kept_whole > 0.0,
        "folding {keep} rows and folding 6 leave one state"
    );
    // Rounding moves the state a little (the chunked and the step scans sum
    // in other orders); a row the fold did not keep moves it by about the
    // distance between the two folds.
    assert!(
        to_kept < 0.1 * kept_whole,
        "a partial fold of {keep} rows left a state {to_kept:.4} from a fold of {keep}, against \
         {kept_whole:.4} between the two folds — the rows it did not keep moved it ({to_whole:.4} \
         from the whole fold)"
    );
}

/// The steps [`golden_staged_experts`] fires: a prefill, then decodes.
const STAGED_SCRIPT: &[Step] = &[
    Step::Prefill(37),
    Step::Decode,
    Step::Decode,
    Step::Prefill(90),
    Step::Decode,
];

/// The budget probe: `PIE_GOLDEN_MODELS` loaded under each of
/// `PIE_GOLDEN_BUDGETS` (bytes, comma-separated), its banks staged where
/// `PIE_GOLDEN_UNTABLED` is set, and fired over [`STAGED_SCRIPT`]: each
/// budget's load refused, or served.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_budget_probe() {
    let cache = cache_dir();
    let budgets: Vec<u64> = std::env::var("PIE_GOLDEN_BUDGETS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|b| b.trim().parse().ok())
        .collect();
    for model in models() {
        let fixture = fixture(&model, &cache);
        // `PIE_GOLDEN_SEARCH`: the least budget a load serves every fire
        // under, between the two stated, and the refusal a byte under it.
        if std::env::var_os("PIE_GOLDEN_SEARCH").is_some() {
            let untabled = std::env::var_os("PIE_GOLDEN_UNTABLED").is_some();
            let (least, said) = least_budget(&fixture, &cache, untabled, budgets[0], budgets[1]);
            println!("budget {model}: the least budget served is {least}");
            println!("budget {model}: a byte under it: {said}");
            continue;
        }
        for &budget in &budgets {
            let mut knobs = knobs();
            if std::env::var_os("PIE_GOLDEN_UNTABLED").is_some() {
                knobs.dense_weights = false;
                knobs.diagnostics.untabled_experts = true;
            }
            let allowance = engine::Allowance {
                budget: Some(budget),
            };
            match load_under(&fixture, &cache, knobs, allowance) {
                Err(why) => println!("budget {model} {budget}: refused at load: {why}"),
                Ok(mut shell) => {
                    for (class, owner, bytes) in shell.ledger().entries() {
                        println!("budget {model} {budget}: ledger {class:?} {owner} {bytes}");
                    }
                    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        staged_steps(&mut shell, &fixture.trace).len()
                    }));
                    println!("budget {model} {budget}: loaded; fires: {served:?}");
                }
            }
        }
    }
}

/// The least budget (bytes, in `lo..=hi`, `lo` refused and `hi` served)
/// under which `fixture` loads and serves every fire of [`STAGED_SCRIPT`],
/// its banks staged where `untabled`, and the load's refusal a byte under
/// it. No budget between loads and then fails a fire: a load is refused,
/// or serves.
fn least_budget(
    fixture: &Fixture,
    cache: &Path,
    untabled: bool,
    mut lo: u64,
    mut hi: u64,
) -> (u64, String) {
    let knobs = || {
        let mut knobs = knobs();
        if untabled {
            knobs.dense_weights = false;
            knobs.diagnostics.untabled_experts = true;
        }
        knobs
    };
    let serves = |budget: u64| -> Result<(), String> {
        let allowance = engine::Allowance {
            budget: Some(budget),
        };
        let mut shell = load_under(fixture, cache, knobs(), allowance)?;
        let fired = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            staged_steps(&mut shell, &fixture.trace).len()
        }));
        assert!(
            fired.is_ok(),
            "{} under {budget} bytes: a fire failed after the load was ready",
            fixture.model
        );
        Ok(())
    };
    assert!(serves(hi).is_ok(), "{}: serves under {hi}", fixture.model);
    assert!(serves(lo).is_err(), "{}: refused under {lo}", fixture.model);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match serves(mid) {
            Ok(()) => hi = mid,
            Err(_) => lo = mid,
        }
    }
    (hi, serves(hi - 1).expect_err("a byte under it refuses"))
}

/// `shell` fired over [`STAGED_SCRIPT`] from a fresh slot: each step's logits.
/// A prefill longer than the rows one fire carries under the load's budget
/// goes in fires of at most that many, the last one's logits the step's.
fn staged_steps(shell: &mut Shell, trace: &Trace) -> Vec<Vec<f32>> {
    shell.open(0).expect("slot 0 opens");
    let most = usize::try_from(shell.budget().max_tokens.max(1)).expect("a count");
    let mut fed = 0u32;
    let mut last = None;
    let mut steps = Vec::with_capacity(STAGED_SCRIPT.len());
    for step in STAGED_SCRIPT {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last.expect("a decode follows a step")],
        };
        fed += u32::try_from(tokens.len()).expect("a count");
        let logits = tokens
            .chunks(most)
            .map(|chunk| fire(shell, trace, chunk))
            .last()
            .expect("a step fires some rows");
        last = Some(argmax(&logits));
        steps.push(logits);
    }
    steps
}

/// The staged-experts probe: `PIE_GOLDEN_MODELS` (quantized MoE miniatures;
/// DeepSeek-V4-Flash's u4g64 one by default) loaded with its routed banks on
/// the host tier and no table of the weight cache reading them
/// (`untabled-experts`), so each fire stages the experts it routes to into
/// the run the load laid. The staging is the bound times its streams, the
/// steps serve the device-resident banks' logits, a fire past the first
/// round takes no device memory, the ledger enters it, no budget loads it
/// and then fails a fire, and a budget a byte short of the least it serves
/// at is refused by the fit, by the staging's name.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_staged_experts() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS").map_or_else(
        |_| vec!["dsv4-flash-mini-u4g64-u2g64-kv-bf16".to_string()],
        |_| models(),
    );
    for model in models {
        let fixture = fixture(&model, &cache);
        let trace = &fixture.trace;
        // The banks on the device, as the base engine seats them.
        let mut resident = load(&fixture, &cache);
        assert_eq!(
            resident.expert_staging().bytes(),
            0,
            "{model}: a load with its banks on the device stages nothing"
        );
        let want = staged_steps(&mut resident, trace);
        drop(resident);

        let staged = || {
            let mut knobs = knobs();
            knobs.dense_weights = false;
            knobs.diagnostics.untabled_experts = true;
            knobs
        };
        let mut shell = load_with(&fixture, &cache, staged());
        let staging = shell.expert_staging().clone();
        assert_eq!(
            shell
                .ledger()
                .of(engine::memory::Class::Fire, "staged experts"),
            staging.bytes(),
            "{model}: the ledger enters the staging as a fire's"
        );
        let fact = staging
            .fact()
            .unwrap_or_else(|| panic!("{model}: no bank is staged"));
        println!("staged {model}: {fact}");
        // The bound, from the trace: each staged bank's codes hold the most
        // experts one fire routes to, the routes one fire's.
        let classes = poem_ir::residency::residency(trace);
        let bytes = engine_cuda::weights::plane_bytes(trace).expect("sized");
        let rows = u64::from(staging.rows());
        let (mut codes, mut routes) = (0u64, 0u64);
        for bank in staging.banks() {
            let at = trace
                .params
                .iter()
                .position(|param| &param.name == bank)
                .unwrap_or_else(|| panic!("the trace declares `{bank}`"));
            let poem_ir::residency::Residency::Expert(atom) = classes[at] else {
                panic!("`{bank}` is staged and routes no expert");
            };
            let experts = u64::from(atom.experts);
            let slots = experts.min(rows * u64::from(atom.top_k));
            codes = codes.max(slots * (bytes[at] / experts));
            routes = routes.max(rows * u64::from(atom.top_k) * 4);
        }
        let runs = staging.runs();
        assert_eq!(runs[0], codes, "{model}: the codes' run is the bound");
        assert_eq!(runs[3], routes, "{model}: the routes' run is one fire's");
        let lane: u64 = runs.iter().map(|run| run.next_multiple_of(256)).sum();
        assert!(!staging.streams().is_empty());
        assert_eq!(
            staging.bytes(),
            lane * staging.streams().len() as u64,
            "{model}: a lane a stream"
        );
        println!(
            "staged {model}: {} bank(s), runs {runs:?}, {} stream(s), {} bytes for {rows} rows",
            staging.banks().len(),
            staging.streams().len(),
            staging.bytes(),
        );

        // The first round builds what a fire builds; the second takes
        // nothing more of the device.
        let first = staged_steps(&mut shell, trace);
        let free = engine_cuda::device::free_bytes();
        let heaps = engine_cuda::device::held::heap_bytes();
        let second = staged_steps(&mut shell, trace);
        assert_eq!(
            engine_cuda::device::free_bytes(),
            free,
            "{model}: a staged fire took device memory"
        );
        assert_eq!(
            engine_cuda::device::held::heap_bytes(),
            heaps,
            "{model}: a staged fire grew a heap"
        );
        assert_eq!(second, first, "{model}: the second round is the first's");
        for (at, (got, want)) in first.iter().zip(&want).enumerate() {
            let most = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!(
                "staged {model} {at}: most |diff| {most:e} from resident, tops {} {}",
                argmax(got),
                argmax(want)
            );
            assert_eq!(
                argmax(got),
                argmax(want),
                "{model} step {at}: the staged top token"
            );
            assert!(most < 0.25, "{model} step {at}: |diff| {most:e}");
        }

        drop(shell);
        // The staging is a term of the fit: under the least budget the
        // load serves at, less a byte, the fit refuses it, naming the banks
        // and the staging's bytes.
        let (least, said) = least_budget(&fixture, &cache, true, 1 << 30, 16 << 30);
        println!("staged {model}: least budget {least}; a byte under it: {said}");
        assert!(
            staging
                .banks()
                .iter()
                .all(|bank| said.contains(&format!("`{bank}`"))),
            "{model}: the refusal names the banks: {said}"
        );
        assert!(
            said.contains(&format!("{} bytes of staged experts", staging.bytes())),
            "{model}: and the bytes: {said}"
        );
    }
}

/// The decisions `model` reads served over [`SCRIPT`] under the profile the
/// probe's loads resolve: those a dispatch consulted for its shapes and kinds.
fn read_set(model: &str, cache: &Path) -> Vec<&'static str> {
    let (name, facts) = engine_cuda::chip::probe(0).expect("a CUDA device answers");
    let (profile, _) = engine_cuda::chip::resolve(&name, &facts, knobs().profiles.as_deref());
    let fixture = fixture(model, cache);
    let mut knobs = knobs();
    knobs.chip = Some(profile.clone());
    let mut shell = load_with(&fixture, cache, knobs);
    shell.open(0).expect("slot 0 opens");
    let mut fed = 0u32;
    let mut last = 0;
    for step in SCRIPT {
        let tokens: Vec<u32> = match *step {
            Step::Prefill(rows) => (fed..fed + rows).map(|t| (t * 7 + 11) % 1000).collect(),
            Step::Decode => vec![last],
        };
        fed += u32::try_from(tokens.len()).expect("a count");
        last = argmax(&fire(&mut shell, &fixture.trace, &tokens));
    }
    drop(shell);
    profile.read()
}

/// A kernel decision only a routed, recurrent, MLA, indexed or NVFP4 model
/// can reach. The compiler's prices (`compiler.cost.*`) and fusion rules
/// (`fuse.*`) are not among them though every load reads them: `pie tune`
/// serves a candidate of those only where it changes the compiled plan
/// (`profile::search`'s `plan_only`), and a dense model's plan never moves
/// with an ssm price or an MoE fusion.
fn family_only(id: &str) -> bool {
    [
        "kernel.moe.",
        "kernel.ssm.",
        "kernel.mla.",
        "kernel.sdpa.mla",
        "kernel.sdpa.index",
        "kernel.sdpa.compressed",
    ]
    .iter()
    .any(|p| id.starts_with(p))
        || id.contains("nvfp4")
}

/// A dense bf16 model reads no routed, recurrent, MLA, indexed or NVFP4
/// decision, so `pie tune` times none of them on it; a routed model (the
/// control) reads its routed ones. `PIE_GOLDEN_MODELS` names more models
/// whose read sets are printed. Fixtures as [`golden_fixtures`].
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn a_dense_model_reads_no_routed_or_recurrent_decision() {
    let cache = cache_dir();
    let dense = read_set("qwen3-0.6b", &cache);
    println!("reads qwen3-0.6b {dense:?}");
    let stray: Vec<&str> = dense.iter().copied().filter(|id| family_only(id)).collect();
    assert!(stray.is_empty(), "qwen3-0.6b reads {stray:?}");
    // The control: a dense model reads its products' decisions, whichever
    // a backend's dense dispatch consults (CUDA's skinny rows, ROCm's turn
    // and live tiles).
    assert!(
        dense.iter().any(|id| id.starts_with("kernel.gemm.")),
        "qwen3-0.6b reads no dense product decision: {dense:?}"
    );
    let routed = read_set("gptoss-20b-mini-u4g64-mxfp4-kv-bf16", &cache);
    println!("reads gptoss-20b-mini-u4g64-mxfp4-kv-bf16 {routed:?}");
    assert!(
        routed.iter().any(|id| id.starts_with("kernel.moe.")),
        "the routed control reads no routed decision: {routed:?}"
    );
    if std::env::var_os("PIE_GOLDEN_MODELS").is_some() {
        for model in models() {
            println!("reads {model} {:?}", read_set(&model, &cache));
        }
    }
}

/// CONTRACT-BC 19: a load serving its windowed rows as ring + buffer arms
/// every composition it fires, as its window lists do (the arming pass's
/// synthetic lanes fold on records of their own): `PIE_GOLDEN_MODELS`
/// (default the mini Gemma and MiMo, windowed beside full attention, MiMo's
/// routed), many seats.
#[test]
#[ignore = "a probe run by hand: see the module's doc"]
fn golden_windowed_arms() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let cache = cache_dir();
    let models = std::env::var("PIE_GOLDEN_MODELS").map_or_else(
        |_| vec!["gemma4-e4b-mini-l6".to_string(), "mimo26-flash-mini".to_string()],
        |_| models(),
    );
    for model in models {
        let fixture = fixture(&model, &cache);
        let mut knobs = knobs();
        knobs.diagnostics.windowed_state = true;
        let shell = load_lanes(&fixture, &cache, knobs, 16, None);
        assert!(
            !shell.windowed_layers().is_empty(),
            "{model}: the load keeps rings"
        );
        let armed = shell.armed().expect("the load armed its lattice");
        println!(
            "windowed arms {model}: {} of {} compositions",
            armed.armed, armed.wanted
        );
        assert!(
            armed.wanted > 0 && armed.armed == armed.wanted,
            "{model}: {} of {} compositions armed ({:?})",
            armed.armed,
            armed.wanted,
            armed.last_refusal
        );
    }
}
