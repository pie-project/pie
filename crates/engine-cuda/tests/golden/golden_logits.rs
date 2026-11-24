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
//! cargo test -p engine-cuda --features cuda --test golden \
//!     golden_fixtures -- --ignored --nocapture
//! flock /tmp/pie-gpu.lock cargo test -p engine-cuda --features cuda \
//!     --test golden golden_logits -- --ignored --nocapture --test-threads=1
//! ```
//!
//! It prints one `golden <model> <step> <hash> <argmax>` line a step and one
//! `golden <model> all <hash>` line a model. `PIE_GOLDEN_MODELS` (a comma
//! list) picks and orders the models; `PIE_GOLDEN_OUT` writes the lines to a
//! file; `PIE_GOLDEN_CHECK` compares them with such a file and fails on the
//! first model whose hash moved, unless `PIE_GOLDEN_AGAINST` names the
//! earlier run's `PIE_GOLDEN_LOGITS` rows and each of this run's lies within
//! the tolerance of its own ([`within_tolerance`]).
//!
//! The decode products' kernels are held across builds: every golden load
//! keeps the product times it chose its kernels by in the cache's `picks`,
//! and every later one is served with them as they stand (`Knobs::chip`,
//! a test's pinned profile), so a build whose kernel crate changed picks
//! what the earlier build picked and its hashes answer for its code, not for
//! a re-timing. Only a candidate whose own code changed is timed again.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Lane, Shell};
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
        kv_budget: std::env::var("PIE_GOLDEN_POOL_MIB")
            .ok()
            .and_then(|mib| mib.parse::<u64>().ok())
            .map(|mib| engine_cuda::store::KvSize::Bytes(mib << 20)),
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
        steps: engine::StepRules::runtime(),
    })
    .map_err(|why| why.to_string())
    .inspect(|shell| record_picks(cache, shell))
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
    knobs.diagnostics.arm_trace = std::env::var_os("PIE_GOLDEN_ARM_TRACE").is_some();
    // `PIE_GOLDEN_TIER_INLINE`: tier copies ride the compute stream.
    knobs.diagnostics.tier_inline = std::env::var_os("PIE_GOLDEN_TIER_INLINE").is_some();
    // No weight shape stated: the tables and routed banks dense on the
    // device, as the base engine seats them. The weight cache's own probes
    // load with its shapes ([`load_cached`]).
    knobs.weights = engine_cuda::WeightPin::Dense;
    knobs.chip = picked(&cache_dir(), knobs.profiles.as_deref());
    knobs
}

/// The cache's product picks ([`record_picks`]): the decode products' times
/// an earlier golden load chose their kernels by, one `kind rows k m kernel
/// ns spread` line each.
fn picks_path(cache: &Path) -> PathBuf {
    cache.join("picks")
}

fn read_picks(cache: &Path) -> Vec<profile::rates::ProductTime> {
    let text = std::fs::read_to_string(picks_path(cache)).unwrap_or_default();
    text.lines()
        .filter_map(|line| {
            let w: Vec<&str> = line.split_whitespace().collect();
            let [kind, rows, k, m, kernel, ns, spread] = w[..] else {
                return None;
            };
            Some(profile::rates::ProductTime {
                kind: kind.parse().ok()?,
                rows: rows.parse().ok()?,
                k: k.parse().ok()?,
                m: m.parse().ok()?,
                kernel: kernel.parse().ok()?,
                ns: ns.parse().ok()?,
                spread: spread.parse().ok()?,
            })
        })
        .collect()
}

/// Keeps what `shell` chose its decode products' kernels by in the cache's
/// picks, beside those already kept (a kept entry stays; a shape no load
/// met before joins). A golden load served after takes them as they stand
/// ([`picked`]), so its logits answer for the code, not for a re-timing.
fn record_picks(cache: &Path, shell: &Shell) {
    let mut picks = read_picks(cache);
    let before = picks.len();
    for time in shell.product_times() {
        if picks.iter().all(|kept| kept.key() != time.key()) {
            picks.push(*time);
        }
    }
    if picks.len() == before {
        return;
    }
    let text: String = picks
        .iter()
        .map(|t| {
            format!(
                "{} {} {} {} {} {} {}\n",
                t.kind, t.rows, t.k, t.m, t.kernel, t.ns, t.spread
            )
        })
        .collect();
    std::fs::write(picks_path(cache), text)
        .unwrap_or_else(|why| panic!("{}: {why}", picks_path(cache).display()));
}

/// The device's profile with the cache's product picks laid over its
/// rates, as a pinned profile (`Knobs::chip`): a golden load serves each
/// product shape an earlier one met with the kernel it chose then, whatever
/// this build's stored measurements say, and times only the shapes (or the
/// changed kernels) none met. `None` while the cache keeps no picks.
fn picked(cache: &Path, profiles: Option<&Path>) -> Option<profile::Profile> {
    let picks = read_picks(cache);
    if picks.is_empty() {
        return None;
    }
    let (name, facts) = engine_cuda::chip::probe(0)?;
    let (profile, _) = engine_cuda::chip::resolve(&name, &facts, profiles);
    let mut tuned = profile::rates::Tuned::default();
    tuned.rates.product_ns = Some(picks);
    Some(profile.with_tuned(tuned))
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
            1,
            1,
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
/// tiers below the device: the kv list, the window lists
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
        if !moved.is_empty() && within_tolerance() {
            eprintln!(
                "golden: hashes moved, every row within tolerance of the earlier run's:\n  {}",
                moved.join("\n  ")
            );
            return;
        }
        assert!(moved.is_empty(), "logits moved:\n  {}", moved.join("\n  "));
    }
}

/// A step's logits within this share of their spread of the earlier run's
/// (the bf16 rule the synthesized-region tests hold a reordered sum to).
const TOLERANCE: f32 = 2e-2;

/// Whether this run's rows (`PIE_GOLDEN_LOGITS`) each lie within
/// [`TOLERANCE`] of an earlier run's (`PIE_GOLDEN_AGAINST`, that run's
/// `PIE_GOLDEN_LOGITS`), row for row, with the same top token unless the
/// earlier row's top two are that close: what a moved hash is judged by
/// where the picks could not be held (a product kernel whose code changed is
/// timed again). Unset either, and a moved hash fails as it stands.
fn within_tolerance() -> bool {
    let (Some(ours), Some(theirs)) = (
        std::env::var_os("PIE_GOLDEN_LOGITS"),
        std::env::var_os("PIE_GOLDEN_AGAINST"),
    ) else {
        return false;
    };
    let read = |dir: &std::ffi::OsStr, at: usize| -> Option<Vec<f32>> {
        let bytes = std::fs::read(Path::new(dir).join(format!("{at}.f32"))).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        )
    };
    let mut rows = 0usize;
    while let Some(want) = read(&theirs, rows) {
        let Some(got) = read(&ours, rows) else {
            eprintln!("golden: row {rows} of the earlier run has none of this run's");
            return false;
        };
        let (lo, hi) = want
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let deviation = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        let bound = TOLERANCE * (hi - lo);
        let (top, picked) = (argmax(&want), argmax(&got));
        let gap = want[top as usize] - want[picked as usize];
        if got.len() != want.len() || deviation > bound || (top != picked && gap > 2.0 * deviation)
        {
            eprintln!(
                "golden: row {rows} {deviation} apart over a spread of {} (bound {bound}); top \
                 {picked}, the earlier run's {top}",
                hi - lo
            );
            return false;
        }
        rows += 1;
    }
    rows > 0
}

/// The md5 of every stored input a load selects its kernels and bodies by:
/// the profile its measurements are kept in (`knobs`: the cache's
/// `profiles`), the bodies records and the host covers, every file in path
/// order, by `md5sum`; "none" where there is no file. A cold cache reads as
/// another configuration.
fn profile_md5(cache: &Path) -> String {
    let mut files = vec![picks_path(cache)].into_iter().filter(|p| p.is_file()).collect::<Vec<_>>();
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
        kv_budget: None,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        ..crate::common::boot(
            fixture.trace.clone(),
            &fixture.contract,
            &fixture.container,
            Budget::new(1, TOKENS),
            CONTEXT,
            4,
        )
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
        kv_budget: None,
        knobs: buffering(knobs()),
        cache_dir: Some(cache),
        ..crate::common::boot(
            fixture.trace.clone(),
            &fixture.contract,
            &fixture.container,
            Budget::new(1, tokens),
            CONTEXT,
            4,
        )
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
        // row's step alone as much as a lag of eleven.
        assert!(
            worst < 0.05 && flips == 0,
            "at a lag of {lag} the lagged lane's logits lie {worst:e} of their scale from the \
             folded lane's, {flips} greedy token(s) flipped"
        );
        assert_eq!(unbodied, 0, "a lagged decode walked with no body");
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
        steps: engine::StepRules::runtime(),
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
        kv_budget: kv_budget.map(engine_cuda::store::KvSize::Bytes),
        knobs,
        cache_dir: Some(cache),
        ..crate::common::boot(
            fixture.trace.clone(),
            &fixture.contract,
            &fixture.container,
            Budget::new(lanes, TOKENS),
            CONTEXT,
            lanes,
        )
    })
    .unwrap_or_else(|why| panic!("{} loads: {why}", fixture.model))
}

/// A load serving its windowed rows as ring + buffer arms
/// every composition it fires (the arming pass's synthetic lanes fold on
/// records of their own): `PIE_GOLDEN_MODELS`
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
        let shell = load_lanes(&fixture, &cache, knobs(), 16, None);
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
