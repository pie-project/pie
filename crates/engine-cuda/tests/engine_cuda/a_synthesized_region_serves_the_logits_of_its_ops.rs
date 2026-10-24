#![cfg(feature = "cuda")]

//! A model served with `compiler.synth_regions` on, each chain of
//! elementwise and row ops one generated kernel, answers the logits it
//! answers with the flag off, where the chain's ops run as their hand
//! kernels: a prefill and the decode steps after it, armed and captured.
//!
//! Bit for bit, on this draw. A residual norm's chain stores the stream as
//! the hand kernels do and computes the norm in their order, but for its
//! sum of squares: one at a time where the hand register kernel adds them
//! two at a time, which moves a normed element by one bf16 ulp about once
//! in two million (`kernels_cuda`'s `elemwise::synth` tests count them).

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const PAGE: u32 = 16;

/// The residual norm unfused, so each layer's `residual_add` and `rmsnorm`
/// form a chain, folded under the flag into one generated kernel.
const UNFUSED: &str = "[fuse]\nresidual_add_rmsnorm = false\n";

const SYNTH: &str = "[compiler]\nsynth_regions = true\n";

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().expect("one at a time lock poisoned")
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A bf16 checkpoint of `trace`'s own planes, seeded by name: norms near
/// one, the rest small.
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
        assert_eq!(param.dtype, Dtype::Bf16, "`{}` is not bf16", param.name);
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut seed = fnv(&param.name);
        let mut bytes = Vec::with_capacity(count * 2);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                ztensor::Leaf::BF16,
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// A catalog model's trace and a seeded checkpoint of its planes.
struct Fixture {
    model: &'static str,
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    _dir: Scratch,
}

fn fixture(model: &'static str) -> Fixture {
    let trace = poem_compiler::catalog::trace_of(model, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let dir = std::env::temp_dir().join(format!("pie-synth-{model}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    let container = dir.join("model.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        model,
        trace,
        contract,
        container,
        _dir: Scratch(dir),
    }
}

/// The model loaded for one lane of `tokens` a fire, graphs on, under the
/// profile `overrides` states.
fn load(fixture: &Fixture, overrides: &str, tokens: u32) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, tokens),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: 2 * tokens,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::profile(overrides)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("{} loads under {overrides:?}: {why}", fixture.model))
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

/// `tokens` fired as slot 0's next step: its logits, and how long the fire
/// took in microseconds.
fn step(shell: &mut Shell, trace: &Trace, tokens: &[u32]) -> (Vec<f32>, f64) {
    let word = trace.facts.word(&Request::new(
        u32::try_from(tokens.len()).expect("a count"),
        false,
    ));
    let start = std::time::Instant::now();
    let mut rows = shell
        .fire(&[Lane {
            slot: 0,
            word,
            tokens,
        }])
        .expect("the step fires");
    (rows.swap_remove(0), start.elapsed().as_secs_f64() * 1e6)
}

fn prompt(tokens: u32) -> Vec<u32> {
    (0..tokens).map(|t| (t * 7 + 11) % 1000).collect()
}

/// `got` is `want` bit for bit, step by step.
fn bit_for_bit(got: &[Vec<f32>], want: &[Vec<f32>], against: &str) {
    assert_eq!(got.len(), want.len());
    for (step, (got, want)) in got.iter().zip(want).enumerate() {
        assert_eq!(got.len(), want.len(), "step {step}: the logits' width");
        let apart: Vec<usize> = (0..got.len())
            .filter(|&at| got[at].to_bits() != want[at].to_bits())
            .collect();
        assert!(
            apart.is_empty(),
            "step {step}: {} of {} logits are not {against}'s, the first at {}: {} and {}",
            apart.len(),
            got.len(),
            apart[0],
            got[apart[0]],
            want[apart[0]]
        );
    }
}

#[test]
fn a_synthesized_region_serves_the_logits_of_its_ops() {
    const TOKENS: u32 = 256;
    const DECODES: usize = 4;
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture("qwen3-micro-text");
    // A prefill's logits, then each greedy decode step's after it.
    let served = |overrides: &str, generates: bool| {
        let mut shell = load(&fixture, overrides, TOKENS);
        assert_eq!(
            !shell.regions().is_empty(),
            generates,
            "under {overrides:?}, the load generates a kernel exactly where a chain folds"
        );
        shell.open(0).expect("slot 0 opens");
        let mut steps = vec![step(&mut shell, &fixture.trace, &prompt(37)).0];
        for _ in 0..DECODES {
            let next = argmax(steps.last().expect("a step"));
            steps.push(step(&mut shell, &fixture.trace, &[next]).0);
        }
        steps
    };
    let ops = served(UNFUSED, false);
    let hand = served("", false);
    let synth = served(&format!("{UNFUSED}{SYNTH}"), true);
    bit_for_bit(&synth, &ops, "the unfused ops");
    bit_for_bit(&synth, &hand, "the hand residual norm");
    // With the hand rule on, the residual norm is one fused op with matmuls
    // on both sides: no chain forms and the flag folds nothing.
    let lone = served(SYNTH, false);
    bit_for_bit(&lone, &hand, "the hand residual norm");
}

/// Qwen3-8B's shape (seeded weights): a prefill step and a decode step with
/// its residual norms as the hand fused kernel, as their two ops, and as
/// one generated kernel each, the median of each fire's wall time:
/// `cargo test --release ... -- --ignored synth_regions_step_rate --nocapture`.
#[test]
#[ignore = "a rate bench: --release -- --ignored synth_regions_step_rate --nocapture"]
fn synth_regions_step_rate() {
    const PREFILL: u32 = 2048;
    const PREFILLS: usize = 8;
    const DECODES: usize = 128;
    let _one = serialized();
    let fixture = fixture("qwen3-8b");
    let median = |mut us: Vec<f64>| {
        us.sort_by(f64::total_cmp);
        us[us.len() / 2]
    };
    eprintln!("residual norm       prefill {PREFILL} ms  decode us  generated kernels");
    for (what, overrides) in [
        ("hand fused kernel", String::new()),
        ("two hand kernels", UNFUSED.to_owned()),
        ("generated kernel", format!("{UNFUSED}{SYNTH}")),
    ] {
        let mut shell = load(&fixture, &overrides, PREFILL);
        let mut prefills = Vec::with_capacity(PREFILLS);
        let mut next = 0;
        for _ in 0..PREFILLS {
            shell.open(0).expect("slot 0 opens");
            let (logits, us) = step(&mut shell, &fixture.trace, &prompt(PREFILL));
            prefills.push(us);
            next = argmax(&logits);
        }
        let mut decodes = Vec::with_capacity(DECODES);
        for _ in 0..DECODES {
            let (logits, us) = step(&mut shell, &fixture.trace, &[next]);
            decodes.push(us);
            next = argmax(&logits);
        }
        eprintln!(
            "{what:<18}  {:>15.3}  {:>9.1}  {:>17}",
            median(prefills) / 1e3,
            median(decodes),
            shell.regions().len()
        );
    }
}
