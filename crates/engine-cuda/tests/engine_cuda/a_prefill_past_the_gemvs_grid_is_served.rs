#![cfg(feature = "device")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const TOP_K: u32 = 4;
const WIDE: u32 = 16_400;

const BOTH: u32 = 8_192;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

const PAGE: u32 = 16;
struct Text {
    name: &'static str,
    ceiling: u32,
}

fn traced(id: &str) -> Trace {
    poem_compiler::catalog::trace_of(id, Dtype::Bf16, Dtype::Bf16, Platform::Cuda)
}

const MICRO: Text = Text {
    name: "qwen3-a3b-micro",
    ceiling: WIDE,
};

const UNCACHED: Text = Text {
    name: "qwen3-a3b-uncached-bank",
    ceiling: BOTH,
};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(tag: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("pie-moe-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    Scratch(dir)
}

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

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
        assert_eq!(
            param.dtype,
            Dtype::Bf16,
            "`{}` is {:?}; this fixture writes a bf16 text",
            param.name,
            param.dtype
        );
        let mut bytes = Vec::with_capacity(count * 2);
        let mut seed = fnv(&param.name);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
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

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "{what}: logit {at} is {value}");
    }
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

struct Fixture {
    text: Text,
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    _dir: Scratch,
}

fn fixture(text: Text) -> Fixture {
    let trace = traced(text.name);
    let dir = scratch(text.name);
    let container = dir.0.join("micro.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes is read by the plan's own names");
    drop(source);
    Fixture {
        text,
        trace,
        contract,
        container,
        _dir: dir,
    }
}

/// The routed matmul's leg: the grouped tcgen05 GEMM (where the device has
/// it), the cuBLAS grouped GEMM, or one GEMV a route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    #[cfg_attr(feature = "rocm", allow(dead_code))]
    Tcgen05,
    Grouped,
    Gemv,
}

/// The grouped legs this build has: ROCm's grouped GEMM is hipBLASLt's.
#[cfg(not(feature = "rocm"))]
const GROUPED: [Leg; 2] = [Leg::Grouped, Leg::Tcgen05];
#[cfg(feature = "rocm")]
const GROUPED: [Leg; 1] = [Leg::Grouped];

fn load(fixture: &Fixture, leg: Leg) -> engine_cuda::Result<Shell> {
    let (tc5, grouped) = match leg {
        Leg::Tcgen05 => (true, true),
        Leg::Grouped => (false, true),
        Leg::Gemv => (false, false),
    };
    let tc5 = if cfg!(feature = "rocm") {
        String::new()
    } else {
        format!("moe.tc5 = {tc5}\n")
    };
    let chip = engine_cuda::chip::profile(&format!(
        "[kernel]\nmoe.grouped = {grouped}\n{tc5}[engine]\ngraphs = \"off\"\n"
    ));
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(1, fixture.text.ceiling),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: fixture.text.ceiling.next_multiple_of(PAGE),
        slots: 2,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(chip),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
}

fn fire_at(fixture: &Fixture, tokens: u32, leg: Leg) -> engine_cuda::Result<(Vec<Vec<f32>>, f64)> {
    let mut shell = load(fixture, leg)?;
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let prompt: Vec<u32> = (0..tokens).map(|t| (t * 7 + 11) % 2048).collect();
    let word = fixture.trace.facts.word(&Request::new(tokens, false));
    shell.fire(&[Lane {
        slot: 0,
        word,
        tokens: &prompt,
    }])?;
    let start = std::time::Instant::now();
    let rows = shell.fire(&[Lane {
        slot: 1,
        word,
        tokens: &prompt,
    }])?;
    Ok((rows, start.elapsed().as_secs_f64() * 1e3))
}

#[test]
fn a_prefill_past_the_gemvs_grid_is_served_every_case() {
    a_prefill_past_the_gemvs_grid_is_served();
    the_two_legs_answer_the_same_model();
    a_bank_past_the_cache_is_where_grouping_pays();
}

fn a_prefill_past_the_gemvs_grid_is_served() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture(MICRO);
    const {
        assert!(
            WIDE * TOP_K > 65_535,
            "the fixture's route run has to pass the GEMV's grid ceiling or this proves nothing"
        )
    };

    for leg in GROUPED {
        let (logits, _) = fire_at(&fixture, WIDE, leg).unwrap_or_else(|why| {
            panic!(
                "a prefill of 65600 routes fires on the {leg:?} leg — if this refused, the \
                 engine never reached it and the routed matmul is still one GEMV per route: \
                 {why}"
            )
        });
        finite(&logits[0], "the wide prefill");
    }

    // The GEMV carries a route run past its grid's y axis on z, so it serves the fire too.
    #[cfg(not(feature = "rocm"))]
    {
        let (logits, _) = fire_at(&fixture, WIDE, Leg::Gemv).unwrap_or_else(|why| {
            panic!("a prefill of 65600 routes fires on the GEMV leg past grid y: {why}")
        });
        finite(&logits[0], "the wide prefill on the GEMV");
    }
    // ROCm's refuses it, naming the route run against the grid it does not fit.
    #[cfg(feature = "rocm")]
    {
        let why = fire_at(&fixture, WIDE, Leg::Gemv)
            .expect_err("with the grouped leg declined, a 65600-route fire is past the GEMV's grid")
            .to_string();
        assert!(why.contains("65600") || why.contains("65535"), "{why}");
    }
}

fn the_two_legs_answer_the_same_model() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture(MICRO);

    let (gemv, gemv_ms) = fire_at(&fixture, BOTH, Leg::Gemv).expect("the per-route GEMV fires");
    finite(&gemv[0], "gemv");
    for leg in GROUPED {
        let (grouped, grouped_ms) = fire_at(&fixture, BOTH, leg).expect("the grouped leg fires");
        finite(&grouped[0], "grouped");
        agree(leg, &grouped, grouped_ms, &gemv, gemv_ms);
    }
}

/// A grouped leg's logits against the GEMV's: a reordered sum moves the
/// tail, not the middle.
fn agree(leg: Leg, grouped: &[Vec<f32>], grouped_ms: f64, gemv: &[Vec<f32>], gemv_ms: f64) {
    assert_eq!(
        grouped[0].len(),
        gemv[0].len(),
        "the two legs read out different vocabularies"
    );
    let spread = gemv[0].iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - gemv[0].iter().copied().fold(f32::INFINITY, f32::min);
    let worst = grouped[0]
        .iter()
        .zip(&gemv[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mut gaps: Vec<f32> = grouped[0]
        .iter()
        .zip(&gemv[0])
        .map(|(a, b)| (a - b).abs())
        .collect();
    gaps.sort_by(f32::total_cmp);
    let median = gaps[gaps.len() / 2];
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    let pick = |row: &[f32]| {
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (at, v)| {
                if *v > best.1 { (at, *v) } else { best }
            })
            .0
    };
    eprintln!(
        "{BOTH} tokens, {} routes: {leg:?} {grouped_ms:.1} ms, gemv {gemv_ms:.1} ms \
         ({:.2}x); gap worst {worst:.4} mean {mean:.4} median {median:.4} over a spread \
         of {spread:.2}; argmax {} vs {}",
        BOTH * TOP_K,
        gemv_ms / grouped_ms,
        pick(&grouped[0]),
        pick(&gemv[0]),
    );
    assert!(
        mean <= 0.01 * spread,
        "the two legs answer differently in the MEAN: {mean} over a spread of {spread}. \
         A reordered sum moves the tail, not the middle"
    );
    assert!(
        worst <= 0.05 * spread,
        "one logit differs by {worst} over a spread of {spread}, which is more than a \
         near-tie rounding the other way"
    );
}

fn a_bank_past_the_cache_is_where_grouping_pays() {
    let _one = serialized();
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let fixture = fixture(UNCACHED);

    let (gemv, gemv_ms) = fire_at(&fixture, BOTH, Leg::Gemv).expect("the per-route GEMV fires");
    finite(&gemv[0], "gemv");
    for leg in GROUPED {
        let (grouped, grouped_ms) = fire_at(&fixture, BOTH, leg).expect("the grouped leg fires");
        finite(&grouped[0], "grouped");
        pays(leg, &grouped, grouped_ms, &gemv, gemv_ms);
    }
}

/// A grouped leg reading the bank once a batch against the GEMV reading it
/// once a route: the same logits, sooner.
fn pays(leg: Leg, grouped: &[Vec<f32>], grouped_ms: f64, gemv: &[Vec<f32>], gemv_ms: f64) {
    let spread = gemv[0].iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - gemv[0].iter().copied().fold(f32::INFINITY, f32::min);
    let mean = grouped[0]
        .iter()
        .zip(&gemv[0])
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / grouped[0].len() as f32;

    eprintln!(
        "{} tokens, {} routes over a 134 MiB bank: {leg:?} {grouped_ms:.1} ms, \
         gemv {gemv_ms:.1} ms ({:.1}x); mean logit gap {mean:.4} over a spread of {spread:.2}",
        BOTH,
        BOTH * TOP_K,
        gemv_ms / grouped_ms,
    );
    assert!(
        mean <= 0.01 * spread,
        "the two legs answer differently in the MEAN: {mean} over a spread of {spread}"
    );
    assert!(
        grouped_ms < gemv_ms,
        "grouping a bank that does not fit in cache ({leg:?}) took {grouped_ms:.1} ms against the \
         per-route GEMV's {gemv_ms:.1} ms; this is the shape it exists for"
    );
}
