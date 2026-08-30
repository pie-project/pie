//! **Routed experts stream, and the logits do not notice** (alto design §7,
//! wave W-a — the Metal plane's reading of it).
//!
//! The claim under test is the one sentence a residency mechanism has to
//! honour: *a load that holds less says exactly what a load that holds
//! everything says*. Routing is computed on device, so no host decision can
//! precede a fire and arrange for the experts it will need to be there. The
//! CUDA sibling's answer is an indirection table and a UVA read on a miss;
//! this plane's answer is different because the platform is, and the
//! difference is what this file measures:
//!
//! ```text
//! (a) a load whose device budget seats HALF the experts fires, and its
//!     logits are byte for byte the logits full residency produces
//! (b) the mechanism actually moved — seats were copied, segments were cut,
//!     and the occupancy after the run is not the identity prefix it began at
//! (c) the refusals: a budget under the dense planes, a budget that seats no
//!     whole expert, and a capped budget over a plan with nothing routed
//! ```
//!
//! # Why there is no "(b) costs no synchronize" claim here
//!
//! Because on this plane it WOULD cost one, and saying otherwise would be the
//! lie the design note exists to prevent. A wired seat is bytes an
//! already-committed dispatch may still be reading, and this shell has no
//! fence and no second copy of the weight store — so each segment is closed
//! with a BLOCKING commit and the run-ahead of a streamed load collapses to
//! one. `engine_metal::serve`'s header prices that in full; what this file
//! asserts is that the price buys the right answer.
//!
//! # The fixture, and why it is not a catalog SKU
//!
//! `Model::a3b_micro` is `qwen35-a3b`'s own text at a size two loads of which
//! fit in a gate's patience: 4 layers, 32 experts at top-k 4, hidden 512,
//! vocab 2048. `a3b` itself is 64 GiB and this file's central claim needs BOTH
//! loads on one device. Its checkpoint is written here, from the trace's own
//! params, with deterministic pseudo-random bytes: what is under test is the
//! residency machinery, and a machinery that moves the wrong bytes fails
//! against arbitrary weights exactly as it fails against trained ones.
//!
//! # Every fire is one token wide, and that is the mechanism's shape
//!
//! Every distinct expert ONE SEGMENT routes to must be seated at once — the
//! segment's matmuls all run behind its cut — so a wide prefill over a small
//! slab is refused by name rather than served wrong (`experts::Tier::evict`
//! says so, and the fix it names is a larger budget or fewer tokens). A
//! one-token fire touches at most `top_k` experts per layer, which is four
//! here, so the budgets below are comfortably above the floor and the
//! refusal is not what this file is exercising.
//!
//! ```text
//! cargo test -p engine-metal --release --test routed_experts_stream -- --nocapture
//! ```
//!
//! # Gating
//!
//! As `serve_smoke.rs`: skipped at run time when the machine has no device,
//! rather than `#[ignore]`d — an ignored test on the one box that could run
//! it is a test nobody runs. The refusals below need no device at all and are
//! asserted unconditionally.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use checkpoint::contract::ModelContract;
use engine_metal::experts::{Attachments, Plan};
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Dtype, Platform, Request};
use model_ir::{ParamSource, Trace};

/// The token every fire of this file feeds, and the ones it feeds back. An
/// arbitrary id inside the micro text's 2048-token vocabulary — the model is
/// synthetic, so a prompt is a number and nothing else.
const PROMPT: [u32; 1] = [233];

/// How many greedy decode fires follow the prefill. Long enough for the
/// routing to wander off the identity prefix the slab was opened at, which is
/// what (b) reads.
const STEPS: usize = 24;

/// **ONE SHELL AT A TIME, PER PROCESS**, for `serve_smoke`'s reason: the
/// measurements are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    use model_dsl::Classify;
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "logit {at} is {value}");
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

// ── the fixture ──────────────────────────────────────────────────────────

/// The reduced routed text, traced for this shell.
fn micro() -> (model::qwen_3::model::Model, Trace) {
    let m = model::qwen_3::model::Model::a3b_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen35-a3b-micro", &m, Platform::Metal);
    (m, trace)
}

/// **THE PAIRING, AND WHY IT IS EMPTY HERE.** `experts::Plan::of` takes the
/// load plan's attachments because a quantized bank's factors and zero points
/// are part of an expert's seat. This fixture is bf16 end to end — every
/// routed bank is one dense plane — so the honest map is the empty one, and
/// stating it directly is what lets the refusals below be asserted with no
/// checkpoint and no device in the room. A load through `api.rs` reads the
/// real map off the contract (`weights::attachments`).
fn dense_planes() -> Attachments {
    Attachments::new()
}

/// A scratch directory of this process's own.
fn scratch(what: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pie-wa-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

/// **A checkpoint for a text nobody ships**, written from the trace's own
/// params under the plan's own names — which is the door `Model::load` opens.
///
/// The bytes are a deterministic function of `(name, element index)`, so the
/// two loads this file compares read the SAME weights and a difference in
/// their logits is a difference in the residency machinery and nothing else.
/// Norm scales are ~1 and everything else is small: a random norm near zero
/// would make the logits a rectangle of noise, and the point of `finite` is
/// to notice when they are.
fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let mut planes: Vec<_> = trace
        .params
        .iter()
        // A REGISTERED PLANE IS ONE THE CHECKPOINT DOES NOT HAVE: the adapter
        // banks are reserved and zeroed by the shell, and publishing one here
        // would be a checkpoint claiming to ship a fine-tune.
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let dtype = match param.dtype {
            model_ir::Dtype::Bf16 => ztensor::DType::BF16,
            model_ir::Dtype::F16 => ztensor::DType::F16,
            model_ir::Dtype::F32 => ztensor::DType::F32,
            other => panic!(
                "`{}` is {other:?}, which this fixture does not state",
                param.name
            ),
        };
        let mut bytes = Vec::with_capacity(count * 4);
        let mut seed = fnv(&param.name);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            match param.dtype {
                model_ir::Dtype::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                model_ir::Dtype::F16 => bytes.extend_from_slice(&f16_bits(value).to_le_bytes()),
                _ => bytes.extend_from_slice(&bf16_bits(value).to_le_bytes()),
            }
        }
        writer
            .add(param.name.as_str(), param.shape.clone(), dtype, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash | 1
}

/// f32 to bf16, round-to-nearest-even — the conversion the loader does.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exponent <= 0 {
        return sign;
    }
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exponent as u16) << 10) | mantissa
}

/// The fixture, built once per test: the trace, the contract, and the
/// container both loads read.
struct Fixture {
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    #[allow(dead_code)]
    dir: PathBuf,
}

fn fixture(what: &str) -> Fixture {
    let (m, trace) = micro();
    let dir = scratch(what);
    let container = dir.join("micro.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = m.load(&source).expect("the text lands its own checkpoint");
    drop(source);
    Fixture {
        trace,
        contract,
        container,
        dir,
    }
}

/// What the whole table demands on the device, off the trace alone.
fn full_demand(trace: &Trace) -> u64 {
    Plan::of(trace, &dense_planes(), None)
        .expect("a bf16 routed text plans")
        .device_demand()
}

/// **How many bytes of this trace are routed bands** — what the budget the
/// parity test states is measured down from.
///
/// Read off a plan that had to form one: a budget one byte under full
/// residency is the smallest ask that still streams, and its bands state the
/// arity and the stride each whole band is the product of.
fn band_bytes(trace: &Trace) -> u64 {
    Plan::of(trace, &dense_planes(), Some(full_demand(trace) - 1))
        .expect("one byte under full residency streams")
        .bands()
        .iter()
        .map(|band| u64::from(band.experts) * band.stride)
        .sum()
}

/// A shell over the fixture, at a stated residency.
fn load(fixture: &Fixture, residency: Plan) -> engine_metal::Result<Shell> {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 64),
        profile: None,
        page_size: 16,
        context: 128,
        slots: 2,
        // F1: the streamed arm collapses to this anyway (the segment cuts
        // block), so the golden is fired at the same depth its comparand can
        // reach. A parity test that compared depth two against depth one
        // would be measuring two things at once.
        runahead: engine::runahead::Runahead::F1,
        residency,
    })
}

/// A prefill and `STEPS` greedy decodes, feeding the argmax back — every fire
/// one token wide, for the reason this file's header gives.
fn run(shell: &mut Shell) -> Vec<Vec<f32>> {
    shell.open(0).expect("slot 0 opens");
    let mut rows = Vec::with_capacity(STEPS + 1);
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(PROMPT.len() as u32),
            tokens: &PROMPT,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");
    let mut fed = argmax(&prefill[0]);
    rows.push(prefill[0].clone());
    for step in 0..STEPS {
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[fed],
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        finite(&decode[0], "decode");
        fed = argmax(&decode[0]);
        rows.push(decode[0].clone());
    }
    rows
}

fn ready(what: &str) -> bool {
    if engine_metal::device::present() {
        return true;
    }
    eprintln!("skipping {what}: no Metal device on this machine");
    false
}

// ── (a) and (b) ──────────────────────────────────────────────────────────

#[test]
fn a_half_resident_load_says_what_a_fully_resident_one_says() {
    let _one = serialized();
    if !ready("a_half_resident_load_says_what_a_fully_resident_one_says") {
        return;
    }
    let fixture = fixture("parity");
    let full = full_demand(&fixture.trace);

    // ── THE GOLDEN. Uncapped: the whole table on the device, no tier, one
    //    command buffer per fire — the fire this shell fired before W-a
    //    existed.
    let mut resident = load(&fixture, Plan::default()).expect("the resident shell loads");
    assert!(
        resident.weights_resident(),
        "an uncapped load holds the whole table"
    );
    assert!(
        resident.expert_residency().is_empty(),
        "and opens no tier to report on"
    );
    assert_eq!(
        resident.expert_motion(),
        (0, 0),
        "and cuts no segment and copies no seat"
    );
    let golden = run(&mut resident);
    drop(resident);

    // ── THE STREAMED LOAD. A budget sized so that the slab seats roughly
    //    half the experts: the dense planes whole, plus half the band bytes.
    let budget = full - band_bytes(&fixture.trace) / 2;
    let planned = Plan::of(&fixture.trace, &dense_planes(), Some(budget))
        .expect("half the experts stream");
    assert!(planned.streams(), "half the table cannot be held whole");
    assert_eq!(
        planned.host_demand(),
        0,
        "unified memory has no second tier to demand of"
    );
    let arity = planned.bands()[0].experts;
    let seated = planned.slots();
    assert!(
        seated > 0 && seated < arity,
        "the slab seats {seated} of {arity} experts, which is neither half nor whole"
    );
    // A one-token fire routes to at most `top_k` experts per layer, and the
    // slab must seat every one of them at once. Stated here so that a fixture
    // change that broke it fails on this line rather than inside a fire.
    assert!(
        seated >= 4,
        "a3b_micro routes at top-k 4 and the slab seats {seated}; one segment cannot \
         pin more seats than it has"
    );
    eprintln!(
        "micro: {full} bytes whole, {} planned ({seated} of {arity} experts per group, \
         {} of source bytes behind them)",
        planned.device_demand(),
        planned.source_bytes(),
    );

    let mut streamed = load(&fixture, planned).expect("the streamed shell loads");
    assert!(
        !streamed.weights_resident(),
        "a streamed load says so rather than claiming the table"
    );
    let groups = streamed.expert_residency();
    assert!(!groups.is_empty(), "and reports the tier it opened");
    let opened: Vec<Vec<Option<u32>>> = groups.iter().map(|g| g.in_seat.clone()).collect();
    for group in &groups {
        assert_eq!(
            group.in_seat.len(),
            seated as usize,
            "`{}` seats what the plan said",
            group.name
        );
        assert!(
            group.in_seat.iter().all(Option::is_some),
            "`{}` opened with an empty seat; the identity prefix is copied in at \
             `Tier::open`",
            group.name
        );
    }

    let streamed_logits = run(&mut streamed);

    // ── (a) THE PARITY. Byte for byte: the arithmetic is the same kernel over
    //    the same bytes in the same order, and only the seat each expert's
    //    weights were copied into differs. A tolerance here would be a
    //    tolerance for the machinery having moved the wrong bytes.
    assert_eq!(golden.len(), streamed_logits.len());
    for (step, (want, got)) in golden.iter().zip(&streamed_logits).enumerate() {
        assert_eq!(
            want.len(),
            got.len(),
            "step {step}: two readouts of one vocabulary"
        );
        let differ = want
            .iter()
            .zip(got)
            .enumerate()
            .find(|(_, (a, b))| a.to_bits() != b.to_bits());
        assert!(
            differ.is_none(),
            "step {step}: the streamed load's logits differ from the resident load's at \
             column {:?} — residency changed a number, which is the one thing it may \
             never do",
            differ.map(|(at, (a, b))| (at, *a, *b)),
        );
    }

    // ── (b) THE MECHANISM MOVED. Segments were cut (one per mixture layer
    //    per fire) and seats were copied, and the occupancy is no longer the
    //    identity prefix the tier opened at — which is a change to which
    //    bytes sit where, and to nothing the logits above can see.
    let (swaps, segments) = streamed.expert_motion();
    let after = streamed.expert_residency();
    let moved = after
        .iter()
        .zip(&opened)
        .any(|(group, first)| &group.in_seat != first);
    eprintln!("micro: {segments} segments cut, {swaps} bands copied, occupancy moved: {moved}");
    assert!(
        segments >= (STEPS + 1) as u64 * groups.len() as u64,
        "{segments} segment cuts over {} fires of {} mixtures — a streamed fire cuts \
         once per mixture",
        STEPS + 1,
        groups.len()
    );
    assert!(
        swaps > 0,
        "{segments} segments and not one band was copied; the tier is opened and the \
         swap is not connected"
    );
    assert!(
        moved,
        "seats were copied and no group's occupancy changed; the bookkeeping and the \
         copies are not the same seats"
    );
}

// ── (c) the refusals, host-only ──────────────────────────────────────────

#[test]
fn a_budget_under_the_dense_planes_is_refused_by_name() {
    let (_, trace) = micro();
    let why = Plan::of(&trace, &dense_planes(), Some(1 << 16))
        .expect_err("64 KiB holds no model");
    let said = why.to_string();
    assert!(
        said.contains("DENSE") && said.contains("do not stream"),
        "the refusal names the tier that cannot hold less: {said}"
    );
}

#[test]
fn a_budget_that_seats_no_whole_expert_is_refused_by_name() {
    let (_, trace) = micro();
    // The floor is the dense planes plus one seat of every band, and the plan
    // states it in its own refusal — so it is read out of one refusal and
    // handed back one byte short, which is the tightest ask that still cannot
    // be met.
    let said = Plan::of(&trace, &dense_planes(), Some(1 << 16))
        .expect_err("64 KiB holds no model")
        .to_string();
    let floor: u64 = said
        .split("Raise it to at least ")
        .nth(1)
        .and_then(|tail| tail.split(',').next())
        .and_then(|number| number.trim().parse().ok())
        .unwrap_or_else(|| panic!("the refusal states the floor it wants: {said}"));

    let why = Plan::of(&trace, &dense_planes(), Some(floor - 1))
        .expect_err("one byte under one seat per band seats no whole expert");
    let said = why.to_string();
    assert!(
        said.contains(&format!("{}", floor - 1)) && said.contains(&format!("{floor}")),
        "the refusal names both numbers: {said}"
    );

    // And the floor itself IS servable — one seat per band, which is the
    // smallest streamed load there is.
    let planned =
        Plan::of(&trace, &dense_planes(), Some(floor)).expect("the floor seats one expert");
    assert!(planned.streams());
    assert_eq!(planned.slots(), 1, "the floor is one seat per band");
}

#[test]
fn a_capped_budget_over_a_plan_with_nothing_routed_is_refused_by_name() {
    // A dense SKU: nothing in it is a routed bank, so a budget under its
    // table has no tier to hold less of. `attachments` would be this trace's
    // quantized pairing and this SKU is bf16, so the empty map is honest.
    let trace = model::trace_of("qwen35-d0.8b-bf16-kv-bf16")
        .expect("the catalog ships the dense SKU")(Platform::Metal);
    let why = Plan::of(&trace, &dense_planes(), Some(1 << 20))
        .expect_err("1 MiB holds no dense model");
    let said = why.to_string();
    assert!(
        said.contains("Nothing in it is a routed-expert bank"),
        "the refusal says there is nothing to hold less of: {said}"
    );
}

#[test]
fn an_uncapped_plan_is_the_degenerate_case() {
    let (_, trace) = micro();
    let plan = Plan::of(&trace, &dense_planes(), None).expect("a bf16 routed text plans");
    assert!(!plan.streams(), "uncapped holds everything");
    assert_eq!(plan.slots(), 0, "and seats nothing, because there is no slab");
    assert_eq!(plan.host_demand(), 0);
    assert_eq!(plan.source_bytes(), 0, "and holds no host band table");
    assert!(plan.device_demand() > 0);

    // A budget at or above the whole table is the same answer, and it is
    // answered as the degenerate case rather than as a streamed load that
    // happens to seat everything.
    let whole = plan.device_demand();
    let held = Plan::of(&trace, &dense_planes(), Some(whole)).expect("the budget covers it");
    assert!(!held.streams());
    assert_eq!(held.device_demand(), whole);
}
