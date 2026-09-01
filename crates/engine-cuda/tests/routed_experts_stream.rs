//! **Routed experts stream, and the logits do not notice** (alto design §7,
//! wave D2).
//!
//! The claim under test is design §7's one sentence about the dynamic demand
//! shape: *residency is a performance promotion, never a correctness
//! condition*. Routing is computed on device, so no host decision can precede
//! a fire and arrange for the experts it will need to be there. The engine's
//! answer is an indirection table — a device-resident `expert_id -> base
//! address` row per bank, whose entries point into a device slab when the
//! expert is resident and at PINNED HOST bytes over UVA when it is not — plus
//! per-expert usage counters the fire path notes its routing in and the host
//! reads between fires.
//!
//! ```text
//! (a) a load whose device budget holds HALF the experts fires, and its
//!     logits are the logits full residency produces
//! (b) a fire that routes to non-resident experts completes with no sync on
//!     the fire path — asserted by construction (below) and witnessed by the
//!     counters showing hits on experts the slab does not hold
//! (c) after repeated fires the promotion moves what was used on-device — the
//!     resident set changes — and the logits do not move with it
//! (d) the refusals: a budget under the dense planes, and a budget under the
//!     pinned tier
//! (e) and the refusal §M-3 added: a streamed load offered no weight cache
//!     directory does not serve, and its sentence names the field to set and
//!     the command to run
//! ```
//!
//! # (b) is a claim about a call graph, and this is where it is stated
//!
//! No test can prove the absence of a synchronize by watching a fire succeed.
//! What makes (b) true is that the fire path grew exactly TWO new operations
//! and neither of them can block:
//!
//! * **one read of `expert_table[expert]`** inside
//!   `moe_matmul_select_gemv_body` — a device load from a device address,
//!   replacing the `weight_base + expert * expert_stride` arithmetic that was
//!   there. When the table pointer is null (full residency) the arithmetic is
//!   what it always was and no load happens at all.
//! * **one `atomicAdd` per routed expert per fire**, from one thread of one
//!   block, into a device counter buffer at a fixed address.
//!
//! Everything else the tier does happens on the HOST between fires
//! (`experts::Tier::promote`, called at the top of `enqueue`) or on the NOTIFY
//! stream behind a settlement event (`experts::Tier::drain`, called in
//! `settle_step`). There is no `cudaLaunchHostFunc` on the compute stream, no
//! `cudaMemcpy` without a stream, no `cudaStreamSynchronize`, no readback the
//! next wave waits on. A miss reads pinned memory over PCIe and the kernel
//! keeps going — which is exactly what article 2 asks for and exactly what
//! `d(a)` measures the *result* of.
//!
//! # (a) boots THREE times now, and the middle one is a prepare
//!
//! §M wave M-3 made the streamed load WARM-ONLY. Under `Intent::Serve` a plan
//! that streams is served out of a prepared serving artifact — `<key>.tiers`
//! under the weight cache directory — or it is REFUSED before the pinned tier
//! is allocated; `Shell::prepare` is the only door in the process that writes
//! one. The fully-resident load is untouched, so the golden below is the load
//! it always was.
//!
//! This file used to hand its streamed boot `weight_cache_dir: None` over a
//! comment saying the cache was off for a gate and a streamed load formed no
//! key anyway. The first half was a choice; the second was a fact about §K's
//! engine, and M-3 is exactly the wave that deleted it. An unkeyed streamed
//! load is now the loudest refusal in the loader, and it is (e).
//!
//! So (a) prepares into a scratch directory and boots warm out of it, at the
//! SAME plan and the SAME `Boot` document — the artifact's key is a function
//! of the whole document, so two documents would name two files and the warm
//! boot would refuse against the one it just wrote. The cost is one extra
//! landing of a 58 MiB synthetic checkpoint, which is why this claim can be
//! made here and not beside the gpt-oss gates.
//!
//! # The fixture, and why it is not a catalog SKU
//!
//! `Model::a3b_micro` is `qwen35-a3b`'s own text at a size two loads of which
//! fit on one card: 4 layers, 32 experts, hidden 512, vocab 2048, ~58 MiB.
//! `a3b` itself is 64 GiB and this file's central claim — that a HALF-resident
//! load says what a FULLY resident one says — needs both loads on one device.
//! Its checkpoint is written here, from the trace's own params, with
//! deterministic pseudo-random bytes: what is under test is the residency
//! machinery, and a machinery that moves the wrong bytes fails against
//! arbitrary weights exactly as it fails against trained ones.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --test routed_experts_stream -- --nocapture
//! ```
//!
//! # Gating
//!
//! As `serve_smoke.rs`: skipped at run time when the machine has no device,
//! rather than `#[ignore]`d — an ignored test on the one box that could run it
//! is a test nobody runs. Nothing here needs a checkpoint on disk either: the
//! one it reads it writes itself, as it does the serving artifact §M-3 now
//! requires of a streamed boot — tens of megabytes under `TMPDIR`, removed
//! however the test leaves, which is small enough that this file states no
//! disk condition.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use checkpoint::contract::ModelContract;
use engine_cuda::experts::{Attachments, Budgets, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request};
use model_ir::{ParamSource, Trace};

/// The tokens every fire of this file feeds. Arbitrary ids inside the micro
/// text's 2048-token vocabulary — the model is synthetic, so a prompt is a
/// vector of numbers and nothing else.
const PROMPT: [u32; 6] = [11, 233, 7, 1904, 42, 900];

/// How many greedy decode fires follow the prefill. Long enough for the
/// promotion loop to have opinions: it moves at most two experts per bank per
/// gap, and there are eight banks.
const STEPS: usize = 24;

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name (`serve_smoke.rs` argues it whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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
fn micro() -> (models::qwen_3::model::Model, Trace) {
    let m = models::qwen_3::model::Model::a3b_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen35-a3b-micro", &m, Platform::Cuda);
    (m, trace)
}

/// A scratch directory of this process's own, which removes itself however
/// the test leaves.
///
/// It used to be a bare `PathBuf` and the only thing that ever collected it
/// was the NEXT run's `remove_dir_all` below — acceptable while all it held
/// was a 58 MiB fixture checkpoint. Since §M-3 it also holds the serving
/// artifact the streamed boot is prepared out of, which is one image per
/// plane of the whole table, so it is worth a `Drop`.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(what: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("pie-d2-{what}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    Scratch(dir)
}

/// **A checkpoint for a text nobody ships**, written from the trace's own
/// params under the plan's own names — which is the door `Model::load` opens.
///
/// The bytes are a deterministic function of `(name, element index)`, so the
/// two loads this file compares read the SAME weights and a difference in
/// their logits is a difference in the residency machinery and nothing else.
/// Norm scales are ~1 and everything else is small: a random norm near zero
/// would make the logits a rectangle of noise, and the point of `finite` is to
/// notice when they are.
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
            Dtype::Bf16 => ztensor::DType::BF16,
            Dtype::F16 => ztensor::DType::F16,
            Dtype::F32 => ztensor::DType::F32,
            other => panic!("`{}` is {other:?}, which this fixture does not state", param.name),
        };
        let mut bytes = Vec::with_capacity(count * 4);
        let mut seed = fnv(&param.name);
        for _ in 0..count {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            match param.dtype {
                Dtype::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                Dtype::F16 => bytes.extend_from_slice(&f16_bits(value).to_le_bytes()),
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

/// The fixture, built once per test: the trace, the contract, the container
/// both loads read, and the directory a prepare writes this deployment's
/// serving artifact into.
struct Fixture {
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
    /// **Where the streamed boot's weights come from** (§M-3). Empty until a
    /// test calls `prepare`, and a streamed `Shell::load` against it while it
    /// is empty refuses — which is (e)'s neighbour claim and what
    /// `a_prepare_writes_the_tiers_a_bare_boot_refuses` states in full.
    cache: PathBuf,
    /// Held for its `Drop` and read by nothing: the checkpoint and the
    /// artifact both live under it and neither outlives its test.
    #[allow(dead_code)]
    dir: Scratch,
}

fn fixture(what: &str) -> Fixture {
    let (m, trace) = micro();
    let dir = scratch(what);
    let container = dir.0.join("micro.zt");
    let cache = dir.0.join("tiers");
    std::fs::create_dir_all(&cache).unwrap_or_else(|why| panic!("{}: {why}", cache.display()));
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = m.load(&source).expect("the text lands its own checkpoint");
    drop(source);
    Fixture {
        trace,
        contract,
        container,
        cache,
        dir,
    }
}

/// What the whole table demands on the device, off the trace alone.
fn full_demand(trace: &Trace) -> u64 {
    Plan::of(trace, &Attachments::new(), Budgets::uncapped())
        .expect("a bf16 routed text plans")
        .device_demand()
}

/// **ONE DOCUMENT, TWO DOORS** (§M-3). The prepare and the boot it feeds have
/// to describe the same deployment in every field or they name two different
/// files: the artifact's key is a function of the trace, the recipe and the
/// ranking. So the document is stated once here and handed to both.
fn doc<'a>(fixture: &'a Fixture, residency: Plan, cache: Option<&'a Path>) -> Boot<'a> {
    Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 64),
        patches: None,
        profile: None,
        page_size: 16,
        context: 128,
        slots: 2,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // **NOT A CACHE ANY MORE — THE ROAD** (§M-3). This field read `None`
        // unconditionally, over a comment saying the warm-boot cache was off
        // for a gate and a streamed load formed no key anyway. It forms one
        // now and it has nowhere else to read: a serving load that streams
        // opens a prepared artifact or it refuses. `None` is still what the
        // uncapped golden is given — the fully-resident path never looks for
        // a file — and it is what (e) stands on.
        weight_cache_dir: cache,
        residency,
    }
}

/// A shell over the fixture, at a stated residency, reading whatever `cache`
/// says: `None` for a resident load or for a refusal, the fixture's own
/// directory for a streamed one that has been prepared into it.
fn load(fixture: &Fixture, residency: Plan, cache: Option<&Path>) -> engine_cuda::Result<Shell> {
    Shell::load(doc(fixture, residency, cache))
}

/// **THE WRITER**, and since §M-3 the only one in the process. `pie model
/// import --prepare-only` reaches it through `Cuda::prepare`; this file
/// reaches it directly, because what the boot below needs is the file and not
/// the plumbing.
fn prepare(fixture: &Fixture, residency: Plan) -> engine_cuda::Result<()> {
    Shell::prepare(doc(fixture, residency, Some(&fixture.cache)))
}

/// A prefill and `STEPS` greedy decodes, feeding the argmax back.
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
    if engine_cuda::device::present() {
        return true;
    }
    eprintln!("skipping {what}: no CUDA device on this machine");
    false
}

// ── (a) and (c) ──────────────────────────────────────────────────────────

#[test]
fn a_half_resident_load_says_what_a_fully_resident_one_says() {
    let _one = serialized();
    if !ready("a_half_resident_load_says_what_a_fully_resident_one_says") {
        return;
    }
    let fixture = fixture("parity");
    let full = full_demand(&fixture.trace);

    // ── THE GOLDEN. Uncapped: the whole table on the device, the tier never
    //    opened, the select kernels handed two nulls — the fire this shell
    //    fired before D2 existed.
    let mut resident = load(&fixture, Plan::default(), None).expect("the resident shell loads");
    assert!(
        resident.weights_resident(),
        "an uncapped load holds the whole table"
    );
    assert!(
        resident.expert_residency().is_empty(),
        "and opens no tier to report on"
    );
    let golden = run(&mut resident);
    drop(resident);

    // ── THE STREAMED LOAD. A budget sized so that the slab can hold roughly
    //    half the experts: the dense planes whole, plus half the expert bytes.
    let budget = full - expert_bytes(&fixture.trace) / 2;
    let planned = Plan::of(&fixture.trace, &Attachments::new(), Budgets::device(budget))
        .expect("half the experts stream");
    assert!(planned.streams(), "half the table cannot be held whole");
    let arity = planned.banks()[0].experts;
    let seated = planned.banks()[0].resident;
    assert!(
        seated > 0 && seated < arity,
        "the slab seats {seated} of {arity} experts, which is neither half nor whole"
    );
    eprintln!(
        "micro: {full} bytes whole, {} planned ({seated} of {arity} experts per bank, \
         {} pinned)",
        planned.device_demand(),
        planned.host_demand(),
    );

    // ── THE PREPARE, WHICH IS WHERE THE STREAMED LOAD'S BYTES NOW COME FROM
    //    (§M-3). Not a warm-boot optimization measured here — that is
    //    `a_second_streamed_boot_maps_the_tiers_it_wrote`'s subject — but the
    //    only road there is: a serving load at this plan refuses against an
    //    empty directory, which is what (e) below stands on. It costs one
    //    more landing of the 58 MiB fixture, and it lands the same table at
    //    the same plan, so the parity claim under it is unmoved.
    prepare(&fixture, planned.clone()).expect("the prepare writes this seat's artifact");
    assert_eq!(
        std::fs::read_dir(&fixture.cache)
            .expect("the cache directory")
            .flatten()
            .count(),
        1,
        "one prepare, one serving artifact, and no `.part` left beside it"
    );

    let mut streamed =
        load(&fixture, planned, Some(&fixture.cache)).expect("the prepared streamed shell loads");
    assert!(
        !streamed.weights_resident(),
        "a streamed load says so rather than claiming the table"
    );
    assert!(
        streamed.weights_from_cache(),
        "and it came off the artifact the prepare wrote — since §M-3 there is \
         no other place a streamed serve can have got them"
    );
    let banks = streamed.expert_residency();
    assert!(!banks.is_empty(), "and reports the tier it opened");
    let held: Vec<Vec<u32>> = banks.iter().map(|bank| bank.in_slot.clone()).collect();

    let streamedlogits = run(&mut streamed);

    // ── (a) THE PARITY. Byte for byte: the arithmetic is the same kernel over
    //    the same bytes in the same order, and only the base address each
    //    expert's weights were read from differs. A tolerance here would be a
    //    tolerance for the machinery having moved the wrong bytes.
    assert_eq!(golden.len(), streamedlogits.len());
    for (step, (want, got)) in golden.iter().zip(&streamedlogits).enumerate() {
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

    // ── (b)'s WITNESS. The counters say the fires routed to experts the slab
    //    did not hold: those reads went to pinned host memory over UVA, and
    //    the fire completed anyway. (The absence of a sync is a property of
    //    the call graph and is argued in this file's header.)
    let after = streamed.expert_residency();
    let mut missed = 0u64;
    for (bank, first) in after.iter().zip(&held) {
        for expert in 0..bank.experts {
            if !first.contains(&expert) {
                missed += u64::from(bank.hits[expert as usize]);
            }
        }
    }
    assert!(
        missed > 0,
        "no fire of this run routed to an expert the slab did not hold at load, so \
         nothing here exercised the pinned tier — raise STEPS or lower the budget"
    );

    // ── (c) THE PROMOTION. The hot experts moved on-device, which is a change
    //    to the table's entries and to nothing else: the logits above are the
    //    logits of a run during which the residency was moving.
    let (promoted, demoted, skipped) = streamed.expert_motion();
    eprintln!(
        "micro: {missed} routed reads to non-resident experts, {promoted} promoted, \
         {demoted} demoted, {skipped} gaps skipped"
    );
    assert!(
        promoted > 0,
        "{STEPS} fires and not one expert was promoted; the counters or the promotion \
         loop are not connected"
    );
    assert_eq!(promoted, demoted, "a promotion takes a slot from a demotion");
    let moved = after
        .iter()
        .zip(&held)
        .any(|(bank, first)| &bank.in_slot != first);
    assert!(moved, "experts were promoted and no slab's occupancy changed");
}

/// **How many bytes of this trace are routed experts** — what the budget the
/// parity test states is measured down from.
///
/// Read off a plan that had to form one: a budget one byte under full
/// residency is the smallest ask that still streams, and its banks state the
/// arity and the stride the whole bank is the product of.
fn expert_bytes(trace: &Trace) -> u64 {
    let full = full_demand(trace);
    Plan::of(trace, &Attachments::new(), Budgets::device(full - 1))
        .expect("one byte under full residency streams")
        .banks()
        .iter()
        .map(|bank| u64::from(bank.experts) * bank.stride)
        .sum()
}

// ── (d) and (e) the refusals ───────────────────────────────────────────────

#[test]
fn a_budget_under_the_planes_that_cannot_move_is_refused_by_name() {
    // **THE FLOOR MOVED AT D2b.** It used to be the DENSE planes — none of
    // them could leave the device, so a budget under them was the end of the
    // conversation. They can leave now (streaming §2's static demand shape),
    // and what is left under any budget is the planes that genuinely cannot:
    // a REGISTERED adapter bank, whose store offset `register_adapter` writes
    // at, plus one expert slot of every routed bank.
    let (_, trace) = micro();
    let why = Plan::of(&trace, &Attachments::new(), Budgets::device(1 << 16))
        .expect_err("64 KiB holds no model");
    let said = why.to_string();
    assert!(
        said.contains("REGISTERED") && said.contains("cannot be moved to another tier"),
        "the refusal names the planes that cannot hold less: {said}"
    );
}

#[test]
fn a_host_budget_under_the_pinned_tier_is_refused_by_name() {
    let (_, trace) = micro();
    let full = full_demand(&trace);
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::device(full * 3 / 4))
        .expect("three quarters streams");
    assert!(plan.streams());
    let residency = engine::load::Residency {
        device_weight_budget: Some(full * 3 / 4),
        host_weight_budget: Some(plan.host_demand() - 1),
    };
    let why = residency
        .admit(plan.device_demand(), plan.host_demand())
        .expect_err("a pinned tier one byte short does not admit");
    let said = why.to_string();
    assert!(
        said.contains("host_weight_budget") && said.contains("pinned host"),
        "the refusal names the tier and the field: {said}"
    );
}

#[test]
fn an_uncapped_budget_opens_no_tier_at_all() {
    let (_, trace) = micro();
    let plan = Plan::of(&trace, &Attachments::new(), Budgets::uncapped()).expect("uncapped plans");
    assert!(!plan.streams());
    assert_eq!(
        plan.host_demand(),
        0,
        "a fully-resident load pins nothing — dev's `place_all` allocates no host tier"
    );
}

/// **A streamed load with nowhere to have been prepared does not serve**
/// (§M-3, claim (e)).
///
/// The load refused here is the load THIS FILE USED TO PERFORM: until M-3 the
/// streamed boot above was handed `weight_cache_dir: None`, streamed the
/// checkpoint, ran the landing transforms and served — and the comment over
/// that field said a streamed load formed no key anyway. Both halves are
/// gone. It forms a key, it has exactly one place to read, and when the
/// deployment names no directory it cannot even say which file is missing —
/// so the sentence it refuses with is about the CONFIG, and it names the
/// field to set and the command to fill it.
///
/// `Fault::Residency`, which `Cuda` renders as `engine::Error::Impossible`:
/// nothing the deployment frees changes the answer. And it costs nothing —
/// the refusal is raised before the pinned tier is allocated and before a
/// byte of the checkpoint is read — which is why the claim lives beside the
/// cheap synthetic fixture rather than behind an `--ignored` gpt-oss gate.
#[test]
fn a_streamed_load_with_no_cache_directory_is_refused_by_name() {
    let _one = serialized();
    if !ready("a_streamed_load_with_no_cache_directory_is_refused_by_name") {
        return;
    }
    let fixture = fixture("unkeyed");
    let budget = full_demand(&fixture.trace) - expert_bytes(&fixture.trace) / 2;
    let planned = Plan::of(&fixture.trace, &Attachments::new(), Budgets::device(budget))
        .expect("half the experts stream");
    assert!(
        planned.streams(),
        "a resident load reads no artifact and there would be nothing here to refuse"
    );

    let why = load(&fixture, planned, None)
        .err()
        .expect("a streamed serve with no weight cache directory does not load");
    let said = format!("{why:?}");
    eprintln!("the unkeyed streamed load refused: {said}");
    assert!(
        said.contains("weight_cache_dir"),
        "the refusal names the field an operator has to state: {said}"
    );
    assert!(
        said.contains("pie model import --prepare-only"),
        "and the command that fills the directory it names: {said}"
    );
}
