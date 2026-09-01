//! **TIER 3 — THE EAGER DOCTRINE, STATED AS COUNTERS** (the tier-2 campaign's
//! third tier; `crate::rotate`'s header, alto streaming §3 item 4, D2b).
//!
//! Tier 1 is a body. Tier 2 is a body with islands in it. Tier 3 is NO body at
//! all: a load whose dense planes rotate takes the eager walk on every fire
//! for the life of it, whatever mode the deployment asked for, because a
//! rotation's backpressure is a HOST cursor the walk advances and a replayed
//! graph has no walk. `Shell::enqueue_on`'s `records` line is where that is
//! decided and `Shell::arm_bodies`'s fourth clause is where the load stops
//! paying for it.
//!
//! `a_spilled_dense_model_says_what_it_said` already gates the MECHANISM at
//! this SKU — the 2/5 budget that forces the spill, the prepare that writes
//! the serving artifact, the pump that moves the bytes, the identity that says
//! it moved the right ones — and it asks all of it under `Graphs::Off`, where
//! eagerness is the mode and there is nothing to be told. This is its sister
//! under `Graphs::On`, and what it adds is not another mechanism but the
//! DOCTRINE: that the same load, asked for graphs, quietly and completely
//! declines to record, says so once at boot, spends nothing trying, and
//! answers the recorded arm's bytes anyway.
//!
//! ```text
//! (1) A ROTATING LOAD ARMS NOTHING AND PAYS NOTHING FOR IT. Straight off
//!     `Shell::load`, under `Graphs::On` with `[engine] bodies` on:
//!     `armed_at_load == 0`, `captures == 0`, `bodies == 0`, and
//!     `eager_rotating == 0`. The first three say the arming pass did not
//!     run; the fourth says it did not run PARTWAY — every rung it climbed
//!     would have executed `record::WARM_FIRES` real walks and captured
//!     nothing, so a boot that spent them would show up here before any
//!     caller had connected.
//! (2) EVERY CALLER FIRE IS COUNTED, AND NONE OF THEM REACHED THE CACHE.
//!     `eager_rotating == FIRES` after a prefill and `STEPS` decodes, with
//!     `eager_buffered == 0` beside it — the two count REASONS and not fires
//!     (`record::BodyTally::eager_buffered`), so the second being zero is what
//!     makes the first a fire count here. And `hits`, `misses`, `reshapes`,
//!     `declines`, `refusals`, `evictions` and `sealed_declines` are all zero:
//!     the router did not leak a rotating fire into the graph cache in any
//!     direction, not as a hit, not as a warming miss, and not as a refusal.
//! (3) THE PUMP RAN UNDER REAL FIRES. `rotation()` is `Some` and its observed
//!     `copies >= STEPS` — the eager cursor is the thing the rotation rides
//!     (`Cursor::pumping`), so tier 3's walk is not a degradation the load
//!     tolerates but the mechanism the load REQUIRES.
//! (4) AND THE BYTES ARE THE RECORDED ARM'S. Token for token and logit bit for
//!     logit bit against an uncapped `Graphs::On` load that armed its bodies
//!     and replayed them. This is the claim that makes the other four worth
//!     making: the doctrine costs launches, and it costs NOTHING ELSE.
//! (5) AND THE PREPARE ROAD SERVED IT — `weights_from_cache()`, because since
//!     §M-3 a streamed serve reads a prepared artifact or is refused.
//! ```
//!
//! # Why tier 3 is a WARNING tier, and why that makes it a gate
//!
//! `record::BodyTally::eager_rotating`'s own doctrine: *an eager walk under
//! `Graphs::Off` or `Graphs::Shaped` is the mode the deployment asked for and
//! there is nothing to report; an eager walk under `Graphs::On` is a fire that
//! ran outside every graph while a graph mode was on, and that is a WARNING
//! condition — a replay that was bought and is not being delivered.* Tiers 1
//! and 2 are capabilities and their gates assert they were DELIVERED. Tier 3
//! is the honest confession that one was not, and a confession has exactly two
//! failure modes: being wrong about the arithmetic, and being silent. Claim
//! (2) is the first; claims (1) and (3) together are the second, because a
//! load that is silent about its rotor is one whose counter reads zero while
//! its fires walk.
//!
//! # Each counter is the decidable form of one sentence in the boot line
//!
//! A rotating load under a recording mode prints, once, at load:
//!
//! ```text
//! [engine] graphs is on but this load armed a dense rotor, so every fire
//! walks eagerly and nothing is recorded — a rotation's backpressure is a
//! host cursor and a replayed graph has no walk; the bodies path's load-time
//! arming is skipped for the same reason, since every rung it climbed would
//! execute its warm fires and capture nothing
//! ```
//!
//! That line is four claims, and each has a counter that decides it:
//!
//! * *"armed a dense rotor"* — `rotation()` is `Some`, and its `copies` move.
//! * *"every fire walks eagerly"* — `eager_rotating == FIRES`, not `>= 1`.
//!   The doctrine is UNIVERSAL over the load's fires, so the gate asks for the
//!   exact count and gets to fail on a router that recorded even one.
//! * *"nothing is recorded"* — `captures == 0` after the traffic, with `hits`
//!   and `misses` zero beside it. Zero captures alone would also be true of a
//!   load that thrashed misses forever.
//! * *"the arming is skipped"* — `armed_at_load == 0` at the instant of load,
//!   AND `eager_rotating == 0` at that same instant. The second is the one
//!   that bites: `Shell::arm_bodies`'s clause is worth having only because the
//!   rungs it does not climb are walks the load does not execute, and the
//!   count of walks is the only place that shows.
//!
//! The gate asserts the STATE the line stands for and never the line's text,
//! which is `tests/bodies_gate.rs`'s standing rule for boot-line claims in
//! this suite — *"a test cannot read stderr without capturing the process, and
//! the number it wants is not the print but what the print stands for"*. Run
//! it with `--nocapture` and the sentence is there to read beside the numbers.
//!
//! # `refusals == 0` is a claim about SHORT-CIRCUIT ORDER, not about the shape
//!
//! Worth stating because it is the one assertion here that could be read as
//! saying something it is not. `Prepared::bodied`'s conjunction puts
//! `!self.weights.rotating()` AHEAD of `Shell::cuttable`, so on a rotating
//! load the admissibility question is never asked at all and `body_refuse` is
//! never reached. A nonzero `refusals` on this load would therefore not mean
//! "this composition is inadmissible" — it would mean the rotor clause moved
//! behind the cut, and the load is paying to classify templates it will never
//! capture. That is the failure this zero is watching for.
//!
//! # The golden is the RECORDED arm, deliberately
//!
//! `a_spilled_dense_model_says_what_it_said` compares its spilled load against
//! an uncapped `Graphs::Off` one, which is the right golden for a claim about
//! the TIER: two eager walks, one reading a spilled table. This gate wants the
//! other diagonal. Its golden is uncapped and `Graphs::On`, so it arms its
//! bodies and replays them — the gate asserts that it did (`armed_at_load >= 1`,
//! `hits >= 1`) — and claim (4) then crosses the doctrine boundary rather than
//! running alongside it: the eager walk of a rotating load and the graph replay
//! of a resident one produce the same bits.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_rotating_load_walks_eagerly_and_says_so -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";
const PROMPT: &str = "The capital of France is";
const STEPS: usize = 12;

/// **HOW MANY FIRES A RUN IS**, which is the number claim (2) is about: the
/// prefill and every decode after it. Spelled rather than open-coded because
/// the doctrine is a statement about ALL of them and an off-by-one would read
/// as a router that recorded exactly one fire.
const FIRES: u64 = STEPS as u64 + 1;

/// **The budget, as a fraction of the table** — the same two fifths
/// `a_spilled_dense_model_says_what_it_said` arms its pump with, for its
/// reason: low enough that most of the plan's layers leave the device, high
/// enough that the embedding stays.
const CAP: u64 = 2;
const OF: u64 = 5;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// A temporary directory that removes itself, however the test leaves —
/// including the serving artifact the prepare puts in it, which at this SKU
/// is under two gibibytes.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(what: &str) -> Scratch {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    let dir = std::env::temp_dir().join(format!("pie-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temporary directory");
    Scratch(dir)
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

struct Rig {
    trace: model_ir::Trace,
    contract: checkpoint::contract::ModelContract,
    checkpoint: PathBuf,
    tokenizer: tokenizer::Tokenizer,
}

fn rig(what: &str) -> Option<Rig> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(one) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index_all(&[one]).expect("the checkpoint opens");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    Some(Rig {
        trace,
        contract,
        checkpoint,
        tokenizer,
    })
}

/// **ONE DOCUMENT, TWO DOORS** (§M-3). The prepare and the boot that reads
/// what it wrote have to state the same deployment in every field or they name
/// two different files, so the document is written once and handed to both.
///
/// **AND EVERY WORD THIS GATE DIFFS IS SPELLED HERE RATHER THAN INHERITED.**
/// `graphs` is a parameter because the two loads state the same mode and the
/// gate would be worthless if one of them quietly did not; `bodies` is written
/// out although `Knobs::default()` already says `true`, because since the
/// tier-2 campaign it is the DEFAULT that carries the meaning and a reader of
/// this file should not have to go and check which way it fell.
fn doc<'a>(rig: &'a Rig, graphs: Graphs, residency: Plan, cache: Option<&'a Path>) -> Boot<'a> {
    Boot {
        trace: rig.trace.clone(),
        contract: &rig.contract,
        checkpoint: &rig.checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs,
        knobs: engine_cuda::Knobs {
            bodies: true,
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // A fresh directory per run for the rotating load — sharing one
        // between runs would be asserting about the last one — and `None` for
        // the uncapped golden, which looks for no file and is refused for
        // nothing.
        weight_cache_dir: cache,
        residency,
    }
}

fn load(rig: &Rig, graphs: Graphs, residency: Plan, cache: Option<&Path>) -> engine_cuda::Result<Shell> {
    Shell::load(doc(rig, graphs, residency, cache))
}

/// **THE WRITER**, and since §M-3 the only one in the process — `pie model
/// import --prepare-only` reaches the same call through `Cuda::prepare`.
///
/// It BINDS THE DEVICE (`Shell::prepare` bakes before it lands), which is why
/// it stands inside this test's serialization and not before it.
fn prepare(rig: &Rig, graphs: Graphs, residency: Plan, cache: &Path) -> engine_cuda::Result<()> {
    Shell::prepare(doc(rig, graphs, residency, Some(cache)))
}

/// A prefill and `STEPS` greedy decodes — [`FIRES`] fires, which is the
/// number the doctrine is asserted against. Answers the tokens and the logit
/// rows; nothing here is timed, because tier 3's cost is
/// `a_spilled_dense_model_says_what_it_said`'s measurement and this file's
/// subject is what the load SAYS about itself.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    shell.open(0).expect("slot 0 opens");
    let mut chosen = Vec::with_capacity(STEPS + 1);
    let mut rows = Vec::with_capacity(STEPS + 1);
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");
    let mut fed = argmax(&prefill[0]);
    chosen.push(fed);
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
        chosen.push(fed);
        rows.push(decode[0].clone());
    }
    (chosen, rows)
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

// ─────────────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "real-hardware: needs a CUDA device, a local Qwen3.5-0.8B snapshot \
            and room under TMPDIR for one serving artifact; run it with \
            `-- --ignored`"]
fn a_rotating_load_walks_eagerly_and_says_so() {
    let _one = serialized();
    let Some(rig) = rig("tier 3's eager doctrine") else {
        return;
    };
    let prompt = rig.tokenizer.encode(PROMPT);

    let full = Plan::of(&rig.trace, &Default::default(), Budgets::uncapped())
        .expect("a dense plan plans")
        .device_demand();
    let budget = full * CAP / OF;
    eprintln!("qwen35-d0.8b: {full} bytes of table; the budget is {budget}");

    // ── THE PLAN THAT ARMS THE ROTOR. Two fifths of a dense table, which is
    //    `a_spilled_dense_model_says_what_it_said`'s mix and is asserted here
    //    only far enough to know the gate is about a ROTATING load: a plan
    //    that held its planes would arm no pump, take the ordinary body path,
    //    and pass every claim below vacuously.
    let plan = Plan::of(&rig.trace, &Default::default(), Budgets::device(budget))
        .expect("a dense plan under a budget spills rather than refusing");
    assert!(plan.streams(), "two fifths of the table cannot be held whole");
    assert!(plan.banks().is_empty(), "nothing here is a routed bank");
    assert!(!plan.groups().is_empty(), "and dense planes are what left");

    // ── THE GOLDEN, AND IT IS THE RECORDED ARM. Uncapped, `Graphs::On`,
    //    bodies armed at load and replayed from the first fire — which the
    //    counters below are made to say out loud, because a golden that had
    //    quietly walked would make claim (4) a comparison of two eager passes
    //    and this gate's whole subject is the boundary between them.
    let mut resident =
        load(&rig, Graphs::On, Plan::default(), None).expect("the uncapped shell loads");
    assert!(resident.weights_resident());
    assert!(
        resident.rotation().is_none(),
        "a load that holds its whole table arms no pump — this golden rotates, \
         so there is no doctrine boundary left for claim (4) to cross"
    );
    let armed = resident.body_stats();
    eprintln!("golden at load:  {armed}");
    assert!(
        armed.tally.armed_at_load >= 1,
        "the uncapped load armed nothing, so the arm this gate diffs against \
         is not the recorded one: {armed}"
    );
    let (golden, golden_rows) = run(&mut resident, &prompt);
    let served = resident.body_stats();
    eprintln!("golden served:   {served}");
    assert!(
        served.tally.hits >= 1,
        "the uncapped load replayed nothing over {FIRES} fires: {served}"
    );
    assert_eq!(
        served.tally.eager_rotating, 0,
        "a load with no rotor counted a rotating walk: {served}"
    );
    let says = rig.tokenizer.decode(&golden, false);
    eprintln!("golden answers:  {says:?}");
    drop(resident);

    // ── THE PREPARE. A spilled plan streams, and since §M-3 a streamed serve
    //    has one road to its weights: this file, written here, or a refusal
    //    naming `pie model import --prepare-only`.
    let cache = scratch("rotating-doctrine");
    prepare(&rig, Graphs::On, plan.clone(), &cache.0)
        .expect("the prepare writes this seat's artifact");

    // ── THE ROTATING LOAD, UNDER A MODE THAT RECORDS. The boot line quoted in
    //    this file's header is printed HERE, once, and everything below is
    //    that line made decidable.
    let mut rotating =
        load(&rig, Graphs::On, plan, Some(&cache.0)).expect("a spilled dense model serves");

    // ── (5) THE PREPARE ROAD SERVED IT.
    assert!(
        !rotating.weights_resident(),
        "a spilled load says so rather than claiming the table"
    );
    assert!(
        rotating.weights_from_cache(),
        "and it came off the prepared artifact — the host-side transform \
         pipeline runs in a prepare now, never in a serve"
    );

    // ── (1) THE ARMING PASS DID NOT RUN, AND DID NOT RUN PARTWAY.
    let booted = rotating.body_stats();
    eprintln!("rotating at load: {booted}");
    assert_eq!(
        booted.tally.armed_at_load, 0,
        "`Shell::arm_bodies` climbed a rung on a rotating load, whose every \
         fire the router refuses to record: {booted}"
    );
    assert_eq!(
        booted.tally.captures, 0,
        "a rotating load captured a body nothing can replay: {booted}"
    );
    assert_eq!(
        booted.census.bodies, 0,
        "a rotating load's map holds {} bodies: {booted}",
        booted.census.bodies
    );
    assert_eq!(
        booted.tally.eager_rotating, 0,
        "the boot spent {} eager walks before a caller connected — which is \
         exactly the cost `Shell::arm_bodies`'s rotor clause exists to refuse: \
         {booted}",
        booted.tally.eager_rotating
    );

    // ── THE TRAFFIC. [`FIRES`] real caller fires, none of them the load's own.
    let (tokens, rows) = run(&mut rotating, &prompt);
    let also = rig.tokenizer.decode(&tokens, false);
    let walked = rotating.body_stats();
    eprintln!("rotating served:  {walked}");
    eprintln!("rotating answers: {also:?}");

    // ── (2) EVERY FIRE WALKED, AND EVERY ONE OF THEM WAS COUNTED.
    assert_eq!(
        walked.tally.eager_rotating, FIRES,
        "the doctrine is universal over this load's fires: {FIRES} were fired \
         and {} were counted as rotating walks: {walked}",
        walked.tally.eager_rotating
    );
    assert_eq!(
        walked.tally.eager_buffered, 0,
        "this SKU moves no buffered RS bytes, and the two counters count \
         REASONS rather than fires — a nonzero one here would mean the count \
         above is not a fire count: {walked}"
    );

    // ── (2, second half) AND THE ROUTER LEAKED NOTHING INTO THE CACHE.
    assert_eq!(
        walked.tally.hits, 0,
        "a rotating fire REPLAYED a body, which is a graph baking one fire's \
         ring state into an exec that outlives it: {walked}"
    );
    assert_eq!(
        walked.tally.misses, 0,
        "a rotating fire warmed toward a capture it can never reach: {walked}"
    );
    assert_eq!(
        walked.tally.captures, 0, "a rotating load recorded a graph: {walked}"
    );
    assert_eq!(walked.tally.reshapes, 0, "a load with no body demoted one: {walked}");
    assert_eq!(
        walked.tally.declines, 0,
        "a fire the router never handed the cache was declined by it: {walked}"
    );
    assert_eq!(
        walked.tally.refusals, 0,
        "`Prepared::bodied` asks `!weights.rotating()` BEFORE `Shell::cuttable`, \
         so a rotating load never reaches `body_refuse` at all — a moving \
         counter here says the rotor clause has fallen behind the cut and the \
         load is classifying templates it will never capture: {walked}"
    );
    assert_eq!(
        walked.tally.evictions, 0,
        "a map that holds nothing evicted something: {walked}"
    );
    assert_eq!(
        walked.tally.sealed_declines, 0,
        "the map was never sealed, because there was never an arming pass to \
         seal it: {walked}"
    );

    // ── (3) THE PUMP RAN, AND IT RAN UNDER THESE FIRES. The eager cursor is
    //    what the rotation rides (`Cursor::pumping`), so this is not a
    //    measurement beside the doctrine — it is why the doctrine holds.
    let pumped = rotating.rotation();
    match pumped {
        Some((observed, slots, arena, rotates)) => eprintln!(
            "the pump: {slots} slots over {arena} bytes of arena rotate {rotates} \
             bytes a step; {observed:?}",
        ),
        None => eprintln!("the pump: nothing armed"),
    }
    let (observed, _, _, rotates) =
        pumped.expect("a spilled dense load arms the rotating pump — tier 3 IS that pump");
    assert!(rotates > 0, "the rotation moves no bytes at all");
    assert!(
        observed.copies >= (STEPS as u64),
        "the pump issued {} copies over {FIRES} fires, so the walks above did \
         not move the weights they walked",
        observed.copies,
    );

    assert!(
        tokens.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the rotating load answered {tokens:?}, which is one token repeated"
    );

    // ── (4) AND THE BYTES ARE THE RECORDED ARM'S. Tier 3 costs launches and
    //    nothing else.
    assert_eq!(
        tokens, golden,
        "the rotating load chose {tokens:?} and the recorded uncapped one \
         chose {golden:?}"
    );
    for (step, (a, b)) in golden_rows.iter().zip(&rows).enumerate() {
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}, logit {at}: replayed {x}, walked {y} — the eager \
                 doctrine moved a number"
            );
        }
    }
}
