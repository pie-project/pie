//! The bodies path's gate (`[engine] bodies`, the bodies design's chunks B
//! and C):
//! what an eager walk says, a fire served from a recorded BODY must say —
//! token for token, over a real checkpoint, with the path's own counters
//! saying it actually captured a body and actually replayed one.
//!
//! **AND SINCE THE KEYED PATH WAS DELETED THIS IS THE SHIPPING GRAPH GATE.**
//! There used to be two recording paths under `Graphs::On` and this file
//! gated the newer one against the older; the tier-2 campaign's first wave
//! removed the exact-shape runtime cache outright, so the router is a binary
//! now — a fire is a BODY's or it is an EAGER WALK — and `Graphs::On` means
//! tiered. Every claim below that used to read "against the keyed arm" reads
//! against the eager walk instead, which was always the stronger oracle
//! (`a_fire_served_from_a_body_...` argues why), and the counters that used
//! to live beside `BodyStats` are gone with the path that moved them.
//!
//! # What a body is, and what the diff is therefore checking
//!
//! A body is keyed on the COMPOSITION: the lattice point, and which classes
//! have rows. What is NOT in that key is the per-class row counts — the
//! numbers a fire actually brings — and what puts them back is not a host
//! write into the exec but a device READ: every guarded entry takes its live
//! row count off the staged live-rows seat (`kernels_cuda::Ctx::arm_stage`),
//! which the shell fills from `Windows::live` on exactly the fires it routes
//! here.
//!
//! So the construction argues identity — a body captured for this composition
//! holds exactly the launches the eager walk of this fire would issue, at the
//! key's ceilings rather than at the fire's live span — and what construction
//! cannot check is the seat: that the words the shell staged are the geometry
//! the launches then ran at, and that a guard reading them admits neither a
//! row too few nor a row too many. That is visible only in the numbers, so
//! the numbers are diffed where they leave the machine.
//!
//! The claims:
//!
//! 1. **identity** — one prefill and sixteen greedy decodes, eager then
//!    bodies-on, ONE load, byte-identical tokens.
//! 2. **it was a body** — `BodyTally::captures >= 1` and
//!    `BodyTally::hits >= 1`, so the diff is not eager against eager. A
//!    steady decode stream presents one composition, so the second decode
//!    fire captures it (`record::WARM_FIRES`) and every fire after replays
//!    it. **`captures` ALONE IS NOT EVIDENCE ON AN ARMED LOAD** — the boot's
//!    arming pass moves it before any caller fires — so every claim that
//!    wants to say SERVING happened says `hits`, or says it with a diff.
//! 3. **the load armed it** (the bodies design's chunk C, then the tier-1
//!    key-collapse wave) — a shell that states `[engine] bodies` at BOOT
//!    captures every composition its lattice can realize — decode-only,
//!    prefill-only and mixed — before it serves anything, and then SEALS the
//!    map. So the first fire of each of those shapes REPLAYS rather than
//!    capturing, and no fire past the load records a graph at all.
//!    `BodyTally::armed_at_load` says how many keys made it, `captures` must
//!    not move afterwards, and `sealed_declines` is where a shape the boot
//!    did not arm shows up.
//! 4. **the limit is stated, not hidden** — `BodyTally::refusals` is printed,
//!    and since the tier-2 campaign it counts something narrower than it
//!    used to. A windowed region whose ops do not all read the staged seat's
//!    START no longer refuses its composition: it is an `Admit::Island`, the
//!    body is captured in SEGMENTS around it, and the island is re-issued
//!    eagerly between the execs (`LastCapture::islands` is where that shows,
//!    and `an_island_body_replays_at_another_split.rs` is its gate). What
//!    `refusals` counts now is a key no cut can rescue: an artifact with two
//!    row axes, or a composition whose islands, GROWN to their nearest legal
//!    boundaries (`record::widen` — over a fork group, over a conditional's
//!    arms, over a schedule's readers), left no captured stretch at all
//!    (`record::Uncut::Eager`). A mixed fire of this catalog was
//!    once refused because FA2 took no seat to read; chunk 2c-b promoted the
//!    five names, the chunked-arm wave promoted the mixer's, and this script
//!    presents no island at all — so a moving counter here is a sentence
//!    about the artifact rather than about this wave's reach.
//!    `a_mixed_fire_says_at_one_split_...` below is where that is met head-on.
//! 5. **the three modes still say one thing** (inherited from the deleted
//!    `graph_replay.rs`, whose two survivable claims landed here) — `Off`,
//!    `Shaped` and `On` over one script give one continuation, so a
//!    difference has an author: a break between `Off` and `Shaped` is
//!    flashinfer's padded split and a break between `Shaped` and `On` is the
//!    body. And compositions that ALTERNATE inside one bucket — decode-only
//!    against decode-beside-prefill — are token-identical to their own eager
//!    walk, which is the claim that two resident bodies do not disturb each
//!    other.
//!
//! # Gating
//!
//! `serve_smoke.rs`'s, whole: skipped at run time when the machine, the
//! checkpoint or the tokenizer is missing, and `#[ignore]`d so a plain
//! workspace sweep does not wait for a model to load. The one box that runs
//! it is CI's self-hosted `pie-worker (engine-cuda)` job, with `-- --ignored`.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine-cuda --features cuda-13 --test bodies_gate -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::record::BodyStats;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this suite serves, as `serve_smoke` serves it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt.
const PROMPT: &str = "The capital of France is";

/// How many decode fires follow the prefill — `serve_smoke`'s number, for
/// `serve_smoke`'s reasons: past the warm fires it is a steady state, and it
/// crosses a page boundary under an exec captured before the crossing.
const STEPS: usize = 16;

/// One shell at a time, per process (`serve_smoke.rs` states the argument).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The lane word the model's own `Classify` computes.
fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// **THE ADAPTER THE HYBRID DECODE GATE ROUTES TO.** Slot zero, because the
/// arming pass's own synthetic for an adapted class binds slot zero too
/// (`Shell::synthetic_lanes`: `self.corrected.contains(class).then_some(0)`) —
/// a fire that named a different row would be asking a body captured over one
/// bank's arithmetic to serve another's.
const ADAPTER: u32 = 0;

/// The same word with `Facts::has_adapter` set — the second decode class of
/// this bake, and the one that makes a decode fire present two decode words.
fn adapted_word(query_len: u32, adapted: bool) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false).adapted(adapted)).word()
}

/// A lane whose word is derived from the same fact the seat states — the
/// standing rule and not a convenience: the word decides the CLASS and the
/// seat carries the PAYLOAD, and a shell that found them disagreeing refuses
/// the fire by name (`Fault::AdapterWord`).
fn routed_seat<'a>(slot: u32, tokens: &'a [u32], adapter: Option<u32>) -> Seated<'a> {
    let lane = Lane {
        slot,
        word: adapted_word(tokens.len() as u32, adapter.is_some()),
        tokens,
    };
    Seated {
        adapter,
        ..Seated::of(lane)
    }
}

/// **ONE ADAPTER, WRITTEN INTO EVERY BANK THE PLAN DECLARES**, with values
/// large enough that the correction moves the logits it touches — a bank of
/// zeros would register fine and prove nothing about the class it unlocks.
///
/// A copy of `an_island_body_replays_at_another_split`'s helper, which is the
/// other file that has to reach an adapted class; the shell's own
/// `register_adapter` is the only door either of them has and neither can lend
/// a bank to the other across a test binary.
fn register_loud(shell: &mut Shell, id: u32) {
    let built: Vec<(String, Vec<u8>)> = shell
        .banks()
        .iter()
        .map(|&(name, _, slot)| {
            let count = usize::try_from(slot).expect("a slot fits this host") / 2;
            let mut bytes = Vec::with_capacity(count * 2);
            for at in 0..count {
                let value = if name.ends_with(".lora_a") {
                    0.20 - ((at % 5) as f32) * 0.07
                } else {
                    0.15 + ((at % 3) as f32) * 0.05
                };
                bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
            }
            (name.to_string(), bytes)
        })
        .collect();
    assert!(
        !built.is_empty(),
        "this plan declares no adapter bank, so the second decode class is \
         unreachable and the two-word fire this file makes cannot be composed"
    );
    let planes: Vec<engine_cuda::AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| engine_cuda::AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .register_adapter(id, &planes)
        .unwrap_or_else(|why| panic!("registering adapter {id}: {why}"));
}

/// f32 to bf16, round-to-nearest-even — the loader's conversion, restated so
/// the adapter registered is the one described.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// A loaded shell, or `None` and a sentence saying what was missing.
///
/// `bodies` is the word the BOOT states, which is not the same question as
/// the word a test flips afterwards: `[engine] bodies` at load is what makes
/// `Shell::arm_bodies` run, and a shell that states it here has its decode
/// bodies captured before this function returns. The two A/B tests below want
/// `false` — they arm nothing at load and turn the path on between arms, so
/// that what they diff is one load's two arms — and the arming gate wants
/// `true`.
fn ready(what: &str, bodies: bool) -> Option<(Shell, tokenizer::Tokenizer)> {
    ready_with(what, Graphs::On, bodies, 4, 4, 0)
}

/// The same load with the deployment dials exposed — the growth gate below
/// needs more seats than the siblings so the load can arm MORE THAN ONE
/// rung, which is what makes slab growth happen between two captures.
///
/// **AND `adapters` IS A DIAL RATHER THAN A CONSTANT BECAUSE ONE CLAIM NEEDS
/// A SECOND DECODE WORD** (the hybrid decode wave). This bake's decode
/// classes are `1` (plain) and `3` (plain + `Facts::has_adapter`), so the
/// composition `a_two_word_decode_fire_replays_from_the_load` fires cannot be
/// staged at `max_adapters: 0` — a lane may not claim the adapter bit without
/// a bank to route to (`Fault::AdapterWord`). It is a BAKE-time capacity
/// check and feeds no other pass: it moves no window, no class, no order and
/// no bucket, so every sibling below keeps its zero and keeps the load it
/// always had.
fn ready_with(
    what: &str,
    graphs: Graphs,
    bodies: bool,
    slots: u32,
    max_lanes: u32,
    adapters: u32,
) -> Option<(Shell, tokenizer::Tokenizer)> {
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
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let trace = models::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
            .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let mut shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget {
            max_adapters: adapters,
            ..Budget::new(max_lanes, 256)
        },
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots,
        ordinal: 0,
        // The load states `On` once; each arm below states its own mode, so a
        // test that forgot would still be diffing two graph paths.
        graphs,
        knobs: engine_cuda::Knobs {
            bodies,
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // The warm-boot weight artifact cache is off for a gate: a test that
        // shared one would be asserting about the last run.
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    // The word is stated here rather than inherited: `Knobs::default()` above
    // is this `Boot`'s own word — and its `bodies` DEFAULTS TO TRUE since the
    // keyed path died, so a test that wanted the off arm and said nothing
    // would silently be testing the on arm. Every arm a test diffs is spelled
    // by the test.
    shell.set_bodies(bodies);
    Some((shell, tokenizer))
}

/// One prefill and `STEPS` greedy decodes in slot 0, in whatever mode and
/// bodies arm the shell is in. Returns the tokens and per-decode
/// milliseconds.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    let mut produced = vec![argmax(&prefill[0])];
    let mut millis = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let fed = [*produced.last().expect("a step feeds the last token back")];
        let at = Instant::now();
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        produced.push(argmax(&decode[0]));
    }
    (produced, millis)
}

/// The mean of the warm half — the steady state, past warming and capture.
fn warm(millis: &[f64]) -> f64 {
    let warm = &millis[millis.len() / 2..];
    warm.iter().sum::<f64>() / warm.len() as f64
}

/// Claims 1, 2 and 4: identity against the eager walk, with the counters
/// saying a body was captured and replayed, and the wave's limit printed.
///
/// **THE EAGER ARM IS THE ONLY ORACLE, AND IT WAS ALWAYS THE RIGHT ONE.**
/// This gate used to have a second arm available — an exact-shape keyed
/// capture of the same fire — and declined it on purpose: a keyed graph and a
/// body of one single-class fire hold the same launches at the same extents,
/// so the diff would have been a graph against very nearly itself. The eager
/// walk shares nothing with either: no capture, no baked argument, no staged
/// seat read by a guard. If the seat's words were wrong, or a guard retired a
/// row it should have run, the eager arm is what would notice. The keyed arm
/// is gone now, and this file loses nothing by it.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_fire_served_from_a_body_says_token_for_token_what_an_eager_fire_says() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the bodies A/B", false) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ONE LOAD, TWO ARMS — two loads would be two residencies and two tuner
    // histories, and a difference could be either (`set_mode`'s argument,
    // inherited whole).
    shell.set_mode(Graphs::Off);
    shell.set_bodies(false);
    let (eager, eager_ms) = run(&mut shell, &prompt);

    shell.set_mode(Graphs::On);
    shell.set_bodies(true);
    let (bodied, bodied_ms) = run(&mut shell, &prompt);

    let bodies = shell.body_stats();
    eprintln!(
        "decode ms/fire (warm half of {STEPS}): eager {:.3}  bodies {:.3}",
        warm(&eager_ms),
        warm(&bodied_ms),
    );
    // `BodyStats` carries the eager tallies too since the router went binary
    // (`eager_rotating`, `eager_buffered`): "how many fires ran outside every
    // graph" is one question with one answer, and it is printed here beside
    // the hits so the two halves of the second arm are read together.
    eprintln!("{bodies}");
    eprintln!(
        "continuations: eager {:?} / bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );

    assert!(
        bodies.tally.captures >= 1,
        "no body was ever captured, so this test compared the eager walk \
         against itself; refusals says whether the admissibility rule turned \
         every fire away: {bodies}"
    );
    assert!(
        bodies.tally.hits >= 1,
        "a body was captured and never replayed, so nothing was served FROM \
         one and the seat went unread: {bodies}"
    );
    assert_eq!(
        eager, bodied,
        "the recorded body disagreed with the eager walk it stands for: \
         eager {:?} against bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );
}

/// **AND THE OFF ARM IS THE EAGER WALK NOW** — which is a stronger sentence
/// than the one it replaces, and a differently-shaped one.
///
/// It used to read: a shell that never states the word presents exactly the
/// KEYED path, so the proof was two-sided — zero bodies counters, and a keyed
/// `captures` that moved to show the fires had gone somewhere. There is no
/// second recording path to go to any more. `Graphs::On` with `bodies` off is
/// the eager walk, byte for byte, and the claim is therefore: the tokens are
/// the ones `Graphs::Off` produces, no body is captured, no body is replayed,
/// no composition is refused — and, invisibly but as importantly, no live-rows
/// words are staged and no context is armed with the seat.
///
/// The counters are what a test can see; they are zero, and a moving one would
/// mean the router had reached a fire nobody sent it. The identity beside them
/// is what says the fires happened at all, which is the job the keyed
/// `captures` used to do.
///
/// **AND THE WORD HAS TO BE SAID OUT LOUD TO BE OFF.** `Knobs::bodies`
/// defaults to TRUE since the keyed path died — bodies are the shipping graph
/// path, not an opt-in beside one — so this test states `bodies: false` at
/// load and re-states it on the shell. A version of it that relied on the
/// default would be quietly testing the on arm.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_shell_that_states_the_word_off_walks_eagerly_and_captures_no_body() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the bodies off arm", false) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // The golden: no graph path exists at all under `Off`.
    shell.set_mode(Graphs::Off);
    shell.set_bodies(false);
    let (golden, _) = run(&mut shell, &prompt);

    // And the subject: the tiered mode with the word off, which routes every
    // fire down the same eager walk `Off` just took.
    shell.set_mode(Graphs::On);
    shell.set_bodies(false);
    let (walked, _) = run(&mut shell, &prompt);

    let bodies = shell.body_stats();
    eprintln!("{bodies}");
    assert!(
        !shell.bodying(),
        "this load stated `bodies: false` and the shell came up bodying"
    );
    assert_eq!(
        bodies,
        engine_cuda::record::BodyStats::default(),
        "a shell nobody armed moved a bodies counter: {bodies}"
    );
    assert_eq!(
        golden,
        walked,
        "`Graphs::On` with the word off is supposed to BE the eager walk, and \
         it said {:?} where `Graphs::Off` said {:?}",
        tokenizer.decode(&walked, false),
        tokenizer.decode(&golden, false),
    );
}

/// **CLAIM 3: THE LOAD ARMED THE DECODE BODIES, SO THE FIRST DECODE REPLAYS.**
///
/// The bodies design's chunk C. A shell that states `[engine] bodies` at boot
/// runs `Shell::arm_bodies` at the tail of its load: for every lattice rung
/// the deployment can seat, a synthetic decode composition — `bucket` lanes of
/// one token each, in the class the template's `attention.decode` arm runs in
/// — is fired `record::WARM_FIRES` times through the ordinary bodied path, so
/// the ordinary warm ladder walks it eagerly twice and captures off the second
/// walk. (Since the tier-1 key collapse the same pass also enumerates prefill
/// and mixed keys and then seals the map; this test still asserts the DECODE
/// half of it, and `an_enumerated_load_arms_prefill_and_mixed_keys_...`
/// asserts the rest.)
///
/// **WHAT THIS ASSERTS THAT THE A/B ABOVE CANNOT.** The A/B turns the path on
/// between two arms of one load, so its bodies are captured by TRAFFIC and its
/// first decode fires are misses by construction. Here nothing has fired at
/// all when the counters are first read, and the interesting number is the one
/// that must NOT move afterwards: a decode stream whose bodies were armed at
/// load pays no capture, so its misses stay where the load left them and every
/// one of its fires is a hit.
///
/// A moving `reshapes` is what would break the second half — a body captured
/// against a synthetic geometry whose plan payloads hash differently from a
/// real fire's would be demoted on arrival, and the counter is named in the
/// failure so the finding is legible rather than a bare inequality.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_load_that_states_the_word_arms_its_decode_bodies_before_it_serves_anything() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the bodies arming gate", true) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ── Before a single caller's fire. Everything below this line is the
    //    LOAD's own doing.
    let armed = shell.body_stats();
    eprintln!("after load, before any fire: {armed}");
    assert!(
        shell.bodying(),
        "the boot stated `[engine] bodies` and the shell came up without it"
    );
    assert!(
        armed.tally.armed_at_load >= 1,
        "the load armed no decode body at all, so nothing below is testing \
         load-time arming; the boot line says how many rungs were wanted and \
         `refusals`/`declines` say what turned them away: {armed}"
    );
    assert_eq!(
        armed.tally.captures, armed.tally.armed_at_load,
        "an armed body is a captured one and nothing but the arming had run: \
         {armed}"
    );
    assert_eq!(
        armed.tally.hits, 0,
        "the arming pass replayed something, which means a rung was fired \
         more times than the warm ladder asks for: {armed}"
    );

    // ── The prefill. It used to be the composition nobody armed — this pass
    //    was decode-only — and it was fired here, outside the measured
    //    window, because it was expected to MISS. The tier-1 key collapse
    //    enumerates prefill keys too, so it is an armed key now and replays;
    //    it stays outside the window anyway, because what this test measures
    //    is the DECODE stream behind it and
    //    `an_enumerated_load_arms_prefill_and_mixed_keys_...` is where the
    //    prefill's own first fire is asserted.
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt[..],
        }])
        .expect("the prefill fires");
    let mut produced = vec![argmax(&prefill[0])];

    // ── THE FIRST DECODE FIRE, ALONE, AND IT REPLAYS. The claim this test
    //    could not make until the decode ceiling landed
    //    (`engine_cuda::run::Run::planning`, the plan-at-bucket-ceiling
    //    design's chunk 3), and the reason it can make it now.
    //
    //    `Run::schedule_shape` hashes the plan builders' numbers, and the one
    //    that used to follow the batch was the decode schedule's carved lane
    //    count: the arming pass synthesized a rung's worth of lanes, this
    //    fire brings one, and two different lane counts hashed to two
    //    different schedules — so the first real fire of an armed key was a
    //    RESHAPE (one demotion, one re-capture, and the eager pass that rides
    //    it) unless the synthesis had guessed the batch size exactly. What
    //    chunk 3 does is stop the decode plan from being carved at the batch
    //    at all: under a body it is carved at the BUCKET's lane ceiling,
    //    which is a field of `record::BodyKey` and therefore the same number
    //    for the arming fire and for every fire that keys to it. The lanes
    //    between the batch and that ceiling are staged as genuinely empty
    //    ones and their work items are retired by `block_valid_mask`
    //    (`kernels_cuda::attn::sched_decode::schedule` carries the argument),
    //    so the widening costs correctness nothing.
    //
    //    So the claim is now the strong one, and `reshapes` is asserted as
    //    part of it: the first fire REPLAYS. A reshape here would say the
    //    hashed image still moves between the synthesis and the traffic, and
    //    a miss would say the key holds no body at all.
    let before = shell.body_stats();
    let fed = [*produced.last().expect("the prefill produced a token")];
    let first = shell
        .fire(&[Lane {
            slot: 0,
            word: word(1),
            tokens: &fed,
        }])
        .expect("the first decode fires");
    produced.push(argmax(&first[0]));
    let after = shell.body_stats();
    eprintln!("across the first decode fire: {after}");
    assert_eq!(
        (
            after.tally.hits - before.tally.hits,
            after.tally.misses - before.tally.misses,
            after.tally.reshapes - before.tally.reshapes,
            after.tally.captures - before.tally.captures,
        ),
        (1, 0, 0, 0),
        "the FIRST decode fire of a load-armed shell did not replay the body \
         the load captured. A moved `reshapes` says the decode plan's hashed \
         image still follows this fire rather than the bucket — which is what \
         the ceiling carve exists to stop — and a moved `misses` says the key \
         held no body at all: before {before} / after {after}"
    );

    // ── And the rest of the stream, which must add nothing but hits.
    let steady = shell.body_stats();
    for step in 1..STEPS {
        let fed = [*produced.last().expect("a step feeds the last token back")];
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        produced.push(argmax(&decode[0]));
    }
    let done = shell.body_stats();
    eprintln!("after {STEPS} decodes: {done}");
    eprintln!("continuation: {:?}", tokenizer.decode(&produced, false));
    assert_eq!(
        done.tally.misses, steady.tally.misses,
        "a decode fire past the first one missed, so the armed body stopped \
         serving mid-stream: {done}"
    );
    // **AND `reshapes` STAYS WHERE THE LOAD LEFT IT, ACROSS THE WHOLE
    // STREAM.** Asked against `before` and not against `steady`, so that the
    // window covers the first fire too: the number this stream must not move
    // is the one that says a schedule shape wandered inside one key, and
    // under a ceiling carve there is nothing left for it to wander with —
    // the decode plan's every hashed field is a function of the `BodyKey` and
    // the load.
    assert_eq!(
        done.tally.reshapes, before.tally.reshapes,
        "the decode stream reshaped its body, so some hashed plan number is \
         still following the fire rather than the bucket: {done}"
    );
    assert_eq!(
        done.tally.hits - steady.tally.hits,
        (STEPS - 1) as u64,
        "every decode fire past the first was supposed to be a hit: {done}"
    );
}

/// **THE ENUMERATED LATTICE, AND ZERO RUNTIME CAPTURE** (the tier-1
/// key-collapse wave's chunk B): a load that states `[engine] bodies` arms
/// PREFILL and MIXED keys as well as decode ones, seals the map behind them,
/// and then serves a whole script of real fires without recording a single
/// graph.
///
/// # Why the boot line is asserted through the counters rather than by reading
///
/// `Shell::arm_bodies` prints armed-of-wanted per composition kind, which is
/// the sentence an operator reads; a test cannot read stderr without
/// capturing the process, and the number it wants is not the print but what
/// the print stands for. So the claim is made where it bites: fire a
/// PREFILL-only composition and a MIXED one, and demand that each REPLAYS on
/// its first arrival. A prefill key that was not armed would miss; a mixed
/// key that was not armed would miss; and — because the map is SEALED — a
/// miss cannot become a capture, so it would show up as a
/// `sealed_declines` instead, which is the counter this test names.
///
/// # And the split that is not the synthetic's
///
/// The arming pass synthesizes a mixed composition as ONE decode lane beside
/// an even spread of prefill lanes. The script below brings two decode lanes
/// and one prefill lane, then one decode lane and two prefill lanes, both
/// landing in the same bucket — geometries no synthesis guessed. They replay
/// anyway, and the reason is the whole of grid-at-ceiling: a bodied fire's
/// launches are issued at the ceilings the KEY spells (`Run::carve_rows`,
/// `Run::carve_lanes`) rather than at the fire's live span, so the arming
/// capture is maximal for its key whatever geometry took it and every split of
/// the bucket finds a body it fits.
///
/// A `captures` that moves anywhere past the load is the failure this test
/// exists to catch: it would mean the serving path minted a graph, which is
/// exactly what the seal is there to make impossible.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn an_enumerated_load_arms_prefill_and_mixed_keys_and_then_captures_nothing() {
    let _serial = serialized();
    // The ladder gate's deployment, and for the ladder gate's reason: eight
    // seats and `max_lanes` at or above a mixed key's LADDER LANE REACH (both
    // classes carved to the lane ceiling — eight and eight). Step 4d clamps
    // its lane padding at `max_lanes`, so a deployment under the reach cannot
    // take the second class's lane ceiling at all and its mixed fires reshape
    // as their lane split moves — which would be this test failing on the
    // DEPLOYMENT rather than on the wave.
    let Some((mut shell, tokenizer)) =
        ready_with("the enumerated arming gate", Graphs::On, true, 8, 16, 0)
    else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    let armed = shell.body_stats();
    eprintln!("after load, before any fire: {armed}");
    assert!(
        shell.bodying(),
        "the boot stated `[engine] bodies` and the shell came up without it"
    );
    // The default lattice at `max_tokens = 256` is six points, and a
    // single-decode-class bake enumerates ONE decode key (eight seats round to
    // the lattice's first rung and no higher one is reachable), six prefill
    // keys and six mixed ones. Asserting the exact thirteen would be asserting
    // about the catalog's class table; asserting that it is well past the
    // decode rungs alone is the claim — a decode-only arming could not reach
    // it.
    assert!(
        armed.tally.armed_at_load > 2,
        "the load armed {} bodies, which is at most its decode rungs — the \
         prefill and mixed enumeration did not run, or every key of it \
         refused. The boot line names which: {armed}",
        armed.tally.armed_at_load,
    );
    assert_eq!(
        armed.tally.captures, armed.tally.armed_at_load,
        "an armed body is a captured one and nothing but the arming had run: \
         {armed}"
    );

    // ── THE PREFILL, WHICH USED TO BE THE COMPOSITION NOBODY ARMED. It is
    //    the first real fire of the load and it must replay.
    for slot in 0..4u32 {
        shell.open(slot).unwrap_or_else(|why| panic!("slot {slot} opens: {why}"));
    }
    let before = shell.body_stats();
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt[..],
        }])
        .expect("the prefill fires");
    let after = shell.body_stats();
    assert_eq!(
        moved(&before, &after),
        (1, 0, 0, 0, 0),
        "the FIRST prefill fire of an enumerated load did not replay. \
         (hits, misses, reshapes, captures, sealed_declines) moved by {:?}: a \
         sealed decline says this bucket's prefill key was never armed — the \
         boot line says whether it was wanted and refused or dropped by the \
         truncation — and a reshape says the synthetic prefill's hashed plan \
         image is not a real one's: before {before} / after {after}",
        moved(&before, &after),
    );
    let mut fed = vec![argmax(&prefill[0])];

    // A second decode lane needs kv of its own, and this prefill is a second
    // fire of the same armed key.
    let primed = shell
        .fire(&[Lane {
            slot: 1,
            word: word(prompt.len() as u32),
            tokens: &prompt[..],
        }])
        .expect("the second slot's prefill fires");
    fed.push(argmax(&primed[0]));

    // ── THE MIXED FIRES, AT TWO SPLITS, NEITHER OF THEM THE SYNTHESIS'S.
    //    Two decode lanes beside one prefill lane of five rows (seven rows,
    //    bucket eight), then one decode lane beside two prefill lanes of
    //    three and two (six rows, bucket eight) — one key, two geometries,
    //    and the arming fired a third.
    let filler = vec![prompt[0]; 8];
    let splits: [(usize, &[u32]); 2] = [(2, &[5u32]), (1, &[3u32, 2])];
    for (round, (decodes, chunks)) in splits.into_iter().enumerate() {
        let before = shell.body_stats();
        let held: Vec<[u32; 1]> = (0..decodes).map(|lane| [fed[lane]]).collect();
        let mut lanes: Vec<Lane<'_>> = held
            .iter()
            .enumerate()
            .map(|(lane, tokens)| Lane {
                slot: lane as u32,
                word: word(1),
                tokens: tokens.as_slice(),
            })
            .collect();
        for (at, rows) in chunks.iter().copied().enumerate() {
            lanes.push(Lane {
                slot: (2 + at) as u32,
                word: word(rows),
                tokens: &filler[..rows as usize],
            });
        }
        let answered = shell
            .fire(&lanes)
            .unwrap_or_else(|why| panic!("mixed round {round} fires: {why}"));
        for lane in 0..decodes {
            fed[lane] = argmax(&answered[lane]);
        }
        let after = shell.body_stats();
        assert_eq!(
            moved(&before, &after),
            (1, 0, 0, 0, 0),
            "mixed round {round} — {decodes} decode lane(s) beside {chunks:?} \
             prefill row(s) — did not replay an armed body. \
             (hits, misses, reshapes, captures, sealed_declines) moved by {:?}. \
             A moved `captures` is the sentence this gate exists to refuse: the \
             serving path minted a graph past the seal. A `sealed_declines` says \
             this mixed key was never armed. A `misses` with the map unsealed \
             says the seal did not take: before {before} / after {after}",
            moved(&before, &after),
        );
    }

    let done = shell.body_stats();
    eprintln!("after the enumerated script: {done}");
    assert_eq!(
        done.tally.captures, armed.tally.captures,
        "the serving path recorded a graph. Every capture on an enumerated \
         load belongs to `Shell::arm_bodies`, and the seal is what makes that \
         a property rather than a hope: {done}"
    );
}

/// `(hits, misses, reshapes, captures, sealed_declines)` across two readings —
/// the five numbers every claim in the gate above is made of.
fn moved(before: &BodyStats, after: &BodyStats) -> (u64, u64, u64, u64, u64) {
    (
        after.tally.hits - before.tally.hits,
        after.tally.misses - before.tally.misses,
        after.tally.reshapes - before.tally.reshapes,
        after.tally.captures - before.tally.captures,
        after.tally.sealed_declines - before.tally.sealed_declines,
    )
}

/// **A DECODE FIRE THAT PRESENTS TWO DECODE WORDS REPLAYS FROM THE LOAD**
/// (the hybrid decode wave, `serve::arming::ensemble_keys`) — the claim the
/// arming enumeration's blind spot cost, stated where it bites.
///
/// # The blind spot
///
/// Every arm of the enumeration presented decode as a SINGLETON. `decode_keys`
/// walks one decode class per rung; `mixed_keys` pairs a decode class with a
/// NON-decode one; `fragmented_keys` walks the present sets that break a mask.
/// So a bake with two decode words had no key for `{both}` — and on such a
/// bake `{both}` is what a batched decode fire brings whenever the scheduler
/// has one sequence of each word in flight, which for a steady stream is
/// almost always. Past the seal every one of those fires walked eagerly and was
/// counted in `sealed_declines`, for the life of the load. It was found on a
/// `gemma4-e4b` load — a hybrid ATTENTION bake, whose sliding arm and full arm
/// are two classes that both run `Attention::Decode` — reading
/// `the sealed map holds no body for b256[c0:256 c1:256]`.
///
/// # Why THIS bake has two decode words, and why that is the same claim
///
/// Not a hybrid attention: this SKU's attention is uniform. Its two decode
/// classes are `1` (plain, one row) and `3` (plain, one row, WITH an adapter)
/// — `Facts::has_adapter` crossing `Facts::qo_one`, which is the other way a
/// class table reaches two `Attention::Decode` masks. The composition is the
/// same shape: one present set, two classes, both decode, one row per lane,
/// and a `record::Ladder` with two rungs of `min(lane_ceiling, bucket)`. What
/// the enumerator produces for it is what it produces for a hybrid attention
/// bake, and the arm cannot tell the two apart — it reads `Shell::decoding`
/// and nothing else. The `gemma4-e4b` SKU has no CI gate on this tree; this is
/// where the claim rides.
///
/// # And it is `hits` that says it
///
/// The map is SEALED, so a key the boot did not arm cannot become a capture on
/// the serving path: it walks and shows up as a `sealed_decline`. So a fire
/// that replays on its FIRST arrival is a fire the load armed, and the three
/// numbers below say which of the two happened.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_two_word_decode_fire_replays_from_the_load() {
    let _serial = serialized();
    // One adapter bank's worth of capacity, which is what buys the right to
    // compose class `3` at all — and four seats, so the one reachable rung is
    // the lattice floor at four lanes and the ensemble arm splits it two and
    // two.
    let Some((mut shell, tokenizer)) =
        ready_with("the hybrid decode gate", Graphs::On, true, 4, 4, 1)
    else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    register_loud(&mut shell, ADAPTER);

    let armed = shell.body_stats();
    eprintln!("after load, before any fire: {armed}");
    assert!(
        armed.tally.armed_at_load >= 1,
        "the boot armed nothing at all, so nothing below is about arming: {armed}"
    );

    // ── TWO SLOTS, PRIMED ONE WORD EACH. The prefills are ordinary fires and
    //    are not what this test is about; what they buy is kv under each lane,
    //    so that the fire below is a DECODE of two words rather than a prefill
    //    of them.
    for slot in 0..2u32 {
        shell.open(slot).unwrap_or_else(|why| panic!("slot {slot} opens: {why}"));
    }
    let plain = shell
        .fire_seated(&[routed_seat(0, &prompt, None)])
        .expect("the plain prefill fires");
    let routed = shell
        .fire_seated(&[routed_seat(1, &prompt, Some(ADAPTER))])
        .expect("the routed prefill fires");
    let fed = [argmax(&plain[0]), argmax(&routed[0])];

    // ── AND THE COMPOSITION THE DEFECT WAS ABOUT: one lane of each decode
    //    word, one row each, in ONE fire.
    let before = shell.body_stats();
    let two = shell
        .fire_seated(&[
            routed_seat(0, &fed[..1], None),
            routed_seat(1, &fed[1..], Some(ADAPTER)),
        ])
        .expect("the two-word decode fires");
    let after = shell.body_stats();
    assert_eq!(two.len(), 2, "a two-lane fire answers two lanes");
    assert_eq!(
        moved(&before, &after),
        (1, 0, 0, 0, 0),
        "the FIRST two-word decode fire of an armed load did not replay.          (hits, misses, reshapes, captures, sealed_declines) moved by {:?}. A          sealed decline is the defect itself: the enumeration armed each decode          word alone and never the pair, so this composition — which is what a          hybrid load's decode traffic IS — walks eagerly for the life of the          load. The boot line's `ensemble` column says whether the arm ran:          before {before} / after {after}",
        moved(&before, &after),
    );
}

/// **AND A LOAD THAT STANDS THE WORD DOWN ARMS NOTHING**, which is the other
/// half of chunk C being a choice rather than a fact.
///
/// The sentence used to be "a load that NEVER STATES the word", and the
/// keyed path's deletion inverted it: `Knobs::bodies` defaults to TRUE now, so
/// silence arms everything and the deployment that wants an unarmed load has
/// to say so. What did not change is the mechanism this asserts.
/// `Shell::arm_bodies` is called unconditionally at the tail of every load and
/// returns on its gate when `[engine] bodies` is stood down — no synthetic
/// fire, no capture, no counter, and no boot line. (The gate has three more
/// clauses and each returns the same way: a mode that records nothing, an
/// artifact with more capture units than a `BodyKey` names — none today, since
/// the key carries a lattice point per row axis — and a load whose weights
/// rotate. This test states the first, which is the one a deployment
/// chooses.) The counters are what a
/// test can see, and they are the default value; a moving one would mean the
/// load had fired something nobody asked it to.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_load_that_stands_the_word_down_arms_nothing_at_all() {
    let _serial = serialized();
    let Some((shell, _tokenizer)) = ready("the unarmed load", false) else {
        return;
    };
    let bodies = shell.body_stats();
    eprintln!("after an unarmed load: {bodies}");
    assert!(
        !shell.bodying(),
        "this load stated `bodies: false` and the shell came up bodying"
    );
    assert_eq!(
        bodies,
        engine_cuda::record::BodyStats::default(),
        "a load that stated nothing armed something anyway: {bodies}"
    );
}

/// **THE MIXED FIRE, CAPTURED AT ONE SPLIT AND REPLAYED AT ANOTHER** — the
/// gate this campaign has owed since chunk 2b-ii, and the first one whose
/// subject is the LANE axis rather than the row axis (bodies design, chunk
/// 2c-b).
///
/// # What a mixed fire asks that a decode stream does not
///
/// Every gate above serves ONE class. Its regions are whole-fire, `lane_offset`
/// is zero everywhere, and the two readings of every table — the window's and
/// the fire's — are the same bytes. So none of them can see the thing chunk
/// 2c-b built: a windowed region whose launches are handed the PLANE's base
/// and have to find their own lanes inside it.
///
/// A mixed fire is that composition. Two lanes decode while two prefill, so
/// the attention regions window, the second class's window begins at
/// `lane_offset = 2`, and every FA2 launch in it reads request ids the
/// schedule staged (`lane_offset + r`) against page bounds, last-page fills
/// and mask spans handed over whole (`Run::pool_absolute`). Get either half
/// wrong and the fire reads another lane's pages — silently, with no fault and
/// no shape error, which is why the oracle here is TOKENS and the arm it is
/// diffed against is the eager walk.
///
/// # And the two re-splits are the point
///
/// The script fires the same composition and moves the rows twice. The first
/// move is BETWEEN THE TWO PREFILL LANES: `6 + 2` becomes `4 + 4`. Same
/// classes, same lane counts, same bucket, same per-window row totals — so
/// `Run::schedule_shape` does not move and a body captured on an earlier round
/// is eligible to REPLAY on this one rather than reshape. What replays is a
/// graph holding baked pointers, and the whole of chunk 2c-b's lane-axis work
/// is the claim that those pointers are functions of the LOAD and not of this
/// fire's split.
///
/// The second move is the ROW TOTAL: `5 + 2`, seven prefill rows inside the
/// same bucket of sixteen. That one has nothing to do with lanes and
/// everything to do with what a prefill SCHEDULE is carved at — the
/// plan-at-bucket-ceiling design's chunk 4 — and the assertion at the end of
/// this test is its own.
///
/// # What this gate asserts, and what the wall in front of it used to be
///
/// Token identity is asserted unconditionally: it holds whichever path the
/// bodies arm actually took. The REPLAY used to be asserted only weakly, and
/// two walls stood in front of it in turn. The first was the ADMISSIBILITY
/// rule — a graph may hold a region only when its window is whole-fire or
/// shifting (`Windows::admits`), and a mixed fire windows the plan-op regions
/// too — which `PLANNED` answered: a plan op is prepare-phase host work that
/// puts no node in the graph, so it can be windowed without addressing
/// anything wrongly. The second was this model's own mixer.
///
/// **AND NEITHER WALL WOULD REFUSE THE COMPOSITION TODAY**, which is worth
/// saying because it changes what a failure here MEANS. Since the tier-2
/// campaign a region the rule turns away is an island: the body is cut around
/// it and this fire replays with `LastCapture::islands` nonzero. So a hit is
/// the whole of what is owed, and the ISLAND COUNT IS NOT ASSERTED — which is
/// a change from what this note used to promise and is the lane axis's doing.
///
/// **THE LANE AXIS TAKES ONE REGION BACK ON THIS SKU, AND THAT IS CORRECT**
/// (`crate::LANE_SHIFTED`). `SHIFTED` speaks for ROWS; a region admitted on it
/// alone is handed the plane's base and then reads its per-lane tables off
/// pointers this shell advanced by `lane_offset`, which a body bakes and a
/// `record::BodyKey` does not fix. The gdn DECODE region is exactly that: it
/// runs the per-STEP scans, whose slot map, fold predicate and commit length
/// arrive sliced (`Run::recurrent`), and on a mixed fire its window begins
/// above lane zero. So it is an island now, the stretches around it are
/// captured, and it is re-issued eagerly at this fire's own lane offset. The
/// PREFILL mixer regions begin at lane zero and are untouched, and so is every
/// full-attention class region, whose two names took the absolute door in
/// chunk 2c-b.
///
/// **THE MODEL THIS GATE LOADS IS AN SSM HYBRID, AND THAT USED TO BE THE
/// WALL.** `qwen35-d0.8b` runs three gated-delta layers to every full
/// attention one (`Model::d0_8b_dims`: twenty-four layers, `attn_every = 4`),
/// and `qwen_3::forward::gdn_mixer` splits each of those on `Facts::qo_one()`
/// — so a mixed fire windows a PREFILL mixer region per gdn layer, holding
/// `layout.split_rows`, `attention.ssm_causal_conv1d_chunked`,
/// `attention.ssm_gdn_prep` and `attention.ssm_gated_delta_chunked`. The two
/// chunked names were off `SHIFTED` for a plain reason — they took no seat at
/// all — and one windowed region on no list is enough to refuse the whole
/// composition. So the honest claim here was the refusal plus token identity.
///
/// **THE CHUNKED-ARM WAVE PROMOTED THEM, AND THIS GATE IS WHERE THAT IS
/// SPENT.** The seat grew a lane half — `win[2]` retires a request grid's
/// padded lanes, `win[3]` names the fire lane of a window-local request — and
/// the four chunked arms read it, against per-lane tables
/// `Run::recurrent_absolute` now hands over whole. With those two names on the
/// list, every windowed region a d0.8b mixed fire presents is shifting or
/// planned: the attention layers' four arms (masked and score-capture empty
/// here, `attention.decode` and `attention.prefill` on the list) beside their
/// plan ops, and both arms of every gdn layer. The `linear.lora_correct` and
/// media windows this text also cuts are EMPTY in this script — no lane
/// carries an adapter or an image — and `Windows::admits` reads a region with
/// no rows as capturable.
///
/// So the assertion below is the replay itself. A failure of it is not a
/// missing feature any more: it says a mixed composition this shell should
/// have served was refused, or captured and never replayed, and
/// `BodyStats` printed above says which.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_mixed_fire_says_at_one_split_what_the_eager_walk_says_at_another() {
    let _serial = serialized();
    // **THE LADDER GATE'S DEPLOYMENT, AND FOR THE LADDER GATE'S REASON** —
    // which this test did not need until its script started moving the LANE
    // SPLIT, and needs absolutely now. `Ladder::lane_reach` states the
    // inequality a mixed key's second class needs before it can take a lane
    // ceiling at all: `max_lanes >= 2 x min(slots, max_lanes, max_tokens)`,
    // because step 4d clamps its lane padding at `max_lanes` and
    // `Carve::lanes` carves the second class between the prefix sum in front
    // of it and that clamp. `ready`'s four-and-four is UNDER it — lane ceiling
    // four, prefix four, nothing left to carve — so the decode class's
    // `Shape::num_requests` and `Shape::lane_offset` fall back to following
    // this fire's batch (`Run::planning`'s `None` arm), and a hashed number
    // that follows the batch is a reshape on every fire whose split moved.
    //
    // While every round fired the same 2+2 that cost nothing: the fallback
    // answered `lane_offset = 2` five times running and the hash sat still.
    // Round 4 moves the split, so it is the first fire of this file that could
    // ever have seen it — the tokens stayed identical (the fallback is
    // CORRECT, it is only unhashable) and the replay went away. `(8, 16, 0)`
    // is the sibling gate's deployment and satisfies the inequality: lane
    // ceiling eight, reach sixteen, `max_lanes` sixteen — so both classes
    // carve, both hashed numbers are key functions, and the moved split
    // replays instead of reshaping.
    let Some((mut shell, tokenizer)) =
        ready_with("the mixed-fire bodies gate", Graphs::On, false, 8, 16, 0)
    else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ONE LOAD, TWO ARMS — `a_fire_served_from_a_body_...`'s reason exactly.
    shell.set_mode(Graphs::Off);
    shell.set_bodies(false);
    let (eager, _, _) = mixed(&mut shell, &prompt);

    shell.set_mode(Graphs::On);
    shell.set_bodies(true);
    let (bodied, moved, split_move) = mixed(&mut shell, &prompt);

    let bodies = shell.body_stats();
    eprintln!("{bodies}");
    eprintln!(
        "mixed continuations: eager {:?} / bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );

    assert_eq!(
        eager, bodied,
        "a mixed fire disagreed with the eager walk it stands for — the lane \
         axis is what a mixed fire tests and nothing else here does: eager \
         {:?} against bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );

    // **AND A CAPTURE IS NOT AN EARLY OUT ANY MORE.** While the chunked arms
    // were off `SHIFTED` this block took a refusal as the honest answer and
    // returned on it. It cannot now: the composition is admissible, so a run
    // that captured nothing at all did not reach the gate — the bodies arm was
    // never engaged, which is a different fault from a refusal and worth its
    // own sentence before the replay claim below.
    assert!(
        bodies.tally.captures >= 1,
        "the bodies arm captured nothing at all, so it never reached the \
         mixed gate: {bodies}"
    );
    // A capture alone does not attribute: the opening prompt's prefill-only
    // composition seats a body of its own, so `captures >= 1` says nothing
    // about the MIXED key. What attributes is a HIT, and a hit is what this
    // gate now claims. On the catalog model it loads — qwen35-d0.8b, an SSM
    // hybrid — the mixed composition used to be refused by name, because its
    // windowed mixer regions run the CHUNKED ssm prefill arms and those
    // carried no seat. They carry one now (`crate::SHIFTED`, the chunked-arm
    // wave), every other windowed region this fire presents is shifting or
    // planned or empty, and the script's fourth round re-splits the rows
    // without moving a class, a lane count or a per-window row total — so the
    // body captured on an earlier round is eligible, and serving it is the
    // whole claim. (The FIFTH round moves the row total too, and the
    // assertion under this one is that claim on its own.)
    assert!(
        bodies.tally.hits >= 1,
        "no mixed replay: a composition whose windowed regions are on \
         `SHIFTED` or `PLANNED`, cut around the ones the LANE axis turns away, \
         was not served from a body at all. A refusal here names a region this \
         shell will not shift OR will not cut — read it off the counters, \
         because it is a NEW wall and not the chunked-arm one: {bodies}"
    );
    // **AND THE LAST ROUND IS A DIFFERENT CLAIM FROM THE ONES BEFORE IT**
    // (plan-at-bucket-ceiling, chunk 4). Rounds 0-4 re-split eight rows
    // between two prefill lanes; the totals never move, so the prefill plan's
    // hashed image could not move either and the replay there says nothing
    // about the row axis. Round 5 fires `5 + 2` — seven prefill rows against
    // eight, nine fire rows against ten, the SAME bucket (16), the same
    // classes, the same lane counts.
    //
    // Before chunk 4 that fire reshaped by construction: `total_num_rows` rode
    // `PrefillPlanInfo` and `PrefillPlan::total_tokens`, both hashed, and a
    // window that carried seven rows hashed differently from one that carried
    // eight. Now the schedule is carved at the bucket and the seven reach the
    // device through the staged image alone, so the body captured on an
    // earlier round serves this one. A moved `reshapes` is the whole failure:
    // it says some hashed prefill number is still following the fire's rows.
    //
    // **AND THE MOVED-SPLIT ROUND IS ITS OWN CLAIM, ONE AXIS OVER**
    // (`mixed`'s own header). Round 4 puts eight prefill rows on ONE lane
    // where the rounds around it put them on two: same classes, same per-class
    // row totals, same rungs, same bucket, so the key does not move and a body
    // captured earlier is eligible. What DOES move is where the decode class's
    // window begins — lane 2 becomes lane 1 — and a shell that baked a
    // per-lane pointer at the first split reads ANOTHER LANE's state at the
    // second, which is the defect `cuda_width_invariance` found. The token
    // diff above is the oracle; these two clauses are what say the diff was
    // taken over a REPLAY, because a round that reshaped instead would have
    // re-captured at the new split and hidden the fault.
    assert_eq!(
        split_move.1, 0,
        "the moved-split round RESHAPED rather than replaying, so this gate \
         still cannot see a body read at a lane offset other than its \
         recording one. Nothing in a `record::BodyKey` moved between that fire \
         and the ones around it, so a reshape names a hashed schedule number \
         that is following the LANE SPLIT. Read the DEPLOYMENT first: \
         `Ladder::lane_reach` needs `max_lanes >= 2 x min(slots, max_lanes, \
         max_tokens)` before a mixed key's second class can take a lane \
         ceiling at all, and this test is seated at the sibling gate's \
         `(8, 16)` precisely so that it can. Under that inequality this \
         failure is the deployment; at or above it, it is a schedule field \
         `Run::planning` has not frozen: {split_move:?} — {bodies}"
    );
    assert!(
        split_move.0 >= 1,
        "the round that moves the lane split did not replay at all, so the \
         lane axis was never crossed under a body and the token diff above \
         proves only what the rounds before it proved: {split_move:?} — \
         {bodies}"
    );
    assert_eq!(
        moved,
        (1, 0),
        "the re-totalled mixed fire did not replay: the last round moved \
         (hits, reshapes) by {moved:?} where chunk 4 promises (1, 0). A \
         reshape says the prefill plan's hashed image still follows this \
         fire's row total rather than its bucket; a zero hit with a zero \
         reshape says the body was short or was never captured: {bodies}"
    );
}

/// The mixed script both arms run: two primed decode lanes beside a prefill
/// class fired at one split, then at a second, then at a third that moves the
/// prefill class's LANE COUNT, then at a fourth whose row total is different.
///
/// Rounds 0-3 move the rows only INSIDE the prefill class, so those fires
/// present the same classes, the same lane counts and the same per-window row
/// totals. Round 5 moves the TOTAL — the whole fire carries nine rows where
/// the others carried ten — and it lands in the same bucket (16), which is the
/// case chunk 4 exists for.
///
/// # AND ROUND 4 MOVES THE SPLIT ITSELF, WHICH IS WHAT THIS SCRIPT USED TO
/// MISS
///
/// Every round of this script used to fire TWO decode lanes beside TWO prefill
/// lanes. The rows moved; the lane counts never did. So the second class's
/// window began at `lane_offset = 2` in every fire of the key, every pointer
/// this shell advances by that number was baked and re-read at the SAME
/// number, and the one thing a mixed body can get wrong on the lane axis was
/// the one thing no round crossed.
///
/// `cuda_width_invariance` crossed it by accident — a decode probe beside ONE
/// prefill neighbour, replaying a body armed against SEVEN — and answered
/// eleven logits wrong with a flipped argmax and no fault anywhere. Round 4 is
/// that fire's coordinates inside this gate's key: EIGHT prefill rows on ONE
/// lane where the rounds around it put eight rows on two. Same classes, same
/// per-class row totals, same rungs, same bucket — so the key does not move
/// and the body captured on an earlier round is eligible — while the decode
/// class's window slides from lane 2 to lane 1. A shell that bakes a per-lane
/// pointer at one split and replays it at another reads another lane's state
/// here, and the token diff is what says so.
///
/// **AND A ROUND THAT MOVES THE SPLIT ASKS SOMETHING OF THE DEPLOYMENT THAT
/// THE OTHERS DID NOT.** A hashed schedule number that follows the batch costs
/// nothing while every fire brings the same batch, and costs a RESHAPE the
/// moment one does not — so the caller has to be seated above
/// `record::Ladder::lane_reach`'s inequality before this round can replay at
/// all. The test above says which seating and why; an author adding a sixth
/// round that moves the split again inherits the same requirement.
///
/// Returns the tokens, the `(hits, reshapes)` the LAST round alone moved, and
/// the same pair for the MOVED-SPLIT round — because a round that reshaped
/// instead of replaying would prove nothing at all.
fn mixed(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, (u64, u64), (u64, u64)) {
    /// The PREFILL class's chunk widths, round by round — one entry per
    /// prefill lane. Rounds 0-4 hold the class's row total at 8 and the last
    /// one drops it to 7; round 4 is the one that holds the total while
    /// moving the LANE COUNT, which is what slides the decode class's window.
    const SPLITS: [&[u32]; 6] = [&[6, 2], &[6, 2], &[6, 2], &[4, 4], &[8], &[5, 2]];
    /// Which round moves the split rather than the rows.
    const MOVED_SPLIT: usize = 4;

    for slot in 0..4 {
        shell.open(slot).unwrap_or_else(|why| panic!("slot {slot} opens: {why}"));
    }

    // The two lanes that will decode have to hold kv first, so each is primed
    // with a prefill of its own — single-class fires, which is why they are
    // outside the loop and not part of what this gate diffs.
    let mut fed = Vec::new();
    for slot in 0..2u32 {
        let primed = shell
            .fire(&[Lane {
                slot,
                word: word(prompt.len() as u32),
                tokens: prompt,
            }])
            .unwrap_or_else(|why| panic!("slot {slot}'s priming prefill fires: {why}"));
        fed.push(argmax(&primed[0]));
    }

    let mut produced = fed.clone();
    // The prefill lanes are fed a repeating token so a chunk of any width is
    // always available: what this gate is about is the SHAPE of the fire, and
    // a prompt long enough to slice four ways would only add a dependency on
    // the tokenizer's output.
    let filler = vec![prompt[0]; 8];
    let mut last = (0, 0);
    let mut split_move = (0, 0);
    for (round, chunks) in SPLITS.iter().copied().enumerate() {
        // The counters bracketing the LAST round, so the claim is about the
        // fire whose row total moved and not about the ones in front of it —
        // and the same bracket around the round that moves the SPLIT.
        let before = shell.body_stats();
        let a = [fed[0]];
        let b = [fed[1]];
        // The two decode lanes first, then one prefill lane per chunk. The
        // prefill slots start at 2 so a round with fewer lanes drops the
        // LAST of them rather than renumbering the ones that stay.
        let mut lanes = vec![
            Lane { slot: 0, word: word(1), tokens: &a },
            Lane { slot: 1, word: word(1), tokens: &b },
        ];
        for (at, chunk) in chunks.iter().copied().enumerate() {
            lanes.push(Lane {
                slot: at as u32 + 2,
                word: word(chunk),
                tokens: &filler[..chunk as usize],
            });
        }
        let answered = shell
            .fire(&lanes)
            .unwrap_or_else(|why| panic!("mixed round {round} fires: {why}"));
        // Every lane's answer goes into the diff, not only the decodes': a
        // lane axis read one place off would show up in whichever lane it
        // landed on.
        for lane in &answered {
            produced.push(argmax(lane));
        }
        fed[0] = argmax(&answered[0]);
        fed[1] = argmax(&answered[1]);
        if round + 1 == SPLITS.len() || round == MOVED_SPLIT {
            let after = shell.body_stats();
            let moved = (
                after.tally.hits - before.tally.hits,
                after.tally.reshapes - before.tally.reshapes,
            );
            if round == MOVED_SPLIT {
                split_move = moved;
            } else {
                last = moved;
            }
        }
    }
    (produced, last, split_move)
}

/// **THE GROWTH-AFTER-CAPTURE GATE** — the regression the grown-slab commit
/// (`81b793c9e`, grow-retires-instead-of-freeing) deserves and never got.
///
/// `Shell::arm_bodies` walks its rungs ASCENDING: the smallest rung's bodies
/// are captured first, and every later rung's eager warm pass GROWS the
/// rows-sized scratch slabs — the gdn staged planes above all — moving the
/// live base while the small bodies keep the address they baked. Retirement
/// is what keeps that address mapped; put the free back and the smallest
/// rung's replay reads a dead block. This gate is that sentence as a test:
/// arm at least two rungs, then serve the SMALLEST one and diff its tokens
/// against an eager load that never captured anything.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B snapshot; run with -- --ignored"]
fn a_body_armed_before_the_slabs_grew_still_reads_its_own_scratch() {
    let _serial = serialized();
    let Some((mut eager, tokenizer)) =
        ready_with("the growth gate's oracle", Graphs::Off, false, 16, 16, 0)
    else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let (eager_tokens, _) = run(&mut eager, &prompt);
    drop(eager);

    let Some((mut shell, _)) = ready_with("the growth gate", Graphs::On, true, 16, 16, 0)
    else {
        return;
    };
    let armed = shell.body_stats();
    eprintln!("after the wide load: {armed}");
    assert!(
        armed.tally.armed_at_load >= 3,
        "fewer than two rungs' worth of bodies armed, so no slab grew between \
         two captures and this gate is not testing growth-after-capture: {armed}"
    );

    // One lane, one row per fire: the SMALLEST rung's bucket, whose body was
    // captured before every later rung's warm pass grew the slabs.
    let (body_tokens, _) = run(&mut shell, &prompt);
    let after = shell.body_stats();
    eprintln!("after the smallest rung's stream: {after}");
    assert!(
        after.tally.hits >= 1,
        "the smallest rung's fires never replayed, so nothing read a \
         pre-growth capture: {after}"
    );
    assert_eq!(
        eager_tokens, body_tokens,
        "the smallest rung's body disagrees with the eager walk — a replay \
         read scratch at an address growth has since abandoned"
    );
}

/// **THE LADDER GATE** (the ceiling design's Option B, finished by the tier-1
/// key collapse): a key whose per-class CEILINGS hold replays however the
/// split moves inside them, and a BUCKET that still splits the key when the
/// fire outgrows it.
///
/// # Why this is not the mixed gate over again
///
/// `a_mixed_fire_says_at_one_split_...` moves rows inside the PREFILL class
/// and then moves the fire's row total, and both of those are claims the
/// bucket alone could make: one class's carve was the fire's bucket, and a
/// bucket does not move while the total stays inside it. What it could not
/// claim is anything about the DECODE class beside it. A windowed class's
/// lanes were carved at its own live count and its `Shape::lane_offset` was
/// the fire's own — so a mixed key whose decode lane count wandered reshaped
/// on every wander, which is what the chunk 5 note called the half-frozen
/// windowed class.
///
/// Option B put a lattice rung per class into `record::BodyKey`, so a
/// window's three carved numbers — the rows in front of it, its rows, its
/// lanes — are prefix sums over that ladder. This gate is the two sentences
/// that follow.
///
/// # The script, and what each round is for
///
/// Three decode lanes beside two prefill lanes, on the default lattice, whose
/// first rung is eight and whose lane ceiling here is eight seats. Rounds 0
/// and 1 warm and capture the key `b16[decode:8 prefill:16]`. Round 2
/// re-splits the prefill rows — eight rows as `4 + 4` rather than `5 + 3` —
/// which the mixed gate already covers. **ROUND 3 IS THIS GATE'S FIRST
/// CLAIM**: it drops a DECODE lane and a prefill row at once — two decode
/// lanes of one row each, seven prefill rows — so BOTH axes of BOTH classes
/// move while neither class leaves the key. Before Option B the decode plan's
/// `num_requests` followed that lane count and the fire reshaped; now it is
/// the key's ceiling, and the fire replays.
///
/// **ROUND 4 IS THE SECOND CLAIM, AND THE KEY COLLAPSE INVERTED IT.** Nine
/// prefill rows against eight — one row across what used to be the lattice's
/// first rung — with the fire's total (twelve rows) landing in the same
/// bucket of sixteen the other rounds landed in. It used to be a MISS, and
/// the assertion here used to say so: a rung was read off the class's own
/// measured rows, so `{decode: 8, prefill: 8}` and `{decode: 8, prefill: 16}`
/// were two keys, two captures and two instantiations of launches that differ
/// in nothing a replay can see.
///
/// The tier-1 key collapse made a rung a CEILING instead
/// (`record::Ladder::rung`): a prefill class is carved to the BUCKET and a
/// decode class to the load's lane ceiling, neither of them a reading of any
/// fire. So all five rounds are one key — `b16[c:8 c:16]` on this deployment —
/// and round 4 is a HIT with nothing reshaped and nothing captured. That
/// equality is the wave's whole deliverable and this round is where it is
/// measured against a real device.
///
/// **AND ROUND 5 IS WHAT STILL SPLITS.** The BUCKET does: twenty-three rows
/// round to thirty-two, a fire of that key needs rows the sixteen-bucket
/// capture never runs, and no ceiling can spell them. So it is a first fire of
/// a second key — a miss, warming toward its own capture — and it is the
/// honest survivor of the round the collapse retired.
///
/// The three assertions are three different counter triples and each would
/// fail differently: a reshape on round 3 says some hashed field still follows
/// the split, a MISS on round 4 says a measurement is back in the key, and a
/// hit on round 5 says the bucket left it.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_split_inside_one_bucket_replays_and_a_split_across_one_is_a_second_key() {
    let _serial = serialized();
    // Five lanes, so more seats than the siblings; and `max_lanes` at or above
    // the LADDER'S LANE REACH — two present classes, each carved to the lane
    // ceiling this eight-seat load states, so 8 + 8 — which is what step 4d
    // clamps its padding at and therefore what bounds the carve a windowed
    // class can take. A deployment whose `max_lanes` sits under that reach
    // still serves every fire, but the class standing LAST in row order finds
    // the staging already spent by the prefix in front of it and takes no lane
    // ceiling at all: its schedule's `num_requests` goes back to following the
    // batch, which is round 3's whole claim.
    //
    // (`record::Ladder::lane_reach` is why the number is 8 + 8 and not
    // 16 + 8. A prefill class's ROW rung is the whole bucket, but a lane needs
    // a SEAT, so the lane reading caps every rung at `min(slots, max_lanes,
    // max_tokens)`. Uncapped, the prefill class's origin alone would consume
    // the sixteen lanes step 4d can stage and leave the decode class behind it
    // with nothing.)
    let Some((mut shell, tokenizer)) =
        ready_with("the ladder gate", Graphs::On, false, 8, 16, 0)
    else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ONE LOAD, TWO ARMS — `a_fire_served_from_a_body_...`'s reason exactly.
    shell.set_mode(Graphs::Off);
    shell.set_bodies(false);
    let (eager, _, _, _) = laddered(&mut shell, &prompt);

    shell.set_mode(Graphs::On);
    shell.set_bodies(true);
    let (bodied, inside, collapsed, across) = laddered(&mut shell, &prompt);

    let bodies = shell.body_stats();
    eprintln!("{bodies}");
    assert_eq!(
        eager, bodied,
        "the ladder script disagreed with the eager walk it stands for: \
         eager {:?} against bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );

    assert_eq!(
        inside,
        (1, 0, 0),
        "a split that moved BOTH classes on BOTH axes inside their rungs did \
         not replay: round 3 moved (hits, misses, reshapes) by {inside:?} \
         where Option B promises (1, 0, 0). A reshape says a hashed field \
         still follows the split — the decode plan's `num_requests` and its \
         `lane_offset` are the two this round exists to pin; a miss with no \
         reshape says the body was short or was never captured: {bodies}"
    );
    assert_eq!(
        collapsed,
        (1, 0, 0),
        "nine prefill rows against eight, inside ONE bucket, was not one body: \
         round 4 moved (hits, misses, reshapes) by {collapsed:?} where the key \
         collapse promises (1, 0, 0). A MISS says a rung is being read off the \
         class's measured rows again — which is exactly what \
         `record::Ladder::rung` retired — or that the capture was too SHORT for \
         this fire, which would say the launches are gridded at the live span \
         rather than at the key's ceiling (`Run::carve_rows`). A RESHAPE says \
         a hashed plan field followed the extra row: {bodies}"
    );
    assert_eq!(
        across,
        (0, 1, 0),
        "a fire that crossed a BUCKET was not a new key: round 5 moved \
         (hits, misses, reshapes) by {across:?} where the collapse promises \
         (0, 1, 0) — a first fire of a second key, warming toward its own \
         capture. A HIT says the bucket left the key, which is the one axis a \
         body genuinely cannot span: {bodies}"
    );
}

/// The ladder script both arms run, and the three counter triples the rounds
/// that matter moved.
///
/// Returns the tokens, then `(hits, misses, reshapes)` for round 3 — the split
/// that moves both classes on both axes inside one key — for round 4, the one
/// that crosses what used to be a rung, and for round 5, the one that crosses
/// a BUCKET.
fn laddered(
    shell: &mut Shell,
    prompt: &[u32],
) -> (Vec<u32>, (u64, u64, u64), (u64, u64, u64), (u64, u64, u64)) {
    /// `(decode lanes, long prefill chunk, short prefill chunk)` per round.
    ///
    /// Rounds 0-2 hold three decode lanes and eight prefill rows (eleven, so
    /// bucket sixteen); round 3 drops to two decode lanes and seven prefill
    /// rows — nine rows, still bucket sixteen, both classes moved on both
    /// axes; round 4 takes the prefill class to NINE, which is the first row
    /// past what used to be the lattice's first rung and is still bucket
    /// sixteen; and round 5 takes the fire to twenty-three rows, which is
    /// bucket THIRTY-TWO and therefore a second key.
    ///
    /// **AND THE ORDER OF THE GEOMETRIES IS NO LONGER A CONSTRAINT.** It used
    /// to be: `record::Body::grids` recorded what each LAUNCH was issued over,
    /// those were the fire's live spans, and a key climbed by MISSING when a
    /// window grew — so the biggest geometry had to come first or a round
    /// would be asserting about the climb instead of about its own claim. The
    /// grid-at-ceiling wave issues a bodied fire's launches at the key's
    /// ceiling instead (`Run::carve_rows`), so the first capture of a key is
    /// already maximal for it and rounds 3 and 4 grow their windows on
    /// purpose. That growth is now part of what round 4 asserts.
    const ROUNDS: [(u32, u32, u32); 6] =
        [(3, 5, 3), (3, 5, 3), (3, 4, 4), (2, 4, 3), (3, 5, 4), (3, 12, 8)];
    /// Which slots decode, and which prefill.
    const DECODES: u32 = 3;

    for slot in 0..DECODES + 2 {
        shell.open(slot).unwrap_or_else(|why| panic!("slot {slot} opens: {why}"));
    }

    // The decode lanes have to hold kv first, so each is primed with a prefill
    // of its own — single-class fires, outside what this gate diffs.
    let mut fed = Vec::new();
    for slot in 0..DECODES {
        let primed = shell
            .fire(&[Lane {
                slot,
                word: word(prompt.len() as u32),
                tokens: prompt,
            }])
            .unwrap_or_else(|why| panic!("slot {slot}'s priming prefill fires: {why}"));
        fed.push(argmax(&primed[0]));
    }

    let mut produced = fed.clone();
    let filler = vec![prompt[0]; 16];
    let mut inside = (0, 0, 0);
    let mut collapsed = (0, 0, 0);
    let mut across = (0, 0, 0);
    for (round, (decodes, long, short)) in ROUNDS.iter().copied().enumerate() {
        let before = shell.body_stats();
        let held: Vec<[u32; 1]> = (0..decodes as usize).map(|lane| [fed[lane]]).collect();
        let mut lanes: Vec<Lane<'_>> = held
            .iter()
            .enumerate()
            .map(|(lane, tokens)| Lane {
                slot: lane as u32,
                word: word(1),
                tokens: tokens.as_slice(),
            })
            .collect();
        lanes.push(Lane {
            slot: DECODES,
            word: word(long),
            tokens: &filler[..long as usize],
        });
        lanes.push(Lane {
            slot: DECODES + 1,
            word: word(short),
            tokens: &filler[..short as usize],
        });
        let answered = shell
            .fire(&lanes)
            .unwrap_or_else(|why| panic!("ladder round {round} fires: {why}"));
        // Every lane's answer goes into the diff, not only the decodes': a
        // lane axis read one place off would show up in whichever lane it
        // landed on.
        for lane in &answered {
            produced.push(argmax(lane));
        }
        for lane in 0..decodes as usize {
            fed[lane] = argmax(&answered[lane]);
        }
        let after = shell.body_stats();
        let moved = (
            after.tally.hits - before.tally.hits,
            after.tally.misses - before.tally.misses,
            after.tally.reshapes - before.tally.reshapes,
        );
        if round == 3 {
            inside = moved;
        }
        if round == 4 {
            collapsed = moved;
        }
        if round == 5 {
            across = moved;
        }
    }
    (produced, inside, collapsed, across)
}

/// **THE THREE MODES SAY ONE THING** — `graph_replay.rs`'s first claim, which
/// outlived the file that carried it.
///
/// That suite was the keyed path's A/B and died with it. Two of its three
/// claims were definitionally keyed (a shape captures once; a second fire of a
/// key does not recapture) and are answered here by the arming and seal gates
/// instead. The third is not about keying at all: it is that `Graphs` has
/// THREE values and a difference between them has an AUTHOR.
///
/// `Off` is the golden — no graph path exists, and the attention schedules are
/// carved at the fire's own split. `Shaped` is the same eager walk with
/// graph-shaped (padded) attention schedules and nothing recorded: it is the
/// arm that isolates flashinfer's padded split, which is a numerics question
/// and not a capture one. `On` is tiered — bodies where the composition is
/// admissible, the eager walk where it is not.
///
/// So a break between `Off` and `Shaped` is the padded split, and a break
/// between `Shaped` and `On` is the BODY. Without the middle arm a failure
/// here would name only "graphs", and the two candidates live in different
/// crates.
///
/// The `On` arm asserts `hits >= 1` and not `captures >= 1`: on a load that
/// armed at boot the capture counter is already nonzero before a caller fires,
/// so only a hit says a fire was SERVED from a body. This load states
/// `bodies: false` so that the arming pass does not run at all and each arm is
/// spelled by the test — the same one-load, three-arms discipline the A/B at
/// the top of this file uses, for the same reason.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_shaped_walk_and_the_tiered_one_say_what_the_eager_walk_says() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the three-mode sweep", false) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ONE LOAD, THREE ARMS. Two loads would be two residencies, two arenas and
    // two tuner histories, and a difference could be any of them.
    shell.set_mode(Graphs::Off);
    shell.set_bodies(false);
    let (golden, golden_ms) = run(&mut shell, &prompt);

    shell.set_mode(Graphs::Shaped);
    shell.set_bodies(false);
    let (shaped, shaped_ms) = run(&mut shell, &prompt);

    shell.set_mode(Graphs::On);
    shell.set_bodies(true);
    let (tiered, tiered_ms) = run(&mut shell, &prompt);

    let bodies = shell.body_stats();
    eprintln!(
        "decode ms/fire (warm half of {STEPS}): eager {:.3}  shaped {:.3}  tiered {:.3}",
        warm(&golden_ms),
        warm(&shaped_ms),
        warm(&tiered_ms),
    );
    eprintln!("{bodies}");
    eprintln!(
        "continuations: eager {:?} / shaped {:?} / tiered {:?}",
        tokenizer.decode(&golden, false),
        tokenizer.decode(&shaped, false),
        tokenizer.decode(&tiered, false),
    );

    assert_eq!(
        golden, shaped,
        "graph-shaped attention schedules changed the continuation before any \
         graph existed, so the break is the padded split and not the capture: \
         eager {:?} against shaped {:?}",
        tokenizer.decode(&golden, false),
        tokenizer.decode(&shaped, false),
    );
    assert!(
        bodies.tally.hits >= 1,
        "the tiered arm never replayed a body, so it was the shaped arm over \
         again and the third column asserts nothing: {bodies}"
    );
    assert_eq!(
        shaped, tiered,
        "the tiered arm disagreed with the shaped walk its bodies were \
         captured from — the break is the graph: shaped {:?} against tiered \
         {:?}",
        tokenizer.decode(&shaped, false),
        tokenizer.decode(&tiered, false),
    );
}

/// **AND TWO RESIDENT BODIES DO NOT DISTURB EACH OTHER** — `fold_gate.rs`'s
/// alternating workload, which is the one claim of that suite that was never
/// about the fold.
///
/// The fold gate alternated compositions because that made every fire flip the
/// enable bits the previous one had set: the workload was an instrument for a
/// mechanism that no longer exists. What the workload IS, though, is two
/// compositions of one bucket arriving in turn — decode-only, then
/// decode-beside-prefill — and under the bodies path that is two resident
/// bodies serving alternately out of one map.
///
/// Nothing else in this file asks that. Every other script settles into ONE
/// composition and re-splits it; here the map is asked to hand back the right
/// body when the previous fire was somebody else's, with the scratch slabs,
/// the staged seat and the arena column all having been written by the
/// composition in between. A body that had baked something belonging to its
/// NEIGHBOUR — a seat address, a plane base, a schedule the other fire's
/// prepare wrote — would produce exactly the right tokens on the steady
/// scripts and wrong ones here.
///
/// So the oracle is the eager walk of the same alternation, and `hits >= 1`
/// says the bodies arm actually served from the map rather than walking both
/// compositions.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn compositions_that_alternate_say_what_the_eager_walk_says() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the alternating gate", false) else {
        return;
    };
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");

    // ONE LOAD, TWO ARMS — `a_fire_served_from_a_body_...`'s reason exactly.
    shell.set_mode(Graphs::Off);
    shell.set_bodies(false);
    let eager = alternating(&mut shell, &carried, &fresh, STEPS);

    shell.set_mode(Graphs::On);
    shell.set_bodies(true);
    let bodied = alternating(&mut shell, &carried, &fresh, STEPS);

    let bodies = shell.body_stats();
    eprintln!("{bodies}");
    eprintln!(
        "alternating continuations: eager {:?} / bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );

    assert!(
        bodies.tally.hits >= 1,
        "neither composition of the alternation was ever served from a body, \
         so this diffed the eager walk against itself; `refusals` says whether \
         the admissibility rule turned the mixed half away: {bodies}"
    );
    assert_eq!(
        eager, bodied,
        "the alternation disagreed with the eager walk it stands for — what \
         only this script can see is one body baking something that belongs to \
         the composition fired in between: eager {:?} against bodies {:?}",
        tokenizer.decode(&eager, false),
        tokenizer.decode(&bodied, false),
    );
}

/// One carried decode lane, with a fresh prefill lane re-seated beside it on
/// every ODD step — compositions alternate inside one bucket, so consecutive
/// fires never key to the same body. Returns the decode lane's tokens.
fn alternating(
    shell: &mut Shell,
    carried: &[u32],
    fresh: &[u32],
    steps: usize,
) -> Vec<u32> {
    shell.open(0).expect("slot 0 opens");
    let seated = shell
        .fire(&[Lane {
            slot: 0,
            word: word(carried.len() as u32),
            tokens: carried,
        }])
        .expect("the carried prefill fires");
    let mut decode = vec![argmax(&seated[0])];
    for step in 0..steps {
        let fed = [*decode.last().expect("a step has a last token")];
        let out = if step % 2 == 1 {
            shell.open(1).expect("slot 1 opens");
            shell
                .fire(&[
                    Lane {
                        slot: 0,
                        word: word(1),
                        tokens: &fed,
                    },
                    Lane {
                        slot: 1,
                        word: word(fresh.len() as u32),
                        tokens: fresh,
                    },
                ])
                .unwrap_or_else(|why| panic!("mixed step {step} fires: {why}"))
        } else {
            shell
                .fire(&[Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &fed,
                }])
                .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"))
        };
        decode.push(argmax(&out[0]));
    }
    decode
}
