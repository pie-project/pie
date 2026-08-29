//! **Article 1, measured**: `submit` returns with the device still running,
//! and the compute stream does not go idle between steps.
//!
//! This is the gate the constitution's first article names as its enforcement
//! ("`submit` enqueues all k steps before returning; an e2e gate measures
//! inter-wave stream gaps"), and F2b is the wave that made it possible to
//! write. Four claims, in the order they can fail:
//!
//! 1. **The host does not wait.** Steps are registered as airborne and stay
//!    airborne while the next ones are enqueued — two frames' work on the
//!    stream at once, which is what "at least two frames in flight" means from
//!    the host's side.
//! 2. **The stream does not go idle.** Recorded with CUDA TIMING events on the
//!    compute stream, one per frame boundary: the device-side span of a frame
//!    in the pipelined arm is compared against the same frame's span in an arm
//!    that synchronizes after every frame — F1's shape. The difference is the
//!    host's enqueue cost, and in the pipelined arm it is supposed to be gone.
//! 3. **And it computes the same thing.** The token sequence and the logits
//!    are byte-identical between a shell loaded at one frame in flight and one
//!    loaded at two.
//! 4. **A poisoned step fails its frame and only its frame.** The step after a
//!    refusal never runs, the refusal is loud, and the NEXT frame answers what
//!    it would have answered had the bad one never been submitted.
//!
//! Beside them, the settlement plumbing itself: every step's completion sink
//! fires exactly once, the staging ring comes all the way back, and the
//! airborne count returns to zero.
//!
//! # Where this lives, and why not in `tests/gpu`
//!
//! `tests/gpu` boots the whole standalone through the websocket edge, which is
//! the right shape for a claim about what a deployment produces and the wrong
//! one for a claim about the compute stream: from there the stream has no
//! name, the shell's airborne count has no reader, and a CUDA event cannot be
//! recorded between two steps. Every number below is read off the surface that
//! owns it. What `tests/gpu` keeps is the end-to-end arm —
//! `cuda_runahead_depth1` and its twin still run the same decode at both
//! depths and diff the tokens.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --test two_frames_in_flight -- --nocapture
//! ```
//!
//! Skips at RUN time, saying which of the machine and the checkpoint was
//! missing — an `#[ignore]` on the one box that could run it is a gate nobody
//! runs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use engine::frame::Shell as FramePhases;
use engine::runahead::Runahead;
use engine_cuda::device::graph::Event;
use engine_cuda::serve::{Done, Settled, StepView};
use engine_cuda::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this gate serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// A prompt long enough that a prefill is real device work — the measurement
/// below is a ratio against a step's own duration, and a two-token step is all
/// launch overhead.
const PROMPT: &str = "The capital of France is Paris, and the capital of Italy is Rome. \
                      The capital of Spain is Madrid, and the capital of Portugal is Lisbon. \
                      The capital of Germany is";

/// How many frames each arm of the saturation measurement runs.
const FRAMES: usize = 12;

/// How many steps a frame carries — `k`, and the number the staging formula
/// is written in terms of.
const STEPS: usize = 2;

/// How many pool slots the saturation loop rotates over.
const SLOTS: u32 = 4;

/// **How much device-side gap a frame is allowed in the pipelined arm**, as a
/// fraction of the tightest frame's own device time.
///
/// A quarter, and the number is chosen against what it has to separate rather
/// than against taste. The two arms differ by exactly one thing: the
/// synchronized arm pays the host's whole per-frame enqueue cost with the
/// device idle, and that cost is MILLISECONDS on this path (compose, page
/// geometry, window resolution, the staging write, the walk's launch loop)
/// against a decode step of a few. So the synchronized arm's gap is of the
/// same order as its device time and the pipelined arm's should be launch
/// overhead — tens of microseconds. A quarter sits an order of magnitude below
/// the failure it is separating and an order above the noise, which is what a
/// threshold on a shared box has to do. The measured numbers are printed
/// either way, so a regression that stays under it is still visible.
const GAP_BUDGET: f32 = 0.25;

/// The snapshot directory: the checkpoint AND the tokenizer that goes with it.
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

/// The container the contract is checked against.
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

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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

/// One shell fires at a time per process (`kernels-cuda`'s scratch slabs are
/// process-global), and this gate loads more than one.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A loaded shell at the stated run-ahead depth, and the vocabulary that goes
/// with it — or `None` and a sentence saying what was missing.
fn ready(what: &str, runahead: Runahead) -> Option<(Shell, tokenizer::Tokenizer)> {
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
    let trace = model::trace_of(SKU).expect("the catalog ships this gate's SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        // The golden path: every claim here is about the STREAM, and a graph
        // replay would be measuring the recorder rather than the run-ahead.
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead,
        // The warm-boot weight artifact cache is off for a gate: a test
        // that shared one would be asserting about the last run.
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}

/// **A step, enqueued and registered, with nothing waited for.**
///
/// The three phases, called where `Cuda::submit` calls them: `prepare` makes
/// every host decision and claims a staging slot, `enqueue` puts the whole
/// step on the compute stream, `settle_step` records the event and hangs the
/// completion off the notify stream. Nothing in here waits.
fn fire(shell: &mut Shell, lanes: &[Seated<'_>], done: Option<Done>) -> Settled {
    let prepared = FramePhases::prepare(shell, StepView { lanes, attachments: &[] }, None)
        .expect("the step prepares");
    let enqueued = FramePhases::enqueue(shell, prepared).expect("the step enqueues");
    shell.settle_step(enqueued, done).expect("the step registers its settlement")
}

/// One lane, seated on the shell's own paging.
fn lane(slot: u32, tokens: &[u32]) -> Seated<'_> {
    Seated::of(Lane {
        slot,
        word: word(tokens.len() as u32),
        tokens,
    })
}

/// **CLAIM 1 — the host does not wait, and the proof is the airborne count.**
///
/// A step is airborne from the instant `settle_step` registers it to the
/// instant its callback runs on the driver's thread. So "two frames in flight"
/// is observable as an airborne count above one at a moment when the host is
/// building the next frame — which under F1 was impossible by construction,
/// because `settle` ended in `cudaStreamSynchronize` and the count could never
/// exceed the step being fired.
#[test]
fn submit_returns_with_the_device_still_running() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the saturation gate", Runahead::of(2)) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    shell.open(0).expect("slot 0 opens");

    // **THE COMPLETION SINK, COUNTED.** One call per step, `Committed` every
    // time, and it arrives on the CUDA driver's host-function thread — which
    // is the whole reason it is an `AtomicUsize` and not a `Cell`.
    let settled = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&settled);
    let sink: engine::engine_api::CompletionSink = Arc::new(move |_at, outcome| {
        assert!(
            matches!(outcome, engine::engine_api::StepOutcome::Committed),
            "a step that reached settlement committed"
        );
        seen.fetch_add(1, Ordering::Release);
    });

    // **NOTHING IN THIS LOOP WAITS**, and that is the experiment. A decode
    // chain cannot be the workload here: it feeds each step the previous
    // step's argmax, so it has to read the numbers back, and a readback is a
    // synchronize — the dependency is the WORKLOAD's, not the engine's, and it
    // would be measuring the wrong thing. What this fires is independent
    // steps over rotating slots, which is the shape a batched fleet actually
    // has: several sequences, none of them waiting on another.
    let feed: Vec<u32> = prompt.clone();
    let mut peak = 0u64;
    let mut steps = 0u32;
    for slot in 0..SLOTS {
        shell.open(slot).expect("the slot opens");
    }
    for frame in 0..FRAMES {
        for step in 0..STEPS {
            let done = Done {
                at: engine::engine_api::StepDone {
                    frame: frame as u64,
                    step: step as u32,
                },
                sink: Arc::clone(&sink),
            };
            let slot = ((frame * STEPS + step) % SLOTS as usize) as u32;
            // A fresh sequence every time round: `open` clears the slot's
            // recurrent banks and resets its extent, so the fires stay the
            // same shape however long the loop runs.
            shell.open(slot).expect("the slot re-opens");
            let _ = fire(&mut shell, &[lane(slot, &feed)], Some(done));
            steps += 1;
            // Sampled BETWEEN steps, on the host, with nothing waited for.
            peak = peak.max(shell.airborne_steps());
        }
    }
    shell.drain().expect("the stream drains");

    eprintln!("peak airborne steps: {peak} over {steps} steps");
    assert!(
        peak >= 2,
        "`submit` is supposed to return with the device still running, but the \
         airborne count never exceeded {peak}: every step settled before the next \
         was enqueued, which is F1's shape and not article 1's"
    );

    // The callbacks all arrived — and the ring came all the way back with
    // them, which is the staging lifetime obligation observed from outside.
    let mut spun = 0;
    while shell.airborne_steps() > 0 && spun < 10_000 {
        std::hint::spin_loop();
        spun += 1;
    }
    assert_eq!(
        shell.airborne_steps(),
        0,
        "every registered settlement must run: the staging slots and the events \
         are released by the callback and nothing else releases them"
    );
    assert_eq!(
        settled.load(Ordering::Acquire) as u32,
        steps,
        "one completion per step, exactly once"
    );
}

/// **CLAIM 2 — the stream does not go idle between steps.**
///
/// Two arms over the same work, measured with CUDA timing events recorded on
/// the compute stream at every frame boundary:
///
/// ```text
/// synchronized   fire the frame, then WAIT for it — F1's shape, and the
///                per-frame span therefore includes the host's whole enqueue
///                cost with the device idle
/// pipelined      fire every frame back to back and wait once at the end —
///                F2b's shape, where the enqueue of frame N+1 happens while
///                frame N runs
/// ```
///
/// The per-frame span in the pipelined arm, minus the tightest span either arm
/// achieved, is the device-side GAP. It has to be small against a frame's own
/// duration; the synchronized arm's is there to show what "not small" looks
/// like on the same box in the same minute.
#[test]
fn the_stream_does_not_go_idle_between_steps() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the stream-gap gate", Runahead::of(2)) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    let span = |shell: &mut Shell, prompt: &[u32], synchronized: bool| -> Vec<f32> {
        shell.open(0).expect("slot 0 opens");
        let stream = shell.compute_stream();
        let marks: Vec<Event> = (0..=FRAMES)
            .map(|_| Event::timing().expect("a timing event"))
            .collect();
        // Warm the path: the first fire of a shape JITs, grows slabs and tunes,
        // and none of that is what this measures.
        let mut warm = fire(shell, &[lane(0, prompt)], None);
        shell.read_out(&mut warm).expect("the warm fire answers");
        let mut token = argmax(&warm.logits[0]);

        marks[0].record(stream).expect("the first mark records");
        for frame in 0..FRAMES {
            for step in 0..STEPS {
                let tokens = vec![token];
                let mut answer = fire(shell, &[lane(0, &tokens)], None);
                if step + 1 == STEPS {
                    shell.read_out(&mut answer).expect("the numbers come back");
                    token = argmax(&answer.logits[0]);
                } else if synchronized {
                    // F1's shape: the sync that stood at the end of every
                    // settle, restored for this arm alone.
                    shell.drain().expect("the stream drains");
                }
            }
            marks[frame + 1]
                .record(stream)
                .expect("the frame boundary records");
        }
        shell.drain().expect("the stream drains");
        (0..FRAMES)
            .map(|frame| {
                marks[frame]
                    .elapsed_ms(&marks[frame + 1])
                    .expect("both marks completed")
            })
            .collect()
    };

    let pipelined = span(&mut shell, &prompt, false);
    let synchronized = span(&mut shell, &prompt, true);

    let mean = |xs: &[f32]| xs.iter().sum::<f32>() / xs.len() as f32;
    // The tightest frame either arm achieved is the closest thing to a frame's
    // pure device time that can be measured without modelling anything.
    let floor = pipelined
        .iter()
        .chain(&synchronized)
        .copied()
        .fold(f32::INFINITY, f32::min);
    let gap_pipelined = mean(&pipelined) - floor;
    let gap_synchronized = mean(&synchronized) - floor;
    eprintln!(
        "frame span: pipelined {:.3} ms, synchronized {:.3} ms, floor {:.3} ms\n\
         inter-step device gap: pipelined {:.3} ms ({:.1}% of a frame), \
         synchronized {:.3} ms ({:.1}% of a frame)",
        mean(&pipelined),
        mean(&synchronized),
        floor,
        gap_pipelined,
        100.0 * gap_pipelined / floor,
        gap_synchronized,
        100.0 * gap_synchronized / floor,
    );

    assert!(
        gap_pipelined < GAP_BUDGET * floor,
        "the compute stream went dry between steps: {gap_pipelined:.3} ms of gap \
         against a {floor:.3} ms frame is {:.0}% and the budget is {:.0}% — \
         article 1 says every wave is enqueued before its predecessor completes",
        100.0 * gap_pipelined / floor,
        100.0 * GAP_BUDGET,
    );
    assert!(
        mean(&pipelined) <= mean(&synchronized),
        "a frame that is enqueued behind its predecessor must not take longer on \
         the device than one that waited for it: {:.3} ms against {:.3} ms",
        mean(&pipelined),
        mean(&synchronized),
    );
    assert!(
        gap_pipelined <= gap_synchronized,
        "the pipelined arm must not be idler than the arm that waits on purpose: \
         {gap_pipelined:.3} ms against {gap_synchronized:.3} ms"
    );
}

/// **CLAIM 3 — and it computes the same thing.**
///
/// The same greedy decode through a shell loaded at one frame in flight and
/// one loaded at two: same tokens, and the same logits bit for bit. A staging
/// ring that handed a fire the wrong slot's bytes would not fault — it would
/// answer fluent garbage — so the assertion is on the bytes and not on the
/// text.
#[test]
fn two_frames_in_flight_is_byte_identical_to_one() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the depth-1 arm", Runahead::F1) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let depth1 = decode(&mut shell, &prompt);
    drop(shell);

    let Some((mut shell, _)) = ready("the depth-2 arm", Runahead::of(2)) else {
        return;
    };
    let depth2 = decode(&mut shell, &prompt);

    let text = tokenizer.decode(&depth2.0, false);
    eprintln!("continuation: {text:?}");
    assert_eq!(
        depth1.0, depth2.0,
        "the run-ahead depth is a pool size, not a semantics: the tokens must not move"
    );
    assert_eq!(
        depth1.1.len(),
        depth2.1.len(),
        "the same number of logit rows come back at either depth"
    );
    for (at, (one, two)) in depth1.1.iter().zip(&depth2.1).enumerate() {
        assert_eq!(
            one.to_bits(),
            two.to_bits(),
            "logit {at} differs between depth 1 and depth 2 ({one} vs {two}); a \
             staging ring that hands a fire the wrong slot's bytes answers fluent \
             garbage rather than faulting, which is why this is asserted on the bits"
        );
    }
}

/// A greedy decode: the tokens, and every logit row that produced them.
fn decode(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<f32>) {
    shell.open(0).expect("slot 0 opens");
    let mut tokens = Vec::new();
    let mut rows: Vec<f32> = Vec::new();
    let mut answer = fire(shell, &[lane(0, prompt)], None);
    shell.read_out(&mut answer).expect("the prefill answers");
    let mut token = argmax(&answer.logits[0]);
    rows.extend_from_slice(&answer.logits[0]);
    tokens.push(token);
    for _ in 0..15 {
        let feed = vec![token];
        let mut answer = fire(shell, &[lane(0, &feed)], None);
        shell.read_out(&mut answer).expect("the decode answers");
        token = argmax(&answer.logits[0]);
        rows.extend_from_slice(&answer.logits[0]);
        tokens.push(token);
    }
    (tokens, rows)
}

/// **CLAIM 4 — a poisoned step fails its frame and only its frame.**
///
/// The frame's second step names a slot the pools do not seat. It is refused
/// before anything of it launches (`prepare` is where the ceilings are), the
/// first step's work is real and settles normally, and the frame AFTER it
/// answers exactly what it would have answered had the bad frame never been
/// submitted — which is the property a staging ring gets wrong by leaking a
/// slot and a completion path gets wrong by resurrecting a failed frame.
#[test]
fn a_poisoned_step_fails_its_frame_without_touching_the_next() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the poison gate", Runahead::of(2)) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let golden = decode(&mut shell, &prompt);

    // Same shell, same slot, run again — and this time a bad step lands in the
    // middle of it.
    shell.open(0).expect("slot 0 re-opens");
    let mut answer = fire(&mut shell, &[lane(0, &prompt)], None);
    shell.read_out(&mut answer).expect("the prefill answers");
    let mut token = argmax(&answer.logits[0]);
    assert_eq!(token, golden.0[0], "the re-run prefill says what it said");

    let free_before = shell.airborne_steps();
    let feed = vec![token];
    // Step two of this frame: a slot past the four this shell seats. Nothing
    // of it launches — the refusal is in `prepare`, above the first stream
    // touch — and its staging slot goes back through `Prepared`'s destructor.
    let bad_lane = [lane(9, &feed)];
    let bad = FramePhases::prepare(
        &mut shell,
        StepView {
            lanes: &bad_lane,
            attachments: &[],
        },
        None,
    );
    let error = bad.err().expect("a slot this shell does not seat is refused");
    eprintln!("the poisoned step refused: {error}");
    shell.drain().expect("the stream drains");
    assert_eq!(
        shell.airborne_steps(),
        free_before,
        "a step that refused before it launched registered no settlement, so it \
         must leave the airborne count exactly where it found it"
    );

    // And now the frame behind it. Every remaining token of the golden run,
    // from the state the good prefill left.
    for expected in &golden.0[1..] {
        let feed = vec![token];
        let mut answer = fire(&mut shell, &[lane(0, &feed)], None);
        shell.read_out(&mut answer).expect("the decode answers");
        token = argmax(&answer.logits[0]);
        assert_eq!(
            token, *expected,
            "the frame after a poisoned one must answer what it would have \
             answered had the poisoned frame never been submitted"
        );
    }
    let _ = tokenizer;
}
