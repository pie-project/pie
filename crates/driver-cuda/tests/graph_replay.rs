//! The A/B: what an eager fire says, a replayed fire must say — token for
//! token, over a real checkpoint (palo porting order, step 5).
//!
//! **WHY THIS SUITE EXISTS EVEN THOUGH CAPTURED IS EAGER BY CONSTRUCTION.**
//! Decision #11 puts the equivalence in the architecture: one walk, two
//! sinks, and `record.rs` captures the same regions the eager pass just ran.
//! What construction CANNOT give is the other half of the claim — that
//! everything a captured launch froze is actually fire-invariant. A pointer
//! that moved, an extent that was this fire's rather than this key's, a plan
//! whose offsets shifted under new contents: each of those produces a graph
//! that runs, finishes, and returns slightly wrong numbers forever. So the
//! subject is diffed against the golden at the only place the difference is
//! visible from outside, which is the token stream.
//!
//! Three claims:
//!
//! 1. **identity** — one prompt, one slot, sixteen greedy decode steps, run
//!    eagerly and then replayed: the same tokens. Run in three modes, not
//!    two, so that a difference has an author: `Off` is the golden, `Shaped`
//!    is the same eager walk with graph-shaped (padded) attention schedules,
//!    and `On` adds the capture. A break between Off and Shaped is
//!    flashinfer's padded split; a break between Shaped and On is the graph.
//! 2. **mixed** — design §0's headline fire (a decode lane beside a prefill
//!    lane, over disjoint row windows) replayed rather than walked.
//! 3. **cache** — the second fire of a key does not capture again, and a new
//!    shape captures exactly once. Watched through the capture COUNTER,
//!    because "it did not recapture" is not a property any output has.
//!
//! # Gating
//!
//! As `serve_smoke.rs`: skipped at run time when the machine, the checkpoint
//! or the tokenizer is missing, rather than `#[ignore]`d.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p driver-cuda --features cuda-13 --test graph_replay -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use driver_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this suite serves, as `serve_smoke` serves it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt.
const PROMPT: &str = "The capital of France is";

/// How many decode fires follow the prefill.
///
/// Sixteen for two reasons. It is enough for a STEADY state — the first two
/// fires of the decode key warm and capture, and the fourteen after them are
/// the ones under test. And it CROSSES A PAGE BOUNDARY: the prompt is five
/// tokens, the pool pages sixteen, so the sequence grows from one page to two
/// underneath a graph that was captured at one. Page count is the first thing
/// that would be wrong if a captured launch had frozen an extent that belongs
/// to the fire rather than to the key.
const STEPS: usize = 16;

/// One shell at a time, per process — `kernels-cuda`'s scratch slabs are
/// process-global (`serve_smoke.rs` states the whole argument).
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
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// A loaded shell, or `None` and a sentence saying what was missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !driver_cuda::device::present() {
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

    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        // Every test below sets the mode it means before it fires; the load
        // states the golden so that a test which forgot would be diffing the
        // golden against itself rather than silently recording.
        graphs: Graphs::Off,
    })
    .expect("the shell loads");
    Some((shell, tokenizer))
}

/// One prefill and `STEPS` greedy decodes in slot 0, in whatever mode the
/// shell is in. Returns the tokens and the per-fire decode milliseconds.
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

/// The mean of the warm half — the steady state, past the fires that warmed
/// and captured.
fn warm(millis: &[f64]) -> f64 {
    let warm = &millis[millis.len() / 2..];
    warm.iter().sum::<f64>() / warm.len() as f64
}

/// Claim 1: eager and replayed produce the same tokens, and the three modes
/// say which layer any difference belongs to.
#[test]
fn a_replayed_fire_says_token_for_token_what_an_eager_fire_says() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the capture A/B") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ONE LOAD, THREE MODES. Two loads would be two residencies, two arenas
    // and two tuner histories, and a difference could be any of them.
    shell.set_mode(Graphs::Off);
    let (golden, eager_ms) = run(&mut shell, &prompt);

    shell.set_mode(Graphs::Shaped);
    let (shaped, shaped_ms) = run(&mut shell, &prompt);

    shell.set_mode(Graphs::On);
    let (replayed, replay_ms) = run(&mut shell, &prompt);

    let stats = shell.graph_stats();
    eprintln!(
        "decode ms/fire (warm half of {STEPS}): eager {:.3}  shaped {:.3}  replay {:.3}",
        warm(&eager_ms),
        warm(&shaped_ms),
        warm(&replay_ms),
    );
    eprintln!(
        "graphs: {} captured ({} nodes, {:.1} ms), {} replayed, {} warming, {} declined, \
         {} resident",
        stats.captures,
        stats.nodes,
        stats.capture_millis,
        stats.replays,
        stats.warming,
        stats.declined,
        stats.execs,
    );
    eprintln!(
        "continuations: eager {:?} / shaped {:?} / replay {:?}",
        tokenizer.decode(&golden, false),
        tokenizer.decode(&shaped, false),
        tokenizer.decode(&replayed, false),
    );

    assert!(
        stats.captures >= 1 && stats.replays >= 1,
        "the graph mode neither captured nor replayed anything, so this test \
         compared eager against eager"
    );
    assert_eq!(
        golden, shaped,
        "graph-shaped attention schedules changed the continuation before any \
         graph existed: eager {:?} against shaped {:?}",
        tokenizer.decode(&golden, false),
        tokenizer.decode(&shaped, false),
    );
    assert_eq!(
        shaped, replayed,
        "the replayed fire disagreed with the eager fire it was captured from: \
         {:?} against {:?}",
        tokenizer.decode(&shaped, false),
        tokenizer.decode(&replayed, false),
    );
}

/// Claim 2: design §0's headline fire, replayed.
///
/// A decode lane beside a prefill lane is two windows of one batch, and under
/// capture it is also two attention schedules whose shapes must be a function
/// of the key rather than of this fire's kv contents. The prefill lane is
/// re-seated every step so the mixed shape repeats and therefore captures.
#[test]
fn a_mixed_fire_replays_what_it_says_eagerly() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the mixed-fire replay") else {
        return;
    };
    const MIXED: usize = 8;
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");

    // The golden: the same mixed fires, eagerly.
    let eager = mixed(&mut shell, Graphs::Off, &carried, &fresh, MIXED);
    let replay = mixed(&mut shell, Graphs::On, &carried, &fresh, MIXED);

    let stats = shell.graph_stats();
    eprintln!(
        "mixed: {} captured, {} replayed — decode lane {:?}, prefill lane {:?}",
        stats.captures,
        stats.replays,
        tokenizer.decode(&replay.0, false),
        tokenizer.decode(&replay.1, false),
    );
    assert!(
        stats.replays >= 1,
        "no mixed fire replayed, so nothing about capture was tested: {stats:?}"
    );
    assert_eq!(
        eager.0, replay.0,
        "the decode lane of a replayed mixed fire said {:?} where the eager one \
         said {:?}",
        tokenizer.decode(&replay.0, false),
        tokenizer.decode(&eager.0, false),
    );
    assert_eq!(
        eager.1, replay.1,
        "the prefill lane of a replayed mixed fire disagreed with the eager one",
    );
}

/// One carried decode lane beside a freshly re-seated prefill lane, `steps`
/// times. Returns `(the decode lane's tokens, the prefill lane's tokens)`.
fn mixed(
    shell: &mut Shell,
    mode: Graphs,
    carried: &[u32],
    fresh: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<u32>) {
    shell.set_mode(mode);
    shell.open(0).expect("slot 0 opens");
    let seated = shell
        .fire(&[Lane {
            slot: 0,
            word: word(carried.len() as u32),
            tokens: carried,
        }])
        .expect("the carried prefill fires");
    let mut decode = vec![argmax(&seated[0])];
    let mut prefilled = Vec::with_capacity(steps);
    for step in 0..steps {
        shell.open(1).expect("slot 1 opens");
        let fed = [*decode.last().expect("a step has a last token")];
        let out = shell
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
            .unwrap_or_else(|why| panic!("the mixed fire at step {step} fires: {why}"));
        decode.push(argmax(&out[0]));
        prefilled.push(argmax(&out[1]));
    }
    (decode, prefilled)
}

/// Claim 3: a key is captured once, and only a new shape captures again.
///
/// **THE PROPERTY IS AN ABSENCE**, and no output has it: a shell that
/// recaptured every fire would produce exactly the right tokens, slowly. So
/// the counter is the instrument.
#[test]
fn a_key_captures_once_and_a_new_shape_captures_once_more() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the graph cache") else {
        return;
    };
    let first = tokenizer.encode(PROMPT);
    let second = tokenizer.encode("The largest planet in the solar system is");
    shell.set_mode(Graphs::On);

    // One decode lane, over and over: warm, capture, then replay forever.
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(first.len() as u32),
            tokens: &first,
        }])
        .expect("the prefill fires");
    let mut carried = vec![argmax(&prefill[0])];
    for _ in 0..6 {
        carried.push(step(&mut shell, &[0], &[*carried.last().expect("a token")]));
    }
    let one_lane = shell.graph_stats();
    assert_eq!(
        one_lane.captures, 1,
        "six fires of one decode shape captured {} graphs, and one shape is one \
         graph (design §5)",
        one_lane.captures,
    );
    assert_eq!(
        one_lane.replays,
        6 - u64::from(driver_cuda::record::WARM_FIRES),
        "every decode fire past the warm ones and the capture should have replayed",
    );
    assert!(one_lane.nodes > 0, "the captured graph holds no nodes");

    // More of the same key: not one more capture.
    for _ in 0..4 {
        carried.push(step(&mut shell, &[0], &[*carried.last().expect("a token")]));
    }
    let again = shell.graph_stats();
    assert_eq!(
        again.captures, one_lane.captures,
        "the same shape captured a second graph"
    );
    assert_eq!(again.replays, one_lane.replays + 4);

    // A NEW shape — two decode lanes — captures exactly once, behind the same
    // two warm fires, and leaves the first exec alone.
    shell.open(1).expect("slot 1 opens");
    let seated = shell
        .fire(&[Lane {
            slot: 1,
            word: word(second.len() as u32),
            tokens: &second,
        }])
        .expect("the second prefill fires");
    let mut other = vec![argmax(&seated[0])];
    let mut fed = (
        *carried.last().expect("a token"),
        *other.last().expect("a token"),
    );
    for _ in 0..4 {
        let out = shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &[fed.0],
                },
                Lane {
                    slot: 1,
                    word: word(1),
                    tokens: &[fed.1],
                },
            ])
            .expect("the two-lane decode fires");
        fed = (argmax(&out[0]), argmax(&out[1]));
        carried.push(fed.0);
        other.push(fed.1);
    }
    let two_lane = shell.graph_stats();
    eprintln!(
        "cache: {} captures over {} execs, {} replays, {:.1} ms captured, {} nodes",
        two_lane.captures,
        two_lane.execs,
        two_lane.replays,
        two_lane.capture_millis,
        two_lane.nodes,
    );
    assert_eq!(
        two_lane.captures,
        again.captures + 1,
        "a two-lane decode is a different shape and should have captured once",
    );
    assert_eq!(
        two_lane.execs, 2,
        "two shapes, two execs — and the first was not evicted",
    );
    assert_eq!(
        two_lane.replays,
        again.replays + 4 - u64::from(driver_cuda::record::WARM_FIRES),
        "the new shape should have warmed, captured on its second fire and replayed \
         the rest",
    );
}

/// One decode fire over `slots`, greedy, returning the first lane's token.
fn step(shell: &mut Shell, slots: &[u32], tokens: &[u32]) -> u32 {
    let lanes: Vec<Lane> = slots
        .iter()
        .zip(tokens)
        .map(|(slot, token)| Lane {
            slot: *slot,
            word: word(1),
            tokens: core::slice::from_ref(token),
        })
        .collect();
    let out = shell.fire(&lanes).expect("the decode fires");
    argmax(&out[0])
}
