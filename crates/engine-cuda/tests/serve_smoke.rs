//! The end-to-end smoke: a real checkpoint, a real prompt, real tokens back.
//!
//! **WHAT IT IS FOR.** Every layer under it is checked on its own — the fire
//! substrate has 26 tests with a mock dispatch, the compiler 55, the page
//! arithmetic its own — and none of that can tell you the model says
//! anything. A shell can bind every seat, refuse nothing, launch every kernel
//! and produce fluent-looking garbage, which is what `.wiki/tart`'s gpt-oss
//! hunt was: a load that ran. So this test asserts the one property no unit
//! test reaches, which is that the continuation is *right*.
//!
//! Three claims, in the order they can fail:
//!
//! 1. **finite** — no NaN, no infinity in the logits. A pool read at the
//!    wrong stride or a state slab of poison shows up here first.
//! 2. **deterministic** — two identical runs produce identical tokens. This
//!    is the class-resolution property `resolve_classes` exists for, observed
//!    from the outside: nondeterminism under a fixed batch means two arms
//!    wrote the same rows.
//! 3. **coherent** — the continuation of "The capital of France is" begins
//!    with " Paris". Pinned against what this shell actually produced, and it
//!    is the assertion that would have caught the saga above.
//!
//! Two more tests stand beside those three, and they are the batching claims:
//! two lanes of one class say what they say alone, and — design §0's headline
//! case — a fire carrying a decode lane BESIDE a prefill lane says what each
//! of them says alone, token for token.
//!
//! # Gating
//!
//! A build that names CUDA is not a machine that has it, and neither is a
//! machine that has it a machine with a 1.7 GB checkpoint on its disk. So the
//! test skips at RUN time, saying which of the three it was missing, rather
//! than being `#[ignore]`d — an ignored test on the one box that could run it
//! is a test nobody runs.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine-cuda --features cuda-13 --test serve_smoke -- --nocapture
//! ```
//!
//! `PIE_SMOKE_CHECKPOINT` and `PIE_SMOKE_TOKENIZER` override where it looks.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this smoke serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt, and the reason it is this one: the answer is a single
/// well-known token, so a continuation that is merely fluent still fails.
const PROMPT: &str = "The capital of France is";

/// What a correct load produces here. OBSERVED, THEN PINNED — the first run
/// of this shell against this checkpoint answered ` Paris`, greedily, and
/// pinning it is what turns "the fire completed" into "the model computed".
const EXPECTED: &str = " Paris";

/// How many decode fires follow the prefill.
const STEPS: usize = 16;

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, because a vocabulary from another snapshot decodes the right ids into
/// the wrong words.
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

/// The container the contract is checked against — one file of the snapshot,
/// whichever one holds the tensors.
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

/// Greedy: the highest logit, and the row it came from must be readable.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// One prefill and `STEPS` decodes in slot 0, greedy throughout.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(0).expect("slot 0 opens");

    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    assert_eq!(prefill.len(), 1, "one lane in, one row of logits out");
    finite(&prefill[0], "prefill");

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
        finite(&decode[0], "decode");
        produced.push(argmax(&decode[0]));
    }
    (produced, millis)
}

/// The lane word the model's own `Classify` computes — runtime-side work, done
/// here because this test IS the runtime for the length of one fire.
fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    // A row that is entirely one value is finite and still wrong — an
    // untouched arena rectangle reads as zeros.
    let spread = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// ONE TEST AT A TIME, PER PROCESS — AND IT IS ABOUT VRAM NOW, NOT ABOUT
/// CORRECTNESS.
///
/// It used to be about correctness. `kernels-cuda`'s scratch slabs were
/// process-global and keyed by NAME (`Ctx::scratch`), and the dense
/// autotuner's cuBLASLt workspace was one buffer per device beside them — so
/// two shells firing at once on two streams staged into the same bytes and
/// both produced fluent garbage. Measured, not assumed: run these tests in
/// parallel and the continuation came back `"PPP is目前是. \{ a a \)"`;
/// serialize them and it was ` Paris`.
///
/// A slab is keyed by `(arena, name, stream)` now — one arena per CUDA
/// context, one slab per stream inside it — and the Lt workspace is one of
/// them, so two shells share nothing. [`two_shells_firing_at_once_say_what_each_says_alone`]
/// is that claim as a test rather than as a comment, and it deliberately does
/// NOT serialize its two shells.
///
/// What is left is arithmetic: every test here loads its own 1.7 GB of
/// weights, and four of them at once is four copies on one card. Held so that
/// a plain `cargo test` is green without an operator remembering
/// `--test-threads 1`.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Everything the three tests below share: a loaded shell and the vocabulary
/// that goes with it, or `None` and a sentence saying which of the machine,
/// the checkpoint and the tokenizer was missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
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

    // The runtime's half: trace the row, state the load contract. Neither is
    // the shell's — `Trace` crosses the boundary, `CompiledModel` never does.
    let trace = model::trace_of(SKU).expect("the catalog ships the smoke's SKU");
    let trace = trace(Platform::Cuda);
    // A stock safetensors snapshot, projected into the one object model the
    // contract algebra speaks. `Source::open` would refuse it — that door is
    // for a canonical `.zt`.
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    // The IMPORT contract, reached through the catalog's own table: it maps
    // the checkpoint's names onto the plan's, which is what a shell handed a
    // stock hugging-face snapshot needs. `Model::load` is the other door —
    // the same publication under the plan's names, read out of a canonical
    // `.zt` this deployment does not have.
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let booted = Instant::now();
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // Small on purpose: the arena reserves `max_tokens` rows of a
        // 248320-wide logit column, and this test needs a prompt, not a
        // batch.
        budget: Budget::new(4, 256),
        profile: None,
        page_size: 16,
        context: 512,
        // Four, because the mixed-fire golden below seats three sequences at
        // once (two mid-generation, one fresh) and re-opens one of them every
        // step.
        slots: 4,
        ordinal: 0,
        // The golden path. Every claim in this file is about what the model
        // says, and the recorded path is diffed against it in
        // `graph_replay.rs` rather than substituted for it here.
        graphs: engine_cuda::Graphs::Off,
        // F1's depth, kept: these gates fire one step at a time and
        // read its numbers, so a deeper ring would carve slots nothing
        // claims. `Runahead::of` is the door a deployment comes through.
        runahead: engine::runahead::Runahead::F1,
    })
    .expect("the shell loads");
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB",
        booted.elapsed().as_secs_f64(),
        weights as f64 / (1 << 30) as f64,
        arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
    );
    Some((shell, tokenizer))
}

#[test]
fn a_real_checkpoint_prefills_decodes_and_says_something_true() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the serve smoke") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");

    let (first, millis) = run(&mut shell, &prompt);
    let text = tokenizer.decode(&first, false);
    eprintln!("continuation: {text:?}");
    let warm = &millis[millis.len() / 2..];
    eprintln!(
        "decode: {:.2} ms/fire over the warm half of {STEPS} steps",
        warm.iter().sum::<f64>() / warm.len() as f64,
    );

    // 2. The same batch twice is the same tokens twice. A difference here is
    //    two arms writing one class's rows, which is what
    //    `resolve_classes` is for and what nothing else in this suite can
    //    observe.
    let (second, _) = run(&mut shell, &prompt);
    assert_eq!(
        first, second,
        "two identical runs produced different tokens: {:?} against {:?}",
        tokenizer.decode(&first, false),
        tokenizer.decode(&second, false),
    );

    // 3. And it has to be RIGHT. A load that ran is not a load that works.
    assert!(
        text.starts_with(EXPECTED),
        "greedy continuation of {PROMPT:?} began {text:?}, and this shell is \
         supposed to answer {EXPECTED:?}"
    );
}

/// One prefill and `steps` greedy decodes for a lane in `slot`, alone.
fn solo(shell: &mut Shell, slot: u32, prompt: &[u32], steps: usize) -> Vec<u32> {
    shell.open(slot).expect("the slot opens");
    let prefill = shell
        .fire(&[Lane {
            slot,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    let mut produced = vec![argmax(&prefill[0])];
    for _ in 0..steps {
        let fed = [*produced.last().expect("a step has a last token")];
        let decode = shell
            .fire(&[Lane {
                slot,
                word: word(1),
                tokens: &fed,
            }])
            .expect("the decode fires");
        produced.push(argmax(&decode[0]));
    }
    produced
}

/// Two sequences batched into one fire say what they say alone.
///
/// **THE PROPERTY BATCHING IS FOR, AND THE ONE IT BREAKS FIRST.** Two lanes
/// in a fire share every weight read, every arena rectangle and one page
/// table, and the whole of what keeps them apart is arithmetic: the seriated
/// row order, the per-lane page block, the indptr the ragged entries walk,
/// and the slot each recurrent bank is addressed by. Get any of those wrong
/// and both lanes still produce fluent text — one of them is just attending
/// the other's tokens. Running each alone and then together is what says so.
///
/// Lengths differ on purpose: equal-length lanes make an indptr that is right
/// for the wrong reason.
#[test]
fn two_lanes_batched_say_what_they_say_alone() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the batching smoke") else {
        return;
    };
    const STEPS: usize = 4;
    let first = tokenizer.encode("The capital of France is");
    let second = tokenizer.encode("Water freezes at a temperature of");
    assert_ne!(
        first.len(),
        second.len(),
        "two prompts of one length would test an indptr that is right by accident"
    );

    let alone = (
        solo(&mut shell, 0, &first, STEPS),
        solo(&mut shell, 1, &second, STEPS),
    );

    // Now together: one prefill fire over both prompts — legal, because both
    // lanes are ¬qo_one and so fall in one class — then one decode fire per
    // step over both continuations.
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let prefill = shell
        .fire(&[
            Lane {
                slot: 0,
                word: word(first.len() as u32),
                tokens: &first,
            },
            Lane {
                slot: 1,
                word: word(second.len() as u32),
                tokens: &second,
            },
        ])
        .expect("a two-lane prefill fires");
    assert_eq!(prefill.len(), 2, "two lanes in, two rows of logits out");
    let mut together = (vec![argmax(&prefill[0])], vec![argmax(&prefill[1])]);
    for _ in 0..STEPS {
        let fed = (
            [*together.0.last().expect("a step has a last token")],
            [*together.1.last().expect("a step has a last token")],
        );
        let decode = shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &fed.0,
                },
                Lane {
                    slot: 1,
                    word: word(1),
                    tokens: &fed.1,
                },
            ])
            .expect("a two-lane decode fires");
        together.0.push(argmax(&decode[0]));
        together.1.push(argmax(&decode[1]));
    }

    eprintln!(
        "batched: {:?} / {:?}",
        tokenizer.decode(&together.0, false),
        tokenizer.decode(&together.1, false),
    );
    assert_eq!(
        alone.0, together.0,
        "lane 0 said {:?} alone and {:?} in a batch",
        tokenizer.decode(&alone.0, false),
        tokenizer.decode(&together.0, false),
    );
    assert_eq!(
        alone.1, together.1,
        "lane 1 said {:?} alone and {:?} in a batch",
        tokenizer.decode(&alone.1, false),
        tokenizer.decode(&together.1, false),
    );
}

/// Design §0's headline case, and the reason this system exists: ONE fire
/// carrying a decode lane and a prefill lane, over disjoint row windows.
///
/// **THE PROPERTY IS TOKEN-FOR-TOKEN IDENTITY WITH SOLO FIRES.** Two lanes of
/// the same class already share every weight read and one page table
/// (`two_lanes_batched_say_what_they_say_alone` above); what a MIXED fire adds
/// is that the two lanes run different KERNELS over different rows of the same
/// rectangles — decode attention over `[10,11)` while prefill attention runs
/// over `[0,10)`, one arena column, one merge — and every shared op after
/// them reads the union. Get a window bound wrong and both lanes still produce
/// fluent text: the decode lane is simply attending the prefill lane's rows.
/// So each lane is run alone first, and the batch has to say the same thing.
///
/// It repeats, because the two halves fail differently. A fresh prompt is
/// prefilled in a second slot on EVERY step, beside a decode lane that is
/// carrying its own continuation forward — so prefill-in-a-mixed-fire happens
/// as often as decode-in-a-mixed-fire, and a drift that only shows on the
/// second occurrence has somewhere to show.
#[test]
fn a_fire_that_mixes_prefill_and_decode_says_what_each_lane_says_alone() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the mixed-fire golden") else {
        return;
    };
    const STEPS: usize = 6;
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");
    assert_ne!(
        carried.len(),
        fresh.len(),
        "two prompts of one length would test a window that is right by accident"
    );

    // The goldens: each lane alone, at the same point in its own life.
    let alone_decode = solo(&mut shell, 0, &carried, STEPS);
    let alone_prefill = solo(&mut shell, 1, &fresh, 0);

    // Seat the carried sequence with one solo prefill, so every fire below
    // has a real mid-generation decode lane in it.
    shell.open(0).expect("slot 0 opens");
    let seated = shell
        .fire(&[Lane {
            slot: 0,
            word: word(carried.len() as u32),
            tokens: &carried,
        }])
        .expect("the carried prefill fires");
    let mut mixed = vec![argmax(&seated[0])];

    for step in 0..STEPS {
        shell.open(1).expect("slot 1 opens");
        let fed = [*mixed.last().expect("a step has a last token")];
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
                    tokens: &fresh,
                },
            ])
            .unwrap_or_else(|why| panic!("the mixed fire at step {step} fires: {why}"));
        assert_eq!(out.len(), 2, "two lanes in, two rows of logits out");
        finite(&out[0], "the decode lane of a mixed fire");
        finite(&out[1], "the prefill lane of a mixed fire");
        assert_eq!(
            argmax(&out[1]),
            alone_prefill[0],
            "the prefill lane of the mixed fire at step {step} said {:?}, and alone \
             it says {:?}",
            tokenizer.decode(&[argmax(&out[1])], false),
            tokenizer.decode(&alone_prefill, false),
        );
        mixed.push(argmax(&out[0]));
    }

    eprintln!(
        "mixed: decode lane {:?}, prefill lane {:?}",
        tokenizer.decode(&mixed, false),
        tokenizer.decode(&alone_prefill, false),
    );
    assert_eq!(
        alone_decode, mixed,
        "the decode lane said {:?} alone and {:?} beside a prefill",
        tokenizer.decode(&alone_decode, false),
        tokenizer.decode(&mixed, false),
    );
}

/// The same claim one lane wider: two decode lanes and a prefill lane, so the
/// decode window holds more than one request and the seriation has to place
/// both of them before the prefill's rows (or both after).
#[test]
fn three_lanes_two_decoding_and_one_prefilling_agree_with_their_solo_runs() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the three-lane mixed fire") else {
        return;
    };
    const STEPS: usize = 4;
    let first = tokenizer.encode(PROMPT);
    let second = tokenizer.encode("The largest planet in the solar system is");
    let fresh = tokenizer.encode("Water freezes at a temperature of");

    let alone = (
        solo(&mut shell, 0, &first, STEPS),
        solo(&mut shell, 1, &second, STEPS),
        solo(&mut shell, 2, &fresh, 0),
    );

    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let seated = shell
        .fire(&[
            Lane {
                slot: 0,
                word: word(first.len() as u32),
                tokens: &first,
            },
            Lane {
                slot: 1,
                word: word(second.len() as u32),
                tokens: &second,
            },
        ])
        .expect("the two carried prefills fire");
    let mut together = (vec![argmax(&seated[0])], vec![argmax(&seated[1])]);

    for step in 0..STEPS {
        shell.open(2).expect("slot 2 opens");
        let fed = (
            [*together.0.last().expect("a step has a last token")],
            [*together.1.last().expect("a step has a last token")],
        );
        let out = shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &fed.0,
                },
                Lane {
                    slot: 2,
                    word: word(fresh.len() as u32),
                    tokens: &fresh,
                },
                Lane {
                    slot: 1,
                    word: word(1),
                    tokens: &fed.1,
                },
            ])
            .unwrap_or_else(|why| panic!("the three-lane fire at step {step} fires: {why}"));
        assert_eq!(out.len(), 3, "three lanes in, three rows of logits out");
        assert_eq!(
            argmax(&out[1]),
            alone.2[0],
            "the prefill lane of the three-lane fire at step {step} disagreed with \
             its solo run"
        );
        together.0.push(argmax(&out[0]));
        together.1.push(argmax(&out[2]));
    }

    assert_eq!(
        alone.0, together.0,
        "decode lane 0 said {:?} alone and {:?} in a mixed fire",
        tokenizer.decode(&alone.0, false),
        tokenizer.decode(&together.0, false),
    );
    assert_eq!(
        alone.1, together.1,
        "decode lane 1 said {:?} alone and {:?} in a mixed fire",
        tokenizer.decode(&alone.1, false),
        tokenizer.decode(&together.1, false),
    );
}

/// **TWO SHELLS IN ONE PROCESS, FIRING AT THE SAME INSTANT, AND NEITHER SAYS
/// THE OTHER'S WORDS.**
///
/// This test could not be written before. Build log 18 recorded "one shell
/// per process (kernels-cuda scratch slabs are process-global — measured,
/// documented in `serve.rs`)" as a standing property of the plane, and
/// [`ONE_AT_A_TIME`] above is the workaround it forced on every suite in the
/// tree. The slabs are per `(arena, name, stream)` now and the arena is the
/// CUDA context, so the property is supposed to be gone — and the only way to
/// say that is to break the rule on purpose.
///
/// **TWO THREADS, NOT TWO INTERLEAVED CALLS.** A shell synchronizes its
/// stream at the end of every fire, so two shells driven in turn from one
/// thread would never have two launches in flight and a shared slab would
/// never be observed. The garbling needs real overlap, which needs two
/// threads — and a `Context` binds the thread that will fire it
/// (`cudaSetDevice` is per-thread), so each thread loads its own.
///
/// **THE GOLDEN IS THE SAME PROCESS, NOT A PINNED STRING.** Both prompts are
/// run first on one shell, alone, and the two concurrent continuations are
/// diffed against those — so a machine, a driver version or a checkpoint that
/// answers something else still tests the property this is for. The two
/// prompts DIFFER, because two shells computing the same continuation would
/// agree whether or not they were staging over each other.
#[test]
fn two_shells_firing_at_once_say_what_each_says_alone() {
    // Held against the OTHER tests in this file — four shells at once is four
    // copies of the weights — and released inside for the two of its own.
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the two-shell gate") else {
        return;
    };
    const STEPS: usize = 12;
    let first = tokenizer.encode(PROMPT);
    let second = tokenizer.encode("The largest planet in the solar system is");
    assert_ne!(first, second, "two shells saying one thing prove nothing");

    let alone = (
        solo(&mut shell, 0, &first, STEPS),
        solo(&mut shell, 1, &second, STEPS),
    );
    // The golden's shell goes away before the two under test load, so the
    // card holds two sets of weights rather than three.
    drop(shell);

    let gate = std::sync::Barrier::new(2);
    let arm = |what: &'static str, prompt: &[u32]| {
        let Some((mut shell, _)) = ready(what) else {
            return None;
        };
        // Both shells are loaded and warm before either fires, so the
        // overlap is the fires and not the loads.
        gate.wait();
        Some(solo(&mut shell, 0, prompt, STEPS))
    };

    let (left, right) = std::thread::scope(|scope| {
        let a = scope.spawn(|| arm("the two-shell gate's first shell", &first));
        let b = scope.spawn(|| arm("the two-shell gate's second shell", &second));
        (
            a.join().expect("the first shell's thread finishes"),
            b.join().expect("the second shell's thread finishes"),
        )
    });
    let (Some(left), Some(right)) = (left, right) else {
        eprintln!("skipping the two-shell gate: a second shell would not load");
        return;
    };

    eprintln!(
        "concurrent: {:?} / {:?}",
        tokenizer.decode(&left, false),
        tokenizer.decode(&right, false),
    );
    assert_eq!(
        alone.0, left,
        "the first shell said {:?} alone and {:?} beside a second shell",
        tokenizer.decode(&alone.0, false),
        tokenizer.decode(&left, false),
    );
    assert_eq!(
        alone.1, right,
        "the second shell said {:?} alone and {:?} beside a first shell",
        tokenizer.decode(&alone.1, false),
        tokenizer.decode(&right, false),
    );
}
