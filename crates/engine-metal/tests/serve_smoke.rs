//! The end-to-end smoke, on an Apple GPU: a real checkpoint, a real prompt,
//! real tokens back.
//!
//! **WHAT IT IS FOR.** Every layer under it is checked on its own — the fire
//! substrate has its own tests with a mock dispatch, the compiler its own,
//! the page arithmetic its own, and `device_floor` proves the shaders
//! compile and one dispatch computes what it says — and none of that can
//! tell you the model says anything. A shell can bind every seat, refuse
//! nothing, launch every kernel and produce fluent-looking garbage. So this
//! file asserts the one property no unit test reaches: that the
//! continuation is *right*.
//!
//! Three claims, in the order they can fail:
//!
//! 1. **finite** — no NaN, no infinity, and not a rectangle nothing wrote. A
//!    pool read at the wrong stride or a state slab of poison shows up here.
//! 2. **deterministic** — two identical runs produce identical tokens.
//! 3. **coherent** — the continuation of "The capital of France is" begins
//!    with " Paris".
//!
//! Beside those: two lanes of one class say what they say alone, design §0's
//! headline case (a decode lane BESIDE a prefill lane) says what each says
//! alone, and a slot reused for a second sequence through one boot does not
//! continue the first — the recurrent clear at held zero, palo build log 19,
//! which this shell inherits verbatim because it inherits the banks.
//!
//! # Gating
//!
//! An Apple target is not a machine with a GPU, and neither is a machine
//! with a GPU one with a 1.4 GB checkpoint on its disk. So the file is
//! `cfg`'d to Apple and SKIPS at run time, saying which of the three was
//! missing, rather than being `#[ignore]`d — an ignored test on the one box
//! that could run it is a test nobody runs.
//!
//! ```text
//! cargo test -p engine-metal --release --test serve_smoke -- --nocapture
//! ```
//!
//! `PIE_SMOKE_SNAPSHOT` overrides where it looks.

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this smoke serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt, and the reason it is this one: the answer is a single
/// well-known token, so a continuation that is merely fluent still fails.
const PROMPT: &str = "The capital of France is";

/// What a correct load produces here. The CUDA shell answered ` Paris` to
/// this prompt against this checkpoint; a second backend answering the same
/// thing is the claim, not a coincidence to be re-pinned.
const EXPECTED: &str = " Paris";

/// How many decode fires follow the prefill.
const STEPS: usize = 16;

/// **ONE SHELL AT A TIME, PER PROCESS.** Not for the reason the CUDA
/// sibling's twin gives — `kernels-metal` allocates nothing and keeps no
/// process-global scratch, so two metal shells firing at once would be
/// correct — but because these tests each hold ~1.5 GiB resident on a 32 GiB
/// unified machine and the MEASUREMENTS are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, because a vocabulary from another snapshot decodes the right ids into
/// the wrong words.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    // The suite runs as root over tailscale ssh, so `HOME` is not the
    // owner's — the cache the checkpoint actually lives in is named
    // explicitly beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots =
            Path::new(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    })
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

/// Greedy: the highest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The lane word the model's own `Classify` computes — runtime-side work,
/// done here because this test IS the runtime for the length of one fire.
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
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// One prefill and `STEPS` decodes in one slot, greedy throughout.
fn run(shell: &mut Shell, slot: u32, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(slot).expect("the slot opens");

    let prefill = shell
        .fire(&[Lane {
            slot,
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
                slot,
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

/// Everything the tests below share: a loaded shell and the vocabulary that
/// goes with it, or `None` and a sentence saying which of the machine, the
/// checkpoint and the tokenizer was missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
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
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let booted = Instant::now();
    let shell = Shell::load(Boot {
        plan,
        contract: &contract,
        checkpoint: &checkpoint,
        // Small on purpose: the arena reserves `max_tokens` rows of a
        // 248320-wide logit column, and this test needs a prompt, not a
        // batch. Sized for a 32 GiB unified machine.
        budget: Budget::new(4, 256),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
    })
    .expect("the shell loads");
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded on {} in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB",
        shell.device_name(),
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
    let (produced, millis) = run(&mut shell, 0, &prompt);
    let text = tokenizer.decode(&produced, false);
    let warm = &millis[millis.len() / 2..];
    eprintln!(
        "{PROMPT:?} -> {text:?}  |  {:.2} ms/fire warm, {} shader points compiled",
        warm.iter().sum::<f64>() / warm.len() as f64,
        shell.compiled()
    );
    assert!(
        text.starts_with(EXPECTED),
        "the continuation is {text:?}, and a correct load begins it {EXPECTED:?}"
    );
}

#[test]
fn two_identical_runs_produce_identical_tokens() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the determinism gate") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let (first, _) = run(&mut shell, 0, &prompt);
    let (second, _) = run(&mut shell, 0, &prompt);
    assert_eq!(
        first, second,
        "two identical fires answered differently, which means two arms wrote the same rows"
    );
}

/// **DESIGN §0's HEADLINE CASE.** A decode lane beside a prefill lane, in
/// one fire, over one artifact — and each says token for token what it says
/// alone. This is the whole point of the window mechanism: the two lanes
/// share every arena column and read disjoint row intervals of it.
#[test]
fn a_decode_lane_beside_a_prefill_lane_says_what_each_says_alone() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the mixed-fire golden") else {
        return;
    };
    let a = tokenizer.encode(PROMPT);
    let b = tokenizer.encode("The largest planet in our solar system is");

    // Solo: lane A is mid-generation (prefilled, then decoding one token a
    // fire); lane B prefills fresh every round.
    shell.open(0).expect("slot 0 opens");
    let seed = shell
        .fire(&[Lane {
            slot: 0,
            word: word(a.len() as u32),
            tokens: &a,
        }])
        .expect("lane a prefills");
    let mut solo_a = vec![argmax(&seed[0])];
    for _ in 0..4 {
        let fed = [*solo_a.last().expect("a step feeds back")];
        let step = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .expect("lane a decodes");
        solo_a.push(argmax(&step[0]));
    }
    shell.open(1).expect("slot 1 opens");
    let solo_b = {
        let fired = shell
            .fire(&[Lane {
                slot: 1,
                word: word(b.len() as u32),
                tokens: &b,
            }])
            .expect("lane b prefills");
        argmax(&fired[0])
    };

    // Mixed: the same lane A steps, with lane B's prefill beside it.
    shell.open(0).expect("slot 0 re-opens");
    let seed = shell
        .fire(&[Lane {
            slot: 0,
            word: word(a.len() as u32),
            tokens: &a,
        }])
        .expect("lane a prefills again");
    let mut mixed_a = vec![argmax(&seed[0])];
    let mut mixed_b = Vec::new();
    for _ in 0..4 {
        let fed = [*mixed_a.last().expect("a step feeds back")];
        shell.open(1).expect("slot 1 re-opens");
        let fired = shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &fed,
                },
                Lane {
                    slot: 1,
                    word: word(b.len() as u32),
                    tokens: &b,
                },
            ])
            .expect("the mixed fire fires");
        mixed_a.push(argmax(&fired[0]));
        mixed_b.push(argmax(&fired[1]));
    }
    eprintln!(
        "solo a {solo_a:?} | mixed a {mixed_a:?}\nsolo b {solo_b} | mixed b {mixed_b:?}"
    );
    assert_eq!(
        solo_a, mixed_a,
        "the decode lane answered differently with a prefill lane beside it"
    );
    assert!(
        mixed_b.iter().all(|&token| token == solo_b),
        "the prefill lane answered differently with a decode lane beside it"
    );
}

/// **LAUNCH ISOLATION: A SLOT REUSED IS A SEQUENCE BEGUN** (palo build log
/// 19). The recurrent banks are zeroed at load and never again unless
/// something clears them; a linear-attention scan reads its whole state on
/// its first step, so a slot still holding the last sequence's history would
/// continue it. The kv half is safe on its own — `kv_len` bounds what is
/// read — so this is the only observable of the clear, and only a SECOND
/// sequence through one boot can see it.
#[test]
fn a_slot_reused_through_one_boot_does_not_continue_the_last_sequence() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("launch isolation") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let mut said = Vec::new();
    for round in 0..3 {
        // Deliberately a DIFFERENT sequence in between, so the bank holds
        // somebody else's history if the clear does not happen.
        let other = tokenizer.encode("Water boils at");
        let (interference, _) = run(&mut shell, 1, &other);
        assert!(!interference.is_empty());

        let (produced, _) = run(&mut shell, 0, &prompt);
        let text = tokenizer.decode(&produced, false);
        eprintln!("round {round}: {text:?}");
        said.push(text);
    }
    assert!(
        said.windows(2).all(|pair| pair[0] == pair[1]),
        "three sequences through one boot answered differently: {said:?}"
    );
    assert!(
        said[0].starts_with(EXPECTED),
        "and the answer is {:?}, not {EXPECTED:?}",
        said[0]
    );
}

/// A lane whose word puts it in no class this artifact bakes, and a mask
/// this plane does not stage: both refused by name, before anything
/// launches.
#[test]
fn a_mask_this_plane_does_not_stage_is_refused_by_name() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the mask refusal") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    shell.open(0).expect("slot 0 opens");
    let mask = engine::engine_api::fire::Mask {
        runs: vec![0, prompt.len() as u32],
        total: prompt.len() as u64,
    };
    let fault = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 0,
                word: word(prompt.len() as u32),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: Some(&mask),
        }])
        .expect_err("a mask this plane stages no bits for");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("mask"),
        "the refusal names the mask: {said}"
    );
}
