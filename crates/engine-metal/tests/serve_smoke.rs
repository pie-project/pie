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

use engine_metal::{AdapterPlane, Boot, Lane, Seated, Shell};
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
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// The same word, with the `masked` fact set — which is what puts a lane in
/// the class whose window runs `attention.masked`.
fn masked_word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, true)).word()
}

/// The same word with the `has_adapter` fact set — which is what puts a lane
/// in a class whose window runs `linear.lora_correct` (design §8, fact bit 1).
fn adapted_word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false).adapted(true)).word()
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
    let trace = models::trace_of(SKU).expect("the catalog ships the smoke's SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
            .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let booted = Instant::now();
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // §M-4c: this gate lands a raw snapshot, which carries no
        // `pie.serving/1` stamp and proceeds — but the deployment's own facts
        // are stated honestly all the same, because an empty precision means
        // "the runtime never assembled them" and is refused rather than
        // skipped (`weights::serves_this_deployment`).
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        // Small on purpose: the arena reserves `max_tokens` rows of a
        // 248320-wide logit column, and this test needs a prompt, not a
        // batch. Sized for a 32 GiB unified machine.
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        // F1's depth: one step in flight, one A/B seat set. Stated rather
        // than defaulted because these are goldens — the eager shell is what
        // a byte-identity arm compares against — and because a second seat
        // set is a second whole `Inputs` reservation on a machine this test
        // is already sized carefully for.
        runahead: engine::runahead::Runahead::F1,
        // Full residency: the whole weight table on the device, no
        // wired-slab tier, no segment cuts — the load every gate in
        // this directory measures.
        residency: engine_metal::ResidencyPlan::default(),
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

/// **A MASK WHOSE WORD SAYS THE OTHER THING IS REFUSED BY NAME.** The word
/// is what puts a lane in a class and the class is what decides whether an
/// `attention.masked` arm runs over its rows, so a mask beside an unmasked
/// word is a rectangle nothing would read — and a masked word beside no mask
/// is a masked arm reading a plane no lane filled. Both directions are a
/// wrong answer that computes, so both are refused.
#[test]
fn a_mask_the_word_does_not_admit_is_refused_by_name() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the mask/word disagreement") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    shell.open(0).expect("slot 0 opens");
    let mask = engine::fire::Masking::Extent(engine::fire::Mask {
        runs: vec![0, prompt.len() as u32],
        total: prompt.len() as u64,
    });
    let fault = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 0,
                // The UNMASKED word, beside a mask.
                word: word(prompt.len() as u32),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: Some(&mask),
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
        }])
        .expect_err("a mask beside a word that skips the masked arm");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("mask"),
        "the refusal names the mask: {said}"
    );
}

/// **AN ALL-KEEPING MASK IS THE UNMASKED ANSWER, AND THAT IS THE ONLY CHECK
/// THAT SEPARATES A STAGED PLANE FROM A BLANK ONE.** A mask of one kept run
/// over the whole extent restricts nothing, so the masked arm must answer
/// what the plain prefill answered — byte for byte at the argmax. A plane of
/// zeros would blank every logit; a plane addressed by the wrong row would
/// answer some other row's sentence; both are finite, and neither survives
/// this comparison.
#[test]
fn an_all_keeping_mask_answers_what_the_unmasked_prefill_did() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the all-keeping mask") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;

    shell.open(0).expect("slot 0 opens");
    let plain = shell
        .fire(&[Lane {
            slot: 0,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the unmasked prefill fires");
    finite(&plain[0], "unmasked prefill");

    shell.open(1).expect("slot 1 opens");
    let keep_all = engine::fire::Masking::Extent(engine::fire::Mask {
        runs: vec![0, rows],
        total: u64::from(rows),
    });
    let masked = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 1,
                word: masked_word(rows),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: Some(&keep_all),
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
        }])
        .expect("the masked prefill fires");
    finite(&masked[0], "masked prefill");
    assert_eq!(
        argmax(&masked[0]),
        argmax(&plain[0]),
        "an all-keeping mask changed the answer, so the staged plane is not \
         the identity it says it is"
    );
}

/// **THE PER-ROW FORM STAGES, AND ITS ALL-KEEPING SPELLING IS THE SAME
/// IDENTITY.** One `Masking::Rows` per query row, each keeping the whole
/// extent, is the extent form written out row by row — so it must answer the
/// same thing, and a row-indexing mistake in the per-row walk is exactly what
/// would show up here and nowhere else.
#[test]
fn a_per_row_mask_that_keeps_everything_is_the_same_identity() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the per-row mask") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;

    shell.open(0).expect("slot 0 opens");
    let plain = shell
        .fire(&[Lane {
            slot: 0,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the unmasked prefill fires");

    shell.open(1).expect("slot 1 opens");
    let per_row = engine::fire::Masking::Rows(
        (0..rows)
            .map(|_| engine::fire::Mask {
                runs: vec![0, rows],
                total: u64::from(rows),
            })
            .collect(),
    );
    let masked = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 1,
                word: masked_word(rows),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: Some(&per_row),
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
        }])
        .expect("the per-row masked prefill fires");
    finite(&masked[0], "per-row masked prefill");
    assert_eq!(
        argmax(&masked[0]),
        argmax(&plain[0]),
        "a per-row mask that keeps everything changed the answer"
    );
}

/// **A PER-ROW MASK OF THE WRONG HEIGHT IS REFUSED BY NAME**, because
/// `Masking::Rows` is parallel to the lane's tokens and a vector of some
/// other length has no reading at all.
#[test]
fn a_per_row_mask_of_the_wrong_height_is_refused_by_name() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the per-row mask's height") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;
    shell.open(0).expect("slot 0 opens");
    let short = engine::fire::Masking::Rows(
        (0..rows - 1)
            .map(|_| engine::fire::Mask {
                runs: vec![0, rows],
                total: u64::from(rows),
            })
            .collect(),
    );
    let fault = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 0,
                word: masked_word(rows),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: Some(&short),
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
        }])
        .expect_err("a per-row mask one row short of its lane");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("Masking::Rows"),
        "the refusal names the form whose height is wrong: {said}"
    );
}

/// **STATING THE POSITIONS THE SHELL WOULD HAVE DERIVED CHANGES NOTHING.**
/// The derived run is `held .. held + rows`; a caller that writes it out must
/// get the same answer, which is what says the stated vector reaches rope's
/// seat rather than some other one — and a run of the wrong height is refused
/// by name rather than padded.
#[test]
fn stated_positions_that_are_the_derived_run_change_nothing() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("explicit positions") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;

    shell.open(0).expect("slot 0 opens");
    let derived = shell
        .fire(&[Lane {
            slot: 0,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the derived-position prefill fires");
    finite(&derived[0], "derived positions");

    shell.open(1).expect("slot 1 opens");
    let stated: Vec<u32> = (0..rows).collect();
    let restated = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 1,
                word: word(rows),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: None,
            adapter: None,
            positions: &stated,
            readout: None,
            captures_scores: false,
            translation: &[],
        }])
        .expect("the stated-position prefill fires");
    assert_eq!(
        argmax(&restated[0]),
        argmax(&derived[0]),
        "writing out the run the shell derives changed the answer"
    );

    shell.open(2).expect("slot 2 opens");
    let short: Vec<u32> = (0..rows - 1).collect();
    let fault = shell
        .fire_seated(&[Seated {
            lane: Lane {
                slot: 2,
                word: word(rows),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: None,
            adapter: None,
            positions: &short,
            readout: None,
            captures_scores: false,
            translation: &[],
        }])
        .expect_err("a position run one short of its lane");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("position"),
        "the refusal names the positions: {said}"
    );
}

/// **A ROUTED LANE AGAINST ZEROED BANKS IS THE IDENTITY, BYTE FOR BYTE.**
///
/// A bank is a `ParamSource::Registered` param: `weights.rs` reserves it and
/// zeroes it at load, and nothing has registered into it here. So
/// `y += B[0]·(A[0]·x)` adds exactly zero — a zero is exact in bf16 and in
/// f32, so the sum is the addend and the comparison is equality and not a
/// tolerance. Registration is the ONLY thing that may change a number on this
/// axis, and this is the test that says so.
///
/// The two lanes run the same attention arm — an adapted prefill is still
/// `qo_one = false`, `masked = false`, so the top-level split
/// `[masked, captures_scores, qo_one, rest]` puts both in `rest` — and the
/// whole difference between them is the correction region the adapter word
/// opens. A plane addressed at the wrong row, a routes vector nobody wrote,
/// or a bank whose residency came up as garbage all move these numbers.
#[test]
fn a_routed_lane_against_zeroed_banks_answers_what_the_base_model_did() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the zeroed-bank identity") else {
        return;
    };
    // The banks the model text declared, which is what makes the claim above
    // a claim about something: a load with no bank routes nowhere.
    let banks = shell.banks();
    eprintln!("banks: {banks:?}");
    assert!(
        !banks.is_empty(),
        "this SKU's text declares adapter banks, and the identity claim needs one"
    );

    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;

    shell.open(0).expect("slot 0 opens");
    let plain = shell
        .fire(&[Lane {
            slot: 0,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the unrouted prefill fires");
    finite(&plain[0], "unrouted prefill");

    shell.open(1).expect("slot 1 opens");
    let routed = shell
        .fire_seated(&[Seated::adapted(
            Lane {
                slot: 1,
                word: adapted_word(rows),
                tokens: &prompt,
            },
            0,
        )])
        .expect("the routed prefill fires");
    finite(&routed[0], "routed prefill");
    assert_eq!(
        routed[0], plain[0],
        "a zeroed bank moved the logits, so the correction is adding something \
         other than zero"
    );
}

/// **AN UNROUTED LANE BESIDE A ROUTED ONE IS THE `-1` SENTINEL, AND IT IS
/// STILL THE BASE MODEL.** This is the only way the sentinel is reachable
/// through the door: `adapter_routes` is staged whenever ANY lane names an
/// adapter, and an unrouted lane's rows contribute `-1` — the kernel's own
/// floor, which returns on that row before it reads a bank.
///
/// It is also the mixed-fire claim design §0 exists for, on this axis: two
/// lanes of two classes in one fire, each answering what it answers alone.
#[test]
fn an_unrouted_lane_beside_a_routed_one_answers_what_it_answers_alone() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the `-1` sentinel") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;

    shell.open(0).expect("slot 0 opens");
    let alone = shell
        .fire(&[Lane {
            slot: 0,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the lane fires alone");
    finite(&alone[0], "the lane alone");

    shell.open(1).expect("slot 1 opens");
    shell.open(2).expect("slot 2 opens");
    let together = shell
        .fire_seated(&[
            Seated::of(Lane {
                slot: 1,
                word: word(rows),
                tokens: &prompt,
            }),
            Seated::adapted(
                Lane {
                    slot: 2,
                    word: adapted_word(rows),
                    tokens: &prompt,
                },
                0,
            ),
        ])
        .expect("the mixed fire fires");
    assert_eq!(together.len(), 2, "two lanes in, two rows of logits out");
    finite(&together[0], "the unrouted lane of the mixed fire");
    finite(&together[1], "the routed lane of the mixed fire");
    // EXACT WITHIN THE FIRE, because both rows are cut from one rectangle by
    // one set of launches: the only thing between them is that one carried
    // `-1` into the correction and the other carried `0` into a zeroed bank,
    // and both are the base model exactly.
    assert_eq!(
        together[0], together[1],
        "the `-1` row and the zeroed-bank row disagree, so one of them read a \
         bank it should not have"
    );
    // ARGMAX ACROSS THE FIRES, which is the strength the mixed-fire golden
    // above claims for the same reason: a two-lane fire is a different
    // rectangle and its launches tile it differently, so bit equality with a
    // solo fire is not a property this shell promises.
    assert_eq!(
        argmax(&together[0]),
        argmax(&alone[0]),
        "the unrouted lane answered differently with a routed lane beside it"
    );
    assert_eq!(
        argmax(&together[1]),
        argmax(&alone[0]),
        "the routed lane against a zeroed bank answered something other than \
         the base model"
    );
}

/// **AN ADAPTER WHOSE WORD SAYS THE OTHER THING IS REFUSED BY NAME, IN BOTH
/// DIRECTIONS.** The word is what puts a lane in a class and the class is
/// what decides whether `linear.lora_correct` runs over its rows. An id
/// beside an unadapted word would be staged and never read — the lane answers
/// the base model under an adapter's name — and an adapted word beside no id
/// would send the arm at a routes vector this fire never staged. Both compute
/// a wrong answer, so both are refused.
#[test]
fn an_adapter_the_word_does_not_admit_is_refused_by_name() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the adapter/word disagreement") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;
    shell.open(0).expect("slot 0 opens");

    let staged_never_read = shell
        .fire_seated(&[Seated::adapted(
            Lane {
                slot: 0,
                // The UNADAPTED word, beside an id.
                word: word(rows),
                tokens: &prompt,
            },
            0,
        )])
        .expect_err("an id beside a word that skips the correction");
    let said = staged_never_read.to_string();
    eprintln!("refusal: {said}");
    assert!(said.contains("adapter"), "the refusal names the adapter: {said}");

    let named_none = shell
        .fire_seated(&[Seated::of(Lane {
            slot: 0,
            // The ADAPTED word, and no id to route with.
            word: adapted_word(rows),
            tokens: &prompt,
        })])
        .expect_err("an adapted word beside no id");
    let said = named_none.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("lora_correct"),
        "the refusal names the arm the word asked for: {said}"
    );
}

/// **A REGISTRATION THE BANKS CANNOT SEAT IS REFUSED BEFORE ANYTHING IS
/// WRITTEN**, and it names the bank and both numbers. The check is whole
/// before the first `memcpy`, so a refused registration leaves every bank
/// holding what it held — which the identity fire after it asserts.
#[test]
fn a_registration_the_banks_cannot_seat_is_refused_by_name() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the registration refusals") else {
        return;
    };
    let banks = shell.banks();
    let Some(&(name, adapters, slot)) = banks.first() else {
        panic!("this SKU's text declares adapter banks");
    };
    let name = name.to_string();
    eprintln!("bank {name:?}: {adapters} adapters of {slot} bytes");

    let whole = vec![0u8; slot as usize];
    let said = shell
        .register_adapter(
            adapters,
            &[AdapterPlane {
                bank: &name,
                bytes: &whole,
            }],
        )
        .expect_err("an id at the capacity, which is one past the last")
        .to_string();
    eprintln!("refusal: {said}");
    assert!(said.contains(&name) && said.contains("capacity"), "{said}");

    let short = vec![0u8; slot as usize - 1];
    let said = shell
        .register_adapter(
            0,
            &[AdapterPlane {
                bank: &name,
                bytes: &short,
            }],
        )
        .expect_err("a plane one byte short of its slot")
        .to_string();
    eprintln!("refusal: {said}");
    assert!(said.contains(&name) && said.contains("one whole slot"), "{said}");

    let said = shell
        .register_adapter(
            0,
            &[AdapterPlane {
                bank: "a_bank_no_model_text_declares",
                bytes: &whole,
            }],
        )
        .expect_err("a bank this plan never declared")
        .to_string();
    eprintln!("refusal: {said}");
    assert!(said.contains("a_bank_no_model_text_declares"), "{said}");

    // AND NOTHING WAS WRITTEN. A registration of zeros into slot 0 would be
    // the identity too, so this asserts the refusals left the banks alone by
    // asserting the fire they were refused around is still the base model.
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;
    shell.open(0).expect("slot 0 opens");
    let plain = shell
        .fire(&[Lane {
            slot: 0,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the unrouted prefill fires");
    shell.open(1).expect("slot 1 opens");
    let routed = shell
        .fire_seated(&[Seated::adapted(
            Lane {
                slot: 1,
                word: adapted_word(rows),
                tokens: &prompt,
            },
            0,
        )])
        .expect("the routed prefill fires");
    assert_eq!(
        routed[0], plain[0],
        "a refused registration wrote into a bank anyway"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The shared-adapter mount, on the device (alto adapter §3.3)
// ─────────────────────────────────────────────────────────────────────────────

/// This test's own mount, unique per process and per nanosecond.
fn scratch(what: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    let at =
        std::env::temp_dir().join(format!("pie-metal-mount-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&at).expect("a scratch directory");
    at
}

/// f32 to bf16, round-to-nearest-even — the loader's own conversion, stated
/// here because a truncating fixture would be writing a slightly different
/// adapter than the one it describes.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// What the load's banks say one role's `[layers, rank, hidden]` source is.
fn geometry(seats: &[engine_metal::BankSeat], role: &str) -> (u64, u64, u64) {
    let banks: Vec<&engine_metal::BankSeat> = seats
        .iter()
        .filter(|seat| engine_metal::adapter::role_of(&seat.name) == role)
        .collect();
    let seat = banks.first().expect("this SKU declares this role's banks");
    (
        banks.len() as u64,
        seat.rows.min(seat.cols),
        seat.rows.max(seat.cols),
    )
}

/// **WRITE ONE ADAPTER DIRECTORY INTO THE MOUNT.**
///
/// A manifest and two plane files, in the orientations §6.3's statute fixes:
/// `A` rank-major `[layers, rank, hidden]`, `B` out-major
/// `[layers, hidden, rank]`. The bytes are the bank's own dtype, because a
/// blob is prepared once by an operator rather than seeded live by a guest.
fn write_adapter(mount: &Path, name: &str, seats: &[engine_metal::BankSeat], amplitude: f32) {
    let dir = mount.join(name);
    std::fs::create_dir_all(&dir).expect("an adapter directory");
    let (layers, rank, hidden) = geometry(seats, "lora_a");
    let plane = |salt: u32, amp: f32, count: u64| -> Vec<u8> {
        (0..count)
            .flat_map(|at| {
                let mixed = (at as u32).wrapping_mul(2_654_435_761).wrapping_add(salt);
                let value = ((mixed % 2_000) as f32 / 1_000.0 - 1.0) * amp;
                bf16_bits(value).to_le_bytes()
            })
            .collect()
    };
    std::fs::write(
        dir.join("lora_a.bin"),
        plane(0x0a0a_a0a0, 0.05, layers * rank * hidden),
    )
    .expect("the A plane writes");
    std::fs::write(
        dir.join("lora_b.bin"),
        plane(0x0b0b_b0b0, amplitude, layers * hidden * rank),
    )
    .expect("the B plane writes");
    std::fs::write(
        dir.join("adapter.toml"),
        format!(
            "rank = {rank}\n\n\
             [[plane]]\nrole = \"lora_a\"\nfile = \"lora_a.bin\"\nlayout = \"rank_major\"\n\n\
             [[plane]]\nrole = \"lora_b\"\nfile = \"lora_b.bin\"\nlayout = \"out_major\"\n"
        ),
    )
    .expect("the manifest writes");
}

/// **THE SHARED-ADAPTER MOUNT'S DEVICE HALF** (alto adapter §3.3, §6.1).
///
/// `blob.rs`'s own pins judge the whole host side with no GPU in the machine —
/// the manifest grammar, the identity, the single flight, the slicing and
/// every refusal — by never landing anything. That leaves exactly the claims
/// that are about bytes which really arrived on a device, and they are these:
///
/// ```text
/// (a) ONE SLOT, ONE COPY. Two binds naming one mounted adapter answer one
///     slot; the second lands nothing and the fire under it is unchanged.
/// (b) THE SAME ANSWER AS THE PRIVATE PATH. A byte-seeded bind of the SAME
///     A and B gets a slot of its own and fires to the same logits, bit for
///     bit — so what the mount changed is WHERE the bytes came from and
///     nothing else.
/// (c) AND IT IS NOT A NO-OP. Both differ from the base model's row, which is
///     what keeps (b) from being two ways of adding zero.
/// (d) A FILE DROP SERVES, and a release frees at the last hold.
/// ```
#[test]
fn two_instances_of_one_mounted_adapter_share_one_slot_and_one_copy() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the shared-adapter mount") else {
        return;
    };
    let seats = shell.bank_seats();
    assert!(
        !seats.is_empty(),
        "this SKU declares adapter banks; a load with none has nowhere to put one"
    );

    // The mount, stated AFTER the load — §3.3's whole posture: where the
    // shared adapters live is the deployment's, not the bake's.
    let mount = scratch("share");
    write_adapter(&mount, "alice-v2", &seats, 0.5);
    shell.mount_adapters(Some(mount.clone()));

    // ── (a) ONE SLOT. The first bind lands; the second joins it.
    let first = shell
        .bind_adapter(engine_metal::AdapterSource::Shared { name: "/alice-v2" })
        .expect("the mount serves the adapter it holds");
    assert!(first.shared, "a name in the mount is a shared source");
    assert!(first.landed, "the first bind is the one that pays");
    let second = shell
        .bind_adapter(engine_metal::AdapterSource::Shared { name: "alice-v2" })
        .expect("a second instance names the same adapter");
    assert_eq!(
        second.slot, first.slot,
        "two instances naming ONE blob must land on ONE slot (alto adapter §3.3): the \
         whole point of keying residency by blob identity is that the second tenant of \
         an adapter costs the device nothing — and the two spellings are one file, so \
         the sharing claim is about bytes and not about strings"
    );
    assert!(
        !second.landed,
        "the second bind paid a landing; it should have joined the one already resident"
    );
    assert_eq!(
        shell.adapter_slots().live(),
        1,
        "one adapter, one resident slot"
    );
    assert_eq!(
        shell.blob_store().blobs().loads(),
        2,
        "one read per plane FILE, and the second bind re-read nothing"
    );

    // ── The fire, under the shared slot.
    let prompt = tokenizer.encode(PROMPT);
    let rows = prompt.len() as u32;
    shell.open(0).expect("slot 0 opens");
    let shared_says = shell
        .fire_seated(&[Seated::adapted(
            Lane {
                slot: 0,
                word: adapted_word(rows),
                tokens: &prompt,
            },
            first.slot,
        )])
        .expect("the corrected prefill fires")[0]
        .clone();
    finite(&shared_says, "the shared adapter's prefill");

    // ── (b) THE PRIVATE PATH, WITH THE SAME A AND B. The resolver's own
    //    planes are what the shared landing wrote, so handing them to the
    //    byte-seeded verb is the same adapter arriving by the other door.
    let (built, _fingerprint) = shell
        .blob_store()
        .planes("alice-v2", &seats)
        .expect("the resolver reads the adapter this test wrote");
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    let private = shell
        .bind_adapter(engine_metal::AdapterSource::Own {
            instance: 77,
            planes: &planes,
        })
        .expect("a private adapter lands from the caller's own bytes");
    assert!(!private.shared, "bytes are not a name in the mount");
    assert!(private.landed, "a private adapter always pays its own landing");
    assert_ne!(
        private.slot, first.slot,
        "a byte-seeded instance gets a slot of its OWN: content-hash dedup across \
         private adapters is a later optimization, and sharing one here would put one \
         tenant's fine-tune under another tenant's rows"
    );

    shell.open(1).expect("slot 1 opens");
    let private_says = shell
        .fire_seated(&[Seated::adapted(
            Lane {
                slot: 1,
                word: adapted_word(rows),
                tokens: &prompt,
            },
            private.slot,
        )])
        .expect("the private prefill fires")[0]
        .clone();
    assert_eq!(
        shared_says, private_says,
        "one adapter arriving by two doors answered two different rows; the mount is \
         supposed to change WHERE the bytes came from and nothing else"
    );

    // ── (c) AND IT IS NOT A NO-OP.
    shell.open(2).expect("slot 2 opens");
    let base_says = shell
        .fire(&[Lane {
            slot: 2,
            word: word(rows),
            tokens: &prompt,
        }])
        .expect("the uncorrected prefill fires")[0]
        .clone();
    assert_ne!(
        shared_says, base_says,
        "the corrected row is the base model's, so this whole test is comparing two \
         ways of adding zero"
    );

    // ── (d) A FILE DROP SERVES, and the mount was never re-stated.
    write_adapter(&mount, "bob-v1", &seats, 0.25);
    let dropped = shell
        .bind_adapter(engine_metal::AdapterSource::Shared { name: "bob-v1" })
        .expect(
            "an adapter written into the mount while the box serves must bind: §3.3 \
             makes adding a LoRA a file drop, so a refusal here would mean the catalog \
             was snapshotted at boot after all",
        );
    assert!(dropped.landed, "a name nobody has bound before pays a landing");
    assert_ne!(dropped.slot, first.slot, "a different adapter is a different identity");
    assert_ne!(dropped.slot, private.slot, "and it is not the private one's either");

    // A name nobody wrote is still an absence, and it says so.
    let why = shell
        .bind_adapter(engine_metal::AdapterSource::Shared { name: "carol-v9" })
        .expect_err("a name nobody wrote is not in the mount")
        .to_string();
    assert!(why.contains("carol-v9"), "the refusal names the adapter: {why}");

    // ── THE RELEASE, AND IT TAKES BOTH HOLDS. A slot two instances of one
    //    blob hold stays pinned until both are gone, which is what keeps a
    //    shared adapter from being taken out from under a fire.
    let held = shell.adapter_slots().live();
    shell.release_adapter(&first);
    assert_eq!(
        shell.adapter_slots().live(),
        held,
        "one of two holds went and the slot is still pinned"
    );
    shell.release_adapter(&second);
    assert_eq!(
        shell.adapter_slots().live(),
        held - 1,
        "the last release frees the shared slot"
    );
    shell.release_adapter(&private);
    shell.release_adapter(&dropped);
    assert_eq!(
        shell.adapter_slots().live(),
        0,
        "every bind was given back, so the table this test leaves behind is empty"
    );
    let _ = std::fs::remove_dir_all(&mount);
}

// ─────────────────────────────────────────────────────────────────────────────
// Guest programs at the fire's epilogue (design §9)
// ─────────────────────────────────────────────────────────────────────────────
//
// **THE CLAIM THESE MAKE THAT NO UNIT TEST REACHES**, and it is the same
// shape as the claim at the top of this file: a shell can bind the intrinsic,
// encode the pass, settle the verdict and commit the cursors, and still have
// pointed the guest at the wrong bytes. The only way to know is to make the
// guest compute something the host can compute too, from a row the host can
// name, and diff them.
//
// The subject is `greedy_argmax` out of `eta-compiler`'s golden corpus — the
// program the inferlet corpus is built on: take a token from one channel,
// argmax the readout, publish the winner. It is bound at the corpus profile's
// vocabulary of EIGHT, not at qwen's 248320, and that is what makes the diff
// sharp rather than what weakens it: the guest reads the first eight elements
// of whatever row the shell pointed it at, so the assertion is
// `argmax(guest) == argmax(host_row[..8])` and it is a statement about WHICH
// ROW was bound. A shell off by one row, or off by the lane's first row
// instead of its last, fails it; a shell that argmaxed the whole vocabulary
// correctly but bound row zero would pass a full-vocab test on a one-row lane
// and fail this one on a prefill.

/// The corpus `program_parity` reads, reached the same way.
fn golden_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eta-compiler/tests/golden"
    ))
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len() / 2)
        .map(|index| u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).expect("hex"))
        .collect()
}

/// One golden trace, all the way to what `Shell::register_program` takes.
///
/// The bind-time profile is transcribed from `eta-compiler`'s corpus helper
/// for the same reason `program_parity` transcribes it: the goldens do not
/// carry one, and binding at the wrong vocabulary refuses rather than
/// misbehaves.
fn guest_registration(name: &str) -> engine::program::ProgramRegistration {
    let path = golden_dir().join(format!("{name}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path:?} is missing"));
    let line = text
        .lines()
        .find_map(|line| line.strip_prefix("container: "))
        .unwrap_or_else(|| panic!("{name} has no container line"));
    let container = eta_ir::container::decode(&unhex(line))
        .unwrap_or_else(|why| panic!("{name} does not decode: {why:?}"));

    let mut profile = eta_ir::registry::ModelProfile::dummy();
    profile.vocab = GUEST_VOCAB;
    let bound = eta_ir::validate::bind(container, profile)
        .unwrap_or_else(|why| panic!("{name} does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Metal;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);

    engine::program::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted,
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

/// How wide the golden corpus's readout is. Eight, and the module note above
/// argues why that is the point rather than a limitation.
const GUEST_VOCAB: u32 = 8;

/// The channel `greedy_argmax` publishes its winner on, read off the golden's
/// own declaration rather than assumed: the one channel the host reads.
fn output_channel(registration: &engine::program::ProgramRegistration) -> u32 {
    registration
        .launch
        .channels
        .iter()
        .position(|declared| declared.host_role == eta_ir::container::HostRole::Reader)
        .map(|at| at as u32)
        .expect("greedy_argmax publishes on a channel the host reads")
}

/// One seed cell per channel the program declares a seed for — a single
/// zeroed cell, which is all `greedy_argmax`'s input channel needs to satisfy
/// its `NeedsFull` requirement.
fn guest_seeds(registration: &engine::program::ProgramRegistration) -> Vec<(u32, Vec<u8>)> {
    registration
        .launch
        .channels
        .iter()
        .enumerate()
        .filter(|(_, declared)| declared.seeded)
        .map(|(at, declared)| {
            let lanes = declared
                .shape
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            let dtype = eta_exec::concrete_dtype(declared.dtype);
            (at as u32, vec![0u8; eta_exec::wire_cell_bytes(dtype, lanes)])
        })
        .collect()
}

/// The channel a non-const descriptor port reads, off the launch package's
/// own binding rather than assumed. `None` for a program that binds the port
/// as a trace-time constant, or not at all.
fn port_channel(
    registration: &engine::program::ProgramRegistration,
    port: eta_ir::registry::Port,
) -> Option<u32> {
    registration
        .launch
        .ports
        .iter()
        .find(|bound| bound.port == port && !bound.is_const)
        .map(|bound| bound.channel as u32)
}

/// Register and bind one instance of `name` on `shell`, ready to be attached.
fn guest_instance(shell: &mut Shell, name: &str) -> (u64, u32) {
    guest_instance_in(shell, name, eta_ir::registry::GeometryClass::Host, Some(0))
}

/// The same, in a stated geometry class and with the first cell of the
/// `embed_tokens` port's channel stated too.
///
/// **THE SEED IS THE FIRST FIRE'S TOKEN, WHICH IS WHY IT IS AN ARGUMENT.**
/// `greedy_argmax`'s channel 0 is its own `embed_tokens` port and its own
/// output — the epilogue publishes the winner into it and the next fire's
/// port reads it back, which is the loop-carried decode channel with nothing
/// added. On the FIRST fire there is no previous epilogue, so whatever the
/// bind seeded is what that fire embeds: zero (the value
/// [`guest_seeds`] writes) says nothing, and a real token id says everything.
///
/// `None` seeds that channel with nothing at all, which is the instance that
/// never published — an empty ring, and the readiness gate's business.
fn guest_instance_in(
    shell: &mut Shell,
    name: &str,
    geometry: eta_ir::registry::GeometryClass,
    carried: Option<i32>,
) -> (u64, u32) {
    let registration = guest_registration(name);
    let channel = output_channel(&registration);
    let mut seeds = guest_seeds(&registration);
    if let Some(embed) = port_channel(&registration, eta_ir::registry::Port::EmbedTokens) {
        match carried {
            Some(token) => {
                let cell = seeds
                    .iter_mut()
                    .find(|(at, _)| *at == embed)
                    .map(|(_, cell)| cell)
                    .unwrap_or_else(|| {
                        panic!("{name}'s `embed_tokens` channel {embed} declares no seed")
                    });
                assert_eq!(
                    cell.len(),
                    4,
                    "{name}'s `embed_tokens` cell is {} bytes and a token id is four",
                    cell.len()
                );
                cell.copy_from_slice(&token.to_le_bytes());
            }
            None => seeds.retain(|(at, _)| *at != embed),
        }
    }
    let program = shell
        .register_program(&registration)
        .unwrap_or_else(|error| panic!("{name} does not compile on this device: {error}"));
    let instance = shell
        .bind_program(
            program,
            &seeds,
            eta_exec::Extents {
                sampled_rows: 1,
                ..eta_exec::Extents::default()
            },
            geometry,
            // No registered channels: this fixture's rings are its own, which
            // is what `&[]` says.
            &[],
        )
        .unwrap_or_else(|error| panic!("{name} does not bind: {error}"));
    (instance, channel)
}

/// The i32 the guest published, taken off its output ring.
fn published(shell: &mut Shell, instance: u64, channel: u32) -> i32 {
    let cell = shell
        .program_instance(instance)
        .expect("the fence lands the flight that carried the epilogue")
        .expect("the instance is still bound")
        .take(channel)
        .expect("the ring reads")
        .expect("the epilogue published a cell");
    i32::from_le_bytes([cell[0], cell[1], cell[2], cell[3]])
}

/// **THE HEADLINE ATTACHMENT CLAIM: A GUEST EPILOGUE READS THE ROW THE HOST
/// READS.**
///
/// One fire, one lane, one attached instance. The host is handed the lane's
/// last row out of the arm's readout seat; the guest is handed the same row
/// out of the arena rectangle, on the device, inside the same command buffer,
/// and argmaxes it. Two paths, two readers, one row — and if the shell bound
/// the intrinsic anywhere else the two answers are unrelated numbers.
///
/// It is also the whole of what a naive-baseline inferlet does, which is why
/// this test standing up is what says the corpus can run here at all.
#[test]
fn an_attached_epilogue_argmaxes_the_row_the_host_reads() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the attached-epilogue smoke") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);

    let (instance, channel) = guest_instance(&mut shell, "greedy_argmax");
    shell.open(0).expect("slot 0 opens");

    let rows = shell
        .fire_attached(
            &[Seated {
                lane: Lane {
                    slot: 0,
                    word: word(prompt.len() as u32),
                    tokens: &prompt,
                },
                pages: &[],
                held: None,
                mask: None,
                adapter: None,
                positions: &[],
                readout: None,
                captures_scores: false,
                translation: &[],
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect("the attached prefill fires");

    finite(&rows[0], "the attached prefill");
    let host = argmax(&rows[0][..GUEST_VOCAB as usize]);
    let guest = published(&mut shell, instance, channel);
    eprintln!("host argmax over the first {GUEST_VOCAB}: {host}, guest published: {guest}");
    assert_eq!(
        i64::from(guest),
        i64::from(host),
        "the epilogue argmaxed a different row from the one the host was handed: the \
         `logits` intrinsic is bound somewhere other than this lane's last row"
    );
}

/// **A ROW LIST POINTS THE EPILOGUE AT THAT ROW AND NOT AT THE LAST ONE.**
///
/// This is the refusal `api.rs` used to answer `row-selected readout` to, and
/// it is the whole of what stands between this plane and the speculative
/// corpus: a verifier states `k` rows and reads them on the device, and a
/// shell that pointed it at the lane's last row would hand it that row
/// followed by `k - 1` rows past the rectangle.
///
/// The host has no seat for an interior row — that ceiling is real and is
/// still refused — so the control is a SECOND fire, on a fresh slot, whose
/// prompt is the prefix ending at the row under test. Its last row is the
/// same teacher-forced position, so its logits are the row the epilogue
/// should have been handed.
#[test]
fn a_stated_readout_row_is_the_row_the_epilogue_is_handed() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the stated-readout-row smoke") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    assert!(prompt.len() >= 3, "the prompt is too short to name an interior row");
    let want = 1u32;

    // The control: a prefill of the prefix, whose LAST row is `want`.
    shell.open(0).expect("slot 0 opens");
    let prefix = &prompt[..=want as usize];
    let control = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prefix.len() as u32),
            tokens: prefix,
        }])
        .expect("the prefix prefill fires");
    let expected = argmax(&control[0][..GUEST_VOCAB as usize]);

    // The subject: the whole prompt, with the epilogue pointed at row `want`.
    let (instance, channel) = guest_instance(&mut shell, "greedy_argmax");
    shell.open(1).expect("slot 1 opens");
    let rows = shell
        .fire_attached(
            &[Seated {
                lane: Lane {
                    slot: 1,
                    word: word(prompt.len() as u32),
                    tokens: &prompt,
                },
                pages: &[],
                held: None,
                mask: None,
                adapter: None,
                positions: &[],
                readout: Some(&[want]),
                captures_scores: false,
                translation: &[],
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect("the row-selected attached prefill fires");

    let last = argmax(&rows[0][..GUEST_VOCAB as usize]);
    let guest = published(&mut shell, instance, channel);
    eprintln!("row {want} expects {expected}, the guest published {guest}, the last row is {last}");
    assert_eq!(
        i64::from(guest),
        i64::from(expected),
        "the epilogue was not handed row {want}: a stated row list has to reach the \
         intrinsic's offset, or every speculative verifier reads the wrong rows"
    );
}

/// **AN ATTACHMENT NAMING A LANE THIS FIRE DOES NOT HAVE IS REFUSED BY NAME,
/// AND BEFORE ANYTHING IS STAGED.** The row it would be pointed at does not
/// exist, so the alternative is a binding into whatever the rectangle holds
/// past the fire's rows — zeros, and an argmax over zeros is token 0.
#[test]
fn an_attachment_naming_an_absent_lane_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the absent-lane attachment refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let (instance, _) = guest_instance(&mut shell, "greedy_argmax");
    shell.open(0).expect("slot 0 opens");

    let fault = shell
        .fire_attached(
            &[Seated {
                lane: Lane {
                    slot: 0,
                    word: word(prompt.len() as u32),
                    tokens: &prompt,
                },
                pages: &[],
                held: None,
                mask: None,
                adapter: None,
                positions: &[],
                readout: None,
                captures_scores: false,
                translation: &[],
            }],
            &[engine_metal::Attached {
                lane: 3,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("an attachment naming lane 3 of a one-lane fire");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("names lane 3"),
        "the refusal does not name the lane it refused: {said}"
    );
}

/// **ONE INSTANCE, ONE PASS, ONE COMMIT.** A second attachment of the same
/// instance would gate against cursors the first pass has not committed and
/// would publish its channel effects twice — so it is refused rather than run.
#[test]
fn an_instance_attached_twice_to_one_fire_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the double-attach refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let (instance, _) = guest_instance(&mut shell, "greedy_argmax");
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");

    let lanes: Vec<Seated<'_>> = (0..2)
        .map(|slot| Seated {
            lane: Lane {
                slot,
                word: word(prompt.len() as u32),
                tokens: &prompt,
            },
            pages: &[],
            held: None,
            mask: None,
            adapter: None,
            positions: &[],
            readout: None,
            captures_scores: false,
            translation: &[],
        })
        .collect();
    let fault = shell
        .fire_attached(
            &lanes,
            &[
                engine_metal::Attached {
                    lane: 0,
                    instance,
                    at: engine::fire::Boundary::Epilogue,
                },
                engine_metal::Attached {
                    lane: 1,
                    instance,
                    at: engine::fire::Boundary::Epilogue,
                },
            ],
        )
        .expect_err("one instance attached to two lanes of one fire");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("attached twice"),
        "the refusal does not say what it refused: {said}"
    );
}

/// **`Boundary::Prologue` IS REFUSED BY NAME, AND THE SENTENCE SAYS WHY.**
///
/// A prologue's channel writes are INPUTS to the forward — token ids,
/// positions, a mask — and this shell stages every fire input on the host, at
/// `prepare`, before it opens a command buffer. So there is no point in the
/// step at which one could be encoded, and running it after the walk would
/// answer the fire with tokens the guest had not written yet.
#[test]
fn a_prologue_attachment_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the prologue attachment refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let (instance, _) = guest_instance(&mut shell, "greedy_argmax");
    shell.open(0).expect("slot 0 opens");

    let fault = shell
        .fire_attached(
            &[Seated {
                lane: Lane {
                    slot: 0,
                    word: word(prompt.len() as u32),
                    tokens: &prompt,
                },
                pages: &[],
                held: None,
                mask: None,
                adapter: None,
                positions: &[],
                readout: None,
                captures_scores: false,
                translation: &[],
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Prologue,
            }],
        )
        .expect_err("a guest program attached before the graph");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("Epilogue"),
        "the refusal does not name the boundary this plane does serve: {said}"
    );
}

/// **A READOUT LIST THAT SKIPS IS REFUSED, AND A CONSECUTIVE ONE IS NOT.**
///
/// The M2 emitter points the intrinsic at one buffer with one offset and the
/// op walks it with the stride it was planned with, so `start .. start + k` is
/// a base and nothing else. A list that skips or descends has no such
/// spelling here — the CUDA plane pays a row-pointer table for it — and
/// serving it would hand the guest the first row followed by whatever the
/// stride landed on.
#[test]
fn a_readout_list_that_is_not_one_run_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the non-consecutive readout refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    assert!(prompt.len() >= 4, "the prompt is too short to skip a row");
    let (instance, _) = guest_instance(&mut shell, "greedy_argmax");
    shell.open(0).expect("slot 0 opens");

    let fault = shell
        .fire_attached(
            &[Seated {
                lane: Lane {
                    slot: 0,
                    word: word(prompt.len() as u32),
                    tokens: &prompt,
                },
                pages: &[],
                held: None,
                mask: None,
                adapter: None,
                positions: &[],
                readout: Some(&[0, 2]),
                captures_scores: false,
                translation: &[],
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("a readout list that skips a row");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("ascending run"),
        "the refusal does not say what shape it can serve: {said}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The decode envelope: a fire whose tokens come off a guest's device ring
// ─────────────────────────────────────────────────────────────────────────────
//
// **THE ONE VALUE A HOST CANNOT KNOW IS THE SAMPLED TOKEN**, and everything
// below is one claim about it. A `GeometryClass::Host` instance's fire reads
// its ids out of the submission, because the runtime folded them and stated
// them; a `GeometryClass::DecodeEnvelope` instance's submission carries
// PLACEHOLDERS for exactly that, and `serve::stage` step 0b reads the real
// ones off the instance's own `embed_tokens` ring — the cell the previous
// fire's epilogue wrote, which no host has seen.
//
// The subject is the corpus's `greedy_argmax` again and it needs nothing
// added: its `embed_tokens` port is bound to channel 0, and channel 0 is
// where its epilogue publishes the winner. Port-consumed and put in one pass,
// which `ExecPlan::takes_channel` counts as a take, so the pass-atomic commit
// advances head and tail together and the ring stays at depth one forever.
// That IS the loop-carried decode channel: seed the first token, and every
// fire after it embeds what the fire before it sampled.
//
// **WHAT PINS THE LOOP TO ONE STEP HERE IS THE GOLDEN'S CONST PORTS, NOT THE
// SHELL.** `greedy_argmax` binds `positions` as the constant `[0]` and
// `kv_len` as the constant `1`, because it was authored as a one-token
// decode over a fresh sequence. `Envelope::positions_for` refuses a run that
// is not `have .. have + rows` and `Envelope::check_extent` refuses an extent
// the fire does not reach, so an envelope lane through this program is a
// one-token fire on a freshly-opened slot and the second step on that slot is
// a named refusal rather than a wrong answer. Both halves are tested: the
// first is the claim, the second is the check working.

/// **AN ENVELOPE LANE EMBEDS WHAT THE GUEST PUBLISHED AND NOT WHAT THE
/// SUBMISSION STATES.**
///
/// Three lanes in one fire, and they are a two-sided diff. Lane 0 is bound in
/// `GeometryClass::DecodeEnvelope` and its submission carries a DECOY id;
/// lane 1 is the same program one enum apart — `GeometryClass::Host` — and
/// its submission carries the id the guest's ring holds; lane 2 carries the
/// decoy with no attachment at all. A shell that resolved the envelope
/// answers lane 0 as lane 1; a shell that read the submission answers it as
/// lane 2. Nothing else can move the row: same word, same fresh slot, same
/// single row, same class.
///
/// `program::ports::resolved` is the third assertion and it is the one no row
/// can make. The claim run-ahead rests on is a NEGATIVE — the token never
/// travelled to the host — and nothing happens when a round trip does not
/// happen, so the counter is what says the envelope was resolved, and
/// resolved ONCE: the host twin beside it must cost no ring read at all.
#[test]
fn an_envelope_lane_embeds_what_the_guest_published_and_not_what_the_submission_states() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the decode-envelope smoke") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    assert!(
        prompt.len() >= 2,
        "the prompt is too short to carry two distinct ids"
    );
    let carried = prompt[0];
    let decoy = *prompt
        .iter()
        .find(|&&id| id != carried)
        .expect("the prompt holds two distinct ids");

    // Two instances of one program, one enum apart. The device one is seeded
    // with the id its port will read; the host one is seeded with it too, so
    // that the ONLY difference between the two lanes is which side of the
    // boundary the id travelled on.
    let (device, _) = guest_instance_in(
        &mut shell,
        "greedy_argmax",
        eta_ir::registry::GeometryClass::DecodeEnvelope,
        Some(carried as i32),
    );
    let (host, _) = guest_instance_in(
        &mut shell,
        "greedy_argmax",
        eta_ir::registry::GeometryClass::Host,
        Some(carried as i32),
    );
    for slot in 0..3 {
        shell.open(slot).expect("the slot opens");
    }

    let placeholder = [decoy];
    let stated = [carried];
    let before = engine_metal::program::ports::resolved();
    let rows = shell
        .fire_attached(
            &[
                Seated::of(Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &placeholder,
                }),
                Seated::of(Lane {
                    slot: 1,
                    word: word(1),
                    tokens: &stated,
                }),
                Seated::of(Lane {
                    slot: 2,
                    word: word(1),
                    tokens: &placeholder,
                }),
            ],
            &[
                engine_metal::Attached {
                    lane: 0,
                    instance: device,
                    at: engine::fire::Boundary::Epilogue,
                },
                engine_metal::Attached {
                    lane: 1,
                    instance: host,
                    at: engine::fire::Boundary::Epilogue,
                },
            ],
        )
        .expect("the decode-envelope fire fires");
    let resolved = engine_metal::program::ports::resolved() - before;

    finite(&rows[0], "the envelope lane");
    finite(&rows[1], "the host twin");
    finite(&rows[2], "the control lane");
    eprintln!(
        "carried {carried}, decoy {decoy} — envelope argmax {}, host twin {}, \
         control {}; envelopes resolved this fire: {resolved}",
        argmax(&rows[0]),
        argmax(&rows[1]),
        argmax(&rows[2]),
    );

    // The precondition, asserted rather than assumed: if the two ids answered
    // the same row the diff below would pass for a shell that read either.
    assert_ne!(
        rows[1], rows[2],
        "the carried id and the decoy answer the same logits, so this fire \
         cannot tell which of them the envelope lane embedded"
    );
    assert_eq!(
        rows[0], rows[1],
        "the envelope lane did not answer what the host twin answered for the \
         id the guest published: `serve::stage` read this lane's tokens \
         somewhere other than its `embed_tokens` port"
    );
    assert_ne!(
        rows[0], rows[2],
        "the envelope lane answered what its own SUBMISSION states, which is \
         the placeholder the runtime ships precisely because it could not know \
         the token"
    );
    assert_eq!(
        resolved, 1,
        "one attached device-carried lane is one envelope resolved: the host \
         twin beside it must cost no ring read at all"
    );
}

/// **AN ENVELOPE ON AN INSTANCE THAT NEVER PUBLISHED IS REFUSED BY NAME, AND
/// THE FENCE IS WHY THE ANSWER IS ABOUT THIS FIRE.**
///
/// The port's value for a fire is the committed front of its ring — the cell
/// the guest's own pass takes — so an instance nothing has published into has
/// no such cell, and the alternative to refusing is embedding whatever the
/// allocation came with. Two gates stand in front of it and this pins the
/// outer one: `Shell::admit_attachments` fences the instance (landing every
/// flight that carried a pass of its own, so the cursors are this fire's and
/// not the fire before last's) and then asks
/// `Session::blocked_channel`, which reads the `NeedsFull` requirement the
/// program declares on the channel its port is bound to. The inner one is
/// `program::ports::read_cell`'s own `head == tail` refusal, for a program
/// that declares no readiness on a channel it binds a port to.
///
/// Nothing has launched when either speaks, which is the price a refusal on
/// this path has to cost: the alternative is discovering it after the forward
/// has written the lane's KV, and that fire the caller cannot retry.
#[test]
fn an_envelope_the_guest_has_not_published_into_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the unpublished-envelope refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let (instance, _) = guest_instance_in(
        &mut shell,
        "greedy_argmax",
        eta_ir::registry::GeometryClass::DecodeEnvelope,
        None,
    );
    shell.open(0).expect("slot 0 opens");

    let placeholder = [prompt[0]];
    let fault = shell
        .fire_attached(
            &[Seated::of(Lane {
                slot: 0,
                word: word(1),
                tokens: &placeholder,
            })],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("an envelope over a ring nothing has published into");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("not ready to fire"),
        "the refusal does not say the instance could not have fired: {said}"
    );
    assert!(
        said.contains("channel 0"),
        "the refusal does not name the channel the port reads: {said}"
    );
}

/// **AN ENVELOPE WHOSE EXTENT HAS DRIFTED FROM THE SEAT IS REFUSED BY BOTH
/// NUMBERS.**
///
/// `kv_len` is served as a CHECK and not as a source, and this is what makes
/// the difference observable. The extent this shell fires with is the seat's
/// own `have + rows` — which is also what the page CSR, the write descriptor
/// and the attention schedules are carved from, because a decode-envelope
/// lane's page table is the SHELL's. Taking the guest's number instead would
/// let one port silently disagree with the four the shell derives, and the
/// failure is a fire that attends the wrong pages.
///
/// `greedy_argmax` states the constant `1`, so its second step on one slot is
/// where the two numbers part: the seat reaches two and the port still says
/// one. The output ring is drained first so that the readiness gate is not
/// what answers — the claim is about the extent, and a refusal from the gate
/// beside it would prove nothing about `check_extent` at all.
#[test]
fn an_envelope_extent_that_has_drifted_from_the_seat_is_refused_by_both_numbers() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the drifted-extent refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let (instance, out) = guest_instance_in(
        &mut shell,
        "greedy_argmax",
        eta_ir::registry::GeometryClass::DecodeEnvelope,
        Some(prompt[0] as i32),
    );
    shell.open(0).expect("slot 0 opens");

    let placeholder = [prompt[0]];
    let attachment = [engine_metal::Attached {
        lane: 0,
        instance,
        at: engine::fire::Boundary::Epilogue,
    }];
    let first = shell
        .fire_attached(
            &[Seated::of(Lane {
                slot: 0,
                word: word(1),
                tokens: &placeholder,
            })],
            &attachment,
        )
        .expect("the first envelope step fires");
    finite(&first[0], "the first envelope step");
    // Drained so that `NeedsEmpty` on the output ring is satisfied for the
    // step below: what that step must be refused for is the extent.
    let sampled = published(&mut shell, instance, out);
    eprintln!("the guest sampled {sampled} and published it into its own embed ring");

    let fault = shell
        .fire_attached(
            &[Seated::of(Lane {
                slot: 0,
                word: word(1),
                tokens: &placeholder,
            })],
            &attachment,
        )
        .expect_err("a second step on a slot whose `kv_len` port states the first");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("kv_len"),
        "the refusal does not name the port that disagreed: {said}"
    );
    assert!(
        said.contains("extent of 1") && said.contains("reaches 2"),
        "the refusal names only one of the two numbers, leaving the reader to \
         guess which side drifted: {said}"
    );
}

/// **A LANE'S POSITIONS HAVE ONE AUTHOR, AND STATING BOTH IS REFUSED RATHER
/// THAN RESOLVED BY PRECEDENCE.**
///
/// The class IS the statement of who resolves. A submission that states
/// positions for a lane bound in a device-resolved class has said two things
/// at once, and either way of honouring it drops one of them under the
/// caller's nose — the guest's run, or the caller's. So the contradiction is
/// the refusal, and it arrives at `prepare` where nothing has launched.
///
/// The runtime never authors it: `pipeline::fire::geometry` ships
/// `Lane::positions` empty whenever the run is the natural one, which is
/// every decode. What this closes is the hand-built submission.
#[test]
fn a_device_class_lane_that_also_states_positions_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the two-author positions refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let (instance, _) = guest_instance_in(
        &mut shell,
        "greedy_argmax",
        eta_ir::registry::GeometryClass::DecodeEnvelope,
        Some(prompt[0] as i32),
    );
    shell.open(0).expect("slot 0 opens");

    let placeholder = [prompt[0]];
    let stated = [0u32];
    let fault = shell
        .fire_attached(
            &[Seated {
                positions: &stated,
                ..Seated::of(Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &placeholder,
                })
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("a device-class lane whose submission states positions too");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("device-resolved geometry class"),
        "the refusal does not say which of the two authors the class names: {said}"
    );
    assert!(
        said.contains("also states"),
        "the refusal does not say what the submission did: {said}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The pooled device geometry: a fire whose PAGE TABLE comes off a guest's ring
// ─────────────────────────────────────────────────────────────────────────────
//
// **THE DECODE ENVELOPE MOVED THE TOKEN; THIS CLASS MOVES THE WRITE.** A
// `GeometryClass::DecodeEnvelope` lane's page table is the SHELL's — the seat's
// `have .. have + rows` is what the page CSR, the write descriptor and the
// attention schedules are all carved from — and only the ids it embeds are the
// guest's. A `GeometryClass::DeviceGeometry` lane states all of it: how many
// lanes the instance carries (`embed_indptr`), which pages each may address
// (`pages`/`page_indptr`), how long each is (`kv_len`), which cell THIS row
// lands in (`w_slot`/`w_off`) and which keys it may reach (`attn_mask`). That
// is the class every attention-shaped fixture in `tests/inferlets` needs —
// beam search, sliding window, attention sink, snapkv, consensus decoding —
// because each of them owns a pool the runtime only leases it.
//
// **THE SUBJECT IS `beam_epilogue`, WHICH IS THE CORPUS'S ONLY MEMBER THAT
// BINDS THE WHOLE CLASS.** Nine ports, two lanes through one instance, a dense
// `[B, P * PAGE_T]` bool mask on a channel. It is also the corpus's widest
// program at sixteen channels, which `program_parity`'s ceiling test already
// says compiles and fires here; what is new below is that a MODEL fire takes
// its geometry from it.
//
// **AND THE GEOMETRY IS SEEDED RATHER THAN GROWN.** The port values a fire
// reads are the committed cells of the instance's rings, and a bind states
// those directly — so the test states the geometry it wants to be fired
// against instead of running the beam loop far enough to produce one. That is
// the same lever `guest_instance_in` pulls for the envelope tests one section
// up, widened from one port to nine.

/// **THE SUBJECT IS AUTHORED HERE AND NOT QUOTED, AND THE REASON IS A
/// MEASURED REFUSAL.** The golden corpus's only member that binds the whole
/// port class is `beam_epilogue`, and it reads `intrinsics::logits()` as
/// `[B, V]` — two rows. This plane's emitted intrinsic handler walks a
/// rectangle's rows consecutively and has no row stride to be told, so
/// `program::launch` refuses a multi-row read by name ("every row after the
/// first would land 248312 elements short"), which makes `beam_epilogue`
/// unattachable to a MODEL fire here however well its geometry resolves. That
/// is a real residual blocker for beam search on this plane and it is a
/// different piece of work from this one.
///
/// So the fixture below is a ONE-lane pooled device-geometry epilogue — the
/// shape every other member of the class has (`sliding-window-attention`,
/// `attention-sink`, `snapkv-eviction`: one sequence, one row, a pool the
/// guest owns) — authored through the same `eta-dsl` builder the goldens were
/// authored through. It binds all nine ports and it re-publishes each one, so
/// it is also what `lease::detect_pooled_device_geometry` calls a pooled pass.
const GUEST_LANES: usize = 1;

/// How wide the fixture's mask rectangle is. The POOL's width, which is what a
/// guest builds a mask at — larger than any extent this test fires, so the
/// surplus takes `crate::mask`'s clip.
const GUEST_MASK_KEYS: usize = 32;

/// How many pages the fixture's page run holds, and how many keys one holds.
///
/// The second is the SHELL's, quoted: `ready`'s boot states `page_size: 16`,
/// and the write descriptor a guest publishes is a page of the run plus an
/// offset inside it, so a fixture that guessed a different size would state a
/// cell the attention does not read. `GUEST_MASK_KEYS` is the two multiplied
/// out, which is what makes the rectangle cover the run.
const GUEST_PAGE_RUN: u32 = 2;
const GUEST_PAGE_SIZE: u32 = 16;

/// The fixture, traced.
fn device_geometry_registration() -> engine::program::ProgramRegistration {
    device_geometry_registration_with(true)
}

/// The same, with the page run's channel declared UNSEEDED when
/// `seeded_pages` is false — the instance that never published, which is what
/// the empty-ring refusal is about.
fn device_geometry_registration_with(seeded_pages: bool) -> engine::program::ProgramRegistration {
    use eta_dsl::builder::Builder;
    use eta_dsl::prelude::*;
    use eta_dsl::{Channel, dtype};

    fn leak<T>(value: T) -> &'static T {
        Box::leak(Box::new(value))
    }

    // THE NINE PORTS, ONE CHANNEL APIECE. The four the wide class adds to the
    // envelope — `pages`, `page_indptr`, `w_slot`, `w_off` — plus the row
    // split and the mask, which is what makes this the whole class rather
    // than the trio.
    let toks: &'static Channel = leak(Channel::from([1i32]).named("toks"));
    let qo: &'static Channel = leak(Channel::from([0u32, 1]).named("qo"));
    let pos: &'static Channel = leak(Channel::from([0u32]).named("pos"));
    let klen: &'static Channel = leak(Channel::from([1u32]).named("klen"));
    let pages: &'static Channel = leak(
        if seeded_pages {
            Channel::seeded([GUEST_PAGE_RUN], dtype::u32)
        } else {
            Channel::new([GUEST_PAGE_RUN], dtype::u32)
        }
        .named("pages"),
    );
    let page_indptr: &'static Channel =
        leak(Channel::from([0u32, GUEST_PAGE_RUN]).named("page_indptr"));
    let w_slot: &'static Channel = leak(Channel::from([0u32]).named("w_slot"));
    let w_off: &'static Channel = leak(Channel::from([0u32]).named("w_off"));
    let mask: &'static Channel = leak(
        Channel::seeded([1u32, GUEST_MASK_KEYS as u32], dtype::bool).named("mask"),
    );
    let out: &'static Channel = leak(Channel::new([1], dtype::i32).named("out"));

    let mut builder = Builder::new(GUEST_VOCAB, 16);
    builder.bind_port(Port::EmbedTokens, toks);
    builder.bind_port(Port::EmbedIndptr, qo);
    builder.bind_port(Port::Positions, pos);
    builder.bind_port(Port::KvLen, klen);
    builder.bind_port(Port::Pages, pages);
    builder.bind_port(Port::PageIndptr, page_indptr);
    builder.bind_port(Port::WSlot, w_slot);
    builder.bind_port(Port::WOff, w_off);
    builder.bind_port(Port::AttnMask, mask);
    // THE PASS: argmax the row this fire computed, publish it as the next
    // fire's token, and ADVANCE the geometry the way a real pooled guest
    // does. The four peeked ports (`embed_indptr`, `kv_len`... no: `kv_len`
    // is peeked and re-put, while `embed_indptr`, `pages`, `page_indptr` and
    // `attn_mask` are peeked and left alone) keep their seeded cells; the
    // four consuming ones (`embed_tokens`, `positions`, `w_slot`, `w_off`)
    // are re-published, because a consumed cell is gone after the pass.
    builder.stage(Stage::Epilogue, move || {
        let next = reduce_argmax(intrinsics::logits());
        toks.put(&next);
        out.put(next);
        pos.put(add(pos.take(), 1u32));
        klen.put(add(klen.take(), 1u32));
        w_slot.put(w_slot.take());
        w_off.put(add(w_off.take(), 1u32));
        // THE PAGE RUN IS RE-PUBLISHED TOO, and not because it moved: `pages`
        // is a PEEK port, so its cell would stand untouched forever. What the
        // explicit drain-and-refill buys is the loop-carried shape a real
        // pooled guest has — `lease::detect_pooled_device_geometry` requires
        // every descriptor channel to be re-published before it will call a
        // pass pooled — and it is what makes the UNSEEDED variant below a
        // legal program at all: a channel a port consumes and nothing
        // produces is a trace-time lint, so "nothing has published one" has
        // to be a first-fire fact rather than a declaration.
        pages.put(pages.take());
    });
    let traced = builder
        .build()
        .expect("the one-lane device-geometry epilogue traces");
    registration_of(traced.container().clone(), "the device-geometry fixture")
}

/// One bound, compiled trace, as `Shell::register_program` takes it.
///
/// The tail of [`guest_registration`], lifted so that a trace this file
/// AUTHORS reaches the shell down the same path a golden does — same profile,
/// same bind, same backend, same emitter.
fn registration_of(
    container: eta_ir::container::TraceContainer,
    what: &str,
) -> engine::program::ProgramRegistration {
    let mut profile = eta_ir::registry::ModelProfile::dummy();
    profile.vocab = GUEST_VOCAB;
    let bound = eta_ir::validate::bind(container, profile)
        .unwrap_or_else(|why| panic!("{what} does not bind: {why:?}"));
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = eta_compiler::codegen::program::Backend::Metal;
    let emitted = eta_compiler::codegen::program::emit_program(backend, &stages, &bound);

    engine::program::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emitted,
        emitter_version: backend.emitter_version(),
        region_analysis: Vec::new(),
        launch,
        reference_ptir: Vec::new(),
    }
}

/// One instance of [`DEVICE_GEOMETRY_GUEST`], bound in the wide class with
/// every descriptor port seeded to `geometry`.
///
/// **THE SEEDS ARE FOUND BY PORT AND NOT BY CHANNEL NUMBER.** A golden's
/// channel order is an artefact of the trace that authored it; what this test
/// is about is which PORT carries which fact, and `port_channel` reads that
/// off the launch package's own bindings. A port the golden folded as a
/// trace-time constant has no channel and needs no seed — its value reaches
/// `program::ports::resolve` out of `ExecPlan::const_ports` — so a `None` here
/// is a port that already says what it should.
fn device_geometry_instance(
    shell: &mut Shell,
    geometry: &[(eta_ir::registry::Port, Vec<u8>)],
) -> u64 {
    bind_device_geometry(shell, device_geometry_registration(), geometry)
}

/// The same over a registration the caller states, so that a fixture variant
/// (an unseeded page channel, say) reaches the shell down this same path.
fn bind_device_geometry(
    shell: &mut Shell,
    registration: engine::program::ProgramRegistration,
    geometry: &[(eta_ir::registry::Port, Vec<u8>)],
) -> u64 {
    let mut seeds = guest_seeds(&registration);
    for (port, cell) in geometry {
        let Some(channel) = port_channel(&registration, *port) else {
            continue;
        };
        // A channel the fixture declared UNSEEDED takes no seed here either:
        // stating one would be publishing the very cell the caller withheld.
        if !seeds.iter().any(|(at, _)| *at == channel) {
            continue;
        }
        let seat = seeds
            .iter_mut()
            .find(|(at, _)| *at == channel)
            .map(|(_, cell)| cell)
            .unwrap_or_else(|| {
                panic!(
                    "the fixture's `{}` port reads channel {channel}, which declares no \
                     seed",
                    port.name()
                )
            });
        assert_eq!(
            seat.len(),
            cell.len(),
            "the `{}` port's channel {channel} holds {} wire byte(s) and this test \
             stated {}",
            port.name(),
            seat.len(),
            cell.len()
        );
        seat.copy_from_slice(cell);
    }
    let program = shell
        .register_program(&registration)
        .unwrap_or_else(|error| {
            panic!("the device-geometry fixture does not compile on this device: {error}")
        });
    let instance = shell
        .bind_program(
            program,
            &seeds,
            eta_exec::Extents {
                // Two rows, because the epilogue reads `[B, V]` of the logits
                // rectangle: a seat carved for one would leave the second
                // beam's row zero-filled.
                sampled_rows: GUEST_LANES as u32,
                ..eta_exec::Extents::default()
            },
            eta_ir::registry::GeometryClass::DeviceGeometry,
            &[],
        )
        .unwrap_or_else(|error| panic!("the device-geometry fixture does not bind: {error}"));
    // ANY CHANNEL A BIND CANNOT SEAT. A pooled pass declares none today — its
    // whole geometry is loop-carried — but the Track B lease's `fresh` arm is
    // a host-writer channel declared `NeedsFull`, and a fixture that grew one
    // would wedge at the readiness gate rather than fire. Zeros, because
    // nothing here consumes the value.
    let package = registration.launch.clone();
    for (at, declared) in package.channels.iter().enumerate() {
        if declared.host_role != eta_ir::container::HostRole::Writer {
            continue;
        }
        let lanes = declared
            .shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>()
            .max(1);
        let dtype = eta_exec::concrete_dtype(declared.dtype);
        let cell = vec![0u8; eta_exec::wire_cell_bytes(dtype, lanes)];
        shell
            .program_instance(instance)
            .expect("the bind landed")
            .expect("the instance is bound")
            .publish(at as u32, &cell)
            .unwrap_or_else(|error| panic!("publishing the host-writer channel {at}: {error}"));
    }
    instance
}

/// `lanes` little-endian `u32` words, as a wire cell.
fn u32_cell(lanes: &[u32]) -> Vec<u8> {
    lanes.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// `lanes` little-endian `i32` words, as a wire cell.
fn i32_cell(lanes: &[i32]) -> Vec<u8> {
    lanes.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// A `Bool` wire cell: one BIT per lane, LSB-first inside each byte.
///
/// The packing this plane's rings hold, which is `eta_exec`'s `encode_wire`
/// and is the thing `program::ports::read_bool_cell` has to undo. Spelled here
/// too, on purpose: a test that shared the shell's own unpacking could not
/// tell a wrong order from a consistent one.
fn bool_cell(lanes: &[bool]) -> Vec<u8> {
    let mut packed = vec![0u8; lanes.len().div_ceil(8)];
    for (at, &kept) in lanes.iter().enumerate() {
        if kept {
            packed[at / 8] |= 1u8 << (at % 8);
        }
    }
    packed
}

/// The geometry cells for a one-lane instance whose sequence holds `have`
/// tokens, embeds `token` as its next row, and addresses its own page run.
///
/// **EVERY NUMBER HERE IS THE ONE THE SEAT WOULD HAVE DERIVED**, which is the
/// whole design of the parity claim below: the host twin's page CSR, write
/// descriptor and causal reach are computed by `store::kv::geometry_with` from
/// `have + 1`, and these state the same thing through nine ports instead. A
/// test whose device geometry described a DIFFERENT fire would prove that the
/// ports were read and nothing about whether they were read correctly.
fn guest_geometry(have: u32, token: u32) -> Vec<(eta_ir::registry::Port, Vec<u8>)> {
    use eta_ir::registry::Port;
    let kv = have + 1;
    // The row keeps exactly the keys the append leaves readable, and drops the
    // pool's surplus — the "a mask may be LONGER" shape `crate::mask` clips.
    let mask: Vec<bool> = (0..GUEST_MASK_KEYS)
        .map(|key| (key as u32) < kv)
        .collect();
    vec![
        // One lane, one row: `[0, 1]`.
        (Port::EmbedIndptr, u32_cell(&[0, 1])),
        // The id this fire embeds — the one value no host could have known,
        // and on this fixture an `I32` channel rather than a `U32` one.
        (Port::EmbedTokens, i32_cell(&[token as i32])),
        (Port::KvLen, u32_cell(&[kv])),
        (Port::Positions, u32_cell(&[have])),
        // The page run in WORKING-SET space: `0 .. GUEST_PAGE_RUN`, which is
        // what `ws.reserve` hands a guest and what it holds forever. Only the
        // live prefix is read (`ceil(kv / page_size)` of them); the rest is
        // the headroom a guest keeps ahead of itself.
        (
            Port::Pages,
            u32_cell(&(0..GUEST_PAGE_RUN).collect::<Vec<_>>()),
        ),
        (Port::PageIndptr, u32_cell(&[0, GUEST_PAGE_RUN])),
        // The write descriptor: which page of the run this row lands in and
        // where inside it. Derivable for one lane appending to its own tail —
        // which is exactly why the parity claim below can be an EQUALITY —
        // and not derivable at all for the `B` lanes of a beam sharing a pool,
        // which is why the ports exist.
        (Port::WSlot, u32_cell(&[have / GUEST_PAGE_SIZE])),
        (Port::WOff, u32_cell(&[have % GUEST_PAGE_SIZE])),
        (Port::AttnMask, bool_cell(&mask)),
    ]
}

/// **A DEVICE-GEOMETRY LANE ANSWERS WHAT ITS HOST-GEOMETRY TWIN ANSWERS.**
///
/// Two slots hold the same prefix and two lanes take one step over it. Lane 0
/// is bound in `GeometryClass::DeviceGeometry`: its row split, token, position,
/// extent, page run, write descriptor and mask are all cells on that
/// instance's rings, translated out of working-set space through
/// `Seated::translation` and used INSTEAD of the seat's arithmetic. Lane 1 is
/// the same step with the same token and the same mask stated on the
/// submission and the page table left to the shell.
///
/// Every number the guest states is the number the seat would have derived, so
/// the two rows must agree bit for bit. What that pins is the whole derivation
/// swap at once: a shell that ignored `pages` would attend another slot's
/// cache, one that ignored `w_slot`/`w_off` would append the row somewhere the
/// attention does not look, one that ignored `kv_len` would carve the CSR at
/// the wrong extent, and one that ignored `embed_indptr` would place zero rows
/// for a lane whose submission states none.
///
/// **AND THE SUBMISSION SIDE OF LANE 0 IS EMPTY ON PURPOSE**, because that is
/// what the runtime ships for this class: `Lane::tokens` is `vec![0; 0]` for
/// every lane of a pooled pass (`pipeline::fire::fire_device_geometry` builds
/// them off an all-zero `qo_indptr`), `KvDelta::pages` is empty because the
/// pages are in a channel, and `KvDelta::held` is zero because the runtime
/// could not know the extent either. A shell that read any of the three would
/// fire a zero-row lane.
#[test]
fn a_device_geometry_lane_answers_what_its_host_geometry_twin_answers() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the device-geometry smoke") else {
        return;
    };
    // The prefix is cut to the guest's own mask width: a mask SHORTER than the
    // extent it rides on is refused by name (`Fault::Mask`), so the fire this
    // test wants is one whose post-append extent the rectangle covers.
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let have = prompt.len().min(GUEST_MASK_KEYS - 1) as u32;
    let prefix = &prompt[..have as usize];
    assert!(have >= 2, "the prefix is too short to be a sequence");

    for slot in 0..2u32 {
        shell.open(slot).expect("the slot opens");
        let rows = shell
            .fire_seated(&[Seated::of(Lane {
                slot,
                word: word(have),
                tokens: prefix,
            })])
            .expect("the prefill fires");
        finite(&rows[0], "the prefill");
    }

    let next = prompt.get(have as usize).copied().unwrap_or(prompt[0]);
    let instance = device_geometry_instance(&mut shell, &guest_geometry(have, next));
    // THE TABLE THAT CROSSES THE TWO PAGE SPACES. Entry `i` is the pool page
    // backing the guest's relative index `i`, and the guest states relative
    // page 0 — so this is the shell's own block base for slot 0, quoted. The
    // runtime mints one per working set (`KvDelta::translation`); a test that
    // handed the identity would be testing nothing, because the identity is
    // exactly the bug the field exists to end.
    let paging = shell.paging();
    let translation: Vec<u32> = (0..GUEST_PAGE_RUN)
        .map(|page| {
            u32::try_from(paging.base(0) + u64::from(page)).expect("a pool page id")
        })
        .collect();
    // The step the twin takes on the submission, stated as the restriction the
    // guest's rectangle encodes: everything the append leaves readable.
    let kv = u64::from(have) + 1;
    let twin_mask = engine::fire::Masking::Extent(engine::fire::Mask::new(vec![0, kv as u32], kv));
    let stated = [next];

    let before = engine_metal::program::ports::resolved();
    let rows = shell
        .fire_attached(
            &[
                // The device lane. Its submission carries NO rows at all — the
                // split is `embed_indptr`'s — and no page table, no held count
                // and no positions, which is what the runtime ships for this
                // class.
                Seated {
                    translation: &translation,
                    mask: None,
                    ..Seated::of(Lane {
                        slot: 0,
                        word: masked_word(1),
                        tokens: &[],
                    })
                },
                // The twin: the same step, the same mask, the shell's table.
                Seated {
                    mask: Some(&twin_mask),
                    ..Seated::of(Lane {
                        slot: 1,
                        word: masked_word(1),
                        tokens: &stated,
                    })
                },
            ],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect("the device-geometry fire fires");
    let resolved = engine_metal::program::ports::resolved() - before;

    finite(&rows[0], "the device-geometry lane");
    finite(&rows[1], "the host-geometry twin");
    eprintln!(
        "prefix {have} token(s), extent {kv} — device argmax {}, twin {}; \
         envelopes resolved this fire: {resolved}",
        argmax(&rows[0]),
        argmax(&rows[1]),
    );
    assert_eq!(
        rows[0], rows[1],
        "the device-geometry lane did not answer what its host-geometry twin \
         answered over the same prefix, the same token and the same mask: the \
         geometry this fire ran was not the one the guest published"
    );
    assert_eq!(
        resolved, 1,
        "one attached instance is one envelope resolved: the host twin beside it \
         must cost no ring read at all"
    );
}

/// **A GUEST PAGE ID ITS TRANSLATION TABLE DOES NOT COVER IS REFUSED BY
/// NAME.**
///
/// The two page spaces are the whole hazard of this class. A guest holds
/// working-set-RELATIVE indexes — `reserve` hands back `0 .. n`, and an O(1)
/// fork is possible precisely because a relative index survives the copy that
/// moves the physical page under it — while everything past `serve::prepare`
/// is in the POOL's space. The runtime crosses between them for every other
/// class before it submits; for this one it ships the table and the crossing
/// happens in the engine. "Translate by identity" is the bug, and it is the
/// quiet one: every lane in the process would address pages `0, 1, ...` and
/// read back somebody else's cache, which is invisible under one guest and
/// wrong the moment two share a device.
///
/// So a table the index runs past is a refusal on the fire, before anything
/// launches, naming the index and the table's own size. An EMPTY table is the
/// same refusal, which is the point: a lane that states page references and
/// carries no table has not been given the crossing at all.
#[test]
fn a_device_page_id_past_its_translation_table_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the untranslatable-page refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let have = prompt.len().min(GUEST_MASK_KEYS - 1) as u32;
    shell.open(0).expect("the slot opens");
    let instance = device_geometry_instance(&mut shell, &guest_geometry(have, prompt[0]));

    let fault = shell
        .fire_attached(
            &[Seated {
                translation: &[],
                ..Seated::of(Lane {
                    slot: 0,
                    word: masked_word(1),
                    tokens: &[],
                })
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("a relative page index no table maps");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("working-set page 0"),
        "the refusal does not name the index that could not be translated: {said}"
    );
    assert!(
        said.contains("0 page(s)"),
        "the refusal does not name how much of a table this fire was handed: {said}"
    );
}

/// **AN EXTENT SHORTER THAN THE APPEND IT DESCRIBES IS REFUSED BY BOTH
/// NUMBERS.**
///
/// `kv_len` is a CHECK against the seat for a decode-envelope lane and a
/// SOURCE for this one: there is no seat here, so `have` is derived back from
/// the guest's extent as `kv_len - rows`, and every number the fire needs —
/// the page count, the last page's fill, the causal reach the mask is
/// intersected with — follows from it. An extent shorter than the rows this
/// fire adds has no such reading: the extent is AFTER the append, so a
/// subtraction would wrap and the lane would attend a length no cache holds.
#[test]
fn a_device_extent_shorter_than_its_own_append_is_refused_by_both_numbers() {
    let _guard = serialized();
    let Some((mut shell, _tokenizer)) = ready("the short-extent refusal") else {
        return;
    };
    shell.open(0).expect("the slot opens");
    // Every other port as it should be, and `kv_len` at zero: one row is
    // appended and the guest says nothing is readable afterwards.
    let mut geometry = guest_geometry(4, 1);
    for (port, cell) in &mut geometry {
        if *port == eta_ir::registry::Port::KvLen {
            *cell = u32_cell(&[0u32; GUEST_LANES]);
        }
    }
    let instance = device_geometry_instance(&mut shell, &geometry);
    let paging = shell.paging();
    let translation: Vec<u32> = (0..GUEST_PAGE_RUN)
        .map(|page| {
            u32::try_from(paging.base(0) + u64::from(page)).expect("a pool page id")
        })
        .collect();

    let fault = shell
        .fire_attached(
            &[Seated {
                translation: &translation,
                ..Seated::of(Lane {
                    slot: 0,
                    word: masked_word(1),
                    tokens: &[],
                })
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("an extent of zero over a fire that appends a row");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains("extent of 0"),
        "the refusal does not name what the port stated: {said}"
    );
    assert!(
        said.contains("1 row(s)"),
        "the refusal does not name what this fire adds: {said}"
    );
}

/// **AN UNPUBLISHED GEOMETRY IS A REFUSAL AND NOT A ZERO.**
///
/// A port's value for a fire is the committed front of its ring — the cell the
/// guest's own pass takes — so an instance whose page channel nothing has
/// published into has no such cell, and the cell at `head` then holds whatever
/// the allocation came with. A shell that read it would attend page zero, or
/// garbage, and never say so.
///
/// Two gates stand in front of it and either is a correct answer:
/// `Shell::admit_attachments` asks `Session::blocked_channel`, which reads the
/// `NeedsFull` requirement the program declares, and
/// `program::ports::cell_of`'s own `head == tail` catches a program that
/// declares no readiness on a channel it binds a port to. What must not happen
/// is a fire, and the subject is the one fact this class adds to the envelope:
/// the page run.
#[test]
fn a_device_geometry_the_guest_has_not_published_is_refused_by_name() {
    let _guard = serialized();
    let Some((mut shell, tokenizer)) = ready("the unpublished-geometry refusal") else {
        return;
    };
    let prompt: Vec<u32> = tokenizer.encode(PROMPT);
    let have = prompt.len().min(GUEST_MASK_KEYS - 1) as u32;
    shell.open(0).expect("the slot opens");
    // Every port stated, and the PAGE run's channel declared with no seed at
    // all — so nothing has ever committed into it and nothing in the pass
    // publishes one either.
    let registration = device_geometry_registration_with(false);
    let pages = port_channel(&registration, eta_ir::registry::Port::Pages)
        .expect("the fixture binds `pages` to a channel");
    let instance = bind_device_geometry(
        &mut shell,
        registration,
        &guest_geometry(have, prompt[0]),
    );
    let paging = shell.paging();
    let translation: Vec<u32> = (0..GUEST_PAGE_RUN)
        .map(|page| u32::try_from(paging.base(0) + u64::from(page)).expect("a pool page id"))
        .collect();

    let fault = shell
        .fire_attached(
            &[Seated {
                translation: &translation,
                ..Seated::of(Lane {
                    slot: 0,
                    word: masked_word(1),
                    tokens: &[],
                })
            }],
            &[engine_metal::Attached {
                lane: 0,
                instance,
                at: engine::fire::Boundary::Epilogue,
            }],
        )
        .expect_err("a page run over a ring nothing has published into");
    let said = fault.to_string();
    eprintln!("refusal: {said}");
    assert!(
        said.contains(&format!("channel {pages}")),
        "the refusal does not name the channel the page run reads: {said}"
    );
}
