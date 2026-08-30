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
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// The same word, with the `masked` fact set — which is what puts a lane in
/// the class whose window runs `attention.masked`.
fn masked_word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, true)).word()
}

/// The same word with the `has_adapter` fact set — which is what puts a lane
/// in a class whose window runs `linear.lora_correct` (design §8, fact bit 1).
fn adapted_word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false).adapted(true)).word()
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
        trace,
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
