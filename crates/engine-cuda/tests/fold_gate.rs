//! The fold's gate (`Knobs::fold`, `.wiki/palo/cuda-abi.md` §7 steps 4-5):
//! what the keyed path says, the folded path must say — token for token,
//! over a real checkpoint, with the fold's own counters saying it actually
//! folded rather than quietly running the arm it is diffed against.
//!
//! **WHY A TOKEN DIFF, WHEN THE BINDING IS A RESTATEMENT OF A REAL CAPTURE.**
//! The fold's construction argues identity — the applied exec holds exactly
//! the launches a keyed capture of the same fire would hold, plus disabled
//! nodes that do not run — but construction cannot check the two things the
//! fold adds: that the segment ALIGNMENT paired every template node with the
//! launch it stands for (a wrong pairing computes, forever), and that a
//! disabled node really contributes nothing. Both are visible only in the
//! numbers, so the numbers are diffed where they leave the machine: the
//! token stream.
//!
//! Step 4's claims:
//!
//! 1. **identity** — sixteen greedy decode steps, keyed then folded, one
//!    load, byte-identical tokens, and `folds >= 1` so the comparison is not
//!    the arm against itself. The decode fires' bindings hold the template's
//!    prefill windows DISABLED, so this is also the disabled-library-node
//!    correctness case: an all-decode fire through the folded exec against
//!    the keyed all-decode graph.
//! 2. **the enable machinery** — compositions ALTERNATE (decode-only against
//!    decode-beside-prefill, same bucket), so every fire flips the windows
//!    the previous one held. Token-identical to the keyed path's same fires.
//! 3. **revisits do not re-capture** — once both compositions are bound,
//!    `throwaways` stops moving and `rebinds` counts instead. Watched
//!    through the counters, because "it did not capture" is not a property
//!    any output has.
//! 4. **the ledger** — ms/fire folded against keyed for steady decode, and
//!    the counts the fold exists to move: captures and resident execs.
//! 5. **the steady mixed column** — one repeated mixed composition, whose
//!    binding disables NOTHING, token-identical and timed. It is the
//!    attribution instrument for the ms/fire columns: with the streams off
//!    it measures exact parity (4.511 against 4.511 on this SKU), which
//!    pins the two real costs by elimination — the all-decode gap
//!    (~0.16 ms) is the DISABLED-node dispatch tax at ~1.3 µs/node on real
//!    nodes (the PoC's 0.24 was synthetic kernels), and the mixed
//!    streams-on gap (0.29 ms AT STEP 4, since reclaimed — see claim 6) was
//!    the P6 fork overlap a serially captured template forfeited. Neither
//!    is a host-updated-exec penalty, which measured zero.
//!
//! Step 5 adds four more (`.wiki/palo/cuda-abi.md` §6d):
//!
//! 6. **the mixed gap is closed** — the template captures FORKED (the
//!    per-stream frontier census), so the same steady-mixed column now
//!    measures parity WITH the streams on: 4.217 folded against 4.223
//!    keyed on this SKU. The old assertion surface (claim 5's test) is
//!    unchanged; the number it prints is the evidence.
//! 7. **the nine-cell gate** — eager, keyed, folded(+pipeline), each over
//!    steady decode, steady mixed and alternating compositions, one load,
//!    tokens byte-identical per workload across all three modes.
//! 8. **the pipeline** — alternating compositions rebind every fire under
//!    step 4's fold; the ping-pong pair turns those into swaps (zero host
//!    writing), and a THREE-composition rotation — which two seats cannot
//!    hold — moves the rebind off the critical path exactly when the
//!    caller states the next fire (`Shell::expect`).
//! 9. **the disable policy** — `all` against `library` (zero-formed pie
//!    nodes stay enabled, empty), token-identical, measured at parity;
//!    `all` ships.
//!
//! # Gating
//!
//! As `serve_smoke.rs`: skipped at run time when the machine, the checkpoint
//! or the tokenizer is missing, rather than `#[ignore]`d.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine-cuda --features cuda-13 --test fold_gate -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this suite serves, as `serve_smoke` serves it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt.
const PROMPT: &str = "The capital of France is";

/// How many decode fires follow the prefill — `graph_replay`'s number, for
/// `graph_replay`'s reasons: past the warm fires it is a steady state, and it
/// crosses a page boundary under an exec bound before the crossing.
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
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

/// A loaded shell, or `None` and a sentence saying what was missing.
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

    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let mut shell = Shell::load(Boot {
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
        // Both arms below run under On; the load states it once so a test
        // that forgot would still be diffing two graph paths, not eager.
        graphs: Graphs::On,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        // F1's depth, kept: these gates fire one step at a time and
        // read its numbers, so a deeper ring would carve slots nothing
        // claims. `Runahead::of` is the door a deployment comes through.
        runahead: engine::runahead::Runahead::F1,
        // The warm-boot weight artifact cache is off for a gate: a test
        // that shared one would be asserting about the last run.
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    // The fold is stated per ARM by the tests (`set_fold`). It cannot be
    // inherited any more — `Knobs::fold` above is this `Boot`'s own word, and
    // there is no environment left to inherit it from (alto wave P) — but the
    // arm is still stated here, because what this file diffs is the keyed path
    // against the folded one and both start from the same load.
    shell.set_fold(false);
    Some((shell, tokenizer))
}

/// One prefill and `STEPS` greedy decodes in slot 0, in whatever mode and
/// fold arm the shell is in. Returns the tokens and per-decode milliseconds.
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

/// The mean of the warm half — the steady state, past warming and binding.
fn warm(millis: &[f64]) -> f64 {
    let warm = &millis[millis.len() / 2..];
    warm.iter().sum::<f64>() / warm.len() as f64
}

/// Claims 1 and 4: identity on steady decode, and the ledger. The decode
/// bindings disable the template's prefill windows, so the identity here IS
/// the disabled-library-node case — an all-decode fire through the folded
/// exec against the keyed all-decode graph, byte for byte.
#[test]
fn a_folded_fire_says_token_for_token_what_a_keyed_fire_says() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the fold A/B") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // ONE LOAD, TWO ARMS — two loads would be two residencies and two tuner
    // histories, and a difference could be either (`set_mode`'s argument,
    // inherited whole).
    shell.set_fold(false);
    let (keyed, keyed_ms) = run(&mut shell, &prompt);

    shell.set_fold(true);
    let (folded, folded_ms) = run(&mut shell, &prompt);

    let stats = shell.graph_stats();
    let fold = shell.fold_stats();
    eprintln!(
        "decode ms/fire (warm half of {STEPS}): keyed {:.3}  folded {:.3}",
        warm(&keyed_ms),
        warm(&folded_ms),
    );
    eprintln!(
        "keyed: {} captures ({:.1} ms), {} replays, {} execs resident",
        stats.captures, stats.capture_millis, stats.replays, stats.execs,
    );
    eprintln!("{fold}");
    eprintln!(
        "continuations: keyed {:?} / folded {:?}",
        tokenizer.decode(&keyed, false),
        tokenizer.decode(&folded, false),
    );

    assert!(
        fold.folds >= 1,
        "no fire ran through a folded exec, so this test compared the keyed \
         path against itself; the refusals above say why: {fold}"
    );
    assert!(
        fold.disabled > 0,
        "the decode binding disabled nothing, so the template held no \
         absent-window nodes and the enable machinery went untested: {fold}"
    );
    assert_eq!(
        keyed, folded,
        "the folded exec disagreed with the keyed path it restates: keyed \
         {:?} against folded {:?}",
        tokenizer.decode(&keyed, false),
        tokenizer.decode(&folded, false),
    );
}

/// One carried decode lane, with a fresh prefill lane re-seated beside it on
/// every ODD step — compositions alternate inside one bucket, which is what
/// makes every fire flip the enables (or, under the pipeline, turn the
/// ping-pong pair) the previous one set. Returns the decode lane's tokens
/// and per-step milliseconds.
fn alternating(
    shell: &mut Shell,
    carried: &[u32],
    fresh: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<f64>) {
    shell.open(0).expect("slot 0 opens");
    let seated = shell
        .fire(&[Lane {
            slot: 0,
            word: word(carried.len() as u32),
            tokens: carried,
        }])
        .expect("the carried prefill fires");
    let mut decode = vec![argmax(&seated[0])];
    let mut millis = Vec::with_capacity(steps);
    for step in 0..steps {
        let fed = [*decode.last().expect("a step has a last token")];
        let mixed = step % 2 == 1;
        let at = Instant::now();
        let out = if mixed {
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
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        decode.push(argmax(&out[0]));
    }
    (decode, millis)
}

/// Claims 2 and 3: alternating compositions through one folded exec are
/// token-identical to the keyed path's same fires, and once both are bound,
/// revisiting captures nothing — the counters say rebind, not throwaway.
#[test]
fn an_absent_window_is_an_enable_bit_and_a_revisit_is_not_a_capture() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the fold's enable machinery") else {
        return;
    };
    const ALTERNATIONS: usize = 12;
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");

    shell.set_fold(false);
    let (keyed, _) = alternating(&mut shell, &carried, &fresh, ALTERNATIONS);

    shell.set_fold(true);
    let (folded, _) = alternating(&mut shell, &carried, &fresh, ALTERNATIONS);

    let fold = shell.fold_stats();
    let keyed_stats = shell.graph_stats();
    eprintln!("{fold}");
    eprintln!(
        "the ledger, on a workload with more than one composition: keyed \
         path holds {} execs ({} captures, {:.1} ms); the fold holds {} \
         (every composition of the bucket is a binding, not an exec)",
        keyed_stats.execs, keyed_stats.captures, keyed_stats.capture_millis, fold.execs,
    );
    eprintln!(
        "alternating: keyed {:?} / folded {:?}",
        tokenizer.decode(&keyed, false),
        tokenizer.decode(&folded, false),
    );
    assert!(
        fold.folds >= 1,
        "no alternating fire ran through the folded exec: {fold}"
    );
    assert_eq!(
        keyed, folded,
        "the fold's enable flips computed a different continuation than the \
         keyed path's per-composition graphs",
    );

    // Claim 3: the steady alternation is rebinds or ping-pong swaps, never
    // captures. Both compositions are bound by now; six more flips must move
    // `rebinds + swaps` and leave `throwaways` exactly where it stands.
    let before = shell.fold_stats();
    let _ = alternating(&mut shell, &carried, &fresh, 6);
    let after = shell.fold_stats();
    eprintln!(
        "revisits: throwaways {} -> {}, rebinds {} -> {}, swaps {} -> {}, enable_flips {} -> {}",
        before.throwaways,
        after.throwaways,
        before.rebinds,
        after.rebinds,
        before.swaps,
        after.swaps,
        before.enable_flips,
        after.enable_flips,
    );
    assert!(
        after.rebinds + after.swaps > before.rebinds + before.swaps,
        "revisited compositions were neither re-bound nor swapped: {after}"
    );
    // The re-run's own carried prefill (slot 0 reopens) is one solo-prefill
    // fire, and that composition reaches its binding fire here — one late
    // throwaway is legitimate. A throwaway per FLIP is the failure this
    // assertion exists for: proportional to six, not bounded by one.
    assert!(
        after.throwaways <= before.throwaways + 1,
        "a revisited composition paid a capture again: {after}"
    );
    assert!(
        after.enable_flips > before.enable_flips,
        "alternating compositions flipped no enables: {after}"
    );
}

/// One carried decode lane beside a re-seated prefill lane, EVERY step —
/// one composition, repeated, so past the binding every fire is a steady
/// folded launch. Returns the decode lane's tokens and per-fire millis.
fn steady_mixed(
    shell: &mut Shell,
    carried: &[u32],
    fresh: &[u32],
    steps: usize,
) -> (Vec<u32>, Vec<f64>) {
    shell.open(0).expect("slot 0 opens");
    let seated = shell
        .fire(&[Lane {
            slot: 0,
            word: word(carried.len() as u32),
            tokens: carried,
        }])
        .expect("the carried prefill fires");
    let mut decode = vec![argmax(&seated[0])];
    let mut millis = Vec::with_capacity(steps);
    for step in 0..steps {
        let fed = [*decode.last().expect("a step has a last token")];
        shell.open(1).expect("slot 1 opens");
        let at = Instant::now();
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
            .unwrap_or_else(|why| panic!("mixed step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        decode.push(argmax(&out[0]));
    }
    (decode, millis)
}

/// The steady MIXED column: one composition whose binding disables NOTHING
/// (every template segment present), so its warm-half ms/fire against the
/// keyed replay isolates what a folded launch costs when no node is
/// disabled — the attribution instrument for the all-decode column's gap,
/// and one more identity surface while it is at it.
#[test]
fn a_steady_mixed_fire_folds_token_for_token_and_prices_the_zero_disabled_launch() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the fold's mixed steady state") else {
        return;
    };
    const MIXED: usize = 16;
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");

    shell.set_fold(false);
    let (keyed, keyed_ms) = steady_mixed(&mut shell, &carried, &fresh, MIXED);

    shell.set_fold(true);
    let (folded, folded_ms) = steady_mixed(&mut shell, &carried, &fresh, MIXED);

    let fold = shell.fold_stats();
    eprintln!(
        "mixed ms/fire (warm half of {MIXED}): keyed {:.3}  folded {:.3}  \
         (disabled under the last binding: {})",
        warm(&keyed_ms),
        warm(&folded_ms),
        fold.disabled,
    );
    eprintln!("{fold}");
    assert!(
        fold.folds >= 1,
        "no steady mixed fire ran through the folded exec: {fold}"
    );
    assert_eq!(
        keyed, folded,
        "the folded mixed fire disagreed with the keyed one: keyed {:?} \
         against folded {:?}",
        tokenizer.decode(&keyed, false),
        tokenizer.decode(&folded, false),
    );
}

/// The wave's step-5 gate (`.wiki/palo/cuda-abi.md` §7 step 5): one load,
/// three modes — eager, keyed replay, folded(+pipeline) — three workloads,
/// sixteen greedy steps each, tokens byte-identical across modes per
/// workload, and the nine-cell ms/fire table said out loud.
///
/// The eager column is the golden (`Graphs::Off`: no graph exists at all),
/// the keyed column is step 4's A/B arm, and the folded column carries
/// everything this step added: the forked template (the mixed column is
/// where the reclaimed 0.29 ms shows), the ping-pong pair (the alternating
/// column is where swaps replace rebinds), and whatever disable policy is
/// the shipped default.
#[test]
fn one_load_three_modes_three_workloads_agree_and_the_table_says_what_each_costs() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the step-5 nine-cell gate") else {
        return;
    };
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");

    // Mode setters, one load throughout — `set_mode`'s argument, three ways.
    let modes: [(&str, Graphs, bool); 3] = [
        ("eager ", Graphs::Off, false),
        ("keyed ", Graphs::On, false),
        ("folded", Graphs::On, true),
    ];

    let mut table: Vec<String> = Vec::new();
    let mut tokens_of: Vec<Vec<Vec<u32>>> = vec![Vec::new(); 3];
    for (at, (name, mode, fold)) in modes.iter().enumerate() {
        shell.set_mode(*mode);
        shell.set_fold(*fold);
        let (decode, decode_ms) = run(&mut shell, &carried);
        let (mixed, mixed_ms) = steady_mixed(&mut shell, &carried, &fresh, STEPS);
        let (alt, alt_ms) = alternating(&mut shell, &carried, &fresh, STEPS);
        table.push(format!(
            "{name}  decode {:.3}  mixed {:.3}  alternating {:.3}  ms/fire (warm half)",
            warm(&decode_ms),
            warm(&mixed_ms),
            warm(&alt_ms),
        ));
        tokens_of[at] = vec![decode, mixed, alt];
    }
    eprintln!("the nine cells, one load:");
    for row in &table {
        eprintln!("  {row}");
    }
    eprintln!("{}", shell.fold_stats());

    let fold = shell.fold_stats();
    assert!(
        fold.folds >= 1,
        "the folded row never ran through a folded exec: {fold}"
    );
    for (workload, name) in ["steady decode", "steady mixed", "alternating"]
        .iter()
        .enumerate()
    {
        assert_eq!(
            tokens_of[0][workload], tokens_of[1][workload],
            "keyed disagreed with eager on {name}"
        );
        assert_eq!(
            tokens_of[0][workload], tokens_of[2][workload],
            "folded disagreed with eager on {name}"
        );
    }
}

/// The pipelined revisit gate (step 5, T2): alternating two known
/// compositions so that under step 4's fold EVERY fire pays a critical-path
/// rebind, then the same fires with the pipeline on — the ping-pong pair
/// holds one composition per exec, so the rebind leaves the critical path
/// entirely (swaps move, rebinds stop), and the warm ms/fire says what that
/// bought. Tokens byte-identical across the arms, because the same bindings
/// land on an exec either way.
#[test]
fn the_pipeline_takes_the_rebind_off_the_critical_path() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the pipelined revisit gate") else {
        return;
    };
    const ALTERNATIONS: usize = 16;
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");
    shell.set_fold(true);

    shell.set_pipeline(false);
    let before_off = shell.fold_stats();
    let (unpipelined, off_ms) = alternating(&mut shell, &carried, &fresh, ALTERNATIONS);
    let after_off = shell.fold_stats();

    shell.set_pipeline(true);
    let before_on = shell.fold_stats();
    let (pipelined, on_ms) = alternating(&mut shell, &carried, &fresh, ALTERNATIONS);
    let after_on = shell.fold_stats();

    let off_rebinds = after_off.rebinds - before_off.rebinds;
    let on_rebinds = after_on.rebinds - before_on.rebinds;
    let on_swaps = after_on.swaps - before_on.swaps;
    eprintln!(
        "alternating ms/fire (warm half of {ALTERNATIONS}): unpipelined {:.3} \
         ({off_rebinds} rebinds, {:.1} us on the critical path)  pipelined {:.3} \
         ({on_rebinds} rebinds, {on_swaps} swaps)",
        warm(&off_ms),
        after_off.rebind_micros - before_off.rebind_micros,
        warm(&on_ms),
    );
    eprintln!("{after_on}");

    assert_eq!(
        unpipelined, pipelined,
        "the ping-pong computed a different continuation than the single exec"
    );
    assert!(
        off_rebinds >= (ALTERNATIONS as u64 - 4),
        "the unpipelined arm did not rebind per fire, so this gate is not \
         measuring the rebind cost: {after_off}"
    );
    assert!(
        on_swaps >= 2 && on_rebinds <= 2,
        "the pipelined arm still rebinds on the critical path instead of \
         turning the pair: {after_on}"
    );
}

/// The prebind gate (step 5, T2's other half): THREE compositions rotating
/// through one bucket, so two seats can never hold them all and every fire
/// would rebind — until the caller states the next fire (`Shell::expect`),
/// and the binding is applied to the idle exec under the previous fire's
/// execution. `prebinds` moves, critical-path `rebinds` stops, tokens match
/// the unhinted arm exactly.
#[test]
fn a_stated_next_fire_is_bound_under_the_running_one() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the prebind gate") else {
        return;
    };
    const ROTATIONS: usize = 18;
    let carried = tokenizer.encode(PROMPT);
    let long = tokenizer.encode("Water freezes at a temperature of");
    let short = tokenizer.encode("The answer is");
    shell.set_fold(true);

    /// One rotation position's lanes: decode-only, decode+long,
    /// decode+short — three signatures, one bucket.
    fn lanes_of<'a>(
        step: usize,
        fed: &'a [u32],
        long: &'a [u32],
        short: &'a [u32],
    ) -> Vec<Lane<'a>> {
        let decode = Lane {
            slot: 0,
            word: word(1),
            tokens: fed,
        };
        match step % 3 {
            0 => vec![decode],
            1 => vec![
                decode,
                Lane {
                    slot: 1,
                    word: word(long.len() as u32),
                    tokens: long,
                },
            ],
            _ => vec![
                decode,
                Lane {
                    slot: 1,
                    word: word(short.len() as u32),
                    tokens: short,
                },
            ],
        }
    }

    let mut arms: Vec<(Vec<u32>, u64, u64, u64)> = Vec::new();
    for hint in [false, true] {
        shell.open(0).expect("slot 0 opens");
        let seated = shell
            .fire(&[Lane {
                slot: 0,
                word: word(carried.len() as u32),
                tokens: &carried,
            }])
            .expect("the carried prefill fires");
        let mut decode = vec![argmax(&seated[0])];
        let before = shell.fold_stats();
        for step in 0..ROTATIONS {
            let fed = [*decode.last().expect("a last token")];
            if step % 3 != 0 {
                shell.open(1).expect("slot 1 opens");
            }
            if hint {
                // The NEXT fire's composition, stated before this one
                // launches — the tokens are placeholders, which is the
                // point: a hint is a composition, not contents.
                shell.expect(&lanes_of(step + 1, &fed, &long, &short));
            }
            let out = shell
                .fire(&lanes_of(step, &fed, &long, &short))
                .unwrap_or_else(|why| panic!("rotation step {step} fires: {why}"));
            decode.push(argmax(&out[0]));
        }
        shell.expect(&[]);
        let after = shell.fold_stats();
        arms.push((
            decode,
            after.rebinds - before.rebinds,
            after.prebinds - before.prebinds,
            after.swaps - before.swaps,
        ));
    }

    let (unhinted, unhinted_rebinds, _, _) = &arms[0];
    let (hinted, hinted_rebinds, hinted_prebinds, hinted_swaps) = &arms[1];
    eprintln!(
        "three compositions rotating: unhinted {unhinted_rebinds} rebinds; \
         hinted {hinted_rebinds} rebinds, {hinted_prebinds} prebinds, \
         {hinted_swaps} swaps"
    );
    eprintln!("{}", shell.fold_stats());
    assert_eq!(
        unhinted, hinted,
        "a hint changed the numbers; a hint may only change WHEN a binding \
         is applied, never what it says"
    );
    assert!(
        *hinted_prebinds >= 2,
        "no stated fire was prebound; the hint went nowhere"
    );
    assert!(
        hinted_rebinds < unhinted_rebinds,
        "the hints did not move rebinds off the critical path \
         (unhinted {unhinted_rebinds}, hinted {hinted_rebinds})"
    );
}

/// The disabled-policy measurement (step 5, T3; §6c finding 2). The steady
/// decode binding turns off the template's ~120 absent-window nodes; real
/// disabled nodes cost ~1.3 µs each at dispatch, and an enabled EMPTY pie
/// launch costs ~1 µs on the zero-row contract — so the `library` policy
/// keeps pie windowed nodes enabled with their fitted zero forms and
/// disables only the library residue. This test measures both policies on
/// one load, pins tokens across them (a zero form that is not actually
/// empty computes garbage, and the token diff is where that surfaces), and
/// prints the numbers the shipped default stands on.
#[test]
fn the_disable_policy_is_a_measurement_and_both_arms_say_the_same_tokens() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the disable-policy measurement") else {
        return;
    };
    let carried = tokenizer.encode(PROMPT);
    let fresh = tokenizer.encode("Water freezes at a temperature of");
    shell.set_fold(true);

    // Policy `all` (the default): decode fires disable every absent node.
    shell.set_fold_library(false);
    let (all_tokens, all_ms) = run(&mut shell, &carried);
    let all_stats = shell.fold_stats();

    // Policy `library`: a mixed fire first, so the next decode fire
    // re-applies its binding under the new policy (a policy flip takes
    // effect at the next application — the steady path launches without
    // touching the exec, by design).
    shell.set_fold_library(true);
    let _ = steady_mixed(&mut shell, &carried, &fresh, 1);
    let (library_tokens, library_ms) = run(&mut shell, &carried);
    let library_stats = shell.fold_stats();

    eprintln!(
        "steady decode ms/fire: disable-all {:.3} (disabled {})  \
         disable-library-only {:.3} (disabled {}, zeroed {})",
        warm(&all_ms),
        all_stats.disabled,
        warm(&library_ms),
        library_stats.disabled,
        library_stats.zeroed,
    );
    eprintln!("{library_stats}");

    assert_eq!(
        all_tokens, library_tokens,
        "a zero-formed pie node computed something: the policies must be \
         indistinguishable in the tokens"
    );
    assert!(
        library_stats.zeroed > 0,
        "the library policy zero-formed nothing — the arm probe fitted no \
         pie windowed node, so this test compared disable-all against \
         itself: {library_stats}"
    );
    assert!(
        library_stats.disabled < all_stats.disabled,
        "the library policy disabled as much as disable-all"
    );
}
