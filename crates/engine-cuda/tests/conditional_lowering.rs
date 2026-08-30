//! P3 at the device: **what a conditional node would save on this shell, and
//! the two reasons it is zero** (palo design §4, build log 27).
//!
//! The case conditionals exist for is an all-decode fire: the prefill windows
//! and the masked windows are empty, and design §4 says their launches are
//! ~1 µs each and a conditional could skip them. This suite goes and looks,
//! and the answer is that there is nothing there to skip.
//!
//! 1. **The artifact this shell bakes holds no conditional.** P3 reads the
//!    profile the shell measured and chooses `AlwaysLaunch` on every region of
//!    every SKU this shell can load — so every golden in this directory is
//!    byte-identical by construction and not by luck. The one region in the
//!    whole catalog that clears both gates is qwen36-27b's MTP head, and that
//!    SKU does not fit this card.
//! 2. **An empty window's launches are already not in the graph.** Build log
//!    10's exec key is the per-class `(rows, lanes)` vector, zeros included,
//!    and the walk skips a zero-row region at RECORD time — so an all-decode
//!    exec holds only the all-decode launches. Measured here as the captured
//!    node count of one key against another.
//! 3. **A conditionalized artifact RECORDS, and says what the eager walk
//!    says.** An eager walk may ignore the bracket and be right; a recording
//!    one may not, and since the graphs wave it does not have to — it places a
//!    real `CU_GRAPH_NODE_TYPE_CONDITIONAL` node and captures the region's
//!    launches into its child graph, with the predicate stored by a kernel
//!    reading the region's row count off the device. This claim is the
//!    CORRECTNESS one: same prompt, same greedy tokens, conditionals on and
//!    conditionals off, over an artifact whose every windowed region is behind
//!    a node.
//!
//! # Gating
//!
//! As `graph_replay.rs`: skipped at run time when the machine, the checkpoint
//! or the tokenizer is missing, rather than `#[ignore]`d.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p engine-cuda --release --features cuda-13 \
//!     --test conditional_lowering -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::{CompiledModel, Budget, DeviceProfile, Lowering, Phase, compile};
use model_dsl::{Classify, Platform, Request};

/// The catalog row this suite serves, as `graph_replay` serves it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt.
const PROMPT: &str = "The capital of France is";

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

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn budget() -> Budget {
    Budget::new(4, 256)
}

/// A shell loaded with a profile of the caller's choosing.
fn ready(what: &str, profile: Option<DeviceProfile>) -> Option<(Shell, tokenizer::Tokenizer)> {
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

    let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: budget(),
        patches: None,
        profile,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
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
    Some((shell, tokenizer))
}

/// How many capture-phase nodes stand in regions this composition has no rows
/// for — **the empty launches a conditional would have to skip to be worth
/// anything.**
///
/// A pure function of the artifact and the set of classes present, which is
/// why it needs no device: a region's window is its mask's rows, and a mask
/// disjoint from the fire's classes has none.
fn empty_launches(compiled: &CompiledModel, present: &[usize]) -> (usize, usize) {
    let mut empty = 0;
    let mut live = 0;
    for region in &compiled.regions {
        if region.phase != Phase::Capture {
            continue;
        }
        let nodes = region.nodes.clone().count();
        if present.iter().any(|class| region.mask.contains(*class)) {
            live += nodes;
        } else {
            empty += nodes;
        }
    }
    (empty, live)
}

/// Which class a lane of this word lands in.
fn class_of(compiled: &CompiledModel, word: u64) -> usize {
    compiled
        .classes
        .class_of(word & compiled.classes.mask)
        .expect("the sweep names this word's class")
}

/// Claim 1, and it needs no device: **the artifact this shell bakes for a SKU
/// it can load holds no conditional region**, so every golden beside this file
/// is unmoved by P3 for a reason rather than by hope.
#[test]
fn the_shells_own_artifact_holds_no_conditional_region() {
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    // The profile `serve.rs` builds when a caller states none, with an L40S's
    // SM count in it — the one number the shell measures.
    let profile = DeviceProfile {
        sms: 142,
        exclusive: engine_cuda::EXCLUSIVE.iter().map(|op| (*op).to_string()).collect(),
        ..DeviceProfile::default()
    };
    let compiled = compile(&trace, &budget(), &profile).expect("the SKU bakes");
    let conditional: Vec<usize> = compiled
        .regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.lowering != Lowering::AlwaysLaunch)
        .map(|(at, _)| at)
        .collect();
    assert!(
        conditional.is_empty(),
        "P3 chose {conditional:?} on `{SKU}`, and every golden in this directory \
         was written against an always-launch artifact",
    );
}

/// Claim 2 — **THE MEASUREMENT.** An all-decode fire against a fire with a
/// prefill lane beside it: how many launches the empty windows would have
/// held, and how many the captured graph actually holds.
#[test]
fn an_all_decode_graph_already_holds_no_empty_launch() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the empty-launch census", None) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    // The host half: what the composition says, off the artifact alone.
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let compiled = compile(
        &trace,
        &budget(),
        &DeviceProfile {
            sms: 142,
            exclusive: engine_cuda::EXCLUSIVE
                .iter()
                .map(|op| (*op).to_string())
                .collect(),
            ..DeviceProfile::default()
        },
    )
    .expect("the SKU bakes");
    let decode = class_of(&compiled, word(1));
    let prefill = class_of(&compiled, word(prompt.len() as u32));
    let (empty_alone, live_alone) = empty_launches(&compiled, &[decode]);
    let (empty_mixed, live_mixed) = empty_launches(&compiled, &[decode, prefill]);
    eprintln!(
        "composition census on `{SKU}`:\n  \
         all-decode   {live_alone} live launches, {empty_alone} empty\n  \
         decode+prefill {live_mixed} live launches, {empty_mixed} empty",
    );
    assert!(
        empty_alone > 0,
        "an all-decode fire has no empty window, and there is nothing to measure",
    );

    // The device half: what the captured graphs actually hold.
    shell.set_mode(Graphs::On);
    shell.open(0).expect("slot 0 opens");
    let seeded = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        }])
        .expect("the prefill fires");
    let mut carried = argmax(&seeded[0]);
    let mut millis: Vec<f64> = Vec::new();
    for _ in 0..8 {
        let at = Instant::now();
        let out = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[carried],
            }])
            .expect("the decode fires");
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        carried = argmax(&out[0]);
    }
    let alone = shell.graph_stats();
    let decode_nodes = alone.nodes;
    let decode_ms = {
        let warm = &millis[millis.len() / 2..];
        warm.iter().sum::<f64>() / warm.len() as f64
    };

    // A second lane, prefilling beside the first's decode: a different key, a
    // different exec, and the launches the all-decode one did not hold.
    shell.open(1).expect("slot 1 opens");
    let second = tokenizer.encode("The largest planet in the solar system is");
    for _ in 0..3 {
        let out = shell
            .fire(&[
                Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &[carried],
                },
                Lane {
                    slot: 1,
                    word: word(second.len() as u32),
                    tokens: &second,
                },
            ])
            .expect("the mixed fire fires");
        carried = argmax(&out[0]);
        // Re-opening resets the slot's kv length, so the next fire presents
        // the same key rather than a longer prefill.
        shell.open(1).expect("slot 1 reopens");
    }
    let mixed = shell.graph_stats();

    eprintln!(
        "captured graphs: all-decode {decode_nodes} nodes at {decode_ms:.3} ms/fire, \
         decode+prefill {} nodes; {} execs, {} captures",
        mixed.nodes, mixed.execs, mixed.captures,
    );

    // **THE RULING.** The all-decode exec holds strictly fewer nodes than the
    // mixed one, which is what says the prefill launches are not in it — the
    // walk skipped them at record time and the key remembers which
    // composition it was recorded for (build log 10). There are no empty
    // launches on the replay path, so a conditional node has nothing to skip
    // and the ms it would save is 0.000 by arithmetic rather than by
    // measurement.
    assert!(
        decode_nodes < mixed.nodes,
        "the all-decode exec holds {decode_nodes} nodes and the mixed one \
         {} — if they were equal, the empty windows WOULD be in the graph and \
         a conditional would have something to skip",
        mixed.nodes,
    );
}

/// Claim 3 — **THE CORRECTNESS GATE.** A conditionalized artifact reaches the
/// capture, records real conditional nodes, and replays the tokens the eager
/// walk produced.
///
/// The profile is the lever, exactly as `Knobs::side_streams` is P6's: costs
/// are DATA the caller passes, so a deployment that states a zero fatness
/// floor gets conditionals on every windowed region. That is a far heavier
/// artifact than anything P3 chooses at the default profile — dozens of nodes
/// rather than the catalog's one — which is exactly what makes it the right
/// instrument: if the bracket were wrong anywhere, this is where it shows.
///
/// **THE EAGER ARM IS THE REFERENCE AND IT IS TAKEN FIRST**, on the same
/// shell, from the same slot, over the same prompt. Two shells would be two
/// loads of the same weights and a second source of disagreement; one shell
/// switching modes is the smallest instrument that can tell a conditional
/// apart from everything else in a fire.
#[test]
fn a_conditionalized_artifact_records_its_nodes_and_replays_what_the_eager_walk_said() {
    let _serial = serialized();
    let forcing = DeviceProfile {
        fat_region_us: 0.0,
        cond_fixed_us: 0.5,
        cond_per_arm_us: 0.0,
        ..DeviceProfile::default()
    };
    // How many regions this profile actually conditionalizes — printed, so
    // that a run which forced NOTHING is visible rather than silently vacuous.
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let forced = {
        let trace = trace(Platform::Cuda);
        let compiled = compile(&trace, &budget(), &forcing).expect("the forced bake bakes");
        compiled
            .regions
            .iter()
            .filter(|region| region.lowering != Lowering::AlwaysLaunch)
            .count()
    };
    assert!(
        forced > 0,
        "this profile conditionalized nothing, so the gate below would pass          over an always-launch artifact and prove nothing",
    );

    let Some((mut shell, tokenizer)) = ready("the conditional capture", Some(forcing)) else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);

    /// How many tokens each arm generates. Long enough that a wrong window or
    /// a body that ran when it should not have has somewhere to show up, short
    /// enough that a capture happens well inside it.
    const STEPS: usize = 12;

    // EAGER FIRST, and it is the reference. The walk's zero-row rule decides
    // what the conditional decides, so an eager pass over a conditionalized
    // artifact runs the same nodes over the same rows (design §4).
    shell.set_mode(Graphs::Off);
    shell.open(0).expect("slot 0 opens");
    let seeded = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        }])
        .expect("an eager fire ignores the bracket and computes");
    let mut carried = argmax(&seeded[0]);
    let mut eager = vec![carried];
    for _ in 0..STEPS {
        let out = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[carried],
            }])
            .expect("an eager decode fires");
        carried = argmax(&out[0]);
        eager.push(carried);
    }

    // **AND NOW THE SAME WALK, WRITTEN DOWN.** The prefill re-seeds the slot,
    // two warm fires pass, and the third records — conditional nodes and all.
    shell.set_mode(Graphs::On);
    shell.open(0).expect("slot 0 reopens");
    let seeded = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        }])
        .expect("a recorded prefill fires");
    let mut carried = argmax(&seeded[0]);
    let mut recorded = vec![carried];
    for _ in 0..STEPS {
        let out = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[carried],
            }])
            .expect("a recorded decode fires");
        carried = argmax(&out[0]);
        recorded.push(carried);
    }

    let stats = shell.graph_stats();
    eprintln!(
        "{forced} regions conditionalized; {} captures, {} execs, {} nodes
           eager    {eager:?}
  recorded {recorded:?}",
        stats.captures, stats.execs, stats.nodes,
    );
    assert!(
        stats.captures > 0,
        "no capture happened, so nothing recorded a conditional node and the          two arms are the same eager walk twice",
    );
    assert_eq!(
        eager, recorded,
        "the recorded arm of a conditionalized artifact answered different          tokens from the eager one",
    );
}
