//! **THE DOOR M-2 AND M-4 SHARE**: a SKU whose plan bakes a conditional,
//! served with `graphs = on`, answering what `graphs = off` answers
//! (`.wiki/alto/multimodal.md` §10.2).
//!
//! A drafting plan bakes exactly one conditional — the draft head, which is
//! `model-compiler`'s `which_skus_get_a_conditional` saying "the catalog's
//! answer at this profile is 1, the MTP head and nothing else" — and until the
//! graphs wave the recorded path refused it by name. So no drafting SKU was
//! servable on cuda under `graphs = on`: not `qwen36-27b`, which is M-2's "MTP
//! must keep serving", and not the EAGLE overlay, which is M-4's identity
//! gate. Both rows were reachable only eagerly.
//!
//! **THE ARTIFACT IS THE CHEAP ONE ON PURPOSE.** qwen36-27b is 54 GB and does
//! not fit this card; `qwen35-d0.8b-eagle-bf16-kv-bf16` is the same MECHANISM
//! at 1.7 GiB — a base checkpoint with a synthetic head overlaid by
//! `pie model import <base> --aux <head>` (`tests/eagle/synthesize_head.py`
//! builds the head) — and P3 stamps a conditional on its head region at the
//! DEFAULT profile, 23 nodes and 564 µs against a 250 µs floor. That is the
//! real bake and not a forced one, which is what makes this the gate and
//! `conditional_lowering.rs`'s forcing profile the instrument.
//!
//! What is asserted, and it is one thing: the two arms produce the same greedy
//! tokens. **The eager arm is the reference** — an eager walk ignores the
//! conditional bracket and is right to (design §4, the zero-row rule decides
//! the same thing at the same instant), so it is the answer the recorded arm
//! owes.
//!
//! ```text
//! PIE_EAGLE_SNAPSHOT=<dir with the overlay .zt and a tokenizer.json> \
//!   cargo test -p engine-cuda --release --features cuda-13 \
//!     --test a_drafting_sku_serves_under_graphs_on -- --nocapture
//! ```
//!
//! # Gating
//!
//! As `graph_replay.rs`: skipped at run time when the machine or the artifact
//! is missing, rather than `#[ignore]`d. The overlay is not in the hugging
//! face cache and has no canonical home, so it is named by environment and
//! nothing is guessed.

use std::path::{Path, PathBuf};

use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::{Budget, Lowering, compile, DeviceProfile};
use model_dsl::{Classify, Platform, Request};

/// The overlay row: `-eagle` in front of the row it drafts for.
const SKU: &str = "qwen35-d0.8b-eagle-bf16-kv-bf16";

const PROMPT: &str = "The capital of France is";

/// How many greedy steps each arm takes. Long enough that the capture happens
/// well inside it — a key is recorded on its third fire — and that a body
/// which ran when it should not have has somewhere to show up.
const STEPS: usize = 12;

fn budget() -> Budget {
    Budget::new(4, 256)
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

fn snapshot() -> Option<PathBuf> {
    let stated = std::env::var("PIE_EAGLE_SNAPSHOT").ok()?;
    let path = PathBuf::from(stated);
    (path.join("tokenizer.json").exists()).then_some(path)
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

fn shell(checkpoint: &Path, container: &Path) -> Shell {
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(container).expect("the overlay opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the overlay's import contract fits its own checkpoint");
    drop(source);

    Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint,
        budget: budget(),
        patches: None,
        // **THE PROFILE THE SHELL MEASURES**, not a forced one: this gate is
        // about the artifact a deployment actually bakes.
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the overlay loads")
}

/// `STEPS + 1` greedy tokens from one prompt, at whatever mode the shell is in.
fn greedy(shell: &mut Shell, prompt: &[u32]) -> Vec<u32> {
    shell.open(0).expect("slot 0 opens");
    let seeded = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    let mut carried = argmax(&seeded[0]);
    let mut out = vec![carried];
    for _ in 0..STEPS {
        let step = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[carried],
            }])
            .expect("a decode fires");
        carried = argmax(&step[0]);
        out.push(carried);
    }
    out
}

/// **THE GATE.** The drafting row's own artifact, both modes, one answer.
#[test]
fn the_eagle_overlay_answers_the_same_tokens_recorded_as_it_does_eagerly() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping: no EAGLE overlay — set PIE_EAGLE_SNAPSHOT to a directory \
             holding the `--aux` import's .zt and a tokenizer.json"
        );
        return;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping: {checkpoint:?} holds no tensor container");
        return;
    };

    // **THE PRECONDITION, ASKED FIRST AND WITHOUT A DEVICE.** If this row
    // stopped baking a conditional, everything below would pass while testing
    // nothing — a graphs=on/off identity over an always-launch artifact is
    // `graph_replay`'s claim and not this one.
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let baked = compile(&trace(Platform::Cuda), &budget(), &DeviceProfile::default())
        .expect("the overlay's plan bakes");
    let conditional: Vec<usize> = baked
        .regions
        .iter()
        .enumerate()
        .filter(|(_, region)| region.lowering != Lowering::AlwaysLaunch)
        .map(|(at, _)| at)
        .collect();
    assert_eq!(
        conditional.len(),
        1,
        "this row is supposed to bake exactly one conditional — its draft head \
         — and it baked {conditional:?}",
    );

    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the tokenizer loads");
    let prompt = tokenizer.encode(PROMPT);

    let mut shell = shell(&checkpoint, &container);
    shell.set_mode(Graphs::Off);
    let eager = greedy(&mut shell, &prompt);

    shell.set_mode(Graphs::On);
    let recorded = greedy(&mut shell, &prompt);

    let stats = shell.graph_stats();
    eprintln!(
        "`{SKU}`: region {} is baked `If`; {} captures, {} execs, {} nodes\n  \
         eager    {eager:?}\n  recorded {recorded:?}",
        conditional[0], stats.captures, stats.execs, stats.nodes,
    );
    assert!(
        stats.captures > 0,
        "the recorded arm never captured, so the conditional was never \
         recorded and both arms are the same eager walk",
    );
    assert_eq!(
        eager, recorded,
        "the drafting row answered different tokens with graphs on",
    );
}
