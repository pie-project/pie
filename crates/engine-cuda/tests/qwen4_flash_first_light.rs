//! **QWEN4'S FIRST LIGHT**: the shipped 4-bit `Qwen3.8-Flash-Next` artifact,
//! served whole on one 46 GB card.
//!
//! Everything the campaign built passes through this one boot: every
//! projection LANDS AS STORED — the eight-bit triplets seat as planes and
//! the dense affine gemm point dequantizes inside the dot (stored-form
//! wave; nothing is dequantized at load) — the expert banks seat as
//! three-plane affine groups under the residency ladder (the table is
//! ~68 GiB against a ~40 GiB device budget, so most of it streams from
//! pinned host), the 51-billion-row n-gram table lands packed at group 32
//! and is dequantized by the gather for the sixteen rows a token touches,
//! and the forward runs the gated residual, the PLE and the sigmoid-gated
//! delta net the parity gate proved against the reference.
//!
//! Three claims, `serve_smoke`'s own: finite, deterministic, and a
//! continuation that says something true. The continuation is OBSERVED,
//! THEN PINNED — the first run of this shell against this artifact is what
//! the expectation records.
//!
//! # And it PREPARES before it boots, because this SKU streams (§M wave M-3)
//!
//! `PIE_QWEN4_TIER_DIR` used to be an opt-in: point it somewhere and the
//! streamed load would write its tiers on the way past, leave it unset and the
//! load would stream, transform and serve without leaving anything behind.
//! There is no such road any more. A serving load that streams is served out
//! of a prepared serving artifact or it is REFUSED
//! (`engine_cuda::weights::Intent`), and this SKU's ~68 GiB table under a
//! 38 GiB device budget streams by construction — so the variable is a
//! REQUIREMENT, and the gate runs `Shell::prepare` against that directory
//! before it runs `Shell::load`. The first run of the gate pays the cold
//! landing in the prepare; every later run finds the artifact and pays a cut
//! and a verify.
//!
//! ```text
//! PIE_COMPILER_LAUNCHER=env PIE_QWEN4_TIER_DIR=/scratch/qwen4-tiers \
//!   cargo test -p engine-cuda --features cuda-13 \
//!   --release --test qwen4_flash_first_light -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen38-flash-mlxu4-kv-bf16";

const PROMPT: &str = "The capital of France is";

/// What a correct load produces here. OBSERVED, THEN PINNED — the first
/// light of this shell against this artifact answered ` Paris. The capital
/// of Germany is Berlin. …`, greedily, with a 38 GiB device budget
/// streaming the expert tiers from pinned host. The stored-form wave kept
/// the pin and improved the sentence: with the projections read in place
/// (and the n-gram gather's group recovered RIGHT — the byte-rectangle fix
/// in `weights::packed`) the same boot walks on through ` Italy is Rome.
/// The capital of Spain is Madrid.` at 8.5 tok/s, loading in ~590 s where
/// the dequantizing load took ~820.
///
/// **THOSE LOAD SECONDS ARE THE PREPARE'S NOW** (§M-3). The ~590 s was the
/// cold landing — the source walk, the transforms, the sink — and that work
/// has moved to `Shell::prepare`, which is the only door that still runs it.
/// The `Shell::load` timed below is a restore out of the artifact the prepare
/// left, so what it prints is a cut and a verify and nothing else. The TOKENS
/// are the pin, and they do not move with either.
const EXPECTED: &str = " Paris.";

const STEPS: usize = 24;

/// T0: room for the dense planes, the packed n-gram table and a working set
/// of experts, under the card's 46 GiB with kv and scratch beside it.
const DEVICE: u64 = 38 << 30;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_QWEN4_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = Path::new(&home).join(
        ".cache/huggingface/hub/models--pipenetwork--Qwen3.8-Flash-Next-MLX-mixed-4_8bit/snapshots",
    );
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn shards(snapshot: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .map(|dir| {
            dir.filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                name.ends_with(".safetensors").then_some(path)
            })
            .collect()
        })
        .unwrap_or_default();
    found.sort();
    found
}

fn word(query_len: u32) -> u64 {
    models::qwen_4::forward::Facts::of(&Request::new(query_len, false)).word()
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

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(0).expect("slot 0 opens");
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
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

#[test]
#[ignore = "real-hardware: needs a CUDA device, the ~99 GiB Qwen3.8-Flash-Next 4-bit \
            snapshot, and PIE_QWEN4_TIER_DIR pointing at ~102 GiB of scratch to prepare \
            the serving artifact into; run with `-- --ignored`"]
fn the_flash_artifact_prefills_decodes_and_says_something() {
    if !engine_cuda::device::present() {
        eprintln!("skipping qwen4 first light: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping qwen4 first light: no Qwen3.8-Flash-Next 4-bit snapshot in the \
             hugging face cache (set PIE_QWEN4_SNAPSHOT)"
        );
        return;
    };
    let shards = shards(&checkpoint);
    assert!(!shards.is_empty(), "{checkpoint:?} holds no tensor container");

    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let trace = models::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index_all(&shards).expect("the checkpoint's shards open as one");
    let contract = models::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits the shipped artifact");
    drop(source);

    let prospect = engine_cuda::weights::prospect(&trace, &contract, &checkpoint)
        .expect("the load prospects");
    let full = Plan::of(&trace, &prospect.planes, Budgets::uncapped())
        .expect("the flash text plans")
        .device_demand();
    eprintln!(
        "qwen38-flash: {:.1} GiB of table against a {:.1} GiB device budget",
        full as f64 / (1u64 << 30) as f64,
        DEVICE as f64 / (1u64 << 30) as f64,
    );
    let residency = Plan::of(
        &trace,
        &prospect.planes,
        Budgets {
            device: Some(DEVICE),
            host: None,
        },
    )
    .expect("the capped load plans");

    // **THE TIER DIRECTORY IS A REQUIREMENT NOW, NOT AN OPT-IN** (§M-3). It
    // used to read "point PIE_QWEN4_TIER_DIR at a directory with ~102 GiB free
    // and the streamed load writes (then reuses) its tiers", and every word of
    // that described the cold serving path this wave deleted. This plan
    // streams — 68 GiB of table against a 38 GiB budget — so with no directory
    // there is nothing for `Shell::load` to be served out of and nowhere for
    // `Shell::prepare` to write one. That is a SKIP and not a failure: the box
    // has the snapshot and the card, and what it is missing is a hundred
    // gigabytes of scratch, which is the same kind of missing as the snapshot
    // itself.
    let Some(tier_dir) = std::env::var("PIE_QWEN4_TIER_DIR").ok().map(PathBuf::from) else {
        eprintln!(
            "skipping qwen4 first light: this SKU streams its expert banks at a \
             {:.0} GiB device budget, and since §M-3 a streamed load serves only out \
             of a prepared serving artifact — set PIE_QWEN4_TIER_DIR to a directory \
             with ~102 GiB free for the gate to prepare into",
            DEVICE as f64 / (1u64 << 30) as f64,
        );
        return;
    };

    // Everything the prepare and the boot have to agree on, stated once: the
    // artifact's key is a function of the trace, the recipe and the ranking,
    // and two documents that differ in any field name two different files.
    let doc = |residency: Plan| Boot {
        residency,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 2,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: Some(tier_dir.as_path()),
    };

    // ── THE PREPARE, AND IT IS THE ONLY WRITER THERE IS. It runs the bake and
    //    the landing `Shell::load` used to run on its way to serving, writes
    //    the `.tiers` file and keeps nothing — no arena, no pools, no shell.
    //    Against a directory that already holds this deployment's artifact it
    //    opens it, cuts it and VERIFIES it instead, which is why running this
    //    gate twice costs a cold landing once.
    let prepared = Instant::now();
    Shell::prepare(doc(residency.clone())).expect("the flash deployment prepares");
    eprintln!(
        "prepared into {} in {:.1}s",
        tier_dir.display(),
        prepared.elapsed().as_secs_f64(),
    );

    let booted = Instant::now();
    let mut shell = Shell::load(doc(residency)).expect("the shell loads");
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

    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");

    let (first, millis) = run(&mut shell, &prompt);
    let text = tokenizer.decode(&first, false);
    let mean = millis.iter().sum::<f64>() / millis.len().max(1) as f64;
    eprintln!("greedy continuation: {text:?}");
    eprintln!("tokens: {first:?}");
    eprintln!("decode: {mean:.1} ms/token ({:.2} tok/s)", 1000.0 / mean);

    assert!(
        text.starts_with(EXPECTED),
        "greedy continuation of {PROMPT:?} was {text:?}, and the first light \
         answered {EXPECTED:?}"
    );

    let (again, _) = run(&mut shell, &prompt);
    assert_eq!(first, again, "twice is not once");
}
