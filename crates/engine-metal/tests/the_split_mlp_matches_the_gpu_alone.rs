#![cfg(target_vendor = "apple")]

//! A prompt long enough for the Neural Engine to take its share of every
//! dense MLP and wide projection reads the same as the GPU alone reads it.
//! The same 27B artifact is served twice: once with `PIE_ANE=off`, once
//! with the Neural Engine on, waiting for its programs to compile; the
//! second serve must actually hand MLPs and projections over, its readout
//! must sit on the first's — the
//! split is int8 activations and weights with fp16 sums, so a small gap is
//! the expected cost — and a decode step after it must stay on the GPU.
//!
//! Needs a Metal device, the private Neural Engine framework, and a
//! `metal`-stamped Qwen3.6-27B U4 artifact in `~/.pie/artifacts` (`pie model
//! import mlx-community/Qwen3.6-27B-4bit --deployment qwen36-27b-u4g64-kv-bf16`
//! writes one); skips without them. On a box whose supervisor kills `ANECompilerService`, the
//! compile is reported refused and the test fails saying so.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use engine_metal::{Boot, Lane, Shell};
use poem::{Platform, Request};
use poem_compiler::Budget;

/// Any `metal`-stamped text-only Qwen3.6-27B U4 artifact in the store:
/// dense MLPs with a hidden size of 5120 and an intermediate of 17408, which
/// the Neural Engine's shape takes. (A `vision` row would need a patch
/// ladder this boot does not state.)
const FAMILY: &str = "qwen36-27b-";
/// At least 512 rows, so the prefill fits a Neural Engine procedure.
const ROWS: usize = 640;
const CONTEXT: u32 = 1024;

/// The artifact and the deployment it was imported as.
fn artifact() -> Option<(PathBuf, String)> {
    let store = PathBuf::from(std::env::var("HOME").ok()?).join(".pie/artifacts");
    for entry in std::fs::read_dir(store).ok()?.flatten() {
        for file in std::fs::read_dir(entry.path()).ok()?.flatten() {
            let path = file.path();
            if path.extension().is_none_or(|e| e != "zt") {
                continue;
            }
            let Some(stamp) = checkpoint::file::serve::stamp_of(&path).ok().flatten() else {
                continue;
            };
            if stamp.backend == "metal"
                && stamp.deployment.starts_with(FAMILY)
                && stamp.deployment.contains("u4g64")
                && !stamp.deployment.contains("bonsai")
                && !stamp.deployment.contains("vision")
            {
                return Some((path, stamp.deployment));
            }
        }
    }
    None
}

/// The first `ROWS` ids of the committed WikiText-2 stream.
fn ids() -> Vec<u32> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/bonsai/ppl_ids.bin");
    let bytes = std::fs::read(path).expect("the committed id fixture");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .take(ROWS)
        .collect()
}

fn boot(artifact: &PathBuf, name: &str) -> (Shell, poem::Trace) {
    let deployment = poem_compiler::catalog::deployment(name).expect("the catalog ships the row");
    let trace = deployment.trace(Platform::Metal);
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .unwrap_or_else(|why| panic!("the artifact holds every plane of {name}: {why}"));
    drop(source);
    let started = Instant::now();
    let shell = Shell::load(Boot {
        voxels: None,
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: Budget::new(1, CONTEXT),
        patches: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: 1,
        pages: CONTEXT / 16,
        runahead: engine::runahead::Runahead::F1,
        gpu_mem_utilization: engine_metal::store::accounting::DEFAULT_GPU_MEM_UTILIZATION,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the shell loads");
    eprintln!("loaded {name} in {:.1}s", started.elapsed().as_secs_f64());
    (shell, trace)
}

fn fire(shell: &mut Shell, trace: &poem::Trace, tokens: &[u32]) -> Vec<f32> {
    let started = Instant::now();
    let out = shell
        .fire(&[Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
            tokens,
        }])
        .expect("the fire lands");
    eprintln!(
        "fired {} tokens in {:.1} ms",
        tokens.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    out.into_iter().next().expect("one readout row")
}

fn argmax(logits: &[f32]) -> usize {
    (0..logits.len()).fold(
        0,
        |best, at| if logits[at] > logits[best] { at } else { best },
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        aa += f64::from(x) * f64::from(x);
        bb += f64::from(y) * f64::from(y);
    }
    dot / (aa * bb).sqrt()
}

#[test]
fn the_split_mlp_matches_the_gpu_alone() {
    if !engine_metal::device::present() {
        eprintln!("not asked: no Metal device");
        return;
    }
    if let Err(why) = kernels_metal::ane::available() {
        eprintln!("not asked: no Neural Engine ({why})");
        return;
    }
    let Some((artifact, name)) = artifact() else {
        eprintln!("not asked: no metal-stamped {FAMILY}*u4g64* artifact in ~/.pie/artifacts");
        return;
    };
    let prompt = ids();
    assert_eq!(prompt.len(), ROWS, "the fixture holds the prompt");
    let next = [prompt[ROWS - 1]];

    // The GPU alone. The switch is read at load, and the whole model is
    // dropped before the second load so the box holds one at a time.
    // SAFETY: this test is the process's only thread reading the variable.
    unsafe { std::env::set_var("PIE_ANE", "off") };
    let whole = {
        let (mut shell, trace) = boot(&artifact, &name);
        assert!(
            shell.neural_engine().is_none(),
            "`PIE_ANE=off` keeps the Neural Engine out of the load"
        );
        shell.open(0).expect("the slot opens");
        let _warm = fire(&mut shell, &trace, &prompt);
        shell.open(0).expect("the slot reopens");
        fire(&mut shell, &trace, &prompt)
    };

    // The split.
    // SAFETY: as above.
    unsafe { std::env::set_var("PIE_ANE", "on") };
    let (mut shell, trace) = boot(&artifact, &name);
    let ane = shell
        .neural_engine()
        .expect("the Neural Engine is set up for the 27B's dense MLPs");
    let waited = Instant::now();
    let compiled = loop {
        match ane.compiled() {
            Some(outcome) => break outcome,
            None if waited.elapsed() > Duration::from_secs(600) => {
                panic!("the Neural Engine program did not compile within 10 minutes")
            }
            None => std::thread::sleep(Duration::from_millis(250)),
        }
    };
    compiled.unwrap_or_else(|why| {
        panic!("the Neural Engine program was refused: {why} (a supervisor killing ANECompilerService does this)")
    });
    eprintln!(
        "Neural Engine program ready after {:.1}s",
        waited.elapsed().as_secs_f64()
    );
    shell.open(0).expect("the slot opens");
    let _warm = fire(&mut shell, &trace, &prompt);
    shell.open(0).expect("the slot reopens");
    let split = fire(&mut shell, &trace, &prompt);
    let ane = shell.neural_engine().expect("still there");
    let (handed, columns) = (ane.splits(), ane.column_splits());
    assert!(handed > 0, "no MLP was handed to the Neural Engine");
    assert!(columns > 0, "no projection was handed to the Neural Engine");
    let decoded = fire(&mut shell, &trace, &next);
    let ane = shell.neural_engine().expect("still there");
    assert_eq!(
        (ane.splits(), ane.column_splits()),
        (handed, columns),
        "a one-token decode stays on the GPU"
    );
    assert!(
        decoded.iter().all(|v| v.is_finite()),
        "the decode is finite"
    );

    assert_eq!(whole.len(), split.len(), "the readouts are vocab-wide rows");
    assert!(
        split.iter().all(|v| v.is_finite()),
        "the split readout is finite"
    );
    let cos = cosine(&whole, &split);
    let (top_whole, top_split) = (argmax(&whole), argmax(&split));
    eprintln!(
        "{handed} MLPs and {columns} projections handed over; readout cosine {cos:.5}; top-1 {top_whole} vs {top_split}"
    );
    assert!(cos > 0.99, "the split readout drifted: cosine {cos:.5}");
    assert_eq!(top_whole, top_split, "the split changes the next token");
}
