//! A real checkpoint, and whether the text's names find it.
//!
//! `model_dispatch.rs` proves every name the text states has a *spelling*.
//! That is a claim about the map. This is the claim about the **checkpoint**:
//! that the spelling names a tensor the plan actually published.
//!
//! The two can disagree, and did. An earlier draft of `Names::mlx` assumed the
//! HuggingFace convention (`model.layers.3.…`, `model.embed_tokens`), was
//! self-consistent, and passed its own test — because both sides of that test
//! were the same file. Only a checkpoint can settle it, so this test exists
//! and is gated on one.
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT`, the same variable `device_smoke.rs`
//! takes. **It has been run.** Against
//! `mlx-community/Llama-3.2-1B-Instruct-4bit` (372 tensors) every name both
//! fire classes state binds, which is the first claim about this text that a
//! real checkpoint rather than another source file settles.
//!
//! Two defects were between here and that result, and neither was in the name
//! map:
//!
//!   * The gate stated `qwen3_0_6b()` facts against a llama snapshot and
//!     reported 308 missing names -- every one a `qkv` or a `q_norm`, which is
//!     to say the FIXTURE's bindings. It derives facts from the checkpoint
//!     now, through the chain the seam uses.
//!   * `geometry_from_facts` read only the `q35_*` block, so a llama config
//!     -- which `from_descriptor` reads into `ll_*` -- was refused as
//!     "carrying no decoder shape" while carrying it in the other block. And
//!     it demanded a linear-attention block of a stack that has no linear
//!     layers.

#![cfg(target_vendor = "apple")]

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;

use driver_metal_new::metal::Context;
use driver_metal_new::model::load::load;
use driver_metal_new::model::resolve::{Names, Store};
use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Arg, Fire, Row, lower};
use model_compiler::trace::FireClass;

fn snapshot() -> Option<PathBuf> {
    std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT").map(PathBuf::from)
}

/// The `pie.model/1` descriptor for a snapshot.
///
/// A TEST may normalize a checkpoint — `crates/model/tests/one_normalizer.rs`
/// scans `crates/model/src` and `crates/engine/src`, and the rule it enforces
/// is that the RUNTIME has one normalizer, not that nothing may call it. The
/// seam itself takes the descriptor the worker hands over.
fn descriptor_for(snapshot: &std::path::Path) -> String {
    let raw = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let root: serde_json::Value = serde_json::from_str(&raw).expect("config.json parses");
    model::config::descriptor(&root, snapshot.to_str().expect("utf8 path"))
        .expect("the config normalizes to a descriptor")
        .to_string()
}

/// Every weight name the Metal text states, over both fire classes.
///
/// The facts come from the CHECKPOINT, through the same chain the seam uses:
/// descriptor -> `ModelFacts` -> `DecodeGeometry` -> `text::facts_from`. An
/// earlier draft named `qwen3_0_6b()` here and reported 308 missing names
/// against a llama-3.2 snapshot -- every one of them a `qkv` or a `q_norm`,
/// which is to say the fixture's bindings and not the checkpoint's. A gate
/// that states its own facts is not testing the checkpoint.
fn names_the_text_states(
    facts: &LlamaLikeFacts,
    metal: &LlamaLikeMetalFacts,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (class, rows) in [(FireClass::Decode, 1usize), (FireClass::Prefill, 16)] {
        let plan = llama_like_metal(facts, metal, class);
        let low = lower(
            &plan,
            &vec![
                Row {
                    samples: true,
                    ..Row::default()
                };
                rows
            ],
            Fire {
                captures_across_splits: false,
            },
        )
        .expect("the metal text lowers");
        for arg in &low.args {
            if let Arg::Weight(name) = arg {
                // A `scale.` marker is a constant riding the weight slot; the
                // binder never looks it up.
                if !name.starts_with("scale.") {
                    out.insert(name.clone());
                }
            }
        }
    }
    out
}

#[test]
fn the_checkpoint_answers_the_names_the_text_states() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let descriptor = descriptor_for(&snapshot);
    let loaded = load(&context, &snapshot, &descriptor).expect("the checkpoint loads");
    assert!(
        !loaded.tensors.is_empty(),
        "the plan published no tensors at all"
    );

    // The checkpoint's own facts, derived the way the seam derives them.
    let model_facts = driver_metal_new::facts::ModelFacts::from_descriptor(&descriptor)
        .expect("the descriptor states the model's facts");
    let geometry =
        driver_metal_new::batch::geometry_from_facts(&model_facts).expect("a decodable geometry");
    let (facts, metal) =
        driver_metal_new::model::text::facts_from(&geometry, |t| loaded.tensors.contains_key(t));

    let named = HashMap::new();
    let mut store = Store::new(Names::mlx(), &loaded.tensors, &named);
    let mut missing: BTreeSet<String> = BTreeSet::new();
    for name in names_the_text_states(&facts, &metal) {
        use driver_metal_new::model::executor::Resolver as _;
        if store.weight(&name).is_none() {
            missing.insert(name);
        }
    }

    assert!(
        missing.is_empty(),
        "the text states {} name(s) this checkpoint does not answer:\n  {}\n\n\
         Either `Names::mlx` spells one wrong, or the plan did not publish it \
         — and the two are told apart by looking. The checkpoint published:\n  {}",
        missing.len(),
        missing.iter().cloned().collect::<Vec<_>>().join("\n  "),
        loaded
            .names()
            .iter()
            .take(40)
            .copied()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// Not an assertion — a report, so a run against a new checkpoint says what it
/// holds without anyone editing a test to find out.
#[test]
fn what_this_checkpoint_published() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let descriptor = descriptor_for(&snapshot);
    let loaded = load(&context, &snapshot, &descriptor).expect("the checkpoint loads");
    let names = loaded.names();
    eprintln!("{} tensors published; layer 0 and the globals:", names.len());
    for name in &names {
        if name.starts_with("layers.0.") || !name.starts_with("layers.") {
            eprintln!("  {name}");
        }
    }
}

/// **Every name a text states, against a checkpoint's PLAN — no device.**
///
/// The same claim as `the_checkpoint_answers_the_names_the_text_states` and a
/// hundredth of the cost: `compile_load_plan` reads a snapshot's metadata and
/// a descriptor and publishes every tensor the load WILL stage, without
/// staging any of it. The 26B gemma4 on this machine is SIGKILLed by the
/// staging test — fifteen gigabytes — and answers this one in milliseconds.
///
/// The published names are what matters, and they are NOT the checkpoint's own
/// keys: the plan renames. Comparing `Names::mlx` against a
/// `safetensors.index.json` was the first draft of this test and it reported
/// fifty mismatches that were all the rename.
///
/// Gated on `PIE_METAL_NAMES_SNAPSHOT`, and on `PIE_METAL_NAMES_ARCH` saying
/// which naming convention to read it with.
#[test]
fn every_name_the_text_states_is_a_tensor_the_load_plan_publishes() {
    let dirs = snapshots_to_check();
    if dirs.is_empty() {
        eprintln!(
            "SKIP: no snapshots. Set PIE_METAL_NAMES_SNAPSHOT (one path, or \
             several separated by `:`), or populate the HuggingFace cache."
        );
        return;
    }
    for dir in dirs {
        eprintln!("--- {}", dir.display());
        check_one_snapshot(&dir);
    }
}

/// Which snapshots this run holds the text to.
///
/// `PIE_METAL_NAMES_SNAPSHOT` when set -- one path, or several separated by
/// `:`. Otherwise every MLX snapshot in the HuggingFace cache, because a
/// machine that HAS six checkpoints should be told about all six and not the
/// one somebody remembered to name. Discovery is what turned this from a gate
/// somebody ran once into the gate that found gpt-oss's 246 misses.
fn snapshots_to_check() -> Vec<PathBuf> {
    if let Some(v) = std::env::var_os("PIE_METAL_NAMES_SNAPSHOT") {
        return std::env::split_paths(&v).collect();
    }
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let Ok(entries) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for repo in entries.flatten() {
        // MLX snapshots only: a GGUF export is a different container and this
        // gate reads safetensors.
        if !repo.file_name().to_string_lossy().contains("mlx-community") {
            continue;
        }
        let Ok(snaps) = std::fs::read_dir(repo.path().join("snapshots")) else {
            continue;
        };
        for snap in snaps.flatten() {
            if snap.path().join("config.json").is_file() {
                found.push(snap.path());
            }
        }
    }
    found.sort();
    found
}

/// Hold the text's names to one snapshot's load plan.
///
/// An architecture no Metal text serves is SKIPPED rather than failed: the
/// llama-like text does not model `qwen3_5`'s linear-attention interleave, and
/// reporting its every tensor as missing says nothing except that.
fn check_one_snapshot(dir: &std::path::Path) {
    let dir = dir.to_path_buf();
    let descriptor = descriptor_for(&dir);
    let target = driver_metal_new::loader::metal_storage_target();
    let (plan, _) = driver_metal_new::loader::compile_load_plan(&dir, &target, &descriptor)
        .expect("the plan compiles");
    let published: BTreeSet<&str> = plan.tensors.iter().map(|t| t.name.as_str()).collect();

    // ONE map. gemma4 used to need its own, and the arch variable said which
    // -- a driver picking a name map per checkpoint. Both conventions are
    // candidates of `Names::mlx` now, so the checkpoint picks and this does
    // not.
    let names = Names::mlx();
    // The DEPLOYMENT's facts, derived from the same descriptor, so a name the
    // text states is one this checkpoint's shape actually asks for.
    let model_facts = driver_metal_new::facts::ModelFacts::from_descriptor(&descriptor)
        .expect("the descriptor states the model's facts");
    let Ok(geometry) = driver_metal_new::batch::geometry_from_facts(&model_facts) else {
        eprintln!("    SKIP: no decodable shape (an architecture no text serves)");
        return;
    };
    // Whether a text serves this checkpoint at all, asked of the DRIVER
    // rather than of a list kept here. `qwen3_5` interleaves linear
    // attention, which the metal text does not model -- and every one of its
    // tensors would then report as missing for that one reason, which is a
    // page of output saying nothing.
    let arch = std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
        .and_then(|c| c.get("model_type")?.as_str().map(str::to_string))
        .unwrap_or_default();
    if !driver_metal_new::model::text::serves(&arch) {
        eprintln!("    SKIP: no metal text serves `{arch}`");
        return;
    }
    let (facts, metal) =
        driver_metal_new::model::text::facts_from(&geometry, |t| published.contains(t));

    let tensors = HashMap::new();
    let named = HashMap::new();
    let store = Store::new(names, &tensors, &named);
    let mut missing: BTreeSet<String> = BTreeSet::new();
    for traced in names_the_text_states(&facts, &metal) {
        // EVERY spelling, not the first: this store has no staged tensors to
        // choose with, and a role that has several candidates resolves at run
        // time against the ones the checkpoint published. Asking only the
        // first would fail a checkpoint that spells it the second way, which
        // is every checkpoint but one.
        let candidates = store.checkpoint_names(&traced);
        if candidates.is_empty() {
            missing.insert(format!("{traced} -> (no spelling)"));
            continue;
        }
        if !candidates.iter().any(|c| published.contains(c.as_str())) {
            missing.insert(format!("{traced} -> {}", candidates.join(" | ")));
        }
    }

    assert!(
        missing.is_empty(),
        "{}: {} name(s) the text states are not tensors this load plan publishes:\n  \
         {}\n\nEither the map spells one wrong or the deployment's facts claim \
         something it does not ship — and the two are told apart by looking. \
         The plan publishes:\n  {}",
        dir.display(),
        missing.len(),
        missing.iter().cloned().collect::<Vec<_>>().join("\n  "),
        published
            .iter()
            .take(30)
            .copied()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
