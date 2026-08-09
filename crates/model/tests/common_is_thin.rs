//! The shared root names no family — and stays thin enough that it cannot.
//!
//! The rule the crate's lib.rs opens with: *"What every family shares: vocabulary and
//! building blocks, never knowledge about any one family."* A rule stated only
//! in prose is one a hurried change can break without anyone noticing, and
//! this one breaks quietly: a family name reaching the shared root does not
//! fail a build, it just moves a fact one directory away from the checkpoints
//! it is about, where the next family to need something similar will copy it.
//!
//! This was `model`, a crate. It is now the `.rs` files at the root
//! of `src/`, and the guard got MORE work rather than less: with no crate
//! boundary to say what is shared, every root file must be classified as
//! shared vocabulary or as something else, and an unclassified one fails.
//!
//! Two things are checked, because "thin" has two failure modes:
//!
//! 1. **No family names.** Not in code, not in a test fixture. A helper that
//!    takes a family's name to *quote it in a diagnostic* is fine — that is a
//!    string the caller supplies — but `common/` may not write one down.
//! 2. **No family-shaped API.** A function whose name contains a family is a
//!    family's rule wearing a shared function's clothes.
//!
//! What this deliberately does NOT check is size. `builder.rs` is the biggest
//! file in the crate and that is correct: it is the authoring DSL, and every
//! generation calls its passes directly. Bulk in a shared crate is only a
//! smell when it is *knowledge*; the passes are mechanism.
//!
//! The generation directories are what tell it what a family is called, so the
//! needle list is derived rather than typed out.

#![cfg(feature = "contract")]

use std::path::PathBuf;

/// Every family this repo knows, in the spellings a source file might use.
///
/// Derived from the directory names rather than typed out, so a family added
/// under `model/src/` is covered the moment it exists. `common` itself is
/// excluded, obviously, and so are the aspect modules that are not families.
fn family_names() -> Vec<String> {
    // A generation is a DIRECTORY module under `src/`; the shared root is the
    // loose `.rs` files beside them. Four directories are not generations:
    // `families/` is cross-generation sharing, `ffi/` is the C boundary (one
    // door for all of them), `config/` is the descriptor aspect, and `bin/`
    // is cargo's, not this crate's layout at all.
    //
    // `config/` in particular MUST be here: the word "config" occurs all over
    // the shared root in code (`crate::config::VERSION` in `facts.rs`), so
    // leaving it in would make the guard below fire on the very reference
    // that is correct.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let not_a_generation = ["families", "ffi", "config", "bin"];
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .expect("src/ exists")
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !not_a_generation.contains(&name.as_str()))
        .collect();
    assert!(
        names.len() >= 15,
        "found {} generation directories under src/; the layout moved and \
         this guard is now looking in the wrong place",
        names.len()
    );
    // Directories are generations (`llama_3`, `qwen_3_5`), so the bare vendor
    // stems have to be listed: `common/` naming "llama" is the violation, and
    // no directory is spelled that way any more. The `model_type` spellings a
    // registry row dispatches on go in for the same reason.
    names.extend(
        [
            "llama", "qwen", "gemma", "olmo", "mistral", "ministral", "mixtral",
            "phi", "phi3", "deepseek", "kimi", "nemotron", "glm", "gpt_oss",
            "gptoss", "csm",
        ]
        .into_iter()
        .map(str::to_string),
    );
    assert!(names.len() > 10, "the family list came back suspiciously short");
    names.sort();
    names
}

/// The shared root: every `.rs` beside the generation directories that holds
/// vocabulary rather than dispatch.
///
/// `SHARED` and `NOT_SHARED` must between them name every root file. That
/// exhaustiveness IS the guard now that no crate boundary draws the line: a
/// new root module cannot be quietly shared, because an unclassified one fails
/// here before any family check runs.
const SHARED: &[&str] = &[
    "builder.rs", "decoders.rs", "facts.rs", "instruct.rs", "metadata.rs",
    "mlx.rs", "moe.rs", "policy.rs", "probe.rs",
    // `deployment.rs` belongs here and the fact that it PASSES is the
    // whole claim: it is what a driver receives, and a driver that
    // could read a family name off it could branch on one. The guard
    // holding here is the same guard `driver-cuda/tests/no_family_names.rs`
    // holds from the other side.
    "deployment.rs",
    // `boot.rs` is the load path both drivers walk. It belongs here for the
    // same reason `deployment.rs` does: a driver calls it, so a family name
    // reachable from it is a family name reachable from a driver. It asks
    // `catalog.rs` — which IS the registry — but asking a registry is not
    // knowing its rows, and that distinction is what this list checks.
    "boot.rs",
    // `contract.rs` used to be a REGISTRY of its own: `model_type` ->
    // author, thirty-odd family rows. It is ten lines over a
    // `&dyn Variant` now, so it holds no family names at all — which is
    // why it moved from the exceptions below to the perimeter here.
    "contract.rs",
];
/// Root files that name families ON PURPOSE, with the reason each does.
const NOT_SHARED: &[(&str, &str)] = &[
    ("lib.rs", "the crate doc, which names the generations it declares"),
    (
        "catalog.rs",
        "THE REGISTRY, and the only one. It names every generation because \
         `GENERATIONS` is the flattening of their tables — that is a list \
         of MODULES rather than a switch on a string, which is the whole \
         difference between this and the three `model_type` tables it \
         replaced",
    ),
    ("multimodal.rs", "family-aware by design -- it dispatches on a VisionArch"),
    (
        "ffi.rs",
        "a DOOR, not vocabulary: one `extern \"C\"` entry per family is what \
         the C surface of a per-family declaration looks like",
    ),
    (
        "weight_names.rs",
        "a MAP between two of this crate's own vocabularies: the trace names \
         the DSL invents and the published names a contract author invents. \
         Each family's walk recognises itself by a tensor only it ships, so \
         naming families is what the module IS",
    ),
    (
        "emissions.rs",
        "a DEPLOYMENT LIST: which families' static forms are committed, from \
         which fact sets -- shared between the emit-cuda bin and the \
         regeneration check so neither holds a copy, and per-family by its \
         nature",
    ),
];

/// `instruct.rs` is half vocabulary and half registry, in that order, split by
/// this marker. Only the half above it is shared.
const INSTRUCT_REGISTRY_MARKER: &str = "── The registry ──";

fn common_sources() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("src/ exists")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert!(!on_disk.is_empty(), "found no sources at the root of src/");

    let mut unclassified: Vec<&String> = on_disk
        .iter()
        .filter(|f| {
            !SHARED.contains(&f.as_str()) && !NOT_SHARED.iter().any(|(n, _)| n == f)
        })
        .collect();
    unclassified.sort();
    assert!(
        unclassified.is_empty(),
        "root module(s) classified as neither shared vocabulary nor dispatch: {unclassified:?}.\n\
         Add it to SHARED (and it must then name no family) or to NOT_SHARED with the reason \
         it is allowed to. Silence here would mean a new shared module nothing guards.",
    );

    let mut out = Vec::new();
    for name in SHARED {
        if !on_disk.iter().any(|f| f == name) {
            continue;
        }
        let text = std::fs::read_to_string(dir.join(name)).expect("read a shared source");
        let text = match text.find(INSTRUCT_REGISTRY_MARKER) {
            Some(at) if *name == "instruct.rs" => text[..at].to_string(),
            _ => text,
        };
        out.push(((*name).to_string(), text));
    }
    assert!(!out.is_empty(), "no shared sources found");
    out
}

/// Strip `//` comments. Prose in `common/` may discuss which families use a
/// pass and why — that is the argument for the pass being shared, and it is
/// worth keeping. What may not appear is a family in the *code*.
fn code_only(text: &str) -> String {
    text.lines()
        .map(|line| line.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

/// No family name appears in `common/`'s code — including its tests.
#[test]
fn common_writes_down_no_family_name() {
    let families = family_names();
    let mut found = Vec::new();
    for (label, text) in common_sources() {
        let code = code_only(&text);
        for (i, line) in code.lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            for family in &families {
                // Word-ish match: `llama` must not fire on a longer identifier
                // that merely contains it, and `csm` is short enough to appear
                // inside unrelated words.
                if let Some(at) = lower.find(family.as_str()) {
                    let before = lower[..at].chars().next_back();
                    let after = lower[at + family.len()..].chars().next();
                    let boundary = |c: Option<char>| {
                        c.is_none_or(|c| !c.is_alphanumeric() && c != '_')
                    };
                    if boundary(before) && boundary(after) {
                        found.push(format!("{label}:{}: {} ({family})", i + 1, line.trim()));
                    }
                }
            }
        }
    }
    assert!(
        found.is_empty(),
        "`common/` names a family in code. It holds vocabulary and building \
         blocks; a fact about one family's checkpoints belongs in that \
         family's directory, where someone reading that family finds it:\n{}",
        found.join("\n")
    );
}

/// The guard is not vacuous: the needle list matches something somewhere.
///
/// Without this, a typo in `family_names` — or a directory layout that stopped
/// yielding family names — would turn the test above into an assertion about
/// strings nothing contains, passing forever while `common/` fills up.
#[test]
fn the_family_list_matches_real_code() {
    let families = family_names();
    let llama = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/llama_3/contract.rs");
    let text = std::fs::read_to_string(&llama).expect("read Llama 3's contract");
    let hits = families
        .iter()
        .filter(|f| text.to_ascii_lowercase().contains(f.as_str()))
        .count();
    assert!(
        hits >= 2,
        "the family-name list matched {hits} names in llama's own contract; \
         it is not looking for the right strings"
    );
}

/// No `pub` item in `common/` is named after a family.
///
/// The subtler shape of the same violation: `author_dense_contract` was not a
/// family name, but it *was* a family's pass sequence living in the shared
/// crate, and six families' contracts could not be read where they lived
/// because of it. A name that says "llama" or "qwen" is the version of that
/// mistake a grep can catch.
#[test]
fn no_shared_item_is_named_after_a_family() {
    let families = family_names();
    let mut found = Vec::new();
    for (label, text) in common_sources() {
        for (i, line) in code_only(&text).lines().enumerate() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub fn ")
                && !trimmed.starts_with("pub struct ")
                && !trimmed.starts_with("pub enum ")
            {
                continue;
            }
            let lower = trimmed.to_ascii_lowercase();
            if families.iter().any(|f| lower.contains(f.as_str())) {
                found.push(format!("{label}:{}: {trimmed}", i + 1));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a shared item at the root is named after a family:\n{}",
        found.join("\n")
    );
}
