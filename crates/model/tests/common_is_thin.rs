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
    // A generation is a DIRECTORY module under `src/`. Three directories are
    // not generations: `shared/` is the vocabulary a generation may name,
    // `ffi/` is the C boundary (one door for all of them), and `bin/` is
    // cargo's layout, not this crate's.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let not_a_generation = ["shared", "ffi", "bin"];
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .expect("src/ exists")
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !not_a_generation.contains(&name.as_str()))
        .collect();
    // 21, the same count `sibling_isolation.rs` walks. Both said `>= 15`.
    assert_eq!(
        names.len(),
        21,
        "found {} generation directories under src/, not 21; a generation \
         arrived or left, or the layout moved and this guard is now looking \
         in the wrong place",
        names.len()
    );
    // Directories are generations (`llama_3`, `qwen_3_5`), so the bare vendor
    // stems have to be listed: `common/` naming "llama" is the violation, and
    // no directory is spelled that way any more. The `model_type` spellings a
    // registry row dispatches on go in for the same reason.
    names.extend(
        [
            "llama",
            "qwen",
            "gemma",
            "olmo",
            "mistral",
            "ministral",
            "mixtral",
            "phi",
            "phi3",
            "deepseek",
            "kimi",
            "nemotron",
            "glm",
            "gpt_oss",
            "gptoss",
            "csm",
        ]
        .into_iter()
        .map(str::to_string),
    );
    assert!(
        names.len() > 10,
        "the family list came back suspiciously short"
    );
    names.sort();
    names
}

/// The shared perimeter: every `.rs` that a generation may reach for, and
/// that must therefore hold vocabulary rather than knowledge.
///
/// Two directories now, and both are listed with their path from `src/` so
/// they cannot be confused: the crate ROOT (the catalog and its answers) and
/// `shared/` (the vocabulary a generation is allowed to name). Before those
/// were one, the loose files at the root WERE the shared root; the split
/// moved the sharing into `shared/` and left the root as the table's own
/// surface, so this guard has to walk both to mean what it meant.
///
/// `SHARED` and `NOT_SHARED` must between them name every file in both. That
/// exhaustiveness IS the guard now that no crate boundary draws the line: a
/// new module cannot be quietly shared, because an unclassified one fails
/// here before any family check runs.
const SHARED: &[&str] = &[
    // ── The crate root: the table's own surface ──
    "encoding.rs",
    "instruct.rs",
    "manifest.rs",
    "metadata.rs",
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
    // ── `shared/`: the general vocabulary half ──
    "shared/builder.rs",
    "shared/decoders.rs",
    "shared/mlx.rs",
    "shared/moe.rs",
    "shared/policy.rs",
    "shared/probe.rs",
    // `tower_names.rs` is `weight_names.rs`'s encoder half and lands on the
    // opposite side of this list from it, which is worth a sentence because
    // the two are siblings. `weight_names` names families on purpose — each
    // family's walk recognises itself by a tensor only it ships. A TOWER's
    // slot list does not: it is an ordered `const void**` layout, the order
    // IS the ABI, and no entry in it asks which family is being staged.
    "shared/tower_names.rs",
];
/// Files that name families ON PURPOSE, with the reason each does.
const NOT_SHARED: &[(&str, &str)] = &[
    (
        "shared/vocabulary.rs",
        "THE TYPE A GENERATION STATES ITS NAMES IN, and its doc argues from \
         the generations that differ: `nemotron_h` publishes \
         `backbone.layers.{}.mixer.q_proj`, `csm` publishes five towers, and \
         llama.cpp ships four of the families and not the rest. Those \
         sentences are the REASON the table has the shape it has, and a \
         reader who cannot see which generations disagree cannot check it. \
         No CODE here names one -- the names are the columns a generation \
         fills, and the module fills none of them.",
    ),
    (
        "lib.rs",
        "the crate doc, which names the generations it declares",
    ),
    (
        "catalog.rs",
        "THE REGISTRY, and the only one. It names every generation because \
         `GENERATIONS` is the flattening of their tables — that is a list \
         of MODULES rather than a switch on a string, which is the whole \
         difference between this and the three `model_type` tables it \
         replaced",
    ),
    (
        "multimodal.rs",
        "family-aware by design -- it dispatches on a VisionArch",
    ),
    (
        "ingest.rs",
        "THE GGUF DISPATCH. `pass_for` answers a `general.architecture` \
         string -- `qwen2`, `qwen3`, `qwen3moe`, `gemma3`, `llama` -- with \
         the family's own ingest pass, and its doc names two more \
         (`gpt-oss`, `gemma4`) to record why they are REFUSED. Naming \
         families is what the module IS, and the string it switches on is \
         llama.cpp's rather than pie's: nothing but a table can bridge a \
         foreign vocabulary to this crate's modules. It sits outside the \
         perimeter -- no driver reaches it, only `pie model import` does",
    ),
    (
        "shared/weight_names.rs",
        "a MAP between two of this crate's own vocabularies: the trace names \
         the DSL invents and the published names a contract author invents. \
         Each family's walk recognises itself by a tensor only it ships, so \
         naming families is what the module IS",
    ),
    (
        "shared/mod.rs",
        "the directory's own doc and its `pub mod` list, which necessarily \
         names the shared IMPLEMENTATIONS below — a module list is not a \
         switch on a family, and the rule it states is the one this file \
         enforces",
    ),
    // The shared IMPLEMENTATIONS. Each is one specific answer that several
    // generations happen to have in common, and each is named for the
    // generation that wrote it down first — which is a fact about who wrote
    // it, not a dispatch on who binds it. They are exempt for the same reason
    // a generation directory is: naming the family is what the file is about.
    // The rule they are held to instead is the sibling-isolation one, which
    // is what makes `shared/` the legitimate home rather than a sibling edge.
    (
        "shared/chatml.rs",
        "ChatML, which qwen3 wrote first and qwen3, qwen3.5, nemotron-h \
         and glm-5 parameterize. NAMED rather than counted: the sentence \
         said `five generations` and there are four, and a wrong count \
         is invisible in a way a wrong name is not",
    ),
    (
        "shared/gemma_chat.rs",
        "gemma's `<start_of_turn>` template, bound by gemma-2, gemma-3 and \
         gemma-3n",
    ),
    (
        "shared/deepseek.rs",
        "deepseek's shared pieces, bound by more than one deepseek generation",
    ),
    (
        "shared/kimi.rs",
        "kimi's shared pieces, bound by kimi-k2 and kimi-k3",
    ),
    (
        "test_rows.rs",
        "ONE ROW THAT IS NOT A MODEL, behind a feature no shipped build \
         enables. It names llama-3 because it IS a `Llama3` — deliberately, \
         so that a test exercises the same type, table and authoring pass \
         production does rather than a second implementation of them. Sharing \
         is not the question for a file the perimeter never links",
    ),
];

/// `instruct.rs` is half vocabulary and half registry, in that order, split by
/// this marker. Only the half above it is shared.
const INSTRUCT_REGISTRY_MARKER: &str = "── The registry ──";

fn common_sources() -> Vec<(String, String)> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    // Both halves of the perimeter, labelled by their path from `src/`. The
    // family DIRECTORIES under `shared/` are deliberately not walked: a
    // `shared/llama_like/` names its family in every file, and it is held to
    // the sibling-isolation rule instead.
    let mut on_disk: Vec<String> = Vec::new();
    for (prefix, dir) in [("", src.clone()), ("shared/", src.join("shared"))] {
        for entry in std::fs::read_dir(&dir).expect("a perimeter directory") {
            let path = entry.expect("readable entry").path();
            if path.extension().is_some_and(|x| x == "rs") {
                let name = path.file_name().unwrap().to_string_lossy();
                on_disk.push(format!("{prefix}{name}"));
            }
        }
    }
    assert!(
        on_disk.len() > 10,
        "found {} sources across the root and `shared/`; the layout moved and \
         this guard is now looking in the wrong place",
        on_disk.len()
    );

    let mut unclassified: Vec<&String> = on_disk
        .iter()
        .filter(|f| !SHARED.contains(&f.as_str()) && !NOT_SHARED.iter().any(|(n, _)| n == f))
        .collect();
    unclassified.sort();
    assert!(
        unclassified.is_empty(),
        "module(s) classified as neither shared vocabulary nor dispatch: {unclassified:?}.\n\
         Add it to SHARED (and it must then name no family) or to NOT_SHARED with the reason \
         it is allowed to. Silence here would mean a new shared module nothing guards.",
    );

    // And the other direction: a classification naming a file that no longer
    // exists is a rule pointing at nothing, which reads as green forever.
    let mut vanished: Vec<&str> = SHARED
        .iter()
        .chain(NOT_SHARED.iter().map(|(n, _)| n))
        .copied()
        .filter(|n| !on_disk.iter().any(|f| f == n))
        .collect();
    vanished.sort_unstable();
    assert!(
        vanished.is_empty(),
        "classified module(s) that are not on disk: {vanished:?}. A rule about \
         a file that moved is a rule that guards nothing.",
    );

    let mut out = Vec::new();
    for name in SHARED {
        let text = std::fs::read_to_string(src.join(name)).expect("read a shared source");
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
                    let boundary =
                        |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
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
    // A GENERATION's own module, which is the thing the list is meant to
    // recognise. It used to be `llama_3/contract.rs`, which moved to
    // `shared/llama_like/contract.rs` when the three authoring passes ten
    // generations bind stopped pretending to be llama-3's — and a shared
    // file is the wrong oracle for this, because the whole claim above is
    // that shared files hold no family names. `llama_3/mod.rs` is llama-3's
    // and stays llama-3's: it is where its rows are written down.
    let llama = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/llama_3/mod.rs");
    let text = std::fs::read_to_string(&llama).expect("read Llama 3's own module");
    let hits = families
        .iter()
        .filter(|f| text.to_ascii_lowercase().contains(f.as_str()))
        .count();
    assert!(
        hits >= 2,
        "the family-name list matched {hits} names in llama's own module; \
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
