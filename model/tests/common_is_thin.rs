//! `common/` names no family — and stays thin enough that it cannot.
//!
//! `pie-model-common`'s lib.rs opens with the rule: *"What every family shares: vocabulary and
//! building blocks, never knowledge about any one family."* A rule stated only
//! in prose is one a hurried change can break without anyone noticing, and
//! this one breaks quietly: a family name reaching `common/` does not fail a
//! build, it just moves a fact one directory away from the checkpoints it is
//! about, where the next family to need something similar will copy it.
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
//! file under `model/` and that is correct: it is the authoring DSL, and every
//! generation calls its passes directly. Bulk in a shared crate is only a
//! smell when it is *knowledge*; the passes are mechanism.
//!
//! The test lives in `pie-model` rather than in `pie-model-common` because it
//! needs to see the generation directories to know what a family is called.

#![cfg(feature = "contract")]

use std::path::{Path, PathBuf};

/// Every family this repo knows, in the spellings a source file might use.
///
/// Derived from the directory names rather than typed out, so a family added
/// under `model/src/` is covered the moment it exists. `common` itself is
/// excluded, obviously, and so are the aspect modules that are not families.
fn family_names() -> Vec<String> {
    // The generation crates live beside this one under `model/`. Everything
    // that is not a generation is named here, so a new generation is covered
    // the moment its directory exists.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let not_a_generation = ["common", "config", "src", "tests", "target"];
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .expect("model/ exists")
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !not_a_generation.contains(&name.as_str()))
        .collect();
    assert!(
        names.len() >= 15,
        "found {} generation directories under model/; the layout moved and \
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

fn common_sources() -> Vec<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("common/src");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("model/common/src exists")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    assert!(!out.is_empty(), "found no sources under model/common/src");
    out.sort();
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
    for path in common_sources() {
        let text = std::fs::read_to_string(&path).expect("read a common source");
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
                        found.push(format!(
                            "{}:{}: {} ({family})",
                            path.file_name().unwrap().to_string_lossy(),
                            i + 1,
                            line.trim()
                        ));
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
    let llama = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("llama_3/src/contract.rs");
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
    for path in common_sources() {
        let text = std::fs::read_to_string(&path).expect("read a common source");
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
                found.push(format!(
                    "{}:{}: {}",
                    path.file_name().unwrap().to_string_lossy(),
                    i + 1,
                    trimmed
                ));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a shared item in `common/` is named after a family:\n{}",
        found.join("\n")
    );
}
