//! A generation may name `families/` and the shared root. Never a sibling.
//!
//! Twenty generations were twenty crates, and the reason given was always this
//! rule: a crate boundary is the only thing Rust has that stops one generation
//! reaching into another. Folded into modules, `use crate::qwen_3::chat::…`
//! from inside `qwen_2` compiles, so the rule is this file.
//!
//! It is worth being precise about what the crate boundary actually bought,
//! because it is less than the argument claimed. `qwen_2` DID depend on
//! `qwen_3` — declared in Cargo.toml, with a comment explaining that Qwen3
//! owns ChatML and the others bind it. The boundary did not prevent the
//! reach; it made the reach a package edge instead of a `use`, which is a
//! record of the violation rather than a defence against it.
//!
//! So this test comes with the thing that makes the rule keepable:
//! [`families`](model::families), a legitimate home for what more than one
//! generation binds. ChatML lives there now. The rule and the escape hatch
//! arrive together on purpose — a rule with nowhere to put the exception is a
//! rule that gets an exception written into it.
//!
//! ## What "may not name a sibling" means
//!
//! Only code counts, and only paths. A generation's prose may say that its
//! template is Qwen3's with thinking off — that is the argument for the row
//! that dispatches to it, and it belongs where a reader will look.
//!
//! The two registries are exempt, and cannot not be: naming every generation
//! is what a registry IS. They live at the root, not inside a generation, so
//! they are simply not walked.

#![cfg(any(feature = "chat", feature = "contract"))]

use std::path::{Path, PathBuf};

fn src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// The generation modules: directory modules under `src/`, minus the four
/// that are not generations — `families/` (cross-generation sharing), `ffi/`
/// (the C boundary, one door for all of them), `config/` (the `pie.model/1`
/// descriptor, an aspect every generation is parameterized by rather than one
/// generation's property) and `bin/` (cargo's layout, not this crate's).
fn generations() -> Vec<String> {
    let not_a_generation = ["families", "ffi", "config", "bin"];
    let mut names: Vec<String> = std::fs::read_dir(src())
        .expect("src/ exists")
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !not_a_generation.contains(&n.as_str()))
        .collect();
    assert!(
        names.len() >= 15,
        "found {} generation modules; the layout moved and this guard is \
         looking in the wrong place",
        names.len()
    );
    names.sort();
    names
}

fn rust_files(dir: &Path) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("a generation directory") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let label = path
                    .strip_prefix(src())
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.push((label, std::fs::read_to_string(&path).expect("readable")));
            }
        }
    }
    out
}

#[test]
fn no_generation_names_a_sibling() {
    let generations = generations();
    let mut found = Vec::new();

    for generation in &generations {
        for (label, text) in rust_files(&src().join(generation)) {
            for (i, line) in text.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for sibling in &generations {
                    if sibling == generation {
                        continue;
                    }
                    if line.contains(&format!("crate::{sibling}::")) {
                        found.push(format!("  {label}:{}: {}", i + 1, line.trim()));
                    }
                }
            }
        }
    }

    assert!(
        found.is_empty(),
        "a generation names a sibling:\n{}\n\n\
         What two generations share is not one generation's property. Move it \
         to `families/` and let both name it there -- that is what the module \
         is for, and it is why this rule is keepable at all.",
        found.join("\n")
    );
}

/// The guard is not vacuous: `crate::<generation>::` is a shape that occurs.
///
/// Without this, a rename of the generation modules would turn the test above
/// into an assertion about strings nothing contains, passing forever.
#[test]
fn the_sibling_path_shape_is_real() {
    let generations = generations();
    let registry = std::fs::read_to_string(src().join("contract.rs"))
        .or_else(|_| std::fs::read_to_string(src().join("instruct.rs")))
        .expect("at least one registry exists");
    let hits = generations
        .iter()
        .filter(|g| registry.contains(&format!("crate::{g}::")))
        .count();
    assert!(
        hits >= 2,
        "matched {hits} `crate::<generation>::` paths in a registry, which is \
         where they are supposed to be everywhere; the guard is not looking \
         for the right shape"
    );
}
