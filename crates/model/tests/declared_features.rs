//! A library's declared features must cover what its library code uses.
//!
//! # The trap
//!
//! Cargo unifies a crate's dependency features with its DEV-dependency
//! features for any build that resolves dev targets. So a crate whose library
//! says `model = { features = ["contract"] }` and whose dev-dependency says
//! `features = ["config", "contract", "forward"]` compiles perfectly under
//! `cargo check`, `cargo test`, `cargo clippy --all-targets` and every other
//! command anyone runs — while the library ALONE, which is what a downstream
//! consumer links, does not build at all.
//!
//! `cargo tree -e features,no-dev` shows the truth. Nothing else does, and
//! nothing in CI runs it.
//!
//! # Found three times
//!
//! | where | declared | actually needed |
//! |---|---|---|
//! | `model::deployment_cuda` | ungated | `all(forward, config)` |
//! | `model::descriptor` | ungated | `config` |
//! | `driver-cuda` -> `model` | `["config"]`, then `["contract"]` | all three |
//!
//! The third is the one this file exists for, because it was got wrong
//! TWICE — once by omission and once by a fix that replaced the missing
//! feature instead of adding to it — and both times the workspace was green.
//! CI's `cargo test --workspace` excludes `driver-cuda`, so the one command
//! that could have observed it is the one command that cannot.
//!
//! # What is checked
//!
//! For each consumer: every `model::<root>` path appearing in its `src/` must
//! have its gate satisfied by the features that consumer's LIBRARY dependency
//! on `model` declares. The gates are read from `model/src/lib.rs` rather
//! than listed here, so adding a `#[cfg]` to a module updates this test by
//! itself.
//!
//! # What is NOT checked
//!
//! Only `model`, only its direct consumers listed below, and only the root
//! module of each path. A feature-gated item INSIDE an ungated module is
//! invisible here. That is a deliberate floor: the three instances found were
//! all whole modules, and a check that tried to be complete would need to be
//! the compiler.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The crates whose library depends on `model`.
///
/// Listed rather than discovered, so that a new consumer is a deliberate
/// addition. `engine` is absent on purpose: it takes `model` through feature
/// forwarding rather than a fixed set, which is a different question.
const CONSUMERS: &[&str] = &["driver-metal", "driver-cuda"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model/../.. is the workspace root")
        .to_path_buf()
}

/// Every `pub mod X;` in `model/src/lib.rs`, with the features its `#[cfg]`
/// demands.
///
/// An ungated module maps to an empty set, which every feature set satisfies.
/// Only `all(..)` and bare `feature = ".."` are read as REQUIREMENTS; an
/// `any(..)` gate is satisfiable more than one way, so it is recorded as no
/// requirement rather than as a wrong one.
fn module_gates(root: &Path) -> BTreeMap<String, BTreeSet<String>> {
    let src = std::fs::read_to_string(root.join("crates/model/src/lib.rs"))
        .expect("model/src/lib.rs is readable");
    let mut gates = BTreeMap::new();
    let mut pending: Option<String> = None;
    for line in src.lines() {
        let line = line.trim();
        if line.starts_with("#[cfg(") {
            pending = Some(line.to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix("pub mod ") {
            let name = rest.trim_end_matches(';').trim_end_matches(" {").trim();
            let mut needs = BTreeSet::new();
            if let Some(cfg) = &pending
                && !cfg.contains("any(")
            {
                let mut rest = cfg.as_str();
                while let Some(i) = rest.find("feature = \"") {
                    rest = &rest[i + 11..];
                    if let Some(j) = rest.find('"') {
                        needs.insert(rest[..j].to_string());
                        rest = &rest[j..];
                    }
                }
            }
            gates.insert(name.to_string(), needs);
        }
        if !line.starts_with("///") && !line.starts_with("//") && !line.is_empty() {
            pending = None;
        }
    }
    gates
}

/// The features a consumer's LIBRARY dependency on `model` declares.
///
/// Reads the `[dependencies]` table only. A `[dev-dependencies]` entry for
/// the same crate is what makes this whole class of defect invisible, so
/// finding it here would defeat the test.
fn declared_features(root: &Path, consumer: &str) -> BTreeSet<String> {
    let manifest = std::fs::read_to_string(root.join("crates").join(consumer).join("Cargo.toml"))
        .unwrap_or_else(|e| panic!("{consumer}/Cargo.toml is readable: {e}"));

    let mut section = String::new();
    let mut lines = manifest.lines().peekable();
    while let Some(line) = lines.next() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            section = t.to_string();
            continue;
        }
        if section != "[dependencies]" || !t.starts_with("model = ") {
            continue;
        }
        // The entry may be inline or span lines up to its closing brace.
        let mut entry = t.to_string();
        while !entry.contains('}') {
            let Some(next) = lines.next() else { break };
            entry.push(' ');
            entry.push_str(next.trim());
        }
        let mut features = BTreeSet::new();
        if let Some(i) = entry.find("features") {
            let rest = &entry[i..];
            if let (Some(a), Some(b)) = (rest.find('['), rest.find(']')) {
                for part in rest[a + 1..b].split(',') {
                    let f = part.trim().trim_matches('"').trim();
                    if !f.is_empty() {
                        features.insert(f.to_string());
                    }
                }
            }
        }
        return features;
    }
    panic!("{consumer}/Cargo.toml has no `model = ` line in [dependencies]");
}

/// Every `model::<root>` used under a consumer's `src/`, with one file that
/// uses it.
fn used_roots(root: &Path, consumer: &str) -> BTreeMap<String, String> {
    let src = root.join("crates").join(consumer).join("src");
    let mut used = BTreeMap::new();
    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("a readable source directory") {
            let path = entry.expect("a readable entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a readable source file");
            let shown = path
                .strip_prefix(&src)
                .unwrap_or(&path)
                .display()
                .to_string();
            for line in text.lines() {
                let code = line.split("//").next().unwrap_or("");
                let mut rest = code;
                while let Some(i) = rest.find("model::") {
                    rest = &rest[i + 7..];
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        used.entry(name).or_insert_with(|| shown.clone());
                    }
                }
            }
        }
    }
    used
}

/// A library that uses a gated module must declare the gate.
///
/// The failure names the module, the file, and the feature to add — because
/// the two historical fixes were "add `config`" and "add `contract`", and the
/// second was applied by REPLACING the first.
#[test]
fn a_library_declares_the_features_its_library_code_uses() {
    let root = workspace_root();
    let gates = module_gates(&root);
    let mut wrong: Vec<String> = Vec::new();

    for consumer in CONSUMERS {
        let declared = declared_features(&root, consumer);
        for (module, file) in used_roots(&root, consumer) {
            let Some(needs) = gates.get(&module) else {
                continue;
            };
            let missing: Vec<&String> = needs.iter().filter(|f| !declared.contains(*f)).collect();
            if !missing.is_empty() {
                let missing: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                wrong.push(format!(
                    "  {consumer}: uses `model::{module}` (src/{file}), which needs \
                     {needs:?}, but its [dependencies] entry declares {declared:?} \
                     — ADD {missing:?}, do not replace"
                ));
            }
        }
    }

    assert!(
        wrong.is_empty(),
        "a library dependency does not cover what the library uses. This builds \
         anyway — the dev-dependency unifies the features back on — and breaks \
         only for a downstream consumer linking the library alone:\n{}",
        wrong.join("\n")
    );
}

/// The gate reader must actually be reading gates.
///
/// If `module_gates` silently returned an empty map — a changed `lib.rs`
/// layout, a parser that stopped matching — the test above would pass for
/// every possible declaration, which is the failure mode a guard cannot
/// have. So pin the three gates the historical defects were about.
#[test]
fn the_gate_reader_still_finds_gates() {
    let gates = module_gates(&workspace_root());
    assert!(
        gates.len() > 10,
        "model/src/lib.rs declares more than ten modules; the reader found {}",
        gates.len()
    );
    assert_eq!(
        gates.get("descriptor").map(BTreeSet::len),
        Some(1),
        "`descriptor` is gated on `config` alone"
    );
    assert_eq!(
        gates.get("deployment_cuda").map(BTreeSet::len),
        Some(2),
        "`deployment_cuda` is gated on `forward` AND `config` — the `all(..)` \
         case, which is the one a reader that only handles a single `feature = \
         \"..\"` gets wrong"
    );
    assert!(
        gates
            .get("deployment")
            .is_some_and(std::collections::BTreeSet::is_empty),
        "`deployment` is ungated, and an ungated module must read as \
         no requirement rather than as absent"
    );
}
