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
//! Two of those three modules no longer exist and neither does the
//! `config` feature — the catalog refactor deleted all of it. The table
//! stays because the LESSON is about the resolver, not about those
//! modules: the same trap is available to `model::catalog`'s
//! `forward`-gated methods the moment a consumer forgets to name the
//! aspect, and the gates are read from `lib.rs` rather than listed here
//! precisely so this test keeps working across that kind of change.
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
    let mut gates = BTreeMap::new();
    // TWO files, because the vocabulary a consumer names moved.
    //
    // It used to be one: `model::builder`, `model::policy` and the rest
    // were `pub mod` lines in `lib.rs`, each carrying its own aspect
    // gate, and reading that file was reading every gate. They live in
    // `shared/` now, behind a `pub mod shared;` that is UNGATED — so a
    // reader that stopped at `lib.rs` would resolve every
    // `model::shared::builder` to "no requirement" and report nothing,
    // forever. That is the disarmed-guard failure this file exists to
    // prevent, arriving by a layout change rather than a code one.
    //
    // The inner modules are keyed `shared::<name>`, which is the path a
    // consumer actually writes.
    for (file, prefix) in [
        ("crates/model/src/lib.rs", ""),
        ("crates/model/src/shared/mod.rs", "shared::"),
    ] {
        read_gates_into(&mut gates, &std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("{file} is readable: {e}")), prefix);
    }
    gates
}

/// One file's `pub mod` lines, folded into the map under `prefix`.
fn read_gates_into(
    gates: &mut BTreeMap<String, BTreeSet<String>>,
    src: &str,
    prefix: &str,
) {
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
            gates.insert(format!("{prefix}{name}"), needs);
        }
        if !line.starts_with("///") && !line.starts_with("//") && !line.is_empty() {
            pending = None;
        }
    }
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
        if let Some(i) = entry.find("features = [") {
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
        // A `features = [..]` list is what a consumer ADDS, not what it
        // gets. Unless it opts out, cargo also turns on `default`, and a
        // reader that ignored that would report a consumer wrong for a
        // feature it holds — `engine` names `model::instruct`, declares
        // only `forward`, and compiles, because `default = ["chat"]`.
        //
        // A guard that cries wolf gets deleted, and this one is worth
        // keeping, so it has to read the same rule cargo does.
        if !entry.contains("default-features = false") {
            features.extend(default_features(root));
        }
        return features;
    }
    panic!("{consumer}/Cargo.toml has no `model = ` line in [dependencies]");
}

/// The features `model` turns on when a consumer does not opt out.
fn default_features(root: &Path) -> BTreeSet<String> {
    let manifest = std::fs::read_to_string(root.join("crates/model/Cargo.toml"))
        .expect("model/Cargo.toml is readable");
    let mut section = String::new();
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            section = t.to_string();
            continue;
        }
        if section != "[features]" || !t.starts_with("default = [") {
            continue;
        }
        let (Some(a), Some(b)) = (t.find('['), t.find(']')) else {
            break;
        };
        return t[a + 1..b]
            .split(',')
            .map(|p| p.trim().trim_matches('"').trim().to_string())
            .filter(|f| !f.is_empty())
            .collect();
    }
    BTreeSet::new()
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
                    let ident = |t: &str| -> String {
                        t.chars()
                            .take_while(|c| {
                                c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_'
                            })
                            .collect()
                    };
                    let name = ident(rest);
                    if name.is_empty() {
                        continue;
                    }
                    // `shared` is a door and not a module a gate hangs on, so
                    // the name that matters is the one BEHIND it. Recording
                    // `shared` alone would resolve every use of the authoring
                    // DSL to the ungated outer module and report nothing.
                    let name = if name == "shared" {
                        let after = &rest[name.len()..];
                        match after.strip_prefix("::").map(ident) {
                            Some(inner) if !inner.is_empty() => format!("shared::{inner}"),
                            _ => name,
                        }
                    } else {
                        name
                    };
                    used.entry(name).or_insert_with(|| shown.clone());
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
        gates.get("shared::weight_names").map(BTreeSet::len),
        Some(1),
        "`weight_names` is gated on `contract` alone — and is keyed under \
         `shared::` because that is the path a consumer writes"
    );
    assert!(
        gates.contains_key("shared::builder"),
        "the reader must descend into `shared/mod.rs`. Every module a \
         consumer names for its vocabulary lives there now, behind an \
         UNGATED `pub mod shared;` — a reader that stopped at lib.rs would \
         resolve all of them to `shared`, find no requirement, and pass \
         forever"
    );
    assert!(
        gates
            .get("deployment")
            .is_some_and(std::collections::BTreeSet::is_empty),
        "`deployment` is ungated, and an ungated module must read as \
         no requirement rather than as absent"
    );
}

/// THE `any(..)` CASE, asked of the reader directly.
///
/// This is the branch that decides whether a consumer gets reported
/// wrong for a gate it satisfies: `any(chat, forward)` is satisfiable
/// two ways, so recording it as a requirement of BOTH names would fail a
/// driver for a module it can legitimately see.
///
/// It used to be pinned against real source — `deployment_cuda` was
/// `all(forward, config)` and `families` was `any(chat, forward)`. Both
/// are gone: `config` was deleted with the descriptor, and `families`
/// became `shared/`, whose modules each carry a single-feature gate. No
/// `any(..)` is left in the crate's surface.
///
/// So it is fed one. A branch with no live example is a branch that
/// rots, and deleting the assertion instead would leave the code with
/// nothing holding it — the same disarmed-guard shape the rest of this
/// file is about. The input is spelled out here rather than found, which
/// is the honest form of the claim: this is what the reader DOES, not
/// what the tree happens to contain today.
#[test]
fn an_any_gate_reads_as_no_requirement_rather_than_as_both() {
    let mut gates = BTreeMap::new();
    read_gates_into(
        &mut gates,
        r#"
#[cfg(any(feature = "chat", feature = "forward"))]
pub mod either;
#[cfg(all(feature = "chat", feature = "forward"))]
pub mod both;
#[cfg(feature = "contract")]
pub mod one;
pub mod ungated;
"#,
        "",
    );

    assert_eq!(
        gates.get("either").map(BTreeSet::len),
        Some(0),
        "`any(..)` is satisfiable more than one way, so it is NO \
         requirement — not a requirement of every name in it"
    );
    assert!(
        gates.contains_key("either"),
        "and it must be SEEN. `None` also reads as 'no requirement' at \
         the call site, so a missed module and an unrequired one would be \
         indistinguishable"
    );
    assert_eq!(
        gates.get("both").map(BTreeSet::len),
        Some(2),
        "`all(..)` demands every name in it"
    );
    assert_eq!(gates.get("one").map(BTreeSet::len), Some(1));
    assert_eq!(
        gates.get("ungated").map(BTreeSet::len),
        Some(0),
        "an ungated module reads as no requirement rather than as absent"
    );
}

/// The two halves of what a consumer actually holds.
///
/// `engine` is the live case: it declares `features = ["forward"]` and
/// names `model::instruct`, which is gated on `chat`. It compiles,
/// because it never wrote `default-features = false` and `default =
/// ["chat"]`. `driver-cuda` is the other half — it opts out, so its
/// list is the whole of it, and `chat` must NOT be granted to it.
#[test]
fn a_consumer_holds_the_defaults_it_did_not_opt_out_of() {
    let root = workspace_root();
    let defaults = default_features(&root);
    assert!(
        defaults.contains("chat"),
        "model's `default` is where this rule gets its teeth; the reader \
         found {defaults:?}"
    );

    assert!(
        declared_features(&root, "engine").contains("chat"),
        "`engine` does not opt out, so it holds `chat` whether or not it \
         names it — reporting it wrong here is how this guard would earn \
         its deletion"
    );
    assert!(
        !declared_features(&root, "driver-cuda").contains("chat"),
        "`driver-cuda` writes `default-features = false` precisely so a \
         driver link carries no chat template. Granting it the defaults \
         anyway would let that regress unnoticed"
    );
}

/// The prefix is applied, so an inner module is keyed by the path a
/// consumer writes.
#[test]
fn an_inner_module_is_keyed_under_its_door() {
    let mut gates = BTreeMap::new();
    read_gates_into(
        &mut gates,
        "#[cfg(feature = \"contract\")]\npub mod builder;\n",
        "shared::",
    );
    assert_eq!(
        gates.get("shared::builder").map(BTreeSet::len),
        Some(1),
        "`model::shared::builder` is what a consumer writes, so that is \
         the key its gate has to hang on"
    );
    assert!(
        !gates.contains_key("builder"),
        "the bare name would collide with a root module of the same name"
    );
}
