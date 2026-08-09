//! The citations resolve.
//!
//! This crate's comments carry an unusual amount of prose, and the prose earns
//! its place by being specific: it names the file that broke, the pass that
//! rewrites the instruction, the test that would have caught it. A citation is
//! the only part of a comment a reader can *act* on — the rest they must take
//! on faith.
//!
//! Which makes a dead citation worse than no citation. It costs a reader a
//! search to discover that the thing they were pointed at is gone, and it
//! costs them the assumption that the surrounding sentence was ever true.
//! Three modules were cited here after being deleted or renamed —
//! weight_store.rs, main.rs, host.rs — and a package this tree never had,
//! model-loader-capi, was named by two manifests and a README. Nothing
//! noticed, because prose has no compiler.
//!
//! This is the compiler, and it reads one convention: **a backticked name is
//! a claim that the thing is there.** Two rules follow, both narrow enough to
//! be unambiguous:
//!
//! * a backticked Rust path must name a file that exists, and
//! * a backticked workspace crate name must name a crate that exists.
//!
//! Which leaves prose a way to talk about what is gone: name it without
//! backticks, as this comment just did four times. That is not a loophole —
//! it is the distinction the check exists to protect. A reader who sees
//! backticks should be able to open the file.
//!
//! Narrow on purpose in two other ways. `architecture.md` and `spec.md` live
//! in a separate repository that is not checked out beside this one, so
//! nothing here could resolve them; and C++ headers like
//! `transcode_engine.hpp` are cited as provenance for what replaced them,
//! where naming the dead thing precisely IS the point. Neither is Rust, which
//! is where the rot was.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/model-loader is two levels below the workspace root")
        .to_path_buf()
}

/// Every file this crate's own text could be pointing at.
///
/// Paths are relative to the workspace root and slash-separated, so that a
/// citation of `plan/passes/tile.rs` can be matched against
/// `crates/model-loader/src/plan/passes/tile.rs` by suffix.
fn workspace_files(root: &Path) -> BTreeSet<String> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeSet<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
            if path.is_dir() {
                walk(&path, root, out);
            } else if let Ok(rel) = path.strip_prefix(root) {
                out.insert(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(root, root, &mut out);
    out
}

/// This crate's text: every `.rs` file under `src/` and `tests/`, plus the
/// files that describe the crate to the outside.
fn crate_text() -> Vec<(String, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut paths: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
        paths.sort();
        for path in paths {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = Vec::new();
    walk(&manifest.join("src"), &mut paths);
    walk(&manifest.join("tests"), &mut paths);
    paths.push(manifest.join("README.md"));
    paths.push(manifest.join("Cargo.toml"));
    paths
        .into_iter()
        .filter_map(|path| {
            let text = std::fs::read_to_string(&path).ok()?;
            let name = path
                .strip_prefix(manifest)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            Some((name, text))
        })
        .collect()
}

/// Every backtick-delimited span in `text`, with the 1-based line it opened on.
///
/// Only backticked spans count. A citation a reader can act on is one the
/// author marked as code; bare prose mentioning a filename is a sentence, not
/// a pointer, and holding it to the same standard would make the check
/// unusable in exactly the files that have the most to say.
fn quoted(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let mut rest = line;
        while let Some(open) = rest.find('`') {
            let after = &rest[open + 1..];
            let Some(close) = after.find('`') else { break };
            out.push((index + 1, after[..close].to_string()));
            rest = &after[close + 1..];
        }
    }
    out
}

/// A citation with its line range stripped: `plan/index.rs:57-61` names a file
/// whatever the numbers say, and line numbers rot on their own schedule.
fn cited_path(span: &str) -> Option<&str> {
    let path = span.split(':').next()?;
    if !path.ends_with(".rs") {
        return None;
    }
    // A path, not a sentence: no spaces, no markup, and nothing that would
    // make it a pattern rather than a name.
    let printable = |ch: char| ch.is_ascii_alphanumeric() || "._-/".contains(ch);
    // `.rs` and `*.rs` are the extension, not a file with a name.
    let named = path.len() > 3 && !path.starts_with('.');
    (named && path.chars().all(printable)).then_some(path)
}

/// A citation with no `/` in it — `verify.rs`, `types.rs` — is read as naming
/// a file in THIS crate, because that is what it means when a comment here
/// says it. Resolving such a name against the whole workspace would let
/// main.rs and host.rs pass on the strength of some other crate's file, which
/// is precisely the confusion the check is for. A citation that carries a
/// path is a deliberate reach outward and resolves workspace-wide.
fn resolves(path: &str, files: &BTreeSet<String>) -> bool {
    let within = if path.contains('/') {
        ""
    } else {
        "crates/model-loader/"
    };
    files.iter().any(|file| {
        file.starts_with(within) && (file == path || file.ends_with(&format!("/{path}")))
    })
}

#[test]
fn every_cited_rust_file_exists() {
    let root = workspace_root();
    let files = workspace_files(&root);
    let mut offences = Vec::new();
    for (source, text) in crate_text() {
        for (line, span) in quoted(&text) {
            let Some(path) = cited_path(&span) else {
                continue;
            };
            if !resolves(path, &files) {
                offences.push(format!("{source}:{line} cites `{span}`"));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "these comments point at Rust files that are not there. A citation is \
         the one part of a comment a reader can act on, so backticks are a \
         claim that the file exists; to name something deleted, drop the \
         backticks:\n{}",
        offences.join("\n")
    );
}

#[test]
fn every_cited_workspace_crate_exists() {
    // The workspace's own naming families. A backticked token shaped like one
    // of these is a package reference and nothing else — which is what makes
    // it checkable, where a bare hyphenated word would not be.
    const FAMILIES: &[&str] = &[
        "client-",
        "controller-",
        "driver-",
        "kernels-",
        "model-",
        "tensor-",
        "worker-",
    ];
    let root = workspace_root();
    let crates: BTreeSet<String> = std::fs::read_dir(root.join("crates"))
        .expect("crates/")
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();

    let mut offences = Vec::new();
    for (source, text) in crate_text() {
        for (line, span) in quoted(&text) {
            let named = FAMILIES.iter().any(|family| span.starts_with(family))
                && span
                    .chars()
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-');
            if named && !crates.contains(&span) {
                offences.push(format!("{source}:{line} cites `{span}`"));
            }
        }
    }
    assert!(
        offences.is_empty(),
        "these comments name workspace packages that are not there. Two \
         manifests and a README named model-loader-capi, a crate this tree \
         never had; to name a package that is gone, drop the backticks:\n{}",
        offences.join("\n")
    );
}
