//! Phase-7 validation — the `context` abstraction must not reappear in the
//! inferlet-facing WIT surface (DoD §11: "repo-wide search confirms no old
//! context abstraction remains").
//!
//! This is a regression guard, not a build test: the Working Set Refactor
//! replaced the `context` resource with explicit `kv-working-set` /
//! `rs-working-set` resources + per-pass descriptors. The forbidden artifacts
//! are the *old* resource (`context.wit`, `import context;`, `borrow<context>`,
//! `resource context`). The new per-pass descriptor records/methods
//! (`kv-context`, `rs-buffer-context`, `rs-context`) legitimately contain the
//! substring "context" and are explicitly allowed.

use std::path::{Path, PathBuf};

/// Repo root = parent of the `pie` crate dir (`runtime/`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime/ has a parent")
        .to_path_buf()
}

/// The canonical core WIT package dir + its 4 vendored mirrors (all must stay
/// byte-for-byte context-free).
const WIT_CORE_DIRS: &[&str] = &[
    "interface/inferlet/core/wit",
    "interface/inferlet/deps/core",
    "interface/inferlet/zo/wit/deps/core",
    "sdk/rust/inferlet/wit/deps/core",
    "sdk/tools/bakery/src/bakery/wit/deps/core",
];

/// Whole-`.wit` substrings that only ever belong to the retired `context`
/// resource — never to the new working-set descriptors.
const FORBIDDEN: &[&str] = &["import context;", "borrow<context>", "resource context"];

#[test]
fn no_context_wit_file_remains() {
    let root = repo_root();
    for d in WIT_CORE_DIRS {
        let ctx = root.join(d).join("context.wit");
        assert!(
            !ctx.exists(),
            "retired inferlet-facing context.wit still present: {}",
            ctx.display()
        );
    }
}

#[test]
fn no_context_resource_usage_in_core_wit() {
    let root = repo_root();
    let mut scanned = 0usize;
    for d in WIT_CORE_DIRS {
        let dir = root.join(d);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue; // a mirror layout may differ; the file checks below still run
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("wit") {
                continue;
            }
            let src = std::fs::read_to_string(&path).unwrap();
            scanned += 1;
            for needle in FORBIDDEN {
                assert!(
                    !src.contains(needle),
                    "retired `context` resource usage ({needle:?}) found in {}",
                    path.display()
                );
            }
        }
    }
    assert!(scanned > 0, "no .wit files scanned — repo layout changed?");
}

#[test]
fn working_set_resources_replace_context() {
    // The replacement must be present in the canonical package.
    let ws = repo_root().join("interface/inferlet/core/wit/working-set.wit");
    let src = std::fs::read_to_string(&ws)
        .unwrap_or_else(|_| panic!("working-set.wit present at {}", ws.display()));
    assert!(
        src.contains("resource kv-working-set"),
        "kv-working-set resource missing"
    );
    assert!(
        src.contains("resource rs-working-set"),
        "rs-working-set resource missing"
    );
    // The constructor takes no handle: the single-model runtime binds the
    // global bound model implicitly (this replaced the old context-creation
    // path; multi-model `constructor(model:)` is gone).
    assert!(
        src.contains("constructor()"),
        "working-set constructor should be the no-arg single-model bind"
    );
}
