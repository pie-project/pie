//! Two tables, one question, and a test that makes them answer it together.
//!
//! "Which `model_type` does pie support?" is asked in three places, on two
//! sides of the C ABI:
//!
//! * [`model::contract::HF_ROWS`] / [`MLX_ROWS`] — which author writes the
//!   storage contract (Rust, this crate);
//! * `crates/driver-metal/csrc/src/model/facts.hpp` — the same question for
//!   Metal, whose `model_family_of` picks both the geometry and the executor.
//!
//! It used to be three, on two sides of the C ABI. The CUDA arch table was
//! `crates/driver-cuda/csrc/src/model/registry.cpp`, and it is deleted with
//! that shell; `driver-cuda-new`'s `FACTS_ROWS` answers for CUDA now, and
//! `driver-cuda-new/tests/facts_registry.rs` holds it against `HF_ROWS`
//! from both sides — a stronger check than a source grep, because both
//! tables are Rust.
//!
//! They are two because they answer *different* things — an author is a
//! storage schema, an arch row is a compute graph, and the two partition the
//! model space differently (`phi3` loads as llama and runs as its own row).
//! What they may not do is disagree about the **set**. A `model_type` with an
//! arch row and no author gets a plan-time "no author for model_type"; one
//! with an author and no arch row gets a boot-time "unsupported model_type".
//! Two unrelated-looking errors, one cause: a family added on one side only.
//!
//! So this is a source grep, in the idiom of `loader/tests/standalone.rs` —
//! the property is about what the *other* language declares, and no amount of
//! Rust-side assertion can reach it. Both C++ tables are plain literal lists
//! by construction, which is what makes them greppable; a table that stopped
//! being one would fail the "did we find anything at all" guard below rather
//! than silently passing with an empty set.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use model::contract::{Author, HF_ROWS, MLX_ROWS};

/// The repository root, from this crate's manifest.
fn repo_root() -> PathBuf {
    // `crates/model` -> the repo root, two levels up.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/model sits two levels below the repo root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

/// Every double-quoted lowercase identifier in `haystack`.
///
/// Deliberately blunt: both call sites hand over a region that contains
/// nothing but the table, so a filter any cleverer than this would be a
/// second thing to keep in step with the C++.
fn quoted(haystack: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = haystack;
    while let Some(open) = rest.find('"') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('"') else { break };
        let token = &rest[..close];
        rest = &rest[close + 1..];
        if !token.is_empty()
            && token
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            out.insert(token.to_string());
        }
    }
    out
}

/// The region of `source` from `start` up to `end`, panicking if either
/// anchor is gone. A moved anchor is a real failure: it means the table was
/// restructured, and a restructured table is exactly when this check is worth
/// re-reading.
fn region<'a>(source: &'a str, what: &str, start: &str, end: &str) -> &'a str {
    let from = source
        .find(start)
        .unwrap_or_else(|| panic!("{what}: anchor `{start}` is gone; has the table moved?"));
    let tail = &source[from..];
    let to = tail
        .find(end)
        .unwrap_or_else(|| panic!("{what}: anchor `{end}` is gone; has the table moved?"));
    &tail[..to]
}

fn metal_family_table() -> BTreeSet<String> {
    let source = read("crates/driver-metal/csrc/src/model/facts.hpp");
    let table = region(
        &source,
        "metal facts.hpp",
        "inline ModelFamily model_family_of",
        "inline bool is_supported_model_type",
    );
    quoted(table)
}

fn rows_of(rows: &[(&str, Author)]) -> BTreeSet<String> {
    rows.iter().map(|(name, _)| (*name).to_string()).collect()
}

/// Report a set difference the way a reader can act on it.
fn assert_same(what: &str, rust: &BTreeSet<String>, native: &BTreeSet<String>, native_where: &str) {
    // An empty native set means the grep stopped matching, not that a driver
    // supports nothing. Fail loudly rather than passing vacuously.
    assert!(
        !native.is_empty(),
        "{native_where}: found no model types; the table's shape changed and \
         this test is no longer reading it"
    );
    let missing_author: Vec<_> = native.difference(rust).cloned().collect();
    let missing_arch: Vec<_> = rust.difference(native).cloned().collect();
    assert!(
        missing_author.is_empty() && missing_arch.is_empty(),
        "{what}: the two tables disagree.\n  \
         in {native_where} but with no author in model::contract: {missing_author:?}\n    \
         → a boot of one of these reaches pie_loader_compile_model and gets \
         \"no author for model_type\"\n  \
         has an author but is absent from {native_where}: {missing_arch:?}\n    \
         → a row that authors a contract for a model the driver cannot run"
    );
}

// THE CUDA ARM IS GONE WITH ITS TABLE. `crates/driver-cuda`'s
// `registry.cpp` was one of the three places "which model_type does pie
// support?" was answered, and it is deleted along with the C++ shell.
//
// The question did not go with it — it moved into Rust, where it is a
// stronger check than a source grep could be. `driver-cuda-new`'s
// `FACTS_ROWS` is the arch table now, and
// `crates/driver-cuda-new/tests/facts_registry.rs` holds it against
// `HF_ROWS` from both sides: nothing openable without an author, and the
// complement stated as `NOT_YET_OPENABLE`.
//
// The Metal arm below still greps a real table, and stays until that
// shell retires the same way.


#[test]
fn metal_family_table_and_the_mlx_authors_name_the_same_models() {
    assert_same(
        "Metal",
        &rows_of(MLX_ROWS),
        &metal_family_table(),
        "crates/driver-metal/csrc/src/model/facts.hpp",
    );
}

/// A row appearing twice would shadow the second silently — the lookup takes
/// the first match — so the table's own shape is worth one assertion.
#[test]
fn no_model_type_is_declared_twice() {
    for (what, rows) in [("HF_ROWS", HF_ROWS), ("MLX_ROWS", MLX_ROWS)] {
        let unique: BTreeSet<_> = rows.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            unique.len(),
            rows.len(),
            "{what}: a model_type is listed more than once; the later row is dead"
        );
    }
}
