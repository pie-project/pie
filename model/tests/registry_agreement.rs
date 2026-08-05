//! Three tables, one question, and a test that makes them answer it together.
//!
//! "Which `model_type` does pie support?" is asked in three places, on two
//! sides of the C ABI:
//!
//! * [`pie_model::contract::HF_ROWS`] / [`MLX_ROWS`] — which author writes the
//!   storage contract (Rust, this crate);
//! * `driver/cuda/src/model/registry.cpp` — which decode DAG binds and runs
//!   (C++, the CUDA driver's arch table);
//! * `driver/metal/src/model/facts.hpp` — the same question for Metal, whose
//!   `model_family_of` picks both the geometry and the executor.
//!
//! They are three because they answer *different* things — an author is a
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

use pie_model::contract::{Author, HF_ROWS, MLX_ROWS};

/// The repository root, from this crate's manifest.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("model/ has a parent")
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

/// The CUDA arch table's `model_type` column.
///
/// Two row shapes, and the reader has to know both because a row carries a
/// *second* string — the binder key — that is not a model type:
///
/// ```text
/// push("phi3", Family::LlamaLike, "phi3", …);
/// for (const char* mt : {"olmo2", "olmo3"}) push(mt, Family::LlamaLike, "olmo3", …);
/// ```
///
/// So take `push`'s first argument when it is a literal, and every element of
/// the loop list when it is not.
fn cuda_arch_table() -> BTreeSet<String> {
    let source = read("driver/cuda/src/model/registry.cpp");
    let table = region(
        &source,
        "cuda registry.cpp",
        "auto push = [&t]",
        "const ArchEntry* find_arch_entry",
    );

    let mut out = BTreeSet::new();
    for row in table.split("push(").skip(1) {
        let first_arg = row.split(',').next().unwrap_or("").trim();
        // `push(mt, …)` inside a loop: the types are in the list above it.
        if first_arg.starts_with('"') {
            out.extend(quoted(first_arg));
        }
    }
    for loop_head in table.split("for (const char* mt : {").skip(1) {
        let list = loop_head.split('}').next().unwrap_or("");
        out.extend(quoted(list));
    }
    out
}

/// Metal's `model_family_of`, which answers the same question for that driver.
fn metal_family_table() -> BTreeSet<String> {
    let source = read("driver/metal/src/model/facts.hpp");
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
         in {native_where} but with no author in pie_model::contract: {missing_author:?}\n    \
         → a boot of one of these reaches pie_loader_compile_model and gets \
         \"no author for model_type\"\n  \
         has an author but is absent from {native_where}: {missing_arch:?}\n    \
         → a row that authors a contract for a model the driver cannot run"
    );
}

#[test]
fn cuda_arch_table_and_the_hf_authors_name_the_same_models() {
    assert_same(
        "CUDA",
        &rows_of(HF_ROWS),
        &cuda_arch_table(),
        "driver/cuda/src/model/registry.cpp",
    );
}

#[test]
fn metal_family_table_and_the_mlx_authors_name_the_same_models() {
    assert_same(
        "Metal",
        &rows_of(MLX_ROWS),
        &metal_family_table(),
        "driver/metal/src/model/facts.hpp",
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
