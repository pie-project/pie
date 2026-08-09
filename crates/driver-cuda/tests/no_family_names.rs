//! The driver must not know what a model is — measured, not asserted.
//!
//! `crates/model/tests/one_normalizer.rs` states the rule: *"what a
//! driver reads is the answer, never the question"*, and names its own
//! ceiling — *"nothing stops a future reader from opening `config.json`
//! again."* A grep is a guard that must be remembered, which is exactly
//! the failure mode it warns about. This is the grep, so that the
//! remembering is the build's job.
//!
//! # Why a ratchet rather than zero
//!
//! The 33-row `model_type` table is gone from this crate — it lives in
//! `model::deployment_cuda`, which is where the question is allowed to
//! be asked — and the fire path reads a `Deployment` with no family
//! name in it. What is left is not dispatch:
//!
//! - `layout/memory_planner.rs` and `model_costs.rs` size a KV pool per
//!   architecture, because the COST of a token genuinely differs. That
//!   is arithmetic about bytes, not a branch about behaviour.
//! - `gpu/bind/` names kernels, and a kernel's name is its own.
//! - `gpu/serve/encode.rs` serves gemma-4's towers, which is a real
//!   family-specific ABI entry and the next thing to generalise.
//!
//! Each is a separate argument, and collapsing them into one number
//! would hide which is which. So this pins the number per FILE: a file
//! may not gain family names, and a file that loses them ratchets the
//! bound down.
//!
//! **A file not in the table may have none at all.** That is the half
//! that matters — it is what stops a new dispatch site appearing
//! somewhere quiet.

use std::collections::BTreeMap;

/// The family names a driver must not be branching on.
const FAMILIES: &[&str] =
    &["gemma", "qwen", "llama", "deepseek", "kimi", "nemotron", "glm", "gpt_oss"];

/// Non-comment lines naming a family, per file, as of the move.
///
/// These are CEILINGS. Lowering one is the point; raising one is the
/// thing this test exists to catch.
fn budget() -> BTreeMap<&'static str, usize> {
    [
        // Cost per token differs by architecture, and that is arithmetic
        // about bytes rather than a branch about behaviour.
        ("layout/memory_planner.rs", 25),
        ("layout/model_costs.rs", 14),
        ("layout/workspace.rs", 7),
        ("layout/kv_geometry.rs", 1),
        ("layout/profile_cache.rs", 1),
        ("layout/profile_key.rs", 1),
        // A kernel's name is its own.
        ("gpu/bind/mod.rs", 14),
        // gemma-4's towers are a real family-specific ABI entry, and the
        // next thing worth generalising.
        ("gpu/serve/encode.rs", 7),
        ("gpu/serve/transfer.rs", 4),
        ("gpu/serve/load.rs", 3),
        ("gpu/serve/state.rs", 2),
        ("gpu/fire/launch.rs", 0),
        ("gpu/fire/lora.rs", 1),
    ]
    .into_iter()
    .collect()
}

fn count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    text.lines()
        .filter(|l| {
            let t = l.trim_start();
            // Comments are provenance — "ports gemma_4/forward" — not
            // dispatch. The claim is about CODE.
            !t.starts_with("//") && FAMILIES.iter().any(|f| l.contains(f))
        })
        .count()
}

fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// No file gains a family name, and no NEW file has one at all.
#[test]
fn the_driver_does_not_learn_a_new_family() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(!files.is_empty(), "the driver has source");

    let budget = budget();
    let mut over = Vec::new();
    let mut unlisted = Vec::new();
    for f in &files {
        let rel = f.strip_prefix(&root).expect("under src").to_string_lossy().into_owned();
        let n = count(f);
        if n == 0 {
            continue;
        }
        match budget.get(rel.as_str()) {
            Some(&cap) if n <= cap => {}
            Some(&cap) => over.push(format!("{rel}: {n} > {cap}")),
            None => unlisted.push(format!("{rel}: {n}")),
        }
    }
    assert!(
        over.is_empty(),
        "these files gained family names. Each one is a driver learning what a \
         model is, which is what `crates/model` exists to prevent — see \
         `model::deployment::Deployment`:\n  {}",
        over.join("\n  ")
    );
    assert!(
        unlisted.is_empty(),
        "these files name a family and are not in the budget. A NEW dispatch \
         site is the thing this test exists to catch; if the mention is \
         legitimate, add it with the argument for why:\n  {}",
        unlisted.join("\n  ")
    );
}

/// The dispatch itself is gone: no `model_type` table remains here.
///
/// This is the strong half of the claim. The 33-row `FACTS_ROWS` and its
/// eleven derivations moved to `model::deployment_cuda`; what a driver
/// receives is a `Deployment`, and a value with no family name in it is
/// a value a driver cannot branch on.
#[test]
fn no_model_type_table_remains_in_the_driver() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue };
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            assert!(
                !t.contains("FACTS_ROWS"),
                "{}:{} names the model_type table, which lives in \
                 `model::deployment_cuda` — a driver reads the answer, never \
                 the question",
                f.display(),
                i + 1
            );
        }
    }
}
