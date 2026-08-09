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
//! `model::catalog`, which is where the question is allowed to
//! be asked — and the fire path reads a `Deployment` with no family
//! name in it. What is left is not dispatch:
//!
//! - `layout/memory_planner.rs` sizes a KV pool per architecture,
//!   because the COST of a token genuinely differs. That is arithmetic
//!   about bytes, not a branch about behaviour.
//! - `bind/` names kernels, and a kernel's name is its own.
//! - `serve/encode.rs` serves gemma-4's towers, which is a real
//!   family-specific ABI entry and the next thing to generalise.
//!
//! Each is a separate argument, and collapsing them into one number
//! would hide which is which. So this pins the number per FILE: a file
//! may not gain family names, and a file that loses them ratchets the
//! bound down.
//!
//! # What the count reads
//!
//! Two corrections, both found in `driver-metal` and ported here so the
//! backends are held to one standard rather than to whichever names each
//! happened to notice. The count skips `#[cfg(test)]` blocks, because a
//! test naming the family it covers is an asset; five files left this
//! table outright once their only mentions were their own fixtures. And
//! it compares case- and punctuation-blind, which raised
//! `memory_planner.rs` from 25 to 30 and surfaced `fire/lora.rs`, a file
//! that had never been listed.
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
        ("layout/memory_planner.rs", 30),
        // A kernel's name is its own.
        // SYMBOL names, not a family branch: `ssm::nemotron_mamba_split_bf16`
        // is what the kernel is CALLED, and the join reads its arity. The
        // budget was fourteen while the arms were hand-written; the ten
        // that left were the arms.
        ("bind/mod.rs", 4),
        // gemma-4's towers are a real family-specific ABI entry, and the
        // next thing worth generalising.
        ("serve/encode.rs", 7),
        ("serve/transfer.rs", 4),
        ("serve/load.rs", 3),
        ("serve/state.rs", 2),
        // Never listed until the counter learned to read `LoraGemma`.
        ("fire/lora.rs", 3),
    ]
    .into_iter()
    .collect()
}

fn count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    // `#[cfg(test)]` opens a block that closes when its braces balance.
    // A test that names the family it covers is what makes the test
    // readable; charging it as debt taxes exactly the coverage this rule
    // needs. Depth, not "the rest of the file": a file may have several
    // test modules with real code between them.
    let mut depth = 0i32;
    let mut armed = false;
    let mut n = 0;
    for l in text.lines() {
        let opens = i32::try_from(l.matches('{').count()).unwrap_or(0);
        let closes = i32::try_from(l.matches('}').count()).unwrap_or(0);
        if l.trim_start().starts_with("#[cfg(test)]") {
            armed = true;
        }
        // Comments are provenance -- "ports gemma_4/forward" -- not
        // dispatch. The claim is about CODE.
        if depth == 0 && !armed && !l.trim_start().starts_with("//") && names_a_family(l) {
            n += 1;
        }
        if armed {
            depth += opens - closes;
            if depth <= 0 && closes > 0 {
                armed = false;
                depth = 0;
            }
        }
    }
    n
}

/// Whether a line names a family, whatever spelling it wears.
///
/// Matching raw text saw `gemma_facts` and missed `GemmaFacts`, which is
/// where family names actually appear in Rust. Reducing the line the way
/// `model/text.rs` reduces an architecture string -- drop case and
/// punctuation, then compare -- makes `GptOss`, `gpt_oss` and `gptoss`
/// one name.
fn names_a_family(line: &str) -> bool {
    let flat: String =
        line.chars().filter(|c| *c != '_' && *c != '-').flat_map(char::to_lowercase).collect();
    FAMILIES.iter().any(|f| {
        let f: String = f.chars().filter(|c| *c != '_').collect();
        flat.contains(&f)
    })
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
        // NOT `if n == 0 { continue }`. A budgeted file that fell to
        // zero is the good outcome and it still has to be recorded, or
        // the budget keeps a line that permits a mention nobody is
        // making.
        if n == 0 && !budget.contains_key(rel.as_str()) {
            continue;
        }
        match budget.get(rel.as_str()) {
            Some(&cap) if n == cap => {}
            Some(&cap) => over.push(format!("{rel}: {n}, budgeted {cap}")),
            None => unlisted.push(format!("{rel}: {n}")),
        }
    }
    // A LINE THAT LEAVES MUST LOWER THE BUDGET, which is why this is
    // `==` and not `<=`. A ceiling only ratchets one way: the arm
    // deletions took ten family names out of `bind/mod.rs` and the
    // budget kept saying fourteen, so the guard had ten free mentions to
    // give away and nobody would have noticed. That is §4's own warning
    // about a grep being "a guard that must be remembered" — a guard
    // that silently loosens is worse than one that fails, because it
    // reads as passing.
    //
    // The cost is real and it is the point: deleting a family name is a
    // two-line diff instead of one. The second line is the record that
    // it happened.
    for (rel, &cap) in &budget {
        if cap > 0 && !files.iter().any(|f| {
            f.strip_prefix(&root).is_ok_and(|r| r.to_string_lossy() == *rel)
        }) {
            over.push(format!("{rel}: budgeted {cap}, and the file is gone"));
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
/// This is the strong half of the claim, and it got stronger. The
/// 33-row `FACTS_ROWS` and its eleven derivations first MOVED, out of
/// this driver and into `model::deployment_cuda` — which was the right
/// first step and still left a table keyed on a `config.json` string,
/// merely one crate over.
///
/// They are DELETED now. There is no `model_type` table anywhere: a
/// checkpoint is matched to a `model::catalog` row by its TENSORS, and
/// what a driver receives is a `Deployment`, a value with no family
/// name in it and therefore nothing to branch on.
///
/// So this guard now watches for the table's RETURN rather than for its
/// escape, which is why the message no longer names a place it could
/// legitimately live.
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
            for banned in ["FACTS_ROWS", "HF_ROWS", "MLX_ROWS"] {
                assert!(
                    !t.contains(banned),
                    "{}:{} names `{banned}`, a table keyed on a `config.json` \
                     string. There is no such table any more — a checkpoint \
                     is matched to a `model::catalog` row by its tensors, and \
                     a driver reads the answer, never the question",
                    f.display(),
                    i + 1
                );
            }
        }
    }
}

/// Every budgeted file exists and still spends what it is budgeted.
///
/// The other half of the ratchet, ported from `driver-metal`. Without it a
/// file can be renamed, deleted, or cleaned up and leave its ceiling behind
/// as a permission nobody is using -- and the next file to need that
/// permission inherits it silently. A budget that only ever caps is a budget
/// that only ever grows. Five files left the table the first time this ran.
#[test]
fn no_budget_line_outlives_what_it_was_for() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stale = Vec::new();
    for (rel, cap) in budget() {
        let path = root.join(rel);
        if !path.exists() {
            stale.push(format!("{rel}: budgeted {cap}, file is gone"));
            continue;
        }
        let n = count(&path);
        if n < cap {
            stale.push(format!("{rel}: budgeted {cap}, actually {n} -- lower the ceiling"));
        }
    }
    assert!(
        stale.is_empty(),
        "the budget is stale. Every line below is a permission larger than \
         what uses it, and the next file to want that permission gets it for \
         free:\n  {}",
        stale.join("\n  ")
    );
}
