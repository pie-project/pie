//! The driver must not know what a model is — measured, not asserted.
//!
//! `crates/model/tests/one_normalizer.rs` states the rule; this file is the
//! family-level half, the one that needs a per-crate budget.
//!
//! A checkpoint is matched to a `model::catalog` row BY ITS TENSORS and the
//! row projects a `Deployment` with no family name to branch on, so the strong
//! half below watches for the table's RETURN rather than its continued
//! existence. A ratchet rather than zero because not every mention is
//! dispatch, and **a file not in the table may have none at all.**

use std::collections::BTreeMap;

/// The abbreviations a family name hides behind once a field is named for it.
///
/// `FAMILIES` spells names out, so it saw `model_type == "gpt_oss"` and missed
/// `go_hidden_size`, `ll_rms_norm_eps`, `g4_num_experts` and `q35_head_dim`
/// entirely — backwards, since the spelled-out mentions are mostly key strings
/// this crate does not choose while the prefixes ARE north-star #7.
///
/// Matched at the start of an identifier, not anywhere in the line: `ll_` is a
/// substring of `full_attn_interval` and `go_` of `dialog_state`.
const PREFIXES: &[&str] = &["go_", "ll_", "g4_", "q35_"];

/// Whether a line names a family, spelled out or abbreviated.
fn names_a_family(line: &str) -> bool {
    // The same reduction the driver applied to an architecture string, for the
    // same reason: a publisher's capitalisation and punctuation are not the
    // identity. Matching raw text saw `gemma_facts` and missed `GemmaFacts`,
    // which is where family names actually appear in Rust.
    let flat: String = line
        .chars()
        .filter(|c| *c != '_' && *c != '-')
        .flat_map(char::to_lowercase)
        .collect();
    if FAMILIES.iter().any(|f| {
        let f: String = f.chars().filter(|c| *c != '_').collect();
        flat.contains(&f)
    }) {
        return true;
    }
    line.split(|c: char| !c.is_alphanumeric() && c != '_')
        .any(|tok| PREFIXES.iter().any(|p| tok.starts_with(p)))
}

/// The family names a driver must not be branching on — the same list
/// `driver-cuda` uses, so the two backends are held to one standard rather
/// than to whichever names each happened to notice.
const FAMILIES: &[&str] = &[
    "gemma", "qwen", "llama", "deepseek", "kimi", "nemotron", "glm", "gpt_oss",
];

/// Non-comment lines naming a family, per file, as of this test's writing.
///
/// These are CEILINGS. Lowering one is the point; raising one is the thing
/// this test exists to catch.
fn budget() -> BTreeMap<&'static str, usize> {
    [
        // A refusal naming the model whose shape caused it: gemma-4 pages two
        // KV geometries, and an operator reading "this pool's layers have two
        // page sizes" needs to know which model does that.
        ("pools/kv.rs", 1),
        // The same, for a control-path refusal that names where the real
        // check lives.
        ("serve/control.rs", 1),
        // The kernel ABI's family-prefixed kind names — every mention is an
        // abbreviation, so this file was invisible until `PREFIXES` existed.
        // They name real per-architecture kernels (gemma4's PLE, gpt-oss's
        // attention sink), which is a different claim from a duplicated read.
        ("lowering/abi.rs", 45),
        // ONE each, and both are the SHADER's own entrypoint name.
        // `mlp/gated.metal` publishes `gptoss_swiglu` -- a clamped,
        // alpha-scaled SwiGLU no other family's activation matches -- and a
        // driver that dispatches it has to spell it. `routine.rs` names the
        // row, `arm.rs` binds its operands.
        //
        // This is the same claim `abi.rs` carries above and not a new one:
        // the string is a compiled entrypoint's identity, not a fact about
        // which model is running. Nothing here branches on it. Renaming the
        // kernel would move the mention rather than remove it, since the
        // activation really is one deployment's.
        ("lowering/routine.rs", 1),
        ("lowering/arm.rs", 1),
    ]
    .into_iter()
    .collect()
}

/// Lines of DISPATCH that name a family: comments and tests excluded.
///
/// Comments are provenance, not dispatch, and a test is the opposite of the
/// thing being counted — `assert!(serves(row))` is the guard on the code that
/// must know. Counting tests made writing one cost budget, which taxes exactly
/// the work this rule needs.
fn count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    // `#[cfg(test)]` opens a module; it closes when its braces balance.
    // Tracking depth rather than assuming the file ends there matters:
    // `batch/geometry.rs` has real code after its test module.
    let mut depth = 0i32;
    let mut armed = false;
    let mut n = 0;
    for l in text.lines() {
        let opens = i32::try_from(l.matches('{').count()).unwrap_or(0);
        let closes = i32::try_from(l.matches('}').count()).unwrap_or(0);
        if l.trim_start().starts_with("#[cfg(test)]") {
            armed = true;
        }
        let inside = depth > 0;
        if !inside && !armed && !l.trim_start().starts_with("//") && names_a_family(l) {
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

fn sources() -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(
        files.len() > 40,
        "found {} source files under {}; the layout moved and this guard is \
         now looking in the wrong place — which is exactly how \
         `one_normalizer.rs` stopped guarding this crate",
        files.len(),
        root.display()
    );
    (root, files)
}

/// No file gains a family name, and no NEW file has one at all.
#[test]
fn the_driver_does_not_learn_a_new_family() {
    let (root, files) = sources();
    let budget = budget();
    let mut over = Vec::new();
    let mut unlisted = Vec::new();
    for f in &files {
        let rel = f
            .strip_prefix(&root)
            .expect("under src")
            .to_string_lossy()
            .into_owned();
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
        "these files gained family names. Each one is a driver learning what \
         a model is, which is what `crates/model` exists to prevent — see \
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

/// Every budgeted file exists and still spends what it is budgeted.
///
/// The other half of the ratchet. Without it a file can be renamed, deleted or
/// cleaned up and leave its ceiling behind as a permission nobody is using,
/// which the next file to need that permission inherits silently.
#[test]
fn no_budget_line_outlives_what_it_was_for() {
    let (root, _) = sources();
    let mut stale = Vec::new();
    for (rel, cap) in budget() {
        let path = root.join(rel);
        if !path.exists() {
            stale.push(format!("{rel}: budgeted {cap}, file is gone"));
            continue;
        }
        let n = count(&path);
        if n < cap {
            stale.push(format!(
                "{rel}: budgeted {cap}, actually {n} — lower the ceiling"
            ));
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

/// The strong half, INVERTED: no `model_type` table remains here.
///
/// The claim is that no table keyed on a `config.json` string exists anywhere
/// on the path — a checkpoint is matched to a `model::catalog` row BY ITS
/// TENSORS, and what a driver receives is a `Deployment`, a value with no
/// family name in it. So this watches for the table's RETURN, which is why the
/// message names no place it could legitimately live.
///
/// `arch_stem` is banned beside the three table names because it was the
/// FUNCTION that made a dispatch key: it lowercased `Qwen3MoeForCausalLM` and
/// dropped the tail, and the two spellings that produced are how a load gate
/// and the seam came to answer differently about the same checkpoint.
#[test]
fn no_model_type_table_remains_in_the_driver() {
    // `sources()` asserts it read more than forty files, which is this
    // scan's own "a broken audit passes" guard: a walk that finds nothing
    // proves nothing, loudly and forever.
    let (root, files) = sources();
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            for banned in [
                "FACTS_ROWS",
                "HF_ROWS",
                "MLX_ROWS",
                "ModelFamily",
                "arch_stem",
            ] {
                assert!(
                    !t.contains(banned),
                    "{}:{} names `{banned}`, which is a table — or the \
                     reduction that made a key for one — over a `config.json` \
                     string. There is no such table any more: a checkpoint is \
                     matched to a `model::catalog` row by its tensors, and a \
                     driver reads the answer, never the question",
                    f.strip_prefix(&root).unwrap_or(f).display(),
                    i + 1
                );
            }
        }
    }
}
