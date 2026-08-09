//! The driver must not know what a model is — measured, not asserted.
//!
//! `crates/model/tests/one_normalizer.rs` states the rule: *"what a driver
//! reads is the answer, never the question"*, and three documents cite it as
//! the thing enforcing that rule on this crate. **It did not.** Its metal
//! half walked `crates/driver-metal/csrc/src` and was deleted along with that
//! tree — honestly, and the file records the deletion — leaving a Rust walk
//! over `model` and `engine` only. The citations outlived the code they
//! cited, which is `north-star.md` rule 4's failure mode landing on the very
//! test written to name it.
//!
//! Two things close that. `one_normalizer.rs` now walks both driver trees for
//! `config.json` reads, since both are Rust and the walk is the same walk.
//! And this file is the family-level half, which is the one that needs a
//! per-crate budget. It is modelled on
//! `driver-cuda/tests/no_family_names.rs`, which is what the other backend
//! ended up with after its own §7.
//!
//! # This test passes at numbers that are the point
//!
//! It passes today, at numbers that are an indictment. CUDA's
//! equivalent guards a driver that already moved its `model_type` table into
//! `model::deployment_cuda`; this one guards a driver that has not, so the
//! budget below starts at 150 lines across 10 files and three of those files
//! hold real dispatch:
//!
//! | file | lines | what it is |
//! |---|---|---|
//! | `facts.rs` | 68 | the `model_type` table itself, 81 fields and four family blocks |
//! | `model/text.rs` | 46 | per-family text-model construction |
//! | `batch/geometry_facts.rs` | 24 | the projection from those facts into kernel arguments |
//!
//! The point of writing it now, BEFORE north-star #7 moves any of that, is
//! that #7 is a change with no compile error attached to its failure mode:
//! the survey found `lowering/consts.rs` binds `hidden`, `intermediate`,
//! `moe_intermediate`, `n_experts`, `vocab`, `eps`, `gdn_conv_dim` and
//! `gdn_v_total` off `DecodeGeometry`, and none of those exist on
//! `Deployment`. A #7 done as "call `deployment_from` and delete `facts.rs`"
//! produces a **GPU fault, not a compiler message**. A ratchet that has to be
//! lowered by hand, file by file, is how that move stays honest.
//!
//! # Why a ratchet rather than zero
//!
//! Not every mention is dispatch, and collapsing them into one number would
//! hide which is which. Four of the ten files below name a family for a
//! reason that survives #7 entirely — a refusal's message, a test's name, a
//! deprecation note — and pinning per FILE keeps those separable from the
//! three that are the actual question being asked in the wrong crate.
//!
//! **A file not in the table may have none at all.** That is the half that
//! matters: it is what stops a new dispatch site appearing somewhere quiet
//! while the loud ones are being cleaned up.

use std::collections::BTreeMap;

/// The family names a driver must not be branching on.
///
/// The same list `driver-cuda` uses, so the two backends are held to one
/// standard rather than to whichever names each happened to notice.
const FAMILIES: &[&str] =
    &["gemma", "qwen", "llama", "deepseek", "kimi", "nemotron", "glm", "gpt_oss"];

/// Non-comment lines naming a family, per file, as of this test's writing.
///
/// These are CEILINGS. Lowering one is the point; raising one is the thing
/// this test exists to catch.
fn budget() -> BTreeMap<&'static str, usize> {
    [
        // ── The three that ARE the question ──────────────────────────
        //
        // north-star #7 is exactly the work of driving these to zero. Each
        // number here is a debt, not an allowance.
        //
        // `facts.rs` is the `model_type` table: 81 fields, four family
        // blocks, filled by asking a `serde_json::Value` what kind of model
        // it is. `model::deployment_cuda` is where the other backend put
        // its equivalent.
        ("facts.rs", 68),
        // Per-family construction of the text model.
        ("model/text.rs", 46),
        // The projection from those facts into `DecodeGeometry`'s 60
        // kernel-argument fields. This is the half `Deployment` cannot
        // express today, and the reason #7 is not a call swap.
        ("batch/geometry_facts.rs", 24),
        // ── The seven that are not ───────────────────────────────────
        //
        // A refusal naming the model whose shape caused it: gemma-4 pages
        // two KV geometries, and an operator reading "this pool's layers
        // have two page sizes" needs to know which model does that.
        ("gpu/pools/kv.rs", 1),
        // The same, for a control-path refusal that names where the real
        // check lives.
        ("gpu/serve/control.rs", 1),
        // A `#[deprecated]` note pointing at what replaced it.
        ("lowering/resolve.rs", 2),
        // A TEST's name and a test fixture's `model_type` string. Naming
        // the family a test covers is what makes the test readable; this
        // is the one place a family name is an asset.
        ("model/rope.rs", 1),
        ("loader/plan.rs", 2),
        // Two `use` paths into `model::families::llama_like`, which is
        // `crates/model`'s own cross-generation sharing module. Reading a
        // shared module that happens to be named after the generation that
        // first needed it is not branching on a family — but it is also
        // not nothing, and §5 has the rename as an open item.
        ("gpu/serve/state.rs", 2),
        // A bool field named for the family whose attention scaling it
        // selects. Real dispatch, small enough to have hidden: it belongs
        // with the other three above and is listed apart only because
        // `geometry.rs` is otherwise clean.
        ("batch/geometry.rs", 2),
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
/// The other half of the ratchet, and the half `one_normalizer.rs` did not
/// have. Without it a file can be renamed, deleted, or cleaned up and leave
/// its ceiling behind as a permission nobody is using — and the next file to
/// need that permission inherits it silently. A budget that only ever caps is
/// a budget that only ever grows.
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

/// The strong half: `ModelFamily` is the table, and it is still here.
///
/// `driver-cuda`'s equivalent asserts `FACTS_ROWS` is ABSENT, because that
/// driver's move is done. This one asserts the opposite — that the type
/// north-star #7 deletes still exists — so that #7 cannot be reported as
/// finished while it is not. When `ModelFamily` goes, this test fails, and
/// the fix is to invert it into cuda's form.
#[test]
fn the_model_type_table_is_still_here_and_this_is_what_number_seven_deletes() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let facts = std::fs::read_to_string(root.join("facts.rs"))
        .expect("facts.rs is the table north-star #7 removes");
    assert!(
        facts.contains("pub enum ModelFamily"),
        "`ModelFamily` is gone from facts.rs. If north-star #7 landed, this \
         test has done its job and should be inverted into \
         `driver-cuda/tests/no_family_names.rs`'s form: assert the table is \
         ABSENT, and drive the budget above to the four non-dispatch files."
    );
}
