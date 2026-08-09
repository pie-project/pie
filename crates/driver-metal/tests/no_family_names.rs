//! The driver must not know what a model is — measured, not asserted.
//!
//! `crates/model/tests/one_normalizer.rs` states the rule: *"what a driver
//! reads is the answer, never the question"*, and three documents cited it as
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
//! # THE INDICTMENT IS SPENT
//!
//! This file used to open by saying it "passes today, at numbers that are an
//! indictment": 189 lines across 8 files, four of which held real dispatch.
//! Two of those four are DELETED, and they were the two that mattered:
//!
//! | file | was | what it was |
//! |---|---|---|
//! | `facts.rs` | 99 | the `model_type` table — 81 fields in four family blocks, filled by asking a `serde_json::Value` what kind of model it is |
//! | `batch/geometry_facts.rs` | 25 | the projection out of those four blocks into `DecodeGeometry`'s kernel arguments |
//!
//! What replaced them is not a smaller table. A checkpoint is matched to a
//! `model::catalog` row BY ITS TENSORS, the row projects a
//! `model::deployment::Deployment` — a value with no family name in it and
//! therefore nothing to branch on — and `batch::geometry_from_deployment` is
//! arithmetic over that value. There is no `model_type` string anywhere on
//! the path, which is why the strong half of this file below has been
//! INVERTED into `driver-cuda`'s form: it watched for the table's continued
//! existence, and it watches for its RETURN.
//!
//! The prediction that file made about the move is worth keeping, because it
//! was right and it is what made the move survivable: *"#7 is a change with
//! no compile error attached to its failure mode — `lowering/consts.rs` binds
//! `hidden`, `intermediate`, `moe_intermediate`, `n_experts`, `vocab`, `eps`,
//! `gdn_conv_dim` and `gdn_v_total` off `DecodeGeometry`, and none of those
//! exist on `Deployment`. A #7 done as 'call `deployment_from` and delete
//! `facts.rs`' produces a GPU FAULT, NOT A COMPILER MESSAGE."* That is
//! exactly why `DecodeGeometry` survived and only its SOURCE changed, and why
//! every number it carries that a `Deployment` cannot state is a REFUSAL in
//! `batch/geometry.rs` rather than a default.
//!
//! # Why a ratchet rather than zero
//!
//! Not every mention is dispatch, and collapsing them into one number would
//! hide which is which. The five files left below name a family for a reason
//! that survived the move entirely — a refusal's message, a kernel's own
//! name, a deprecation note, a shared module's path — and pinning per FILE is
//! what keeps those separable from a new dispatch site.
//!
//! **A file not in the table may have none at all.** That is the half that
//! matters: it is what stops a new dispatch site appearing somewhere quiet
//! while the loud ones are being cleaned up.

use std::collections::BTreeMap;

/// The abbreviations a family name hides behind once a field is named for it.
///
/// `FAMILIES` spells names out, so it counted `model_type == "gpt_oss"` and
/// every descriptor key with `gemma` in it -- and missed `go_hidden_size`,
/// `ll_rms_norm_eps`, `g4_num_experts` and `q35_head_dim` entirely. That is
/// backwards: the spelled-out mentions are mostly key strings this crate
/// does not choose and test data, while the prefixes ARE north-star #7. A
/// deletion of 27 such fields moved this guard's number by zero, which is how
/// the omission was found.
///
/// Matched at the start of an identifier, not anywhere in the line: `ll_` is
/// a substring of `full_attn_interval` and `go_` of `dialog_state`.
const PREFIXES: &[&str] = &["go_", "ll_", "g4_", "q35_"];

/// Whether a line names a family, spelled out or abbreviated.
fn names_a_family(line: &str) -> bool {
    // The same reduction the deleted `model/text.rs` applied to an
    // architecture string, for the same reason: a publisher's capitalisation
    // and punctuation are not the identity. Matching raw text saw
    // `gemma_facts` and missed `GemmaFacts`, which is where family names
    // actually appear in Rust.
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

/// The family names a driver must not be branching on.
///
/// The same list `driver-cuda` uses, so the two backends are held to one
/// standard rather than to whichever names each happened to notice.
const FAMILIES: &[&str] = &[
    "gemma", "qwen", "llama", "deepseek", "kimi", "nemotron", "glm", "gpt_oss",
];

/// Non-comment lines naming a family, per file, as of this test's writing.
///
/// These are CEILINGS. Lowering one is the point; raising one is the thing
/// this test exists to catch.
fn budget() -> BTreeMap<&'static str, usize> {
    [
        // ── The one that is gone ─────────────────────────────────────
        //
        // `model/text.rs` was budgeted 27, for "the list of architectures a
        // Metal text is written for, plus the `LlamaLikeFacts` synthesis that
        // feeds it", and its entry said in as many words what would retire
        // it: *"it stays budgeted until a catalog row can be asked for a
        // METAL trace the way it can be asked for a CUDA one. `Variant::trace`
        // has no backend parameter today, which is the whole reason this file
        // cannot go to zero."*
        //
        // It has one now — `catalog::Deployed::backend` — and the file is
        // DELETED rather than reduced. The eleven architecture strings, the
        // `canonical()` that folded spellings onto them and the
        // `facts_from_with()` that rebuilt twenty-nine model facts from nine
        // tensor probes all went with it, because the row states every one of
        // them. What replaced it is `model/binding.rs`, which is not in this
        // table at all: it spells no family name in code, because the six
        // things it holds are an affine point, an expert bank's format and
        // three kernel capabilities, and not one of those is a model.
        //
        // `serve/state.rs` was budgeted 2, for two `use` paths into
        // `model::shared::llama_like` — the crate's cross-generation
        // sharing module — which the shell needed because it held that
        // family's facts struct. It holds a `&dyn catalog::Variant` and a
        // `catalog::MetalBinding` now, neither of which is a family's type,
        // so both imports went and the entry with them.
        //
        // ── The six that are not ─────────────────────────────────────
        //
        // A refusal naming the model whose shape caused it: gemma-4 pages
        // two KV geometries, and an operator reading "this pool's layers
        // have two page sizes" needs to know which model does that.
        ("pools/kv.rs", 1),
        // The same, for a control-path refusal that names where the real
        // check lives.
        ("serve/control.rs", 1),
        // `lowering/resolve.rs` left the same way. Its two were a
        // `#[deprecated]` note pointing at what replaced it, on a shim whose
        // whole map has since moved to `model::shared::weight_names::Names`
        // — beside the HuggingFace map that module already owned, and for
        // the reason its doc gives: "the map between them belongs here and
        // not in a driver." What is left in this crate is the resolver, which
        // spells nothing.
        // `model/rope.rs` and `loader/plan.rs` were here for "a TEST's name
        // and a test fixture's `model_type` string", which the count no
        // longer sees. Their entries are gone rather than zeroed: a file
        // whose only mentions were its own coverage was never in debt.
        // `serve/state.rs` left the same way — see the note above.
        //
        // `batch/geometry.rs` held `DecodeGeometry::gemma`, a bool named for
        // the family whose norm convention it selected, and then held the
        // ladder that filled it. Both are gone, and what is left is three
        // lines that are not a dispatch at all: a `RopeScaling::Piecewise` match
        // arm — an enum variant `model::deployment` states, so reading it is
        // reading the ROW rather than deciding what the model is — and two
        // lines of the refusal that fires when a checkpoint asks for a YaRN
        // ladder this driver does not derive. That refusal names llama-3's
        // `rope_type` because an operator who reads "this driver derives only
        // llama-3's piecewise table" knows what to do next, and one that
        // reads "unsupported rope scaling" does not.
        //
        // Budgeted at what those are rather than left unlisted. The comment
        // here used to claim the file spent ZERO — written when `PREFIXES`
        // did not exist and the count could not see a match arm — and an
        // unlisted file that scores is a red build with no argument attached
        // to it, which is the failure mode this table's `unlisted` assertion
        // exists to force into the open.
        //
        // THREE BECAME TWO when `RopeScaling::Llama3` was renamed to
        // `RopeScaling::Piecewise` in `model::deployment`: the match arm was
        // a family name this file had no way to spend differently, because
        // the vocabulary it matched on carried the lineage. The two left were
        // the refusal's own sentence.
        //
        // TWO BECAME NONE, and the entry left with them. Those two lines said
        // "this driver derives only llama-3's piecewise table" to an operator
        // whose checkpoint wanted a YaRN ladder; then the driver learned to
        // derive the ladder, and a refusal nobody can reach has no sentence
        // to spend. The entry is deleted rather than zeroed, for the reason
        // the entries above it were: a budget of zero is not a budget, and an
        // unlisted file that scores fails `unlisted` with the argument
        // attached, which is stricter than a ceiling of zero and says more.
        // The kernel ABI's family-prefixed kind names. Absent from this
        // table until `PREFIXES` was added, because every one of its
        // mentions is an abbreviation: it names no family in full. 62 of
        // these became 45 when six kinds documented in the same words as a
        // second family's twin were merged into one. The rest name real
        // per-architecture kernels -- gemma4's PLE, gpt-oss's attention
        // sink -- which is a different claim from a duplicated read.
        ("lowering/abi.rs", 45),
    ]
    .into_iter()
    .collect()
}

/// Lines of DISPATCH that name a family: comments and tests excluded.
///
/// Comments are provenance -- "ports gemma_4/forward" -- not dispatch. Tests
/// are the opposite of the thing being counted: `assert!(serves(row))` is not
/// a driver learning what a model is, it is the guard on the code that must
/// know. Counting them made writing a test cost budget, which taxes exactly
/// the work #7 needs; the deleted `model/text.rs` was 48 with 37 of them
/// inside `#[cfg(test)]`, so three quarters of its debt was its own coverage.
fn count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    // `#[cfg(test)]` opens a module; it closes when its braces balance.
    // Tracking depth rather than assuming the file ends there matters:
    // `model/text.rs` had two test modules with real code between them, and
    // `batch/geometry.rs` still does.
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

/// The strong half, INVERTED: no `model_type` table remains here.
///
/// This test used to assert the opposite. It read `facts.rs` and required
/// `pub enum ModelFamily` to still be in it, so that north-star #7 could not
/// be reported as finished while the table it deletes was still standing —
/// and it said, in its own doc, that when the type went the fix was to invert
/// it into `driver-cuda`'s form. This is that inversion.
///
/// The claim it now makes is the strong one and it got stronger on the way.
/// The table did not MOVE — `driver-cuda`'s equivalent watched its own
/// `FACTS_ROWS` leave the driver for `model::deployment_cuda`, which was the
/// right first step and still left a table keyed on a `config.json` string
/// one crate over. There is no such table anywhere now: a checkpoint is
/// matched to a `model::catalog` row BY ITS TENSORS, and what a driver
/// receives is a `Deployment`, a value with no family name in it and
/// therefore nothing to branch on.
///
/// So this watches for the table's RETURN rather than for its escape, which
/// is why the message no longer names a place it could legitimately live.
/// `arch_stem` is in the banned list beside the three table names because it
/// was the FUNCTION that made a dispatch key: it lowercased
/// `Qwen3MoeForCausalLM` and dropped the tail, and the two spellings that
/// produced are how a load gate and the seam came to answer differently about
/// the same checkpoint.
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
