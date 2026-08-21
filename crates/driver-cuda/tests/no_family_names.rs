//! The driver must not know what a model is — measured, not asserted.
//!
//! This greps `src/` for model-family names as a ratchet, not a hard zero,
//! because a few real per-file reasons remain: `layout/memory_planner.rs`
//! sizes a KV pool per architecture (cost arithmetic, not a behaviour
//! branch), one `bind/arms` refusal names the scheme it wants, and
//! `serve/encode.rs` still serves gemma-4's towers (a real family-specific
//! ABI entry pending generalisation). [`budget`] pins a ceiling per file: a
//! file may not gain family names, and one that loses them must ratchet its
//! bound down.
//!
//! The count skips `#[cfg(test)]` blocks (a test naming its own coverage is
//! not dispatch) and compares case- and punctuation-blind (`GptOss` ==
//! `gpt_oss` == `gptoss`). A file absent from the budget must have zero
//! mentions — that is what catches a new dispatch site appearing quietly.
//!
//! `crates/model/tests/one_normalizer.rs` states the companion rule for the
//! other direction: what a driver reads must already be the answer, never
//! the question. This file is the enforcement half, run against the code
//! rather than trusted to be remembered.

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
        // Two `use kernels_cuda::tower::gemma4_* as *_tower` imports, reaching
        // the two walks -- the only two launcher references this file makes,
        // not a dispatch branch. `PieEncodeDesc` is still gemma-shaped;
        // generalising that ABI is what lowers this to one.
        ("serve/encode.rs", 2),
        // THE TOWER BUDGETS ARE GONE, and their absence is the record.
        //
        // `tower/mod.rs` (3), `tower/gemma4_vision.rs` (1),
        // `tower/gemma4_audio.rs` (1), `tower/qwen3_vl.rs` (1) and
        // `tower/qwen3_vl/attn.rs` (1) were budgeted here because a walk is
        // the BODY of one named program rather than a dispatch site. They no
        // longer need the exemption: the gemma-4 walks moved to
        // `kernels_cuda::tower`, where a monolithic kernel belongs, and the
        // qwen3-vl walk was deleted with no caller. A driver that holds no
        // walk needs no permission to name one.
        //
        // AND THE ARM BUDGETS ARE GONE THE SAME WAY, which is the second
        // time this table has recorded a directory rather than a permission.
        //
        // `bind/arms/` was thirteen files and 173 hand-written rows; seven of
        // them were budgeted here -- `attn.rs`, `layout.rs`, `mlp.rs`,
        // `norm.rs` and `ssm.rs` at zero, `rope.rs` at one and `quant.rs` at
        // two. The directory is deleted: the rows are derived from the
        // routine column now and what remains of the routing is
        // `bind/route.rs` and `bind/table.rs`.
        //
        // The two non-zero ones are worth naming on the way out, because
        // neither was a branch and a reader who finds their subject again
        // should know what it was allowed to be. `rope.rs`'s one was a
        // fact's NAME -- `rope::rope_yarn_bf16` unbound on llama-3's
        // `low_freq_factor`/`high_freq_factor`, since a refusal that will not
        // say whose scheme it wants is not worth printing. `quant.rs`'s two
        // were a MEASUREMENT's provenance: the MXFP4 decode rows named
        // gpt-oss because that is the export `compute-sanitizer` was run
        // against, and qwen3.5 because `build_moe_ptrs_aligned` is its
        // statement-side builder. Both arms were `arm: None`.
        //
        // Nothing inherits those permissions. The files that replaced them
        // are budgeted at nothing, which is to say they are not in this
        // table, which is to say a family name appearing in either is
        // exactly this test's quarry.
        // Zero, kept explicit rather than dropped: its three mentions were
        // `ffi::pie_k_gemm_act_x_wt_bf16` call sites, now
        // `use crate::fire::gemm::act_x_wt_bf16;` -- neither spelling
        // flattens to a family name because `::` survives flattening.
        ("fire/lora.rs", 0),
    ]
    .into_iter()
    .collect()
}

fn count(path: &std::path::Path) -> usize {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    // `#[cfg(test)]` opens a block that closes when its braces balance,
    // tracked by depth (not "rest of file") since a file may hold several
    // test modules with real code between them; a test naming the family
    // it covers isn't charged.
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
/// Raw text matching sees `gemma_facts` but misses `GemmaFacts`; dropping
/// case and punctuation before comparing makes `GptOss`, `gpt_oss` and
/// `gptoss` one name.
fn names_a_family(line: &str) -> bool {
    let line = without_trace_symbols(line);
    let flat: String =
        line.chars().filter(|c| *c != '_' && *c != '-').flat_map(char::to_lowercase).collect();
    FAMILIES.iter().any(|f| {
        let f: String = f.chars().filter(|c| *c != '_').collect();
        flat.contains(&f)
    })
}

/// The line with every quoted TRACE SYMBOL removed.
///
/// A quoted `"family::kernel"` literal, e.g.
/// `"ssm::nemotron_mamba_split_bf16"`, is a lookup key naming a kernel, not
/// the driver branching on a model -- what this test looks for is a
/// `GemmaFacts`, a `match arch`, a per-family type table, none of which
/// such a key can be. Only a string that is ENTIRELY
/// `lower_snake::lower_snake` is stripped, so a real model type, variant,
/// or field comparison cannot hide behind the same shape. Keeping the
/// shape this narrow is deliberate: a looser rule here would be the free
/// permission this test's ratchet elsewhere refuses.
fn without_trace_symbols(line: &str) -> String {
    let ok = |s: &str| {
        let Some((ns, name)) = s.split_once("::") else { return false };
        let seg = |x: &str| {
            !x.is_empty()
                && x.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        };
        seg(ns) && seg(name)
    };
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find('"') {
        let (before, after_open) = rest.split_at(open);
        out.push_str(before);
        let body = &after_open[1..];
        let Some(close) = body.find('"') else {
            out.push_str(after_open);
            return out;
        };
        if !ok(&body[..close]) {
            out.push('"');
            out.push_str(&body[..close]);
        }
        out.push('"');
        rest = &body[close + 1..];
    }
    out.push_str(rest);
    out
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
///
/// Two failure modes, both fatal: a budgeted file whose count rose past its
/// ceiling (`over`), and a file naming a family that was never budgeted at
/// all (`unlisted`) -- the latter is how a brand-new dispatch site gets
/// caught the moment it appears.
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
    // `==` and not `<=`: a ceiling only ratchets down. Leaving a stale cap
    // in place after mentions leave would hand out free mentions silently
    // -- the "guard that must be remembered" failure this test exists to
    // avoid. Deleting a family name costs a two-line diff; the second line
    // is the record that it happened, which is a cost this test intends.
    for (rel, &cap) in &budget {
        if cap > 0
            && !files
                .iter()
                .any(|f| f.strip_prefix(&root).is_ok_and(|r| r.to_string_lossy() == *rel))
        {
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
/// A checkpoint is matched to a `model::catalog` row by its TENSORS; a
/// driver receives a `Deployment`, with no family name to branch on. This
/// guards against `FACTS_ROWS`/`HF_ROWS`/`MLX_ROWS` (a table keyed on a
/// `config.json` string) reappearing anywhere in the driver. Unlike
/// [`budget`], this check tolerates no count at all: a `model_type` table
/// is not something the driver may have a little of.
#[test]
fn no_model_type_table_remains_in_the_driver() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
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
/// Without this, a file can be renamed, deleted, or cleaned up and leave
/// its ceiling behind as an unused permission the next file to need it
/// would silently inherit.
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
