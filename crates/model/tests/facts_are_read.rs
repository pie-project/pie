//! EVERY FACT A FAMILY DECLARES, READ BY THAT FAMILY'S TEXT.
//!
//! A fact is a promise that something about a deployment changes what the
//! model does. A fact nobody reads is a promise nobody keeps, and it
//! fails in the worst available way: the declaration looks complete, the
//! plan traces, and the answer is wrong on the axis the fact was supposed
//! to control.
//!
//! gemma-2 is why this file exists. It carried `tied_embeddings: true`,
//! correctly — the checkpoint ships no `lm_head.weight` — and its text
//! named `"lm_head"` unconditionally, because the
//! `if tied { "embed" } else { "lm_head" }` fork was written in four
//! other family texts and not in that one. The trace asked the binder for
//! a tensor that does not exist. Nothing caught it: the family is in
//! `NOT_YET_OPENABLE`, so no fire reaches it, and a golden pins whatever
//! the text says rather than what it should say.
//!
//! So this is the same closed-set discipline `UNARMED` and
//! `NOT_YET_OPENABLE` use, applied to facts: the unread set is STATED,
//! and a fact that joins it fails here rather than on a checkpoint.
//!
//! The scan is source-level, like `executor_bind`'s arm scan, and for the
//! same reason — the question is about what the code SAYS, and there is
//! no runtime handle on "did this field influence the trace".

#![cfg(feature = "forward")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Facts declared and not read, by family, with why.
///
/// Every line here belongs to a family in `NOT_YET_OPENABLE` whose TEXT
/// IS UNFINISHED — not to a family that traces a complete plan and
/// forgets one axis, which is what gemma-2 did. The distinction is the
/// whole point of listing them rather than deleting the fields: these
/// facts were read off a real config and are what the text will need
/// when someone finishes it. Deleting them would throw away the reading.
///
/// kimi-k2's `rope_yarn_original` shows the shape: its text has no rope
/// statement at all — `mla_prepare` hands back `q_pe` and `k_pe` and
/// nothing rotates either, with `k_pe` dropped into `_`. The fact is not
/// forgotten, the pass is unwritten.
///
/// A line LEAVES when the text starts reading the fact. A line JOINS only
/// with a commit that says which unfinished pass it belongs to.
const DECLARED_BUT_UNREAD: &[(&str, &[&str])] = &[
    // The DSA indexer, hyper-connections and the routed MLP's clamp.
    (
        "deepseek_v4",
        &["hash_routed", "o_groups", "sliding_window", "swiglu_limit_milli"],
    ),
    // MLA's aligned MoE block and the indexer's top-k.
    ("glm5", &["aligned_block", "index_topk"]),
    // The MXFP4 decode leg's route ceiling.
    ("gpt_oss", &["mxfp4_decode_max_routes"]),
    // No rope statement in the MLA text yet; see the header above.
    ("kimi_k2", &["rope_yarn_original"]),
    // KDA's gate floor.
    ("kimi_k3", &["gate_lower_bound_milli"]),
];

fn crate_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every family with a `forward/facts.rs`, and where its sources live.
fn families() -> BTreeMap<String, PathBuf> {
    let mut out = BTreeMap::new();
    let mut roots = vec![crate_src(), crate_src().join("families")];
    roots.retain(|r| r.is_dir());
    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else { continue };
        for e in entries.flatten() {
            let dir = e.path();
            if dir.join("forward/facts.rs").is_file() {
                let name = dir.file_name().unwrap().to_string_lossy().into_owned();
                out.insert(name, dir);
            }
        }
    }
    out
}

/// Every `pub` field of every `pub struct` in one facts file.
fn declared_fields(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut in_struct = false;
    for line in src.lines() {
        let t = line.trim();
        if t.starts_with("pub struct ") && t.ends_with('{') {
            in_struct = true;
            continue;
        }
        if in_struct && t == "}" {
            in_struct = false;
            continue;
        }
        if !in_struct {
            continue;
        }
        // `pub name: Type,` — the field, not a doc comment or attribute.
        if let Some(rest) = t.strip_prefix("pub ")
            && let Some((name, _)) = rest.split_once(':')
            && name.chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
            && !name.is_empty()
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// Every `.field` mention across a family's own sources.
fn read_fields(dir: &Path) -> BTreeSet<String> {
    let mut text = String::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                text.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
            }
        }
    }
    let mut out = BTreeSet::new();
    let bytes: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == '.' && i + 1 < bytes.len() && (bytes[i + 1].is_ascii_lowercase()) {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == '_') {
                j += 1;
            }
            out.insert(bytes[start..j].iter().collect::<String>());
            i = j;
            continue;
        }
        i += 1;
    }
    out
}

#[test]
fn every_declared_fact_is_read_by_its_family() {
    let stated: BTreeMap<&str, BTreeSet<&str>> = DECLARED_BUT_UNREAD
        .iter()
        .map(|(fam, fields)| (*fam, fields.iter().copied().collect()))
        .collect();

    let families = families();
    assert!(
        families.len() > 8,
        "the family scan found {} families, so its shape assumption broke",
        families.len()
    );

    let mut found: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (fam, dir) in &families {
        let facts = std::fs::read_to_string(dir.join("forward/facts.rs"))
            .expect("the family's facts");
        let unread: BTreeSet<String> =
            declared_fields(&facts).difference(&read_fields(dir)).cloned().collect();
        if !unread.is_empty() {
            found.insert(fam.clone(), unread);
        }
    }

    // A family whose stated list no longer matches is the interesting
    // case in both directions: a fact that LEFT means the text started
    // reading it, and a fact that JOINED is a promise nobody keeps.
    let found_view: BTreeMap<&str, BTreeSet<&str>> = found
        .iter()
        .map(|(k, v)| (k.as_str(), v.iter().map(String::as_str).collect()))
        .collect();
    assert_eq!(
        found_view, stated,
        "the declared-but-unread set moved.\n\
         A fact that LEFT means its text now reads it — delete the line.\n\
         A fact that JOINED is declared and never read, which is how \
         gemma-2 came to ask its binder for a `lm_head.weight` no gemma-2 \
         checkpoint ships. Either read it or say which unfinished pass it \
         belongs to."
    );
}
