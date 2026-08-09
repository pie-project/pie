//! NO TWO FAMILIES MAY DESCRIBE THE SAME THING TWICE.
//!
//! A facts struct is a family's answer to "what varies about this
//! deployment". When two families give field-identical answers, they are
//! describing one thing in two places — and two places can disagree.
//! They did: three families wrote `is_full_attn` and disagreed at
//! `interval == 0`; gemma-2 declared `tied_embeddings` and never read it,
//! because the fork that reads it was written in four OTHER family texts.
//!
//! So this is `cuda.md` §5.C2's method as a standing gate. A sweep
//! collapsed four duplicated shapes into `model_compiler::facts`
//! (`MlaFacts`, `MoeFacts`, `GqaFacts`, and the schedule predicates), and
//! this refuses to let a fifth appear quietly.
//!
//! It compares FIELD NAMES, not types or docs. Two structs with the same
//! field names are the same fact whatever they are called, and calling
//! them different things is exactly how the copies survived.

#![cfg(feature = "forward")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Pairs that are allowed to look alike, with why.
///
/// EMPTY, and that is the interesting part: the sweep left nothing
/// behind. A line joins here only with a reason that survives the
/// question "why can these two never need to change together?" — which
/// is a high bar, because the answer for every pair the sweep found was
/// "they cannot".
const ALLOWED_TWINS: &[(&str, &str, &str)] = &[];

fn families_dir() -> Vec<PathBuf> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for root in [src.clone(), src.join("families")] {
        let Ok(rd) = std::fs::read_dir(&root) else { continue };
        for e in rd.flatten() {
            let f = e.path().join("forward/facts.rs");
            if f.is_file() {
                out.push(f);
            }
        }
    }
    out
}

/// `(family::Struct, sorted field names)` for every `pub struct` with at
/// least three fields.
///
/// Three because below that the "same shape" question stops being
/// meaningful — a two-field struct of `heads` and `head_dim` is a pair of
/// numbers, not a description.
fn shapes() -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for path in families_dir() {
        let family = path
            .parent()
            .and_then(Path::parent)
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let src = std::fs::read_to_string(&path).expect("a family's facts");
        let mut name: Option<String> = None;
        let mut fields: Vec<String> = Vec::new();
        for line in src.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("pub struct ")
                && t.ends_with('{')
            {
                name = Some(rest.trim_end_matches(" {").to_string());
                fields.clear();
                continue;
            }
            if name.is_some() && t == "}" {
                let n = name.take().expect("just checked");
                if fields.len() >= 3 {
                    fields.sort();
                    out.push((format!("{family}::{n}"), std::mem::take(&mut fields)));
                }
                fields.clear();
                continue;
            }
            if name.is_some()
                && let Some(rest) = t.strip_prefix("pub ")
                && let Some((f, _)) = rest.split_once(':')
                && f.chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
                && !f.is_empty()
            {
                fields.push(f.to_string());
            }
        }
    }
    out
}

#[test]
fn no_two_families_describe_the_same_shape() {
    let shapes = shapes();
    assert!(
        shapes.len() > 15,
        "the facts scan found {} structs, so its shape assumption broke",
        shapes.len()
    );

    let mut by_fields: BTreeMap<Vec<String>, Vec<String>> = BTreeMap::new();
    for (who, fields) in shapes {
        by_fields.entry(fields).or_default().push(who);
    }

    let allowed = |a: &str, b: &str| {
        ALLOWED_TWINS
            .iter()
            .any(|(x, y, _)| (*x == a && *y == b) || (*x == b && *y == a))
    };

    let mut twins: Vec<String> = Vec::new();
    for (fields, who) in &by_fields {
        if who.len() < 2 {
            continue;
        }
        // Every pair in the group has to be excused, not just one.
        let excused = who
            .iter()
            .enumerate()
            .all(|(i, a)| who.iter().skip(i + 1).all(|b| allowed(a, b)));
        if !excused {
            twins.push(format!("{who:?} all describe {fields:?}"));
        }
    }

    assert!(
        twins.is_empty(),
        "two families describe the same shape, which is two places that \
         can disagree about one thing.\n\
         Put it in `model_compiler::facts` and leave a `pub type` alias \
         behind, the way `MlaFacts`, `MoeFacts` and `GqaFacts` were done \
         — every call site keeps working and there is one definition.\n\
         If they genuinely must stay apart, add them to `ALLOWED_TWINS` \
         with a reason that answers \"why can these never need to change \
         together?\"\n\n{}",
        twins.join("\n")
    );
}
