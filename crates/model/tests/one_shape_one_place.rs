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
//!
//! # Where a shape lives
//!
//! It used to be `<family>/forward/facts.rs` and only that. The catalog
//! refactor moved the SEMANTIC shape up to `<family>/spec.rs` — ungated,
//! because a `const` catalog row is written in those words and a row must
//! exist under every aspect — and left `forward/facts.rs` holding a
//! re-export plus whatever per-backend facts are genuinely about the
//! tracer.
//!
//! So the scan reads BOTH, and the count assertion below is what caught
//! that it had to: after the split, a facts-only scan found five structs
//! where it used to find thirty-one, and the twin check that runs on them
//! would have passed by having almost nothing left to compare. A test
//! that quietly stops testing is worse than no test, because the header
//! keeps claiming the gate is standing.

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

/// Every file a family declares a shape in, in the order they are read.
///
/// `spec.rs` and `forward/facts.rs`, unioned. A family may have either or
/// both: the split is a migration in progress, and a scan that knows
/// about one half sees half the shapes.
///
/// Two roots, because a shape can be declared in either half of the
/// crate: a generation directory at the root, or a shared family under
/// `shared/`. The second used to be `src/families/` and `read_dir`
/// returns `Err` for a path that is not there — which this loop skips in
/// silence, so a rename would have quietly halved what the guard sees
/// rather than failing it. Hence the assertion below.
fn declaring_files() -> Vec<PathBuf> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for root in [src.clone(), src.join("shared")] {
        let rd = std::fs::read_dir(&root)
            .unwrap_or_else(|e| panic!("{} is readable: {e}", root.display()));
        for e in rd.flatten() {
            let dir = e.path();
            for f in [dir.join("spec.rs"), dir.join("forward/facts.rs")] {
                if f.is_file() {
                    out.push(f);
                }
            }
        }
    }
    // Named rather than counted, because a count cannot see this. One
    // family lives under `shared/`, so losing that root moves the total
    // by one — and any threshold loose enough to survive a new
    // generation is loose enough to miss it.
    assert!(
        out.iter()
            .any(|p| p.components().any(|c| c.as_os_str() == "llama_like")),
        "the shared half of the crate contributed no shapes; found {out:?}"
    );
    assert!(out.len() >= 10, "found only {} declaring files", out.len());
    out
}

/// The family a declaring file belongs to.
///
/// One directory up from `spec.rs` and two up from `forward/facts.rs`,
/// which is the whole difference between the two.
fn family_of(path: &Path) -> String {
    let dir = if path.file_name().is_some_and(|f| f == "spec.rs") {
        path.parent()
    } else {
        path.parent().and_then(Path::parent)
    };
    dir.and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `(family::Struct, sorted field names)` for every `pub struct` with at
/// least three fields.
///
/// Three because below that the "same shape" question stops being
/// meaningful — a two-field struct of `heads` and `head_dim` is a pair of
/// numbers, not a description.
fn shapes() -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for path in declaring_files() {
        let family = family_of(&path);
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
                && f.chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
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
        "the shape scan found {} structs, so its layout assumption broke — \
         the last time this fired, the shapes had moved to `spec.rs` and \
         the scan was still reading only `forward/facts.rs`",
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
