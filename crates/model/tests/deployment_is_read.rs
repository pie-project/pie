//! Every `Deployment` field is read by somebody, or says who will.
//!
//! A `Deployment` is what a driver reads a checkpoint's shape off. It
//! is the one place a family's measurements become a driver's
//! instructions, so a field here is a promise that something acts on
//! it. A field nothing reads is not inert: it reads, to whoever adds
//! the next family, as an axis already handled.
//!
//! `k_eq_v` is why this file exists, and it is no longer a field.
//! gemma-4-31b ships no `v_proj` — its full layers read V out of the K
//! projection — and the row states that, and the Metal text acts on it
//! through `LlamaLikeMetalFacts::v_from_k`. The CUDA text never grew an
//! arm: `Gemma4LayerW` declares a `v_proj` and matmuls it with nothing
//! to branch on, and `project::trace` is not even handed the flag. So
//! the field was read by exactly one thing — a `DecodeGeometry` copy
//! that nothing downstream read — and when that copy went, the
//! statement had no reader at all.
//!
//! It was then excused HERE, and the excuse promised the field "becomes
//! read the day the CUDA text grows the branch". That day cannot come.
//! `Gemma4LayerW` lives in `model`, not in a driver; the branch that
//! needs the flag is handed `&self.shape`, so growing it reads the ROW,
//! exactly as the Metal projection already does through
//! `Gemma4::row()`. No driver was ever going to read this: `k_eq_v`
//! does not move the KV geometry — both halves are still cached — it
//! decides which tensors the TEXT binds, and the text is written in
//! this crate.
//!
//! So it was a second statement of one measurement, and the live one is
//! the row's. Thirteen projections wrote it and nothing anywhere read
//! it. The refusal in `Variant::trace`'s CUDA arm is what this build
//! can honestly do about the missing branch, and it is made of
//! `Gemma4::k_eq_v`. This test is so the NEXT such field is a failure
//! rather than an archaeology.

use std::collections::BTreeSet;

/// Fields no consumer crate reads, and who is expected to.
///
/// An entry is a claim about unwritten work, not permission. It names
/// the pass and what happens meanwhile.
///
/// **Empty, and that is the point.** Its only entry was `k_eq_v`, and
/// the entry is what kept the field: the excuse named a future reader
/// that the crate's own layering ruled out, so the line could never
/// have expired on its own. An entry here must therefore name a reader
/// that could exist in a CONSUMER crate — if the pass that will read
/// the fact lives in `model`, the fact belongs on the row, and this
/// struct is a copy of it.
const UNREAD_BY_CONSUMERS: &[(&str, &str)] = &[];

fn crate_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// The braces-matched body of an item starting at `from`.
fn block(s: &str, from: usize) -> &str {
    let o = s[from..].find('{').expect("a block") + from;
    let (mut d, mut e) = (0usize, o);
    for (i, c) in s[o..].char_indices() {
        match c {
            '{' => d += 1,
            '}' => {
                d -= 1;
                if d == 0 {
                    e = o + i;
                    break;
                }
            }
            _ => {}
        }
    }
    &s[o..=e]
}

fn declared() -> BTreeSet<String> {
    let s = std::fs::read_to_string(crate_dir().join("src/deployment.rs")).expect("deployment.rs");
    let b = block(&s, s.find("pub struct Deployment").expect("the struct"));
    b.lines()
        .filter_map(|l| l.trim().strip_prefix("pub "))
        .filter_map(|l| l.split(':').next())
        .map(str::to_string)
        .collect()
}

/// Every `.field` read in a crate that CONSUMES a `Deployment`.
///
/// `crates/model` itself is excluded on purpose. A projection writing
/// `k_eq_v` into the struct is not a reader of it, and a family reading
/// its own facts back out proves nothing about whether any driver ever
/// asked.
fn read_by_consumers() -> BTreeSet<String> {
    fn walk(d: &std::path::Path, out: &mut String) {
        for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
            }
        }
    }
    let workspace = crate_dir().join("..");
    let mut all = String::new();
    let mut crates = 0usize;
    for e in std::fs::read_dir(&workspace).expect("crates/").flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if name == "model" || !p.join("src").is_dir() {
            continue;
        }
        crates += 1;
        walk(&p.join("src"), &mut all);
    }
    assert!(
        crates > 5,
        "found only {crates} sibling crates — the scan broke"
    );

    let mut seen = BTreeSet::new();
    for (i, _) in all.match_indices('.') {
        let n: String = all[i + 1..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !n.is_empty() {
            seen.insert(n);
        }
    }
    seen
}

#[test]
fn every_deployment_field_reaches_a_driver() {
    let names = declared();
    assert!(
        names.len() > 15,
        "found only {} fields — the scan broke",
        names.len()
    );

    let read = read_by_consumers();
    let excused: BTreeSet<&str> = UNREAD_BY_CONSUMERS.iter().map(|(n, _)| *n).collect();

    let dead: Vec<&String> = names
        .iter()
        .filter(|n| !read.contains(*n) && !excused.contains(n.as_str()))
        .collect();

    assert!(
        dead.is_empty(),
        "no crate outside `model` reads these:\n  {dead:?}\n\nA \
         `Deployment` field is an instruction to a driver. One nobody \
         reads is an axis that LOOKS handled — which is how gemma-4's \
         `k_eq_v` came to be stated by thirteen projections, acted on by \
         the Metal text from a different road, and traced against a \
         `v_proj` on CUDA. That one ended in DELETION: check first \
         whether the fact already travels on the row, because then this \
         copy is the second answer and not the missing reader. Otherwise \
         wire the reader, or name it in `UNREAD_BY_CONSUMERS` with the \
         pass that will read it — in a consumer crate.",
    );
}

/// The excuse does not outlive the field it excuses.
#[test]
fn nothing_is_excused_that_the_deployment_no_longer_declares() {
    let names = declared();
    let stale: Vec<&str> = UNREAD_BY_CONSUMERS
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !names.contains(*n))
        .collect();
    assert!(stale.is_empty(), "no such field any more: {stale:?}");
}

/// And an excuse expires when the field starts being read.
///
/// The same shape as `facts_are_read`'s stale-entry guard, and for the
/// same reason: an entry that says "unwritten" about written work sends
/// the next reader looking for a gap that closed.
#[test]
fn nothing_is_excused_that_a_driver_already_reads() {
    let read = read_by_consumers();
    let live: Vec<&str> = UNREAD_BY_CONSUMERS
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| read.contains(*n))
        .collect();
    assert!(
        live.is_empty(),
        "a driver reads these now, so their entries are wrong: {live:?}",
    );
}
