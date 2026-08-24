//! `src/points_dispatch.rs` is CURRENT against the tables that write it.
//!
//! The `model-loader/tests/golden_plans.rs` idiom, and the one
//! `model-dsl-legacy`'s retired `wrappers_are_current.rs` used for the same
//! kind of file: regenerate into a string, diff it against what is
//! committed, refuse a stale copy. `UPDATE_POINTS_DISPATCH=1` rewrites it —
//! and, like every golden rewrite, the SAME run still tests the code
//! compiled from the old file, so run it once more to prove the new one.
//!
//! CHECKED IN RATHER THAN BUILT INTO `OUT_DIR`, against this crate's own
//! `build.rs` note, and for one reason the note's argument does not cover:
//! the dispatch's arms are read. A carried-header list is a list; this is
//! the plane's answer to every point it claims, and a reviewer who cannot
//! see the diff cannot see a claim quietly changing which slot it reads.
//! The staleness the note warns about is what the test below is.

#[path = "points_dispatch_is_current/generator.rs"]
mod generator;

use std::path::PathBuf;

fn at() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/points_dispatch.rs")
}

#[test]
fn points_dispatch_is_current() {
    let want = generator::generate();
    let at = at();
    let have = std::fs::read_to_string(&at).unwrap_or_default();
    if want == have {
        return;
    }
    if std::env::var_os("UPDATE_POINTS_DISPATCH").is_some() {
        std::fs::write(&at, &want).unwrap_or_else(|e| panic!("rewriting {}: {e}", at.display()));
        return;
    }
    // Point at the first diverging line rather than dumping two files.
    let line = want
        .lines()
        .zip(have.lines())
        .position(|(w, h)| w != h)
        .map_or_else(|| want.lines().count().min(have.lines().count()) + 1, |i| i + 1);
    panic!(
        "src/points_dispatch.rs is STALE against `kernels::points` × this \
         plane's `*_CLAIMS` (first divergence at line {line}). The dispatch is \
         generated, never edited: regenerate with \
         `UPDATE_POINTS_DISPATCH=1 cargo test -p kernels-cuda --test \
         points_dispatch_is_current` and review the diff."
    );
}

/// The committed file answers EVERY point the plane claims, and answers no
/// point it does not.
///
/// The freshness test above proves the file is what the generator writes;
/// this proves the generator did not drop a claim on the way out. It is the
/// half of "the tables are the source of truth" that IS checkable — a family
/// missing from `generator::families()` is invisible to both, and the
/// generator's own header says so.
#[test]
fn every_claim_has_an_arm() {
    let file = std::fs::read_to_string(at()).expect("the committed dispatch");
    let mut missing: Vec<&str> = Vec::new();
    let mut claimed: Vec<&str> = Vec::new();
    for f in generator::families() {
        for point in f.claims {
            claimed.push(point);
            if !file.contains(&format!("        {point:?} =>")) {
                missing.push(point);
            }
        }
    }
    assert!(missing.is_empty(), "claimed and not dispatched: {missing:?}");

    // And nothing else: an arm for an UNCLAIMED point would answer a
    // measured backlog row with a call that refuses one layer deeper.
    let arms: Vec<&str> = file
        .lines()
        .filter_map(|l| l.strip_prefix("        \""))
        .filter_map(|l| l.split_once("\" =>"))
        .map(|(point, _)| point)
        .collect();
    let extra: Vec<&&str> = arms.iter().filter(|a| !claimed.contains(a)).collect();
    assert!(extra.is_empty(), "dispatched and not claimed: {extra:?}");
    assert_eq!(arms.len(), claimed.len(), "one arm per claim, and no arm twice");
}
