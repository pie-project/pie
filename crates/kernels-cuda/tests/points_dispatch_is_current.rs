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

/// The committed file answers EVERY point the plane answers, and answers no
/// point it does not.
///
/// The freshness test above proves the file is what the generator writes;
/// this proves the generator did not drop a claim on the way out. It is the
/// half of "the tables are the source of truth" that IS checkable — a surface
/// missing from `generator::surfaces()` is invisible to both, and the
/// generator's own header says so.
///
/// BOTH TIERS, because both get an arm in the one match: a tier-1 point this
/// plane claims, and every point of its tier-2 surface — which is all of
/// them, an inherent method being its own claim.
#[test]
fn every_claim_has_an_arm() {
    let file = std::fs::read_to_string(at()).expect("the committed dispatch");
    let mut missing: Vec<&str> = Vec::new();
    let mut claimed: Vec<&str> = Vec::new();
    for surface in generator::surfaces() {
        for point in surface.arms() {
            claimed.push(point.name);
            if !file.contains(&format!("        {:?} =>", point.name)) {
                missing.push(point.name);
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

/// Every `CANON` row names a point this plane DOES NOT claim.
///
/// `model_compiler::sweep::resolve` asks the claim tables first and
/// `kernels_cuda::CANON` second, so a row for a point the plane claims is
/// unreachable — a line that reads as a live answer and is never consulted.
/// A row for a point the FLOOR does not declare is worse: nothing will ever
/// ask for it, and a typo in the family half looks exactly like a backlog.
///
/// This replaced `kernels::canon::ROLES`, a closed list of family prefixes
/// the `#[routine]` attribute asserted its `canon` column against at build
/// time. A prefix list could only catch a misspelled FAMILY; the point
/// tables catch a misspelled point, a point that moved, and a row that
/// stopped being a backlog because the plane grew a claim for it.
#[test]
fn every_canon_row_is_an_unclaimed_point() {
    let mut declared: Vec<&str> = Vec::new();
    let mut claimed: Vec<&str> = Vec::new();
    for surface in generator::surfaces() {
        declared.extend(surface.declares().iter().map(|p| p.name));
        claimed.extend(surface.arms().into_iter().map(|p| p.name));
    }
    for (claim, symbol) in kernels_cuda::CANON {
        assert!(
            declared.contains(claim),
            "`CANON` answers `{claim}` with `{symbol}`, and no family declares that point"
        );
        assert!(
            !claimed.contains(claim),
            "`CANON` answers `{claim}`, which this plane CLAIMS -- the claim wins at \
             resolution and this row is never read"
        );
    }
}
