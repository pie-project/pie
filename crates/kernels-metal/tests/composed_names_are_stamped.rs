//! Every name a body can compose is one the shader tree stamps — and the two
//! halves of that sentence are both new.
//!
//! # The hole this closes was written down, not discovered
//!
//! `tests/entrypoints.rs`'s header:
//!
//! > The shader half of the comparison needed a C preprocessor — the axis
//! > product lives in `instantiate_*` macros and nothing else writes it down —
//! > so it arrived as a committed `entrypoints.generated.txt` ... That artifact
//! > is deleted, and with it the only hermetic view a `cargo test` had of what
//! > the shaders instantiate. **Nothing compares them now, in a test or out of
//! > one.**
//! >
//! > What is NOT held is the set itself — a shader instantiating a name no row
//! > declares, or a row whose axes over-generate one no shader stamps, is green
//! > everywhere.
//!
//! `build.rs` expands the macros now, so there is a shader half again and it
//! needs nothing installed. What it produces was checked against the 481 rows
//! of hand-written `ENTRYPOINTS` it replaced before those were deleted: 481
//! against 481, with nothing on either side alone.
//!
//! # And the table half became a composition
//!
//! Nineteen tables of literals, 291 rows, indexed by folding two to four axis
//! values into one integer. The name IS those values, so `qmm_name` builds it
//! and this checks the whole product against the shaders — which is the
//! comparison the header says nothing was making.

use std::collections::BTreeSet;

#[test]
fn every_composable_name_is_stamped() {
    let stamped: BTreeSet<&str> = kernels_metal::STAMPED.iter().map(|(_, n)| *n).collect();
    let missing: Vec<&str> = kernels_metal::quant::composable()
        .into_iter()
        .filter(|name| !stamped.contains(*name))
        .collect();

    assert!(
        missing.is_empty(),
        "{} composable name(s) name a point no `.metal` file stamps. A body \
         reaching one builds a pipeline against a function the library does not \
         hold, which is a nil pipeline on a device:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// The two vacuity guards this shape needs, and it needs both: an empty
/// expansion makes the walk above pass by having nothing to miss, and an empty
/// product makes it pass by having nothing to look up.
#[test]
fn neither_side_of_the_comparison_is_empty() {
    assert_eq!(
        kernels_metal::STAMPED.len(),
        481,
        "the expansion produced a different number of entrypoints than the ten \
         hand-written `ENTRYPOINTS` tables it replaced"
    );
    // 291 ON THE NOSE, which is what the nineteen tables held. A count that
    // moves means an axis moved, and an axis moving without the shader tree
    // moving with it is what the walk above then catches.
    assert_eq!(
        kernels_metal::quant::composable().len(),
        291,
        "the composers no longer produce what the tables they replaced held"
    );
}

/// What the expansion found beyond the affine family, pinned.
///
/// The other direction, and the one the deleted census could not check either:
/// a shader stamping something nothing fires is a translation unit compiled for
/// no reason. It is not an error -- `DECLARED_ELSEWHERE` exists for the points
/// this backend names but does not build, and a body may spell a name this test
/// cannot see -- so the number is pinned rather than required to be zero, and a
/// jump in it is visible.
#[test]
fn what_is_stamped_beyond_the_composed_family_is_a_known_number() {
    let composable: BTreeSet<&str> = kernels_metal::quant::composable().into_iter().collect();
    let beyond: BTreeSet<&str> = kernels_metal::STAMPED
        .iter()
        .map(|(_, n)| *n)
        .filter(|n| !composable.contains(n))
        .collect();
    assert_eq!(
        beyond.len(),
        190,
        "the shader tree stamps {} entrypoints outside the affine family it \
         composes; 481 stamped less the 291 composed is 190",
        beyond.len()
    );
}
