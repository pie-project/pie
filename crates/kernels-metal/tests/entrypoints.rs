//! The table's product, against the shader tree.
//!
//! This is invariant (1) of `.wiki/kernel-metal-refactor.md` §6:
//!
//! > every entrypoint in `kernels/` resolves to exactly one (row, axis point),
//! > and every (row, axis point) to exactly one entrypoint
//!
//! The comparison runs in two hops, and the split is not incidental. Expanding
//! an `instantiate_*` macro needs a C preprocessor, which a `cargo test` should
//! not shell out to — so `scripts/metal-kernel-audit.py` does that half and
//! writes `entrypoints.generated.txt`, and this test does the half that must be
//! hermetic and fast. The same shape as the RNG contract's generated artifacts
//! and for the same reason.
//!
//! When a shader changes, both hops fail and they fail in a useful order:
//! `--check` names the entrypoint that appeared or vanished, and this test then
//! says whether the table has a row for it.

use std::collections::BTreeSet;
use std::path::PathBuf;

fn artifact() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("entrypoints.generated.txt")
}

fn from_the_shaders() -> BTreeSet<String> {
    std::fs::read_to_string(artifact())
        .expect(
            "entrypoints.generated.txt exists; regenerate it with \
             scripts/metal-kernel-audit.py --write",
        )
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn the_table_names_exactly_what_the_shaders_instantiate() {
    let shaders = from_the_shaders();
    let table: BTreeSet<String> = kernels_metal::entrypoints().into_iter().collect();

    let undeclared: Vec<_> = shaders.difference(&table).collect();
    assert!(
        undeclared.is_empty(),
        "{} entrypoints exist in kernels/ that no row declares. A new \
         instantiation needs a row, or a point on an existing row's axis:\n{:#?}",
        undeclared.len(),
        undeclared
    );

    let phantom: Vec<_> = table.difference(&shaders).collect();
    assert!(
        phantom.is_empty(),
        "{} entrypoints are declared that no shader instantiates. An axis whose \
         product over-generates is the usual cause — see `sdpa_paged_decode`, \
         which lists its tails for exactly this reason:\n{:#?}",
        phantom.len(),
        phantom
    );
}

/// Two rows claiming one entrypoint would make `sig_in` order-dependent, and
/// the set comparison above cannot see it: a duplicate is absorbed by the set.
#[test]
fn no_two_rows_claim_the_same_entrypoint() {
    let mut seen: std::collections::BTreeMap<String, &str> = Default::default();
    for row in kernels_metal::KERNELS {
        for name in row.entrypoints() {
            if let Some(other) = seen.insert(name.clone(), row.name) {
                panic!("`{name}` is claimed by both `{other}` and `{}`", row.name);
            }
        }
    }
}

/// The row count is load-bearing prose in three documents, so it is pinned
/// rather than described. Change it here when a kernel is added, deliberately.
///
/// It has earned its keep once already: 99/480 became 98/479 when the census
/// learned that a wrapped `template` parameter list still declares a template,
/// so `affine_qmm_t_aligned` was a BODY and never a dispatchable name. The set
/// comparison above passed either way — it compares the table to whatever the
/// census says — and this is the assertion that made the correction visible.
#[test]
fn the_table_is_ninety_eight_kernels_over_four_hundred_and_seventy_nine_entrypoints() {
    assert_eq!(kernels_metal::KERNELS.len(), 98);
    assert_eq!(kernels_metal::entrypoints().len(), 479);
}

/// Every entrypoint resolves through the public lookup `model-compiler` uses,
/// at every point of every axis. `sig_in` tries exact matches first and axis
/// matches second, so a base that shadows a sibling's point surfaces here.
#[test]
fn every_entrypoint_resolves_through_sig_in() {
    for name in from_the_shaders() {
        assert!(
            kernels::sig_in(kernels_metal::KERNELS, &name).is_some(),
            "`{name}` does not resolve"
        );
    }
}
