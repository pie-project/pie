//! A row that ships a sandwich norm states a GELU gate.
//!
//! This check used to run inside `driver-metal`'s load path, and the
//! reason it was written there is worth keeping: serving a gemma on a
//! SiLU is a 2%-at-the-origin error that diverges from there, produces
//! finite plausible tokens and never faults. Measured on gemma-4-31b,
//! whose three mid-rank probes read 0.95, 0.93 and 0.93 of MLX's logits
//! and whose argmax and top five were RIGHT throughout. A defect that
//! survives every cheap check is one worth keeping a second reader for,
//! and a row added later with its gate left at the default is exactly
//! how it would come back.
//!
//! What was wrong was WHERE it read. The driver asked the staged
//! tensors whether `layers.0.pre_feedforward_layernorm.weight` had
//! arrived, and compared that to the row's gate. Three of
//! `driver-metal`'s own guards forbid that, for the reason the catalog
//! refactor exists: a driver that re-derives a model fact from a tensor
//! name is a second answer to a question the row already answers, and
//! the norm variant came to read `(1 + w)` for gemma-4 precisely
//! because a probe like this one asked whether it shipped the norm.
//!
//! So the pair moves to where both halves are already stated. The
//! manifest declares `pre_feedforward_layernorm` for the rows that ship
//! it; the deployment states `mlp_gate`. Comparing them needs no
//! checkpoint at all, which makes this strictly stronger than what it
//! replaces:
//!
//!   - it covers EVERY row, not the ones a Metal load happens to reach;
//!   - it fails in CI rather than at a user's load;
//!   - a row is wrong when it is WRITTEN, not when it is served.
//!
//! The driver keeps no version of this. Reconciling two answers was
//! only ever necessary because there were two.

use model::catalog::{Deployed, catalog};
use model::deployment::MlpGate;

/// The tensor whose presence means the MLP is wrapped, not just preceded.
const SANDWICH: &str = "pre_feedforward_layernorm";

/// Rows whose manifest declares the sandwich norm, by id.
fn sandwiched() -> Vec<&'static str> {
    catalog()
        .iter()
        .filter(|v| {
            v.manifest()
                .tensors
                .iter()
                .any(|t| t.name.contains(SANDWICH))
        })
        .map(|v| v.id())
        .collect()
}

#[test]
fn a_sandwich_norm_and_a_silu_gate_cannot_both_be_stated() {
    let mut wrong: Vec<(&str, MlpGate)> = Vec::new();

    for v in catalog() {
        let sandwich = v
            .manifest()
            .tensors
            .iter()
            .any(|t| t.name.contains(SANDWICH));
        if !sandwich {
            continue;
        }
        // A row this build cannot serve still states its gate; the
        // refusal is about kernels, not about the checkpoint.
        let Ok(d) = v.deployment(Deployed::single()) else {
            continue;
        };
        if !matches!(d.mlp_gate, MlpGate::GeluTanh) {
            wrong.push((v.id(), d.mlp_gate));
        }
    }

    assert!(
        wrong.is_empty(),
        "these rows declare `{SANDWICH}` in their manifest and state a \
         non-GELU gate: {wrong:?}. A sandwich norm is gemma's, and gemma's \
         MLP gate is a GELU -- a gemma served on a SiLU answers plausibly \
         and wrongly, so this is a row to fix rather than a driver to teach"
    );
}

/// The guard is not vacuous: some row really does ship the norm.
///
/// Written out because the assertion above passes trivially on an empty
/// set, and a manifest rename -- `pre_feedforward_layernorm` to anything
/// else -- would empty it silently. That is the failure this pairing was
/// always vulnerable to, and it is the one the driver's version could
/// not have caught either.
#[test]
fn some_row_ships_the_sandwich_norm() {
    let ids = sandwiched();
    assert!(
        !ids.is_empty(),
        "no row declares `{SANDWICH}`, so the check above is vacuous -- \
         either the manifests renamed it or the generations that ship it left"
    );
    assert!(
        ids.iter().any(|id| id.starts_with("gemma")),
        "the sandwich norm is gemma's and no gemma row declares it: {ids:?}"
    );
}
