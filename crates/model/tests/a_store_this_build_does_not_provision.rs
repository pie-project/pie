//! Four generations plan a KV store this build does not allocate, and
//! all four refuse to trace rather than hand out a fire against it.
//!
//! `glm-5`, `deepseek-v4`, `kimi-k2` and `kimi-k3` compress their KV: an
//! MLA latent plane or a per-layer compressed one, neither of which fits
//! the k/v pair the pager allocates. Their `trace` asks `deployment`
//! first and propagates the refusal, and the doc above each of those
//! guards records what happened before it was there -- the refusal fired
//! at the door and a fire was handed out anyway.
//!
//! ## Why this test exists at all
//!
//! The line AFTER each of those guards is unreachable today, and that is
//! the point: it is unreachable because the refusal always fires, not
//! because the guard is redundant. A coverage report reads the two the
//! same way, and the cheap response -- delete the tail, or delete the
//! guard -- is exactly the regression the doc comments describe.
//!
//! So the fact is written down instead. Every row of every one of the
//! four refuses, by the sentence keyed on its STORE STYLE rather than on
//! its family, and the day a build provisions one of these stores this
//! test fails and names the generation whose tail just became live code.
//!
//! It asserts the sentence, not merely that some error came back: a row
//! that refused for an unrelated reason -- a missing shard, an encoding
//! this build has no kernel for -- would satisfy `is_err` while saying
//! nothing about the store.

use model::catalog::{Deployed, catalog};
use model_ir::trace::FireClass;

/// The two sentences `KvStyle::store_refusal` states, by store.
const NO_MLA: &str = "this build provisions no MLA latent store";
const NO_PLANE: &str = "this build provisions no compressed KV plane store";

/// Which generation each id belongs to, and which store it plans.
///
/// Spelled out rather than derived from the deployment, because a test
/// that read the answer off the thing under test would pass whatever it
/// said. It earned that immediately: this table was first written with
/// all four under `Mla`, and deepseek-v4 turns out to plan a compressed
/// PLANE. "They are the four MLA families" was the shorthand everybody
/// used, including the four doc comments; it was not true.
const COMPRESSED: &[(&str, &str)] = &[
    ("glm-5", NO_MLA),
    ("deepseek-v4", NO_PLANE),
    ("kimi-k2", NO_MLA),
    ("kimi-k3", NO_MLA),
];

#[test]
fn a_compressed_kv_row_refuses_to_trace_and_names_the_store() {
    let mut seen = 0usize;
    for row in catalog() {
        let Some((_, sentence)) = COMPRESSED
            .iter()
            .find(|(stem, _)| row.id().starts_with(stem))
        else {
            continue;
        };
        seen += 1;
        for class in [FireClass::Decode, FireClass::Prefill] {
            let refusal = row
                .trace(class, Deployed::single())
                .expect_err("a fire against a store nothing allocated");
            let said = format!("{refusal:?}");
            assert!(
                said.contains(sentence),
                "{} refused {class:?} for the wrong reason: {said}",
                row.id()
            );
        }
    }
    assert_eq!(
        seen,
        COMPRESSED.len(),
        "a generation this test names has left the catalog, or was renamed \
         out from under the prefix it is matched by; the four are {COMPRESSED:?}"
    );
}

/// The refusal is the DEPLOYMENT's, not a second copy at the row.
///
/// The sentence used to be written once per family, four cosmetic
/// variations of one fact, which is four places for one of them to go
/// stale. It now lives on the store style. This holds the two together:
/// what `trace` refuses with must be exactly what the row's deployment
/// refuses with, so a family that grew its own spelling again is a
/// failure here rather than a divergence nobody reads.
#[test]
fn the_trace_refusal_is_the_deployments_own_sentence() {
    for row in catalog() {
        if !COMPRESSED
            .iter()
            .any(|(stem, _)| row.id().starts_with(stem))
        {
            continue;
        }
        let from_deployment = row
            .deployment(Deployed::single())
            .expect_err("the store is unprovisioned at the deployment too");
        let from_trace = row
            .trace(FireClass::Decode, Deployed::single())
            .expect_err("and therefore at the trace");
        assert_eq!(
            format!("{from_deployment:?}"),
            format!("{from_trace:?}"),
            "{} states the refusal twice and the two have drifted",
            row.id()
        );
    }
}
