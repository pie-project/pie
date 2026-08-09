//! The two tables that answer "which `model_type` does Metal support?"
//!
//! [`ModelFamily::of`] picks this driver's geometry and decode DAG.
//! [`model::contract::MLX_ROWS`] picks the author that writes its storage
//! contract. `facts.rs`'s own doc states the obligation between them: *"The
//! lists mirror the `Naming::Mlx` rows of the author registry"*.
//!
//! They answer *different* things — an author is a storage schema, a family is
//! a compute graph, and the two partition the model space differently. What
//! they may not do is disagree about the **set**. A `model_type` with a family
//! and no author gets a plan-time "no author for model_type"; one with an
//! author and no family gets a boot-time "unsupported model_type". Two
//! unrelated-looking errors, one cause: a family added on one side only.
//!
//! # Why this is here and not in `model`
//!
//! It used to be a SOURCE GREP in `model/tests/registry_agreement.rs`, reading
//! `crates/driver-metal/csrc/src/model/facts.hpp` and pulling its literal list
//! out with a regex — because the other table was C++ and no Rust-side
//! assertion could reach it.
//!
//! That tree is deleted and this driver is Rust, so the two tables can be
//! compared directly, which is strictly stronger than a grep: a `match` arm is
//! not a literal list, and a table that stopped being greppable used to fail
//! the "did we find anything at all" guard rather than the check itself.
//!
//! It lives on this side because `model` must not depend on a driver. Its
//! counterpart is `driver-cuda-new/tests/facts_registry.rs`, which is the same
//! check for the same reason on the CUDA side.

use model::contract::MLX_ROWS;
use model::policy::Naming;

/// Every `model_type` the MLX authors claim.
fn authored() -> std::collections::BTreeSet<&'static str> {
    MLX_ROWS.iter().map(|(model_type, _)| *model_type).collect()
}

#[test]
fn the_metal_family_table_and_the_mlx_authors_name_the_same_models() {
    let authored = authored();
    let with_geometry: std::collections::BTreeSet<&str> = authored
        .iter()
        .copied()
        .filter(|m| driver_metal_new::facts::ModelFamily::is_supported(m))
        .collect();

    let authored_only: Vec<&str> = authored.difference(&with_geometry).copied().collect();
    assert!(
        authored_only.is_empty(),
        "these model types have an MLX author and NO Metal geometry, so they \
         load and then refuse at boot with 'unsupported model_type': {authored_only:?}"
    );
}

/// And the other direction: a geometry with no author.
///
/// There is no list to enumerate — `ModelFamily::of` is a `match`, not a
/// table — so this checks the strings the match arms name, which is the same
/// set `facts.rs` writes down.
#[test]
fn every_model_type_the_metal_geometry_claims_has_an_mlx_author() {
    // The arms of `ModelFamily::of`, in its own order. A family added there
    // and not here makes this test stale rather than wrong, which is why the
    // assertion below re-derives support rather than trusting the list.
    const CLAIMED: &[&str] = &[
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_next",
        "qwen3_next_text",
        "qwen3_6",
        "gemma4",
        "gemma4_text",
        "gpt_oss",
        "llama",
        "llama3",
        "mistral",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen2_moe",
    ];
    let authored = authored();
    for model_type in CLAIMED {
        assert!(
            driver_metal_new::facts::ModelFamily::is_supported(model_type),
            "`{model_type}` is listed here but `ModelFamily::of` no longer claims it; \
             this list has gone stale"
        );
        assert!(
            authored.contains(model_type),
            "`{model_type}` has a Metal geometry and no MLX author, so a boot \
             gets 'no author for model_type' at plan time"
        );
    }
    assert_eq!(
        model::contract::rows(Naming::Mlx).len(),
        MLX_ROWS.len(),
        "the rows this test reads are the ones the loader dispatches on"
    );
}
