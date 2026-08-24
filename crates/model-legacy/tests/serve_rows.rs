//! THE R1 PIN: `model::serve::ROWS` states the same numbers this catalog does.
//!
//! The consumer cutover moved `engine`'s three questions — how many layers,
//! how wide are the logits, which chat template — off `catalog::Variant` and
//! onto a flat `const` table in the new `model` crate. The templates moved as
//! files, so their identity is `git mv` and needs no test. The two NUMBERS did
//! not move: they were read out of this catalog once and written down there,
//! and a number written down twice is a number that can disagree.
//!
//! This test is the third party. It links both crates and holds them to each
//! other row for row, so a table that drifts is a failing test rather than a
//! sampler sized from the wrong model. It dies when this crate does (R3), by
//! which time the new catalog states the numbers itself.

use model_legacy::catalog::{self, Deployed, Variant};

/// The logits width the legacy catalog attributes to a row.
///
/// Two sources, and the second one is not a shortcut. Eleven of the fourteen
/// rows answer through `deployment`, which is the reading `engine::model::
/// register` used. The other three — `glm-5-106b-a12b`, `kimi-k3`,
/// `deepseek-v4` — REFUSE a single-rank deployment in this build, and the
/// refusal is about the PAGER ("this build provisions no MLA latent store"),
/// not about the model: their `vocab` is stated on the row's own facts and is
/// as true as the other eleven's. Reading it there is what lets the pin cover
/// all fourteen instead of the eleven that happen to be deployable today.
fn legacy_vocab(row: &'static dyn Variant) -> u32 {
    if let Ok(deployment) = row.deployment(Deployed::single()) {
        return deployment.shape.vocab;
    }
    let mla: &[(&str, u32)] = &[
        (
            model_legacy::glm_5::VARIANTS[0].id,
            model_legacy::glm_5::VARIANTS[0].shape.vocab,
        ),
        (
            model_legacy::kimi_k3::VARIANTS[0].id,
            model_legacy::kimi_k3::VARIANTS[0].shape.vocab,
        ),
        (
            model_legacy::deepseek_v4::VARIANTS[0].id,
            model_legacy::deepseek_v4::VARIANTS[0].shape.vocab,
        ),
    ];
    mla.iter()
        .find(|(id, _)| *id == row.id())
        .map(|(_, vocab)| *vocab)
        .unwrap_or_else(|| {
            panic!(
                "{}: this build refuses its single-rank deployment and it is not one of \
                 the three MLA rows whose facts state a vocab directly — read the width \
                 off the row and add it here",
                row.id()
            )
        })
}

#[test]
fn serve_rows_cover_exactly_the_catalog() {
    let mut legacy: Vec<&str> = catalog::ids();
    let mut served: Vec<&str> = model::serve::ids();
    legacy.sort_unstable();
    served.sort_unstable();
    assert_eq!(
        legacy, served,
        "`model::serve::ROWS` and this catalog ship different ids"
    );
}

#[test]
fn serve_rows_agree_on_layers_and_vocab() {
    for row in catalog::catalog() {
        let served = model::serve::row(row.id())
            .unwrap_or_else(|| panic!("`{}` is in `model::serve::ROWS`", row.id()));
        assert_eq!(
            served.layers,
            row.load_shape().layers,
            "{}: layer count",
            row.id()
        );
        assert_eq!(
            served.vocab,
            legacy_vocab(*row),
            "{}: logits width",
            row.id()
        );
    }
}
