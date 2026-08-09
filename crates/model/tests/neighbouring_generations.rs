//! WHERE TWO GENERATIONS SHARE A SHAPE, EACH STILL STATES ITS OWN CEILING.
//!
//! OLMo 2 and OLMo 3 are built out of the same `LlamaLikeFacts` and the
//! same projection. What separates them is not structural — it is that
//! one was trained to 4,096 tokens and the other to 65,536, and that
//! each row SAYS so. The shapes are close enough that unifying them
//! behind one shared ceiling would look like a tidy-up, and the result
//! would hand an OLMo 2 sixteen times the context it was trained on.
//!
//! # Why this is a crate-level test
//!
//! Because the property is ABOUT two generations, and neither of them
//! owns it.
//!
//! It used to live inside both, as two halves of one assertion:
//! `olmo_2/mod.rs` checked that its ceiling was lower than OLMo 3's, and
//! `olmo_3/mod.rs` checked that its own was sixteen times OLMo 2's. Each
//! half named the other generation with `crate::olmo_3::` /
//! `crate::olmo_2::`, which is exactly the sibling edge
//! `tests/sibling_isolation.rs` forbids — and forbids for a reason that
//! this case shows plainly: a generation that reaches into its neighbour
//! is a generation you cannot read, delete or replace on its own.
//!
//! Two halves also meant two chances to disagree. They stated the same
//! relation twice, once as `<` and once as `* 16`, and nothing held the
//! two spellings together. Here it is one assertion.
//!
//! The escape from the sibling rule is not "make it a helper" — a helper
//! in `shared/` that knew about OLMo would be the same knowledge one
//! directory further away. It is that a test comparing generations is
//! written where generations are compared: outside all of them.
//!
//! The gemma-2/gemma-3 pair joined later, by the same route and for a
//! reason worth naming: its assertion had been written against a
//! hand-typed `32_768`, which made it a constant the compiler folded to
//! `true`. A cross-generation claim that does not read the other
//! generation is not checking anything, and the only place it CAN read
//! it is here.

use model::catalog::{Deployed, Variant};

/// The advertised facts of a generation's first row.
///
/// Generic over the row type because each generation's `VARIANTS` is a
/// slice of its OWN shape struct — the row IS the facts — so there is no
/// one concrete type to name here.
fn first_row<V: Variant>(variants: &[V]) -> model::deployment::Advertised {
    variants
        .first()
        .expect("a generation ships at least one row")
        .deployment(Deployed::single())
        .expect("the row is servable by this build")
        .advertised
}

/// OLMo 3 extended the context and both generations' rows say so.
///
/// Stated as the exact multiple rather than as an inequality, because
/// the number is the point: 4,096 to 65,536 is what the release did, and
/// an inequality would still pass if a future edit halved OLMo 3.
#[test]
fn the_olmo_generations_state_their_own_context_ceilings() {
    let older = first_row(model::olmo_2::VARIANTS);
    let newer = first_row(model::olmo_3::VARIANTS);

    assert_eq!(
        newer.max_model_len,
        older.max_model_len * 16,
        "OLMo 2 -> OLMo 3 is 4096 -> 65536, and each row states its own"
    );

    // Every OLMo 2 row, not just the first: a generation ships several
    // sizes and the ceiling is the generation's, so one row stating it
    // correctly is not the claim.
    for v in model::olmo_2::VARIANTS {
        let a = v
            .deployment(Deployed::single())
            .expect("servable")
            .advertised;
        assert!(
            a.max_model_len < newer.max_model_len,
            "{}: OLMo 2 is the shorter-context generation and the rows must say so",
            v.id()
        );
    }
}

/// The label is the only name that separates the two.
///
/// `olmo2` and `olmo3` are the whole of what a guest program has to tell
/// a 4k model from a 64k one, so the labels differing is not cosmetic.
#[test]
fn the_olmo_generations_advertise_different_labels() {
    let older = first_row(model::olmo_2::VARIANTS);
    let newer = first_row(model::olmo_3::VARIANTS);
    assert_ne!(older.arch, newer.arch, "the digit is the difference");

    for v in model::olmo_2::VARIANTS {
        let a = v
            .deployment(Deployed::single())
            .expect("servable")
            .advertised;
        assert_ne!(
            a.arch,
            newer.arch,
            "{}: the label must separate them",
            v.id()
        );
    }
}

/// The guard is not vacuous: both generations were actually walked.
#[test]
fn both_generations_ship_rows() {
    assert!(
        !model::olmo_2::VARIANTS.is_empty() && !model::olmo_3::VARIANTS.is_empty(),
        "a generation with no rows would make every assertion above vacuous"
    );
}

/// Gemma 3 extended the context and gemma-2's rows must stay under it.
///
/// The same shape as the OLMo pair, and it arrived here the same way.
/// `gemma_2/mod.rs` had asserted `MAX_MODEL_LEN < 32_768` against a
/// number typed in by hand, which the compiler folded to `true` — it
/// could not fail, and clippy said so. Reading gemma-3's row instead
/// makes it live, and makes it a sibling edge, so it is here.
///
/// The comparison is against `gemma-3-1b`, the shortest gemma-3 row that
/// generates text. `embeddinggemma-300m` states `2048` — a sixty-fourth
/// of its siblings — because an embedding tower is trained on documents
/// that fit, and it is not on this axis at all. A `min()` over the
/// generation would find it and this test would read backwards.
#[test]
fn gemma_2_stays_under_the_ceiling_gemma_3_raised() {
    let successor = model::gemma_3::VARIANTS
        .iter()
        .find(|v| v.id() == "gemma-3-1b")
        .expect("gemma-3-1b is a row")
        .deployment(Deployed::single())
        .expect("servable")
        .advertised
        .max_model_len;

    assert_eq!(
        successor, 32_768,
        "the 1B is the text-only release and states a quarter of its siblings"
    );

    let mut walked = 0usize;
    for v in model::gemma_2::VARIANTS {
        let a = v
            .deployment(Deployed::single())
            .expect("servable")
            .advertised;
        assert!(
            a.max_model_len < successor,
            "{}: gemma-2 is the shorter-context generation and the rows must say so",
            v.id()
        );
        walked += 1;
    }
    assert!(walked > 0, "no gemma-2 row was walked");
}
