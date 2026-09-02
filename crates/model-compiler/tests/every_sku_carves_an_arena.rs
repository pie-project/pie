//! Bakes the real catalog (six model texts, four platforms) and checks the
//! arena is clash-free, finite and non-empty, reused at the busiest instant
//! rather than summed, tiles the node list exactly once, and keeps
//! collectives always-launch — real plans at deployment scale, not hand-built
//! fixtures.

use model_compiler::{
    Budget, PATCH_LATTICE_FLOOR,
};

mod common;
use common::patch_ladder_for;

// `patch_ladder_for` is a second statement of `engine_cuda::api::patch_ladder`
// (model-compiler cannot depend on engine-cuda to diff against it), so this
// checks the copy against the rule in prose rather than a typed-out list.
// Not #[ignore]d: bakes nothing, reads no catalog.
#[test]
fn the_ladder_this_file_derives_is_the_one_the_rule_describes() {
    for max_tokens in [8192u32, 4096, 2048, 1024, 96, 8] {
        let budget = Budget::new(256, max_tokens);
        let ladder = patch_ladder_for(&budget);

        // The ceiling: the token rectangle's, capped at two whole images, and
        // never below one whole image.
        let want = max_tokens.min(4096).max(PATCH_LATTICE_FLOOR);
        assert_eq!(ladder.max_patches, want, "the ceiling at {max_tokens} tokens");

        // The rungs: they start at the floor, they double, and the last one is
        // the ceiling. Asked of the vector rather than of the loop.
        assert_eq!(
            ladder.buckets.first().copied(),
            Some(PATCH_LATTICE_FLOOR),
            "the ladder starts at the smallest whole image: {:?}",
            ladder.buckets,
        );
        assert_eq!(
            ladder.buckets.last().copied(),
            Some(ladder.max_patches),
            "the ladder ends at its ceiling: {:?}",
            ladder.buckets,
        );
        for pair in ladder.buckets.windows(2) {
            let (low, high) = (pair[0], pair[1]);
            assert!(
                high == low * 2 || high == ladder.max_patches,
                "rung {high} follows {low} and is neither its double nor the \
                 ceiling: {:?}",
                ladder.buckets,
            );
        }

        // `max_images` is the ceiling AT the floor, and never zero — a
        // deployment that admits patch rows admits at least one image.
        assert_eq!(
            ladder.max_images,
            (ladder.max_patches / PATCH_LATTICE_FLOOR).max(1),
            "as many images as the ceiling holds at the floor",
        );
        assert!(ladder.max_images >= 1, "a ladder admits at least one image");
    }

    // AND IT IS THE LADDER THE SIBLING FILE STATES BY HAND. At the 8192-token
    // deployment every sweep in this tree uses,
    // `the_second_row_axis_costs_the_first_nothing`'s `also_admitting_patches`
    // writes the rungs out as a literal; a derivation that disagreed with the
    // one hand-written ladder in the crate would be one of the two being wrong.
    let ladder = patch_ladder_for(&Budget::new(256, 8192));
    assert_eq!(ladder.max_patches, 4096);
    assert_eq!(ladder.buckets, vec![64, 128, 256, 512, 1024, 2048, 4096]);
}

