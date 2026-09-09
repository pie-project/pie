use model_compiler::{Budget, PATCH_LATTICE_FLOOR};

mod common;
use common::patch_ladder_for;

#[test]
fn the_ladder_this_file_derives_is_the_one_the_rule_describes() {
    for max_tokens in [8192u32, 4096, 2048, 1024, 96, 8] {
        let budget = Budget::new(256, max_tokens);
        let ladder = patch_ladder_for(&budget);

        let want = max_tokens.clamp(PATCH_LATTICE_FLOOR, 4096);
        assert_eq!(
            ladder.max_patches, want,
            "the ceiling at {max_tokens} tokens"
        );

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

        assert_eq!(
            ladder.max_images,
            (ladder.max_patches / PATCH_LATTICE_FLOOR).max(1),
            "as many images as the ceiling holds at the floor",
        );
        assert!(ladder.max_images >= 1, "a ladder admits at least one image");
    }

    let ladder = patch_ladder_for(&Budget::new(256, 8192));
    assert_eq!(ladder.max_patches, 4096);
    assert_eq!(ladder.buckets, vec![64, 128, 256, 512, 1024, 2048, 4096]);
}
