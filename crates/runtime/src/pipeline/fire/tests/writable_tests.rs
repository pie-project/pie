use super::*;

/// A lane whose rows the device places may write from its set's frozen
/// prefix to what it reserved, and the pages the fire makes its set's
/// own are the ones under those slots: none of the frozen pages below,
/// none of the mapped pages past.
#[test]
fn a_device_placed_lane_writes_only_the_pages_made_its_own() {
    let writable = device_writable(2, 6, 13).expect("a set that reserved past its prefix");
    assert_eq!(writable, 6..13);
    assert_eq!(
        writable_pages(&writable, 4),
        1..4,
        "the page holding slot 6 through 12"
    );
    let lane = ::engine::KvDelta {
        folded: 6,
        write_end: writable.end,
        ..::engine::KvDelta::default()
    };
    assert_eq!(
        ::engine::fire::misplaced_write(lane.writable(), &[6, 12]),
        None
    );
    let below = ::engine::fire::misplaced_write(lane.writable(), &[6, 5]).expect("below");
    assert!(below.contains("kv slot 5"), "{below}");
    assert!(::engine::fire::misplaced_write(lane.writable(), &[13]).is_some());

    let refused = device_writable(2, 8, 8).expect_err("nothing reserved past the prefix");
    assert!(
        refused.contains("lane 2") && refused.contains("froze 8"),
        "{refused}"
    );
}
