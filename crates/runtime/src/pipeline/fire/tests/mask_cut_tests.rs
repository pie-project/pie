use super::*;

/// A mask plane fits the room of the rung its piece runs at, `max(t *
/// t, max_context)` bits, by cutting rows and never by refusing: a
/// long buffer takes more fires, down to one row a fire, and a row
/// over any buffer the context holds fits.
#[test]
fn a_masked_pass_is_cut_to_the_room_of_its_rung() {
    let rung = |r: u32| u64::from(r.next_power_of_two().max(16));
    // An empty buffer: a full-width prompt fits its own rung.
    assert_eq!(mask_rows(512, 512, 4096, rung), 512);
    // A prompt of 512 over a buffer of 3584: 512 * 4096 bits is past
    // what a 512-token rung holds (512 * 512 = 262144, or 4096).
    let r = mask_rows(512, 3584 + 512, 4096, rung);
    assert!(u64::from(r) * 4096 <= rung(r) * rung(r) || u64::from(r) * 4096 <= 4096);
    assert!((1..512).contains(&r));
    // One row over a buffer as long as the context fits.
    assert_eq!(mask_rows(1, 4096, 4096, rung), 1);
    // A decode row over 100,000 buffered keys: one row a fire.
    assert_eq!(mask_rows(64, 100_001, 131_072, u64::from), 1);
}
