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
    assert!(r < 512 && r >= 1);
    // One row over a buffer as long as the context fits.
    assert_eq!(mask_rows(1, 4096, 4096, rung), 1);
    // A decode row over 100,000 buffered keys: one row a fire.
    assert_eq!(mask_rows(64, 100_001, 131_072, |r| u64::from(r)), 1);
}

/// A piece of a cut pass that folds as it goes states the boundary its
/// prepare found, and its mask starts past the columns between.
#[test]
fn a_piece_restates_the_boundary_and_its_mask_starts_there() {
    let bits = [true, false, true, true, false, true];
    let mut runs = vec![0u32];
    let mut keeping = false;
    for &bit in &bits {
        if bit != keeping {
            runs.push(0);
            keeping = bit;
        }
        *runs.last_mut().unwrap() += 1;
    }
    let mask = ::engine::Mask::new(runs, bits.len() as u64);
    let mut lane = ::engine::Lane {
        tokens: vec![0; 2],
        kv: ::engine::KvDelta {
            held: 14,
            folded: 10,
            ..Default::default()
        },
        mask: Some(::engine::Masking::Rows(vec![mask.clone(), mask.clone()])),
        ..::engine::Lane::default()
    };
    restate_boundary(&mut lane, 12);
    assert_eq!(lane.kv.folded, 12);
    let Some(::engine::Masking::Rows(masks)) = &lane.mask else {
        panic!("the lane keeps a mask a row");
    };
    assert_eq!(masks[0], mask.after(2));
    // A boundary past the lane's first row is the lane's first row.
    restate_boundary(&mut lane, 20);
    assert_eq!(lane.kv.folded, 14);
    // One behind what the lane states moves nothing.
    restate_boundary(&mut lane, 3);
    assert_eq!(lane.kv.folded, 14);
}
