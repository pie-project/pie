//! Fitting the cache pool to the device memory a load has left once its
//! weights and working set are down. The page and slot counts arrive from
//! the operator's config (or its defaults) knowing nothing of the device, so
//! they are ceilings the fit only ever lowers.

/// The least state slots a fitted pool holds for `seated` kv sequences: two
/// per seat, one per posted frame, and never under the three one buffered
/// lane holds alone (its folded state and the two pages a window spans).
#[must_use]
pub fn least_state_slots(seated: u32) -> u32 {
    seated.saturating_mul(2).max(3)
}

/// The largest page count at or under `asked` whose watermark fits `room`,
/// or zero when not even the first page does. `declared_at` is the watermark
/// at a page count and only ever grows with it, so this is a bisection over
/// a step function, not a division: each plane rounds up to a map unit, and
/// a model has dozens of planes.
#[must_use]
pub fn pages_within(asked: u64, room: u64, declared_at: impl Fn(u64) -> u64) -> u64 {
    if declared_at(asked) <= room {
        return asked;
    }
    let (mut fits, mut over) = (0u64, asked);
    while over - fits > 1 {
        let probe = fits + (over - fits) / 2;
        if declared_at(probe) <= room {
            fits = probe;
        } else {
            over = probe;
        }
    }
    fits
}

/// The state slots a hybrid's pool holds when `room` does not seat one
/// sequence beside the `asked` count's slabs: the most whole seats of two
/// slots (one a posted frame, as the runtime admits lanes) whose slabs take
/// at most half of what is past `one_sequence`, the kv pages' watermark at
/// the declared context, and never under what one buffered lane holds
/// (`least_state_slots`). `slabs_at` is the slabs' watermark at a slot count.
#[must_use]
pub fn state_slots_within(
    asked: u32,
    room: u64,
    one_sequence: u64,
    slabs_at: impl Fn(u32) -> u64,
) -> u32 {
    let seats = pages_within(
        u64::from(asked / 2),
        room.saturating_sub(one_sequence) / 2,
        |seats| slabs_at(u32::try_from(seats).unwrap_or(u32::MAX).saturating_mul(2)),
    );
    least_state_slots(u32::try_from(seats).unwrap_or(u32::MAX))
}

/// The pages and state slots a pool of `room` bytes seats at or under the
/// asked counts, for an engine that declares its whole pool at load: every
/// page that fits beside the asked slots, or, when that is under one
/// sequence (`one_slot` pages), beside the slots `state_slots_within` cuts
/// to. `None` when not even one sequence seats. `bytes_at(pages, slots)` is
/// what the pool takes at those counts.
#[must_use]
pub fn pool_within(
    pages: u64,
    slots: u32,
    one_slot: u64,
    room: u64,
    bytes_at: impl Fn(u64, u32) -> u64,
) -> Option<(u64, u32)> {
    let fit = pages_within(pages, room, |pages| bytes_at(pages, slots));
    if fit >= one_slot {
        return Some((fit, slots));
    }
    let slots = state_slots_within(slots, room, bytes_at(one_slot, 0), |slots| {
        bytes_at(0, slots)
    })
    .min(slots);
    let fit = pages_within(pages, room, |pages| bytes_at(pages, slots));
    (fit >= one_slot).then_some((fit, slots))
}

#[cfg(test)]
mod tests {
    use super::pool_within;

    #[test]
    fn the_pool_fits_the_room_under_its_ceilings() {
        const PAGE: u64 = 1 << 20;
        const SLOT: u64 = 64 << 20;
        let bytes_at = |pages: u64, slots: u32| pages * PAGE + u64::from(slots) * SLOT;
        assert_eq!(
            pool_within(1024, 16, 256, u64::MAX, bytes_at),
            Some((1024, 16))
        );
        assert_eq!(
            pool_within(65536, 16, 256, 2048 * PAGE, bytes_at),
            Some((1024, 16))
        );
        let (pages, slots) = pool_within(65536, 256, 256, 2048 * PAGE, bytes_at).unwrap();
        assert!(slots < 256 && pages >= 256 && bytes_at(pages, slots) <= 2048 * PAGE);
        assert_eq!(pool_within(65536, 3, 256, 255 * PAGE, bytes_at), None);
    }
}
