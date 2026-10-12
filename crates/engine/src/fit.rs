//! Sizing the cache pool: the kv pages and recurrent-state slots a load
//! keeps on the device. `[model] device_kv_budget` states the pool's bytes;
//! omitted, the pool takes what `[model] device_memory_share` leaves once
//! the weights and working set are down. Either way the page count follows
//! from bytes, so it means the same on every engine and model.

use core::fmt;

/// Why a pool could not be sized: what was asked, what the device had, and
/// what one sequence needs, worded the same on every engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Short {
    /// A stated `device_kv_budget` past the room the share leaves.
    Budget { budget: u64, room: u64 },
    /// A stated `device_kv_budget` under one sequence at the declared context.
    Sequence { budget: u64, least: u64 },
    /// No budget stated, and the room the share leaves holds no sequence.
    Room { room: u64, least: u64 },
}

impl fmt::Display for Short {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mib = |bytes: u64| bytes >> 20;
        match *self {
            Short::Budget { budget, room } => write!(
                f,
                "`[model] device_kv_budget` states {} MiB for the kv pages and state slots, \
                 and the device has {} MiB under `[model] device_memory_share` once the \
                 weights and working set are down; state a smaller budget, a larger share, \
                 or a smaller `device_weight_budget`",
                mib(budget),
                mib(room),
            ),
            Short::Sequence { budget, least } => write!(
                f,
                "`[model] device_kv_budget` states {} MiB, and one sequence at the declared \
                 context beside the state slots takes {} MiB; state a larger budget, a \
                 shorter `max_model_len`, or fewer `max_state_slots`",
                mib(budget),
                mib(least),
            ),
            Short::Room { room, least } => write!(
                f,
                "the device has {} MiB for the kv pages and state slots under `[model] \
                 device_memory_share`, and one sequence at the declared context beside the \
                 state slots takes {} MiB; raise the share, state a smaller \
                 `device_weight_budget`, a shorter `max_model_len`, or fewer `max_state_slots`",
                mib(room),
                mib(least),
            ),
        }
    }
}

/// The pages a pool takes: every page a stated `budget` buys, or with none
/// stated every page `room` holds. `least` is one sequence's pages, and
/// `bytes_at` the pool's bytes at a page count (its slots fixed), which only
/// ever grows with it. A stated budget is never cut to the room: past it,
/// or under one sequence, the load is refused. Pages that cost nothing (a
/// model with no kv rows) are one sequence's: more would hold nothing.
pub fn pages_for(
    budget: Option<u64>,
    room: u64,
    least: u64,
    bytes_at: impl Fn(u64) -> u64,
) -> Result<u64, Short> {
    let limit = match budget {
        Some(budget) if budget > room => return Err(Short::Budget { budget, room }),
        Some(budget) => budget,
        None => room,
    };
    let pages = if bytes_at(least.saturating_mul(2)) == bytes_at(least) {
        least
    } else {
        pages_within(PAGE_CEILING, limit, &bytes_at)
    };
    if pages >= least {
        return Ok(pages);
    }
    let least = bytes_at(least);
    Err(match budget {
        Some(budget) => Short::Sequence { budget, least },
        None => Short::Room { room, least },
    })
}

/// No pool pages past this: a page index is a 32-bit word on every engine.
const PAGE_CEILING: u64 = u32::MAX as u64;

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
/// a model has dozens of planes. It doubles up to the room before it
/// bisects, so an `asked` far past the room costs a few probes, not 32.
#[must_use]
pub fn pages_within(asked: u64, room: u64, declared_at: impl Fn(u64) -> u64) -> u64 {
    if declared_at(asked) <= room {
        return asked;
    }
    let (mut fits, mut over) = (0u64, 1u64);
    while over < asked && declared_at(over) <= room {
        (fits, over) = (over, over.saturating_mul(2).min(asked));
    }
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
    use super::{Short, pages_for, pool_within};

    #[test]
    fn a_budget_buys_its_pages_and_none_fills_the_room() {
        const PAGE: u64 = 1 << 20;
        const FIXED: u64 = 64 << 20;
        let bytes_at = |pages: u64| FIXED + pages * PAGE;
        let room = 4096 * PAGE;
        assert_eq!(pages_for(None, room, 256, bytes_at), Ok(4096 - 64));
        assert_eq!(
            pages_for(Some(1024 * PAGE), room, 256, bytes_at),
            Ok(1024 - 64)
        );
        assert_eq!(
            pages_for(Some(room + 1), room, 256, bytes_at),
            Err(Short::Budget {
                budget: room + 1,
                room
            }),
            "a stated budget is refused past the room, never cut to it"
        );
        assert_eq!(
            pages_for(Some(128 * PAGE), room, 256, bytes_at),
            Err(Short::Sequence {
                budget: 128 * PAGE,
                least: bytes_at(256)
            })
        );
        assert_eq!(
            pages_for(None, 128 * PAGE, 256, bytes_at),
            Err(Short::Room {
                room: 128 * PAGE,
                least: bytes_at(256)
            })
        );
        assert_eq!(
            pages_for(None, room, 256, |_| FIXED),
            Ok(256),
            "pages that cost nothing are one sequence's, not every page there is"
        );
    }

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
