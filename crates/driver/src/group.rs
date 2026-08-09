//! Which fires may share one M3 dispatch.
//!
//! M3 runs many fires as lanes of a single grouped dispatch. Two fires can only
//! share one if they run the *same* stage — same compiled kernel, same
//! bindings — and if their work is close enough in size that the group is not
//! dominated by its widest lane. [`GroupKey`] is that pair of conditions
//! expressed as one comparable value: fires whose keys are equal are grouped,
//! and fires whose keys differ are not.
//!
//! # Why the size condition is a power of two
//!
//! A grouped dispatch runs every lane at the width of the widest, so a lane of
//! one row sitting beside a lane of a thousand pays for a thousand. Grouping on
//! the exact row count would avoid that entirely and group almost nothing --
//! every distinct count would be its own dispatch, and a batch of a hundred
//! differently-sized fires would be a hundred dispatches. [`schedule_bucket`]
//! rounds up to a power of two instead, which bounds the waste at slightly
//! under 2x and collapses the count of distinct groups to about thirty-two.
//!
//! # Why the key is a struct and not a string
//!
//! The C++ built the key by `reinterpret_cast`ing a `std::uint64_t` into a
//! `std::string` and pushing the bucket byte onto it. That works as a map key
//! and nothing else: it is host-endian, it holds embedded NULs, it is not text
//! in any encoding, and it cannot be compared to anything without knowing all
//! of that. It exists only because `std::map` wanted a single key type for two
//! numbers, which is not a reason to serialize them.
//!
//! # An absent key is not an empty one
//!
//! `m3_stage_key` returned `""` when the stage had no canonical identity, and
//! `m3_stage_group_key` also returned `""` when the program was null and when
//! the requested stage was not in it -- three different facts as one value that
//! is, itself, a perfectly usable map key. Every caller happened to test
//! `key.empty()` first. `Option<GroupKey>` is the same discipline with the test
//! moved into the type.

use driver_api::plan::LaunchOp;

use super::extent::Extents;
use super::readiness::Effect;

/// How many channel slots one M3 lane can address.
///
/// The lane table has a fixed number of per-lane channel entries and the
/// emitted kernels index it directly, so this is a hard ABI bound rather than a
/// budget: a stage that wants more cannot be run by these kernels at all.
pub const MAX_CHANNELS: usize = 29;

/// A stage names a channel slot past what a lane can address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooManyChannels {
    /// How many slots the stage's ops would need.
    pub needed: usize,
    /// How many a lane has.
    pub limit: usize,
}

/// Which power-of-two size class a fire's work falls in.
///
/// Zero for a fire of one row or fewer; otherwise `ceil(log2(rows))`, where
/// `rows` is the larger of the sampled and total row counts -- sampling can ask
/// for more rows than the forward pass produced, and the dispatch has to cover
/// whichever is bigger.
#[must_use]
pub fn schedule_bucket(extents: &Extents) -> u8 {
    let rows = extents.sampled_rows.max(extents.row_count);
    if rows <= 1 {
        return 0;
    }
    // `ceil(log2(rows))`, taken as the bit width of `rows - 1` so that an exact
    // power of two lands in its own bucket rather than the next one up.
    (u32::BITS - (rows - 1).leading_zeros()) as u8
}

/// What two fires must agree on to share a dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupKey {
    /// The stage's canonical identity: same identity, same compiled kernel.
    pub identity: u64,
    /// The size class, from [`schedule_bucket`].
    pub bucket: u8,
}

impl GroupKey {
    /// The key for a stage at a fire's extents, or `None` if the stage has no
    /// canonical identity and therefore cannot be grouped with anything --
    /// including with another stage that also has none.
    #[must_use]
    pub fn of(identity: u64, extents: &Extents) -> Option<GroupKey> {
        if identity == 0 {
            return None;
        }
        Some(GroupKey {
            identity,
            bucket: schedule_bucket(extents),
        })
    }
}

/// How many channel slots a stage's ops actually address.
///
/// The count is one past the highest slot named, not the number of distinct
/// slots: the lane table is indexed by slot, so a stage that uses only slot 7
/// still needs eight entries.
///
/// # Errors
///
/// [`TooManyChannels`] if that exceeds [`MAX_CHANNELS`]. The C++ bounded the
/// stage's *declared* channel count when the program was registered and left
/// this one unbounded, so an op naming a slot past the declared count produced
/// a binding count the argument table has no room for -- checked in the wrong
/// place, since it is the slots the ops name that get bound.
pub fn used_channel_slots(ops: &[LaunchOp]) -> Result<usize, TooManyChannels> {
    let needed = ops
        .iter()
        .filter(|op| op.channel != u32::MAX)
        .map(|op| op.channel as usize + 1)
        .max()
        .unwrap_or(0);
    if needed > MAX_CHANNELS {
        return Err(TooManyChannels {
            needed,
            limit: MAX_CHANNELS,
        });
    }
    Ok(needed)
}

/// This entry names a channel at all. Without it the rest of the word is not
/// to be read: an unused lane-table slot is zeroed, and zero must not look
/// like a channel that requires nothing.
pub const CHANNEL_VALID: u32 = 1 << 0;
/// The lane's first touch is a take or read, so it needs a cell.
pub const CHANNEL_NEEDS_FULL: u32 = 1 << 1;
/// The lane's first touch is a put, so it needs room.
pub const CHANNEL_NEEDS_EMPTY: u32 = 1 << 2;
/// The lane takes from this channel.
pub const CHANNEL_TAKE: u32 = 1 << 3;
/// The lane puts to this channel.
pub const CHANNEL_PUT: u32 = 1 << 4;
/// A lane that fails on this channel must not be retried in place.
///
/// Set when the lane has already had a visible effect that a retry would
/// repeat, so the group must be torn down and recomposed instead.
pub const CHANNEL_RETRY_INELIGIBLE: u32 = 1 << 5;

/// Pack one lane's channel effect into the word the grouped kernel reads.
#[must_use]
pub fn channel_flags(effect: &Effect, retry_ineligible: bool) -> u32 {
    let mut flags = CHANNEL_VALID;
    for (set, bit) in [
        (effect.requires_full, CHANNEL_NEEDS_FULL),
        (effect.requires_empty, CHANNEL_NEEDS_EMPTY),
        (effect.take, CHANNEL_TAKE),
        (effect.put, CHANNEL_PUT),
        (retry_ineligible, CHANNEL_RETRY_INELIGIBLE),
    ] {
        if set {
            flags |= bit;
        }
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(row_count: u32, sampled_rows: u32) -> Extents {
        Extents {
            row_count,
            sampled_rows,
            ..Extents::default()
        }
    }

    /// The bucket is `ceil(log2(rows))`, which puts an exact power of two in
    /// its own bucket rather than in the next one up -- the difference between
    /// a 1024-row fire paying for 1024 and paying for 2048.
    #[test]
    fn the_bucket_is_the_ceiling_of_the_base_two_logarithm() {
        for (count, bucket) in [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (8, 3),
            (9, 4),
            (1024, 10),
            (1025, 11),
        ] {
            assert_eq!(schedule_bucket(&rows(count, 0)), bucket, "{count} rows");
        }
    }

    /// Sampling can ask for more rows than the forward pass produced, so the
    /// dispatch has to be sized on whichever is larger.
    #[test]
    fn the_bucket_follows_whichever_row_count_is_larger() {
        assert_eq!(schedule_bucket(&rows(4, 64)), schedule_bucket(&rows(64, 4)));
        assert_eq!(schedule_bucket(&rows(4, 64)), 6);
    }

    /// The widest fire the extents can describe still fits the byte.
    #[test]
    fn the_widest_possible_fire_still_has_a_bucket() {
        assert_eq!(schedule_bucket(&rows(u32::MAX, 0)), 32);
    }

    /// Same stage, similar size: one dispatch.
    #[test]
    fn fires_of_one_stage_in_one_size_class_share_a_key() {
        assert_eq!(GroupKey::of(7, &rows(5, 0)), GroupKey::of(7, &rows(8, 0)));
    }

    /// Different stage, or a size class apart: not one dispatch. Grouping
    /// either would run the wrong kernel or make every lane pay for the widest.
    #[test]
    fn a_different_stage_or_size_class_is_a_different_key() {
        let base = GroupKey::of(7, &rows(8, 0));
        assert_ne!(base, GroupKey::of(8, &rows(8, 0)));
        assert_ne!(base, GroupKey::of(7, &rows(9, 0)));
    }

    /// A stage with no canonical identity groups with nothing -- not even with
    /// another stage that also has none, which is what an empty-string key
    /// would have done to a caller that forgot to test it.
    #[test]
    fn a_stage_with_no_identity_has_no_key_at_all() {
        assert_eq!(GroupKey::of(0, &rows(8, 0)), None);
        assert_eq!(GroupKey::of(0, &rows(8, 0)), GroupKey::of(0, &rows(9, 0)));
        assert!(GroupKey::of(0, &rows(8, 0)).is_none());
    }

    fn op(channel: u32) -> LaunchOp {
        LaunchOp {
            channel,
            ..LaunchOp::default()
        }
    }

    /// The lane table is indexed by slot, so the count is one past the highest
    /// slot named and not the number of distinct slots.
    #[test]
    fn the_slot_count_is_one_past_the_highest_slot_not_the_number_used() {
        assert_eq!(used_channel_slots(&[op(7)]), Ok(8));
        assert_eq!(used_channel_slots(&[op(0), op(7), op(0)]), Ok(8));
    }

    /// An op that touches no channel spells it `u32::MAX`, which must not be
    /// read as the highest slot in the universe.
    #[test]
    fn ops_that_touch_no_channel_do_not_widen_the_table() {
        assert_eq!(used_channel_slots(&[op(u32::MAX)]), Ok(0));
        assert_eq!(used_channel_slots(&[op(2), op(u32::MAX)]), Ok(3));
        assert_eq!(used_channel_slots(&[]), Ok(0));
    }

    /// The C++ bounded the declared channel count at registration and left this
    /// count unbounded, so an op naming a slot past the declaration produced a
    /// binding count with no room in the argument table.
    #[test]
    fn a_slot_past_what_a_lane_can_address_is_refused() {
        assert_eq!(
            used_channel_slots(&[op(MAX_CHANNELS as u32 - 1)]),
            Ok(MAX_CHANNELS)
        );
        assert_eq!(
            used_channel_slots(&[op(MAX_CHANNELS as u32)]),
            Err(TooManyChannels {
                needed: MAX_CHANNELS + 1,
                limit: MAX_CHANNELS
            })
        );
    }

    /// The valid bit is what separates "a channel that requires nothing" from
    /// "a zeroed lane-table entry", which are the same word without it.
    #[test]
    fn an_effect_that_requires_nothing_is_still_marked_valid() {
        assert_eq!(channel_flags(&Effect::default(), false), CHANNEL_VALID);
        assert_ne!(channel_flags(&Effect::default(), false), 0);
    }

    #[test]
    fn each_part_of_an_effect_sets_its_own_bit() {
        let effect = Effect {
            requires_full: true,
            requires_empty: true,
            take: true,
            put: true,
            capacity: 4,
        };
        assert_eq!(
            channel_flags(&effect, true),
            CHANNEL_VALID
                | CHANNEL_NEEDS_FULL
                | CHANNEL_NEEDS_EMPTY
                | CHANNEL_TAKE
                | CHANNEL_PUT
                | CHANNEL_RETRY_INELIGIBLE
        );
        assert_eq!(
            channel_flags(&effect, false) & CHANNEL_RETRY_INELIGIBLE,
            0,
            "retry eligibility is the caller's fact, not the effect's"
        );
    }

    /// Every bit is distinct, so no two facts about a channel can be confused
    /// for one another.
    #[test]
    fn the_flag_bits_do_not_overlap() {
        let bits = [
            CHANNEL_VALID,
            CHANNEL_NEEDS_FULL,
            CHANNEL_NEEDS_EMPTY,
            CHANNEL_TAKE,
            CHANNEL_PUT,
            CHANNEL_RETRY_INELIGIBLE,
        ];
        let mut seen = 0u32;
        for bit in bits {
            assert_eq!(bit.count_ones(), 1, "{bit:#x} is not a single bit");
            assert_eq!(seen & bit, 0, "{bit:#x} collides with an earlier flag");
            seen |= bit;
        }
    }
}
