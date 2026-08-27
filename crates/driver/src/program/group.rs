use driver_api::program::LaunchOp;

use super::extent::Extents;
use super::readiness::Effect;

pub const MAX_CHANNELS: usize = 29;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooManyChannels {
    pub needed: usize,

    pub limit: usize,
}

#[must_use]
pub fn schedule_bucket(extents: &Extents) -> u8 {
    let rows = extents.sampled_rows.max(extents.row_count);
    if rows <= 1 {
        return 0;
    }

    (u32::BITS - (rows - 1).leading_zeros()) as u8
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupKey {
    pub identity: u64,

    pub bucket: u8,
}

impl GroupKey {
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

pub fn used_channel_slots(ops: &[LaunchOp]) -> Result<usize, TooManyChannels> {
    // Was `filter(channel != u32::MAX).map(channel as usize + 1)`, the
    // `PIE_NO_CHANNEL` sentinel read by hand. `filter_map` over the `Option`
    // is the same walk with the sentinel spelled by the type.
    let needed = ops
        .iter()
        .filter_map(|op| op.channel)
        .map(|channel| channel as usize + 1)
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

pub const CHANNEL_VALID: u32 = 1 << 0;

pub const CHANNEL_NEEDS_FULL: u32 = 1 << 1;

pub const CHANNEL_NEEDS_EMPTY: u32 = 1 << 2;

pub const CHANNEL_TAKE: u32 = 1 << 3;

pub const CHANNEL_PUT: u32 = 1 << 4;

pub const CHANNEL_RETRY_INELIGIBLE: u32 = 1 << 5;

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
