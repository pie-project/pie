use engine_api::program::LaunchOp;

use super::extent::Extents;

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

// SIX `CHANNEL_*` FLAG BITS AND THE `channel_flags` THAT PACKED THEM stood
// here: the per-channel word a device lane table used to carry so a kernel
// could read a program's declared requirement out of the table. Nothing in
// either shell packs one — readiness is `Readiness`/`Effect` on the host and
// the ticket words on the device — and the re-exports were the only thing
// keeping them compiled (alto E).
