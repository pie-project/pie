use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch};

const FILE: &str = "channel/channels.cuh";

pub type DevicePtr = u64;

pub const MAX_RING: u32 = 64;

pub const NO_TICKET: u64 = u64::MAX;

const PULL_BLOCK: u32 = 256;

const BUMP_BLOCK: u32 = 1;

const MASK_BLOCK: u32 = 256;

const SETTLE_BLOCK: u32 = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Ticket {
    pub slot: u32,
    pub flags: u32,
    pub expected_head: u64,
    pub expected_tail: u64,
    pub words: DevicePtr,
    pub mirror: DevicePtr,
    pub cells: DevicePtr,
    pub cap1: u32,
    pub wire_bytes: u32,
    pub native_bytes: u32,
}

impl Ticket {
    pub const CONSUME: u32 = 1 << 0;

    pub const PUBLISH: u32 = 1 << 1;

    pub const HOST_WRITER: u32 = 1 << 2;

    pub const PACKED_BOOL: u32 = 1 << 3;

    pub const REQUIRE_INPUT: u32 = 1 << 4;

    pub const HOST_READER: u32 = 1 << 5;

    pub const ADVANCE_HEAD: u32 = 1 << 6;

    pub const ADVANCE_TAIL: u32 = 1 << 7;
    pub const SHADOW: u32 = 1 << 8;
}

const _: () = assert!(
    core::mem::size_of::<Ticket>() == 64,
    "channel::Ticket: sizeof disagrees with `channel/channels.cuh`'s Ticket",
);
const _: () = assert!(
    core::mem::align_of::<Ticket>() == 8,
    "channel::Ticket: alignof disagrees with `channel/channels.cuh`'s Ticket",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PullLane {
    pub full: DevicePtr,
    pub pass_commit: DevicePtr,
    pub ticket_offset: u32,
    pub ticket_count: u32,
    pub initial_commit: u32,
    pub diagnose: u32,
}

const _: () = assert!(
    core::mem::size_of::<PullLane>() == 32,
    "channel::PullLane: sizeof disagrees with `channel/channels.cuh`'s PullLane",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct BumpLane {
    pub full: DevicePtr,
    pub head: DevicePtr,
    pub tail: DevicePtr,
    pub cap1: DevicePtr,
    pub taken: DevicePtr,
    pub taken_count: u32,
    pub put: DevicePtr,
    pub put_count: u32,
    pub commit: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<BumpLane>() == 72,
    "channel::BumpLane: sizeof disagrees with `channel/channels.cuh`'s BumpLane",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PublishLane {
    pub commit: DevicePtr,
    pub ticket_offset: u32,
    pub ticket_count: u32,
}

const _: () = assert!(
    core::mem::size_of::<PublishLane>() == 16,
    "channel::PublishLane: sizeof disagrees with `channel/channels.cuh`'s PublishLane",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SettleLane {
    pub commit: DevicePtr,
    pub ticket_offset: u32,
    pub ticket_count: u32,
}

const _: () = assert!(
    core::mem::size_of::<SettleLane>() == 16,
    "channel::SettleLane: sizeof disagrees with `channel/channels.cuh`'s SettleLane",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Rings {
    pub full: DevicePtr,
    pub head: DevicePtr,
    pub tail: DevicePtr,
    pub cap1: DevicePtr,
    pub slots: u32,
    _pad: u32,
}

impl Rings {
    #[must_use]
    pub const fn new(
        full: DevicePtr,
        head: DevicePtr,
        tail: DevicePtr,
        cap1: DevicePtr,
        slots: u32,
    ) -> Self {
        Self {
            full,
            head,
            tail,
            cap1,
            slots,
            _pad: 0,
        }
    }

    #[must_use]
    pub const fn full_at(slot: u32, ring: u32) -> u64 {
        slot as u64 * MAX_RING as u64 + ring as u64
    }

    #[must_use]
    pub const fn bump_lane(
        self,
        taken: DevicePtr,
        taken_count: u32,
        put: DevicePtr,
        put_count: u32,
        commit: DevicePtr,
    ) -> BumpLane {
        BumpLane {
            full: self.full,
            head: self.head,
            tail: self.tail,
            cap1: self.cap1,
            taken,
            taken_count,
            put,
            put_count,
            commit,
        }
    }
}

const _: () = assert!(
    core::mem::size_of::<Rings>() == 40,
    "channel::Rings: a registry is four addresses, a slot count and its padding",
);

pub fn pull_validate(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.pull_validate";
    if lane_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::channel::pull_validate")
            .apply(Launch::grid([lane_count, 1, 1], [PULL_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(tickets),
            ArgValue::Ptr(lanes),
            ArgValue::U32(lane_count),
        ],
    )
}

pub fn commit_bump(ctx: &Ctx, lanes: DevicePtr, lane_count: u32) -> Result<(), Error> {
    const OP: &str = "channel.commit_bump";
    if lane_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::channel::commit_bump")
            .apply(Launch::grid([lane_count, 1, 1], [BUMP_BLOCK, 1, 1])),
        &[ArgValue::Ptr(lanes), ArgValue::U32(lane_count)],
    )
}

pub fn scatter_publish(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.scatter_publish";
    if lane_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::channel::scatter_publish")
            .apply(Launch::grid([lane_count, 1, 1], [PULL_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(tickets),
            ArgValue::Ptr(lanes),
            ArgValue::U32(lane_count),
        ],
    )
}

pub fn settle(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.settle";
    if lane_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::channel::settle")
            .apply(Launch::grid([lane_count, 1, 1], [SETTLE_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(tickets),
            ArgValue::Ptr(lanes),
            ArgValue::U32(lane_count),
        ],
    )
}

pub fn mask_from_commit(
    ctx: &Ctx,
    commits: DevicePtr,
    indptr: DevicePtr,
    mask: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.mask_from_commit";
    if lane_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::channel::mask_from_commit")
            .apply(Launch::grid([lane_count, 1, 1], [MASK_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(commits),
            ArgValue::Ptr(indptr),
            ArgValue::Ptr(mask),
            ArgValue::U32(lane_count),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_every_case() {
        a_slots_full_bytes_are_max_ring_apart_whatever_its_capacity();
        the_ticket_flags_keep_devs_numbering();
        a_bump_lane_carries_the_registrys_own_four_arrays();
    }

    fn a_slots_full_bytes_are_max_ring_apart_whatever_its_capacity() {
        assert_eq!(Rings::full_at(0, 0), 0);
        assert_eq!(Rings::full_at(0, 63), 63);
        assert_eq!(Rings::full_at(1, 0), u64::from(MAX_RING));
        assert_eq!(Rings::full_at(7, 3), 7 * 64 + 3);
    }

    fn the_ticket_flags_keep_devs_numbering() {
        assert_eq!(Ticket::CONSUME, 1);
        assert_eq!(Ticket::PUBLISH, 2);
        assert_eq!(Ticket::HOST_WRITER, 4);
        assert_eq!(Ticket::PACKED_BOOL, 8);
        assert_eq!(Ticket::REQUIRE_INPUT, 16);
        assert_eq!(Ticket::HOST_READER, 32);
    }

    fn a_bump_lane_carries_the_registrys_own_four_arrays() {
        let rings = Rings::new(0x1000, 0x2000, 0x3000, 0x4000, 8);
        let lane = rings.bump_lane(0x5000, 2, 0x6000, 1, 0x7000);
        assert_eq!(
            (lane.full, lane.head, lane.tail, lane.cap1),
            (rings.full, rings.head, rings.tail, rings.cap1),
        );
        assert_eq!((lane.taken_count, lane.put_count), (2, 1));
    }

}
