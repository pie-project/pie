//! Device kernels for the channel ticket/commit protocol (`pull_validate`,
//! `commit_bump`, `scatter_publish`, `settle`) and their `#[repr(C)]` layouts,
//! mirroring `channels.cuh`.

use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch};

const FILE: &str = "channel/channels.cuh";

/// A device address, as used in these `#[repr(C)]` structs.
pub type DevicePtr = u64;

/// Stride of `full` per slot (dev `kMaxRing`). Capacities above this are
/// clamped at registration; addressing never changes.
pub const MAX_RING: u32 = 64;

/// Sentinel meaning "no claim made" for `expected_head`/`expected_tail`.
pub const NO_TICKET: u64 = u64::MAX;

/// Threads per lane for `pull_validate`.
const PULL_BLOCK: u32 = 256;

/// One thread per lane: the bump is the sole writer of durable ring state.
const BUMP_BLOCK: u32 = 1;

/// Threads per lane for the mask scatter.
const MASK_BLOCK: u32 = 256;

/// Threads per lane for `settle`.
const SETTLE_BLOCK: u32 = 128;

/// One host-visible channel endpoint, as this fire predicted it (`#[repr(C)]`
/// vs dev's `DeviceHostChannelTicket`). `expected_head`/`expected_tail` are
/// monotone guest counters; `Rings`' `head`/`tail` are already mod `cap1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Ticket {
    /// Ring slot, indexing `Rings::full` at `slot * MAX_RING + ring`.
    pub slot: u32,
    /// Any of the `Ticket::*` flag constants, or-ed.
    pub flags: u32,
    /// Head counter the host believes this endpoint stands at.
    pub expected_head: u64,
    /// Tail counter the host believes this endpoint stands at.
    pub expected_tail: u64,
    /// Endpoint's four live words in mapped pinned memory: `[0] head, [1]
    /// tail, [2] poison, [3] closed`.
    pub words: DevicePtr,
    /// Host writer's staging ring, `mirror + ring * wire_bytes`. Null unless
    /// flagged [`Ticket::HOST_WRITER`].
    pub mirror: DevicePtr,
    /// Device cell slab, `cells + ring * native_bytes`.
    pub cells: DevicePtr,
    /// `capacity + 1` (spare-cell convention). Must be at least 1: used as
    /// `cap1 - 1` and as a modulus, so zero underflows or divides by zero.
    pub cap1: u32,
    /// Bytes per mirror cell; bit-packed for a bool channel
    /// (`native_bytes.div_ceil(8)`), else equal to `native_bytes`.
    pub wire_bytes: u32,
    /// Bytes per device cell, one byte per element, always unpacked.
    pub native_bytes: u32,
}

impl Ticket {
    /// This fire takes the committed (head) cell; validated as
    /// `head == expected_head`.
    pub const CONSUME: u32 = 1 << 0;

    /// This fire writes the pending (tail) cell; validated as
    /// `tail == expected_tail` and the ring having room.
    pub const PUBLISH: u32 = 1 << 1;

    /// Producer is the host: the cell arrives in the pinned mirror and
    /// `pull_validate` copies it in. Only meaningful with [`Ticket::CONSUME`].
    pub const HOST_WRITER: u32 = 1 << 2;

    /// Mirror is bit-packed, one bit per element; widened to one byte per
    /// element on the pull.
    pub const PACKED_BOOL: u32 = 1 << 3;

    /// Requires `tail > head` (a committed item present), independent of
    /// [`Ticket::CONSUME`].
    pub const REQUIRE_INPUT: u32 = 1 << 4;

    /// Consumer is the host: the cell `scatter_publish` writes is copied to
    /// the pinned mirror. Only meaningful with [`Ticket::PUBLISH`].
    pub const HOST_READER: u32 = 1 << 5;

    /// A committed fire moves this endpoint's head to `expected_head + 1` —
    /// [`settle`]'s predicate, not [`Ticket::CONSUME`] (a peeking `read` also
    /// sets CONSUME without moving the counter).
    pub const ADVANCE_HEAD: u32 = 1 << 6;

    /// A committed fire moves this endpoint's tail to `expected_tail + 1`,
    /// the mirror of [`Ticket::ADVANCE_HEAD`].
    pub const ADVANCE_TAIL: u32 = 1 << 7;
    /// A follower rank's ticket under tensor parallelism: the ring's words
    /// and mirror are rank 0's to vote on, advance and publish into. The
    /// vote is taken as held, the host writer's cell is still pulled at the
    /// predicted ring position, and nothing durable is written.
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

/// One fire's slice of the ticket table and the commit word it votes on
/// (`#[repr(C)]` vs dev `PullValidateHostChannelLane`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PullLane {
    /// Ring registry's full/empty bytes ([`Rings::full`]); a host-writer
    /// pull sets bytes here.
    pub full: DevicePtr,
    /// Two words: `[0]` pass commit flag, `[1]` kill word. Both re-seeded per
    /// fire so a stale value from a previous snapshot cannot leak in.
    pub pass_commit: DevicePtr,
    /// Start of this lane's tickets in the table handed to [`pull_validate`].
    pub ticket_offset: u32,
    pub ticket_count: u32,
    /// Value `pass_commit[0]` is seeded to before any ticket votes.
    pub initial_commit: u32,
    /// Non-zero makes the kernel print the ticket that vetoed the fire.
    pub diagnose: u32,
}

const _: () = assert!(
    core::mem::size_of::<PullLane>() == 32,
    "channel::PullLane: sizeof disagrees with `channel/channels.cuh`'s PullLane",
);

/// One fire's durable ring bookkeeping and the two slot lists it moves
/// (`#[repr(C)]` vs dev `CommitBumpLane`). Build via [`Rings::bump_lane`]
/// rather than by hand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct BumpLane {
    pub full: DevicePtr,
    pub head: DevicePtr,
    pub tail: DevicePtr,
    pub cap1: DevicePtr,
    /// Slots this fire took from: `full[head]` clears and `head` advances.
    pub taken: DevicePtr,
    pub taken_count: u32,
    /// Slots this fire put to: `full[tail]` sets and `tail` advances.
    pub put: DevicePtr,
    pub put_count: u32,
    /// Word `[0]` of the fire's commit pair. Zero and this lane moves
    /// nothing.
    pub commit: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<BumpLane>() == 72,
    "channel::BumpLane: sizeof disagrees with `channel/channels.cuh`'s BumpLane",
);

/// One fire's outward tickets and the commit word they ride on, the publish
/// counterpart of [`PullLane`] (`#[repr(C)]` vs `channels.cuh`'s
/// `PublishLane`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PublishLane {
    /// Word `[0]` of the fire's commit pair. Zero and this lane copies
    /// nothing.
    pub commit: DevicePtr,
    /// Start of this lane's tickets in the table handed to
    /// [`scatter_publish`].
    pub ticket_offset: u32,
    pub ticket_count: u32,
}

const _: () = assert!(
    core::mem::size_of::<PublishLane>() == 16,
    "channel::PublishLane: sizeof disagrees with `channel/channels.cuh`'s PublishLane",
);

/// One fire's settlement window over the same ticket table (`#[repr(C)]` vs
/// `channels.cuh`'s `SettleLane`); a distinct type from [`PublishLane`] so a
/// field added to one can't corrupt the other. Carries no registry pointer:
/// [`commit_bump`]'s `taken` loop already clears `full` and advances `head`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SettleLane {
    /// Word `[0]` of the fire's commit pair. Zero and this lane advances
    /// nothing.
    pub commit: DevicePtr,
    /// Start of this lane's tickets in the table handed to [`settle`].
    pub ticket_offset: u32,
    pub ticket_count: u32,
}

const _: () = assert!(
    core::mem::size_of::<SettleLane>() == 16,
    "channel::SettleLane: sizeof disagrees with `channel/channels.cuh`'s SettleLane",
);

/// The ring registry: four parallel device arrays indexed by slot (dev
/// `ChannelArena`). `full` is `[slots * MAX_RING]` bytes at
/// `slot * MAX_RING + ring`; `head`/`tail` are `[slots]` ring positions
/// already mod `cap1` (head = cell a take reads, tail = cell a put writes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Rings {
    pub full: DevicePtr,
    pub head: DevicePtr,
    pub tail: DevicePtr,
    pub cap1: DevicePtr,
    /// Number of slots the four arrays are cut for.
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

    /// Byte offset of `(slot, ring)` in the `full` array.
    #[must_use]
    pub const fn full_at(slot: u32, ring: u32) -> u64 {
        slot as u64 * MAX_RING as u64 + ring as u64
    }

    /// This registry plus one fire's two slot lists and its commit word.
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

/// Validates every ticket against the live ring words and pulls what the host
/// wrote (dev `k_pull_validate_host_channels_batch`). One block per lane; a
/// passing [`HOST_WRITER`](Ticket::HOST_WRITER)`|`[`CONSUME`](Ticket::CONSUME)
/// ticket copies its mirror cell into the device slab. Enqueue only — `Ok`
/// means the launch is on the stream, not that any lane has voted.
pub fn pull_validate(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.pull_validate";
    // Empty wave: nothing to enqueue, not a refusal (matches dev).
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

/// The only writer of durable ring state (dev `commit_bump`). Per lane, iff
/// its commit word survived [`pull_validate`]: sets/advances `tail` for
/// each put slot, then clears/advances `head` for each taken slot. Enqueue
/// after the kernels that wrote the cells, on the same stream, so that
/// launch boundary orders payload before tail.
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

/// The publication, as a kernel rather than a copy (dev
/// `k_scatter_host_publish_copies`). Copies each outward ticket's pending
/// cell from the device slab to the guest's mapped pinned mirror, predicated
/// on the same commit word as [`commit_bump`]. Enqueue after [`commit_bump`]
/// on the same stream, so the guest never sees a tail advance before the
/// cell it announces.
pub fn scatter_publish(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.scatter_publish";
    // Nothing outward-bound: not a refusal, same as an empty wave elsewhere.
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

/// Advances the guest endpoint's counters on the device (dev
/// `k_settle_host_channels_batch`). Per lane, iff committed: stores
/// `expected_head + 1`/`expected_tail + 1` into `words[0]`/`words[1]` for
/// [`ADVANCE_HEAD`](Ticket::ADVANCE_HEAD)/[`ADVANCE_TAIL`](Ticket::ADVANCE_TAIL)
/// tickets — always the prediction plus one, never a read-modify-write.
/// Enqueue after [`scatter_publish`] on the same stream; stores are relaxed
/// at system scope, ordering comes from the launch boundary.
pub fn settle(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.settle";
    // No host-visible endpoint: not a refusal, same as an empty wave.
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

/// Scatters each lane's commit word across that lane's rows: a downstream
/// scan folds a row's state only where its mask byte is nonzero.
///
/// * `commits` — `[lane_count]` device addresses, one commit word per lane
///   (null reads as "did not commit").
/// * `indptr` — `[lane_count + 1]` `i32`; lane `l` owns rows
///   `[indptr[l], indptr[l + 1])`.
/// * `mask` — `[indptr[lane_count]]` bytes, one per row.
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

    /// `full` addressing is `MAX_RING`-strided regardless of a slot's `cap1`.
    #[test]
    fn a_slots_full_bytes_are_max_ring_apart_whatever_its_capacity() {
        assert_eq!(Rings::full_at(0, 0), 0);
        assert_eq!(Rings::full_at(0, 63), 63);
        assert_eq!(Rings::full_at(1, 0), u64::from(MAX_RING));
        assert_eq!(Rings::full_at(7, 3), 7 * 64 + 3);
    }

    /// Flag bit values must match dev's numbering.
    #[test]
    fn the_ticket_flags_keep_devs_numbering() {
        assert_eq!(Ticket::CONSUME, 1);
        assert_eq!(Ticket::PUBLISH, 2);
        assert_eq!(Ticket::HOST_WRITER, 4);
        assert_eq!(Ticket::PACKED_BOOL, 8);
        assert_eq!(Ticket::REQUIRE_INPUT, 16);
        assert_eq!(Ticket::HOST_READER, 32);
    }

    /// A bump lane must reuse the registry's own four arrays, or the pull's
    /// writes and the bump's target different memory.
    #[test]
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
