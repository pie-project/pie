use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, kernel};

const FILE: &str = "runtime/channel.hip";

pub type DevicePtr = u64;

pub const MAX_RING: u32 = 64;

pub const NO_TICKET: u64 = u64::MAX;

/// Sixteen 32-lane teams, a ticket each at once (`kTeam`).
const PULL_BLOCK: u32 = 512;

/// A thread per put or take, a wavefront of them at a time.
const BUMP_BLOCK: u32 = 64;

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
    "channel::Ticket: sizeof disagrees with `runtime/channel.hip`'s Ticket",
);
const _: () = assert!(
    core::mem::align_of::<Ticket>() == 8,
    "channel::Ticket: alignof disagrees with `runtime/channel.hip`'s Ticket",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PullLane {
    pub full: DevicePtr,
    pub pass_commit: DevicePtr,
    pub ticket_offset: u32,
    pub ticket_count: u32,
    pub initial_commit: u32,
}

const _: () = assert!(
    core::mem::size_of::<PullLane>() == 32,
    "channel::PullLane: sizeof disagrees with `runtime/channel.hip`'s PullLane",
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
    "channel::BumpLane: sizeof disagrees with `runtime/channel.hip`'s BumpLane",
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
    "channel::PublishLane: sizeof disagrees with `runtime/channel.hip`'s PublishLane",
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SettleLane {
    pub commit: DevicePtr,
    pub ticket_offset: u32,
    pub ticket_count: u32,
    /// Where the host reads the pass's verdict, which the settle copies
    /// from `commit` (in the wave's device tables).
    pub commit_host: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<SettleLane>() == 24,
    "channel::SettleLane: sizeof disagrees with `runtime/channel.hip`'s SettleLane",
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

/// Loads the kernels a guest program's wave fires (pull, bump, publish,
/// settle) on the current device now: every load fires them once a program
/// binds, and nothing builds after ready.
pub fn load_wave_kernels() -> Result<(), Error> {
    for name in ["pull_validate", "commit_bump", "scatter_publish", "settle"] {
        kernel(FILE, &format!("::pie::channel::{name}")).load()?;
    }
    Ok(())
}

/// Loads the kernels a fire's staging gathers with (a tap's words, a lane's
/// device-carried mask bits, a table of them) on the current device now:
/// a serving fire fires them, and nothing builds after ready.
pub fn load_gather_kernels() -> Result<(), Error> {
    for name in [
        "gather_words",
        "gather_bits",
        "gather_bit_runs",
        "beam_step",
        "copy_pages",
        "copy_rs_rows",
    ] {
        kernel(FILE, &format!("::pie::channel::{name}")).load()?;
    }
    Ok(())
}

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
        Fire::at(kernel(FILE, "::pie::channel::pull_validate"))
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
        Fire::at(kernel(FILE, "::pie::channel::commit_bump"))
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
        Fire::at(kernel(FILE, "::pie::channel::scatter_publish"))
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
        Fire::at(kernel(FILE, "::pie::channel::settle"))
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
        Fire::at(kernel(FILE, "::pie::channel::mask_from_commit"))
            .apply(Launch::grid([lane_count, 1, 1], [MASK_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(commits),
            ArgValue::Ptr(indptr),
            ArgValue::Ptr(mask),
            ArgValue::U32(lane_count),
        ],
    )
}

const BITS_BLOCK: u32 = 128;

/// Packs the `bits` bool bytes at `src` into little-endian bits at `dst`
/// (`dst`'s bits past `bits` in its last byte are cleared).
pub fn gather_bits(ctx: &Ctx, src: DevicePtr, dst: DevicePtr, bits: u32) -> Result<(), Error> {
    const OP: &str = "channel.gather_bits";
    if bits == 0 {
        return Ok(());
    }
    let blocks = bits.div_ceil(8).div_ceil(BITS_BLOCK);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::gather_bits"))
            .apply(Launch::grid([blocks, 1, 1], [BITS_BLOCK, 1, 1])),
        &[ArgValue::Ptr(src), ArgValue::Ptr(dst), ArgValue::U32(bits)],
    )
}

/// One run of [`gather_bit_runs`]: `bits` bool bytes at `src` packed into
/// bits `dst_byte` bytes past the destination.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitRun {
    pub src: DevicePtr,
    /// Where the run lands in the fire's masks: 64-bit, as their offsets.
    pub dst_byte: u64,
    pub bits: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<BitRun>() == 24);

/// Packs each of `run_count` [`BitRun`]s at `runs` into `dst`, a block a run.
pub fn gather_bit_runs(
    ctx: &Ctx,
    runs: DevicePtr,
    dst: DevicePtr,
    run_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.gather_bit_runs";
    if run_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::gather_bit_runs"))
            .apply(Launch::grid([run_count, 1, 1], [BITS_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(runs),
            ArgValue::Ptr(dst),
            ArgValue::U32(run_count),
        ],
    )
}

/// One run `gather_words` copies: `words` u32 words from `src` to
/// `dst_word` words past the destination.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WordRun {
    pub src: DevicePtr,
    pub dst_word: u32,
    pub words: u32,
}

const _: () = assert!(size_of::<WordRun>() == 16);

const GATHER_BLOCK: u32 = 32;

/// Copies each of `run_count` runs at `runs` into `dst`, a block a run: one
/// launch where a copy a run would be a driver call and a copy node apiece.
pub fn gather_words(
    ctx: &Ctx,
    runs: DevicePtr,
    dst: DevicePtr,
    run_count: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.gather_words";
    if run_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::gather_words"))
            .apply(Launch::grid([run_count, 1, 1], [GATHER_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(runs),
            ArgValue::Ptr(dst),
            ArgValue::U32(run_count),
        ],
    )
}

/// A lane of the staged rs tables (`channel.cu`'s `RsLane`): where its rows
/// land in its rs buffer pages and which buffered rows it replays first.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RsLane {
    /// The buffer token its first row lands at; [`RsLane::NONE`]: none.
    pub at: u32,
    /// The buffered rows it replays ahead of its own.
    pub replay: u32,
    /// Where its replayed tokens start in the replay table.
    pub replay_off: u32,
    /// Where its pages' slot addresses start in the page table.
    pub page_off: u32,
    /// Where its extended rows start in the fire's extended run: past
    /// every earlier lane's replayed and own rows.
    pub ext: u32,
}

const _: () = assert!(size_of::<RsLane>() == 20);

impl RsLane {
    pub const NONE: u32 = u32::MAX;

    /// A lane that lands no rows and replays none.
    pub const INERT: RsLane = RsLane {
        at: RsLane::NONE,
        replay: 0,
        replay_off: 0,
        page_off: 0,
        ext: 0,
    };
}

/// The staged rs tables a fire's buffered rows are addressed through: a
/// [`RsLane`] a lane, the slot address of each lane's pages, and each
/// replayed row's buffer token.
#[derive(Clone, Copy, Debug)]
pub struct RsTables {
    pub lanes: DevicePtr,
    pub pages: DevicePtr,
    pub replay_tok: DevicePtr,
    /// Each replayed row resolved, at its token's index: its page slot's
    /// address with the row in the page in the top 16 bits. The host writes
    /// a plain lane's; a beam group's head step writes its beams' once it
    /// has resolved their pages (`BeamRsRun`).
    pub resolved: DevicePtr,
    /// The rows a page holds.
    pub page_tokens: u32,
}

/// Where a buffered plane's rows lie in a page's slot: `off` bytes in, a
/// token's row every `stride` bytes.
#[derive(Clone, Copy, Debug)]
pub struct RsPlane {
    pub off: u64,
    pub stride: u64,
}

const RS_BLOCK: u32 = 128;

/// Lands each lane's rows of `src` at its buffer tokens (a block a lane, up
/// to `lanes_stated`; the armed window's live words say how many are live).
pub fn rs_scatter(
    ctx: &Ctx,
    src: DevicePtr,
    row_bytes: u32,
    qo_indptr: DevicePtr,
    lanes_stated: u32,
    tables: RsTables,
    plane: RsPlane,
) -> Result<(), Error> {
    const OP: &str = "channel.rs_scatter";
    if lanes_stated == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::rs_scatter"))
            .apply(Launch::grid([lanes_stated, 1, 1], [RS_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(src),
            ArgValue::U32(row_bytes),
            ArgValue::Ptr(qo_indptr),
            ctx.stage(),
            ArgValue::U32(lanes_stated),
            ArgValue::Ptr(tables.lanes),
            ArgValue::Ptr(tables.pages),
            ArgValue::U32(tables.page_tokens),
            ArgValue::Usize(plane.off),
            ArgValue::Usize(plane.stride),
        ],
    )
}

/// The window's extended layout into `csr` (`lanes_cap + 1` words) and the
/// live words the recurrent kernels read the extended rows by (`ext_win`):
/// from the window's place in the fire's extended run, its first lane's
/// [`RsLane::ext`].
pub fn rs_ext_csr(
    ctx: &Ctx,
    qo_indptr: DevicePtr,
    lanes_stated: u32,
    lanes_cap: u32,
    lanes: DevicePtr,
    csr: DevicePtr,
    ext_win: DevicePtr,
) -> Result<(), Error> {
    const OP: &str = "channel.rs_ext_csr";
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::rs_ext_csr"))
            .apply(Launch::grid([1, 1, 1], [1, 1, 1])),
        &[
            ArgValue::Ptr(qo_indptr),
            ctx.stage(),
            ArgValue::U32(lanes_stated),
            ArgValue::U32(lanes_cap),
            ArgValue::Ptr(lanes),
            ArgValue::Ptr(csr),
            ArgValue::Ptr(ext_win),
        ],
    )
}

/// Each lane's extended rows into `ext`: its replayed buffer rows (of
/// `plane`, when the value is one) then its own rows of `own`, which land in
/// its buffer too. With `layout` (`csr`, `ext_win`), it lays the window's
/// extended layout as `rs_ext_csr` does, for the op's later launches.
#[allow(clippy::too_many_arguments)]
pub fn rs_ext_gather(
    ctx: &Ctx,
    own: DevicePtr,
    ext: DevicePtr,
    row_bytes: u32,
    qo_indptr: DevicePtr,
    layout: Option<(DevicePtr, DevicePtr)>,
    lanes_stated: u32,
    tables: RsTables,
    plane: Option<RsPlane>,
) -> Result<(), Error> {
    const OP: &str = "channel.rs_ext_gather";
    if lanes_stated == 0 {
        return Ok(());
    }
    let (pages, at) = match plane {
        Some(plane) => (tables.pages, plane),
        None => (0, RsPlane { off: 0, stride: 0 }),
    };
    let (csr, ext_win) = layout.unwrap_or((0, 0));
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::rs_ext_gather"))
            .apply(Launch::grid([lanes_stated, 1, 1], [RS_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(own),
            ArgValue::Ptr(ext),
            ArgValue::U32(row_bytes),
            ArgValue::Ptr(qo_indptr),
            ArgValue::Ptr(csr),
            ArgValue::Ptr(ext_win),
            ctx.stage(),
            ArgValue::U32(lanes_stated),
            ArgValue::Ptr(tables.lanes),
            ArgValue::Ptr(pages),
            ArgValue::Ptr(tables.replay_tok),
            ArgValue::U32(tables.page_tokens.max(1)),
            ArgValue::Usize(at.off),
            ArgValue::Usize(at.stride),
        ],
    )
}

/// Each lane's own rows of the extended `ext` back onto `target`.
#[allow(clippy::too_many_arguments)]
pub fn rs_ext_land(
    ctx: &Ctx,
    ext: DevicePtr,
    target: DevicePtr,
    row_bytes: u32,
    qo_indptr: DevicePtr,
    csr: DevicePtr,
    lanes_stated: u32,
    lanes: DevicePtr,
) -> Result<(), Error> {
    const OP: &str = "channel.rs_ext_land";
    if lanes_stated == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::rs_ext_land"))
            .apply(Launch::grid([lanes_stated, 1, 1], [RS_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(ext),
            ArgValue::Ptr(target),
            ArgValue::U32(row_bytes),
            ArgValue::Ptr(qo_indptr),
            ArgValue::Ptr(csr),
            ctx.stage(),
            ArgValue::U32(lanes_stated),
            ArgValue::Ptr(lanes),
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

/// A lane's page copy in one page list (`channel.hip`'s `PageCopyRun`):
/// its `[src, dst, tokens]` at `copy`, the reserve pages' base addresses at
/// `base`, its planes `planes` entries of the plane table from `plane_at`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageCopyRun {
    pub copy: DevicePtr,
    pub base: DevicePtr,
    pub fault: DevicePtr,
    pub reserve: u32,
    pub page: u32,
    pub floor_rel: u32,
    pub plane_at: u32,
    pub planes: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<PageCopyRun>() == 48);

/// Where a kv plane lies in a page, and the bytes one token holds in it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagePlane {
    pub offset: u64,
    pub cell: u64,
}

const _: () = assert!(size_of::<PagePlane>() == 16);

/// The fault a lane's page list states in its fault word: an entry past its
/// set's reserve, a list shorter than the pages it reads, a row written below
/// the set's frozen slots, a copy out of its bounds (`channel.hip`).
pub const PAGES_PAST_RESERVE: u32 = 1;
pub const PAGES_SHORT: u32 = 2;
pub const PAGES_BELOW_FLOOR: u32 = 3;
pub const PAGES_COPY_BOUNDS: u32 = 4;

const PAGES_BLOCK: u32 = 128;

/// Copies each of `run_count` [`PageCopyRun`]s at `runs` over its planes of
/// `planes`, a block a (run, plane); `max_planes` is the most a run spans.
pub fn copy_pages(
    ctx: &Ctx,
    runs: DevicePtr,
    planes: DevicePtr,
    run_count: u32,
    max_planes: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.copy_pages";
    if run_count == 0 || max_planes == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::copy_pages"))
            .apply(Launch::grid([run_count, max_planes, 1], [PAGES_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(runs),
            ArgValue::Ptr(planes),
            ArgValue::U32(run_count),
        ],
    )
}

/// A lane's staged page table [`beam_step`] resolves over its group's list
/// (`channel.hip`'s `BeamListRun`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeamListRun {
    pub indices: DevicePtr,
    pub write_page: DevicePtr,
    pub write_offset: DevicePtr,
    pub lane: u32,
    pub entries: u32,
    pub rows: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<BeamListRun>() == 40);

/// A lane's rs page table [`beam_step`] resolves over its group's list, on a
/// hybrid (`BEAM-PAGE-LISTS.md` §11; `channel.hip`'s `BeamRsRun`): its
/// `entries` entries from `from`, entry e holding the `tokens` rows of slots
/// `base + e * tokens` on, each resolved to the set's entry (`bases`, from
/// its entry 0, `bases_len` of them) holding the same rows of its beam's
/// list's page.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeamRsRun {
    pub pages: DevicePtr,
    pub bases: DevicePtr,
    pub lane: u32,
    pub from: u32,
    pub entries: u32,
    pub base: u32,
    pub tokens: u32,
    pub bases_len: u32,
    /// The fire's staged rs lanes, replay rows and resolved rows: the head
    /// step resolves the lane's replayed rows over the pages it wrote (null
    /// `resolved`: none).
    pub lanes_tab: DevicePtr,
    pub replay_tok: DevicePtr,
    pub resolved: DevicePtr,
}

const _: () = assert!(size_of::<BeamRsRun>() == 64);

/// A beam's rs copy (`BEAM-PAGE-LISTS.md` §6.4 on a hybrid, `channel.hip`'s
/// `RsCopyRun`): the `[src, dst, tokens]` its group's step left at `copy`,
/// moved row by row of each plane through the set's rs entries (`bases`,
/// as [`BeamRsRun`]'s); slots below the buffer have no row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RsCopyRun {
    pub copy: DevicePtr,
    pub bases: DevicePtr,
    pub fault: DevicePtr,
    pub reserve: u32,
    pub page: u32,
    pub tokens: u32,
    pub base: u32,
    pub f0: u32,
    pub floor_rel: u32,
    pub plane_at: u32,
    pub planes: u32,
    pub bases_len: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<RsCopyRun>() == 64);

/// Copies each of `run_count` [`RsCopyRun`]s at `runs` over its planes of
/// `planes`, a block a (run, plane); `max_planes` is the most a run spans.
pub fn copy_rs_rows(
    ctx: &Ctx,
    runs: DevicePtr,
    planes: DevicePtr,
    run_count: u32,
    max_planes: u32,
) -> Result<(), Error> {
    const OP: &str = "channel.copy_rs_rows";
    if run_count == 0 || max_planes == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::copy_rs_rows"))
            .apply(Launch::grid([run_count, max_planes, 1], [PAGES_BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(runs),
            ArgValue::Ptr(planes),
            ArgValue::U32(run_count),
        ],
    )
}

/// One beam group's step (`new-memory/designs/BEAM-PAGE-LISTS.md`,
/// `channel.hip`'s `BeamStepRun`): its resident state, the parents its
/// last epilogue wrote, its pool's device ids, its lanes' list runs and
/// fault words, and its numbers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BeamStepRun {
    pub state: DevicePtr,
    pub parents: DevicePtr,
    pub trans: DevicePtr,
    pub lists: DevicePtr,
    pub faults: DevicePtr,
    pub lanes: u32,
    /// The list width and pool pages the state is laid for.
    pub cap: u32,
    /// The pool's pages now (`reserved` pages past `f0`).
    pub reserve: u32,
    pub page: u32,
    pub folded: u32,
    /// The slot every beam writes this fire.
    pub slot: u32,
    pub list_count: u32,
    /// The ring of pending pages: a page freed at fire t is handed out from
    /// fire t + frames + 1.
    pub frames: u32,
    /// The pool pages the fire stages, from the first: every page the step
    /// names lies below them (the lowest-free rule), and one that does not
    /// faults its lane.
    pub backed: u32,
    /// Its lanes' rs page tables (a hybrid's, §11), `rs_count` of them.
    pub rs_count: u32,
    pub rs: DevicePtr,
}

const _: () = assert!(size_of::<BeamStepRun>() == 88);

/// The most beams one group holds.
pub const BEAM_MAX_LANES: u32 = 64;
/// A lane's parent past its group (`beam_step`).
pub const PAGES_PARENT: u32 = 5;
/// A fire found fewer free pages than it hands out (exact sizing makes it
/// unreachable: a bug where it happens).
pub const PAGES_EXHAUSTED: u32 = 6;

/// The u32 words a group of `lanes` beams keeps for lists of `cap` columns
/// over `cap` pool pages and `frames` pending rings
/// (`channel.hip`'s state layout).
#[must_use]
pub const fn beam_state_words(lanes: u32, cap: u32, frames: u32) -> u64 {
    4 + 2 * lanes as u64 * cap as u64 + cap as u64 + frames as u64 * cap as u64 + 3 * lanes as u64
}

/// The state words where a group's copies for this fire lie.
#[must_use]
pub const fn beam_copies_word(lanes: u32, cap: u32, frames: u32) -> u64 {
    4 + 2 * lanes as u64 * cap as u64 + cap as u64 + frames as u64 * cap as u64
}

const BEAM_BLOCK: u32 = 256;

/// Runs each of `run_count` [`BeamStepRun`]s at `runs`, a block a group.
pub fn beam_step(ctx: &Ctx, runs: DevicePtr, run_count: u32) -> Result<(), Error> {
    const OP: &str = "channel.beam_step";
    if run_count == 0 {
        return Ok(());
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::channel::beam_step"))
            .apply(Launch::grid([run_count, 1, 1], [BEAM_BLOCK, 1, 1])),
        &[ArgValue::Ptr(runs), ArgValue::U32(run_count)],
    )
}
