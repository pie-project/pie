//! **THE DEVICE HALF OF THE TICKET/COMMIT MACHINERY** — the five control
//! kernels a fire's admission decision, its publication and its settlement are
//! made of (alto design §5), and the `#[repr(C)]` layouts they read.
//!
//! The mechanism in one sentence: **the host owns a prediction and the device
//! owns the truth, and the two are reconciled by a kernel rather than by a
//! synchronize.** A fire arrives carrying TICKETS — one per host-visible
//! channel endpoint, each a claim about where that endpoint's head and tail
//! stand and what the fire intends to do there. [`pull_validate`] checks every
//! claim against the live ring words and clears the fire's commit word if any
//! is stale; [`commit_bump`] — the only writer of durable ring state — then
//! publishes and consumes, or does nothing whatsoever; [`scatter_publish`]
//! carries the cells a committed fire put out into the guest's mapped pinned
//! mirror, so the crossing is a kernel and not a `cudaMemcpy`; and [`settle`]
//! advances the guest endpoint's own counters to the prediction the fire was
//! admitted on. Nothing in the sequence reads a device word on the host, and
//! nothing waits.
//!
//! **THE SETTLEMENT IS THE ONE THAT CAME BACK LAST**, and it is why the
//! boundary used to wait. Those endpoint counters are what the NEXT fire's
//! mint predicts off; while a host thread was the only thing that could
//! advance them, every frame boundary took a `cudaStreamSynchronize` to give
//! that thread a turn. On the device they are advanced in stream order, so the
//! kernel that reads them next runs after the kernel that wrote them and no
//! host thread is between the two.
//!
//! A refused fire is a DUMMY RUN, not an error: it computed, it wrote its
//! output into the pending (tail) cell, and because the bump never moved the
//! tail no reader can address those bytes and the next fire overwrites them.
//! Pass-atomic, by construction, with no rollback path to get wrong.
//!
//! # Ordering, which is constitutional
//!
//! **Payload-before-tail comes from the kernel-launch boundary on one stream.**
//! The kernel that writes a cell is enqueued before the kernel that publishes
//! the tail announcing it; kernel completion is itself a system-scope release.
//! So there is **no `__threadfence_system()` in the publish path** and the ring
//! words are stored relaxed at system scope — dev measured one system fence at
//! ~37 µs in this launch shape on an L40S (about the whole cost of the
//! publishing kernel) and per-store release at **13.8×** relaxed, growing
//! linearly in the word count: 159 µs against 12 µs at one ticket, 792 µs
//! against 19 µs at eight. The argument, and the enumeration of every reader
//! that sits on the far side of a kernel boundary, is dev
//! `channels.hpp:263-276` and the ORDERING NOTE at `channels.hpp:389-409`;
//! `kernels/channel/channels.cuh`'s header restates it where the code is.
//!
//! Acquire on the *load* side is a different trade and is kept:
//! [`pull_validate`] is the one kernel here that reads words another agent
//! writes concurrently, so its ring-word loads are `ld.acquire.sys`.
//!
//! # One documented exception to "the bump is the only writer"
//!
//! [`commit_bump`] is the only writer of `head`, `tail`, and every full byte
//! **except one**: a host-writer pull in [`pull_validate`] sets
//! `full[slot][expected_head % cap1]` for the cell it just copied in, and it
//! does so per ticket — so a LATER ticket in the same lane can still veto the
//! fire, leaving that byte set on a pass that did not commit. dev does exactly
//! this (`channels.hpp:366-371`), and it is safe rather than sloppy: the byte
//! records something the GUEST published, not something this fire produced, the
//! head does not move, and the next fire re-pulls the same cell and sets the
//! same byte. The consumption of that cell — clearing the byte and advancing
//! the head — is still the bump's alone, and still predicated. Stated here
//! because "only writer" is otherwise exactly true and a reader who finds this
//! store will assume it is a bug.
//!
//! # The layouts
//!
//! Every struct below is `#[repr(C)]` against a declaration in
//! `kernels/channel/channels.cuh`, which is in turn dev's — field order and
//! size are asserted at compile time rather than described, because a control
//! structure that disagrees with the kernel reading it fails as a wrong ring
//! index, not as a crash.

use kernels::KernelError;

use crate::jit::{ArgValue, Ctx, Fire, Launch};

const FILE: &str = "channel/channels.cuh";

/// A device address, as the handles carry it — the same currency
/// `attn::fa2_abi` uses for the same reason: a control structure crossing to
/// a kernel holds addresses, not references.
pub type DevicePtr = u64;

/// **THE WIDEST RING A SLOT'S FULL/EMPTY BYTES ARE CUT FOR** (dev
/// `channels.hpp:47`, `kMaxRing`).
///
/// `full` is one byte per `(slot, ring)` at `slot * MAX_RING + ring`, so the
/// stride is this constant and *not* the slot's own `cap1`. A slot whose
/// declared capacity would need more cells than this is clamped at
/// registration (dev `ChannelArena::init`: `if (cap1 > kMaxRing) cap1 =
/// kMaxRing`) — the ring shrinks, the addressing never moves.
pub const MAX_RING: u32 = 64;

/// The absent ticket sentinel (dev `channels.hpp:209`, `kNoChannelTicket`) —
/// what an `expected_head`/`expected_tail` holds when this fire makes no claim
/// about that end of the ring.
pub const NO_TICKET: u64 = u64::MAX;

/// The pull-validate block: dev launches 256 threads per lane
/// (`channels.hpp:587`), which is the block the mirror→cell copy is strided
/// over.
const PULL_BLOCK: u32 = 256;

/// The bump is one thread per lane (dev `channels.hpp:196`): it walks two
/// short slot lists and writes the words nobody else may write.
const BUMP_BLOCK: u32 = 1;

/// The mask scatter's block — a lane's rows are contiguous, so one block
/// strides its own span.
const MASK_BLOCK: u32 = 256;

/// The settlement's block: dev launches 128 threads per lane
/// (`channels.hpp:544`), which strides a lane's ticket window. A fire carries
/// a handful of tickets, so this is a ceiling and not a shape.
const SETTLE_BLOCK: u32 = 128;

/// **ONE HOST-VISIBLE CHANNEL ENDPOINT, AS THIS FIRE PREDICTED IT** —
/// `#[repr(C)]` against dev `channels.hpp:216-227`
/// (`DeviceHostChannelTicket`).
///
/// The two currencies in here are not interchangeable and mixing them is the
/// bug this doc exists to prevent. `expected_head`/`expected_tail` are the
/// **monotone 64-bit counters** the guest endpoint keeps — they only ever
/// increase, which is what lets emptiness be `tail > head` and fullness a
/// subtraction. The [`Rings`] registry's `head`/`tail` are **ring positions**,
/// already reduced mod `cap1`. The one place they meet is the pull, which
/// takes `expected_head % cap1` to find the cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Ticket {
    /// The ring slot, indexing [`Rings::full`] at `slot * MAX_RING + ring`.
    pub slot: u32,
    /// Any of the `Ticket::*` flag constants, or-ed.
    pub flags: u32,
    /// The head counter the host believes this endpoint stands at.
    pub expected_head: u64,
    /// The tail counter the host believes this endpoint stands at.
    pub expected_tail: u64,
    /// The endpoint's four live words in **mapped pinned memory**:
    /// `[0] head, [1] tail, [2] poison, [3] closed`. Device-addressable under
    /// UVA, so `pull_validate` reads them in place rather than through a copy
    /// — which is what removes the per-fire D2H the design calls a violation.
    pub words: DevicePtr,
    /// The host writer's staging ring, `mirror + ring * wire_bytes`, also
    /// pinned. Null unless the ticket is flagged [`Ticket::HOST_WRITER`].
    pub mirror: DevicePtr,
    /// The device cell slab, `cells + ring * native_bytes`.
    pub cells: DevicePtr,
    /// **`capacity + 1`** — the spare-cell convention (dev
    /// `channels.hpp:10-13`). The extra cell is what makes `tail == head` mean
    /// empty unambiguously, and it is why a capacity-N channel holds at most N
    /// unconsumed items with `cap1 - 1` as the fullness bound.
    ///
    /// **Must be at least 1.** The kernel takes `cap1 - 1` in `u32` and
    /// `expected_head % cap1`, so a zero here is an unsigned underflow that
    /// admits every publish and a division by zero on the pull. dev's
    /// arithmetic is reproduced unchanged rather than guarded, because the
    /// registration side is where a capacity of `-1` should be refused —
    /// noted here so the precondition is written down somewhere.
    pub cap1: u32,
    /// Bytes per **mirror** cell — bit-packed for a bool channel, so
    /// `wire_bytes == native_bytes.div_ceil(8)` there and equal otherwise.
    pub wire_bytes: u32,
    /// Bytes per **device** cell — one byte per element, always unpacked.
    pub native_bytes: u32,
}

impl Ticket {
    /// This fire takes the committed (head) cell. Validated as
    /// `head == expected_head`: nobody else consumed it first.
    pub const CONSUME: u32 = 1 << 0;

    /// This fire writes the pending (tail) cell. Validated as
    /// `tail == expected_tail` **and** the ring having room.
    pub const PUBLISH: u32 = 1 << 1;

    /// The producer is the HOST: the cell arrives in a pinned mirror and
    /// `pull_validate` copies it into the device slab. Only acted on together
    /// with [`Ticket::CONSUME`], because what is pulled is the cell this fire
    /// is about to take.
    pub const HOST_WRITER: u32 = 1 << 2;

    /// The mirror is bit-packed, one bit per element; the pull widens it to
    /// one byte per element on the way in.
    pub const PACKED_BOOL: u32 = 1 << 3;

    /// There must actually be a committed item: `tail > head`. Separate from
    /// [`Ticket::CONSUME`] because a fire may know *where* the head is without
    /// requiring anything to be sitting on it.
    pub const REQUIRE_INPUT: u32 = 1 << 4;

    /// The consumer is the HOST: the cell this fire put into the device slab
    /// is copied out into the pinned mirror by [`scatter_publish`]. The
    /// mirror-side counterpart of [`Ticket::HOST_WRITER`], and only acted on
    /// together with [`Ticket::PUBLISH`], because what is scattered is the
    /// cell this fire just wrote.
    pub const HOST_READER: u32 = 1 << 5;

    /// **A COMMITTED FIRE MOVES THIS ENDPOINT'S HEAD TO
    /// `expected_head + 1`** — [`settle`]'s predicate, and NOT
    /// [`Ticket::CONSUME`].
    ///
    /// dev predicates its settlement on `kTicketConsume` directly
    /// (`channels.hpp:437`) because there a consuming ticket always consumed.
    /// Here [`Ticket::CONSUME`] means *this fire addresses the committed
    /// cell*, which a `read` that peeks without taking also sets, and which a
    /// take whose ring was empty at mint sets too — neither of those moves the
    /// counter. It also carries an ownership decision the device cannot see:
    /// on a channel the host READS, the head is the GUEST's counter and the
    /// engine may never store it. So the mint states the advance separately,
    /// off the same arithmetic `Session::settle` used to do on the host.
    pub const ADVANCE_HEAD: u32 = 1 << 6;

    /// **A COMMITTED FIRE MOVES THIS ENDPOINT'S TAIL TO
    /// `expected_tail + 1`** — the mirror of [`Ticket::ADVANCE_HEAD`], set
    /// where the fire puts and the ENGINE owns the tail (a channel the host
    /// reads, or a device-only ring).
    pub const ADVANCE_TAIL: u32 = 1 << 7;
}

const _: () = assert!(
    core::mem::size_of::<Ticket>() == 64,
    "channel::Ticket: sizeof disagrees with `channel/channels.cuh`'s Ticket",
);
const _: () = assert!(
    core::mem::align_of::<Ticket>() == 8,
    "channel::Ticket: alignof disagrees with `channel/channels.cuh`'s Ticket",
);

/// **ONE FIRE'S SLICE OF THE TICKET TABLE, AND THE COMMIT WORD IT VOTES ON** —
/// `#[repr(C)]` against dev `channels.hpp:229-239`
/// (`PullValidateHostChannelLane`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PullLane {
    /// The ring registry's full/empty bytes ([`Rings::full`]), which a
    /// host-writer pull sets.
    pub full: DevicePtr,
    /// **TWO WORDS, NOT ONE**: `[0]` the pass commit flag, `[1]` the kill word.
    /// Both are re-seeded per fire — a ringed snapshot may carry a stale kill
    /// from a previous occurrence of the same slot, and a kill nobody cleared
    /// would refuse a fire that is fine.
    pub pass_commit: DevicePtr,
    /// Where this lane's tickets start in the table handed to
    /// [`pull_validate`].
    pub ticket_offset: u32,
    pub ticket_count: u32,
    /// What `pass_commit[0]` is seeded to before any ticket votes. A prologue
    /// that has already failed for a reason of its own seeds 0, and no ticket
    /// can raise it back.
    pub initial_commit: u32,
    /// Non-zero makes the kernel print the ticket that vetoed the fire. A
    /// refusal is otherwise indistinguishable from every other reason a
    /// prologue does not commit — which is exactly the debugging problem dev
    /// added this seat for.
    pub diagnose: u32,
}

const _: () = assert!(
    core::mem::size_of::<PullLane>() == 32,
    "channel::PullLane: sizeof disagrees with `channel/channels.cuh`'s PullLane",
);

/// **ONE FIRE'S DURABLE RING BOOKKEEPING AND THE TWO SLOT LISTS IT MOVES** —
/// `#[repr(C)]` against dev `channels.hpp:150-160` (`CommitBumpLane`).
///
/// Build one with [`Rings::bump_lane`] rather than by hand: the first four
/// fields are the registry's and must be the registry's.
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
    /// Word `[0]` of the fire's commit pair. **Zero and this lane moves
    /// nothing** — the one predicate the whole protocol's atomicity rests on.
    pub commit: DevicePtr,
}

const _: () = assert!(
    core::mem::size_of::<BumpLane>() == 72,
    "channel::BumpLane: sizeof disagrees with `channel/channels.cuh`'s BumpLane",
);

/// **ONE FIRE'S OUTWARD TICKETS AND THE COMMIT WORD THEY RIDE ON** — the
/// publish counterpart of [`PullLane`], `#[repr(C)]` against
/// `channel/channels.cuh`'s `PublishLane`.
///
/// dev spells the same work as a flat copy list
/// (`k_scatter_host_publish_copies`, `channels.hpp:411-470`); it is a lane
/// here for the reason every other structure in this module is one — a fire's
/// commit word is per lane, and a copy that outran its lane's refusal would
/// hand the guest a cell the bump never published.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct PublishLane {
    /// Word `[0]` of the fire's commit pair — the same word [`pull_validate`]
    /// seeded and [`commit_bump`] read. **Zero and this lane copies nothing.**
    pub commit: DevicePtr,
    /// Where this lane's tickets start in the table handed to
    /// [`scatter_publish`].
    pub ticket_offset: u32,
    pub ticket_count: u32,
}

const _: () = assert!(
    core::mem::size_of::<PublishLane>() == 16,
    "channel::PublishLane: sizeof disagrees with `channel/channels.cuh`'s PublishLane",
);

/// **ONE FIRE'S SETTLEMENT WINDOW OVER THE SAME TICKET TABLE** —
/// `#[repr(C)]` against `channel/channels.cuh`'s `SettleLane`, dev
/// `channels.hpp:380-388` (`HostChannelSettlementLane`) less the fields this
/// port does not need.
///
/// Three values identical to [`PublishLane`]'s, and a distinct type on
/// purpose: two kernels reading one array would make a field added to either
/// a silent corruption of the other.
///
/// # What dev carries here and this does not
///
/// * **`host_commit`** — dev's mapped mirror of the commit pair, written so
///   the completion callback can classify a lane without a D2H. Here the
///   commit pair IS mapped pinned memory (`Session`'s own), so the host reads
///   the word the kernels wrote where they wrote it and there is nothing to
///   mirror.
/// * **`full`/`head`/`cap1`/`consume`** — dev's settlement also clears the
///   consumed cell's full byte and advances the registry head for its
///   conditional-consume slots. That is [`commit_bump`]'s `taken` loop here,
///   verbatim and predicated on the same word, so doing it again would
///   advance every consumed ring TWICE. **The registry is the bump's; the
///   endpoint words are the settlement's** — which is why this lane carries
///   no registry pointer at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct SettleLane {
    /// Word `[0]` of the fire's commit pair — the same word [`pull_validate`]
    /// seeded and every other kernel here read. **Zero and this lane advances
    /// nothing**, which is what leaves a refused fire's guest endpoint
    /// exactly where it stood.
    pub commit: DevicePtr,
    /// Where this lane's tickets start in the table handed to [`settle`].
    pub ticket_offset: u32,
    pub ticket_count: u32,
}

const _: () = assert!(
    core::mem::size_of::<SettleLane>() == 16,
    "channel::SettleLane: sizeof disagrees with `channel/channels.cuh`'s SettleLane",
);

/// **THE RING REGISTRY**: the four device arrays every slot's bookkeeping
/// lives in, kept together because they are indexed together and a lane that
/// mixed two registries would corrupt both silently.
///
/// The layout, restated once (dev `ChannelArena`, `channels.hpp:647-700`):
///
/// * **cells** — one slab per channel, cell `r` at `cells + r * cell_bytes`.
///   Not held here: a cell address travels on the [`Ticket`] that names it,
///   because only the ticket knows the cell width.
/// * **`full`** — `[slots * MAX_RING]` bytes, the full/empty bit of
///   `(slot, ring)` at `slot * MAX_RING + ring`. A byte and not a packed bit,
///   so a set and a clear from different slots never collide.
/// * **`head` / `tail`** — `[slots]` `u32` ring positions, already mod `cap1`.
///   `head` is the committed cell a take reads; `tail` is the pending cell a
///   put writes.
/// * **`cap1`** — `[slots]` `u32`, each `capacity + 1`. The spare cell is the
///   empty/full discriminator; it is never a cell anyone addresses on purpose.
///
/// Seeding, for the record: a channel declared `from(seed)` starts with
/// `full[slot * MAX_RING + 0] = 1` and `tail = 1 % cap1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Rings {
    pub full: DevicePtr,
    pub head: DevicePtr,
    pub tail: DevicePtr,
    pub cap1: DevicePtr,
    /// How many slots the four arrays are cut for.
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

    /// The byte offset of `(slot, ring)` in the `full` array — the one piece
    /// of addressing arithmetic both this crate and the kernel do, written
    /// once so a test and a kernel cannot disagree about it.
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

/// **VALIDATE EVERY TICKET AGAINST THE LIVE RING WORDS, AND PULL WHAT THE HOST
/// WROTE** — dev `k_pull_validate_host_channels_batch`
/// (`channels.hpp:277-376`).
///
/// One block per lane. Seeds each lane's commit pair, then per ticket compares
/// the host's prediction against `words[0]`/`words[1]` read straight out of
/// mapped pinned memory, `atomicAnd`-ing the commit word to zero on any
/// mismatch. A ticket that passes and is flagged
/// [`HOST_WRITER`](Ticket::HOST_WRITER)`|`[`CONSUME`](Ticket::CONSUME) also
/// copies its mirror cell into the device slab and sets the full byte.
///
/// `tickets` is the whole wave's table; each lane names its own window into it
/// with `ticket_offset`/`ticket_count`. Enqueue only — an `Ok` means the
/// launch is on the stream, not that any lane has voted.
pub fn pull_validate(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), KernelError> {
    const OP: &str = "channel.pull_validate";
    // A wave with no host-visible endpoint has no admission decision to make.
    // Nothing to enqueue is not a refusal (dev: `if (lanes.empty()) return`).
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

/// **THE ONLY WRITER OF DURABLE RING STATE** — dev `commit_bump`
/// (`channels.hpp:116-137`).
///
/// Per lane, iff its commit word survived [`pull_validate`]: set `full[tail]`
/// and advance `tail` for every put slot, then clear `full[head]` and advance
/// `head` for every taken slot. A refused lane reads zero and leaves the
/// registry byte-for-byte as it found it — the bytes that fire wrote are still
/// in the tail cell, addressable by nobody, and the next fire overwrites them.
///
/// Enqueue this AFTER the kernels that wrote the cells, on the same stream:
/// the launch boundary between them is what orders payload before tail, and it
/// is the only thing that does (see the module header).
pub fn commit_bump(ctx: &Ctx, lanes: DevicePtr, lane_count: u32) -> Result<(), KernelError> {
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

/// **THE PUBLICATION, AS A KERNEL AND NOT A COPY** — dev
/// `k_scatter_host_publish_copies` (`channels.hpp:411-470`).
///
/// A pass's `put` lands in the DEVICE slab's pending cell, because that is
/// where an emitted kernel can write; a guest reads its channel out of a
/// MAPPED PINNED mirror, because that is where it can read without a CUDA
/// call on its own thread. This is the whole of the crossing — one strided
/// copy per outward ticket, device slab to pinned mirror, no `cudaMemcpy` in
/// either direction and therefore no host between the two.
///
/// Predicated on the same commit word as everything else: a refused fire
/// scatters nothing, and the pending cell it wrote stays addressable by
/// nobody. Enqueue it AFTER [`commit_bump`] on the same stream — the guest
/// learns a cell is there when its tail word advances at settle, which is on
/// the far side of this launch, so the launch boundary is the
/// payload-before-tail ordering here as it is everywhere else.
///
/// A ticket is acted on only when flagged
/// [`PUBLISH`](Ticket::PUBLISH)`|`[`HOST_READER`](Ticket::HOST_READER); the
/// ring is `expected_tail % cap1`, arithmetic on the PREDICTION and never a
/// read of the live tail.
pub fn scatter_publish(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), KernelError> {
    const OP: &str = "channel.scatter_publish";
    // A wave with nothing outward-bound has nothing to publish, and that is
    // not a refusal — the same reading `pull_validate` gives an empty wave.
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

/// **THE ENDPOINT'S COUNTERS, ADVANCED BY THE DEVICE** — dev
/// `k_settle_host_channels_batch` (`channels.hpp:411-455`), and the kernel
/// whose absence made every frame boundary take a `cudaStreamSynchronize`.
///
/// These four words are the guest's view of the ring, and until this kernel
/// existed the HOST advanced them: synchronize, read the pinned commit word,
/// then bump each slot's counter. The next fire's mint predicts off exactly
/// those counters, so the wait was never there to make the answer correct —
/// it was there because the answer did not exist until a host thread wrote
/// it. Written by the device, in stream order, it exists when the next kernel
/// that reads it runs and no host thread stands between the two.
///
/// Per lane, iff its commit word survived: for every ticket flagged
/// [`ADVANCE_HEAD`](Ticket::ADVANCE_HEAD) store `expected_head + 1` into
/// `words[0]`, and for every [`ADVANCE_TAIL`](Ticket::ADVANCE_TAIL) store
/// `expected_tail + 1` into `words[1]`. **The prediction plus one, never a
/// read-modify-write** — the prediction is what [`pull_validate`] already
/// proved the word equals, and an increment would race the guest's own
/// counter at the other end of the ring.
///
/// **Enqueue this AFTER [`scatter_publish`]** on the same stream. That launch
/// boundary is the entire payload-before-tail argument: the cells the scatter
/// wrote into the guest's mirror are visible system-wide at its completion,
/// which strictly precedes the first store this kernel makes, so a tail can
/// never reach the guest ahead of the cell it announces. Every store here is
/// relaxed at system scope and there is no fence — see the module header.
pub fn settle(
    ctx: &Ctx,
    tickets: DevicePtr,
    lanes: DevicePtr,
    lane_count: u32,
) -> Result<(), KernelError> {
    const OP: &str = "channel.settle";
    // A wave with no host-visible endpoint has no counter to advance, and
    // that is not a refusal — the same reading every other kernel here gives
    // an empty wave.
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

/// **THE RS FOLD PREDICATE, AS DEVICE DATA** (alto design §6).
///
/// A recurrent-state scan folds a row's state only where `write_state_mask[r]`
/// is non-zero (`attn/ssm.cuh`'s `row_persists`), and the rows that may fold
/// are exactly the rows of the lanes whose fire committed. This scatters each
/// lane's commit word across that lane's rows through the row CSR the fire
/// already carries — so the predicate reaches the scan without anybody reading
/// a device word on the host.
///
/// * `commits` — `[lane_count]` device **addresses**, one commit word per lane
///   (a null entry reads as "did not commit"). Pointers rather than an array
///   of words because a lane's commit pair is allocated with the lane's
///   snapshot and the pairs are not contiguous.
/// * `indptr` — `[lane_count + 1]` `i32`; lane `l` owns rows
///   `[indptr[l], indptr[l + 1])`.
/// * `mask` — `[indptr[lane_count]]` bytes, one per row.
pub fn mask_from_commit(
    ctx: &Ctx,
    commits: DevicePtr,
    indptr: DevicePtr,
    mask: DevicePtr,
    lane_count: u32,
) -> Result<(), KernelError> {
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

    /// The addressing both this crate and the kernel do. A `full` array is
    /// `MAX_RING`-strided whatever a slot's own `cap1` is, which is the whole
    /// reason the constant exists.
    #[test]
    fn a_slots_full_bytes_are_max_ring_apart_whatever_its_capacity() {
        assert_eq!(Rings::full_at(0, 0), 0);
        assert_eq!(Rings::full_at(0, 63), 63);
        assert_eq!(Rings::full_at(1, 0), u64::from(MAX_RING));
        assert_eq!(Rings::full_at(7, 3), 7 * 64 + 3);
    }

    /// The flags are a bit set, and dev's numbering is the wire order — a
    /// ticket built here is read by a kernel written against
    /// `channels.hpp:210-214`.
    #[test]
    fn the_ticket_flags_keep_devs_numbering() {
        assert_eq!(Ticket::CONSUME, 1);
        assert_eq!(Ticket::PUBLISH, 2);
        assert_eq!(Ticket::HOST_WRITER, 4);
        assert_eq!(Ticket::PACKED_BOOL, 8);
        assert_eq!(Ticket::REQUIRE_INPUT, 16);
        assert_eq!(Ticket::HOST_READER, 32);
    }

    /// A registry hands its four arrays to every lane it builds. A lane that
    /// carried a different `full` than the pull that set the byte would
    /// publish into a ring nobody reads.
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

    /// Nothing to enqueue is not a refusal, and the check must come BEFORE
    /// the fire — a zero grid is what `Ctx::fire` refuses, and a wave with no
    /// host-visible endpoint is not an error.
    #[test]
    fn an_empty_wave_enqueues_nothing_and_refuses_nothing() {
        // SAFETY: no branch reached with `lane_count == 0` touches the stream.
        let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
        assert!(pull_validate(&ctx, 0, 0, 0).is_ok());
        assert!(commit_bump(&ctx, 0, 0).is_ok());
        assert!(scatter_publish(&ctx, 0, 0, 0).is_ok());
        assert!(mask_from_commit(&ctx, 0, 0, 0, 0).is_ok());
    }
}
