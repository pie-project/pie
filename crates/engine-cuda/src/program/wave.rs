//! **ONE BOUNDARY'S CONTROL PLANE, IN ONE ALLOCATION AND THREE LAUNCHES.**
//!
//! A boundary is a run of independent guest passes — sixty-four samplers at
//! c=64, one per lane — and every one of them needs the same three control
//! kernels around its regions: `channel::pull_validate` in front,
//! `channel::commit_bump` and `channel::scatter_publish` behind. All three
//! have taken a LANE COUNT and per-lane arrays since they were written; what
//! they never had was a caller with more than one lane to hand them.
//!
//! # The number this module came for
//!
//! An nsys capture at c=64, after the boundary's wait was collapsed to one
//! (wave 6), put `channel::pull_validate` at **362 ms, 11% of all GPU time**,
//! spread over one one-block launch per attachment per token step. Almost
//! none of it is arithmetic: the kernel reads a handful of pinned endpoint
//! words over PCIe, one ticket at a time, in a single block — so sixty-four
//! lanes of it are sixty-four serialized PCIe round trips that could have
//! been sixty-four blocks of one launch overlapping each other's latency.
//! `commit_bump` and `scatter_publish` are the same shape and the same waste.
//!
//! The blocker was never the kernels. It was that the six control structures
//! a fire stages — the ticket table, the two slot lists `commit_bump` walks,
//! and the [`PullLane`], [`BumpLane`] and [`PublishLane`] the fire is — lived
//! in a buffer cut PER SESSION (wave 6's `StagePlan`), so a wave's lanes were
//! sixty-four disjoint allocations and there was no array to point a lane
//! count at. This module is that buffer moved up one level: a boundary's
//! lanes stage into ONE arena, contiguously, and the three kernels launch
//! once each over the whole wave.
//!
//! ```text
//! per boundary, per attachment          per boundary
//! ----------------------------          ------------
//! 1 cudaMemcpyAsync (the six)           1 cudaMemcpyAsync (the whole arena)
//! 1 pull_validate  grid=1               1 pull_validate  grid=lanes
//! the regions                           the regions, unchanged
//! 1 commit_bump    grid=1               1 commit_bump    grid=lanes
//! 1 scatter_publish grid=1              1 scatter_publish grid=lanes
//! ```
//!
//! # The arena's layout, and why the bump lanes are patched
//!
//! Seven regions, in this order and 8-aligned: the ticket table, the taken
//! list, the put list, then the four lane arrays. Every lane's tickets are a
//! contiguous window of the first region named by its own `ticket_offset`,
//! which is exactly the field [`PullLane`] and [`PublishLane`] have always
//! carried and nobody has ever set to anything but zero.
//!
//! [`BumpLane`] is the one that cannot be written at mint: it carries the
//! DEVICE ADDRESSES of its own slices of the taken and put lists, and those
//! are not known until every lane has staged and the regions have been laid
//! out. So a mint records its spans and [`Wave::fly`] patches the two
//! pointers in before the image is copied. Nothing else about a lane is
//! deferred.
//!
//! # What the arena may not do
//!
//! **IT IS REWRITTEN ONCE PER BATCH AND READ BY THE BATCH'S KERNELS**, so the
//! next batch's copy must not overtake the previous batch's kernels. It does
//! not: the copy and the kernels are on the same stream, and a stream is a
//! queue. That argument is the whole of it and it never involved a
//! synchronize — which matters, because the boundary no longer takes one.
//!
//! **A GROWTH IS THE ONE THING STREAM ORDER DOES NOT COVER.** `cudaFree` is
//! not enqueued: it takes effect when the host calls it, so replacing the
//! arena while a previous batch's kernels still hold its address is a
//! use-after-free that stream ordering says nothing about. While every batch
//! ended in a synchronize that could not happen; now it can. So the arena
//! grows at a HIGH-WATER MARK and the buffer it replaces is **retired, not
//! freed** ([`Wave::retired`]): growth is `next_power_of_two`, so the whole
//! retired set is at most the live arena over again, a few tens of kilobytes
//! at c=64, and it stops growing after the first few boundaries of a run.
//!
//! [`PullLane`]: kernels_cuda::channel::PullLane

use kernels_cuda::channel::{self, BumpLane, PublishLane, PullLane, SettleLane, Ticket};

use crate::device::{Buffer, Context};
use crate::error::Result;

use super::launch::slice_bytes;

/// Eight, because the widest field in any of the three lane records is a
/// pointer and the kernels dereference them where they lie.
const ALIGN: usize = 8;

/// Where a lane's slot lists sit in the wave's two lists, as an index and a
/// length — the halves of a [`BumpLane`]'s two pointers that are known at
/// mint.
#[derive(Clone, Copy, Debug, Default)]
struct Span {
    taken_at: usize,
    taken_len: usize,
    put_at: usize,
    put_len: usize,
}

/// **THE CONTROL STRUCTURES OF EVERY FIRE IN ONE BOUNDARY.**
///
/// Accumulated on the host as one lane per staged fire, committed to the
/// device as one image and three launches, and cleared for the next batch.
/// The device buffer outlives the batch; the host vectors do not.
#[derive(Debug, Default)]
pub struct Wave {
    /// The arena, at its high-water mark. `None` until a first batch says how
    /// big a batch is.
    arena: Option<Buffer>,
    /// Every lane's tickets, concatenated; a lane's window is
    /// `[ticket_offset, ticket_offset + ticket_count)`.
    tickets: Vec<Ticket>,
    /// Every lane's taken slots, concatenated.
    taken: Vec<u32>,
    /// Every lane's put slots, concatenated.
    put: Vec<u32>,
    pull: Vec<PullLane>,
    /// The bump lanes with their two list pointers still zero — see the
    /// module header.
    bump: Vec<BumpLane>,
    publish: Vec<PublishLane>,
    /// The settlement lanes — same three values as [`Wave::publish`]'s and a
    /// region of their own, because two kernels reading one array would make
    /// a field added to either a silent corruption of the other.
    settle: Vec<SettleLane>,
    /// One per lane, parallel to [`Wave::bump`].
    spans: Vec<Span>,
    /// **THE ARENAS A GROWTH REPLACED, KEPT ALIVE** — see the module header.
    /// A previous batch's kernels may still be reading one, and `cudaFree`
    /// happens on the host the moment it is called rather than in stream
    /// order, so the only safe thing to do with a replaced arena is nothing.
    retired: Vec<Buffer>,
}

/// Where each of the six regions starts in the arena, and how long the whole
/// image is.
#[derive(Clone, Copy, Debug, Default)]
struct Regions {
    tickets: usize,
    taken: usize,
    put: usize,
    pull: usize,
    bump: usize,
    publish: usize,
    settle: usize,
    bytes: usize,
}

impl Wave {
    /// **STAGE ONE FIRE'S CONTROL STRUCTURES**, answering the lane index it
    /// took.
    ///
    /// `pull` and `publish` arrive with their `ticket_offset` unset; this is
    /// what sets them, because only the wave knows where the lane's window
    /// landed. `bump` arrives with its two list pointers unset for the same
    /// reason, and they are filled in at [`Wave::fly`].
    pub(super) fn stage(
        &mut self,
        tickets: &[Ticket],
        taken: &[u32],
        put: &[u32],
        mut pull: PullLane,
        bump: BumpLane,
        mut publish: PublishLane,
        mut settle: SettleLane,
    ) -> usize {
        let offset = u32::try_from(self.tickets.len()).unwrap_or(u32::MAX);
        let count = u32::try_from(tickets.len()).unwrap_or(u32::MAX);
        pull.ticket_offset = offset;
        pull.ticket_count = count;
        publish.ticket_offset = offset;
        publish.ticket_count = count;
        settle.ticket_offset = offset;
        settle.ticket_count = count;
        let span = Span {
            taken_at: self.taken.len(),
            taken_len: taken.len(),
            put_at: self.put.len(),
            put_len: put.len(),
        };
        self.tickets.extend_from_slice(tickets);
        self.taken.extend_from_slice(taken);
        self.put.extend_from_slice(put);
        self.pull.push(pull);
        self.bump.push(bump);
        self.publish.push(publish);
        self.settle.push(settle);
        self.spans.push(span);
        self.pull.len() - 1
    }

    /// **Is anything staged and not yet flown?**
    pub(super) fn staged(&self) -> usize {
        self.pull.len()
    }

    /// **THE COPY AND THE PULL**: lay the arena out, patch the bump lanes,
    /// carry the whole image across in one `cudaMemcpyAsync`, and launch
    /// `channel::pull_validate` once for every lane in the batch.
    ///
    /// Enqueue-only, like everything else on the fire path. The caller
    /// launches its regions after this and calls [`Wave::land`] behind them.
    ///
    /// # Errors
    ///
    /// Whatever the allocation, the copy or the launch said.
    pub(super) fn fly(&mut self, context: &Context) -> Result<()> {
        let lanes = self.pull.len();
        if lanes == 0 {
            return Ok(());
        }
        let regions = self.regions();
        self.reserve(regions.bytes)?;
        let base = self.arena.as_ref().map_or(0, Buffer::ptr);

        // **THE ONE THING A MINT COULD NOT KNOW.** A lane's slot lists are at
        // an offset into a list whose base did not exist yet.
        for (lane, span) in self.spans.iter().enumerate() {
            let bump = &mut self.bump[lane];
            bump.taken = base + (regions.taken + span.taken_at * size_of::<u32>()) as u64;
            bump.taken_count = u32::try_from(span.taken_len).unwrap_or(u32::MAX);
            bump.put = base + (regions.put + span.put_at * size_of::<u32>()) as u64;
            bump.put_count = u32::try_from(span.put_len).unwrap_or(u32::MAX);
        }

        let mut image = vec![0u8; regions.bytes];
        let mut put_at = |offset: usize, bytes: &[u8]| {
            image[offset..offset + bytes.len()].copy_from_slice(bytes);
        };
        put_at(regions.tickets, &slice_bytes(&self.tickets));
        put_at(regions.taken, &slice_bytes(&self.taken));
        put_at(regions.put, &slice_bytes(&self.put));
        put_at(regions.pull, &slice_bytes(&self.pull));
        put_at(regions.bump, &slice_bytes(&self.bump));
        put_at(regions.publish, &slice_bytes(&self.publish));
        put_at(regions.settle, &slice_bytes(&self.settle));

        // **THE SOURCE IS A `Vec` THAT DIES HERE, AND THAT IS SAFE**: a
        // pageable source is copied into the driver's staging buffer before
        // `cudaMemcpyAsync` returns (`Buffer::stage`'s own doc). The
        // DESTINATION is this wave's, rewritten once per batch, and the only
        // writer that could race this batch's kernels is the next batch's —
        // a later copy on the same stream, on the far side of the caller's
        // one synchronize.
        let stream = context.stream();
        if let Some(arena) = self.arena.as_mut() {
            arena.stage(stream, 0, &image)?;
        }
        let lane_count = u32::try_from(lanes).unwrap_or(u32::MAX);
        channel::pull_validate(
            context.ctx(),
            base + regions.tickets as u64,
            base + regions.pull as u64,
            lane_count,
        )?;
        Ok(())
    }

    /// **THE BUMP AND THE PUBLICATION**, once for the whole batch, and then
    /// the host state cleared for the next one.
    ///
    /// Enqueue these AFTER the regions that wrote the cells, on the same
    /// stream: the launch boundary between them is what orders payload before
    /// tail, and it is the only thing that does.
    ///
    /// # Errors
    ///
    /// Whatever the launches said.
    pub(super) fn land(&mut self, context: &Context) -> Result<()> {
        let lanes = self.pull.len();
        if lanes == 0 {
            return Ok(());
        }
        let regions = self.regions();
        let base = self.arena.as_ref().map_or(0, Buffer::ptr);
        let lane_count = u32::try_from(lanes).unwrap_or(u32::MAX);
        channel::commit_bump(context.ctx(), base + regions.bump as u64, lane_count)?;
        channel::scatter_publish(
            context.ctx(),
            base + regions.tickets as u64,
            base + regions.publish as u64,
            lane_count,
        )?;
        // **AND THE SETTLEMENT LAST, WHICH IS THE WHOLE ORDERING ARGUMENT.**
        // `scatter_publish` above wrote the cells into the guests' mapped
        // mirrors; this advances the tails that ANNOUNCE those cells. The
        // launch boundary between the two is what orders payload before tail
        // — kernel completion is a system-scope release — and it is the only
        // thing that does, which is why neither kernel fences and why this one
        // may never be hoisted above the scatter.
        channel::settle(
            context.ctx(),
            base + regions.tickets as u64,
            base + regions.settle as u64,
            lane_count,
        )?;
        self.clear();
        Ok(())
    }

    /// Forget a batch that will never fly — the host half only, since nothing
    /// of it reached the device.
    pub(super) fn clear(&mut self) {
        self.tickets.clear();
        self.taken.clear();
        self.put.clear();
        self.pull.clear();
        self.bump.clear();
        self.publish.clear();
        self.settle.clear();
        self.spans.clear();
    }

    /// Where the six regions sit, from what is staged right now.
    fn regions(&self) -> Regions {
        let mut at = 0usize;
        let mut place = |bytes: usize| {
            let offset = at;
            at += bytes.div_ceil(ALIGN) * ALIGN;
            offset
        };
        let tickets = place(self.tickets.len() * size_of::<Ticket>());
        let taken = place(self.taken.len() * size_of::<u32>());
        let put = place(self.put.len() * size_of::<u32>());
        let pull = place(self.pull.len() * size_of::<PullLane>());
        let bump = place(self.bump.len() * size_of::<BumpLane>());
        let publish = place(self.publish.len() * size_of::<PublishLane>());
        let settle = place(self.settle.len() * size_of::<SettleLane>());
        Regions {
            tickets,
            taken,
            put,
            pull,
            bump,
            publish,
            settle,
            bytes: at.max(ALIGN),
        }
    }

    /// Grow the arena to hold `bytes`, and never shrink it.
    ///
    /// **A HIGH-WATER MARK AND NOT A FIT.** Growing to fit every batch would
    /// put a `cudaMalloc` and a `cudaFree` on the fire path (article 7);
    /// growing at the mark puts them on the first few boundaries of a run and
    /// never again.
    ///
    /// **AND THE REPLACED ARENA IS RETIRED RATHER THAN DROPPED.** A previous
    /// batch's kernels may still be reading it, and `cudaFree` is not
    /// enqueued — it takes effect when the host calls it. The boundary used to
    /// end in a synchronize, which made "nothing is airborne here" true by
    /// accident; it does not any more, so the module header's rule is enforced
    /// by keeping the bytes instead of by assuming the wait.
    fn reserve(&mut self, bytes: usize) -> Result<()> {
        if self.arena.as_ref().is_some_and(|arena| arena.bytes() >= bytes) {
            return Ok(());
        }
        if let Some(outgrown) = self.arena.take() {
            self.retired.push(outgrown);
        }
        // A little slack, so a wave that gains one attachment does not
        // reallocate: the arena is a few tens of kilobytes at c=64 and the
        // ceiling is the boundary's lane count, which is bounded by the
        // batch.
        self.arena = Some(Buffer::zeroed(bytes.next_power_of_two())?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ALIGN, Wave};
    use kernels_cuda::channel::{BumpLane, PublishLane, PullLane, SettleLane, Ticket};

    fn ticket(slot: u32) -> Ticket {
        Ticket {
            slot,
            ..Ticket::default()
        }
    }

    #[test]
    fn each_lane_names_its_own_window_of_the_wave_s_ticket_table() {
        let mut wave = Wave::default();
        let first = wave.stage(
            &[ticket(0), ticket(1)],
            &[0],
            &[1],
            PullLane::default(),
            BumpLane::default(),
            PublishLane::default(),
            SettleLane::default(),
        );
        let second = wave.stage(
            &[ticket(7)],
            &[],
            &[7],
            PullLane::default(),
            BumpLane::default(),
            PublishLane::default(),
            SettleLane::default(),
        );
        assert_eq!((first, second), (0, 1));
        assert_eq!((wave.pull[0].ticket_offset, wave.pull[0].ticket_count), (0, 2));
        assert_eq!((wave.pull[1].ticket_offset, wave.pull[1].ticket_count), (2, 1));
        assert_eq!(
            (wave.publish[1].ticket_offset, wave.publish[1].ticket_count),
            (2, 1)
        );
        // The settlement reads the SAME window: it is the same fire's tickets,
        // read after the publication rather than instead of it.
        assert_eq!(
            (wave.settle[1].ticket_offset, wave.settle[1].ticket_count),
            (2, 1)
        );
        assert_eq!(wave.tickets.iter().map(|t| t.slot).collect::<Vec<_>>(), vec![0, 1, 7]);
    }

    #[test]
    fn a_lane_with_no_tickets_still_takes_a_lane() {
        let mut wave = Wave::default();
        wave.stage(
            &[],
            &[],
            &[],
            PullLane::default(),
            BumpLane::default(),
            PublishLane::default(),
            SettleLane::default(),
        );
        // The pull seeds the commit word every stage's early-return reads, so
        // a fire with nothing to validate is still a lane of the launch.
        assert_eq!(wave.staged(), 1);
        assert_eq!(wave.pull[0].ticket_count, 0);
    }

    #[test]
    fn the_six_regions_are_eight_aligned_and_in_order() {
        let mut wave = Wave::default();
        wave.stage(
            &[ticket(0)],
            &[0, 1, 2],
            &[3],
            PullLane::default(),
            BumpLane::default(),
            PublishLane::default(),
            SettleLane::default(),
        );
        let regions = wave.regions();
        for offset in [
            regions.tickets,
            regions.taken,
            regions.put,
            regions.pull,
            regions.bump,
            regions.publish,
            regions.settle,
        ] {
            assert_eq!(offset % ALIGN, 0, "region at {offset} is not {ALIGN}-aligned");
        }
        assert!(regions.tickets < regions.taken);
        assert!(regions.taken < regions.put);
        assert!(regions.put < regions.pull);
        assert!(regions.pull < regions.bump);
        assert!(regions.bump < regions.publish);
        assert!(regions.publish < regions.settle);
        assert!(regions.settle < regions.bytes);
    }

    #[test]
    fn clearing_leaves_nothing_of_the_batch_behind() {
        let mut wave = Wave::default();
        wave.stage(
            &[ticket(0)],
            &[0],
            &[0],
            PullLane::default(),
            BumpLane::default(),
            PublishLane::default(),
            SettleLane::default(),
        );
        wave.clear();
        assert_eq!(wave.staged(), 0);
        assert!(wave.tickets.is_empty());
        assert!(wave.taken.is_empty());
        assert!(wave.put.is_empty());
        assert!(wave.spans.is_empty());
    }
}
