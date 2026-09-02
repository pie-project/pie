//! One boundary's control plane: batches every lane's pull/bump/publish/
//! settle control structs into one arena, one `cudaMemcpyAsync`, and three
//! launches (`pull_validate`, `commit_bump`, `scatter_publish`) instead of
//! one launch per lane. The arena is rewritten once per batch on the same
//! stream the kernels run on (stream order, no synchronize needed); on
//! growth the outgrown buffer is retired rather than freed, since `cudaFree`
//! is not stream-ordered and a previous batch's kernels may still read it.

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

/// Every fire's control structures in one boundary: accumulated on the host
/// as one lane per staged fire, committed to the device as one image and
/// three launches, and cleared for the next batch. The device buffer
/// outlives the batch; the host vectors do not.
#[derive(Debug, Default)]
pub struct Wave {
    /// The arena, at its high-water mark. `None` until a first batch says how
    /// big a batch is.
    arena: Option<Buffer>,
    /// Every lane's tickets, concatenated; a lane's window is
    /// `[ticket_offset, ticket_offset + ticket_count)`.
    tickets: Vec<Ticket>,
    taken: Vec<u32>,
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
    /// Arenas a growth replaced, kept alive: a previous batch's kernels may
    /// still be reading one, and `cudaFree` is not stream-ordered.
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
    /// Stage one fire's control structures, answering the lane index it
    /// took. Sets `ticket_offset` on `pull`/`publish`/`settle`, since only
    /// the wave knows where the lane's window landed; `bump`'s two list
    /// pointers are filled in at [`Wave::fly`].
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

    /// Is anything staged and not yet flown?
    pub(super) fn staged(&self) -> usize {
        self.pull.len()
    }

    /// Lay the arena out, patch the bump lanes, carry the whole image across
    /// in one `cudaMemcpyAsync`, and launch `channel::pull_validate` once per
    /// lane. Enqueue-only; caller launches its regions after this, then
    /// calls [`Wave::land`].
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

        // bump's list pointers need the arena base, unknown until now
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

        // `image` is copied into the driver's staging buffer before
        // cudaMemcpyAsync returns, so it dying here is safe; only the next
        // batch's copy could race this one, and it's ordered behind it on
        // the same stream.
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

    /// Bump and publish, once for the whole batch, then clear host state.
    /// Must be enqueued after the regions that wrote the cells, on the same
    /// stream, to order payload before tail.
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
        // settlement last: scatter_publish wrote the cells, this advances the
        // tails that announce them; must never be hoisted above the scatter.
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

    /// Grow the arena to hold `bytes`; never shrinks. Grows at a high-water
    /// mark (not per-batch) to keep cudaMalloc/cudaFree off the fire path;
    /// the outgrown buffer is retired rather than freed, since a previous
    /// batch's kernels may still be reading it and `cudaFree` is not
    /// stream-ordered.
    fn reserve(&mut self, bytes: usize) -> Result<()> {
        if self.arena.as_ref().is_some_and(|arena| arena.bytes() >= bytes) {
            return Ok(());
        }
        if let Some(outgrown) = self.arena.take() {
            self.retired.push(outgrown);
        }
        // next_power_of_two: slack so gaining one attachment doesn't reallocate
        self.arena = Some(Buffer::zeroed(bytes.next_power_of_two())?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Wave;
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
        // settlement reads the same window as the fire's tickets
        assert_eq!(
            (wave.settle[1].ticket_offset, wave.settle[1].ticket_count),
            (2, 1)
        );
        assert_eq!(wave.tickets.iter().map(|t| t.slot).collect::<Vec<_>>(), vec![0, 1, 7]);
    }

}
