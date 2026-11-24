use kernels_cuda::runtime::channel::{self, BumpLane, PublishLane, PullLane, SettleLane, Ticket};

use crate::device::Context;
use crate::error::Result;

use super::launch::slice_bytes;

const ALIGN: usize = 8;

/// A lane's live commit words: the verdict and the word beside it the pull
/// clears.
const COMMIT_BYTES: usize = 2 * size_of::<u32>();

#[derive(Clone, Copy, Debug, Default)]
struct Span {
    taken_at: usize,
    taken_len: usize,
    put_at: usize,
    put_len: usize,
}

#[derive(Debug, Default)]
pub struct Wave {
    /// Where the wave's tables lie in a carried pair's state, laid at each
    /// fly.
    arena: u64,
    tickets: Vec<Ticket>,
    taken: Vec<u32>,
    put: Vec<u32>,
    pull: Vec<PullLane>,
    bump: Vec<BumpLane>,
    publish: Vec<PublishLane>,
    settle: Vec<SettleLane>,
    spans: Vec<Span>,
}

#[derive(Clone, Copy, Debug, Default)]
struct Regions {
    /// Each lane's live commit words (two `u32`s a lane, as the pull writes).
    commits: usize,
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
    #[allow(clippy::too_many_arguments)]
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

    pub(super) fn staged(&self) -> usize {
        self.pull.len()
    }

    /// Where lane `lane`'s live commit word lies, once the wave has flown.
    pub(super) fn commit_at(&self, lane: usize) -> u64 {
        self.arena + (self.regions().commits + lane * COMMIT_BYTES) as u64
    }

    /// Stages the wave's tables at `base`, in a carried pair's state.
    pub(super) fn fly(&mut self, context: &Context, base: u64) -> Result<()> {
        let lanes = self.pull.len();
        if lanes == 0 {
            return Ok(());
        }
        let regions = self.regions();
        self.arena = base;

        // Each pass's verdict lives in the wave's own device tables, where
        // every region reads it without a trip to the host; the settle hands
        // it to the host word each lane was staged with.
        for lane in 0..lanes {
            let live = base + (regions.commits + lane * COMMIT_BYTES) as u64;
            let host = self.settle[lane].commit;
            self.pull[lane].pass_commit = live;
            self.bump[lane].commit = live;
            self.publish[lane].commit = live;
            self.settle[lane].commit = live;
            self.settle[lane].commit_host = host;
        }

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

        let stream = context.stream();
        crate::device::stage_raw(stream, base, &image)?;
        let lane_count = u32::try_from(lanes).unwrap_or(u32::MAX);
        channel::pull_validate(
            context.ctx(),
            base + regions.tickets as u64,
            base + regions.pull as u64,
            lane_count,
        )?;
        Ok(())
    }

    pub(super) fn land(&mut self, context: &Context) -> Result<()> {
        let lanes = self.pull.len();
        if lanes == 0 {
            return Ok(());
        }
        let regions = self.regions();
        let base = self.arena;
        let lane_count = u32::try_from(lanes).unwrap_or(u32::MAX);
        channel::commit_bump(context.ctx(), base + regions.bump as u64, lane_count)?;
        channel::scatter_publish(
            context.ctx(),
            base + regions.tickets as u64,
            base + regions.publish as u64,
            lane_count,
        )?;
        channel::settle(
            context.ctx(),
            base + regions.tickets as u64,
            base + regions.settle as u64,
            lane_count,
        )?;
        self.clear();
        Ok(())
    }

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

    fn regions(&self) -> Regions {
        let mut at = 0usize;
        let mut place = |bytes: usize| {
            let offset = at;
            at += bytes.div_ceil(ALIGN) * ALIGN;
            offset
        };
        let commits = place(self.pull.len() * COMMIT_BYTES);
        let tickets = place(self.tickets.len() * size_of::<Ticket>());
        let taken = place(self.taken.len() * size_of::<u32>());
        let put = place(self.put.len() * size_of::<u32>());
        let pull = place(self.pull.len() * size_of::<PullLane>());
        let bump = place(self.bump.len() * size_of::<BumpLane>());
        let publish = place(self.publish.len() * size_of::<PublishLane>());
        let settle = place(self.settle.len() * size_of::<SettleLane>());
        Regions {
            commits,
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

    /// The bytes the staged wave's tables take.
    pub(super) fn bytes(&self) -> u64 {
        if self.pull.is_empty() {
            0
        } else {
            self.regions().bytes as u64
        }
    }

    /// The most a lane that names `channels` channels adds to a wave's
    /// tables, and what the tables take whatever their lanes: the room
    /// each of them is aligned in.
    pub(super) fn lane_bytes(channels: usize) -> (u64, u64) {
        let lane = channels * (size_of::<Ticket>() + 2 * size_of::<u32>())
            + COMMIT_BYTES
            + size_of::<PullLane>()
            + size_of::<BumpLane>()
            + size_of::<PublishLane>()
            + size_of::<SettleLane>();
        (lane as u64, (9 * ALIGN) as u64)
    }
}
