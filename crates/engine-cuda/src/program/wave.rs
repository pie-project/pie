use kernels_cuda::channel::{self, BumpLane, PublishLane, PullLane, SettleLane, Ticket};

use crate::device::{Buffer, Context};
use crate::error::Result;

use super::launch::slice_bytes;

const ALIGN: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
struct Span {
    taken_at: usize,
    taken_len: usize,
    put_at: usize,
    put_len: usize,
}

#[derive(Debug, Default)]
pub struct Wave {
    arena: Option<Buffer>,
    tickets: Vec<Ticket>,
    taken: Vec<u32>,
    put: Vec<u32>,
    pull: Vec<PullLane>,
    bump: Vec<BumpLane>,
    publish: Vec<PublishLane>,
    settle: Vec<SettleLane>,
    spans: Vec<Span>,
    retired: Vec<Buffer>,
}

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

    pub(super) fn fly(&mut self, context: &Context) -> Result<()> {
        let lanes = self.pull.len();
        if lanes == 0 {
            return Ok(());
        }
        let regions = self.regions();
        self.reserve(regions.bytes)?;
        let base = self.arena.as_ref().map_or(0, Buffer::ptr);

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

    fn reserve(&mut self, bytes: usize) -> Result<()> {
        if self
            .arena
            .as_ref()
            .is_some_and(|arena| arena.bytes() >= bytes)
        {
            return Ok(());
        }
        if let Some(outgrown) = self.arena.take() {
            self.retired.push(outgrown);
        }
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
        assert_eq!(
            (wave.pull[0].ticket_offset, wave.pull[0].ticket_count),
            (0, 2)
        );
        assert_eq!(
            (wave.pull[1].ticket_offset, wave.pull[1].ticket_count),
            (2, 1)
        );
        assert_eq!(
            (wave.publish[1].ticket_offset, wave.publish[1].ticket_count),
            (2, 1)
        );
        assert_eq!(
            (wave.settle[1].ticket_offset, wave.settle[1].ticket_count),
            (2, 1)
        );
        assert_eq!(
            wave.tickets.iter().map(|t| t.slot).collect::<Vec<_>>(),
            vec![0, 1, 7]
        );
    }
}
