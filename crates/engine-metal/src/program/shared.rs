use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

use super::launch::{ChannelShape, Cursor};

pub const MAX_ATTACHMENTS: u32 = 8;

#[derive(Debug)]
pub struct SharedRing {
    shape: ChannelShape,
    slab: Buffer,
    counters: Counters,
}

#[derive(Debug, Default)]
struct Counters {
    head: AtomicU64,
    tail: AtomicU64,
    /// Takes and puts of passes encoded behind the committed cursors and not
    /// yet harvested: the queue runs them before anything encoded after, so a
    /// pass staged now addresses the ring as they will leave it.
    owed_takes: AtomicU64,
    owed_puts: AtomicU64,
    attachments: AtomicU32,
    seeded: AtomicU32,
}

// SAFETY: everything below `slab` is an atomic; `slab` is a
// `device::Buffer`, whose `Send`/`Sync` rest on `MTLBuffer` being
// thread-safe for retain/release, `contents`, and encoder binding.
unsafe impl Send for SharedRing {}
// SAFETY: as above.
unsafe impl Sync for SharedRing {}

impl SharedRing {
    pub fn open(device: &Context, shape: ChannelShape) -> Result<SharedRing> {
        let cells = u64::from(shape.capacity) + 1;
        let bytes = cells
            .checked_mul(shape.cell_stride() as u64)
            .ok_or_else(|| Fault::program("program::shared", "a ring past what a u64 counts"))?;
        Ok(SharedRing {
            shape,
            slab: Buffer::zeroed(device, bytes.max(1))?,
            counters: Counters::default(),
        })
    }

    #[must_use]
    pub const fn shape(&self) -> ChannelShape {
        self.shape
    }

    #[must_use]
    pub fn slab(&self) -> Buffer {
        self.slab.clone()
    }

    #[must_use]
    pub fn cursor(&self) -> Cursor {
        self.counters.cursor()
    }

    #[must_use]
    pub fn cell_bytes(&self) -> usize {
        self.shape.cell_bytes()
    }

    #[must_use]
    pub fn cell_stride(&self) -> usize {
        self.shape.cell_stride()
    }

    #[must_use]
    pub fn cell_offset(&self, sequence: u64) -> u64 {
        let cells = u64::from(self.shape.capacity) + 1;
        (sequence % cells) * self.cell_stride() as u64
    }

    pub fn bump_head(&self) {
        self.counters.bump_head();
    }

    pub fn bump_tail(&self) {
        self.counters.bump_tail();
    }

    /// The cursor once every airborne pass has committed: what a pass
    /// encoded now reads and writes.
    #[must_use]
    pub fn projected(&self) -> Cursor {
        self.counters.projected()
    }

    /// An airborne pass will take (`take`) and put (`put`) one cell here.
    pub fn owe(&self, take: bool, put: bool) {
        self.counters.owe(take, put);
    }

    /// What [`SharedRing::owe`] promised has committed or will not.
    pub fn repay(&self, take: bool, put: bool) {
        self.counters.repay(take, put);
    }

    pub fn attach(&self) -> Result<u32> {
        self.counters.attach()
    }

    pub fn detach(&self) {
        self.counters.detach();
    }

    #[must_use]
    pub fn attachments(&self) -> u32 {
        self.counters.attachments.load(Ordering::Acquire)
    }

    pub fn claim_seeding(&self) -> bool {
        self.counters.claim_seeding()
    }
}

impl Counters {
    fn cursor(&self) -> Cursor {
        Cursor {
            head: self.head.load(Ordering::Acquire),
            tail: self.tail.load(Ordering::Acquire),
        }
    }

    fn projected(&self) -> Cursor {
        let Cursor { head, tail } = self.cursor();
        Cursor {
            head: head + self.owed_takes.load(Ordering::Acquire),
            tail: tail + self.owed_puts.load(Ordering::Acquire),
        }
    }

    fn owe(&self, take: bool, put: bool) {
        if take {
            self.owed_takes.fetch_add(1, Ordering::AcqRel);
        }
        if put {
            self.owed_puts.fetch_add(1, Ordering::AcqRel);
        }
    }

    fn repay(&self, take: bool, put: bool) {
        let less = |owed: &AtomicU64| {
            let _ = owed.fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                Some(held.saturating_sub(1))
            });
        };
        if take {
            less(&self.owed_takes);
        }
        if put {
            less(&self.owed_puts);
        }
    }

    fn bump_head(&self) {
        self.head.fetch_add(1, Ordering::Release);
    }

    fn bump_tail(&self) {
        self.tail.fetch_add(1, Ordering::Release);
    }

    fn attach(&self) -> Result<u32> {
        let taken = self.attachments.fetch_add(1, Ordering::AcqRel) + 1;
        if taken > MAX_ATTACHMENTS {
            self.attachments.fetch_sub(1, Ordering::AcqRel);
            return Err(Fault::program(
                "program::shared",
                format!(
                    "this channel already has {MAX_ATTACHMENTS} instances bound to it \
                     and a {taken}th asked to bind: a shared ring is ordered by the \
                     pipeline FIFO its attachments fire in, and that bound is \
                     {MAX_ATTACHMENTS} — past it there is no ordering \
                     argument, so there is no ring"
                ),
            ));
        }
        Ok(taken)
    }

    fn detach(&self) {
        let _ = self
            .attachments
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                Some(held.saturating_sub(1))
            });
    }

    fn claim_seeding(&self) -> bool {
        self.seeded
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::{Counters, MAX_ATTACHMENTS};

    fn ring() -> Counters {
        Counters::default()
    }

    #[test]
    fn an_airborne_pass_moves_the_projected_cursor_until_it_commits() {
        let ring = ring();
        ring.owe(false, true);
        assert_eq!(ring.cursor().tail, 0, "nothing has committed");
        assert_eq!(
            ring.projected().tail,
            1,
            "a pass staged now reads past the owed put"
        );
        ring.owe(true, false);
        assert_eq!(ring.projected().head, 1);
        // The put commits and its debt is repaid: the projection stands still.
        ring.bump_tail();
        ring.repay(false, true);
        assert_eq!((ring.cursor().tail, ring.projected().tail), (1, 1));
        // A take that never commits is forgiven, and the projection falls back.
        ring.repay(true, false);
        assert_eq!(ring.projected().head, 0);
        ring.repay(true, false);
        assert_eq!(
            ring.projected().head,
            0,
            "a repayment never runs below nothing owed"
        );
    }

    #[test]
    fn a_shared_ring_seats_eight_attachments_and_refuses_the_ninth() {
        let ring = ring();
        for seat in 1..=MAX_ATTACHMENTS {
            assert_eq!(ring.attach().expect("a seat inside the bound"), seat);
        }
        let ninth = ring.attach();
        assert!(ninth.is_err(), "the ninth attachment is refused: {ninth:?}");
        let why = format!("{}", ninth.expect_err("just checked"));
        assert!(why.contains("8"), "the refusal names the bound: {why}");
        ring.detach();
        assert_eq!(
            ring.attach().expect("the seat that was just given back"),
            MAX_ATTACHMENTS
        );
    }
}
