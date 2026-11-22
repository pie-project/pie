use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::graph::Event;
use crate::error::Result;
use crate::inputs::Free;

#[derive(Debug, Clone, Default)]
pub struct Airborne {
    counts: Arc<Counts>,
}

#[derive(Debug, Default)]
struct Counts {
    issued: AtomicU64,
    settled: AtomicU64,
}

impl Airborne {
    pub const NEVER: u64 = u64::MAX;

    #[must_use]
    pub fn new() -> Airborne {
        Airborne::default()
    }

    #[must_use]
    pub fn next_seq(&self) -> u64 {
        self.counts.issued.load(Ordering::Acquire)
    }

    pub fn enter(&self) -> u64 {
        self.counts.issued.fetch_add(1, Ordering::AcqRel)
    }

    pub fn leave(&self) {
        self.counts.settled.fetch_add(1, Ordering::Release);
    }

    pub fn abandon(&self) {
        self.leave();
    }

    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .issued
            .load(Ordering::Acquire)
            .saturating_sub(self.counts.settled.load(Ordering::Acquire))
    }

    #[must_use]
    pub fn settled_past(&self, seq: u64) -> bool {
        seq == Airborne::NEVER || self.counts.settled.load(Ordering::Acquire) > seq
    }
}

#[derive(Debug)]
pub struct Settlement {
    events: Vec<Event>,
    free: Arc<Free>,
}

impl Settlement {
    pub fn open(depth: usize) -> Result<Settlement> {
        let mut events = Vec::with_capacity(depth);
        for _ in 0..depth {
            events.push(Event::new()?);
        }
        Ok(Settlement {
            events,
            free: Free::of(depth),
        })
    }

    /// A settle event, as [`Free::claim`] claims: waited for while a step's
    /// landing holds one, refused by name at once where the claimant's own
    /// steps hold them all.
    pub fn claim(&self) -> Result<crate::inputs::SlotGuard> {
        self.free.claim(
            "settlement events (one per in-flight step; every one held by a step not yet \
             handed to its landing)",
        )
    }

    #[must_use]
    pub fn event(&self, at: u32) -> &Event {
        &self.events[at as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::Settlement;
    use std::time::{Duration, Instant};

    /// A settle event is claimed as a staging slot is: with one held by a
    /// step's landing, a claim waits for it and then has it; with every one
    /// held by the claimant's own steps, none landing, it is refused by name
    /// at once.
    #[test]
    fn a_settle_event_waits_for_a_landing_and_refuses_a_claim_deeper_than_its_set() {
        if !crate::device::present() {
            eprintln!("skipping: no CUDA device on this machine");
            return;
        }
        let settlement = Settlement::open(2).expect("two settle events");
        let first = settlement.claim().expect("a free event");
        let landing = settlement.claim().expect("a free event").landing();
        let at = landing.at();
        let lands = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(landing);
        });
        let began = Instant::now();
        let got = settlement.claim().expect("the landing's event comes back");
        assert!(began.elapsed() >= Duration::from_millis(150), "it waited for the landing");
        assert_eq!(got.at(), at, "the landing's event");
        lands.join().expect("the landing thread");
        let began = Instant::now();
        let refused = settlement.claim().expect_err("a third event of two, none landing");
        assert!(began.elapsed() < Duration::from_millis(100), "refused at once");
        assert!(refused.to_string().contains("settlement events"), "{refused}");
        drop((first, got));
    }
}
