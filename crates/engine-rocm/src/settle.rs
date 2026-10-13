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

    pub fn claim(&self) -> Result<u32> {
        self.free.take().ok_or(crate::error::Fault::Ceiling {
            what: "settlement events (one per in-flight step)",
            need: self.events.len() as u64 + 1,
            have: self.events.len() as u64,
        })
    }

    #[must_use]
    pub fn event(&self, at: u32) -> &Event {
        &self.events[at as usize]
    }

    #[must_use]
    pub fn recycler(&self) -> Arc<Free> {
        Arc::clone(&self.free)
    }
}
