use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    #[must_use]
    pub fn new() -> Airborne {
        Airborne::default()
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
        self.counts.settled.load(Ordering::Acquire) > seq
    }
}

pub struct Done {
    pub at: engine::StepDone,

    pub sink: engine::CompletionSink,
}

impl std::fmt::Debug for Done {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Done").field("at", &self.at).finish()
    }
}

#[derive(Debug, Clone)]
pub struct Arms {
    taken: Vec<bool>,
}

impl Arms {
    #[must_use]
    pub fn of(depth: usize) -> Arms {
        Arms {
            taken: vec![false; depth.max(1)],
        }
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.taken.len()
    }

    #[must_use]
    pub fn free(&self) -> Option<usize> {
        self.taken.iter().position(|held| !held)
    }

    pub fn take(&mut self, at: usize) {
        if let Some(held) = self.taken.get_mut(at) {
            *held = true;
        }
    }

    pub fn give(&mut self, at: usize) {
        if let Some(held) = self.taken.get_mut(at) {
            *held = false;
        }
    }

    #[must_use]
    pub fn held(&self) -> usize {
        self.taken.iter().filter(|held| **held).count()
    }
}
