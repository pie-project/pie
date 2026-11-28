//! Which chunks may open, the reservations and claims that hold chunks back
//! for a ticket, and the room counters a waiter reads.

use super::*;

impl SlotPool {
    /// Whether physical chunk `chunk` may be opened: free, in a committed
    /// segment, below the activation and not fenced.
    pub(super) fn openable(&self, chunk: u32) -> bool {
        self.free.get(chunk)
            && !self.fenced.get(chunk)
            && chunk < self.state_limit()
            && chunk < self.ceiling
            && self.segments.state(self.segments.of(chunk)) == SegmentState::Committed
            && !self.held_for.values().any(|held| held.contains(&chunk))
    }

    /// No chunk at or past physical id `ceiling` is opened from now on, the
    /// engine binding none there yet (while it holds back part of the range
    /// it stated, [`::engine::chunk::ChunkFacts::held_bytes`]); chunks below it open as
    /// before. A chunk already open past it stays its holder's.
    pub fn set_ceiling(&mut self, ceiling: u32) {
        let ceiling = ceiling.min(self.total);
        let (from, to) = (ceiling.min(self.ceiling), ceiling.max(self.ceiling));
        self.ceiling = ceiling;
        for chunk in from..to {
            self.reopen(chunk);
        }
    }

    /// See [`Self::set_ceiling`].
    pub fn ceiling(&self) -> u32 {
        self.ceiling
    }

    /// Whether `segment` may be committed: it lies wholly under the
    /// ceiling (all of it, with no ceiling). One that straddles it would map
    /// chunks the engine holds back, past the pool's hard limit.
    pub fn backs(&self, segment: u32) -> bool {
        self.ceiling >= self.total || self.segments.range(segment).end <= self.ceiling
    }

    /// Brings `chunk`'s place in [`Self::open`] up to date.
    pub(super) fn reopen(&mut self, chunk: u32) {
        if self.openable(chunk) {
            if !self.open.get(chunk) {
                self.open.set(chunk);
                self.given.give();
            }
        } else {
            self.open.clear(chunk);
        }
    }

    /// How many times room was given back (a chunk opened for state) since
    /// the pool was made.
    #[must_use]
    pub fn room_gives(&self) -> u64 {
        self.given.gives.load(std::sync::atomic::Ordering::Acquire)
    }

    /// How many times slots were taken or given back since the pool was
    /// made: unmoved, every placement read on the pool reads the same.
    #[must_use]
    pub fn room_changes(&self) -> u64 {
        self.changes
    }

    /// Holds free physical `chunks` for one claimant: closed to every
    /// other take until [`Self::release_reservation`], opened to its own
    /// inside [`Self::with_claim`]. Chunks not free now are left out. Its
    /// ticket.
    pub fn reserve_chunks(&mut self, chunks: impl IntoIterator<Item = u32>) -> u64 {
        self.next_ticket += 1;
        let ticket = self.next_ticket;
        let held: BTreeSet<u32> = chunks
            .into_iter()
            .filter(|&chunk| chunk < self.total && self.open.get(chunk))
            .collect();
        for &chunk in &held {
            self.open.clear(chunk);
        }
        self.changes += 1;
        self.held_for.insert(ticket, held);
        ticket
    }

    /// Whether a reservation holds a chunk in `start..end`.
    pub(super) fn reserved_in(&self, start: u32, end: u32) -> bool {
        self.held_for
            .values()
            .any(|held| held.range(start..end).next().is_some())
    }

    /// The chunks `ticket` still holds.
    #[must_use]
    pub fn reserved(&self, ticket: u64) -> Vec<u32> {
        self.held_for
            .get(&ticket)
            .map(|held| held.iter().copied().collect())
            .unwrap_or_default()
    }

    /// `ticket`'s chunks still free go back to every take.
    pub fn release_reservation(&mut self, ticket: u64) {
        let Some(held) = self.held_for.remove(&ticket) else {
            return;
        };
        for chunk in held {
            self.reopen(chunk);
        }
        self.changes += 1;
        self.slots_freed();
    }

    /// `f` run with `ticket`'s held chunks open to it: what it takes of
    /// them leaves the reservation, what it leaves stays held.
    pub fn with_claim<R>(&mut self, ticket: Option<u64>, f: impl FnOnce(&mut Self) -> R) -> R {
        let opened = self.claim_open(ticket);
        let out = f(self);
        self.claim_close(opened);
        out
    }

    /// Opens `ticket`'s held chunks to the takes that follow, until
    /// [`Self::claim_close`]: what [`Self::with_claim`] does around its
    /// takes, for a caller whose takes go through another owner's methods.
    pub fn claim_open(&mut self, ticket: Option<u64>) -> Option<(u64, BTreeSet<u32>)> {
        let ticket = ticket?;
        let held = self.held_for.remove(&ticket)?;
        for &chunk in &held {
            if self.openable(chunk) {
                self.open.set(chunk);
            }
        }
        Some((ticket, held))
    }

    /// Closes what [`Self::claim_open`] opened: the chunks taken leave the
    /// reservation, the rest stay held.
    pub fn claim_close(&mut self, opened: Option<(u64, BTreeSet<u32>)>) {
        let Some((ticket, held)) = opened else {
            return;
        };
        let mut kept = BTreeSet::new();
        for chunk in held {
            if self.open.get(chunk) {
                self.open.clear(chunk);
                kept.insert(chunk);
            }
        }
        self.held_for.insert(ticket, kept);
    }
}
