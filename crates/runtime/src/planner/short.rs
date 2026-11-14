//! What the planner's waiters lack that its own moves cannot give them.
//!
//! The planner funds a waiter by what it may do on its own: commit more of
//! the pool, drop idle cache whose keys say it may go, and move whole
//! processes to the host. Past that, room is the policy's to make: a short
//! waiter stays parked and its planner says, through
//! [`crate::policy::pressure::raise`], how short each tier is. The waiter is
//! released by a discard or new retention keys, by the governor's growth, or,
//! at its deadline, by an explicit error. Nothing here drops anything.
//!
//! Streamed weights go before any kv or state: a device shortfall first
//! sheds the weight cache to its floor ([`WeightShed`]), and the policy
//! hears of it only once a report finds nothing left to shed.

use std::collections::HashMap;
use std::sync::Arc;

use crate::policy::pressure::{self, Tier};
use crate::rt::Instant;

/// Says that `tier` is short of `bytes` in all now (0 once it is not).
pub type Sink = Arc<dyn Fn(Tier, u64) + Send + Sync>;

/// Sheds the engine's weight cache to its floor; the slots it gave back
/// (0 when nothing was above the floor, or the weights do not stream).
pub type WeightShed = Arc<dyn Fn() -> u64 + Send + Sync>;

/// The sink a planner on `engine` raises tier pressure through.
pub fn sink(engine: usize) -> Sink {
    Arc::new(move |tier, bytes| pressure::raise(tier, engine, bytes))
}

/// What one waiter lacks on each tier, since when it has lacked any, and
/// how many times each tier's pressure had been answered with nothing
/// freed then ([`pressure::nothing_freed`]).
#[derive(Debug, Clone, Copy)]
struct Short {
    bytes: [u64; 2],
    since: Instant,
    answered: [u64; 2],
}

/// The shortfalls one planner reports.
#[derive(Default)]
pub(super) struct Shortfall {
    /// By waiter id (the second half of its entry key).
    waiters: HashMap<u64, Short>,
    /// What was last said per tier, so a planner with nothing short stays
    /// quiet.
    said: [u64; 2],
}

impl Shortfall {
    /// Waiter `id` lacks `bytes` (device, host) now. Its clock runs from the
    /// first note that says it lacks anything, and stops when one says not.
    pub(super) fn note(&mut self, id: u64, bytes: [u64; 2]) {
        if bytes == [0, 0] {
            self.waiters.remove(&id);
            return;
        }
        self.waiters
            .entry(id)
            .and_modify(|short| short.bytes = bytes)
            .or_insert(Short {
                bytes,
                since: Instant::now(),
                answered: Tier::ALL.map(pressure::nothing_freed),
            });
    }

    /// Waiter `id` lacks `bytes` on the host, whatever it lacks on the
    /// device.
    pub(super) fn note_host(&mut self, id: u64, bytes: u64) {
        let device = self.waiters.get(&id).map_or(0, |short| short.bytes[0]);
        self.note(id, [device, bytes]);
    }

    /// Nothing is short and nothing was said: there is nothing to tell.
    pub(super) fn quiet(&self) -> bool {
        self.waiters.is_empty() && self.said == [0, 0]
    }

    /// Forgets the waiters that are no longer waiting.
    pub(super) fn retain(&mut self, waiting: impl Fn(u64) -> bool) {
        self.waiters.retain(|&id, _| waiting(id));
    }

    /// Waiter `id` starts over: it fails only on an answer of nothing
    /// after now.
    pub(super) fn restart(&mut self, id: u64) {
        if let Some(short) = self.waiters.get_mut(&id) {
            short.since = Instant::now();
            short.answered = Tier::ALL.map(pressure::nothing_freed);
        }
    }

    /// Whether the policy answered the pressure of a tier waiter `id` is
    /// short on and freed nothing, since it became short (or started over).
    pub(super) fn answered_nothing(&self, id: u64) -> bool {
        self.waiters.get(&id).is_some_and(|short| {
            Tier::ALL.into_iter().any(|tier| {
                short.bytes[tier.index()] > 0
                    && pressure::nothing_freed(tier) > short.answered[tier.index()]
            })
        })
    }

    pub(super) fn since(&self, id: u64) -> Option<Instant> {
        self.waiters.get(&id).map(|short| short.since)
    }

    /// Whether waiter `id` lacks room on the host.
    pub(super) fn host_short(&self, id: u64) -> bool {
        self.waiters
            .get(&id)
            .is_some_and(|short| short.bytes[Tier::Host.index()] > 0)
    }

    /// What waiter `id` lacks most of: the tier and its bytes.
    pub(super) fn worst(&self, id: u64) -> Option<(Tier, u64)> {
        let short = self.waiters.get(&id)?;
        Tier::ALL
            .into_iter()
            .map(|tier| (tier, short.bytes[tier.index()]))
            .max_by_key(|&(tier, bytes)| (bytes, std::cmp::Reverse(tier)))
    }

    /// What each tier is short of in all.
    pub(super) fn sums(&self) -> [u64; 2] {
        let mut sums = [0u64; 2];
        for short in self.waiters.values() {
            for (sum, bytes) in sums.iter_mut().zip(short.bytes) {
                *sum = sum.saturating_add(bytes);
            }
        }
        sums
    }

    /// The tiers to tell, with what to tell them: each one short now, and
    /// one that was and no longer is.
    pub(super) fn take_said(&mut self) -> Vec<(Tier, u64)> {
        let sums = self.sums();
        Tier::ALL
            .into_iter()
            .filter_map(|tier| {
                let (now, said) = (sums[tier.index()], self.said[tier.index()]);
                self.said[tier.index()] = now;
                (now > 0 || said > 0).then_some((tier, now))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short waiter fails on the policy's answer of nothing for a tier it
    /// is short on, after it became short, not on a clock; starting over
    /// waits for the next such answer.
    #[test]
    fn a_short_waiter_fails_on_an_answer_of_nothing_not_on_a_clock() {
        let mut shortfall = Shortfall::default();
        shortfall.note(7, [1 << 20, 0]);
        assert!(!shortfall.answered_nothing(7), "nothing answered yet");
        pressure::answered_nothing(Tier::Device);
        assert!(shortfall.answered_nothing(7), "answered with nothing freed");
        shortfall.restart(7);
        assert!(!shortfall.answered_nothing(7), "started over");
        // An answer on a tier it is not short on is not its.
        pressure::answered_nothing(Tier::Host);
        assert!(!shortfall.answered_nothing(7));
    }
}
