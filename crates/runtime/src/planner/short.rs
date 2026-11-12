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

/// What one waiter lacks on each tier, and since when it has lacked any.
#[derive(Debug, Clone, Copy)]
struct Short {
    bytes: [u64; 2],
    since: Instant,
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

    /// Since when waiter `id` has been short, if it is.
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

    #[test]
    fn a_shortfall_is_told_while_it_stands_and_once_as_it_clears() {
        let mut short = Shortfall::default();
        assert!(short.take_said().is_empty(), "nothing short, nothing said");
        short.note(1, [100, 0]);
        short.note(2, [20, 5]);
        assert_eq!(short.take_said(), [(Tier::Device, 120), (Tier::Host, 5)]);
        let since = short.since(1).unwrap();
        short.note(1, [50, 0]);
        assert_eq!(short.since(1), Some(since), "its clock runs on");
        assert_eq!(short.take_said(), [(Tier::Device, 70), (Tier::Host, 5)]);
        short.retain(|id| id == 1);
        assert_eq!(short.take_said(), [(Tier::Device, 50), (Tier::Host, 0)]);
        assert_eq!(
            short.take_said(),
            [(Tier::Device, 50)],
            "host said clear once"
        );
        short.note(1, [0, 0]);
        assert_eq!(short.since(1), None);
        assert_eq!(short.take_said(), [(Tier::Device, 0)]);
        assert!(short.take_said().is_empty());
    }
}
