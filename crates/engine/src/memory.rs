//! A device's memory as one ledger.
//!
//! Every byte an engine puts on its device belongs to one [`Class`] and is
//! entered here under the owner that holds it. The ledger keeps one
//! inequality, in bytes of whole chunks:
//!
//! ```text
//! model + state + fire(rung) + outside <= capacity
//! ```
//!
//! - `capacity` is what the device has for this process: measured, never a
//!   share of the card, and capped only where the operator states a budget.
//! - Nothing here decides what stays: the governor decides how much, the
//!   policy what, the pool how. The ledger is what they read.
//!
//! An engine realises the ledger as one address range where its device
//! allows, as several buffers under one budget where a buffer is bounded,
//! or by measurement where it cannot lay a class out itself.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// What a device byte is for, by how long it lives and what sizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Class {
    /// Load to unload, sized by the model: weights and what is made of
    /// them, compiled bodies.
    Model,
    /// Kept between fires, sized by demand: kv, recurrent state, channels.
    State,
    /// One fire, or the frames in flight, sized by the rung: activations,
    /// kernel workspaces, staged inputs, readout rows.
    Fire,
    /// What the driver and its libraries take beside what the engine lays
    /// out, priced by measuring it.
    Outside,
}

impl Class {
    pub const ALL: [Class; 4] = [Class::Model, Class::State, Class::Fire, Class::Outside];

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Class::Model => "model",
            Class::State => "state",
            Class::Fire => "fire",
            Class::Outside => "outside",
        }
    }
}

/// The ledger is over: what is entered against what the device has.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Over {
    pub held: u64,
    pub capacity: u64,
}

impl fmt::Display for Over {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the device holds {} MiB of {} MiB it has for this process",
            self.held >> 20,
            self.capacity >> 20
        )
    }
}

/// What a device reports against what the ledger holds for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Audit {
    pub entered: u64,
    pub device: u64,
}

impl Audit {
    /// Device bytes no owner entered (positive) or entered bytes the device
    /// does not hold (negative).
    #[must_use]
    pub fn gap(&self) -> i128 {
        i128::from(self.device) - i128::from(self.entered)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ledger {
    chunk_bytes: u64,
    capacity: u64,
    owners: BTreeMap<(Class, &'static str), u64>,
}

impl Ledger {
    /// A ledger of chunks of `chunk_bytes` over a device with `capacity`
    /// bytes for this process.
    #[must_use]
    pub fn new(chunk_bytes: u64, capacity: u64) -> Ledger {
        Ledger {
            chunk_bytes: chunk_bytes.max(1),
            capacity,
            owners: BTreeMap::new(),
        }
    }

    #[must_use]
    pub const fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    #[must_use]
    pub const fn capacity(&self) -> u64 {
        self.capacity
    }

    /// What the device has now: a governor's target moves it, a measured
    /// `outside` does not (that is an entry).
    pub fn set_capacity(&mut self, capacity: u64) {
        self.capacity = capacity;
    }

    /// `owner` holds `bytes` of `class` now, whatever it held before.
    pub fn enter(&mut self, class: Class, owner: &'static str, bytes: u64) {
        if bytes == 0 {
            self.owners.remove(&(class, owner));
        } else {
            self.owners.insert((class, owner), bytes);
        }
    }

    #[must_use]
    pub fn of(&self, class: Class, owner: &'static str) -> u64 {
        self.owners.get(&(class, owner)).copied().unwrap_or(0)
    }

    /// The bytes entered for `class`.
    #[must_use]
    pub fn bytes(&self, class: Class) -> u64 {
        self.owners
            .iter()
            .filter(|((c, _), _)| *c == class)
            .fold(0u64, |sum, (_, bytes)| sum.saturating_add(*bytes))
    }

    /// The whole chunks `class` takes.
    #[must_use]
    pub fn chunks(&self, class: Class) -> u64 {
        self.bytes(class).div_ceil(self.chunk_bytes)
    }

    /// Every class, in whole chunks' bytes.
    #[must_use]
    pub fn held(&self) -> u64 {
        Class::ALL.iter().fold(0u64, |sum, class| {
            sum.saturating_add(self.chunks(*class).saturating_mul(self.chunk_bytes))
        })
    }

    /// What no class has taken.
    #[must_use]
    pub fn room(&self) -> u64 {
        self.capacity.saturating_sub(self.held())
    }

    /// The one inequality.
    pub fn check(&self) -> Result<(), Over> {
        let held = self.held();
        if held > self.capacity {
            return Err(Over {
                held,
                capacity: self.capacity,
            });
        }
        Ok(())
    }

    /// The largest rung, a power of two of tokens from `least` up, whose
    /// fire fits beside everything entered but the fire class and
    /// `floor_state` more bytes of state (one working set at the declared
    /// context): the top of the ladder, derived. `fire_at` is a rung's
    /// whole footprint and only grows with it. `None`: not even `least`
    /// fits, and the load is refused.
    #[must_use]
    pub fn largest_rung(
        &self,
        least: u32,
        floor_state: u64,
        fire_at: impl Fn(u32) -> u64,
    ) -> Option<u32> {
        let chunked = |bytes: u64| {
            bytes
                .div_ceil(self.chunk_bytes)
                .saturating_mul(self.chunk_bytes)
        };
        let beside = Class::ALL
            .iter()
            .filter(|class| **class != Class::Fire)
            .fold(0u64, |sum, class| {
                sum.saturating_add(chunked(self.bytes(*class)))
            })
            .saturating_add(chunked(floor_state));
        let fits = |rung: u32| beside.saturating_add(chunked(fire_at(rung))) <= self.capacity;
        let mut rung = least.max(1).next_power_of_two();
        if !fits(rung) {
            return None;
        }
        while let Some(next) = rung.checked_mul(2) {
            if !fits(next) {
                break;
            }
            rung = next;
        }
        Some(rung)
    }

    /// The ledger against what the device reports as in use.
    #[must_use]
    pub fn audit(&self, device: u64) -> Audit {
        Audit {
            entered: Class::ALL
                .iter()
                .fold(0u64, |sum, class| sum.saturating_add(self.bytes(*class))),
            device,
        }
    }

    /// Every entry, largest first within its class.
    pub fn entries(&self) -> impl Iterator<Item = (Class, &'static str, u64)> + '_ {
        let mut entries: Vec<_> = self
            .owners
            .iter()
            .map(|((class, owner), bytes)| (*class, *owner, *bytes))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(b.2.cmp(&a.2)).then(a.1.cmp(b.1)));
        entries.into_iter()
    }
}

/// The rungs of a ladder whose top is `largest`: every power of two of
/// tokens from `least` up to it.
#[must_use]
pub fn rungs(least: u32, largest: u32) -> Vec<u32> {
    let mut rungs = Vec::new();
    let mut rung = least.max(1).next_power_of_two();
    while rung <= largest {
        rungs.push(rung);
        match rung.checked_mul(2) {
            Some(next) => rung = next,
            None => break,
        }
    }
    rungs
}

impl fmt::Display for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for class in Class::ALL {
            write!(f, "{} {} MiB", class.name(), self.bytes(class) >> 20)?;
            let mut first = true;
            for (_, owner, bytes) in self.entries().filter(|entry| entry.0 == class) {
                write!(
                    f,
                    "{}{owner} {}",
                    if first { " (" } else { ", " },
                    bytes >> 20
                )?;
                first = false;
            }
            f.write_str(if first { "; " } else { "); " })?;
        }
        write!(f, "{} of {} MiB", self.held() >> 20, self.capacity >> 20)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn the_ledger_holds_whole_chunks_and_refuses_past_capacity() {
        let mut ledger = Ledger::new(16 * MIB, 100 * MIB);
        ledger.enter(Class::Model, "weights", 20 * MIB + 1);
        ledger.enter(Class::Fire, "arena", 30 * MIB);
        assert_eq!(ledger.chunks(Class::Model), 2);
        assert_eq!(ledger.held(), (32 + 32) * MIB);
        assert_eq!(ledger.room(), 36 * MIB);
        assert_eq!(ledger.check(), Ok(()));
        ledger.enter(Class::State, "kv", 40 * MIB);
        assert_eq!(
            ledger.check(),
            Err(Over {
                held: 112 * MIB,
                capacity: 100 * MIB
            })
        );
        ledger.enter(Class::State, "kv", 0);
        assert_eq!(ledger.of(Class::State, "kv"), 0);
        assert_eq!(ledger.check(), Ok(()));
    }

    #[test]
    fn the_top_rung_is_the_largest_power_of_two_that_fits_beside_the_floor() {
        let mut ledger = Ledger::new(MIB, 1000 * MIB);
        ledger.enter(Class::Model, "weights", 400 * MIB);
        ledger.enter(Class::Outside, "bodies", 50 * MIB);
        // A fire entered at some rung is not what the next rung stands beside.
        ledger.enter(Class::Fire, "arena", 300 * MIB);
        let fire_at = |rung: u32| u64::from(rung) * MIB / 16;
        // 400 + 50 + 100 of floor state leave 450: 4096 tokens take 256, 8192 take 512.
        assert_eq!(ledger.largest_rung(256, 100 * MIB, fire_at), Some(4096));
        assert_eq!(ledger.largest_rung(256, 540 * MIB, fire_at), None);
        assert_eq!(rungs(256, 4096), vec![256, 512, 1024, 2048, 4096]);
        assert_eq!(rungs(300, 4096), vec![512, 1024, 2048, 4096]);
    }

    #[test]
    fn an_audit_states_what_no_owner_entered() {
        let mut ledger = Ledger::new(MIB, 100 * MIB);
        ledger.enter(Class::Model, "weights", 10 * MIB);
        ledger.enter(Class::Fire, "inputs", 3 * MIB);
        assert_eq!(ledger.audit(15 * MIB).gap(), i128::from(2 * MIB));
        assert_eq!(
            ledger.to_string(),
            "model 10 MiB (weights 10); state 0 MiB; fire 3 MiB (inputs 3); outside 0 MiB; 13 of 100 MiB"
        );
    }
}
