//! Adapter slots: which slot of the registered banks holds which bundle. A
//! bundle (`engine::bundle`) is keyed by the digest of its bytes, so every
//! pass that attaches the same bytes shares one slot and writes it once; a
//! slot no pass holds keeps its bundle until another bundle needs the room.

use std::collections::BTreeMap;

use crate::error::{Fault, Result};
use crate::weights::BankSeat;

/// This load's banks as `engine::bundle` fills them.
#[must_use]
pub fn banks(seats: &[BankSeat]) -> Vec<engine::bundle::Bank<'_>> {
    seats
        .iter()
        .map(|seat| engine::bundle::Bank {
            name: &seat.name,
            rows: seat.rows,
            cols: seat.cols,
            slot: seat.slot,
            dtype: seat.dtype,
        })
        .collect()
}

/// A pass's hold on the slot its bundle is in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub slot: u32,
    /// Whether this bind wrote the slot.
    pub landed: bool,
    pub key: Key,
}

/// A bundle, by the blake3 digest of its bytes (`engine::bundle::Bundle::key`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key(pub [u8; 32]);

/// The slots of a load's banks, by the bundle each holds.
#[derive(Debug, Default)]
pub struct Slots {
    seats: u32,
    held: BTreeMap<Key, (u32, u32)>,
    /// The slots no bind holds that still keep a bundle, the oldest release
    /// first: a bind of the same bytes takes its slot back unwritten, and a
    /// new bundle takes a free slot before it takes one of these.
    kept: Vec<(Key, u32)>,
}

/// The slot an acquire gives, and whether it must be written (`fresh`).
#[derive(Debug, Clone, Copy)]
pub struct Grant {
    pub slot: u32,
    pub fresh: bool,
}

impl Slots {
    #[must_use]
    pub fn new(seats: u32) -> Slots {
        Slots {
            seats,
            held: BTreeMap::new(),
            kept: Vec::new(),
        }
    }

    #[must_use]
    pub fn seats(&self) -> u32 {
        self.seats
    }

    #[must_use]
    pub fn live(&self) -> usize {
        self.held.len()
    }

    /// A hold on the slot of `key`: the slot that holds it or keeps it, else
    /// a free slot, else the slot kept longest. Refuses when every slot is
    /// held.
    pub fn acquire(&mut self, key: Key) -> Result<Grant> {
        if let Some((slot, count)) = self.held.get_mut(&key) {
            *count += 1;
            return Ok(Grant {
                slot: *slot,
                fresh: false,
            });
        }
        if let Some(at) = self.kept.iter().position(|(kept, _)| *kept == key) {
            let (_, slot) = self.kept.remove(at);
            self.held.insert(key, (slot, 1));
            return Ok(Grant { slot, fresh: false });
        }
        let taken: Vec<u32> = self
            .held
            .values()
            .map(|(slot, _)| *slot)
            .chain(self.kept.iter().map(|(_, slot)| *slot))
            .collect();
        let slot = match (0..self.seats).find(|slot| !taken.contains(slot)) {
            Some(slot) => slot,
            None if !self.kept.is_empty() => self.kept.remove(0).1,
            None => return Err(Fault::AdapterSlots { seats: self.seats }),
        };
        self.held.insert(key, (slot, 1));
        Ok(Grant { slot, fresh: true })
    }

    /// Gives back a hold whose write failed: its slot keeps nothing.
    pub fn abandon(&mut self, key: &Key) {
        self.release_key(key);
    }

    /// Gives back a hold. The slot keeps its bundle once the last hold goes,
    /// until another bundle needs it.
    pub fn release(&mut self, key: &Key) {
        if let Some(slot) = self.release_key(key) {
            self.kept.push((*key, slot));
        }
    }

    /// Drops one hold of `key`, and answers its slot if that was the last.
    fn release_key(&mut self, key: &Key) -> Option<u32> {
        let (slot, count) = self.held.get_mut(key)?;
        *count = count.saturating_sub(1);
        let (slot, last) = (*slot, *count == 0);
        if last {
            self.held.remove(key);
        }
        last.then_some(slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> Key {
        Key([byte; 32])
    }

    #[test]
    fn adapter_every_case() {
        the_residency_pins_releases_and_refuses_by_name();
        one_blob_is_one_slot_however_many_instances_name_it();
        an_abandoned_acquire_leaves_the_slot_free();
    }

    fn the_residency_pins_releases_and_refuses_by_name() {
        let mut slots = Slots::new(2);
        let first = slots.acquire(key(1)).expect("a free slot");
        assert!(first.fresh);
        let second = slots.acquire(key(2)).expect("the other one");
        assert_ne!(first.slot, second.slot, "two instances are two slots");
        let why = slots.acquire(key(3)).expect_err("both are pinned");
        assert!(why.to_string().contains('2'), "names the capacity: {why}");
        let again = slots.acquire(key(1)).expect("its own slot");
        assert_eq!(again.slot, first.slot);
        assert!(!again.fresh, "a held key does not re-land");
        slots.release(&key(1));
        assert_eq!(slots.live(), 2, "the second hold keeps it pinned");
        slots.release(&key(1));
        assert_eq!(slots.live(), 1, "the last release frees it");
        let reused = slots.acquire(key(3)).expect("the freed slot");
        assert_eq!(reused.slot, first.slot, "the lowest free slot is taken");
    }

    fn one_blob_is_one_slot_however_many_instances_name_it() {
        let mut slots = Slots::new(2);
        let alice = key(10);

        let first = slots.acquire(alice).expect("a free slot");
        assert!(first.fresh, "the first bind is the one that pays");
        let second = slots.acquire(alice).expect("the same identity");
        assert_eq!(second.slot, first.slot, "one blob, one slot");
        assert!(!second.fresh, "the second joins what is already there");
        let third = slots.acquire(alice).expect("and a third");
        assert_eq!(third.slot, first.slot);
        assert_eq!(slots.live(), 1, "three binds, one resident identity");

        let private = slots.acquire(key(77)).expect("the other slot");
        assert_ne!(private.slot, first.slot);
        assert!(private.fresh);

        let rewritten = key(11);
        let why = slots
            .acquire(rewritten)
            .expect_err("a new identity wants a seat and both are pinned");
        assert!(why.to_string().contains('2'), "names the capacity: {why}");

        slots.release(&alice);
        slots.release(&alice);
        assert_eq!(slots.live(), 2, "two holds are gone and one remains");
        slots.release(&alice);
        assert_eq!(slots.live(), 1, "the last release frees the shared slot");
        let after = slots
            .acquire(key(12))
            .expect("the freed slot seats the next identity");
        assert_eq!(after.slot, first.slot);
    }

    fn an_abandoned_acquire_leaves_the_slot_free() {
        let mut slots = Slots::new(1);
        let grant = slots.acquire(key(7)).expect("the only slot");
        slots.abandon(&key(7));
        assert_eq!(slots.live(), 0);
        let after = slots.acquire(key(8)).expect("it is free again");
        assert_eq!(after.slot, grant.slot);
    }
}
