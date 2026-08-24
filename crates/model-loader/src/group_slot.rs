//! Which instance of a group is in which slot.
//!
//! A group is `arity` interchangeable instances of one plan. The loader
//! stops there on purpose: making all `arity` of them resident and paging a
//! few of them through a bounded slab are the same program, and choosing
//! between those is policy, which lives here.
//!
//! This is the bookkeeping half of that policy and nothing else. It holds no
//! device memory, opens no files, and names no API, so the eviction rule --
//! the part that decides whether streaming thrashes -- is testable on a
//! machine with no GPU. The half that moves bytes is each backend's own:
//! CUDA pages into a device slab, Metal into a heap slot, and neither
//! disagrees with the other about which instance was supposed to be there.
//!
//! It lives beside the plan rather than in a driver for that last reason.
//! Two backends deciding residency by two eviction rules is two ways for the
//! same checkpoint to thrash, and one of them would be found much later than
//! the other.
//!
//! # Who reads this today, measured
//!
//! ONE backend, and it is out of the workspace: `driver-metal`'s
//! `loader/slab.rs` is the only consumer of anything here, and its own doc
//! saying this is "shared with CUDA rather than restated" is ahead of the
//! facts — no CUDA path names this module. `release` and `evictions` have no
//! caller at all outside the tests below.
//!
//! It was considered for relocation into `driver-metal` beside its reader and
//! kept, for two reasons that are about evidence rather than tidiness. The
//! tests below RUN, in `cargo test -p model-loader`, and `driver-metal` does
//! not compile — moving the file would convert five checked properties into
//! unchecked text. And this module imports nothing from the rest of the crate
//! and nothing imports it, so what it costs the loader is its own length; what
//! it would cost a driver is the second eviction rule the paragraph above
//! exists to prevent.
//!
//! The rule: a free slot always beats a used one (free slots have age 0, and
//! nothing else ever does), ties go to the lowest slot id, and among used
//! slots the least recently touched loses. Slots pinned by the batch in
//! flight are never victims, because a kernel may still be reading one; a
//! batch that wants more instances than there are slots is a configuration
//! error and says so rather than corrupting a slot out from under a running
//! kernel.

/// One slot's bookkeeping.
#[derive(Clone, Copy, Debug)]
struct Slot {
    instance: Option<u32>,
    /// 0 = never filled, so free slots win the LRU without a special case.
    age: u64,
    pinned: bool,
}

/// What [`GroupSlotIndex::acquire`] decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Acquired {
    /// The slot the instance goes in. Touched and pinned already; the caller
    /// still has to fill it -- this says where the instance goes, not that
    /// it is there.
    pub slot: u32,
    /// Whether the slot held a different instance, which the caller must
    /// treat as invalidating any pointer it handed out for that instance.
    pub evicted: bool,
}

/// Every slot is pinned by the batch in flight: the batch wants more
/// instances at once than the cache holds. A configuration error, reported
/// rather than resolved by corrupting a slot a kernel may still read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllSlotsPinned {
    /// How many slots the index has, all of them pinned.
    pub num_slots: u32,
}

impl std::fmt::Display for AllSlotsPinned {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "all {} slots are pinned by the batch in flight, so it wants more \
             instances at once than the cache holds",
            self.num_slots
        )
    }
}

impl std::error::Error for AllSlotsPinned {}

/// The residency map for one group: instance number in, slot number out.
#[derive(Clone, Debug)]
pub struct GroupSlotIndex {
    arity: u32,
    slot_of: Vec<Option<u32>>,
    slots: Vec<Slot>,
    tick: u64,
    evictions: u64,
}

impl GroupSlotIndex {
    /// `arity` is the group's instance count, `num_slots` how many fit the
    /// budget. `num_slots` may exceed `arity` -- the caller clamps if it
    /// cares.
    ///
    /// # Panics
    ///
    /// If either count is zero: an index over nothing is not a small cache,
    /// it is one that cannot run, and every caller has already refused that
    /// configuration in its own words.
    #[must_use]
    pub fn new(arity: u32, num_slots: u32) -> Self {
        assert!(
            arity > 0 && num_slots > 0,
            "GroupSlotIndex: arity and slot count must be positive"
        );
        GroupSlotIndex {
            arity,
            slot_of: vec![None; arity as usize],
            slots: vec![
                Slot {
                    instance: None,
                    age: 0,
                    pinned: false,
                };
                num_slots as usize
            ],
            tick: 0,
            evictions: 0,
        }
    }

    /// The group's instance count.
    #[must_use]
    pub fn arity(&self) -> u32 {
        self.arity
    }

    /// How many slots the budget holds.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn num_slots(&self) -> u32 {
        self.slots.len() as u32
    }

    /// The slot holding `instance`, or `None`.
    ///
    /// # Panics
    ///
    /// If `instance` is outside the arity. Callers translate their own
    /// coordinates (layer, expert) and bounds-check them first; an index
    /// past the arity here is their bug, not their data's.
    #[must_use]
    pub fn find(&self, instance: u32) -> Option<u32> {
        self.slot_of[self.checked(instance)]
    }

    /// Mark a slot as used by the batch in flight. A hit still has to say
    /// so, or a later miss in the same batch could evict what this one just
    /// found.
    pub fn touch_and_pin(&mut self, slot: u32) {
        self.tick += 1;
        let s = &mut self.slots[slot as usize];
        s.age = self.tick;
        s.pinned = true;
    }

    /// Give `instance` a slot, evicting the LRU unpinned one if no slot is
    /// free. The returned slot is touched and pinned.
    ///
    /// # Errors
    ///
    /// [`AllSlotsPinned`] when the batch in flight holds every slot.
    ///
    /// # Panics
    ///
    /// If `instance` is outside the arity, as for [`find`](Self::find).
    pub fn acquire(&mut self, instance: u32) -> Result<Acquired, AllSlotsPinned> {
        let key = self.checked(instance);

        let mut victim: Option<(u32, u64)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            if s.pinned {
                continue;
            }
            // Strictly-less keeps the tie on the lowest slot id.
            if victim.is_none_or(|(_, best)| s.age < best) {
                victim = Some((u32::try_from(i).expect("slot count fits u32"), s.age));
            }
        }
        let Some((victim, _)) = victim else {
            return Err(AllSlotsPinned {
                num_slots: self.num_slots(),
            });
        };

        self.tick += 1;
        let s = &mut self.slots[victim as usize];
        let out = Acquired {
            slot: victim,
            evicted: s.instance.is_some(),
        };
        if let Some(old) = s.instance {
            self.slot_of[old as usize] = None;
            self.evictions += 1;
        }
        s.instance = Some(instance);
        s.age = self.tick;
        s.pinned = true;
        self.slot_of[key] = Some(victim);
        Ok(out)
    }

    /// Give back the pin an [`acquire`](Self::acquire) took, leaving the
    /// slot's contents and its recency alone.
    ///
    /// For a caller that acquires a slot to write it rather than to read it,
    /// and so has no batch to hold it against. Placing every instance in
    /// turn is the one that does: pinning each as it went would run the
    /// index out of slots before it finished -- a failure, not a slowdown,
    /// since `acquire` has nothing left to evict.
    pub fn release(&mut self, slot: u32) {
        if let Some(s) = self.slots.get_mut(slot as usize) {
            s.pinned = false;
        }
    }

    /// End the batch. Until this is called every slot the batch touched is
    /// off limits as a victim.
    pub fn unpin_all(&mut self) {
        for s in &mut self.slots {
            s.pinned = false;
        }
    }

    /// How many instances have been displaced so far.
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    fn checked(&self, instance: u32) -> usize {
        assert!(
            instance < self.arity,
            "GroupSlotIndex: instance {instance} outside arity {}",
            self.arity
        );
        instance as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_free_slot_always_beats_a_used_one_and_ties_go_low() {
        let mut index = GroupSlotIndex::new(8, 3);
        assert_eq!(
            index.acquire(0),
            Ok(Acquired {
                slot: 0,
                evicted: false
            })
        );
        index.unpin_all();
        // Slot 0 is used and recently touched; 1 and 2 are free. The free
        // ones win, lowest id first.
        assert_eq!(
            index.acquire(1),
            Ok(Acquired {
                slot: 1,
                evicted: false
            })
        );
        assert_eq!(
            index.acquire(2),
            Ok(Acquired {
                slot: 2,
                evicted: false
            })
        );
    }

    #[test]
    fn the_least_recently_touched_unpinned_slot_is_the_victim() {
        let mut index = GroupSlotIndex::new(8, 2);
        index.acquire(0).unwrap();
        index.acquire(1).unwrap();
        index.unpin_all();
        // Touch instance 0's slot; instance 1's becomes the LRU.
        let slot0 = index.find(0).unwrap();
        index.touch_and_pin(slot0);
        index.release(slot0);
        let got = index.acquire(2).unwrap();
        assert_eq!(
            got,
            Acquired {
                slot: index.find(2).unwrap(),
                evicted: true
            }
        );
        assert_eq!(index.find(1), None, "the evicted instance is forgotten");
        assert_eq!(index.find(0), Some(slot0));
        assert_eq!(index.evictions(), 1);
    }

    #[test]
    fn a_pinned_slot_is_never_a_victim_and_all_pinned_says_so() {
        let mut index = GroupSlotIndex::new(8, 2);
        index.acquire(0).unwrap();
        index.acquire(1).unwrap();
        // Both slots pinned by the batch in flight: a third instance is a
        // configuration error, not an eviction.
        assert_eq!(index.acquire(2), Err(AllSlotsPinned { num_slots: 2 }));
        index.unpin_all();
        assert!(index.acquire(2).is_ok(), "unpinning ends the batch");
    }

    #[test]
    fn a_hit_repins_what_the_batch_already_found() {
        let mut index = GroupSlotIndex::new(4, 2);
        index.acquire(0).unwrap();
        index.unpin_all();
        let slot = index.find(0).unwrap();
        index.touch_and_pin(slot);
        // The other slot is free; a miss takes it rather than the pinned hit.
        let got = index.acquire(1).unwrap();
        assert_ne!(got.slot, slot);
    }

    #[test]
    fn release_gives_back_a_pin_without_touching_recency() {
        let mut index = GroupSlotIndex::new(4, 2);
        // The place-all pattern: acquire to write, release as you go.
        for instance in 0..2 {
            let got = index.acquire(instance).unwrap();
            index.release(got.slot);
        }
        // Nothing pinned, so a third instance evicts the oldest (slot 0).
        assert_eq!(
            index.acquire(2),
            Ok(Acquired {
                slot: 0,
                evicted: true
            })
        );
    }
}
