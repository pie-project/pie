//! Giving slots back: release, the room signal it tells, and the fences a
//! released slot waits on before it is free.

use super::*;

impl SlotPool {
    /// Drops `holder`'s reference on each of `slots`. A slot no one holds
    /// any more is handed out again once the later of `epoch` and the last
    /// fire that touched it retires.
    #[track_caller]
    pub fn release(&mut self, holder: Holder, slots: &[Slot], epoch: u64) {
        let live = holder.is_live();
        let mut immediate = Vec::new();
        // Let go slots by the epoch they wait on; few epochs per call.
        let mut waiting: Vec<(u64, Vec<Slot>)> = Vec::new();
        for &slot in slots {
            let (state, at) = self.locate(slot);
            let was = state.refs[at];
            let mut now = was;
            if live {
                assert!(now.live > 0, "{slot:?} has no live reference of {holder:?} to drop");
                now.live -= 1;
            } else {
                assert!(now.cache > 0, "{slot:?} has no cache reference to drop");
                now.cache -= 1;
            }
            state.refs[at] = now;
            state.reclassify(was, now);
            if now.live + now.cache == 0 {
                let after = now.touched.max(epoch);
                if after <= self.retired {
                    immediate.push(slot);
                } else if let Some((_, batch)) = waiting.iter_mut().find(|(at, _)| *at == after) {
                    batch.push(slot);
                } else {
                    waiting.push((after, vec![slot]));
                }
            }
        }
        if !slots.is_empty() {
            let count = self
                .holders
                .get_mut(&holder)
                .expect("a holder releases only what it holds");
            *count -= slots.len() as u64;
            if *count == 0 {
                self.holders.remove(&holder);
            }
        }
        for (after, batch) in waiting {
            match self.pending.entry(after) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(batch);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().extend(batch);
                }
            }
        }
        let freed = !immediate.is_empty();
        for slot in immediate {
            self.give_back(slot);
        }
        if freed {
            self.slots_freed();
        }
    }

    /// Calls `signal` with this pool's counts each time slots come free
    /// ([`RoomGiven::signal`]), once a release or a retire however many
    /// came: set once, by the planner of the engine this pool backs, its
    /// tier and the shape whose free slots it counts.
    pub fn set_room_signal(&self, signal: RoomSignal, tier: RoomTier, counts: Option<ShapeId>) {
        let _ = self.given.signal.set((signal, tier, counts));
    }

    /// Test builds: each slot given back from here on is pushed to `queue`
    /// as its physical byte range, for the engine that backs this pool to
    /// poison before its next verb ([`::engine::chunk::PoisonQueue`]).
    #[cfg(feature = "test-slot-fill")]
    pub fn poison_into(&mut self, queue: ::engine::chunk::PoisonQueue) {
        self.poison = PoisonTap(Some(queue));
    }

    pub(super) fn slots_freed(&self) {
        if let Some((signal, tier, counts)) = self.given.signal.get() {
            (signal.0)(RoomCount {
                tier: *tier,
                chunks: u64::from(self.free_chunks()),
                slots: counts.map_or(0, |shape| self.available(shape)),
            });
        }
    }

    /// Every fire through `epoch` has retired: what waited on them is free.
    /// The chunks fenced until `from` stay fenced until `to` instead.
    pub fn refence(&mut self, from: u64, to: u64) {
        if let Some(chunks) = self.fences.remove(&from) {
            self.fences.entry(to).or_default().extend(chunks);
        }
    }

    pub fn retire_through(&mut self, epoch: u64) {
        self.retired = self.retired.max(epoch);
        let later = self.fences.split_off(&(epoch.saturating_add(1)));
        let mut due = std::mem::replace(&mut self.fences, later);
        if epoch == u64::MAX {
            due.append(&mut self.fences);
        }
        for chunk in due.into_values().flatten() {
            self.fenced.clear(chunk);
            self.reopen(chunk);
        }
        let later = self.pending.split_off(&(epoch.saturating_add(1)));
        let due = std::mem::replace(&mut self.pending, later);
        let mut freed = false;
        if epoch == u64::MAX {
            // split_off(MAX) leaves MAX itself behind.
            let rest = std::mem::take(&mut self.pending);
            for slot in rest.into_values().flatten() {
                self.give_back(slot);
                freed = true;
            }
        }
        for slot in due.into_values().flatten() {
            self.give_back(slot);
            freed = true;
        }
        if freed {
            self.slots_freed();
        }
    }

    /// Per class of `shape` ([`ChunkMeta::class`]), its partly filled
    /// chunks and their free slots: what segregating by class holds back.
    pub fn partial_classes(&self, shape: ShapeId) -> Vec<(u32, usize, u64)> {
        let Some(state) = self.shapes.get(shape as usize) else {
            return Vec::new();
        };
        state
            .partial
            .iter()
            .map(|(&class, chunks)| {
                let free = chunks.iter().map(|&(std::cmp::Reverse(used), _, _)| u64::from(state.spc() - used)).sum();
                (class, chunks.len(), free)
            })
            .collect()
    }

    /// Slots let go and waiting on a fire.
    pub fn pending_slots(&self) -> usize {
        self.pending.values().map(Vec::len).sum()
    }
}
