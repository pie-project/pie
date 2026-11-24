//! A chunk's own life: opened for a shape and class, filled, a slot given
//! back, closed.

use super::*;

impl SlotPool {
    /// A chunk taken or moved onto lies in memory the engine backs for
    /// state: a committed segment (the bottom's, or a span segment the
    /// walk committed). Checked at every take and
    /// every move's destination in debug builds, naming the logical chunk,
    /// its physical and the segment: met, a fire on it reads or writes
    /// memory the engine gave back (a chunk of a segment just released).
    #[track_caller]
    pub(super) fn assert_backed(&self, what: &str, logical: u32, physical: u32) {
        if cfg!(debug_assertions) {
            let segment = self.segments.of(physical);
            let state = self.segments.state(segment);
            assert!(
                physical >= self.reserved && state == SegmentState::Committed,
                "{what} of logical chunk {logical} on physical {physical} in segment {segment} \
                 ({state:?}), which the engine does not back for state"
            );
        }
    }

    #[track_caller]
    pub(super) fn locate(&mut self, slot: Slot) -> (&mut ShapeState, usize) {
        let meta = self.chunks[slot.chunk as usize];
        assert!(meta.shape != NO_SHAPE, "{slot:?} lies in a free chunk");
        let state = &mut self.shapes[meta.shape as usize];
        assert!(slot.index < state.spc(), "{slot:?} is past its chunk");
        let at = (meta.frame * state.spc() + slot.index) as usize;
        (state, at)
    }

    /// Opens the lowest free committed chunk for `shape`. The caller has
    /// checked the room.
    pub(super) fn open_chunk(&mut self, shape: ShapeId, class: u32) -> u32 {
        let physical = self.open.first().expect("room was checked");
        self.open_at(physical, shape, class)
    }

    /// Opens free physical chunk `physical` for `shape`, to hold `class`
    /// ([`ChunkMeta::class`]); its logical id.
    #[track_caller]
    pub(super) fn open_at(&mut self, physical: u32, shape: ShapeId, class: u32) -> u32 {
        let logical = self.physical_to_logical[physical as usize];
        self.assert_backed("a take", logical, physical);
        self.open.clear(physical);
        self.free.clear(physical);
        debug_assert_eq!(self.chunks[logical as usize].shape, NO_SHAPE);
        self.segments.hold(physical);
        self.held += 1;
        let frame = self.shapes[shape as usize].open_frame();
        self.chunks[logical as usize] = ChunkMeta {
            shape,
            frame,
            used: 0,
            class,
        };
        logical
    }

    /// Takes up to `want` of `chunk`'s lowest free slots.
    pub(super) fn fill(&mut self, chunk: u32, live: bool, want: u64, taken: &mut Vec<Slot>) {
        self.changes += 1;
        let meta = self.chunks[chunk as usize];
        let physical = self.logical_to_physical[chunk as usize];
        let state = &mut self.shapes[meta.shape as usize];
        let base = meta.frame as usize * state.words;
        let mut got = 0u32;
        for word in 0..state.words {
            while state.free[base + word] != 0 && u64::from(got) < want {
                let bits = &mut state.free[base + word];
                let index = word as u32 * WORD + bits.trailing_zeros();
                *bits &= *bits - 1;
                let refs = state.ref_at(meta.frame, index);
                let was = *refs;
                *refs = SlotRefs {
                    live: u32::from(live),
                    cache: u32::from(!live),
                    touched: 0,
                };
                let now = *refs;
                state.reclassify(was, now);
                taken.push(Slot { chunk, index });
                got += 1;
            }
            if u64::from(got) == want {
                break;
            }
        }
        let now = meta.used + got;
        state.reposition(physical, chunk, meta.class, meta.used, now);
        self.chunks[chunk as usize].used = now;
    }

    /// Takes slots `first .. first + n` of `chunk`, every one free.
    pub(super) fn fill_run(
        &mut self,
        chunk: u32,
        live: bool,
        first: u32,
        n: u32,
        taken: &mut Vec<Slot>,
    ) {
        self.changes += 1;
        let meta = self.chunks[chunk as usize];
        let physical = self.logical_to_physical[chunk as usize];
        let state = &mut self.shapes[meta.shape as usize];
        let base = meta.frame as usize * state.words;
        for index in first..first + n {
            let (word, bit) = Bits::locate(index);
            let bits = &mut state.free[base + word];
            debug_assert_ne!(*bits & bit, 0, "slot {index} of chunk {chunk} is taken");
            *bits &= !bit;
            let refs = state.ref_at(meta.frame, index);
            let was = *refs;
            *refs = SlotRefs {
                live: u32::from(live),
                cache: u32::from(!live),
                touched: 0,
            };
            let now = *refs;
            state.reclassify(was, now);
            taken.push(Slot { chunk, index });
        }
        let now = meta.used + n;
        state.reposition(physical, chunk, meta.class, meta.used, now);
        self.chunks[chunk as usize].used = now;
    }

    pub(super) fn give_back(&mut self, slot: Slot) {
        self.changes += 1;
        let meta = self.chunks[slot.chunk as usize];
        let physical = self.logical_to_physical[slot.chunk as usize];
        #[cfg(feature = "test-slot-fill")]
        if let Some(queue) = &self.poison.0 {
            let bytes = self.shapes[meta.shape as usize].shape.slot_bytes;
            let offset = u64::from(physical) * self.chunk_bytes + u64::from(slot.index) * bytes;
            queue
                .lock()
                .expect("the poison queue's lock is never poisoned")
                .push((offset, bytes));
        }
        let state = &mut self.shapes[meta.shape as usize];
        let (word, bit) = Bits::locate(slot.index);
        let at = meta.frame as usize * state.words + word;
        debug_assert_eq!(state.free[at] & bit, 0, "{slot:?} freed twice");
        state.free[at] |= bit;
        let now = meta.used - 1;
        state.reposition(physical, slot.chunk, meta.class, meta.used, now);
        self.chunks[slot.chunk as usize].used = now;
        if now == 0 {
            state.spare.push(meta.frame);
            self.close_chunk(slot.chunk, physical);
        }
    }

    pub(super) fn close_chunk(&mut self, logical: u32, physical: u32) {
        self.runs.remove(&logical);
        self.chunks[logical as usize] = FREE_CHUNK;
        self.free.set(physical);
        self.segments.let_go(physical);
        self.held -= 1;
        self.reopen(physical);
    }

    // ---- the span backed on demand ----
}
