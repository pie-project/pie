//! What the pool holds, read: a slot's ref and chunk, free and partial room
//! by shape, what fits, the demand, and the waiters short of room.

use super::*;

impl SlotPool {
    /// Committed chunks neither held nor set aside for activation, and
    /// below the ceiling ([`Self::set_ceiling`]): a committed chunk the
    /// engine binds none of yet is no room, so the walker commits ahead
    /// of a waiter rather than count it.
    pub fn headroom(&self) -> u32 {
        let all = self
            .committed_chunks()
            .saturating_sub(self.reserved + self.held + self.act);
        if self.ceiling >= self.total {
            return all;
        }
        let below: u32 = (0..self.segments.count())
            .filter(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(self.ceiling).saturating_sub(range.start)
            })
            .sum();
        all.min(below.saturating_sub(self.reserved + self.held))
    }

    pub fn shape(&self, shape: ShapeId) -> Option<&SlotShape> {
        self.shapes.get(shape as usize).map(|state| &state.shape)
    }

    /// The engine's slot reference: `chunk * slots_per_chunk + index`.
    pub fn slot_ref(&self, slot: Slot) -> u32 {
        let shape = self.chunks[slot.chunk as usize].shape;
        slot.chunk * self.shapes[shape as usize].spc() + slot.index
    }

    /// Where logical chunk `chunk` lies now, when it is reserved or open.
    pub fn physical(&self, chunk: u32) -> Option<u32> {
        (chunk < self.reserved || self.chunks[chunk as usize].shape != NO_SHAPE)
            .then(|| self.logical_to_physical[chunk as usize])
    }

    /// The slots chunk `chunk` (logical) holds or keeps for an epoch: 0 for
    /// a free chunk.
    #[must_use]
    pub fn chunk_used(&self, chunk: u32) -> u32 {
        self.chunks.get(chunk as usize).map_or(0, |meta| meta.used)
    }

    /// The physical chunks state may open now ([`Self::free_chunks`]'s).
    #[must_use]
    pub fn open_chunks(&self) -> Vec<u32> {
        self.open.iter_in(0, self.total).collect()
    }

    /// The open chunks a guest's state may take ([`Self::chunk_runs_for`]'s):
    /// below the activation the rung lends and the ceiling.
    #[must_use]
    pub fn state_open_chunks(&self) -> Vec<u32> {
        self.open.iter_in(0, self.state_limit().min(self.ceiling)).collect()
    }

    /// The logical -> physical table the engine keeps: a permutation, free
    /// logical ids included.
    pub fn chunk_table(&self) -> &[u32] {
        &self.logical_to_physical
    }

    /// Whether `slot` of `shape` is handed out: its chunk open for the shape
    /// and the slot not free in it (held, or waiting on a fire's epoch).
    pub fn is_held(&self, shape: ShapeId, slot: Slot) -> bool {
        let Some(meta) = self.chunks.get(slot.chunk as usize) else {
            return false;
        };
        let Some(state) = self.shapes.get(shape as usize) else {
            return false;
        };
        if meta.shape != shape || slot.index >= state.spc() {
            return false;
        }
        let word = meta.frame as usize * state.words + (slot.index / WORD) as usize;
        state.free[word] & (1 << (slot.index % WORD)) == 0
    }

    /// The shape logical chunk `chunk` is open for.
    pub fn chunk_shape(&self, chunk: u32) -> Option<ShapeId> {
        Some(self.chunks[chunk as usize].shape).filter(|&shape| shape != NO_SHAPE)
    }

    /// The bytes the chunks opened now for any of `shapes` hold free: their
    /// unused slots, room no other shape takes until a chunk empties.
    #[must_use]
    pub fn slack_of(&self, shapes: &[ShapeId]) -> u64 {
        self.chunks
            .iter()
            .filter(|meta| meta.shape != NO_SHAPE && shapes.contains(&meta.shape))
            .map(|meta| {
                let shape = &self.shapes[meta.shape as usize].shape;
                u64::from(shape.slots_per_chunk.saturating_sub(meta.used)) * shape.slot_bytes
            })
            .sum()
    }

    /// Slots `holder` holds references on.
    pub fn held_by(&self, holder: &Holder) -> u64 {
        self.holders.get(holder).copied().unwrap_or(0)
    }

    /// References on `slot`: `(working sets and snapshots, cache entries)`.
    pub fn refs(&self, slot: Slot) -> (u32, u32) {
        let meta = self.chunks[slot.chunk as usize];
        if meta.shape == NO_SHAPE {
            return (0, 0);
        }
        let shape = &self.shapes[meta.shape as usize];
        let refs = shape.refs[(meta.frame * shape.spc() + slot.index) as usize];
        (refs.live, refs.cache)
    }

    /// Free chunks a shape could open now without breaking the budget.
    pub fn free_chunks(&self) -> u32 {
        self.open.count.min(self.headroom())
    }

    /// Free slots `shape` could take now without breaking the budget.
    /// Free slots of `shape` in its partly filled chunks: what a take of
    /// its slots fills before it opens a chunk.
    #[must_use]
    pub fn partial_free(&self, shape: ShapeId) -> u64 {
        self.shapes
            .get(shape as usize)
            .map_or(0, |state| state.partial_free)
    }

    pub fn available(&self, shape: ShapeId) -> u64 {
        let Some(state) = self.shapes.get(shape as usize) else {
            return 0;
        };
        let chunks = self.free_chunks();
        state.partial_free + u64::from(chunks) * u64::from(state.spc())
    }

    /// Whether every `(shape, slots)` of `needs` could be taken at once:
    /// each shape's partly filled chunks first, then free chunks, which the
    /// shapes share.
    pub fn fits(&self, needs: &[(ShapeId, u64)]) -> bool {
        let room = u64::from(self.free_chunks());
        let mut chunks = 0u64;
        for &(shape, slots) in needs {
            let Some(state) = self.shapes.get(shape as usize) else {
                return slots == 0;
            };
            chunks += slots
                .saturating_sub(state.partial_free)
                .div_ceil(u64::from(state.spc()));
        }
        chunks <= room
    }

    /// Free chunks a shape could open once every segment below the
    /// activation the facts allow were committed: what the pool can grow
    /// to, short of the reserved chunks, the activation and what is held.
    pub fn growable_chunks(&self) -> u32 {
        let free = (0..self.state_limit())
            .filter(|&chunk| {
                self.free.get(chunk)
                    && !self.fenced.get(chunk)
                    && !matches!(
                        self.segments.state(self.segments.of(chunk)),
                        SegmentState::Draining { .. }
                    )
            })
            .count() as u32;
        free.min(
            self.total
                .saturating_sub(self.reserved + self.held + self.act),
        )
    }

    /// [`Self::fits`] against what the pool can grow to
    /// ([`Self::growable_chunks`]), committed or not.
    pub fn fits_grown(&self, needs: &[(ShapeId, u64)]) -> bool {
        self.chunks_for(needs)
            .is_some_and(|chunks| chunks <= u64::from(self.growable_chunks()))
    }

    pub(super) fn chunks_for(&self, needs: &[(ShapeId, u64)]) -> Option<u64> {
        let mut chunks = 0u64;
        for &(shape, slots) in needs {
            let Some(state) = self.shapes.get(shape as usize) else {
                if slots == 0 {
                    continue;
                }
                return None;
            };
            chunks += slots
                .saturating_sub(state.partial_free)
                .div_ceil(u64::from(state.spc()));
        }
        Some(chunks)
    }

    /// Commits the lowest uncommitted segments below the activation and
    /// the ceiling, a prefix, until `needs` fit the committed chunks and every chunk below
    /// physical chunk `through` lies in a committed segment. The segments
    /// committed, lowest first: the engine backs them (`commit_segments`
    /// of the prefix they end) before anything lands in them. Stops short
    /// where no segment is left; the take that follows says so.
    pub fn commit_for(&mut self, needs: &[(ShapeId, u64)], through: u32) -> Vec<u32> {
        let mut committed = Vec::new();
        loop {
            let covered = (0..through.min(self.total)).all(|chunk| {
                self.segments.state(self.segments.of(chunk)) != SegmentState::Uncommitted
            });
            if covered && self.fits(needs) {
                return committed;
            }
            // Never into what the engine holds back (the ceiling): a
            // segment there is no room the engine binds, and backing it
            // takes the memory the engine keeps for its bodies.
            let Some(segment) = self
                .segments
                .uncommitted()
                .find(|&segment| segment < self.top_segment && self.backs(segment))
            else {
                return committed;
            };
            self.commit_segment(segment)
                .expect("an uncommitted segment commits");
            committed.push(segment);
        }
    }

    pub fn shape_demand(&self, shape: ShapeId) -> ShapeDemand {
        self.shapes
            .get(shape as usize)
            .map_or_else(ShapeDemand::default, |state| ShapeDemand {
                live: state.live,
                cached: state.cached,
                waiting: state.waiting,
            })
    }

    /// Demand counted slot by slot: what is live, and what is wanted.
    pub fn demand(&self) -> Demand {
        self.shapes.iter().fold(Demand::default(), |demand, state| {
            let bytes = state.shape.slot_bytes;
            Demand {
                live: demand.live + state.live * bytes,
                wanted: demand.wanted + (state.live + state.cached + state.waiting) * bytes,
            }
        })
    }

    /// Who holds the pool's slots, by kind of holder (the store's own, as
    /// guest state and role slots are taken, apart), and the chunks free
    /// and opened: what a stall is read by.
    #[must_use]
    pub fn held_census(&self) -> String {
        let (mut sets, mut cache, mut snapshots, mut weights, mut store) = (0, 0, 0, 0, 0);
        for (holder, &slots) in &self.holders {
            match holder {
                h if *h == super::super::chunks::STORE => store += slots,
                Holder::WorkingSet(_) => sets += slots,
                Holder::CacheEntry(_) => cache += slots,
                Holder::Snapshot(_) => snapshots += slots,
                Holder::Weights { .. } => weights += slots,
            }
        }
        format!(
            "slots: working_sets={sets} cache={cache} snapshots={snapshots} weights={weights} \
             store={store} ({} holders); chunks: opened={} free={} committed={} total={} \
             pending={}",
            self.holders.len(),
            self.held,
            self.free_chunks(),
            self.committed_chunks(),
            self.total,
            self.pending.values().map(Vec::len).sum::<usize>(),
        )
    }

    /// Records that `waiter` is parked short `slots` of `shape`, replacing
    /// what it was short before.
    pub fn wait(&mut self, waiter: u64, shape: ShapeId, slots: u64) {
        self.unwait(waiter);
        if let Some(state) = self.shapes.get_mut(shape as usize) {
            state.waiting += slots;
            self.waiters.insert(waiter, (shape, slots));
        }
    }

    /// Whether no holder holds a slot of the pool: no working set, cache
    /// entry, snapshot, store table or weight lies in any chunk.
    pub fn holds_nothing(&self) -> bool {
        self.holders.is_empty()
    }

    /// Whether any waiter is parked short of a slot.
    pub fn waiting(&self) -> bool {
        self.shapes.iter().any(|state| state.waiting > 0)
    }

    pub fn unwait(&mut self, waiter: u64) {
        if let Some((shape, slots)) = self.waiters.remove(&waiter) {
            self.shapes[shape as usize].waiting -= slots;
        }
    }

    // ---- slots ----
}
