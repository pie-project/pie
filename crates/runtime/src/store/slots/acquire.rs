//! Taking slots: singly, in runs, in whole-chunk runs, where a placement
//! puts an unpark's pieces, by claim, and shared.

use super::*;

impl SlotPool {
    /// `n` fresh slots of `shape` for `holder`, from the fullest partly
    /// filled chunks first, then from chunks opened lowest address first.
    /// All or nothing.
    pub fn acquire(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        n: u64,
    ) -> Result<Vec<Slot>, SlotError> {
        let state = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?;
        let spc = u64::from(state.spc());
        let fresh = n.saturating_sub(state.partial_free).div_ceil(spc);
        let room = u64::from(self.free_chunks());
        if fresh > room {
            let short = fresh - room;
            return Err(SlotError::Short {
                shape,
                slots: n - state.partial_free - room * spc,
                chunks: short as u32,
            });
        }
        let mut taken = Vec::with_capacity(n as usize);
        while (taken.len() as u64) < n {
            let chunk = match self.shapes[shape as usize].fullest(SINGLE) {
                Some(logical) => logical,
                None => self.open_chunk(shape, SINGLE),
            };
            self.fill(chunk, holder.is_live(), n - taken.len() as u64, &mut taken);
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Ok(taken)
    }

    /// The chunks of `shape`'s single slots ([`SINGLE`]), each with the
    /// slots it holds or keeps for an epoch: what a list compaction reads
    /// its sources and destinations from. `settled` is false for one a
    /// chunk move or a residency copy holds (fenced, or moving out): its
    /// slots stay where they lie meanwhile, neither a source nor a
    /// destination.
    #[must_use]
    pub fn single_chunks(&self, shape: ShapeId) -> Vec<SingleChunk> {
        self.chunks
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.shape == shape && meta.class == SINGLE)
            .map(|(logical, meta)| {
                let physical = self.logical_to_physical[logical];
                SingleChunk {
                    logical: logical as u32,
                    used: meta.used,
                    settled: !self.fenced.get(physical) && !self.moving_out.contains_key(&physical),
                }
            })
            .collect()
    }

    /// `n` fresh slots of `shape` for `holder` from the partly filled
    /// chunks of its single slots outside `outside` (logical), fullest
    /// first, opening none: a list compaction's destinations. All or
    /// nothing.
    pub fn acquire_outside(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        n: u64,
        outside: &BTreeSet<u32>,
    ) -> Option<Vec<Slot>> {
        let state = self.shapes.get(shape as usize)?;
        let order: Vec<u32> = state
            .partial
            .get(&SINGLE)
            .into_iter()
            .flatten()
            .map(|&(_, _, logical)| logical)
            .filter(|logical| !outside.contains(logical))
            .collect();
        let spc = u64::from(state.spc());
        let room: u64 = order
            .iter()
            .map(|&logical| spc - u64::from(self.chunks[logical as usize].used))
            .sum();
        if room < n {
            return None;
        }
        let mut taken = Vec::with_capacity(n as usize);
        for logical in order {
            if taken.len() as u64 == n {
                break;
            }
            self.fill(
                logical,
                holder.is_live(),
                n - taken.len() as u64,
                &mut taken,
            );
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Some(taken)
    }

    /// `n` adjacent slots of `shape` for `holder`, in index order, in one
    /// chunk: the lowest run of `n` free slots in the fullest partly filled
    /// chunk that has one, else the first `n` slots of a chunk opened for
    /// the shape. A run is never split across chunks: where no chunk has
    /// room for it whole, nothing is taken.
    ///
    /// # Errors
    ///
    /// [`SlotError::RunPastChunk`] where `n` is past a chunk's slots;
    /// [`SlotError::Short`] where no chunk has room for the run.
    pub fn acquire_run(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        n: u64,
    ) -> Result<Vec<Slot>, SlotError> {
        let state = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?;
        let per_chunk = state.spc();
        if n > u64::from(per_chunk) {
            return Err(SlotError::RunPastChunk {
                shape,
                slots: n,
                per_chunk,
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let n = n as u32;
        let found = self.partial_run(shape, n, |_, _| false);
        let (chunk, first) = match found {
            Some(found) => found,
            None if self.free_chunks() > 0 => (self.open_chunk(shape, n), 0),
            None => {
                return Err(SlotError::Short {
                    shape,
                    slots: u64::from(n),
                    chunks: 1,
                });
            }
        };
        let mut taken = Vec::with_capacity(n as usize);
        self.fill_run(chunk, holder.is_live(), first, n, &mut taken);
        *self.holders.entry(holder).or_default() += u64::from(n);
        Ok(taken)
    }

    /// Where a run of `n` slots of `shape` lies in a chunk already open for
    /// runs of its length, the slots `held(logical, index)` reads as held
    /// taken too: its logical chunk and first slot. A run lies only among
    /// runs of its own length: a run's hole left behind is one the next run
    /// of that length takes exactly, and a chunk whose runs all leave is
    /// free whole again. What [`Self::acquire_run`] takes, and what
    /// [`Self::chunk_runs_beside`] places on its view.
    pub(super) fn partial_run(
        &self,
        shape: ShapeId,
        n: u32,
        held: impl Fn(u32, u32) -> bool,
    ) -> Option<(u32, u32)> {
        let state = self.shapes.get(shape as usize)?;
        state.partial.get(&n).and_then(|chunks| {
            chunks.iter().find_map(|&(_, _, logical)| {
                let frame = self.chunks[logical as usize].frame;
                state
                    .free_run_beside(frame, n, |index| held(logical, index))
                    .map(|first| (logical, first))
            })
        })
    }

    /// The free runs of `run` adjacent chunks state may open now, each
    /// inside one committed segment, below the current rung's activation
    /// and the ceiling ([`Self::state_limit`], [`Self::set_ceiling`]): each
    /// run's first physical chunk, lowest first, no two sharing a chunk,
    /// and no more than the budget's room holds ([`Self::headroom`]).
    pub(super) fn chunk_runs_for(&self, run: u32) -> Vec<u32> {
        self.runs_where(run, self.headroom(), |chunk| self.open.get(chunk))
    }

    /// [`Self::chunk_runs_for`] over the chunks `free` reads as free and
    /// `room` chunks of the budget's headroom.
    pub(super) fn runs_where(&self, run: u32, room: u32, free: impl Fn(u32) -> bool) -> Vec<u32> {
        self.runs_below(run, room, self.state_limit(), free)
    }

    /// [`Self::runs_where`] below physical chunk `below` (and the ceiling).
    pub(super) fn runs_below(
        &self,
        run: u32,
        room: u32,
        below: u32,
        free: impl Fn(u32) -> bool,
    ) -> Vec<u32> {
        let run = run.max(1);
        if run > self.segments.segment_chunks() {
            return Vec::new();
        }
        let limit = below.min(self.ceiling);
        let most = (room / run) as usize;
        let mut firsts = Vec::new();
        for segment in 0..self.segments.count() {
            let range = self.segments.range(segment);
            let mut stretch = 0u32;
            for chunk in range.start..range.end.min(limit) {
                if !free(chunk) {
                    stretch = 0;
                    continue;
                }
                stretch += 1;
                if stretch == run {
                    firsts.push(chunk + 1 - run);
                    stretch = 0;
                }
            }
        }
        firsts.truncate(most);
        firsts
    }

    /// Whether `pieces` free runs of `run` adjacent chunks would stand
    /// once `own` has been placed first, in order, exactly where the takes
    /// would put it, read only: a [`OwnPiece::Chunks`] as
    /// [`Self::acquire_chunk_runs`] places it (the lowest free runs), a
    /// [`OwnPiece::Slots`] as [`Self::acquire_run`] does (a chunk open for
    /// runs of its length first, else the lowest free chunk, which runs of
    /// its length placed after it share). Nothing is taken: the pool is the
    /// same after.
    #[must_use]
    pub fn chunk_runs_beside(&self, own: &[OwnPiece], run: u32, pieces: u32) -> bool {
        let placement = self.place(own);
        placement.stands() && placement.free_runs(run) >= pieces
    }

    /// The free chunks `takes` (single slots by shape, as
    /// [`Self::acquire`] places them) would open, lowest first, the shapes'
    /// partly filled chunks filled first: read only. The one placement the
    /// planner's map and need read a restore's state by, so they see the
    /// room its unpark sees once its serve has taken that state. `None`
    /// where they do not fit.
    #[must_use]
    pub fn opens_for(&self, takes: &[(ShapeId, u64)]) -> Option<Vec<u32>> {
        let own: Vec<OwnPiece> = takes
            .iter()
            .map(|&(shape, slots)| OwnPiece::Take { shape, slots })
            .collect();
        self.free_beside(&own)
            .map(|(opened, _)| opened.into_iter().collect())
    }

    /// Where `own` (placed in order as the takes would place it) fits at
    /// all, the free chunks it opens: the runs of whole chunks, then the
    /// chunks its runs and takes of slots open beyond the partly filled
    /// ones. `None` where it does not fit: the one placement a restore's
    /// need and its standing are read by ([`super::super::roles::Roles::unpark_need`]).
    #[must_use]
    pub fn opens_beside(&self, own: &[OwnPiece]) -> Option<u32> {
        self.free_beside(own)
            .map(|(opened, _)| u32::try_from(opened.len()).unwrap_or(u32::MAX))
    }

    /// The chunks `own` opens placed in order as the takes would place it,
    /// and the headroom it leaves; `None` where it does not fit.
    /// The chunks `own` opens placed alone, on a pool holding nothing
    /// else, by the store's own placement: whole runs as they are, single
    /// slots of a shape packed into its chunks, runs of slots of one length
    /// and shape sharing chunks of their class (a chunk holds `spc / len`
    /// of them). What a never-fits check reads: [`Self::opens_beside`] is
    /// the room now, and a part placed alone in a chunk of its own each was
    /// no placement (a beam's runs of slots each read as a whole chunk,
    /// past the pool, refusing beams the budget serves).
    #[must_use]
    pub fn opens_alone(&self, own: &[OwnPiece]) -> u32 {
        use std::collections::BTreeMap;
        let mut chunks = 0u64;
        let mut takes: BTreeMap<ShapeId, u64> = BTreeMap::new();
        let mut classes: BTreeMap<(ShapeId, u32), u64> = BTreeMap::new();
        for piece in own {
            match *piece {
                OwnPiece::Chunks { run, pieces } => {
                    chunks += u64::from(run.max(1)) * u64::from(pieces);
                }
                OwnPiece::Take { shape, slots } => *takes.entry(shape).or_default() += slots,
                OwnPiece::Slots { shape, slots } if slots > 0 => {
                    *classes.entry((shape, slots)).or_default() += 1;
                }
                OwnPiece::Slots { .. } => {}
            }
        }
        let spc = |shape: ShapeId| {
            self.shapes
                .get(shape as usize)
                .map_or(1, |state| u64::from(state.spc().max(1)))
        };
        for (shape, slots) in takes {
            chunks += slots.div_ceil(spc(shape));
        }
        for ((shape, len), runs) in classes {
            let per = (spc(shape) / u64::from(len)).max(1);
            chunks += runs.div_ceil(per);
        }
        u32::try_from(chunks).unwrap_or(u32::MAX)
    }

    pub(super) fn free_beside(
        &self,
        own: &[OwnPiece],
    ) -> Option<(std::collections::BTreeSet<u32>, u32)> {
        let placement = self.place(own);
        placement
            .stands()
            .then(|| (placement.opened.clone(), placement.room))
    }

    /// `own` placed in order by the takes themselves ([`Self::acquire`],
    /// [`Self::acquire_run`], [`Self::acquire_chunk_runs`]) on a scratch
    /// copy of the pool: the one placement every check reads, so what
    /// stands here is what the takes find. A view that mirrored the takes
    /// counted a shape's partly filled slots once per take and stood a
    /// pending bind's runs the restore's own takes then broke
    /// (`PendingBindShort`, its unpark asking no run).
    /// It stops at the first piece the takes refuse, as a restore does.
    #[must_use]
    pub fn place(&self, own: &[OwnPiece]) -> Placement {
        let mut scratch = self.scratch();
        let mut placed = 0usize;
        for piece in own {
            let took = match *piece {
                OwnPiece::Take { shape, slots } => scratch.acquire(PLACING, shape, slots).map(drop),
                OwnPiece::Slots { shape, slots } => scratch
                    .acquire_run(PLACING, shape, u64::from(slots))
                    .map(drop),
                // Whole chunks are placed alike whatever shape opens
                // them (no slot of another class shares one): the first.
                OwnPiece::Chunks { run, pieces } => scratch
                    .acquire_chunk_runs(PLACING, 0, run, pieces)
                    .map(drop),
            };
            if took.is_err() {
                break;
            }
            placed += 1;
        }
        let opened = self
            .open
            .iter_in(0, self.total)
            .filter(|&chunk| !scratch.open.get(chunk))
            .collect();
        let room = scratch.headroom();
        Placement {
            opened,
            room,
            placed,
            asked: own.len(),
            scratch,
        }
    }

    /// A copy of the pool for [`Self::place`]: every field a take reads
    /// and writes, none of the bookkeeping no take reads (the slots
    /// waiting on a fire, the holders' counts, the waiters, the fences'
    /// epochs), so a placement costs the chunk tables' copy alone.
    pub(super) fn scratch(&self) -> SlotPool {
        SlotPool {
            chunk_bytes: self.chunk_bytes,
            total: self.total,
            reserved: self.reserved,
            shapes: self.shapes.clone(),
            act_chunks: self.act_chunks.clone(),
            act: self.act,
            carved: self.carved,
            moving_out: BTreeMap::new(),
            fire: self.fire.clone(),
            standing: self.standing,
            readout_row_bytes: self.readout_row_bytes,
            readout_rows: self.readout_rows,
            max_lanes: self.max_lanes,
            act_max: self.act_max,
            top_segment: self.top_segment,
            span_on_demand: self.span_on_demand,
            act_span: self.act_span,
            chunks: self.chunks.clone(),
            logical_to_physical: self.logical_to_physical.clone(),
            physical_to_logical: self.physical_to_logical.clone(),
            free: self.free.clone(),
            open: self.open.clone(),
            fences: BTreeMap::new(),
            fenced: self.fenced.clone(),
            ceiling: self.ceiling,
            segments: self.segments.clone(),
            held: self.held,
            pending: BTreeMap::new(),
            retired: self.retired,
            holders: HashMap::new(),
            waiters: HashMap::new(),
            runs: HashMap::new(),
            compacted: Vec::new(),
            given: std::sync::Arc::default(),
            changes: 0,
            held_for: self.held_for.clone(),
            next_ticket: self.next_ticket,
            // A placement's scratch gives nothing back for real.
            #[cfg(feature = "test-slot-fill")]
            poison: PoisonTap::default(),
        }
    }

    /// Free runs of `run` adjacent chunks inside one segment live state may
    /// take now ([`Self::acquire_chunk_runs`]), counted disjoint.
    #[must_use]
    pub fn free_chunk_runs(&self, run: u32) -> u32 {
        u32::try_from(self.chunk_runs_for(run).len()).unwrap_or(u32::MAX)
    }

    /// Runs of `run` adjacent chunks the segments still to commit hold
    /// below the activation: uncommitted segments under the top rung's
    /// span that the engine binds ([`Self::backs`]), the room
    /// [`super::super::shared::SharedPool::grow_step`] commits into, counted no
    /// further than what the pool may grow to ([`Self::growable_chunks`]).
    /// What a guest bind short of a free run waits on a commit for; none
    /// where no segment is left to commit within the budget.
    #[must_use]
    pub fn committable_runs(&self, run: u32) -> u32 {
        let run = run.max(1);
        if run > self.segments.segment_chunks() {
            return 0;
        }
        let limit = self.state_limit().min(self.ceiling);
        let prefix: u32 = self
            .segments
            .uncommitted()
            .filter(|&segment| segment < self.top_segment && self.backs(segment))
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(limit).saturating_sub(range.start) / run
            })
            .sum();
        let lent: u32 = self
            .lent_unbacked()
            .iter()
            .map(|&(_, chunks)| chunks / run)
            .sum();
        (prefix + lent).min(self.growable_chunks() / run)
    }

    /// `pieces` runs of `run` adjacent free chunks for `holder`, each run
    /// inside one segment, the lowest first, each chunk opened for `shape`
    /// and all its slots taken: refs in order, a run's chunks in physical
    /// order. A run longer than one is kept so (an evacuation moves it
    /// whole) until its first chunk is free again. All or nothing.
    ///
    /// # Errors
    ///
    /// [`SlotError::RunPastSegment`] where `run` is past a segment's
    /// chunks; [`SlotError::NoChunkRun`] where fewer than `pieces` such
    /// runs are free, naming what was.
    pub fn acquire_chunk_runs(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        run: u32,
        pieces: u32,
    ) -> Result<Vec<Slot>, SlotError> {
        let spc = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?
            .spc();
        let run = run.max(1);
        let segment = self.segments.segment_chunks();
        if run > segment {
            return Err(SlotError::RunPastSegment { run, segment });
        }
        let live = holder.is_live();
        let firsts = self.chunk_runs_for(run);
        if firsts.len() < pieces as usize {
            return Err(SlotError::NoChunkRun {
                run,
                pieces,
                free_runs: u32::try_from(firsts.len()).unwrap_or(u32::MAX),
                free_chunks: self.free_chunks(),
            });
        }
        // Each piece inside one segment, whatever found it: on Metal a
        // segment is a buffer of its own, and a piece across its end reads
        // past it ("a guest piece ... lies across the end of its
        // segment").
        if let Some(&first) = firsts[..pieces as usize]
            .iter()
            .find(|&&first| self.segments.of(first) != self.segments.of(first + run - 1))
        {
            return Err(SlotError::RunAcrossSegment { first, run });
        }
        let mut taken = Vec::with_capacity((pieces * run * spc) as usize);
        for &first in &firsts[..pieces as usize] {
            let head = self.physical_to_logical[first as usize];
            for physical in first..first + run {
                let chunk = self.open_at(physical, shape, WHOLE);
                self.fill(chunk, live, u64::from(spc), &mut taken);
            }
            if run > 1 {
                self.runs.insert(head, run);
            }
        }
        if !taken.is_empty() {
            *self.holders.entry(holder).or_default() += taken.len() as u64;
        }
        Ok(taken)
    }

    /// Exactly slot refs `first .. first + n` of `shape` for `holder`, in
    /// ref order: the chunks they lie in, `first` its chunk's slot 0, each
    /// opened for the shape where it is named rather than lowest first (a
    /// last chunk's slots past `n` stay free for the shape). What a load's
    /// own placement is adopted by, its segments committed first (the
    /// load's `commit_for`, which the boot backs): a claim never commits a
    /// segment itself, so nothing opens that the engine was not told to
    /// back. All or nothing.
    ///
    /// # Errors
    ///
    /// [`SlotError::Unclaimable`] where `first` is not a slot 0 or a chunk
    /// is past the pool, reserved, open already, not free, in the
    /// activation span, draining or in a segment not committed; [`SlotError::Budget`] where the chunks
    /// outgrow what is committed.
    pub fn claim(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        first: u32,
        n: u64,
    ) -> Result<Vec<Slot>, SlotError> {
        let range = self.claimable(shape, first, n)?;
        let count = range.end - range.start;
        let needed = self.reserved + self.held + self.act + count;
        let committed = self.committed_chunks();
        if needed > committed {
            return Err(SlotError::Budget { needed, committed });
        }
        let mut taken = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
        for chunk in range {
            let physical = self.logical_to_physical[chunk as usize];
            let opened = self.open_at(physical, shape, SINGLE);
            debug_assert_eq!(opened, chunk);
            self.fill(chunk, holder.is_live(), n - taken.len() as u64, &mut taken);
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Ok(taken)
    }

    /// Whether [`Self::claim`] of the same slots could take their chunks as
    /// the pool stands, budget aside: the logical chunks they lie in.
    ///
    /// # Errors
    ///
    /// As [`Self::claim`]'s [`SlotError::Unclaimable`].
    pub fn claimable(
        &self,
        shape: ShapeId,
        first: u32,
        n: u64,
    ) -> Result<std::ops::Range<u32>, SlotError> {
        let spc = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?
            .spc();
        let start = first / spc;
        if !first.is_multiple_of(spc) {
            return Err(SlotError::Unclaimable {
                chunk: start,
                why: "the first slot is not its chunk's slot 0",
            });
        }
        let count = u32::try_from(n.div_ceil(u64::from(spc))).unwrap_or(u32::MAX);
        let range = start..start.saturating_add(count);
        for chunk in range.clone() {
            if let Some(why) = self.unclaimable(chunk) {
                return Err(SlotError::Unclaimable { chunk, why });
            }
        }
        Ok(range)
    }

    /// Why logical chunk `chunk` cannot be claimed now, if it cannot.
    pub(super) fn unclaimable(&self, chunk: u32) -> Option<&'static str> {
        if chunk >= self.total {
            return Some("past the pool");
        }
        if chunk < self.reserved {
            return Some("reserved");
        }
        if self.chunks[chunk as usize].shape != NO_SHAPE {
            return Some("open for a shape already");
        }
        let physical = self.logical_to_physical[chunk as usize];
        if !self.free.get(physical) || self.fenced.get(physical) {
            Some("not free")
        } else if physical >= self.state_limit() {
            Some("in the activation span")
        } else if matches!(
            self.segments.state(self.segments.of(physical)),
            SegmentState::Draining { .. }
        ) {
            Some("its segment is draining")
        } else if self.segments.state(self.segments.of(physical)) == SegmentState::Uncommitted {
            // Committed here only as the engine backs it: a claim that
            // committed a segment itself opened chunks the engine was never
            // told to back, and a fire on them was refused (Metal's admit:
            // "committed chunks of the state pool").
            Some(if self.segments.of(physical) >= self.top_segment {
                "its span segment is not backed yet (commit_span)"
            } else {
                "its segment is not backed yet (commit_segments)"
            })
        } else {
            None
        }
    }

    /// Adds `holder`'s reference to slots someone already holds.
    #[track_caller]
    pub fn share(&mut self, holder: Holder, slots: &[Slot]) {
        let live = holder.is_live();
        for &slot in slots {
            let (state, at) = self.locate(slot);
            let was = state.refs[at];
            assert!(was.live + was.cache > 0, "{slot:?} is not held");
            let mut now = was;
            if live {
                now.live += 1;
            } else {
                now.cache += 1;
            }
            state.refs[at] = now;
            state.reclassify(was, now);
        }
        if !slots.is_empty() {
            *self.holders.entry(holder).or_default() += slots.len() as u64;
        }
    }

    /// Notes that a fire at `epoch` touches `slots`: once let go, they wait
    /// for it.
    pub fn touch(&mut self, slots: &[Slot], epoch: u64) {
        for &slot in slots {
            let (state, at) = self.locate(slot);
            let refs = &mut state.refs[at];
            refs.touched = refs.touched.max(epoch);
        }
    }
}
