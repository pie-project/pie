//! The segment walk: a segment's evacuation planned, applied and undone, a
//! segment released or committed, and what a commit ahead takes.

use super::*;

impl SlotPool {
    pub fn segments(&self) -> &Segments {
        &self.segments
    }

    /// The committed segment below the top span holding the fewest chunks,
    /// the highest among equals.
    pub fn emptiest_segment(&self) -> Option<u32> {
        (0..self.top_segment)
            .rev()
            .filter(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .min_by_key(|&segment| self.segments.used(segment))
    }

    /// Moves every held chunk of `segment` into free committed chunks of
    /// lower segments, lowest first, a run of adjacent chunks
    /// ([`Self::acquire_chunk_runs`]) whole into as many adjacent free
    /// chunks of one segment; none when they do not fit there, when the
    /// segment holds reserved chunks, or when the budget would not survive
    /// losing the segment.
    pub fn plan_evacuation(&self, segment: u32) -> Option<Evacuation> {
        if segment >= self.top_segment || self.segments.state(segment) != SegmentState::Committed {
            return None;
        }
        let range = self.segments.range(segment);
        if range.start < self.reserved {
            return None;
        }
        let needed = self.reserved + self.held + self.act;
        if needed > self.committed_chunks() - self.segments.len(segment) {
            return None;
        }
        // A segment a residency copy still reads or writes is never
        // evacuated: the copy's addresses were resolved at its submit, and
        // a segment released (unmapped) under a copy queued on another
        // stream is read unmapped (a device fault on an eviction's
        // kv-to-disk copy).
        if self.moving_out.range(range.start..range.end).next().is_some() {
            return None;
        }
        // Nor one holding chunks a served restore's reservation keeps for
        // its unpark and bind ([`Self::reserve_chunks`]): released, they
        // left the reservation unable to open them, and the bind they were
        // held for came up short, its process suspended again (the
        // simulation's seed 263: a run of three held in a segment the walk
        // gave back).
        if self.reserved_in(range.start, range.end) {
            return None;
        }
        let sources: Vec<u32> = (range.start..range.end)
            .filter(|&chunk| !self.free.get(chunk))
            .collect();
        let mut targets: BTreeSet<u32> = self.open.iter_in(0, range.start).collect();
        let mut moves = Vec::with_capacity(sources.len());
        let mut in_run = BTreeSet::new();
        for &from in &sources {
            let logical = self.physical_to_logical[from as usize];
            let Some(&run) = self.runs.get(&logical) else {
                continue;
            };
            let to = self.free_run_in(&targets, run)?;
            for k in 0..run {
                targets.remove(&(to + k));
                in_run.insert(from + k);
                moves.push(ChunkMove {
                    logical: self.physical_to_logical[(from + k) as usize],
                    from: from + k,
                    to: to + k,
                });
            }
        }
        for &from in sources.iter().filter(|from| !in_run.contains(from)) {
            let to = targets.pop_first()?;
            moves.push(ChunkMove {
                logical: self.physical_to_logical[from as usize],
                from,
                to,
            });
        }
        Some(Evacuation { segment, moves })
    }

    /// The lowest of `free` chunks that opens `run` of them at adjacent
    /// physical ids inside one segment.
    pub(super) fn free_run_in(&self, free: &BTreeSet<u32>, run: u32) -> Option<u32> {
        let mut stretch = 0u32;
        let mut last = None;
        for &chunk in free {
            let adjacent = last.is_some_and(|last: u32| {
                last + 1 == chunk && self.segments.of(last) == self.segments.of(chunk)
            });
            stretch = if adjacent { stretch + 1 } else { 1 };
            last = Some(chunk);
            if stretch == run {
                return Some(chunk + 1 - run);
            }
        }
        None
    }

    /// Carries out `evacuation`, which the fire at `epoch` copies: the
    /// segment then drains, handing nothing out, until that fire retires.
    pub fn apply_evacuation(
        &mut self,
        evacuation: &Evacuation,
        epoch: u64,
    ) -> Result<(), SegmentError> {
        let segment = evacuation.segment;
        self.segments.check(segment)?;
        let stale = SegmentError::Stale(segment);
        if let Some((head, run, at)) = self.splits_run(&evacuation.moves) {
            return Err(SegmentError::SplitsRun {
                segment,
                head,
                run,
                at,
            });
        }
        if segment >= self.top_segment || self.segments.state(segment) != SegmentState::Committed {
            return Err(stale);
        }
        let range = self.segments.range(segment);
        let needed = self.reserved + self.held + self.act;
        let committed = self.committed_chunks() - self.segments.len(segment);
        if needed > committed {
            return Err(SegmentError::Budget {
                segment,
                committed,
                needed,
            });
        }
        if evacuation.moves.len() != self.segments.used(segment) as usize {
            return Err(stale);
        }
        let mut targets = BTreeSet::new();
        for step in &evacuation.moves {
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && range.contains(&step.from)
                && step.to < range.start
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(stale);
            }
        }
        self.segments
            .set_state(segment, SegmentState::Draining { until: epoch });
        for chunk in range {
            self.open.clear(chunk);
        }
        for step in &evacuation.moves {
            self.move_chunk(*step);
        }
        Ok(())
    }

    /// Takes back `evacuation`, applied with [`Self::apply_evacuation`] and
    /// refused by the engine before its copy: each moved chunk goes back to
    /// where it lay (the engine's table never changed), held still or let
    /// go since, and the segment is committed again, its free chunks open.
    pub fn undo_evacuation(&mut self, evacuation: &Evacuation) {
        let segment = evacuation.segment;
        if !matches!(
            self.segments.state(segment),
            SegmentState::Draining { .. }
        ) {
            return;
        }
        self.segments.set_state(segment, SegmentState::Committed);
        for step in evacuation.moves.iter().rev() {
            if self.logical_to_physical[step.logical as usize] != step.to {
                continue;
            }
            let back = ChunkMove {
                logical: step.logical,
                from: step.to,
                to: step.from,
            };
            if self.chunks[step.logical as usize].shape == NO_SHAPE {
                // Let go since: two free chunks change places.
                let displaced = self.physical_to_logical[back.to as usize];
                self.logical_to_physical[back.logical as usize] = back.to;
                self.logical_to_physical[displaced as usize] = back.from;
                self.physical_to_logical[back.to as usize] = back.logical;
                self.physical_to_logical[back.from as usize] = displaced;
            } else {
                self.move_chunk(back);
            }
        }
        for chunk in self.segments.range(segment) {
            self.reopen(chunk);
        }
    }

    /// The draining segment planned to wait past `epoch` waits on it
    /// instead: the engine took its copy with the fires through `epoch`
    /// handed out, the last that may read its chunks.
    pub fn drains_through(&mut self, epoch: u64) {
        for segment in 0..self.segments.count() {
            if let SegmentState::Draining { until } = self.segments.state(segment)
                && until > epoch
            {
                self.segments
                    .set_state(segment, SegmentState::Draining { until: epoch });
            }
        }
    }

    /// Gives `segment` back: a drained one once its evacuation's fire has
    /// retired, or a committed one holding nothing.
    pub fn release_segment(&mut self, segment: u32) -> Result<(), SegmentError> {
        self.segments.check(segment)?;
        if segment >= self.top_segment {
            return Err(SegmentError::State {
                segment,
                state: self.segments.state(segment),
                wanted: "below the activation committed from the load",
            });
        }
        let held = self.segments.used(segment);
        if held > 0 {
            return Err(SegmentError::Held { segment, held });
        }
        // Nor released while a residency copy still reads or writes one of
        // its chunks: it waits for the copy to land ([`Self::moved_out`]).
        let range = self.segments.range(segment);
        // Nor while a served restore's reservation keeps one of its chunks
        // ([`Self::plan_evacuation`]).
        // Nor while a chunk move's copy still reads or writes one of its
        // chunks (fenced until the fire carrying it retires): a compaction
        // moving chunks out leaves the segment empty and committed, and
        // given back then, the copy read memory the engine unmapped.
        let fenced = range.clone().filter(|&chunk| self.fenced.get(chunk)).count();
        let copying = u32::try_from(self.moving_out.range(range.start..range.end).count() + fenced)
            .unwrap_or(u32::MAX)
            .saturating_add(u32::from(self.reserved_in(range.start, range.end)));
        if copying > 0 {
            return Err(SegmentError::Held { segment, held: copying });
        }
        match self.segments.state(segment) {
            SegmentState::Draining { until } if until > self.retired => {
                Err(SegmentError::Draining {
                    segment,
                    until,
                    retired: self.retired,
                })
            }
            SegmentState::Draining { .. } => {
                self.segments.set_state(segment, SegmentState::Uncommitted);
                Ok(())
            }
            SegmentState::Committed => {
                let needed = self.reserved + self.held + self.act;
                let committed = self.committed_chunks() - self.segments.len(segment);
                if needed > committed {
                    return Err(SegmentError::Budget {
                        segment,
                        committed,
                        needed,
                    });
                }
                self.segments.set_state(segment, SegmentState::Uncommitted);
                for chunk in self.segments.range(segment) {
                    self.reopen(chunk);
                }
                Ok(())
            }
            state => Err(SegmentError::State {
                segment,
                state,
                wanted: "drained or empty",
            }),
        }
    }

    /// Marks `segment` committed once the backend has mapped it.
    pub fn commit_segment(&mut self, segment: u32) -> Result<(), SegmentError> {
        self.segments.check(segment)?;
        match self.segments.state(segment) {
            SegmentState::Uncommitted => {
                self.segments.set_state(segment, SegmentState::Committed);
                for chunk in self.segments.range(segment) {
                    debug_assert!(self.free.get(chunk));
                    self.reopen(chunk);
                }
                Ok(())
            }
            state => Err(SegmentError::State {
                segment,
                state,
                wanted: "uncommitted",
            }),
        }
    }

    /// The lowest uncommitted segments to commit so `bytes` of headroom
    /// stand ready, off the fire path; empty when they already do. Never
    /// more than there are.
    pub fn commit_ahead(&self, bytes: u64) -> Vec<u32> {
        let want = bytes.div_ceil(self.chunk_bytes.max(1));
        let mut room = u64::from(self.headroom());
        let mut segments = Vec::new();
        for segment in self
            .segments
            .uncommitted()
            .filter(|&segment| segment < self.top_segment && self.backs(segment))
        {
            if room >= want {
                break;
            }
            room += u64::from(self.segments.len(segment));
            segments.push(segment);
        }
        // The prefix whole, the span the rung lends ([`Self::lent_unbacked`]).
        for (segment, lent) in self.lent_unbacked() {
            if room >= want {
                break;
            }
            room += u64::from(lent);
            segments.push(segment);
        }
        segments
    }
}
