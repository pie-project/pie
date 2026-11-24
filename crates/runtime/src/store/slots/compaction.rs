//! Compaction: the chunk moves that free a run, applied and undone, and a
//! chunk's move itself.

use super::*;

impl SlotPool {
    /// The fewest chunk moves that stand one free run of `run` adjacent
    /// chunks inside one committed segment, where free chunks are enough
    /// but scattered (a guest bind's run: pages and slots of many
    /// owners interleave across chunks). The window of `run` chunks whose
    /// held chunks are fewest is cleared: each held chunk in it moves to a
    /// free chunk outside it, the lowest first. A chunk moves on its
    /// logical id, so its holders keep their refs and the engine copies its
    /// bytes ([`Pool::evacuate`]). `None` where a run stands already,
    /// where a move this pool made is still fenced (one at a time), or
    /// where no window can be cleared: no free chunks outside it for what
    /// it holds, or something in it that does not move (a reserved chunk,
    /// a run taken whole, a fenced one).
    ///
    /// [`Pool::evacuate`]: ::engine::Pool::evacuate
    #[must_use]
    pub fn plan_compaction(&self, run: u32) -> Option<Vec<ChunkMove>> {
        self.plan_compaction_for(run, 1)
    }

    /// [`Self::plan_compaction`] for a need of `pieces` runs: while fewer
    /// than that stand free, one more is cleared beside those that do,
    /// neither its window nor its moves' destinations in them. Planned only
    /// where no run stands, a head short of a second run with free and
    /// emptiable chunks enough would wait with no move.
    pub fn plan_compaction_for(&self, run: u32, pieces: u32) -> Option<Vec<ChunkMove>> {
        self.plan_compaction_below(run, pieces, self.state_limit())
    }

    /// The runs a raise to `rung` finds no free run below its activation
    /// for, as `(run, pieces)` of the longest: each run of chunks in the
    /// raise's way moves whole to a free run below ([`Self::plan_rung`]),
    /// and where free chunks there stand scattered, none does. `None` where
    /// the raise's moves plan as the pool stands, or its way holds no run.
    #[must_use]
    pub fn raise_runs_short(&self, rung: u32) -> Option<(u32, u32)> {
        let act = self.act_chunks(rung);
        let limit = self.total - act;
        let sources = self.held_in(limit, self.state_limit());
        if sources.is_empty() {
            return None;
        }
        let targets: BTreeSet<u32> = self.open.iter_in(0, limit).collect();
        if self
            .moves_keeping_runs(&sources, targets.clone(), false)
            .is_some()
        {
            return None;
        }
        // The runs in the way, by length: the longest that finds no room is
        // the one a compaction opens first.
        let mut lengths: BTreeMap<u32, u32> = BTreeMap::new();
        for (&head, &run) in &self.runs {
            let at = self.logical_to_physical[head as usize];
            if sources.contains(&at) {
                *lengths.entry(run).or_default() += 1;
            }
        }
        let (&run, &pieces) = lengths.iter().next_back()?;
        let standing = self
            .runs_below(run, u32::MAX, limit, |chunk| targets.contains(&chunk))
            .len();
        (standing < pieces as usize).then_some((run, pieces))
    }

    /// [`Self::plan_compaction_for`] below physical chunk `below`: a raise's
    /// activation limit, where the runs in its way go
    /// ([`Self::raise_runs_short`]).
    pub fn plan_compaction_below(
        &self,
        run: u32,
        pieces: u32,
        below: u32,
    ) -> Option<Vec<ChunkMove>> {
        let run = run.max(1);
        if run > self.segments.segment_chunks() || self.compacting() {
            return None;
        }
        let below = below.min(self.state_limit());
        let standing = self.runs_below(run, u32::MAX, below, |chunk| self.open.get(chunk));
        if standing.len() >= pieces.max(1) as usize {
            return None;
        }
        let kept: HashSet<u32> = standing
            .iter()
            .flat_map(|&first| first..first + run)
            .collect();
        let free_here = |chunk: u32| self.open.get(chunk) && !kept.contains(&chunk);
        // Each run's chunks by physical chunk, its first and its length: a
        // run moves whole, to a free run of its length outside the window
        // ([`Self::moves_keeping_runs`]). Held as unmovable, a pool of runs
        // laid side by side would open no window for another: a restore of
        // three runs of 57 chunks with 99 chunks free would wait for good,
        // its planner finding no step.
        let mut member_of: HashMap<u32, (u32, u32)> = HashMap::new();
        for (&head, &len) in &self.runs {
            let at = self.logical_to_physical[head as usize];
            for k in 0..len {
                member_of.insert(at + k, (at, len));
            }
        }
        // Never a chunk a residency move's copy still reads (an eviction's,
        // parked state's moving down): moved under it, the copy read a chunk
        // handed on. Reserved chunks are free and closed: neither a source
        // (not held) nor a destination (not open).
        // Nor one a move before still copies into or out of (fenced until
        // the fire carrying that copy retires): moved again, the first
        // move's retire unfenced the chunk the second still reads, and it
        // was handed on under that copy.
        let still = |chunk: u32| self.moving_out.contains_key(&chunk) || self.fenced.get(chunk);
        let movable = |chunk: u32| {
            chunk >= self.reserved
                && !self.free.get(chunk)
                && !still(chunk)
                && self.chunks[self.physical_to_logical[chunk as usize] as usize].shape != NO_SHAPE
                && member_of
                    .get(&chunk)
                    .is_none_or(|&(at, len)| (at..at + len).all(|member| !still(member)))
        };
        let limit = below.min(self.ceiling);
        let mut windows: Vec<(u32, u32)> = Vec::new();
        for segment in 0..self.segments.count() {
            if self.segments.state(segment) != SegmentState::Committed {
                continue;
            }
            let range = self.segments.range(segment);
            let end = range.end.min(limit);
            let mut start = range.start;
            while start + run <= end {
                let mut held = 0u32;
                let mut blocked = None;
                for chunk in start..start + run {
                    if kept.contains(&chunk) {
                        blocked = Some(chunk);
                        break;
                    }
                    if self.open.get(chunk) {
                        continue;
                    }
                    if movable(chunk) {
                        held += 1;
                    } else {
                        blocked = Some(chunk);
                        break;
                    }
                }
                if let Some(chunk) = blocked {
                    start = chunk + 1;
                    continue;
                }
                windows.push((held, start));
                start += 1;
            }
        }
        // The window that moves fewest first, each tried in turn: the first
        // whose holders (each run among them whole) find room outside it.
        windows.sort_unstable();
        windows.into_iter().find_map(|(_, start)| {
            let window = start..start + run;
            let mut sources: BTreeSet<u32> = BTreeSet::new();
            for chunk in window.clone().filter(|&chunk| !self.open.get(chunk)) {
                match member_of.get(&chunk) {
                    Some(&(at, len)) => sources.extend(at..at + len),
                    None => {
                        sources.insert(chunk);
                    }
                }
            }
            let targets: BTreeSet<u32> = self
                .open
                .iter_in(0, limit)
                .filter(|&chunk| free_here(chunk) && !window.contains(&chunk))
                .collect();
            let sources: Vec<u32> = sources.into_iter().collect();
            self.moves_keeping_runs(&sources, targets, false)
        })
    }

    /// Carries out `moves` ([`Self::plan_compaction`]'s), which the fire at
    /// `epoch` copies: each chunk moved off stays closed until that fire
    /// retires, and then the run it cleared stands free. All or nothing.
    pub fn apply_compaction(&mut self, moves: &[ChunkMove], epoch: u64) -> Result<(), SlotError> {
        if let Some((head, run, at)) = self.splits_run(moves) {
            return Err(SlotError::SplitsRun { head, run, at });
        }
        let mut targets = BTreeSet::new();
        for step in moves {
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && !self.free.get(step.from)
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(SlotError::StaleCompaction);
            }
        }
        for step in moves {
            // Fenced before the move reopens it: closed until the copy.
            self.fenced.set(step.from);
            self.move_chunk(*step);
            // Its destination too, until the copy into it has landed (see
            // `apply_rung`): never handed to another owner meanwhile.
            self.fenced.set(step.to);
        }
        if !moves.is_empty() {
            self.fences
                .entry(epoch)
                .or_default()
                .extend(moves.iter().flat_map(|step| [step.from, step.to]));
        }
        self.compacted = moves.iter().map(|step| step.from).collect();
        Ok(())
    }

    /// Whether logical chunk `logical` is open for a shape: it holds state.
    #[must_use]
    pub fn holds_chunk(&self, logical: u32) -> bool {
        self.chunks
            .get(logical as usize)
            .is_some_and(|meta| meta.shape != NO_SHAPE)
    }

    /// The moves that take the state in logical chunks `logicals` down to
    /// the lowest free chunks below the state limit: chunks the engine
    /// refused a fire for, its state lying where it will not read or write
    /// it (`engine::Error::Placement`). Chunks no longer held, or already
    /// below the limit and below every chunk free there, move nothing.
    #[must_use]
    pub fn plan_moves_of(&self, logicals: &[u32]) -> Vec<ChunkMove> {
        let limit = self.state_limit().min(self.ceiling);
        let sources: Vec<u32> = logicals
            .iter()
            .filter_map(|&logical| {
                let from = *self.logical_to_physical.get(logical as usize)?;
                (self.chunks[logical as usize].shape != NO_SHAPE
                    && !self.free.get(from)
                    && !self.fenced.get(from))
                .then_some(from)
            })
            .collect();
        let targets: BTreeSet<u32> = self.open.iter_in(0, limit).collect();
        self.moves_keeping_runs(&sources, targets, true)
            .unwrap_or_default()
    }

    /// Whether `moves` would split a run of chunks ([`Self::runs`]), and
    /// how, by name: a run lies at its first chunk's physical chunk and the
    /// next ones, in one segment, and the engine addresses it so (a piece
    /// from its first chunk's first byte); moved, every chunk of it moves,
    /// each to the chunk after its predecessor's destination, in one
    /// segment. Read on the moves alone, whoever planned them: the check
    /// the movers' own placement does not stand in for (runs split by a
    /// raise moving them chunk by chunk).
    #[must_use]
    pub fn splits_run(&self, moves: &[ChunkMove]) -> Option<(u32, u32, u32)> {
        let to_of: HashMap<u32, u32> = moves.iter().map(|step| (step.from, step.to)).collect();
        for (&head, &run) in &self.runs {
            let Some(&start) = self.logical_to_physical.get(head as usize) else {
                continue;
            };
            let moved: Vec<Option<u32>> = (start..start + run)
                .map(|at| to_of.get(&at).copied())
                .collect();
            if moved.iter().all(Option::is_none) {
                continue;
            }
            let Some(first) = moved[0] else {
                let at = moved.iter().position(Option::is_some).unwrap_or(0);
                return Some((head, run, u32::try_from(at).unwrap_or(u32::MAX)));
            };
            for (k, to) in (0..run).zip(&moved) {
                let whole = *to == Some(first + k)
                    && self.segments.of(first + k) == self.segments.of(first);
                if !whole {
                    return Some((head, run, k));
                }
            }
        }
        None
    }

    /// Moves of the chunks at physical `sources` to `targets` (free
    /// chunks), each run of chunks ([`Self::runs`]) moved whole to a free
    /// run of its length, its first chunk first: a member named moves its
    /// whole run with it, never alone. `partial`: what finds no room stays
    /// and the rest moves; else `None` unless all fit.
    pub(super) fn moves_keeping_runs(
        &self,
        sources: &[u32],
        mut targets: BTreeSet<u32>,
        partial: bool,
    ) -> Option<Vec<ChunkMove>> {
        // Each run's chunks by physical chunk: its first and its length.
        let mut member_of: HashMap<u32, (u32, u32)> = HashMap::new();
        for (&head, &run) in &self.runs {
            let Some(&start) = self.logical_to_physical.get(head as usize) else {
                continue;
            };
            for k in 0..run {
                member_of.insert(start + k, (start, run));
            }
        }
        let mut runs: BTreeSet<(u32, u32)> = BTreeSet::new();
        let mut singles: Vec<u32> = Vec::new();
        for &from in sources {
            match member_of.get(&from) {
                Some(&run) => {
                    runs.insert(run);
                }
                None => singles.push(from),
            }
        }
        let mut moves = Vec::new();
        for (start, run) in runs {
            let Some(to) = self.free_run_in(&targets, run) else {
                if partial {
                    continue;
                }
                return None;
            };
            for k in 0..run {
                targets.remove(&(to + k));
                moves.push(ChunkMove {
                    logical: self.physical_to_logical[(start + k) as usize],
                    from: start + k,
                    to: to + k,
                });
            }
        }
        for from in singles {
            let Some(to) = targets.pop_first() else {
                if partial {
                    break;
                }
                return None;
            };
            moves.push(ChunkMove {
                logical: self.physical_to_logical[from as usize],
                from,
                to,
            });
        }
        Some(moves)
    }

    /// Takes back `moves`, applied with [`Self::apply_compaction`] and
    /// refused by the engine before its copy: each chunk goes back to where
    /// it lay, and its fence lifts.
    pub fn undo_compaction(&mut self, moves: &[ChunkMove]) {
        for step in moves.iter().rev() {
            for chunks in self.fences.values_mut() {
                chunks.retain(|&chunk| chunk != step.from && chunk != step.to);
            }
            self.fenced.clear(step.from);
            self.fenced.clear(step.to);
            self.move_chunk(ChunkMove {
                logical: step.logical,
                from: step.to,
                to: step.from,
            });
        }
        self.fences.retain(|_, chunks| !chunks.is_empty());
        self.compacted.clear();
    }
}

impl SlotPool {
    /// Whether the chunks the last compaction moved off are still fenced:
    /// the run it clears is on its way. A rung raise's fence is not one.
    #[must_use]
    pub fn compacting(&self) -> bool {
        self.compacted.iter().any(|&chunk| self.fenced.get(chunk))
    }

    /// Re-fences the chunks a compaction fenced until `from` to `to`: the
    /// engine took the copy and waits it out ([`Self::drains_through`]'s
    /// counterpart for chunk moves).
    pub fn compaction_copied(&mut self, from: u64, to: u64) {
        if from > to {
            self.refence(from, to);
        }
    }

    /// Takes back `evacuation`, applied with [`Self::apply_rung`] and
    /// refused by the engine before its copy: each moved chunk goes back to
    /// where it lay (the engine's table never changed), its fence lifts, and
    /// the activation is `act` again, the rung in hand.
    pub fn undo_rung(&mut self, evacuation: &RungEvacuation, act: u32) {
        for step in evacuation.moves.iter().rev() {
            for chunks in self.fences.values_mut() {
                chunks.retain(|&chunk| chunk != step.from && chunk != step.to);
            }
            self.fenced.clear(step.from);
            self.fenced.clear(step.to);
            self.move_chunk(ChunkMove {
                logical: step.logical,
                from: step.to,
                to: step.from,
            });
        }
        self.fences.retain(|_, chunks| !chunks.is_empty());
        self.set_act(act);
        // The engine refused it: its carve is the one it held.
        self.engine_narrowed();
    }

    /// Moves a held chunk onto free physical chunk `step.to`, as the
    /// engine's `ChunkTable::apply` does: the free logical id there takes the
    /// moved one's old place.
    #[track_caller]
    pub(super) fn move_chunk(&mut self, step: ChunkMove) {
        self.changes += 1;
        let meta = self.chunks[step.logical as usize];
        let displaced = self.physical_to_logical[step.to as usize];
        // A chunk let go while its move was under way (its process ended,
        // or was torn down, before the engine took or refused the copy):
        // its logical id goes back as the engine's table has it, both
        // physical chunks free, no shape to keep count of. Read as a shape,
        // its free id (`NO_SHAPE`) would index past the shapes and panic
        // (on a SIGTERM's teardown).
        if meta.shape == NO_SHAPE {
            self.logical_to_physical[step.logical as usize] = step.to;
            self.logical_to_physical[displaced as usize] = step.from;
            self.physical_to_logical[step.to as usize] = step.logical;
            self.physical_to_logical[step.from as usize] = displaced;
            self.reopen(step.from);
            self.reopen(step.to);
            return;
        }
        self.assert_backed("a move", step.logical, step.to);
        self.open.clear(step.to);
        self.free.clear(step.to);
        self.free.set(step.from);
        self.logical_to_physical[step.logical as usize] = step.to;
        self.logical_to_physical[displaced as usize] = step.from;
        self.physical_to_logical[step.to as usize] = step.logical;
        self.physical_to_logical[step.from as usize] = displaced;
        self.segments.let_go(step.from);
        self.segments.hold(step.to);
        let state = &mut self.shapes[meta.shape as usize];
        let spc = state.spc();
        if meta.used > 0 && meta.used < spc {
            let chunks = state.partial.entry(meta.class).or_default();
            chunks.remove(&(Reverse(meta.used), step.from, step.logical));
            chunks.insert((Reverse(meta.used), step.to, step.logical));
        }
        self.reopen(step.from);
    }
}
