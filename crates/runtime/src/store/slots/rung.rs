//! Moving the activation to another rung: the chunks it evacuates, applied
//! and undone.

use super::*;

impl SlotPool {
    pub(super) fn rung_budget(&self, act: u32) -> Result<(), SlotError> {
        let needed = self.reserved + self.held + act;
        let committed = self.committed_for(act);
        if needed > committed {
            return Err(SlotError::Budget { needed, committed });
        }
        Ok(())
    }

    /// State chunks at physical ids `from..to`.
    pub(super) fn held_in(&self, from: u32, to: u32) -> Vec<u32> {
        (from..to)
            .filter(|&chunk| !self.free.get(chunk) && chunk >= self.reserved)
            .collect()
    }

    /// Sets the activation aside for fires up to `rung` tokens; refused when
    /// the state already held leaves too little, or when state sits in the
    /// span the rung's activation takes (raise it with [`Self::plan_rung`]
    /// and [`Self::apply_rung`] then).
    pub fn set_rung(&mut self, rung: u32) -> Result<(), SlotError> {
        let act = self.act_chunks(rung);
        self.rung_budget(act)?;
        let in_way = self.held_in(self.total - act, self.state_limit()).len() as u32;
        if in_way > 0 {
            return Err(SlotError::Occupied {
                rung,
                chunks: in_way,
            });
        }
        self.set_act(act);
        Ok(())
    }

    pub(super) fn set_act(&mut self, act: u32) {
        self.changes += 1;
        let was = self.state_limit();
        self.act = act;
        self.carved = if self.span_on_demand {
            self.carved.max(act)
        } else {
            act
        };
        let now = self.state_limit();
        for chunk in was.min(now)..was.max(now) {
            self.reopen(chunk);
        }
    }

    /// The activation is `act` again, the engine's carve with it, after a
    /// raise whose copy landed and whose carve the engine refused: the
    /// state the copy moved stays where it went.
    pub fn restore_act(&mut self, act: u32) {
        self.set_act(act);
        self.engine_narrowed();
    }

    /// A residency move's copy reads `chunks` (physical) from here until
    /// [`Self::moved_out`]: no raise takes them back meanwhile.
    pub fn moving_out(&mut self, chunks: &[u32]) {
        for &chunk in chunks {
            *self.moving_out.entry(chunk).or_default() += 1;
        }
    }

    /// The copy [`Self::moving_out`] named has landed, or will not.
    pub fn moved_out(&mut self, chunks: &[u32]) {
        for chunk in chunks {
            if let Some(count) = self.moving_out.get_mut(chunk) {
                *count -= 1;
                if *count == 0 {
                    self.moving_out.remove(chunk);
                }
            }
        }
    }

    /// The engine narrowed its carve to the activation set aside now (a
    /// lower's notice landed, `Pool::set_rung`): the span above it opens
    /// for state ([`Self::carved`]).
    pub fn engine_narrowed(&mut self) {
        let was = self.state_limit();
        self.carved = self.act;
        let now = self.state_limit();
        for chunk in was.min(now)..was.max(now) {
            self.reopen(chunk);
        }
    }

    /// The moves that clear rung `rung`'s activation span of state: each
    /// state chunk in it to the lowest free chunk below it. Empty when
    /// nothing is in the way (or the rung takes no more than now).
    pub fn plan_rung(&self, rung: u32) -> Result<RungEvacuation, SlotError> {
        let act = self.act_chunks(rung);
        self.rung_budget(act)?;
        let limit = self.total - act;
        // A move's copy still reads chunks the raise would take back: it
        // waits for the copy to land (its end asks the walk again), never
        // laying its activation over state a copy is reading.
        // Chunks a served restore's reservation keeps there are as taken:
        // laid under the activation, its bind's runs were gone.
        let read = self.moving_out.range(limit..).count()
            + usize::from(self.reserved_in(limit, self.total));
        if read > 0 {
            return Err(SlotError::Occupied {
                rung,
                chunks: u32::try_from(read).unwrap_or(u32::MAX),
            });
        }
        let sources = self.held_in(limit, self.state_limit());
        // A run of chunks moves whole, to a free run below (its first chunk
        // and every chunk after it at the next physical ones): moved chunk
        // by chunk, a run in the span came back split, its first logical
        // chunk's run ending past a gap ("a guest item ... lies in a run of
        // 2 chunk(s) ... ends past a gap", and corrupted reads).
        let targets: BTreeSet<u32> = self.open.iter_in(0, limit).collect();
        let Some(moves) = self.moves_keeping_runs(&sources, targets, false) else {
            return Err(SlotError::NoRoom {
                rung,
                chunks: sources.len() as u32,
            });
        };
        Ok(RungEvacuation {
            rung,
            from: self.rung(),
            act,
            moves,
        })
    }

    /// The chunks raising the activation to `rung` lacks now: past what is
    /// committed, and in free chunks below its span for the state in it.
    /// 0 when the raise fits.
    /// Where `rung`'s activation and the reserved chunks could never fit
    /// the pool, even with all state moved off it: the chunks they take
    /// and the pool's total.
    #[must_use]
    pub fn rung_past(&self, rung: u32, own: u32) -> Option<(u32, u32)> {
        let need = self
            .reserved
            .saturating_add(self.act_chunks(rung))
            .saturating_add(own);
        (need > self.total).then_some((need, self.total))
    }

    pub fn rung_short(&self, rung: u32) -> u32 {
        let act = self.act_chunks(rung);
        let budget = (self.reserved + self.held + act).saturating_sub(self.committed_for(act));
        let limit = self.total - act;
        let sources = self.held_in(limit, self.state_limit()).len() as u32;
        let targets = self.open.iter_in(0, limit).take(sources as usize).count() as u32;
        let short = budget.max(sources - targets);
        // Runs in the way move whole, each to a free run below: free chunks
        // enough, scattered, open none. A compaction below the raise's
        // limit opens them where it can (the walk makes it,
        // [`super::super::shared::SharedPool::compaction_for_raise`]); where none
        // can, what the runs take is short, for the planner to free (the
        // simulation's field seed 603: read as fitting, the raise waited
        // with nothing funded and nothing to move).
        match self.raise_runs_short(rung) {
            Some((run, pieces))
                if !self.compacting()
                    && self.plan_compaction_below(run, pieces, limit).is_none() =>
            {
                let open = self.open.iter_in(0, limit).count() as u32;
                short.max((run * pieces + sources).saturating_sub(open).max(1))
            }
            _ => short,
        }
    }

    /// Carries out `evacuation` and takes its rung; the fire at `epoch`
    /// carries the copy, and the chunks moved off stay closed until it
    /// retires. All or nothing.
    pub fn apply_rung(&mut self, evacuation: &RungEvacuation, epoch: u64) -> Result<(), SlotError> {
        let act = self.act_chunks(evacuation.rung);
        self.rung_budget(act)?;
        let limit = self.total - act;
        if act != evacuation.act {
            return Err(SlotError::StaleRung);
        }
        if let Some((head, run, at)) = self.splits_run(&evacuation.moves) {
            return Err(SlotError::SplitsRun { head, run, at });
        }
        let mut targets = BTreeSet::new();
        let froms: BTreeSet<u32> = evacuation.moves.iter().map(|step| step.from).collect();
        if !self
            .held_in(limit, self.state_limit())
            .iter()
            .all(|chunk| froms.contains(chunk))
        {
            return Err(SlotError::StaleRung);
        }
        for step in &evacuation.moves {
            // A run's chunks below the limit move with it (`plan_rung`).
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && step.to < limit
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(SlotError::StaleRung);
            }
        }
        for step in &evacuation.moves {
            // Fenced before the move reopens it, as a compaction's: a run's
            // member below the span (moved whole with the one in it) was
            // left open while its copy still read it, another owner's to
            // take mid-copy.
            self.fenced.set(step.from);
            self.move_chunk(*step);
            // Its destination too, until the copy into it has landed: let
            // go meanwhile, it would reopen and be handed to another owner
            // whose data the copy then overwrites ("the kernel wrote no
            // status at all").
            self.fenced.set(step.to);
        }
        if !evacuation.moves.is_empty() {
            self.fences.entry(epoch).or_default().extend(
                evacuation
                    .moves
                    .iter()
                    .flat_map(|step| [step.from, step.to]),
            );
        }
        self.set_act(act);
        Ok(())
    }
}
