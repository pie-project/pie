//! The pool said in a line: its budget, its census by run, and the
//! invariants a test checks.

use super::*;

impl SlotPool {
    /// The state budget a guest's room is read against, in one line: what
    /// `free_chunks` (the open chunks within the headroom) and
    /// `committable_runs` (segments still to commit within the budget)
    /// stop at. The committed chunks, less the reserved, the held and the
    /// activation's, are the headroom (bounded by the ceiling where it sits
    /// below the pool); the uncommitted segments below the top span, and
    /// what the pool could still grow to, bound what commits.
    #[must_use]
    pub fn budget_line(&self) -> String {
        let uncommitted: Vec<u32> = self
            .segments
            .uncommitted()
            .filter(|&segment| segment < self.top_segment)
            .collect();
        let backed = uncommitted
            .iter()
            .filter(|&&segment| self.backs(segment))
            .count();
        let lent = self.span_lent();
        format!(
            "budget: total={} reserved={} held={} act={} committed={} headroom={} \
             ceiling={} state_limit={} open={} free_chunks={} segments={} top_segment={} \
             uncommitted_below_top={} of_them_backed={} growable={} committable_one_chunk={} \
             span_lent={} span_lent_backed={} span_reclaim_at_top_rung={}",
            self.total,
            self.reserved,
            self.held,
            self.act,
            self.committed_chunks(),
            self.headroom(),
            self.ceiling,
            self.state_limit(),
            self.open.count,
            self.free_chunks(),
            self.segments.count(),
            self.top_segment,
            uncommitted.len(),
            backed,
            self.growable_chunks(),
            self.committable_runs(1),
            lent.0,
            lent.1,
            lent.2,
        )
    }

    /// What stands between free chunks and a run of `run` adjacent ones,
    /// said in one line for a head short of a run (the soaks' "free chunks
    /// and no run" stalls): the budget's terms and its headroom, the free
    /// and open chunks, chunks opened for a shape and holding no slot,
    /// fenced chunks, the segment's length, each committed segment's
    /// longest open stretch (the five longest), the open runs of `run`,
    /// and what a compaction would move.
    #[must_use]
    pub fn census(&self, run: u32) -> String {
        let limit = self.state_limit().min(self.ceiling);
        let free = (0..self.total)
            .filter(|&chunk| self.free.get(chunk))
            .count();
        let past_limit = (limit..self.total)
            .filter(|&chunk| self.free.get(chunk))
            .count();
        let open = self.open.iter_in(0, self.total).count();
        let fenced = (0..self.total)
            .filter(|&chunk| self.fenced.get(chunk))
            .count();
        let empty_opened = self
            .chunks
            .iter()
            .filter(|meta| meta.shape != NO_SHAPE && meta.used == 0)
            .count();
        let mut by_shape: BTreeMap<ShapeId, (u32, u32)> = BTreeMap::new();
        for meta in self.chunks.iter().filter(|meta| meta.shape != NO_SHAPE) {
            let entry = by_shape.entry(meta.shape).or_default();
            entry.0 += 1;
            if meta.used == 0 {
                entry.1 += 1;
            }
        }
        let mut stretches: Vec<(u32, u32)> = Vec::new();
        for segment in 0..self.segments.count() {
            if self.segments.state(segment) != SegmentState::Committed {
                continue;
            }
            let range = self.segments.range(segment);
            let (mut longest, mut stretch) = (0u32, 0u32);
            for chunk in range.start..range.end.min(limit) {
                stretch = if self.open.get(chunk) { stretch + 1 } else { 0 };
                longest = longest.max(stretch);
            }
            stretches.push((longest, segment));
        }
        stretches.sort_unstable_by(|a, b| b.cmp(a));
        stretches.truncate(5);
        let compaction = self.plan_compaction(run).map(|moves| moves.len());
        format!(
            "pool (scope: every chunk): run={run} committed={} reserved={} opened={} act={} \
             headroom={} free_any={free} free_open_for_state={open} \
             free_past_state_limit={past_limit} fenced={fenced} opened_empty={empty_opened} \
             shapes(opened,empty)={by_shape:?} segment_chunks={} state_limit={} ceiling={} \
             longest_open_by_segment={stretches:?} open_runs={} compaction_moves={compaction:?} \
             compacting={}",
            self.committed_chunks(),
            self.reserved,
            self.held,
            self.act,
            self.headroom(),
            self.segments.segment_chunks(),
            self.state_limit(),
            self.ceiling,
            self.free_chunk_runs(run),
            self.compacting(),
        )
    }
}

impl SlotPool {
    /// Checks every internal invariant; for tests.
    #[cfg(test)]
    pub(crate) fn check(&self) {
        for (&head, &run) in &self.runs {
            let start = self.logical_to_physical[head as usize];
            for at in start..start + run {
                assert!(
                    at < self.total
                        && !self.free.get(at)
                        && self.segments.of(at) == self.segments.of(start),
                    "the run of {run} from logical chunk {head} (physical {start}) is split at {at}"
                );
            }
        }
        let mut held = 0;
        let mut seg_used = vec![0u32; self.segments.count() as usize];
        for logical in 0..self.total {
            let physical = self.logical_to_physical[logical as usize];
            let meta = self.chunks[logical as usize];
            assert_eq!(self.physical_to_logical[physical as usize], logical);
            if meta.shape == NO_SHAPE && logical >= self.reserved {
                assert!(self.free.get(physical), "a free logical id on a held chunk");
                continue;
            }
            assert!(!self.free.get(physical));
            assert!(!self.open.get(physical));
            seg_used[self.segments.of(physical) as usize] += 1;
            if logical < self.reserved {
                assert_eq!(physical, logical);
                continue;
            }
            held += 1;
            assert!(meta.used > 0, "an open chunk with nothing out");
            let state = &self.shapes[meta.shape as usize];
            let base = meta.frame as usize * state.words;
            let free: u32 = state.free[base..base + state.words]
                .iter()
                .map(|word| word.count_ones())
                .sum();
            assert_eq!(free + meta.used, state.spc());
            let partial = meta.used < state.spc();
            assert_eq!(
                state
                    .partial
                    .get(&meta.class)
                    .is_some_and(|chunks| chunks.contains(&(
                        Reverse(meta.used),
                        physical,
                        logical
                    ))),
                partial
            );
            assert!(
                matches!(
                    self.segments.state(self.segments.of(physical)),
                    SegmentState::Committed
                ),
                "a held chunk outside a committed segment"
            );
        }
        assert_eq!(held, self.held);
        for segment in 0..self.segments.count() {
            assert_eq!(seg_used[segment as usize], self.segments.used(segment));
            if segment >= self.top_segment && !self.span_on_demand {
                assert_eq!(self.segments.state(segment), SegmentState::Committed);
            }
            for chunk in self.segments.range(segment) {
                assert_eq!(self.open.get(chunk), self.openable(chunk), "chunk {chunk}");
                if !self.free.get(chunk) && chunk >= self.reserved {
                    assert!(chunk < self.state_limit(), "state in the activation span");
                }
            }
        }
        for state in &self.shapes {
            let mut live = 0;
            let mut cached = 0;
            let mut partial_free = 0;
            for (&class, chunks) in &state.partial {
                assert!(!chunks.is_empty(), "an empty class is no entry");
                for &(Reverse(used), physical, logical) in chunks {
                    assert_eq!(self.logical_to_physical[logical as usize], physical);
                    assert_eq!(self.chunks[logical as usize].used, used);
                    assert_eq!(self.chunks[logical as usize].class, class);
                    if class == SINGLE {
                        partial_free += u64::from(state.spc() - used);
                    }
                }
            }
            assert_eq!(partial_free, state.partial_free);
            for logical in 0..self.total {
                let meta = self.chunks[logical as usize];
                if meta.shape == NO_SHAPE || !std::ptr::eq(&self.shapes[meta.shape as usize], state)
                {
                    continue;
                }
                for index in 0..state.spc() {
                    let refs = state.refs[(meta.frame * state.spc() + index) as usize];
                    live += u64::from(refs.live > 0);
                    cached += u64::from(refs.live == 0 && refs.cache > 0);
                }
            }
            assert_eq!(live, state.live);
            assert_eq!(cached, state.cached);
        }
        assert!(self.reserved + self.held + self.act <= self.committed_chunks());
        assert!(self.committed_chunks() <= self.total);
    }
}
