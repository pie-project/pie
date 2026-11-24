//! The activation span a pool backs on demand: what it lends, commits and
//! releases.

use super::*;

impl SlotPool {
    /// Whether the span is backed on demand ([`SlotPoolFacts::span_on_demand`]).
    pub fn span_on_demand(&self) -> bool {
        self.span_on_demand
    }

    /// The span segments the rung lends and no segment backs yet, each with
    /// the chunks of it the rung lends (below [`Self::state_limit`]),
    /// highest first, the order [`Self::span_to_commit`] backs them. Empty
    /// where the span is backed from the load or the engine holds part of
    /// the range back.
    ///
    /// Span lending. The pool is one range: the top `act_max` chunks are
    /// the largest rung's activation, and a fire at a smaller rung takes
    /// only its own `act_chunks(rung)` of them. The chunks between, below
    /// the state limit, are lent to state (kv pages, records, rs lists,
    /// guest parts alike): they open as any chunk does once their segment
    /// is backed, and a grow for a waiter short of room backs them after
    /// the prefix ([`super::super::shared::SharedPool::grow_step`]). Lending
    /// allocates nothing past load: the span's segments are in the range
    /// reserved at load, backed on demand within it. A fire whose rung
    /// needs lent chunks takes them back first: a raise moves the state
    /// lying there down into free chunks below ([`Self::plan_rung`]); where
    /// too few stand free, [`Self::rung_short`] says how many, and the
    /// planner frees them as it does any head's room (its occupants
    /// evicted or moved down, under the key rule), the raise waiting for
    /// that with no timer. The budget line states the span lent, how much
    /// of it is backed, and the state chunks a raise to the largest rung
    /// takes back.
    pub(super) fn lent_unbacked(&self) -> Vec<(u32, u32)> {
        if !self.span_on_demand || self.ceiling < self.total {
            return Vec::new();
        }
        let limit = self.state_limit();
        (self.top_segment..self.segments.count())
            .rev()
            .filter(|&segment| self.segments.state(segment) == SegmentState::Uncommitted)
            .filter_map(|segment| {
                let range = self.segments.range(segment);
                let lent = range.end.min(limit).saturating_sub(range.start);
                (lent > 0).then_some((segment, lent))
            })
            .collect()
    }

    /// The span the rung lends to state now (`act_max` less the rung's
    /// activation), how much of it is backed, and the state chunks a raise
    /// to the largest rung must take back ([`Self::held_in_span`]).
    #[must_use]
    pub fn span_lent(&self) -> (u32, u32, u32) {
        let lent = self.act_max.saturating_sub(self.act);
        let unbacked: u32 = self.lent_unbacked().iter().map(|&(_, chunks)| chunks).sum();
        let top = self.act_chunks.last().map_or(0, |&(tokens, _)| tokens);
        (lent, lent.saturating_sub(unbacked), self.held_in_span(top))
    }

    /// Span segments committed here, counted from the top of the pool: the
    /// `n` of `commit_span(n)` (0 where the span is backed from the load).
    pub fn span_committed(&self) -> u32 {
        if !self.span_on_demand {
            return 0;
        }
        (self.top_segment..self.segments.count())
            .rev()
            .take_while(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .count() as u32
    }

    /// The span segment counted `n`th from the top (`n` from 1).
    pub(super) fn span_segment(&self, n: u32) -> u32 {
        self.segments.count() - n
    }

    /// The `n` of the next `commit_span(n)` that backs a span chunk the
    /// lowered rung lends to state and no segment backs yet: the highest
    /// such segment's, so the span is backed from the top down. `None`
    /// where every lent span chunk is backed, nothing is lent, or the
    /// engine holds back part of the range (no state opens past it then).
    pub fn span_to_commit(&self) -> Option<u32> {
        self.lent_unbacked()
            .first()
            .map(|&(segment, _)| self.segments.count() - segment)
    }

    /// The engine backed the top `n` span segments (`commit_span(n)`):
    /// their chunks the rung lends open for state.
    pub fn commit_span(&mut self, n: u32) {
        let n = n.min(self.segments.count() - self.top_segment);
        for k in 1..=n {
            let segment = self.span_segment(k);
            if self.segments.state(segment) == SegmentState::Uncommitted {
                self.segments.set_state(segment, SegmentState::Committed);
                for chunk in self.segments.range(segment) {
                    self.reopen(chunk);
                }
            }
        }
    }

    /// The `n` of the next `release_span(n)`: the lowest span segments
    /// committed here that the rung no longer lends (wholly in the
    /// activation, the rung raised past them), hold no state chunk and no
    /// chunk a raise's copy still reads, given back from the bottom up.
    /// `None` where none is.
    pub fn span_to_release(&self) -> Option<u32> {
        let committed = self.span_committed();
        let limit = self.state_limit();
        let mut keep = committed;
        while keep > 0 {
            let segment = self.span_segment(keep);
            let range = self.segments.range(segment);
            let idle = range.start >= limit
                && self.segments.used(segment) == 0
                && range.clone().all(|chunk| !self.fenced.get(chunk));
            if !idle {
                break;
            }
            keep -= 1;
        }
        (keep < committed).then_some(keep)
    }

    /// Gives back the span segments below the top `n` committed here
    /// ([`Self::span_to_release`] says which may go), ahead of the
    /// engine's `release_span(n)`.
    pub fn release_span(&mut self, n: u32) {
        for k in n + 1..=self.span_committed() {
            let segment = self.span_segment(k);
            debug_assert_eq!(self.segments.used(segment), 0);
            self.segments.set_state(segment, SegmentState::Uncommitted);
            for chunk in self.segments.range(segment) {
                self.reopen(chunk);
            }
        }
    }

    // ---- segments ----
}
