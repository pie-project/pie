//! The pool's sizes: chunks committed, reserved and held for state, the
//! activation's rungs, a fire's and the readouts' bytes, and the runs the
//! pool could hold.

use super::*;

impl SlotPool {
    /// A watch on room given back from now on: what a rise short of free
    /// room waits on.
    #[must_use]
    pub fn room_watch(&self) -> RoomWatch {
        RoomWatch {
            seen: self.given.gives.load(std::sync::atomic::Ordering::Acquire),
            given: std::sync::Arc::clone(&self.given),
        }
    }

    /// The span's chunks under `rung`'s activation that hold state: what a
    /// rise to `rung` must move out ([`Self::plan_rung`]).
    #[must_use]
    pub fn held_in_span(&self, rung: u32) -> u32 {
        let limit = self.total - self.act_chunks(rung);
        u32::try_from(self.held_in(limit, self.state_limit()).len()).unwrap_or(u32::MAX)
    }

    // ---- accounting ----

    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    pub fn total_chunks(&self) -> u32 {
        self.total
    }

    pub fn reserved_chunks(&self) -> u32 {
        self.reserved
    }

    /// Chunks opened for a slot shape, waiting slots included.
    pub fn state_chunks(&self) -> u32 {
        self.held
    }

    /// Chunks with memory under them for the pool's use: the committed
    /// segments' and, where the span is backed on demand, the activation's
    /// in span segments not committed here (the engine backs its carve).
    pub fn committed_chunks(&self) -> u32 {
        self.committed_for(self.act)
    }

    /// [`Self::committed_chunks`] with an activation of `act` chunks.
    pub(super) fn committed_for(&self, act: u32) -> u32 {
        let committed = self.segments.committed_chunks();
        if !self.span_on_demand {
            return committed;
        }
        let from = self.total - act;
        let carve: u32 = (self.top_segment..self.segments.count())
            .filter(|&segment| self.segments.state(segment) == SegmentState::Uncommitted)
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.saturating_sub(range.start.max(from))
            })
            .sum();
        committed + carve
    }

    /// Activation chunks a fire at `rung` tokens takes: the smallest rung at
    /// or above it, else the largest; with the chunks the guest term takes
    /// at that rung's lanes ([`Self::set_fire`]) and the engine's standing
    /// guest bytes ([`Self::set_standing`]), one ceil over the two.
    pub fn act_chunks(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(0, |&(tokens, chunks)| {
                chunks.saturating_add(self.fire_chunks(tokens))
            })
    }

    /// The chunks the guests take at a rung of `tokens`: what the worst
    /// fire of `min(tokens, max_lanes)` lanes lays for the pairs open
    /// ([`::engine::GuestTerm::bytes`]), beside it what the engine keeps
    /// standing for every live instance whatever the fire, and the rows
    /// such a fire reads out where they land beside the activation
    /// ([`Self::readout_bytes`]).
    pub(super) fn fire_chunks(&self, tokens: u32) -> u32 {
        let bytes = self
            .fire
            .bytes(tokens.min(self.max_lanes))
            .saturating_add(self.standing)
            .saturating_add(self.readout_bytes(tokens));
        u32::try_from(bytes.div_ceil(self.chunk_bytes.max(1))).unwrap_or(u32::MAX)
    }

    /// The readout term at a rung of `tokens`: a fire there reads out no
    /// more rows than its tokens, nor than the most one fire reads
    /// ([`Self::set_readouts`]), each taking the bytes a row the engine
    /// states beside the activation; 0 where they land inside the span.
    #[must_use]
    pub fn readout_bytes(&self, tokens: u32) -> u64 {
        u64::from(tokens.min(self.readout_rows)).saturating_mul(self.readout_row_bytes)
    }

    /// The readout term from now on, of the engine's
    /// [`::engine::ReadoutFacts`]: `row_bytes` a row beside the activation
    /// (0: inside the span, no term) for up to `rows` rows a fire (0: no
    /// bound but the fire's tokens). Nothing moves here, as with
    /// [`Self::set_fire`].
    pub fn set_readouts(&mut self, row_bytes: u64, rows: u32) {
        self.readout_row_bytes = row_bytes;
        self.readout_rows = if rows == 0 { u32::MAX } else { rows };
    }

    /// The most rows one fire reads out ([`Self::set_readouts`]):
    /// `u32::MAX` where nothing but its tokens bounds them.
    #[must_use]
    pub fn readout_rows(&self) -> u32 {
        self.readout_rows
    }

    /// The bytes the guest term takes at the rung fires of `rung` tokens
    /// take, before they are rounded to chunks.
    #[must_use]
    pub fn fire_bytes(&self, rung: u32) -> u64 {
        self.fire.bytes(self.rung_of(rung).min(self.max_lanes))
    }

    /// The guest term of the pairs that can be in a fire from now on,
    /// built for fires of up to [`Self::max_lanes`] lanes. Nothing moves
    /// here: a term that raises the rung's need is taken by a raise
    /// ([`Self::plan_rung`], [`Self::apply_rung`]), and one that drops
    /// gives the room back at the next lower ([`Self::set_rung`]).
    /// The term it replaces is handed back.
    pub fn set_fire(&mut self, fire: ::engine::GuestTerm) -> ::engine::GuestTerm {
        std::mem::replace(&mut self.fire, fire)
    }

    /// Makes the guest term the one of `pairs` in place
    /// ([`::engine::GuestTerm::fit`]): whether it changed.
    pub fn fit_fire(&mut self, pairs: &[::engine::FirePair]) -> bool {
        self.fire.fit(pairs, self.max_lanes)
    }

    /// See [`Self::set_fire`].
    pub fn fire(&self) -> &::engine::GuestTerm {
        &self.fire
    }

    /// The engine's standing guest bytes from now on
    /// ([`::engine::Engine::standing_after`]): the span bytes it keeps for
    /// the live instances of registered programs beside every fire, with
    /// what the binds under way may add. Set aside below the carve at
    /// every rung ([`Self::act_chunks`]); nothing moves here, as with
    /// [`Self::set_fire`]. The bytes it replaces are handed back.
    pub fn set_standing(&mut self, bytes: u64) -> u64 {
        std::mem::replace(&mut self.standing, bytes)
    }

    /// See [`Self::set_standing`].
    #[must_use]
    pub fn standing(&self) -> u64 {
        self.standing
    }

    /// Whether the guests take nothing of the span at any rung: no fire
    /// term, nothing standing, and no readout row beside the activation.
    #[must_use]
    pub fn guests_zero(&self) -> bool {
        self.fire.is_zero() && self.standing == 0 && self.readout_row_bytes == 0
    }

    /// The most lanes a fire carries: the engine's `FireLimits::max_lanes`.
    pub fn set_max_lanes(&mut self, lanes: u32) {
        self.max_lanes = lanes.max(1);
    }

    /// See [`Self::set_max_lanes`].
    #[must_use]
    pub fn max_lanes(&self) -> u32 {
        self.max_lanes
    }

    /// The engine states its activation span covers `chunks` from the
    /// range's end as it stands (`ChunkFacts::act_span_chunks`), `None`
    /// where nothing of its own bounds it: the statement in force from
    /// here, read where a rung is fitted ([`Self::act_span`]). Nothing is
    /// kept that an earlier statement decided.
    pub fn set_act_span(&mut self, chunks: Option<u32>) {
        self.act_span = chunks;
    }

    /// See [`Self::set_act_span`].
    #[must_use]
    pub fn act_span(&self) -> Option<u32> {
        self.act_span
    }

    /// The rung fires of `rung` tokens take: the smallest at or above it,
    /// else the largest (`rung` itself where no rung is stated).
    #[must_use]
    pub fn rung_of(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(rung, |&(tokens, _)| tokens)
    }

    /// The activation chunks set aside now.
    pub fn act_reserved(&self) -> u32 {
        self.act
    }

    /// Every rung's tokens, ascending.
    pub fn rungs(&self) -> Vec<u32> {
        self.act_chunks.iter().map(|&(tokens, _)| tokens).collect()
    }

    /// The rung the activation is set aside for now: the largest whose
    /// chunks it holds, the fire term's included (0 where no rung is
    /// stated, or where a fire term registered since leaves it holding
    /// none until a raise).
    pub fn rung(&self) -> u32 {
        self.act_chunks
            .iter()
            .map(|&(tokens, _)| tokens)
            .filter(|&tokens| self.act_chunks(tokens) <= self.act)
            .max()
            .unwrap_or(0)
    }

    /// The largest rung's activation chunks: the top span, committed from
    /// the load.
    pub fn act_max(&self) -> u32 {
        self.act_max
    }

    /// The most disjoint runs of `run` adjacent chunks the pool could ever
    /// hold for state at the smallest rung, each inside one segment past the
    /// reserved chunks: what a need of runs is fitting against alone.
    #[must_use]
    pub fn runs_most(&self, run: u32) -> u32 {
        let run = run.max(1);
        let limit = self.total.saturating_sub(self.act_chunks(0));
        (0..self.segments.count())
            .map(|segment| {
                let range = self.segments.range(segment);
                range
                    .end
                    .min(limit)
                    .saturating_sub(range.start.max(self.reserved))
                    / run
            })
            .sum()
    }

    /// The longest run of adjacent chunks any arrangement of the pool lays
    /// for state: inside one segment, past the reserved chunks, below the
    /// narrowest rung's activation. A run longer is never laid, whatever
    /// moves.
    #[must_use]
    pub fn longest_run_most(&self) -> u32 {
        let limit = self.total.saturating_sub(self.act_chunks(0));
        (0..self.segments.count())
            .map(|segment| {
                let range = self.segments.range(segment);
                range
                    .end
                    .min(limit)
                    .saturating_sub(range.start.max(self.reserved))
            })
            .max()
            .unwrap_or(0)
    }

    /// Held state chunks lying where the engine's state pool does not
    /// hold them under the rung in force, `(logical, physical)`: at or past
    /// the current rung's activation (the carve the activation writes), or
    /// in a segment with no memory under it (a span segment backed on
    /// demand among them). Empty
    /// where every holder's refs read memory the engine backs for state.
    #[must_use]
    pub fn held_unbacked(&self) -> Vec<(u32, u32)> {
        let limit = self.state_limit();
        (0..self.total)
            .filter(|&logical| self.chunks[logical as usize].shape != NO_SHAPE)
            .map(|logical| (logical, self.logical_to_physical[logical as usize]))
            .filter(|&(_, physical)| {
                let segment = self.segments.of(physical);
                physical >= limit || self.segments.state(segment) == SegmentState::Uncommitted
            })
            .collect()
    }

    /// Every chunk opened for a shape, `(logical, physical)`.
    #[must_use]
    pub fn held_physical(&self) -> Vec<(u32, u32)> {
        (0..self.total)
            .filter(|&logical| self.chunks[logical as usize].shape != NO_SHAPE)
            .map(|logical| (logical, self.logical_to_physical[logical as usize]))
            .collect()
    }

    /// Every free chunk a take could open now, `(logical, physical)`: what
    /// the engine must resolve to backed memory at the same physical.
    #[must_use]
    pub fn takeable_physical(&self) -> Vec<(u32, u32)> {
        self.open
            .iter_in(0, self.total)
            .map(|physical| (self.physical_to_logical[physical as usize], physical))
            .collect()
    }

    /// The first segment under the largest rung's activation (the segment
    /// count where activation lives apart).
    pub fn top_segment(&self) -> u32 {
        self.top_segment
    }

    /// State chunks lie at physical ids below this: the activation of the
    /// current rung takes the rest.
    pub fn state_limit(&self) -> u32 {
        self.total - self.act.max(self.carved)
    }

    /// Committed segments below the top span: what `commit_segments(n)`
    /// backs as a prefix, `0..n`.
    pub fn prefix_segments(&self) -> u32 {
        (0..self.top_segment)
            .take_while(|&segment| self.segments.state(segment) != SegmentState::Uncommitted)
            .count() as u32
    }
}
