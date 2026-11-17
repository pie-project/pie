//! The state pool's chunks on the device: one address range
//! ([`Range`]), the table that places each logical chunk on a physical one,
//! and the ids the paged kernels read slots by.
//!
//! A lane names slots by slot ref (`logical chunk * slots a chunk + index`).
//! Until the runtime hands those out, a lane names today's page ids, and
//! [`Seating`] seats each on a slot the first time it is named: the lowest
//! free chunk opened for its shape, as the runtime's pool will.

use engine::chunk::{ChunkTable, Evacuation};

use crate::device::elastic::{Commit, PhysicalPool, Range};
use crate::error::{Fault, Result};
use crate::store::layout::{Kind, Layout, SlotShape};

const UNSEATED: u32 = u32::MAX;

/// Today's page ids of one shape, each seated on a slot ref as it is first
/// named.
#[derive(Debug, Default)]
struct Seating {
    refs: Vec<u32>,
    /// The chunk being filled and the next free index in it.
    open: Option<(u32, u32)>,
}

#[derive(Debug)]
pub struct Chunks {
    range: Range,
    layout: Layout,
    /// The chunks state may take: the pool's but the activation's.
    total: u32,
    /// The chunks under the range's end a fire's activation takes.
    act: u32,
    /// The bytes of them asked for: a run laid past these starts in what
    /// the last chunk has left ([`Chunks::widen_activation`]).
    act_asked: u64,
    table: ChunkTable,
    seated: [Seating; Kind::COUNT],
    /// The lowest chunk no shape has opened; chunk 0 is the null chunk.
    next: u32,
    null_zeroed: bool,
    /// Lanes name slot refs: nothing is seated here any more.
    refs: bool,
    /// The chunks the runtime named weight slots in: no page is seated in
    /// one ([`Chunks::keep_clear`]).
    weights: std::collections::BTreeSet<u32>,
    /// The chunk ids of the range the card never backs, what it was
    /// reserved for past what the card came to back ([`Chunks::state_fitted`]):
    /// from `skip.0`, `skip.1` of them. The runtime is told them skipped
    /// (`Reserved::Skipped`): it names none, and every id keeps its place.
    skip: (u32, u32),
}

/// What a run laid in the activation's span past its arenas starts at a
/// multiple of: a device page of any backend.
const RUN_ALIGN: u64 = 1 << 16;

impl Chunks {
    /// Reserves `total` chunks of addresses for state and, above them, the
    /// chunks `act_bytes` of activation take; nothing is backed yet.
    pub fn reserve(
        pool: &PhysicalPool,
        layout: Layout,
        total: u32,
        act_bytes: u64,
    ) -> Result<Chunks> {
        let act = u32::try_from(act_bytes.div_ceil(layout.chunk_bytes.max(1))).unwrap_or(u32::MAX);
        // A flat range (no device memory mapped on demand: ROCm) is one
        // allocation of every chunk at once, so it can hold no more chunks
        // than the pool's ceiling backs; a mapped range only reserves
        // addresses, and the fit decides how many chunks are ever backed.
        let total = if crate::device::elastic::mapped_on_demand() {
            total
        } else {
            total.min(Chunks::ceiling(pool, &layout, act))
        };
        let total = total.max(1);
        let bytes = (u64::from(total) + u64::from(act)) * layout.chunk_bytes;
        let segment = u64::from(layout.segment_chunks) * layout.chunk_bytes;
        Ok(Chunks {
            range: Range::reserve(pool, bytes, segment, "bytes of the state pool")?,
            layout,
            total,
            act,
            act_asked: act_bytes,
            table: ChunkTable::identity(total.saturating_add(act)),
            seated: Default::default(),
            next: 1,
            null_zeroed: false,
            refs: false,
            weights: std::collections::BTreeSet::new(),
            skip: (total, 0),
        })
    }

    /// The chunks the pool's ceiling backs beside an activation of `act`
    /// chunks: the most a range of this layout can ever hold.
    #[must_use]
    pub fn ceiling(pool: &PhysicalPool, layout: &Layout, act: u32) -> u32 {
        let backed =
            pool.hard_pages().saturating_mul(pool.page_bytes()) / layout.chunk_bytes.max(1);
        u32::try_from(backed)
            .unwrap_or(u32::MAX)
            .saturating_sub(act)
    }

    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    #[must_use]
    pub fn base(&self) -> u64 {
        self.range.base()
    }

    #[must_use]
    pub fn total(&self) -> u32 {
        self.total
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.range.bytes()
    }

    /// The bytes the committed segments back; the activation's are apart.
    #[must_use]
    pub fn committed_bytes(&self) -> u64 {
        self.range.segment_bytes_committed()
    }

    /// Every byte the range has backed: its committed segments and held
    /// spans, or, a flat range once allocated, all of it.
    #[must_use]
    pub fn backed_bytes(&self) -> u64 {
        self.range.committed_bytes()
    }

    /// The unit the range is backed in.
    #[must_use]
    pub fn map_unit(&self) -> u64 {
        self.range.map_unit()
    }

    /// The chunks a fire's activation takes, under the range's end.
    #[must_use]
    pub fn act_chunks(&self) -> u32 {
        self.act
    }

    /// The bytes of the activation's span under the range's end.
    #[must_use]
    pub fn activation_bytes(&self) -> u64 {
        u64::from(self.act) * self.layout.chunk_bytes
    }

    /// The bytes of the activation's span not yet held backed: what a
    /// fire's activation will still take beside the state's chunks. None on
    /// a flat range once allocated: its one allocation backs the whole span,
    /// held or not, so no byte of it is free for the cache rows.
    #[must_use]
    pub fn activation_unheld_bytes(&self) -> u64 {
        if !crate::device::elastic::mapped_on_demand() && self.range.committed_bytes() > 0 {
            return 0;
        }
        let chunk = self.layout.chunk_bytes;
        (u64::from(self.act) * chunk).saturating_sub(self.range.held(u64::from(self.total) * chunk))
    }

    /// The bytes of the activation's span its runs were asked
    /// ([`Chunks::reserve`], [`Chunks::widen_activation`]): its chunks are
    /// these, rounded up to a chunk.
    #[must_use]
    pub fn activation_asked(&self) -> u64 {
        self.act_asked
    }

    /// Where the last run held backed in the activation's span ends, from
    /// the span's start.
    #[must_use]
    pub fn activation_held_end(&self) -> u64 {
        self.range
            .held_end_past(u64::from(self.total) * self.layout.chunk_bytes)
    }

    /// The bytes of the activation's span held backed, wherever in it: the
    /// arena's prefix, the carves laid at its end, the workspaces and the
    /// staged runs.
    #[must_use]
    pub fn activation_held_bytes(&self) -> u64 {
        self.range
            .held_past(u64::from(self.total) * self.layout.chunk_bytes)
            .min(self.activation_bytes())
    }

    /// The chunk ids the card never backs ([`Chunks::state_fitted`]).
    fn skipped(&self) -> std::ops::Range<u32> {
        self.skip.0..self.skip.0 + self.skip.1
    }

    /// The chunk id the first `chunks` backed chunks end at: past the
    /// skipped ids where they reach them.
    #[must_use]
    pub fn ids_for(&self, chunks: u32) -> u32 {
        if self.skip.1 > 0 && chunks > self.skip.0 {
            chunks.saturating_add(self.skip.1).min(self.total)
        } else {
            chunks
        }
    }

    /// The chunk id the state's chunks end at: the range's state end, or
    /// where the skipped ids start when they run up to it.
    fn under_state(&self) -> u32 {
        if self.skip.1 > 0 && self.skipped().end >= self.total {
            self.skip.0
        } else {
            self.total
        }
    }

    /// Where the chunks the runtime counts as state end: under it lie the
    /// last of them, in order.
    #[must_use]
    pub fn state_end(&self) -> u64 {
        self.range.base() + u64::from(self.under_state()) * self.layout.chunk_bytes
    }

    /// Where a fire's activation starts: the range's end less its chunks.
    #[must_use]
    pub fn activation_base(&self) -> u64 {
        self.range.base() + u64::from(self.total) * self.layout.chunk_bytes
    }

    /// Backs the first `bytes` of the activation's span, and keeps them
    /// backed; `beside` the pool's budget where it refuses them.
    pub fn hold_activation(
        &mut self,
        pool: &mut PhysicalPool,
        bytes: u64,
        beside: bool,
    ) -> Result<Commit> {
        let from = u64::from(self.total) * self.layout.chunk_bytes;
        let room = u64::from(self.act) * self.layout.chunk_bytes;
        if bytes > room {
            return Err(Fault::Ceiling {
                what: "bytes of activation under the state pool's end",
                need: bytes,
                have: room,
            });
        }
        self.range.hold(pool, from, bytes, beside)
    }

    /// Backs `bytes` of the activation's span from `offset` into it, and
    /// keeps them backed: where a carve laid at the span's top starts.
    pub fn hold_activation_at(
        &mut self,
        pool: &mut PhysicalPool,
        offset: u64,
        bytes: u64,
        beside: bool,
    ) -> Result<Commit> {
        let from = u64::from(self.total) * self.layout.chunk_bytes;
        let room = u64::from(self.act) * self.layout.chunk_bytes;
        if offset.saturating_add(bytes) > room {
            return Err(Fault::Ceiling {
                what: "bytes of activation under the state pool's end",
                need: offset.saturating_add(bytes),
                have: room,
            });
        }
        self.range.hold(pool, from + offset, bytes, beside)
    }

    /// Backs the state's chunks of the segment the runtime's activation
    /// span starts in, and keeps them backed: the runtime counts that
    /// segment committed from the load and opens state below the activation
    /// inside it. The runtime's span starts where its state chunks end
    /// ([`Chunks::stated`]).
    pub fn hold_boundary(&mut self, pool: &mut PhysicalPool) -> Result<Commit> {
        // A size-only walk backs nothing, and zeroes no null chunk it would
        // then count zeroed (`crate::sizing`).
        if crate::sizing::active() {
            return Ok(Commit::Committed);
        }
        let chunk = self.layout.chunk_bytes;
        let start = self.total;
        let segment = self.layout.segment_chunks.max(1);
        let first = start / segment * segment;
        if first >= start {
            return Ok(Commit::Committed);
        }
        // The boundary's chunks but the skipped ids: below them, above
        // them, or both.
        let skipped = self.skipped();
        let mut outcome = Commit::Committed;
        for (from, to) in [
            (first, start.min(skipped.start.max(first))),
            (first.max(skipped.end), start),
        ] {
            if from < to && outcome == Commit::Committed {
                outcome = self.range.hold(
                    pool,
                    u64::from(from) * chunk,
                    u64::from(to - from) * chunk,
                    false,
                )?;
            }
        }
        if outcome == Commit::Committed && first == 0 && !self.null_zeroed {
            crate::device::zero_span(
                self.range.span(0, chunk)?,
                usize::try_from(chunk).unwrap_or(0),
            )?;
            self.null_zeroed = true;
        }
        Ok(outcome)
    }

    /// Holds the state's segments to the pool's first `chunks` chunks: what
    /// the fit left the cache rows of a range reserved for more.
    pub fn limit(&mut self, chunks: u32) {
        self.range
            .limit(u64::from(self.ids_for(chunks)) * self.layout.chunk_bytes);
    }

    /// The chunk a pool of `fitted` chunks parts at ([`Chunks::state_fitted`]):
    /// the first whole segment past the null chunk and every chunk the
    /// load seated weights in, which stay where they lie; `fitted` itself
    /// where that leaves no chunk above it, or the pool is to lie in one
    /// run from its base (`contiguous`: another worker reads it in place).
    #[must_use]
    pub fn parting(&self, fitted: u32, contiguous: bool) -> u32 {
        let fitted = fitted.min(self.total);
        if contiguous || fitted == self.total {
            return fitted;
        }
        self.kept().min(fitted)
    }

    /// The chunks that stay where they lie whatever the pool comes to hold,
    /// in whole segments: the null chunk's and those the load seated
    /// weights in.
    fn kept(&self) -> u32 {
        let segment = self.layout.segment_chunks.max(1);
        let kept = self.weights.last().map_or(1, |last| last + 1);
        kept.div_ceil(segment).saturating_mul(segment)
    }

    /// Gives back the segments past where the pool parts whatever it comes
    /// to hold ([`Chunks::parting`]): what the load's own fires seated
    /// there is read by nothing, and the chunks from there on are placed
    /// under the activation ([`Chunks::state_fitted`]).
    pub fn clear_past_parting(&mut self, pool: &mut PhysicalPool, contiguous: bool) -> Result<()> {
        if contiguous || self.skip.1 > 0 {
            return Ok(());
        }
        let segment = self.layout.segment_chunks.max(1);
        self.range.release_segments(pool, self.kept() / segment)
    }

    /// States the pool to the runtime as the `fitted` chunks the card
    /// backs, of a range reserved for more, with the activation's right
    /// above them. The range's base and the activation's span stay where
    /// they lie (a body recorded over them names both), so the chunks the
    /// range holds past `fitted` are skipped in its middle: the pool's
    /// chunks up to where it parts ([`Chunks::parting`]) lie from the
    /// base, the rest in one run that ends where the activation starts.
    /// The runtime's chunk `fitted + i` is the activation's chunk `i`, so
    /// what a smaller rung lends and the state's last chunks are one span.
    /// The skipped chunks lie under no chunk id the runtime names, and are
    /// never backed. Where the pool is `contiguous`, or too small to part,
    /// the skipped chunks are those right under the activation.
    pub fn state_fitted(&mut self, fitted: u32, contiguous: bool) -> Result<()> {
        let fitted = fitted.min(self.total);
        if self.skip.1 > 0 && self.stated() == fitted {
            return Ok(());
        }
        let parting = self.parting(fitted, contiguous);
        let chunk = self.layout.chunk_bytes;
        if fitted < self.total {
            self.range.part(
                u64::from(parting) * chunk,
                u64::from(self.total - fitted) * chunk,
            )?;
        }
        self.skip = (parting, self.total - fitted);
        self.range.limit(u64::from(self.total) * chunk);
        Ok(())
    }

    /// Whether the pool's chunks lie in one run from the range's base: none
    /// skipped under any of them.
    #[must_use]
    pub fn in_one_run(&self) -> bool {
        self.skip.1 == 0 || self.skipped().end >= self.total
    }

    /// Whether the state's chunks end where the activation starts: only
    /// then is what a smaller rung lends one span with them.
    #[must_use]
    pub fn state_meets_activation(&self) -> bool {
        self.under_state() == self.total
    }

    /// The state chunks the card backs: the range's but those skipped
    /// ([`Chunks::state_fitted`]).
    #[must_use]
    pub fn stated(&self) -> u32 {
        self.total - self.skip.1
    }

    /// Whether the range's chunk `chunk` is backed: in a committed segment,
    /// or in a span held backed.
    #[must_use]
    pub fn backed(&self, chunk: u32) -> bool {
        self.range
            .span(
                u64::from(chunk) * self.layout.chunk_bytes,
                self.layout.chunk_bytes,
            )
            .is_ok()
    }

    /// The chunks a flat range's standing allocation holds.
    fn room_chunks(&self) -> u64 {
        self.range.bytes() / self.layout.chunk_bytes.max(1)
    }

    /// Lays a flat range's chunks out anew inside its standing allocation:
    /// `total` of state and `act` (asked `act_bytes`) of activation above
    /// them, as a fresh reservation would, the activation's span held again
    /// where it then lies.
    fn relay(
        &mut self,
        pool: &mut PhysicalPool,
        total: u32,
        act: u32,
        act_bytes: u64,
        held: u64,
    ) -> Result<()> {
        self.range.unhold();
        self.total = total.max(1);
        self.act = act;
        self.act_asked = act_bytes;
        self.table = ChunkTable::identity(self.total.saturating_add(act));
        self.seated = Default::default();
        self.next = 1;
        self.null_zeroed = false;
        self.refs = false;
        self.weights = std::collections::BTreeSet::new();
        self.skip = (self.total, 0);
        if held > 0 {
            match self.hold_activation(pool, held, true)? {
                Commit::Committed => {}
                _ => {
                    return Err(Fault::program(
                        "store::chunks",
                        "the activation's span was refused the room it held",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Reserves the range anew for `total` chunks of state, before any
    /// segment is backed: the activation's span is backed again where it
    /// then lies.
    pub fn reshape(&mut self, pool: &mut PhysicalPool, total: u32) -> Result<()> {
        if !self.fresh() {
            return Err(Fault::program(
                "store::chunks",
                "the state pool reshaped under committed chunks",
            ));
        }
        let held = self
            .range
            .held(u64::from(self.total) * self.layout.chunk_bytes);
        // The span's runs as asked, not its chunks' bytes: the chunks are
        // these rounded up, and a widening later lays its run past them.
        let act_bytes = self.act_asked;
        // A flat range is allocated once, here at the fit, and laid out
        // anew inside that allocation by every fit after that fits in it;
        // only a fit that needs more (a measuring load's, past the least
        // range it armed over) allocates it again.
        if self.range.flat_allocated()
            && u64::from(total) + u64::from(self.act) <= self.room_chunks()
        {
            return self.relay(pool, total, self.act, self.act_asked, held);
        }
        self.range.release_all(pool);
        let mut fresh = Chunks::reserve(pool, self.layout.clone(), total, act_bytes)?;
        fresh.range.undefer();
        if held > 0 {
            match fresh.hold_activation(pool, held, true)? {
                Commit::Committed => {}
                _ => {
                    return Err(Fault::program(
                        "store::chunks",
                        "the activation's span was refused the room it held",
                    ));
                }
            }
        }
        *self = fresh;
        Ok(())
    }

    /// Reserves the range anew with `bytes` more of activation above what
    /// it was asked: what a load learns its fires take beside their arena
    /// (the kernels' workspaces) once it has walked them. Only while no
    /// segment is backed; the activation it held is laid again. Returns
    /// where the run starts in the span: in what its last chunk had left,
    /// so a run takes whole chunks only past that.
    pub fn widen_activation(&mut self, pool: &mut PhysicalPool, bytes: u64) -> Result<u64> {
        if !self.fresh() {
            return Err(Fault::program(
                "store::chunks",
                "the activation's span widened under committed chunks",
            ));
        }
        let chunk = self.layout.chunk_bytes;
        let held = self.range.held(u64::from(self.total) * chunk);
        let at = self.act_asked.next_multiple_of(RUN_ALIGN);
        let act_bytes = at.saturating_add(bytes);
        let act = u32::try_from(act_bytes.div_ceil(chunk.max(1))).unwrap_or(u32::MAX);
        let allocated = self.range.flat_allocated();
        if allocated && u64::from(self.total) + u64::from(act) <= self.room_chunks() {
            self.relay(pool, self.total, act, act_bytes, held)?;
            return Ok(at);
        }
        self.range.release_all(pool);
        let mut fresh = Chunks::reserve(pool, self.layout.clone(), self.total, act_bytes)?;
        if allocated {
            fresh.range.undefer();
        }
        if held > 0 {
            match fresh.hold_activation(pool, held, true)? {
                Commit::Committed => {}
                _ => {
                    return Err(Fault::program(
                        "store::chunks",
                        "the activation's span was refused the room it held",
                    ));
                }
            }
        }
        *self = fresh;
        Ok(at)
    }

    #[must_use]
    pub fn high_water_bytes(&self) -> u64 {
        self.range.high_water_bytes()
    }

    /// The most the committed segments have backed, the activation's held
    /// spans apart.
    #[must_use]
    pub fn segment_high_water_bytes(&self) -> u64 {
        self.range.segment_high_water_bytes()
    }

    #[must_use]
    pub fn committed_segments(&self) -> u32 {
        self.range.committed_segments()
    }

    /// The segments the state's chunks lie in.
    #[must_use]
    pub fn segments(&self) -> u32 {
        self.segments_for(u64::from(self.total))
    }

    /// Whether no segment is backed yet.
    #[must_use]
    pub fn fresh(&self) -> bool {
        self.range.committed_segments() == 0
    }

    /// The segments that hold the first `chunks` chunks.
    #[must_use]
    pub fn segments_for(&self, chunks: u64) -> u32 {
        u32::try_from(chunks.div_ceil(u64::from(self.layout.segment_chunks.max(1))))
            .unwrap_or(u32::MAX)
            .min(self.range.segments())
    }

    /// The pages backing the first `chunks` chunks would take beyond what
    /// is backed now ([`Range::growth_of`]): as the range lays them now,
    /// or (`placed`, with whether the pool lies in one run) as a pool of
    /// that many would be placed ([`Chunks::state_fitted`]).
    pub fn growth_for(&self, chunks: u32, placed: Option<bool>) -> Result<u64> {
        let chunk = self.layout.chunk_bytes;
        let (end, parted) = match placed {
            Some(contiguous) => {
                let chunks = chunks.min(self.total);
                let parting = self.parting(chunks, contiguous);
                if parting < chunks {
                    (
                        self.total,
                        (
                            u64::from(parting) * chunk,
                            u64::from(self.total - chunks) * chunk,
                        ),
                    )
                } else {
                    (chunks, (u64::MAX, 0))
                }
            }
            None => (self.ids_for(chunks), self.range.parted()),
        };
        self.range.growth_of(
            self.segments_for(u64::from(end)),
            u64::from(end) * chunk,
            parted,
        )
    }

    /// The segments that lie wholly within the first `chunks` chunks.
    #[must_use]
    pub fn segments_under(&self, chunks: u64) -> u32 {
        u32::try_from(chunks / u64::from(self.layout.segment_chunks.max(1)))
            .unwrap_or(u32::MAX)
            .min(self.range.segments())
    }

    /// Backs the first `segments` segments, and zeroes the null chunk once
    /// it is backed: every page behind a window reads it, and a masked
    /// column still multiplies its value, so it must hold no NaN.
    pub fn commit_segments(&mut self, pool: &mut PhysicalPool, segments: u32) -> Result<Commit> {
        // As [`Chunks::hold_boundary`]: a walk counts no chunk zeroed.
        if crate::sizing::active() {
            return Ok(Commit::Committed);
        }
        let outcome = self.range.commit_segments(pool, segments)?;
        if outcome == Commit::Committed && !self.null_zeroed && self.range.committed_segments() > 0
        {
            let bytes = self.layout.chunk_bytes;
            crate::device::zero_span(
                self.range.span(0, bytes)?,
                usize::try_from(bytes).unwrap_or(0),
            )?;
            self.null_zeroed = true;
        }
        Ok(outcome)
    }

    /// Gives back the segments past the first `segments`.
    pub fn release_segments(&mut self, pool: &mut PhysicalPool, segments: u32) -> Result<()> {
        if segments == 0 {
            self.null_zeroed = false;
        }
        self.range.release_segments(pool, segments)
    }

    pub(crate) fn shape(&self, kind: Kind) -> Result<&SlotShape> {
        self.layout.shape(kind).ok_or_else(|| {
            Fault::program(
                "store::chunks",
                format!("a {} named in a load that lays none", kind.name()),
            )
        })
    }

    /// From here on a lane names its slots by slot ref, as the runtime's
    /// pool hands them out: what this pool seated for the load's own fires
    /// is let go, and nothing is seated again.
    pub fn take_refs(&mut self) {
        if !self.refs {
            self.refs = true;
            self.seated = Default::default();
        }
    }

    /// Whether a page this pool seated itself lies in one of `chunks`.
    #[must_use]
    pub fn seated_in(&self, chunks: &std::collections::BTreeSet<u32>) -> bool {
        !self.refs
            && Kind::ALL.iter().any(|kind| {
                let Ok(shape) = self.shape(*kind) else {
                    return false;
                };
                self.seated[kind.index()]
                    .refs
                    .iter()
                    .any(|&slot| slot != UNSEATED && chunks.contains(&(slot / shape.per_chunk)))
            })
    }

    /// Keeps the pages this pool seats itself clear of `chunks`, the ones
    /// the runtime named weight slots in. Where a seated page already lies
    /// in one, every page is let go and seated again as it is next named:
    /// what the pages held is lost, so the caller holds no lane's rows.
    pub fn keep_clear(&mut self, chunks: std::collections::BTreeSet<u32>) {
        if self.seated_in(&chunks) {
            self.seated = Default::default();
            self.next = 1;
        }
        self.weights = chunks;
    }

    /// The chunks under `chunks` the runtime named weight slots in.
    #[must_use]
    pub fn weight_chunks_under(&self, chunks: u32) -> u32 {
        u32::try_from(self.weights.range(..chunks).count()).unwrap_or(u32::MAX)
    }

    /// Whether lanes name slot refs.
    #[must_use]
    pub fn by_refs(&self) -> bool {
        self.refs
    }

    /// Seats page `id` of today's ids on a slot, opening the lowest free
    /// chunk for its shape where the one being filled is full. The windowed
    /// id 0 is the null slot. Returns the physical chunk the slot lies in.
    pub fn seat(&mut self, kind: Kind, id: u32) -> Result<u32> {
        if matches!(kind, Kind::Window(_)) && id == 0 {
            return Ok(0);
        }
        let per_chunk = self.shape(kind)?.per_chunk;
        if self.refs || kind.is_guest() {
            // A slot ref: its chunk is one the runtime opened, in a segment
            // it had backed ahead of this fire.
            let chunk = id / per_chunk;
            let named = self.total.saturating_add(self.act);
            if chunk == 0 || chunk >= named {
                return Err(Fault::Ceiling {
                    what: "chunks of the state pool a slot ref may name",
                    need: u64::from(chunk) + 1,
                    have: u64::from(named),
                });
            }
            return Ok(self.table.physical(chunk).unwrap_or(chunk));
        }
        let stated = self.total;
        let skipped = self.skipped();
        let seating = &mut self.seated[kind.index()];
        if let Some(&slot) = seating.refs.get(id as usize)
            && slot != UNSEATED
        {
            let chunk = slot / per_chunk;
            return Ok(self.table.physical(chunk).unwrap_or(chunk));
        }
        let (chunk, index) = match seating.open {
            Some((chunk, index)) if index < per_chunk => (chunk, index),
            _ => {
                if self.next >= stated {
                    return Err(Fault::Ceiling {
                        what: "chunks of the state pool",
                        need: u64::from(self.next) + 1,
                        have: u64::from(stated),
                    });
                }
                // A chunk the runtime laid weights in is not this pool's
                // to open, nor a skipped one.
                while self.weights.contains(&self.next) || skipped.contains(&self.next) {
                    self.next += 1;
                }
                if self.next >= stated {
                    return Err(Fault::Ceiling {
                        what: "chunks of the state pool clear of the weights' slots",
                        need: u64::from(self.next) + 1,
                        have: u64::from(stated),
                    });
                }
                self.next += 1;
                (self.next - 1, 0)
            }
        };
        seating.open = Some((chunk, index + 1));
        if seating.refs.len() <= id as usize {
            seating.refs.resize(id as usize + 1, UNSEATED);
        }
        // A ref is a u32 the kernels read: past it a product would wrap onto
        // another slot.
        seating.refs[id as usize] = chunk
            .checked_mul(per_chunk)
            .and_then(|first| first.checked_add(index))
            .ok_or(Fault::Ceiling {
                what: "slot refs of the state pool",
                need: u64::from(chunk) * u64::from(per_chunk) + u64::from(index) + 1,
                have: u64::from(u32::MAX),
            })?;
        Ok(self.table.physical(chunk).unwrap_or(chunk))
    }

    /// The slot ref page `id` is seated on.
    pub fn slot_ref(&self, kind: Kind, id: u32) -> Result<u32> {
        if (matches!(kind, Kind::Window(_)) && id == 0) || self.refs || kind.is_guest() {
            return Ok(id);
        }
        self.seated[kind.index()]
            .refs
            .get(id as usize)
            .copied()
            .filter(|&slot| slot != UNSEATED)
            .ok_or_else(|| {
                Fault::program(
                    "store::chunks",
                    format!("{} {id} is read before it is seated", kind.name()),
                )
            })
    }

    /// The range's chunk and the index in it of page `id`'s slot.
    fn place(&self, kind: Kind, id: u32) -> Result<(u32, u32)> {
        let shape = self.shape(kind)?;
        let slot = self.slot_ref(kind, id)?;
        let logical = slot / shape.per_chunk;
        let physical = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total),
        })?;
        Ok((physical, slot % shape.per_chunk))
    }

    /// Bytes from the pool's base to page `id`'s slot.
    pub fn slot_at(&self, kind: Kind, id: u32) -> Result<u64> {
        let (chunk, index) = self.place(kind, id)?;
        Ok(self
            .shape(kind)?
            .slot_at(self.layout.chunk_bytes, chunk, index))
    }

    /// The address of `len` bytes at `offset` of page `id`'s slot.
    pub fn span(&self, kind: Kind, id: u32, offset: u64, len: u64) -> Result<u64> {
        self.range.span(self.slot_at(kind, id)? + offset, len)
    }

    /// The address of weight slot `slot` (a slot ref) of the row-streamed
    /// table `rows`, where its chunk is backed.
    pub fn row_slot(&self, rows: &crate::store::layout::RowsShape, slot: u32) -> Result<u64> {
        self.weight_slot(rows.per_chunk, rows.slot_bytes, slot)
    }

    /// The address of weight slot `slot` (a slot ref) of a weight shape of
    /// `per_chunk` slots of `slot_bytes` a chunk, where its chunk is backed.
    pub fn weight_slot(&self, per_chunk: u32, slot_bytes: u64, slot: u32) -> Result<u64> {
        let logical = slot / per_chunk;
        let physical = self.table.physical(logical).ok_or(Fault::Ceiling {
            what: "chunks of the state pool",
            need: u64::from(logical) + 1,
            have: u64::from(self.total),
        })?;
        self.range.span(
            u64::from(physical) * self.layout.chunk_bytes
                + u64::from(slot % per_chunk) * slot_bytes,
            slot_bytes,
        )
    }

    /// The address of a guest item of `bytes` in `refs`, slot refs of
    /// `role`'s shape: its one slot, or its run of adjacent slots of one
    /// chunk, where the chunk lies now.
    pub fn guest_span(
        &self,
        role: engine::chunk::SlotRole,
        refs: &[u32],
        bytes: u64,
    ) -> Result<u64> {
        let kind = Kind::guest(role).ok_or_else(|| {
            Fault::program("store::chunks", format!("{role:?} is not a guest role"))
        })?;
        let shape = self.shape(kind)?;
        let want = usize::try_from(bytes.div_ceil(shape.slot_bytes)).unwrap_or(usize::MAX);
        let run = refs.windows(2).all(|pair| pair[1] == pair[0] + 1)
            && refs.first().map(|first| first / shape.per_chunk)
                == refs.last().map(|last| last / shape.per_chunk);
        if refs.len() != want || refs.is_empty() || !run {
            return Err(Fault::program(
                "store::chunks",
                format!(
                    "a guest item of {bytes} bytes takes one run of {want} {role:?} slot(s) \
                     of {} bytes in one chunk, and its binding names {refs:?}",
                    shape.slot_bytes
                ),
            ));
        }
        self.span(kind, refs[0], 0, bytes)
    }

    /// The id a paged kernel reads page `id`'s slot by.
    pub fn page_id(&self, kind: Kind, id: u32) -> Result<i32> {
        let (chunk, index) = self.place(kind, id)?;
        let q = self
            .shape(kind)?
            .page_id(self.layout.chunk_bytes, chunk, index);
        i32::try_from(q).map_err(|_| Fault::Ceiling {
            what: "page ids of the state pool",
            need: q,
            have: i32::MAX as u64,
        })
    }

    /// Rewrites today's page ids in `ids` as the ids the kernels read; an id
    /// under zero (a row that writes nothing) stays.
    pub fn page_ids(&self, kind: Kind, ids: &mut [i32]) -> Result<()> {
        for id in ids {
            if *id >= 0 {
                *id = self.page_id(kind, *id as u32)?;
            }
        }
        Ok(())
    }

    /// Moves each logical chunk of `evacuation` onto its new physical chunk:
    /// the bytes in one batched copy on `stream`, then the table, so every
    /// fire enqueued after reads the chunk where it now lies. The runtime
    /// names only free physical chunks as destinations.
    pub fn evacuate(
        &mut self,
        stream: *mut core::ffi::c_void,
        evacuation: &Evacuation,
    ) -> Result<()> {
        let chunk = self.layout.chunk_bytes;
        let mut spans = Vec::with_capacity(evacuation.moves.len());
        for &(logical, to) in &evacuation.moves {
            let from = self.table.physical(logical).ok_or(Fault::Ceiling {
                what: "chunks of the state pool",
                need: u64::from(logical) + 1,
                have: u64::from(self.total),
            })?;
            if from == to {
                continue;
            }
            spans.push((
                self.range.span(u64::from(to) * chunk, chunk)?,
                self.range.span(u64::from(from) * chunk, chunk)?,
                usize::try_from(chunk).unwrap_or(0),
            ));
        }
        let mut table = self.table.clone();
        table
            .apply(evacuation)
            .map_err(|why| Fault::program("store::chunks", why.to_string()))?;
        crate::device::copy_batch_d2d(stream, &spans)?;
        self.table = table;
        Ok(())
    }

    /// The physical chunk logical chunk `logical` lies on.
    #[must_use]
    pub fn physical(&self, logical: u32) -> Option<u32> {
        self.table.physical(logical)
    }
}

impl Chunks {
    /// The pool as the runtime counts it: the range's chunks but those
    /// past the fit ([`Chunks::skip_past`]), the null chunk reserved, each
    /// shape free to take any chunk, and the activation's chunks right
    /// above the state's; `rungs`, each smaller rung's tokens and its
    /// activation's chunks.
    #[must_use]
    pub fn facts(
        &self,
        max_tokens: u32,
        flat: bool,
        rungs: &[(u32, u32)],
    ) -> engine::chunk::ChunkFacts {
        use engine::chunk::{
            ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, SlotKind, SlotShape,
        };
        let layout = &self.layout;
        let shape = |shape: &crate::store::layout::SlotShape, kind: SlotKind| SlotShape {
            id: shape.id,
            kind,
            slot_bytes: shape.slot_bytes,
            slots_per_chunk: shape.per_chunk,
            sets_per_working_set: kind.sets_per_working_set(),
            rows: shape.rows.iter().map(|&(row, _)| row).collect(),
            pinned: None,
            role: engine::chunk::SlotRole::State,
        };
        let mut shapes = Vec::new();
        if let Some(list) = &layout.list {
            shapes.push(shape(
                list,
                SlotKind::List {
                    tokens: list.tokens,
                    rate: 1,
                    window: None,
                },
            ));
        }
        for window in &layout.windows {
            shapes.push(shape(
                window,
                SlotKind::List {
                    tokens: window.tokens,
                    rate: 1,
                    window: window.window,
                },
            ));
        }
        if let Some(rs) = &layout.rs {
            let mut stated = shape(
                &rs.slot,
                SlotKind::List {
                    tokens: rs.slot.tokens,
                    rate: 1,
                    window: None,
                },
            );
            stated.role = engine::chunk::SlotRole::RsBuffer;
            shapes.push(stated);
        }
        if let Some(fixed) = &layout.fixed {
            shapes.push(shape(
                &fixed.slot,
                SlotKind::Fixed {
                    pieces: fixed.pieces,
                },
            ));
        }
        for role in [
            engine::chunk::SlotRole::Instance,
            engine::chunk::SlotRole::Channel,
        ] {
            if let Some(guest) = layout.guest(role) {
                let kind = SlotKind::Fixed { pieces: 1 };
                shapes.push(SlotShape {
                    id: guest.id,
                    kind,
                    slot_bytes: guest.slot_bytes,
                    slots_per_chunk: guest.per_chunk,
                    sets_per_working_set: kind.sets_per_working_set(),
                    rows: Vec::new(),
                    pinned: None,
                    role,
                });
            }
        }
        let total = self.total.saturating_add(self.act);
        let max = u32::try_from(u64::from(total).div_ceil(u64::from(layout.segment_chunks.max(1))))
            .unwrap_or(u32::MAX);
        ChunkFacts {
            tells_rungs: false,
            act_span_chunks: None,
            chunk_bytes: layout.chunk_bytes,
            total_chunks: total,
            segment_chunks: layout.segment_chunks,
            segments: SegmentFacts {
                committed: if flat { max } else { self.committed_segments() },
                max,
                growth: if flat {
                    Growth::Fixed
                } else {
                    Growth::TwoEnded
                },
            },
            reserved: std::iter::once(ReservedRun {
                what: Reserved::Null,
                first: 0,
                chunks: 1,
            })
            .chain((self.skip.1 > 0).then_some(ReservedRun {
                what: Reserved::Skipped,
                first: self.skip.0,
                chunks: self.skip.1,
            }))
            .collect(),
            shapes,
            act_chunks: rungs
                .iter()
                .filter(|&&(tokens, chunks)| tokens < max_tokens && chunks < self.act)
                .copied()
                .chain(std::iter::once((max_tokens, self.act)))
                .collect(),
            eager_rungs: Vec::new(),
            held_bytes: 0,
            outside_bytes: 0,
            host_chunks: 0,
            disk_chunks: 0,
            disk_persistent: false,
            disk_kept: false,
            weights: layout
                .rows
                .iter()
                .map(|rows| engine::chunk::WeightShape {
                    id: rows.id,
                    kind: engine::chunk::WeightKind::Rows,
                    slot_bytes: rows.slot_bytes,
                    slots_per_chunk: rows.per_chunk,
                    count: rows.blocks,
                    floor_slots: rows.floor.min(rows.blocks),
                    name: format!("param {}", rows.param),
                    seated: None,
                })
                .chain(
                    layout
                        .experts
                        .iter()
                        .map(|experts| engine::chunk::WeightShape {
                            id: experts.id,
                            kind: engine::chunk::WeightKind::Expert,
                            slot_bytes: experts.slot_bytes,
                            slots_per_chunk: experts.per_chunk,
                            count: experts.count(),
                            floor_slots: experts.floor.min(experts.count()),
                            name: format!(
                                "experts of param {}",
                                experts
                                    .banks
                                    .first()
                                    .and_then(|l| l.first())
                                    .map_or(0, |p| p.param)
                            ),
                            seated: None,
                        }),
                )
                .collect(),
        }
    }
}

#[cfg(all(test, feature = "device"))]
mod skip_tests {
    use super::*;
    use crate::store::layout::{CHANNEL, CHANNEL_SLOT, INSTANCE, INSTANCE_SLOT};
    use engine::chunk::{Reserved, ReservedRun};

    const CHUNK: u64 = 32 << 20;
    const SEGMENT: u32 = 4;

    fn guest(id: engine::chunk::ShapeId, slot_bytes: u64) -> SlotShape {
        SlotShape {
            id,
            tokens: 0,
            slot_bytes,
            per_chunk: u32::try_from(CHUNK / slot_bytes).expect("slots a chunk"),
            grain: 1,
            window: None,
            rows: Vec::new(),
            pools: Vec::new(),
        }
    }

    /// A range of 16 state chunks in segments of 4, 4 of activation above,
    /// nothing backed.
    fn range(pool: &PhysicalPool) -> Chunks {
        let layout = Layout {
            chunk_bytes: CHUNK,
            segment_chunks: SEGMENT,
            list: None,
            windows: Vec::new(),
            fixed: None,
            rs: None,
            instance: guest(INSTANCE, INSTANCE_SLOT),
            channel: guest(CHANNEL, CHANNEL_SLOT),
            rows: Vec::new(),
            experts: Vec::new(),
        };
        Chunks::reserve(pool, layout, 16, 4 * CHUNK).expect("the range reserves")
    }

    /// Where the runtime's chunk `id` lies, where the card backs it: the
    /// address the engine reads a slot of it at.
    fn address(chunks: &Chunks, id: u32) -> Option<u64> {
        let per_chunk = u32::try_from(CHUNK / INSTANCE_SLOT).expect("slots a chunk");
        chunks
            .weight_slot(per_chunk, INSTANCE_SLOT, id * per_chunk)
            .ok()
    }

    /// Fits a fresh range to `fitted` of its 16 chunks and backs every
    /// segment the runtime may commit: the skipped ids back nothing.
    fn fitted(pool: &mut PhysicalPool, fitted: u32, contiguous: bool) -> Chunks {
        let mut chunks = range(pool);
        chunks
            .state_fitted(fitted, contiguous)
            .expect("the pool fits");
        let segments = chunks.segments_for(u64::from(chunks.total()));
        assert_eq!(
            chunks
                .commit_segments(pool, segments)
                .expect("the segments commit"),
            Commit::Committed
        );
        chunks
    }

    /// The runtime is told the skipped ids as a reserved run, beside the
    /// null chunk, among every id of the range: none is renumbered.
    fn skipped_run(chunks: &Chunks) -> Option<ReservedRun> {
        let facts = chunks.facts(1024, false, &[]);
        assert_eq!(facts.total_chunks, 16 + 4, "every id of the range is named");
        facts
            .reserved
            .iter()
            .copied()
            .find(|run| run.what == Reserved::Skipped)
    }

    /// Every id the runtime may hand out lies at its own place: chunk `id`
    /// at `id` chunks from the base, so each member of a run of adjacent
    /// ids lies a chunk past the one before; a skipped id is backed by
    /// nothing, and no id lies in its place.
    fn ids_keep_their_place(chunks: &Chunks, skip: std::ops::Range<u32>) {
        for id in 1..16 {
            match address(chunks, id) {
                Some(at) => {
                    assert!(!skip.contains(&id), "skipped id {id} is backed");
                    assert_eq!(at, chunks.base() + u64::from(id) * CHUNK, "id {id}");
                }
                None => assert!(skip.contains(&id), "id {id} is not backed"),
            }
        }
        for id in 1..15 {
            if let (Some(a), Some(b)) = (address(chunks, id), address(chunks, id + 1)) {
                assert_eq!(b - a, CHUNK, "ids {id} and {} adjacent", id + 1);
            }
        }
    }

    /// A fitted range whose skip lies mid-segment: a pool of 10 chunks of
    /// 16 stated in one run from its base skips ids 10..16, from the middle
    /// of a segment up to the activation. The runtime is told every id of
    /// the range, ids 10..16 skipped, so a run bound up to the skip's edge
    /// lies below it at adjacent addresses, no id lies in the skip's place,
    /// and the activation's ids above it keep their meaning (id 16 its
    /// first chunk). Parted at a whole segment, the same holds of ids
    /// 4..10. Before, the runtime's ids were renumbered across the skip:
    /// it was told 10 + 4 chunks, and its id 10 was the activation's.
    #[test]
    fn a_run_up_to_a_mid_segment_skip_lies_below_it_and_none_crosses_it() {
        let _context = crate::device::Context::bind(0, None).expect("a device context");
        let mut pool = PhysicalPool::open(0, u64::MAX).expect("the pool opens");
        // Stated in one run from the base (another worker reads it in
        // place): the skip runs mid-segment up to the activation.
        let chunks = fitted(&mut pool, 10, true);
        assert_eq!(chunks.skip, (10, 6));
        assert!(
            !chunks.skip.0.is_multiple_of(SEGMENT),
            "the skip mid-segment"
        );
        assert_eq!(
            skipped_run(&chunks),
            Some(ReservedRun {
                what: Reserved::Skipped,
                first: 10,
                chunks: 6,
            })
        );
        assert!(chunks.in_one_run());
        assert!(
            !chunks.state_meets_activation(),
            "nothing is lent above the skip"
        );
        ids_keep_their_place(&chunks, 10..16);
        drop(chunks);

        // Parted at the first whole segment: ids 4..10 skipped, the state's
        // last chunks (10..16) right under the activation.
        let chunks = fitted(&mut pool, 10, false);
        assert_eq!(chunks.skip, (4, 6));
        assert!(chunks.state_meets_activation());
        assert_eq!(chunks.ids_for(10), 16);
        ids_keep_their_place(&chunks, 4..10);
    }
}
