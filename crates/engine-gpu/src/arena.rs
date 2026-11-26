//! A fire's activation: the arena the compiler carves by liveness and the
//! kernel scratch rooms ([`crate::scratch`]), laid together at the top of
//! the state pool (REDESIGN 2.3, 2.4). The span is the pool's top chunks,
//! in its last segment's buffer and never across two; chunk ids are the
//! pool's.
//!
//! A rung (fires of at most so many tokens, and as many lanes) takes the
//! carve laid for its width, anchored at the span's end: the place each of
//! its values and rooms takes is fixed, so a frame recorded at a rung binds
//! the same bytes whatever the pool's rung is, and the span under the
//! current rung's carve is state's to take.
//!
//! A capturing lane's score rows (`poem_ir::Dim::Capturing`,
//! `Placement::Capture`) are no arena value and no part of the span: a fire
//! lays them in the score room of a pair it carries that reads them, state
//! of that pair in the pool (`crate::guest::state`), one key row for each
//! capturing lane in lane order and one page row for each of those whose
//! program reads pages ([`CaptureRows`]).

use kernels_slang::Tensor;
use poem_compiler::{ArenaMap, Budgets, CompiledModel, FireRows, Placement};
use poem_exec::store::arena::rect;
use poem_ir::{Dim, Dtype, Trace, Ty, ValueId};

use crate::backend::{Backend, Tactics};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::run::{Scored, SlotTable};
use crate::scratch::{FireScratch, Rooms};
use crate::store::kv::Paging;

/// What a rung's carve and its rooms are each aligned to: no storage
/// binding a device asks more of (Vulkan's `minStorageBufferOffsetAlignment`
/// is at most 256).
pub const ALIGN: u64 = 256;

/// One rung: fires of at most `tokens` tokens and as many lanes.
#[derive(Debug)]
pub struct Rung {
    pub tokens: u32,
    map: ArenaMap,
    rooms: Rooms,
    /// The rows each value's placement holds in this carve.
    capacity: Vec<u32>,
    /// Where the rooms start, past the carve.
    rooms_at: u64,
    /// The fire's own windowed rows' slots ([`crate::ring`]), past the
    /// rooms: where they start (a unit short of their place at most, the
    /// first slot on the next unit), and how many.
    fire_at: u64,
    fire_slots: u32,
    /// The carve and the rooms.
    bytes: u64,
    /// The span's regions above every carve ([`Span::with_regions`]): a
    /// rung's footprint holds them, its carve lies below them.
    regions: u64,
}

/// The score rows of a load whose trace captures (`Dim::Capturing`): a
/// capturing lane's key row is a mass a key of the load's context, and its
/// page row a mass a kv page of it. No arena slot holds either: a fire lays
/// its capturing lanes' rows in a reading pair's score room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRows {
    /// The keys of the load's context: a key row's f32s.
    pub keys: u32,
    /// The kv pages of that context: a page row's f32s.
    pub pages: u32,
}

/// Which row of a capturing lane a value is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grain {
    Keys,
    Pages,
}

impl CaptureRows {
    /// The rows of `trace`'s captures at a load of `context` keys a lane in
    /// pages of `page_size`; `None` where it captures nothing. A capture
    /// wider than one run of the context is refused by name.
    pub fn of(trace: &Trace, context: u32, page_size: u32) -> Result<Option<CaptureRows>> {
        let mut any = false;
        for (at, decl) in trace.values.iter().enumerate() {
            let Ty::Tensor { shape, dtype } = &decl.ty else {
                continue;
            };
            if shape.first() != Some(&Dim::Capturing) {
                continue;
            }
            match (&shape[1..], dtype) {
                ([Dim::ContextTimes(1) | Dim::ContextPages], Dtype::F32) => any = true,
                _ => {
                    return Err(Fault::Program {
                        at: "arena::CaptureRows::of",
                        why: format!(
                            "value {at} is a capturing lane's row of {shape:?} {dtype:?}; this \
                             engine lays an f32 row of one run of the load's context, or of \
                             its pages"
                        ),
                    });
                }
            }
        }
        Ok(any.then(|| CaptureRows {
            keys: context,
            pages: context.div_ceil(page_size.max(1)),
        }))
    }

    /// The bytes of one key row.
    #[must_use]
    pub fn key_bytes(&self) -> u64 {
        u64::from(self.keys) * 4
    }

    /// The bytes of one page row.
    #[must_use]
    pub fn page_bytes(&self) -> u64 {
        u64::from(self.pages) * 4
    }

    /// Where the page rows start in a room of `keys` key rows and `pages`
    /// page rows, and the room's bytes: the key rows first, each run
    /// aligned.
    #[must_use]
    pub fn region(&self, keys: u32, pages: u32) -> (u64, u64) {
        let pages_at = (u64::from(keys) * self.key_bytes()).next_multiple_of(ALIGN);
        let pages = (u64::from(pages) * self.page_bytes()).next_multiple_of(ALIGN);
        (pages_at, if keys == 0 { 0 } else { pages_at + pages })
    }
}

/// Which capturing row `value` is, where `map` places it as one.
#[must_use]
pub fn grain(map: &ArenaMap, value: ValueId) -> Option<Grain> {
    match map.placements.get(map.root(value).0 as usize)? {
        Placement::Capture {
            width: Dim::ContextPages,
            ..
        } => Some(Grain::Pages),
        Placement::Capture { .. } => Some(Grain::Keys),
        _ => None,
    }
}

impl Rung {
    #[allow(clippy::too_many_arguments)]
    fn lay<T: Tactics + ?Sized>(
        trace: &Trace,
        compiled: &CompiledModel,
        budgets: &Budgets,
        tokens: u32,
        paging: Paging,
        tuning: &T,
        routers: &[u32],
        planes: &[u64],
        max_buffer: u64,
    ) -> Result<Rung> {
        let map = if tokens >= budgets.tokens.max_tokens {
            compiled.arena.clone()
        } else {
            poem_compiler::arena_rung(trace, budgets, compiled, tokens).map_err(|why| {
                Fault::Program {
                    at: "arena::Rung::lay",
                    why: format!("the carve for fires of {tokens} tokens: {why}"),
                }
            })?
        };
        let rooms = Rooms::lay(
            trace, compiled, &map, budgets, tokens, paging, tuning, routers, planes, max_buffer,
        );
        let rooms_at = map.bytes.next_multiple_of(ALIGN);
        let bytes = (rooms_at + rooms.bytes()).next_multiple_of(ALIGN);
        Ok(Rung {
            tokens,
            capacity: capacities(&map),
            map,
            rooms,
            rooms_at,
            fire_at: bytes,
            fire_slots: 0,
            bytes,
            regions: 0,
        })
    }

    /// This rung with room for its fires' own windowed rows: the slots a
    /// fire of its tokens and lanes takes, `slot_bytes` each, the first on
    /// a `unit` (CONTRACT-BC 19: a fire's activation holds its own folded
    /// rows' K/V for its reads).
    fn with_fire_slots(&mut self, tokens_a_slot: u32, lanes: u32, slot_bytes: u64, unit: u64) {
        // A lane's rows from a slot's first row: one slot more a lane.
        let slots = self
            .tokens
            .div_ceil(tokens_a_slot.max(1))
            .saturating_add(self.tokens.min(lanes));
        let at = self.bytes.next_multiple_of(ALIGN);
        self.fire_at = at;
        self.fire_slots = slots;
        self.bytes = (at + u64::from(slots) * slot_bytes + unit).next_multiple_of(ALIGN);
    }

    /// The rung's footprint: its carve and rooms, aligned, and the span's
    /// regions above them.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes + self.regions
    }

    /// Whether a fire of `rows` carrying `carried` recurrent rows ahead of
    /// its own fits this rung: its own rows and its carried rows each at
    /// most the rung's tokens (`engine::fire::ActivationRows`).
    fn holds(&self, rows: FireRows, carried: u32) -> bool {
        let tokens = u64::from(self.tokens);
        rows.tokens <= tokens
            && rows.lanes <= tokens
            && rows.readouts <= tokens
            && carried <= self.tokens
    }
}

/// The activation span: every rung's carve, and the buffer and place the
/// pool gave it.
#[derive(Debug)]
pub struct Span<B: Backend> {
    /// Ascending; the last is the top fire's (the compiler's own carve).
    rungs: Vec<Rung>,
    /// For each rung, the carve its fires take ([`Span::widen`]).
    widest: Vec<usize>,
    routers: Vec<u32>,
    /// The score rows a capturing lane takes in a pair's score room, where
    /// the trace captures.
    capture: Option<CaptureRows>,
    /// The pool's last segment and the byte in it the span ends at: the
    /// segment is the span a fire's carve may take, and the pool states it
    /// so (`ChunkFacts::act_span_chunks`).
    place: Option<(B::Buffer, u64)>,
    /// The fire-class reservations laid at the span's top, above every
    /// carve, each its owner's and aligned bytes ([`Span::with_regions`]).
    regions: Vec<(crate::books::Owner, u64)>,
}

impl<B: Backend> Span<B> {
    /// The carve and rooms of every rung of `tokens` (ascending, the last
    /// `budgets`' own ceiling); nothing placed yet.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        trace: &Trace,
        compiled: &CompiledModel,
        budgets: &Budgets,
        tokens: &[u32],
        paging: Paging,
        tuning: &B::Tuning,
        max_buffer: u64,
    ) -> Result<Span<B>> {
        let routers = crate::scratch::routers(trace);
        // A projection whose bank outgrows one binding lands its logits in
        // a room of the rung's, a run of rows at a time.
        let planes = crate::weights::plane_bytes(trace)?;
        let capture = CaptureRows::of(trace, paging.context(), paging.page_size)?;
        let rungs = tokens
            .iter()
            .map(|&tokens| {
                Rung::lay(
                    trace, compiled, budgets, tokens, paging, tuning, &routers, &planes, max_buffer,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Span {
            widest: (0..rungs.len()).collect(),
            rungs,
            routers,
            capture,
            place: None,
            regions: Vec::new(),
        })
    }

    #[must_use]
    pub fn rungs(&self) -> &[Rung] {
        &self.rungs
    }

    /// The bytes the widest rung takes: the span's.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.rungs.iter().map(Rung::bytes).max().unwrap_or(0)
    }

    /// The score rows a capturing lane takes in a pair's score room;
    /// `None` where the trace captures nothing.
    #[must_use]
    pub fn capture(&self) -> Option<CaptureRows> {
        self.capture
    }

    /// `(rung tokens, chunks)` of every rung, chunks of `chunk_bytes`,
    /// ascending in both (`ChunkFacts::act_chunks`).
    #[must_use]
    pub fn act_chunks(&self, chunk_bytes: u64) -> Vec<(u32, u32)> {
        let mut most = 0u32;
        self.rungs
            .iter()
            .map(|rung| {
                let chunks =
                    u32::try_from(rung.bytes().div_ceil(chunk_bytes.max(1))).unwrap_or(u32::MAX);
                most = most.max(chunks);
                (rung.tokens, most)
            })
            .collect()
    }

    /// Lays the span in `buffer`, ending at its byte `end`.
    pub fn place(&mut self, buffer: B::Buffer, end: u64) -> Result<()> {
        let end = end - end % ALIGN;
        if end < self.bytes() {
            return Err(Fault::Ceiling {
                what: "bytes of the activation span in the pool's last segment",
                need: self.bytes(),
                have: end,
            });
        }
        self.place = Some((buffer, end));
        Ok(())
    }

    fn placed(&self) -> Result<(&B::Buffer, u64)> {
        self.place
            .as_ref()
            .map(|(buffer, end)| (buffer, *end))
            .ok_or_else(|| Fault::Unbound {
                what: "the activation span, which the pool has not placed".to_string(),
            })
    }

    /// The buffer the span lies in.
    pub fn buffer(&self) -> Result<&B::Buffer> {
        Ok(self.placed()?.0)
    }

    /// The carve a fire of `rows` carrying `carried` recurrent rows takes:
    /// the widest whose bytes lie in the chunks stated for the narrowest
    /// rung that holds it ([`Span::widen`]).
    #[must_use]
    pub fn rung_of(&self, rows: FireRows, carried: u32) -> usize {
        let narrowest = self
            .rungs
            .iter()
            .position(|rung| rung.holds(rows, carried))
            .unwrap_or(self.rungs.len().saturating_sub(1));
        self.widest.get(narrowest).copied().unwrap_or(narrowest)
    }

    /// Takes the rungs the pool states, `(tokens, chunks)` of `chunk_bytes`
    /// each class once: a fire then carves the widest carve that lies in
    /// its rung's stated chunks, so where rungs share their chunks (all of
    /// them, where the span lies past the kernels' state index) it lays
    /// out as at the top fire and lends nothing it would not.
    pub fn widen(&mut self, stated: &[(u32, u32)], chunk_bytes: u64) {
        let room = |tokens: u32| {
            stated
                .iter()
                .find(|&&(rung, _)| rung >= tokens)
                .or(stated.last())
                .map_or(0, |&(_, chunks)| u64::from(chunks) * chunk_bytes)
        };
        self.widest = (0..self.rungs.len())
            .map(|at| {
                let room = room(self.rungs[at].tokens);
                (at..self.rungs.len())
                    .rev()
                    .find(|&wide| self.rungs[wide].bytes() <= room)
                    .unwrap_or(at)
            })
            .collect();
    }

    /// Where rung `rung`'s carve starts in the span's buffer: the span's
    /// end less its bytes.
    pub fn base(&self, rung: usize) -> Result<u64> {
        let (_, end) = self.placed()?;
        Ok(end - self.rungs[rung].bytes())
    }

    /// Room in every rung for its fires' own windowed rows
    /// ([`Rung::with_fire_slots`]); before the pool sizes the span.
    pub fn with_fire_slots(&mut self, tokens_a_slot: u32, lanes: u32, slot_bytes: u64, unit: u64) {
        for rung in &mut self.rungs {
            rung.with_fire_slots(tokens_a_slot, lanes, slot_bytes, unit);
        }
    }

    /// Lays `regions` (each an owner's reservation of so many bytes) at
    /// the span's top, above every rung's carve: each rung's footprint
    /// grows by their bytes, each aligned, so the pool's span holds them
    /// at every rung and no rung lends them to state. After every other
    /// part of the rungs is laid ([`Span::with_fire_slots`]), before the
    /// pool sizes the span.
    pub fn with_regions(&mut self, regions: &[(crate::books::Owner, u64)]) {
        self.regions = regions
            .iter()
            .map(|&(owner, bytes)| (owner, bytes.max(1).next_multiple_of(ALIGN)))
            .collect();
        let bytes: u64 = self.regions.iter().map(|&(_, bytes)| bytes).sum();
        for rung in &mut self.rungs {
            rung.regions = bytes;
        }
    }

    /// The bytes the span's regions take at its top.
    #[must_use]
    pub fn region_bytes(&self) -> u64 {
        self.regions.iter().map(|&(_, bytes)| bytes).sum()
    }

    /// Each region's owner and aligned bytes, in the order laid.
    #[must_use]
    pub fn region_owners(&self) -> &[(crate::books::Owner, u64)] {
        &self.regions
    }

    /// The regions as laid in the placed span, in the order stated
    /// ([`Span::with_regions`]): the first right above the carves, the last
    /// at the span's end.
    pub fn regions(&self) -> Result<Vec<crate::region::Region<B>>> {
        let (buffer, end) = self.placed()?;
        let mut at = end - self.region_bytes();
        self.regions
            .iter()
            .map(|&(owner, bytes)| {
                let region = crate::region::Region::new(buffer.clone(), at, bytes, owner);
                at += bytes;
                region
            })
            .collect()
    }

    /// Each rung's tokens and the bytes its fire slots take in its carve
    /// (what its stated activation chunks count for them).
    #[must_use]
    pub fn fire_bytes(&self, slot_bytes: u64) -> Vec<(u32, u64)> {
        self.rungs
            .iter()
            .map(|rung| (rung.tokens, u64::from(rung.fire_slots) * slot_bytes))
            .collect()
    }

    /// The tokens rung `rung`'s fires bring at most.
    #[must_use]
    pub fn rung_tokens(&self, rung: usize) -> u32 {
        self.rungs.get(rung).map_or(0, |at| at.tokens)
    }

    /// Rung `rung`'s fire slots: the byte of the span's buffer they start
    /// at (before the unit's rounding) and how many.
    pub fn fire_slots(&self, rung: usize) -> Result<(u64, u32)> {
        let at = &self.rungs[rung];
        Ok((self.base(rung)? + at.fire_at, at.fire_slots))
    }

    /// A fire's slots at rung `rung`: its carve at its place and, where a
    /// lane of it captures (`scored`), its capturing lanes' rows in `room`
    /// (a reading pair's score room: its segment buffer, first byte and
    /// bytes): a key row for each of the fire's capturing lanes and a page
    /// row for each of those whose program reads pages. Every layer's
    /// capture is the one key row it adds into; a fire none of whose lanes
    /// captures binds no row.
    pub fn slots(
        &self,
        handles: &Handles<B>,
        rung: usize,
        rows: FireRows,
        scored: Scored,
        room: Option<(&B::Buffer, u64, u64)>,
    ) -> Result<SlotTable> {
        let (buffer, end) = self.placed()?;
        let at = &self.rungs[rung];
        let base = end - at.bytes();
        let captured = match self.capture {
            Some(capture) if scored.rows() > 0 => {
                let (pages_at, region) = capture.region(scored.rows(), scored.page_rows());
                let Some((room, room_at, room_bytes)) = room else {
                    return Err(Fault::Unbound {
                        what: format!(
                            "a score room for the {} capturing lane(s) of a fire, which no \
                             lane's pair keeps: a lane captures only where a program it \
                             runs reads its scores",
                            scored.rows()
                        ),
                    });
                };
                if region > room_bytes {
                    return Err(Fault::Ceiling {
                        what: "bytes of a pair's score room for a fire's capturing lanes",
                        need: region,
                        have: room_bytes,
                    });
                }
                let keys = Tensor::new(
                    handles.bind(
                        room,
                        room_at,
                        u64::from(scored.rows()) * capture.key_bytes(),
                    )?,
                    scored.rows(),
                    capture.keys,
                    Dtype::F32,
                );
                let pages = (scored.page_rows() > 0)
                    .then(|| {
                        Ok::<_, Fault>(Tensor::new(
                            handles.bind(
                                room,
                                room_at + pages_at,
                                u64::from(scored.page_rows()) * capture.page_bytes(),
                            )?,
                            scored.page_rows(),
                            capture.pages,
                            Dtype::F32,
                        ))
                    })
                    .transpose()?;
                Some((keys, pages))
            }
            _ => None,
        };
        let mut slots: Vec<Option<Tensor>> = Vec::with_capacity(at.map.placements.len());
        for value in 0..at.map.placements.len() {
            let value = ValueId(value as u32);
            // A fire that captures nothing lays no row: a capture's value
            // has no rectangle, so it binds none.
            let grain = captured.and_then(|_| grain(&at.map, value));
            slots.push(match (grain, captured) {
                (Some(Grain::Keys), Some((keys, _))) => Some(keys),
                (Some(Grain::Pages), Some((_, pages))) => pages,
                (Some(_), None) => None,
                (None, _) => match rect(&at.map, value, rows) {
                    // Bound at its room, what the rung's map lays for it, not
                    // at the rows this fire brings: a view then depends on
                    // where the value lies, so what a dispatch binds over it
                    // is one for every fire of the rung. The tensor's rows
                    // are the fire's own.
                    Some(rect) if rect.bytes > rect.room => {
                        return Err(Fault::Ceiling {
                            what: "bytes of a value's room at its rung a fire's rows take",
                            need: rect.bytes,
                            have: rect.room,
                        });
                    }
                    // The carve lies in the chunks the rung holds, and the
                    // span below it is lent to state (a guest's among it):
                    // a room past the carve would reach that state.
                    Some(rect) if rect.offset + rect.room > at.map.bytes => {
                        return Err(Fault::Ceiling {
                            what: "bytes of a rung's carve a value's room reaches",
                            need: rect.offset + rect.room,
                            have: at.map.bytes,
                        });
                    }
                    Some(rect) => Some(Tensor::new(
                        handles.bind(buffer, base + rect.offset, rect.room)?,
                        rect.rows,
                        rect.width,
                        rect.dtype,
                    )),
                    None => None,
                },
            });
        }
        Ok(SlotTable(slots, scored))
    }

    /// A fire's rooms at rung `rung`.
    pub fn scratch(&self, rung: usize) -> Result<FireScratch<'_, B>> {
        let (buffer, _) = self.placed()?;
        let at = &self.rungs[rung];
        Ok(FireScratch::new(
            &at.rooms,
            buffer,
            self.base(rung)? + at.rooms_at,
            &at.capacity,
            &self.routers,
        ))
    }
}

/// The top rung among `rungs` (`(tokens, bytes)`, the most any one of the
/// rung's reservations takes: its span, an arm's readout seat): the widest
/// whose every reservation lies in one `binding`, as the span lies in the
/// pool's last segment, no segment is past a binding, and no buffer is past
/// the device's largest. `None`: not even the least does.
#[must_use]
pub fn bound_top(rungs: &[(u32, u64)], binding: u64) -> Option<u32> {
    rungs
        .iter()
        .filter(|&&(_, bytes)| bytes <= binding)
        .map(|&(tokens, _)| tokens)
        .max()
}

/// `budget` with its fires cut to `top` tokens: the buckets past it go,
/// and where any were stated, `top` is the last; a lane carries a token,
/// so no more lanes than `top`.
#[must_use]
pub fn bounded_budget(budget: &poem_compiler::Budget, top: u32) -> poem_compiler::Budget {
    let mut buckets: Vec<u32> = budget
        .buckets
        .iter()
        .copied()
        .filter(|&bucket| bucket < top)
        .collect();
    if !budget.buckets.is_empty() {
        buckets.push(top);
    }
    poem_compiler::Budget {
        max_lanes: budget.max_lanes.min(top),
        max_tokens: top,
        buckets,
        ..budget.clone()
    }
}

#[must_use]
pub fn capacities(map: &ArenaMap) -> Vec<u32> {
    (0..map.placements.len())
        .map(|at| {
            let root = map.root(ValueId(at as u32));
            let Some(poem_compiler::Placement::Arena {
                bytes,
                width,
                dtype,
                ..
            }) = map.placements.get(root.0 as usize)
            else {
                return 0;
            };
            let row = width.saturating_mul(poem_compiler::arena::elem_bytes(*dtype).unwrap_or(0));
            if row == 0 {
                return 0;
            }
            u32::try_from(bytes / row).unwrap_or(u32::MAX)
        })
        .collect()
}
