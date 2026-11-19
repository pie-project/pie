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
    /// The carve and the rooms.
    bytes: u64,
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
    fn lay<T: Tactics + ?Sized>(
        trace: &Trace,
        compiled: &CompiledModel,
        budgets: &Budgets,
        tokens: u32,
        paging: Paging,
        tuning: &T,
        routers: &[u32],
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
            trace, compiled, &map, budgets, tokens, paging, tuning, routers,
        );
        let rooms_at = map.bytes.next_multiple_of(ALIGN);
        let bytes = (rooms_at + rooms.bytes()).next_multiple_of(ALIGN);
        Ok(Rung {
            tokens,
            capacity: capacities(&map),
            map,
            rooms,
            rooms_at,
            bytes,
        })
    }

    /// The carve and the rooms, aligned.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether a fire of `rows` fits this rung's carve.
    fn holds(&self, rows: FireRows) -> bool {
        let tokens = u64::from(self.tokens);
        rows.tokens <= tokens && rows.lanes <= tokens && rows.readouts <= tokens
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
    ) -> Result<Span<B>> {
        let routers = crate::scratch::routers(trace);
        let capture = CaptureRows::of(trace, paging.context(), paging.page_size)?;
        let rungs = tokens
            .iter()
            .map(|&tokens| Rung::lay(trace, compiled, budgets, tokens, paging, tuning, &routers))
            .collect::<Result<Vec<_>>>()?;
        Ok(Span {
            widest: (0..rungs.len()).collect(),
            rungs,
            routers,
            capture,
            place: None,
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
                    u32::try_from(rung.bytes.div_ceil(chunk_bytes.max(1))).unwrap_or(u32::MAX);
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

    /// The carve a fire of `rows` takes: the widest whose bytes lie in the
    /// chunks stated for the narrowest rung that holds it
    /// ([`Span::widen`]).
    #[must_use]
    pub fn rung_of(&self, rows: FireRows) -> usize {
        let narrowest = self
            .rungs
            .iter()
            .position(|rung| rung.holds(rows))
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
                    .find(|&wide| self.rungs[wide].bytes <= room)
                    .unwrap_or(at)
            })
            .collect();
    }

    /// Where rung `rung`'s carve starts in the span's buffer: the span's
    /// end less its bytes.
    pub fn base(&self, rung: usize) -> Result<u64> {
        let (_, end) = self.placed()?;
        Ok(end - self.rungs[rung].bytes)
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
        let base = end - at.bytes;
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

#[cfg(test)]
mod tests {
    use poem_compiler::{ArenaMap, Placement};
    use poem_ir::{Dim, Dtype, ValueId};

    use super::{ALIGN, CaptureRows, Grain, grain};

    /// A score room lays its key rows, then its page rows, each run
    /// aligned, and none where no lane captures; a pair's room, laid for
    /// the most lanes a fire carries, holds every fire's rows.
    #[test]
    fn a_guest_region_lays_a_key_row_a_capturing_lane_and_a_page_row_a_reader() {
        let rows = CaptureRows { keys: 2048, pages: 128 };
        assert_eq!((rows.key_bytes(), rows.page_bytes()), (8192, 512));
        assert_eq!(rows.region(0, 0), (0, 0), "no capturing lane, no region");
        assert_eq!(rows.region(3, 0), (3 * 8192, 3 * 8192));
        assert_eq!(rows.region(3, 2), (3 * 8192, 3 * 8192 + 1024));
        let odd = CaptureRows { keys: 33, pages: 3 };
        for (keys, pages) in [(1, 0), (1, 1), (3, 2), (7, 7), (64, 1)] {
            let (pages_at, bytes) = odd.region(keys, pages);
            assert_eq!(pages_at % ALIGN, 0);
            assert!(
                bytes <= odd.region(64, 64).1,
                "{keys} key rows and {pages} page rows past a 64-lane room"
            );
        }
    }

    /// A capture's value is the key row or the page row by its width, and
    /// an arena value neither.
    #[test]
    fn a_capture_is_a_key_or_a_page_row_by_its_width() {
        let map = ArenaMap {
            placements: vec![
                Placement::Capture {
                    width: Dim::ContextTimes(1),
                    dtype: Dtype::F32,
                },
                Placement::Alias(ValueId(0)),
                Placement::Capture {
                    width: Dim::ContextPages,
                    dtype: Dtype::F32,
                },
                Placement::Param(0),
            ],
            spans: Vec::new(),
            live_in: Vec::new(),
            bytes: 0,
        };
        assert_eq!(grain(&map, ValueId(0)), Some(Grain::Keys));
        assert_eq!(grain(&map, ValueId(1)), Some(Grain::Keys), "an alias");
        assert_eq!(grain(&map, ValueId(2)), Some(Grain::Pages));
        assert_eq!(grain(&map, ValueId(3)), None);
    }
}
