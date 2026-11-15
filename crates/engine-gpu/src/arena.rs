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
//! The planes the compiler lifts over every other slot (`attn.scores`, a
//! layer each: `ArenaMap::scores_from`) are no part of a rung's carve. The
//! span is laid for fires that capture nothing; a fire some of whose lanes
//! capture is seated the planes' rows for those lanes, one a lane, right
//! under its rung's carve, inside the span (`Engine::fire_facts` states
//! them to the runtime, which cuts such a fire to a rung they fit beside).

use kernels_slang::Tensor;
use poem_compiler::{ArenaMap, Budgets, CompiledModel, FireRows, Placement, RowExpr};
use poem_exec::store::arena::rect;
use poem_ir::{Dtype, Trace, ValueId};

use crate::backend::{Backend, Tactics};
use crate::error::{Fault, Result};
use crate::handles::Handles;
use crate::run::{Scored, SlotTable};
use crate::scratch::{FireScratch, Rooms};
use crate::store::kv::Paging;

/// What a rung's carve and its rooms are each aligned to: no storage
/// binding a device asks more of (Vulkan's `minStorageBufferOffsetAlignment`
/// is at most 256).
const ALIGN: u64 = 256;

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
    /// The capture planes, which the carve leaves out.
    lifted: Lifted,
}

/// A capture plane of one carve: its slot, and a row of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Plane {
    /// The placement that holds it.
    root: u32,
    /// The bytes of one row: one capturing lane's.
    row: u64,
    width: u32,
    dtype: Dtype,
}

/// The planes the compiler lifted over every other slot of a carve, bottom
/// up. A fire with `rows` capturing lanes takes [`Lifted::bytes`] for them,
/// each plane `rows` rows, aligned, one after another.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Lifted {
    planes: Vec<Plane>,
}

impl Lifted {
    /// The planes of `map`, every slot from byte `from` on
    /// (`ArenaMap::scores_from`). The carve under `from` must hold every
    /// other slot whole, and a plane a row a lane: the lift is the
    /// compiler's, and a fire's seat for its capturing lanes rests on it.
    fn of(map: &ArenaMap, from: u64) -> Result<Lifted> {
        let mut planes: Vec<(u64, Plane)> = Vec::new();
        for (root, slot) in map.placements.iter().enumerate() {
            let Placement::Arena {
                offset,
                bytes,
                rows,
                width,
                dtype,
            } = slot
            else {
                continue;
            };
            if *bytes == 0 {
                continue;
            }
            if *offset < from {
                if offset + bytes > from {
                    return Err(Fault::Program {
                        at: "arena::Lifted::of",
                        why: format!(
                            "value {root}'s slot at byte {offset} for {bytes} lies across byte \
                             {from}, where the capture planes start"
                        ),
                    });
                }
                continue;
            }
            let row = poem_compiler::arena::elem_bytes(*dtype)
                .map_or(0, |element| width.saturating_mul(element));
            if *rows != RowExpr::Lanes || row == 0 {
                return Err(Fault::Program {
                    at: "arena::Lifted::of",
                    why: format!(
                        "value {root}'s capture plane holds {rows:?} rows of {width} {dtype:?}, \
                         not a row a lane"
                    ),
                });
            }
            planes.push((
                *offset,
                Plane {
                    root: u32::try_from(root).unwrap_or(u32::MAX),
                    row,
                    width: u32::try_from(*width).unwrap_or(u32::MAX),
                    dtype: *dtype,
                },
            ));
        }
        planes.sort_by_key(|&(offset, _)| offset);
        Ok(Lifted {
            planes: planes.into_iter().map(|(_, plane)| plane).collect(),
        })
    }

    /// Which plane backs `value`, if one does.
    fn plane_of(&self, map: &ArenaMap, value: ValueId) -> Option<usize> {
        if self.planes.is_empty() {
            return None;
        }
        let root = map.root(value).0;
        self.planes.iter().position(|plane| plane.root == root)
    }

    /// Where plane `plane` starts among `rows` rows of each.
    fn seat(&self, plane: usize, rows: u32) -> u64 {
        self.planes[..plane]
            .iter()
            .map(|plane| (u64::from(rows) * plane.row).next_multiple_of(ALIGN))
            .sum()
    }

    /// The bytes `rows` rows of every plane take.
    fn bytes(&self, rows: u32) -> u64 {
        self.seat(self.planes.len(), rows)
    }

    /// One row of every plane: what one capturing lane adds.
    fn a_lane(&self) -> u64 {
        self.planes.iter().map(|plane| plane.row).sum()
    }

    /// The most the planes' alignment adds to a fire's rows of them.
    fn slack(&self) -> u64 {
        self.planes.len() as u64 * (ALIGN - 1)
    }
}

/// The bytes from the span's end down to the planes of a fire: its rung's
/// `carve` and, right under it, `planes` bytes of them, within the `room`
/// bytes a fire's activation may take. A fire they do not fit beside was
/// cut no narrower than its rung, which the runtime's fit does for every
/// fire of a program whose term it was told.
fn planes_under(carve: u64, planes: u64, room: u64) -> Result<u64> {
    let need = carve.saturating_add(planes);
    if need > room {
        return Err(Fault::Ceiling {
            what: "bytes of the activation span for a fire's carve and its capture planes",
            need,
            have: room,
        });
    }
    Ok(need)
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
        // The carve ends where the capture planes start: a fire is seated
        // those by its capturing lanes, under the carve.
        let unscored = map.scores_from(trace);
        let lifted = Lifted::of(&map, unscored).map_err(|fault| match fault {
            Fault::Program { at, why } => Fault::Program {
                at,
                why: format!("the carve for fires of {tokens} tokens: {why}"),
            },
            other => other,
        })?;
        let rooms_at = unscored.next_multiple_of(ALIGN);
        let bytes = (rooms_at + rooms.bytes()).next_multiple_of(ALIGN);
        Ok(Rung {
            tokens,
            capacity: capacities(&map),
            map,
            rooms,
            rooms_at,
            bytes,
            lifted,
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
    /// The pool's last segment and the byte in it the span ends at: the
    /// segment is the span a fire's carve and its capture planes may take
    /// together, and the pool states it so (`ChunkFacts::act_span_chunks`).
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
        let rungs = tokens
            .iter()
            .map(|&tokens| Rung::lay(trace, compiled, budgets, tokens, paging, tuning, &routers))
            .collect::<Result<Vec<_>>>()?;
        Ok(Span {
            widest: (0..rungs.len()).collect(),
            rungs,
            routers,
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

    /// What a fire takes beside its rung's carve for its capturing lanes,
    /// as `(bytes a lane, bytes a fire)`: a row of every capture plane a
    /// lane, and the most the planes' alignment adds. Zero where the model
    /// captures nothing.
    #[must_use]
    pub fn capture_term(&self) -> (u64, u64) {
        let most = |of: fn(&Lifted) -> u64| {
            self.rungs
                .iter()
                .map(|rung| of(&rung.lifted))
                .max()
                .unwrap_or(0)
        };
        (most(Lifted::a_lane), most(Lifted::slack))
    }

    /// Whether a capture plane backs `value`: its rows are then a fire's
    /// capturing lanes ([`Scored`]), not its lanes.
    #[must_use]
    pub fn captured(&self, value: ValueId) -> bool {
        self.rungs
            .last()
            .is_some_and(|top| top.lifted.plane_of(&top.map, value).is_some())
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

    /// A fire's slots at rung `rung`: its carve at its place and, right
    /// under it, the capture planes' rows for the fire's capturing lanes
    /// (`scored`); a fire none of whose lanes captures binds no plane.
    pub fn slots(
        &self,
        handles: &Handles<B>,
        rung: usize,
        rows: FireRows,
        scored: Scored,
    ) -> Result<SlotTable> {
        let (buffer, end) = self.placed()?;
        let at = &self.rungs[rung];
        let base = end - at.bytes;
        let captured = scored.rows();
        let planes_at = if captured == 0 {
            base
        } else {
            end - planes_under(at.bytes, at.lifted.bytes(captured), end)?
        };
        let mut slots: Vec<Option<Tensor>> = Vec::with_capacity(at.map.placements.len());
        let mut backed: Vec<u32> = Vec::new();
        for value in 0..at.map.placements.len() {
            let value = ValueId(value as u32);
            let Some(plane) = at.lifted.plane_of(&at.map, value) else {
                slots.push(match rect(&at.map, value, rows) {
                    Some(rect) => Some(Tensor::new(
                        handles.bind(buffer, base + rect.offset, rect.bytes)?,
                        rect.rows,
                        rect.width,
                        rect.dtype,
                    )),
                    None => None,
                });
                continue;
            };
            backed.push(value.0);
            if captured == 0 {
                slots.push(None);
                continue;
            }
            let Plane {
                row, width, dtype, ..
            } = at.lifted.planes[plane];
            slots.push(Some(Tensor::new(
                handles.bind(
                    buffer,
                    planes_at + at.lifted.seat(plane, captured),
                    u64::from(captured) * row,
                )?,
                captured,
                width,
                dtype,
            )));
        }
        Ok(SlotTable(slots, scored.backing(backed)))
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

/// The top rung among `rungs` (`(tokens, span bytes)`): the widest whose
/// span lies in one `binding`, as the span lies in the pool's last segment
/// and no segment is past a binding. `None`: not even the least does.
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
    use poem_compiler::{ArenaMap, Placement, RowExpr};
    use poem_ir::{Dtype, ValueId};

    use super::{ALIGN, Lifted, planes_under};
    use crate::error::Fault;

    /// One row of a fixture plane: 32 f32.
    const ROW: u64 = 128;
    const LANES: u64 = 512;

    fn plane(offset: u64) -> Placement {
        Placement::Arena {
            offset,
            bytes: LANES * ROW,
            rows: RowExpr::Lanes,
            width: 32,
            dtype: Dtype::F32,
        }
    }

    /// A carve of 1024 bytes of rows, then two capture planes (the second
    /// placed first in value order) and an alias of one.
    fn map() -> ArenaMap {
        ArenaMap {
            placements: vec![
                Placement::Arena {
                    offset: 0,
                    bytes: 1024,
                    rows: RowExpr::Tokens,
                    width: 64,
                    dtype: Dtype::F32,
                },
                plane(1024 + LANES * ROW),
                Placement::Alias(ValueId(1)),
                plane(1024),
                Placement::Param(0),
            ],
            spans: Vec::new(),
            live_in: Vec::new(),
            bytes: 1024 + 2 * LANES * ROW,
        }
    }

    /// The capture planes are the slots from where the lift starts, bottom
    /// up; a fire takes a row of each a capturing lane, each plane aligned,
    /// and none where no lane captures.
    #[test]
    fn a_fire_takes_a_row_of_every_capture_plane_a_capturing_lane() {
        let map = map();
        let lifted = Lifted::of(&map, 1024).expect("the fixture's lift");
        assert_eq!(
            lifted
                .planes
                .iter()
                .map(|plane| plane.root)
                .collect::<Vec<_>>(),
            [3, 1],
            "bottom up"
        );
        assert_eq!(lifted.a_lane(), 2 * ROW);
        assert_eq!(lifted.slack(), 2 * (ALIGN - 1));
        assert_eq!(lifted.plane_of(&map, ValueId(3)), Some(0));
        assert_eq!(lifted.plane_of(&map, ValueId(2)), Some(1), "an alias");
        assert_eq!(lifted.plane_of(&map, ValueId(0)), None);
        assert_eq!(lifted.plane_of(&map, ValueId(4)), None);
        assert_eq!(lifted.bytes(0), 0);
        // Three capturing lanes: 384 bytes a plane, each on its own
        // alignment.
        assert_eq!(lifted.seat(1, 3), 512);
        assert_eq!(lifted.bytes(3), 1024);
        // Every lane capturing takes what the compiler's lift did.
        assert_eq!(lifted.bytes(LANES as u32), 2 * LANES * ROW);
        // What the term states covers the planes at every count.
        for lanes in [1, 2, 3, 7, 64, 511, 512] {
            assert!(lifted.bytes(lanes) <= u64::from(lanes) * lifted.a_lane() + lifted.slack());
        }
        // A model that captures nothing lifts none.
        let none = Lifted::of(&map, map.bytes).expect("no lift");
        assert_eq!(none, Lifted::default());
        assert_eq!(none.plane_of(&map, ValueId(3)), None);
        assert_eq!((none.a_lane(), none.slack(), none.bytes(512)), (0, 0, 0));
    }

    /// A slot lying across the byte the lift starts at, or a lifted slot
    /// that is not a row a lane, is refused by name: the seat of a fire's
    /// capturing lanes rests on both.
    #[test]
    fn a_lift_that_is_not_whole_planes_of_a_row_a_lane_is_refused() {
        let map = map();
        let Err(Fault::Program { why, .. }) = Lifted::of(&map, 512) else {
            panic!("a carve slot across the lift is refused");
        };
        assert!(why.contains("lies across byte 512"), "{why}");
        let mut rows = map.clone();
        rows.placements[3] = Placement::Arena {
            offset: 1024,
            bytes: LANES * ROW,
            rows: RowExpr::Tokens,
            width: 32,
            dtype: Dtype::F32,
        };
        let Err(Fault::Program { why, .. }) = Lifted::of(&rows, 1024) else {
            panic!("a plane a row a token is refused");
        };
        assert!(why.contains("not a row a lane"), "{why}");
    }

    /// A fire's planes lie right under its rung's carve, within what a
    /// fire's activation may take; past it the fire is refused with the
    /// bytes it wants and the bytes laid.
    #[test]
    fn a_fires_planes_lie_under_its_carve_within_the_span() {
        assert_eq!(planes_under(4096, 1024, 8192).expect("fits"), 5120);
        assert_eq!(planes_under(4096, 4096, 8192).expect("fits whole"), 8192);
        let Err(Fault::Ceiling { need, have, .. }) = planes_under(4096, 4097, 8192) else {
            panic!("planes past the span are refused");
        };
        assert_eq!((need, have), (8193, 8192));
    }
}
